"""Bounded native-tool agent loop."""

from __future__ import annotations

import asyncio
import logging
import json
import time

from pydantic import ValidationError

from .actions import ActionResult
from .routing_settings import RoutingSettings
from .tool_routing import RouteController, RoutingMemory, choose_route, parse_reroute, router_failure_message, navigation_handoff
from .providers import Message
from .contact_lookup import ContactEvidence
from .search_progress import SearchProgress, procedure_lookup, search_handoff, PROCEDURE_SEARCH_INSTRUCTION
from .search_recovery import empty_search, validate_empty_search_retry, EMPTY_SEARCH_INSTRUCTION
from .navigation_recovery import navigation_owner, resource_controls, resource_pointer_handoff
from .tool_profiles import Route
from .tool_routing import RerouteRequest
from .state_reads import execute_state_read, state_identity
from .tool_execution import BuiltStateReadCall
from .session_control import RunControl, ControlInterrupted
from .agent_control import recover_page
from .agent_support import (
    AgentBridge,
    LoopDetector,
    TokenBudget,
    action_status as _action_status,
    retire_superseded_page_context as _retire_page_context,
    retire_bounded_state_reads as _retire_state_reads,
    tool_exchange as _tool_exchange,
    trim_messages as _trim,
    validation_result as _validation_result,
)
from .config import AgentSettings, PageSettings, SecuritySettings
from .context_control import compact_result
from .agent_authorization import authorize_action
from .agent_runtime import (
    AgentOutcome,
    available_tools as _available_tools,
    elapsed as _elapsed,
    login_outcome as _login_outcome,
    outcome as _outcome,
    prepare_tool_call as _prepare_tool_call,
    refine_page as _refine_page,
    token_budget_outcome as _token_budget_outcome,
)
from .prompting import (
    build_messages,
    untrusted_block,
    model_safe_page,
    model_safe_result,
)
from .protocol import PageState
from .providers import ModelProvider, ProviderError, ToolCall
from .security import (
    ActionClassifier,
    PermissionManager,
)
from .transitions import observe_navigation, observe_same_page_click, observe_stale_snapshot
from app.site_adapters import SiteAdapterRegistry

log = logging.getLogger("browser.agent")


async def run_agent(
    goal: str,
    initial_page: PageState,
    allowlist: list[str],
    provider: ModelProvider,
    settings: AgentSettings,
    bridge: AgentBridge,
    *,
    security_settings: SecuritySettings | None = None,
    read_only: bool = False,
    site_adapters: SiteAdapterRegistry | None = None,
    page_settings: PageSettings | None = None,
    control: RunControl | None = None,
    routing_settings: RoutingSettings | None = None,
    router_provider: ModelProvider | None = None,
    routing_memory: RoutingMemory | None = None,
    initial_routing=None,
) -> AgentOutcome:
    """Run until the model answers or one of the explicit bounds stops it."""
    started = time.monotonic()
    budget = TokenBudget(settings.token_budget)
    detector = LoopDetector()
    page = initial_page
    consecutive_failures = 0
    schema_retry_available = True
    navigation_corrections = 0
    search_progress = SearchProgress()
    procedure_request = procedure_lookup(goal)
    search_corrections = 0
    empty_searches = 0
    stale_refreshes = 0
    contact_corrections = 0
    contact_evidence = ContactEvidence()
    authorization_blocked = False
    blocked_navigation_owners: set[tuple] = set()
    routing_settings = routing_settings or RoutingSettings()
    routing_memory = routing_memory or RoutingMemory()
    route_control = None
    route_source = "disabled"
    classifier = (
        ActionClassifier(goal, page, allowlist)
        if security_settings is not None
        else None
    )
    permissions = (
        PermissionManager(security_settings, read_only)
        if security_settings is not None
        else None
    )

    if page.looks_like_login:
        return _login_outcome(budget, started)

    try:
        async with asyncio.timeout(settings.run_timeout_seconds):
            bootstrap_recorded = False
            while True:
                try:
                    if control is not None:
                        await control.wait_ready()
                        if control.needs_refresh:
                            page = await recover_page(control, bridge, settings, page, site_adapters, page_settings, 1)
                            initial_result = ActionResult(success=True, action="get_page_state", snapshot_id=page.snapshot_id,
                                data={"page_state": page.model_dump()})
                            bootstrap = ToolCall(tool="get_page_state", arguments={"full": True}, snapshot_id=page.snapshot_id, reason="Fresh resume observation")
                            bootstrap_recorded = True
                            break
                    bootstrap = ToolCall(
                        tool="get_page_state",
                        arguments={},
                        snapshot_id=page.snapshot_id,
                        reason="Read a fresh snapshot before the first model decision.",
                    )
                    await bridge.status(
                        "Reading page...",
                        1,
                        settings.max_steps,
                        _elapsed(started),
                        budget.usage,
                    )
                    async with asyncio.timeout(settings.step_timeout_seconds):
                        initial_result = await bridge.execute(
                            bootstrap, 1, settings.max_steps
                        )
                    break
                except ControlInterrupted:
                    if bridge.stopped:
                        return _outcome("Stopped by user.", "stopped", budget, started)
                    continue
            if not bootstrap_recorded:
                await bridge.timeline(bootstrap, initial_result, 1)
                if permissions is not None:
                    await bridge.audit(bootstrap, initial_result, "not_required")
            log.info(
                "agent step",
                extra={
                    "step": 1,
                    "tool": "get_page_state",
                    "ok": initial_result.success,
                    "error": initial_result.error,
                    "duration_ms": initial_result.duration_ms,
                },
            )
            raw_initial = initial_result.data.get("page_state")
            if not initial_result.success or not isinstance(raw_initial, dict):
                return _outcome(
                    "Could not read a fresh page snapshot.",
                    "initial_snapshot_failed",
                    budget,
                    started,
                )
            page = _refine_page(
                PageState.model_validate(raw_initial),
                site_adapters,
                page_settings,
            )
            if classifier is not None:
                classifier.observe(page)
            if page.looks_like_login:
                return _login_outcome(budget, started)
            messages = build_messages(
                goal,
                page,
                (
                    site_adapters.system_prompt_addendum(page.url)
                    if site_adapters is not None
                    else ""
                ),
            )

            for step in range(2, settings.max_steps + 1):
                try:
                    if control is not None:
                        await control.wait_ready()
                        if control.needs_refresh:
                            if permissions is not None:
                                permissions.invalidate_all()
                            page = await recover_page(control, bridge, settings, page, site_adapters, page_settings, step)
                            if classifier is not None:
                                classifier.observe(page)
                            detector = LoopDetector()
                            contact_evidence = ContactEvidence()
                            search_progress = SearchProgress()
                            consecutive_failures = 0
                            _retire_page_context(messages)
                            fresh_messages = build_messages(goal, page,
                                site_adapters.system_prompt_addendum(page.url) if site_adapters else "")
                            messages[0] = fresh_messages[0]
                            messages[2] = fresh_messages[2]
                    if bridge.stopped:
                        return _outcome("Stopped by user.", "stopped", budget, started)
                    if budget.exceeded:
                        return _token_budget_outcome(budget, started)

                    await bridge.status(
                        "Thinking...",
                        step,
                        settings.max_steps,
                        _elapsed(started),
                        budget.usage,
                    )
                    if routing_settings.enabled and route_control is None:
                        route_state = routing_memory.state(page, _available_tools(page.url, site_adapters, tab_bound=getattr(bridge, "tab_bound", False)), adapters=site_adapters)
                        if initial_routing is not None:
                            selected, router_usage, route_source = initial_routing
                            initial_routing = None
                        else:
                            operation = choose_route(goal, route_state, routing_settings, router_provider)
                            selected, router_usage, route_source = (await control.interruptible(operation)
                                if control is not None else await operation)
                        if route_source == "router_failed":
                            return _outcome(router_failure_message(selected.failure), "error", budget, started)
                        budget.record(router_usage)
                        if bridge.stopped:
                            return _outcome("Stopped by user.", "stopped", budget, started)
                        if budget.exceeded:
                            return _token_budget_outcome(budget, started)
                        route_control = RouteController(selected.route, list(selected.plan[1:]), routing_settings.max_reroutes)
                        routing_memory.previous_route = selected.route
                    active_tools = _available_tools(page.url, site_adapters, tab_bound=getattr(bridge, "tab_bound", False))
                    request_messages = messages
                    if route_control is not None:
                        route_state = routing_memory.state(page, active_tools, adapters=site_adapters)
                        active_tools = route_control.tools(active_tools,
                            site_adapters.adapter_for(page.url) if site_adapters else None, route_state.page_kind)
                        request_messages = [messages[0], route_control.instruction(route_state), *messages[1:]]
                        if routing_memory.last_result_end and not procedure_request:
                            request_messages.insert(2, Message(role="system", content=
                                f"Previous bounded result window ended at {routing_memory.last_result_end} on this page. "
                                "If below 10, use that offset for a requested next window. At 10, use supported search pagination "
                                "and retrieve the next page; never request offset 10 or invent records."))
                    if procedure_request:
                        search_progress.sync(page)
                        request_messages = [*request_messages, Message(role="system", content=PROCEDURE_SEARCH_INSTRUCTION)]
                        progress_hint = search_progress.instruction()
                        if progress_hint:
                            request_messages = [*request_messages, Message(role="system", content=progress_hint)]
                        active_tools = search_progress.tools(active_tools)
                    focused_controls = resource_controls(goal, model_safe_page(page), allowlist)
                    if navigation_owner(page) in blocked_navigation_owners:
                        active_tools = [tool for tool in active_tools if tool.name not in {"navigate", "open_tab"}]
                        request_messages = [*request_messages, Message(role="system", content=(
                            "Direct URL navigation failed on this document and is unavailable until the page changes. "
                            "Use click with an observed element_id or the site search tool. Do not repeat the URL, "
                            "open a spaces directory, or describe the current page as completion of the original task."))]
                    if route_control is not None and focused_controls:
                        focus_message = Message(role="user", content=untrusted_block(
                            "CURRENT_RESOURCE_CONTROLS", json.dumps({"snapshot_id":page.snapshot_id,
                                "controls":focused_controls}, ensure_ascii=False)))
                        # Keep leading trusted route instructions and native tool/result pairs intact.
                        position = next((i for i, message in enumerate(request_messages) if message.role != "system"), len(request_messages))
                        request_messages = [*request_messages[:position], focus_message, *request_messages[position:]]
                    if empty_search(page):
                        request_messages = [*request_messages, Message(role="system", content=EMPTY_SEARCH_INSTRUCTION)]
                    contact_evidence.observe(model_safe_page(page))
                    decision_identity = state_identity(page)
                    decision_epoch = control.epoch if control is not None else None
                    try:
                        async with asyncio.timeout(settings.step_timeout_seconds):
                            model_started = time.monotonic()
                            operation = provider.generate(request_messages, tools=active_tools)
                            try:
                                response = await control.interruptible(operation) if control is not None else await operation
                            except ProviderError as exc:
                                if route_control is not None and exc.code == "invalid_tool":
                                    raise ProviderError("tool_not_available_in_current_profile",
                                        code="tool_not_available_in_current_profile") from None
                                raise
                            finally:
                                log.info("model decision", extra={
                                    "route_source": route_source, "route": route_control.route.value if route_control else "ALL",
                                    "main_model_latency_ms": int((time.monotonic() - model_started) * 1000),
                                    "tools_exposed": len(active_tools), "rerouted": bool(route_control and route_control.reroutes),
                                })
                    except TimeoutError:
                        return _outcome(
                            "A model step timed out.", "step_timeout", budget, started
                        )
                    budget.record(response.usage)

                    if bridge.stopped:
                        return _outcome("Stopped by user.", "stopped", budget, started)
                    if budget.exceeded:
                        return _token_budget_outcome(budget, started)
                    if response.decision.type == "final":
                        if route_control is not None:
                            try:
                                reroute = parse_reroute(response.decision.answer)
                            except (ValueError, ValidationError):
                                messages.append(Message(role="system", content="Invalid reroute control object; no capability changed."))
                                continue
                            if reroute is not None:
                                # Graph, bound, and current adapter availability are backend-owned.
                                allowed = not (reroute.route.value == "SEARCH" and
                                    not route_state.search_available and route_state.product_scope != "local_fixture")
                                if reroute.route.value == "DOCUMENT" and not (route_state.selected_pdf or route_state.prepared_pdf or route_state.document_session_active):
                                    allowed = False
                                changed = allowed and route_control.transition(reroute)
                                if changed:
                                    routing_memory.previous_route = route_control.route
                                    if hasattr(bridge, "send"):
                                        await bridge.send("routing_status", {"route":route_control.route.value,
                                            "adapter":route_state.adapter, "router_model":provider.model,
                                            "route_source":"main_request", "prepared_followup":False,
                                            "document_count":0}, getattr(bridge, "turn_id", None))
                                messages.append(Message(role="system", content=(
                                    "Reroute accepted: " + route_control.route.value if changed else
                                    "Reroute forbidden or transition limit reached; no capability changed.")))
                                log.info("route transition", extra={"route": route_control.route.value,
                                    "route_source": "main_request", "rerouted": bool(changed)})
                                continue
                        if not authorization_blocked and search_handoff(response.decision.answer, page):
                            if search_corrections >= 2:
                                return _outcome("The search did not establish an answer. I could not verify the requested information.",
                                    "search_incomplete", budget, started)
                            search_corrections += 1
                            messages.append(Message(role="system", content=(
                                "The task is unfinished. A final answer ends this run; saying you will continue searching does not execute anything. "
                                "Continue now using tools or an internal reroute: open a relevant observed source, or refine the query if results "
                                "are unrelated. Do not keep reading the same result window. Return source-supported information, or explicitly "
                                "state that you could not verify an answer after investigation.")))
                            continue
                        contact_issue = contact_evidence.issue(goal, response.decision.answer, page)
                        if contact_issue and not authorization_blocked:
                            if contact_corrections >= 2:
                                return _outcome("I could not verify a suitable contact for your request and location. "
                                    "No unverified email is recommended.", "error", budget, started)
                            contact_corrections += 1
                            messages.append(Message(role="system", content=(
                                "The contact lookup is unfinished: " + contact_issue + ". Search snippets are candidates, "
                                "not verified recommendations. Open the relevant IT/contact result and read its details. "
                                "Apply the latest location correction to the original task; do not repeat an unsuitable "
                                "site's address or invent another address from its pattern. Search again with the requested "
                                "location and service if necessary. Return only observed, location-supported contacts and "
                                "cite their source page. Continue with tools or an internal reroute, not advice for the user to search.")))
                            continue
                        if (route_control is not None and not authorization_blocked
                                and route_control.route.value in {"SEARCH", "READ", "ACTION", "NAVIGATION"}
                                and (navigation_handoff(response.decision.answer)
                                     or resource_pointer_handoff(goal, response.decision.answer, focused_controls))):
                            if navigation_corrections < 2:
                                navigation_corrections += 1
                                messages.append(Message(role="system", content=(
                                    "The task is unfinished. A sidebar or section link is a route to investigate, "
                                    "not an answer to a request for a particular resource or lab location. "
                                    "Open the relevant observed control, inspect its page, and search for the "
                                    "requested name/details if they are absent. Do not invent a location. "
                                    "Do not ask the user to confirm ordinary navigation, "
                                    "searching or a profile switch. Continue the ORIGINAL user task now: use a "
                                    "current advertised tool or the structured reroute control. If a tool reports "
                                    "a genuine permission denial, stop and explain that denial.")))
                                continue
                            routing_memory.remember_navigation(goal, page)
                            return _outcome("The model did not complete navigation. Retry or continue the original request.",
                                            "error", budget, started)
                        return _outcome(
                            response.decision.answer,
                            response.finish_reason,
                            budget,
                            started,
                        )

                    requested_call = response.decision
                    if route_control is not None and requested_call.tool not in {t.name for t in active_tools}:
                        invalid = ActionResult(success=False, action=requested_call.tool, snapshot_id=page.snapshot_id,
                            error="tool_not_available_in_current_profile", message="Select a tool from the current profile.")
                        await bridge.timeline(requested_call, invalid, step)
                        if permissions is not None:
                            await bridge.audit(requested_call, invalid, "profile_rejected")
                        messages.extend(_tool_exchange(response, requested_call, invalid))
                        messages = _trim(messages, settings.history_steps)
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            return _outcome("Repeated tools outside the current profile.",
                                "tool_not_available_in_current_profile", budget, started)
                        continue
                    try:
                        if procedure_request:
                            search_progress.validate(requested_call)
                        validate_empty_search_retry(goal, page, requested_call)
                        call, adapter_tool = _prepare_tool_call(
                            page.url, requested_call, site_adapters
                        )
                    except (ValidationError, ValueError) as exc:
                        invalid = _validation_result(requested_call, page, exc)
                        if permissions is not None:
                            await bridge.timeline(requested_call, invalid, step)
                            await bridge.audit(
                                requested_call, invalid, "schema_rejected"
                            )
                        messages.extend(
                            _tool_exchange(response, requested_call, invalid)
                        )
                        messages = _trim(messages, settings.history_steps)
                        if schema_retry_available:
                            schema_retry_available = False
                            continue
                        return _outcome(
                            f"The model produced invalid tool arguments twice: {exc}",
                            "invalid_tool_arguments",
                            budget,
                            started,
                        )
                    schema_retry_available = True

                    equivalent_state = bool(
                        adapter_tool
                        and site_adapters is not None
                        and site_adapters.tool_call_satisfied(
                            page.url, requested_call, page
                        )
                    )
                    loop_key = detector.fingerprint(requested_call, page)
                    semantic_loop_key = detector.semantic_fingerprint(
                        requested_call
                    )
                    repeat = detector.record(
                        requested_call,
                        page,
                        equivalent_state=equivalent_state,
                    )
                    if repeat >= 3:
                        repeated = ActionResult(
                            success=False,
                            action=requested_call.tool,
                            snapshot_id=page.snapshot_id,
                            error="invalid_arguments",
                            message=(
                                f"The same {requested_call.tool} action repeated three times; "
                                "the run is being aborted."
                            ),
                        )
                        await bridge.timeline(requested_call, repeated, step)
                        if permissions is not None:
                            await bridge.audit(
                                requested_call, repeated, "loop_aborted"
                            )
                        log.warning(
                            "agent loop aborted",
                            extra={
                                "tool": requested_call.tool,
                                "loop_repeat": repeat,
                                "loop_fingerprint": loop_key,
                                "loop_semantic_fingerprint": semantic_loop_key,
                                "loop_equivalent_state": equivalent_state,
                            },
                        )
                        return _outcome(
                            "Aborted after the same "
                            f"{requested_call.tool} action repeated three times.",
                            "loop_detected",
                            budget,
                            started,
                        )

                    authorization = await authorize_action(
                        call,
                        page,
                        repeat,
                        allowlist,
                        classifier,
                        permissions,
                        bridge,
                        settings,
                        step,
                        _elapsed(started),
                        budget.usage,
                        requested_call=requested_call,
                    )
                    page = authorization.page
                    call = authorization.call
                    result = authorization.result
                    confirmation = authorization.confirmation
                    if (result is not None and result.error in {"permission_denied", "navigation_blocked"}
                            and result.data.get("recovery") != "observed_link_or_adapter_search"):
                        authorization_blocked = True
                    assessment = authorization.assessment
                    password_target = authorization.password_target
                    timeline_call = requested_call if adapter_tool else call

                    if result is None:
                        await bridge.status(
                            _action_status(call),
                            step,
                            settings.max_steps,
                            _elapsed(started),
                            budget.usage,
                        )
                        try:
                            async with asyncio.timeout(settings.step_timeout_seconds):
                                if isinstance(call, BuiltStateReadCall):
                                    assert site_adapters is not None
                                    result = execute_state_read(
                                        call, page, target=getattr(bridge, "target", None),
                                        decision_identity=decision_identity,
                                        decision_epoch=decision_epoch, control=control,
                                        adapters=site_adapters, allowlist=allowlist,
                                    )
                                    if result.success:
                                        publish = getattr(bridge, "offer_document_results", None)
                                        if publish is not None:
                                            await publish(page, result, allowlist)
                                        result = result.model_copy(update={"data": {
                                            **result.data, "context_kind": "bounded_state_read",
                                        }})
                                else:
                                    result = await bridge.execute(
                                        call, step, settings.max_steps
                                    )
                        except TimeoutError:
                            result = ActionResult(
                                success=False,
                                action=call.tool,
                                snapshot_id=page.snapshot_id,
                                error="timeout",
                                message="Browser action exceeded the per-step timeout.",
                                duration_ms=int(settings.step_timeout_seconds * 1000),
                            )

                    same_page_click = (
                        call.tool == "click"
                        and result.success
                        and not result.page_changed
                    )
                    if same_page_click:
                        await bridge.timeline(timeline_call, result, step)
                        if permissions is not None:
                            await bridge.audit(
                                call,
                                result,
                                confirmation,
                                password_target=password_target,
                            )

                    adapter_navigation = bool(
                        adapter_tool
                        and call.tool == "navigate"
                        and result.success
                        and result.page_changed
                    )
                    if adapter_navigation:
                        await bridge.timeline(timeline_call, result, step)
                        if permissions is not None:
                            await bridge.audit(
                                call,
                                result,
                                confirmation,
                                password_target=password_target,
                            )
                        result = await observe_navigation(
                            call,
                            result,
                            page,
                            bridge,
                            settings,
                            step,
                            audit=permissions is not None,
                        )

                    result = await observe_same_page_click(
                        call,
                        result,
                        page,
                        bridge,
                        settings,
                        step,
                        audit=permissions is not None,
                    )

                    if not same_page_click and not adapter_navigation:
                        await bridge.timeline(timeline_call, result, step)
                        if permissions is not None:
                            await bridge.audit(
                                call,
                                result,
                                confirmation,
                                password_target=password_target,
                            )
                    log.info(
                        "agent step",
                        extra={
                            "step": step,
                            "tool": timeline_call.tool,
                            "executed_tool": call.tool,
                            "element_id": result.element_id,
                            "element_name": result.element_name,
                            "ok": result.success,
                            "error": result.error,
                            "page_changed": result.page_changed,
                            "reresolved": result.reresolved,
                            "duration_ms": result.duration_ms,
                            "risk": assessment.risk if assessment else "unclassified",
                            "confirmation": confirmation,
                            "loop_repeat": repeat,
                            "loop_fingerprint": loop_key,
                            "loop_semantic_fingerprint": semantic_loop_key,
                            "loop_equivalent_state": equivalent_state,
                        },
                    )
                    if (result.error == "stale_snapshot" and authorization.result is None
                            and not isinstance(call, BuiltStateReadCall)):
                        if permissions is not None:
                            permissions.invalidate_all()
                        if stale_refreshes >= 2:
                            return _outcome("The page snapshot repeatedly expired. No stale action was retried.",
                                "stale_snapshot", budget, started)
                        stale_refreshes += 1
                        result = await observe_stale_snapshot(result, page, bridge, settings, step,
                            audit=permissions is not None)
                        if not result.data.get("stale_snapshot_refreshed"):
                            return _outcome("The expired snapshot could not be refreshed in the same page. No action was retried.",
                                "stale_snapshot", budget, started)
                        # The browser rejected this attempt before execution. Remove only
                        # its recorded counts after a verified same-owner refresh. The
                        # next model decision still goes through normal authorization.
                        detector.discard_rejected_attempt(loop_key, semantic_loop_key)
                    previous_page = page
                    fresh_page_observed = False
                    raw_page = result.data.get("page_state")
                    if isinstance(raw_page, dict):
                        try:
                            page = _refine_page(
                                PageState.model_validate(raw_page),
                                site_adapters,
                                page_settings,
                            )
                            fresh_page_observed = True
                            result = result.model_copy(
                                update={
                                    "data": {
                                        **result.data,
                                        "page_state": page.model_dump(),
                                    }
                                }
                            )
                            if classifier is not None:
                                classifier.observe(page)
                        except ValidationError:
                            page = previous_page
                    if result.page_changed and not fresh_page_observed:
                        # A successful navigation without an observation cannot
                        # retain authority to read the prior adapter records.
                        page = PageState.model_validate(page.model_dump())
                    if (result.page_changed or state_identity(previous_page) != state_identity(page)
                            or (isinstance(call, BuiltStateReadCall) and result.success)):
                        _retire_state_reads(messages)
                    safe_page = model_safe_page(previous_page)
                    safe_result = model_safe_result(result, previous_page)
                    compacted = compact_result(
                        safe_result,
                        safe_page,
                        force_full=(
                            result.data.get("stale_snapshot_refreshed") is True
                            or (call.tool == "get_page_state"
                            and call.arguments.get("full") is True)
                        ),
                    )
                    observation = compacted.data.get("page_state")
                    crossed_page_identity = (
                        previous_page.tab_id != page.tab_id
                        or previous_page.url != page.url
                        or previous_page.document_id != page.document_id
                        or previous_page.navigation_id != page.navigation_id
                    )
                    if (
                        (crossed_page_identity or result.data.get("stale_snapshot_refreshed")
                         or call.tool in {"switch_tab", "open_tab", "close_tab"})
                        and isinstance(observation, dict)
                        and observation.get("type") == "page_state_full"
                    ):
                        _retire_page_context(messages)
                        messages[0] = build_messages(goal, page, site_adapters.system_prompt_addendum(page.url) if site_adapters else "")[0]
                    messages.extend(
                        _tool_exchange(response, timeline_call, compacted)
                    )
                    messages = _trim(messages, settings.history_steps)

                    if procedure_request:
                        search_progress.observe(page, requested_call, result)
                    if route_control is not None:
                        routing_memory.observe(route_control.route, requested_call, result, page,
                                               _available_tools(page.url, site_adapters, tab_bound=getattr(bridge, "tab_bound", False)))
                        if route_control.after_result(requested_call, result):
                            routing_memory.previous_route = route_control.route
                            log.info("route transition", extra={"route": route_control.route.value,
                                "route_source": "orchestrator", "rerouted": True})
                    if (procedure_request and route_control is not None and route_control.route == Route.SEARCH
                            and requested_call.tool in {"type_text", "press_key", "get_page_files"}
                            and result.success and resource_controls(goal, model_safe_page(page), allowlist)):
                        changed = route_control.transition(RerouteRequest(route=Route.ACTION,
                            reason="Inspect the relevant observed procedure link"))
                        if changed:
                            routing_memory.previous_route = route_control.route
                            messages.append(Message(role="system", content=
                                "A relevant observed procedure link is now available. Use ACTION/click to inspect its source before another search."))
                            if hasattr(bridge, "send"):
                                await bridge.send("routing_status", {"route":route_control.route.value,
                                    "adapter":route_state.adapter, "router_model":provider.model,
                                    "route_source":"orchestrator", "prepared_followup":False, "document_count":0},
                                    getattr(bridge, "turn_id", None))
                    if (route_control is not None and result.error == "navigation_blocked"
                            and result.data.get("recovery") == "observed_link_or_adapter_search"):
                        blocked_navigation_owners.add(navigation_owner(page))
                        # Recovery is backend-directed, bounded by the existing route graph/budget.
                        recovery_route = Route.ACTION if resource_controls(goal, model_safe_page(page), allowlist) else Route.SEARCH
                        state = routing_memory.state(page,
                            _available_tools(page.url, site_adapters, tab_bound=getattr(bridge, "tab_bound", False)),
                            adapters=site_adapters)
                        if recovery_route != Route.SEARCH or state.search_available or state.product_scope == "local_fixture":
                            changed = route_control.transition(RerouteRequest(route=recovery_route,
                                reason="Recover blocked navigation using observed controls or site search"))
                            if changed:
                                routing_memory.previous_route = route_control.route
                                if hasattr(bridge, "send"):
                                    await bridge.send("routing_status", {"route":route_control.route.value,
                                        "adapter":state.adapter, "router_model":provider.model,
                                        "route_source":"orchestrator", "prepared_followup":False,
                                        "document_count":0}, getattr(bridge, "turn_id", None))
                    if requested_call.tool == "type_text" and result.success:
                        empty_searches = empty_searches + 1 if empty_search(page) else 0
                        if empty_searches >= 3:
                            return _outcome("Three searches returned no results. I could not verify the requested information from this website.",
                                "search_exhausted", budget, started)
                    if page.looks_like_login:
                        return _login_outcome(budget, started)

                    if result.success:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            return _outcome(
                                "Aborted after three consecutive browser-tool failures.",
                                "tool_failures",
                                budget,
                                started,
                            )
                except ControlInterrupted:
                    if permissions is not None:
                        permissions.invalidate_all()
                    if bridge.stopped:
                        return _outcome("Stopped by user.", "stopped", budget, started)
                    continue
    except TimeoutError:
        return _outcome("The run time limit was reached.", "run_timeout", budget, started)
    except ProviderError:
        raise

    return _outcome(
        "Reached maximum steps without completing the task.",
        "max_steps",
        budget,
        started,
    )
