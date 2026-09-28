"""Permission checks and TOCTOU grant revalidation for agent actions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic import ValidationError

from .actions import ActionResult
from .agent_support import AgentBridge
from .config import AgentSettings
from .domain_policy import navigation_allowed
from .protocol import PageState
from .providers import ToolCall
from .security import (
    ActionClassifier,
    ActionIdentity,
    PermissionManager,
    RiskAssessment,
    action_identity,
    permission_denied,
    target_element,
)


@dataclass(frozen=True)
class Authorization:
    page: PageState
    call: ToolCall
    result: ActionResult | None
    confirmation: str
    assessment: RiskAssessment | None
    password_target: bool


async def authorize_action(
    call: ToolCall,
    page: PageState,
    repeat: int,
    allowlist: list[str],
    classifier: ActionClassifier | None,
    permissions: PermissionManager | None,
    bridge: AgentBridge,
    settings: AgentSettings,
    step: int,
    elapsed_ms: int,
    usage: dict[str, int],
    requested_call: ToolCall | None = None,
) -> Authorization:
    if call.tool == "close_tab":
        from .tab_control import authorize_close
        closed_call, closed_result, closed_confirmation = await authorize_close(
            call, page, permissions, bridge, settings, step
        )
        return Authorization(page, closed_call, closed_result, closed_confirmation, None, False)
    result: ActionResult | None = None
    confirmation = "not_required"
    approved_identity: ActionIdentity | None = None
    assessment = classifier.classify(call, page) if classifier else None
    target = target_element(call, page)
    password_target = bool(
        target and target.input_type.casefold() == "password"
    )

    if (call.tool in {"navigate", "open_tab"} and assessment is not None
            and "untrusted_argument" in assessment.reasons and "download" not in assessment.reasons):
        # Do not ask the user to approve model-invented page IDs/search paths.
        # Keep acquisition confirmations and all scope checks intact.
        blocked = ActionResult(success=False, action=call.tool, snapshot_id=page.snapshot_id,
            error="navigation_blocked", data={"recovery":"observed_link_or_adapter_search"}, message=(
                "This URL was not supplied by the user or observed as an enabled visible page link. "
                "Do not guess a destination or ask for approval of it. Use a current observed link; "
                "otherwise request SEARCH and use observed search inputs. Keep working in the current tab."))
        return Authorization(page, call, blocked, "not_executed", assessment, password_target)

    if repeat == 2:
        repeated_call = requested_call or call
        confirmation = "loop_blocked"
        result = ActionResult(
            success=False,
            action=repeated_call.tool,
            snapshot_id=page.snapshot_id,
            error="invalid_arguments",
            message=_repeat_correction(repeated_call),
        )
    elif permissions is not None and assessment is not None:
        permission = permissions.check(assessment)
        if permission == "block_read_only":
            confirmation = "read_only_blocked"
            result = permission_denied(
                call,
                page,
                "Read-only mode permits only low-risk tools. "
                "Ask the user to turn read-only mode off if this action is needed.",
            )
        elif permission == "confirm":
            await bridge.status(
                "Authorizing action...",
                step,
                settings.max_steps,
                elapsed_ms,
                usage,
            )
            choice = await bridge.confirm(call, assessment, step)
            confirmation = choice
            identity = action_identity(call, page)
            if not permissions.apply(assessment, choice, identity):
                result = permission_denied(
                    call,
                    page,
                    "The user cancelled this action. Do not retry the "
                    "same action unless the user changes the request.",
                )
            else:
                approved_identity = identity
        elif permissions.has_run_grant(assessment):
            confirmation = "preapproved_for_run"
            approved_identity = (
                permissions.grant_identity(assessment)
                or action_identity(call, page)
            )

    if result is None and call.tool in {"navigate", "open_tab"} and not navigation_allowed(
        str(call.arguments["url"]), allowlist
    ):
        result = ActionResult(
            success=False,
            action=call.tool,
            snapshot_id=page.snapshot_id,
            error="navigation_blocked",
            message=(
                "The destination is not a normal HTTP(S) URL. "
                "Chrome controls site access."
            ),
        )

    if (
        result is None
        and approved_identity is not None
        and permissions is not None
        and assessment is not None
    ):
        page, call, failure = await _revalidate_approved_action(
            call, page, approved_identity, bridge, settings, step
        )
        if failure is not None:
            permissions.invalidate(assessment)
            confirmation = f"{confirmation}_invalidated"
            result = failure

    return Authorization(
        page, call, result, confirmation, assessment, password_target
    )


def _repeat_correction(call: ToolCall) -> str:
    if call.tool == "get_page_state":
        return (
            "This exact get_page_state call already ran on the unchanged page. "
            "Use the current snapshot if it already contains what you need; do "
            "not re-read merely to verify existing site_context. "
            "get_page_state accepts only the optional full=true argument."
        )
    return (
        "This exact action already ran on the unchanged page. "
        "Choose a different action or use the current result."
    )


async def _revalidate_approved_action(
    call: ToolCall,
    page: PageState,
    approved: ActionIdentity,
    bridge: AgentBridge,
    settings: AgentSettings,
    step: int,
) -> tuple[PageState, ToolCall, ActionResult | None]:
    """Re-read and compare an approved target immediately before dispatch."""
    refresh = ToolCall(
        tool="get_page_state",
        arguments={},
        snapshot_id=page.snapshot_id,
        reason="Revalidate the approved target immediately before execution.",
    )
    await bridge.status(
        "Revalidating approved target...",
        step,
        settings.max_steps,
        0,
        {},
    )
    try:
        async with asyncio.timeout(settings.step_timeout_seconds):
            refreshed = await bridge.execute(refresh, step, settings.max_steps)
    except TimeoutError:
        refreshed = ActionResult(
            success=False,
            action="get_page_state",
            snapshot_id=page.snapshot_id,
            error="timeout",
            message="Target revalidation timed out.",
            duration_ms=int(settings.step_timeout_seconds * 1000),
        )
    await bridge.timeline(refresh, refreshed, step)
    await bridge.audit(refresh, refreshed, "grant_revalidation")
    raw = refreshed.data.get("page_state")
    if not refreshed.success or not isinstance(raw, dict):
        return page, call, _stale_result(
            call, page.snapshot_id, "The approved target could not be revalidated."
        )
    try:
        current = PageState.model_validate(raw)
    except ValidationError:
        return page, call, _stale_result(
            call, page.snapshot_id, "The target revalidation snapshot was invalid."
        )
    current_identity = action_identity(call, current)
    refreshed_call = call.model_copy(update={"snapshot_id": current.snapshot_id})
    if current_identity != approved:
        failure = _stale_result(
                call,
                current.snapshot_id,
                "The approved target identity or destination changed before execution.",
        )
        return current, refreshed_call, failure.model_copy(
            update={"data": {"page_state": current.model_dump()}}
        )
    return current, refreshed_call, None


def _stale_result(call: ToolCall, snapshot_id: str, prefix: str) -> ActionResult:
    return ActionResult(
        success=False,
        action=call.tool,
        element_id=str(call.arguments.get("element_id", "")) or None,
        snapshot_id=snapshot_id,
        error="stale_snapshot",
        message=(
            f"{prefix} The grant was invalidated; re-read the page and confirm again."
        ),
    )
