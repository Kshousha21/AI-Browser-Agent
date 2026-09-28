"""One model route per chat query; existing workflows retain all authority."""
import time
from pydantic import ValidationError
from app.site_adapters import SiteAdapterRegistry
from document_workflow.models import ResearchRequest, DocumentError
from document_workflow.research import run_research
from browser.agent_runtime import available_tools
from browser.prompting import clamp_page_state
from browser.product_scope import product_url_allowed, product_pattern_allowed
from browser.protocol import UserMessagePayload
from browser.session_control import ControlInterrupted
from browser.tool_profiles import Route
from browser.tool_routing import choose_route, ROUTER_FAILURE_MESSAGES


async def dispatch_query(provider, settings, envelope, bridge, documents, memory, router, *, research=False):
    try:
        request = ResearchRequest.model_validate(envelope.payload) if research else None
        payload = (UserMessagePayload(text=request.question, page_state=request.page_state,
            allowlist=request.allowlist,
            allow_remote_context=request.allow_remote_context) if request else
            UserMessagePayload.model_validate(envelope.payload))
        if (not product_url_allowed(payload.page_state.url) or
                not all(product_pattern_allowed(p) for p in payload.allowlist)):
            raise DocumentError("research_scope_changed")
        adapters = SiteAdapterRegistry(settings.site_adapters)
        page = adapters.refine_page_state(clamp_page_state(payload.page_state, settings.page))
        bridge.target = page
        active_ids = documents.document_context(page)
        if request is None and not active_ids:
            try:
                continued = memory.resume_navigation(payload.text, page)
            except ValueError:
                await bridge.send("error", {"code":"bad_request", "message":
                    "No navigation task is waiting to continue. Enter the task you want performed. Use the confirmation dialog for protected actions."}, envelope.id)
                await bridge.close()
                return
            payload = payload.model_copy(update={"text":memory.conversation.prepare(continued, page)})
        else:
            memory.clear_navigation()
            memory.conversation.clear()
        if request and request.document_id and request.document_id not in documents.ready:
            raise DocumentError("document_not_ready")
        selected = bool(request and request.mode == "specific")
        state = memory.state(page, available_tools(page.url, adapters), adapters=adapters,
            selected_pdf=selected or bool(active_ids), prepared_pdf=bool(active_ids or (request and request.document_id)),
            document_session_active=bool(active_ids),
            has_previous_document_question=bool(active_ids and documents.last_document_question))
        state = state.model_copy(update={"document_research_requested": request is not None})
        if not settings.routing.enabled:
            if request:
                await run_research(documents, request.model_dump(), bridge, provider)
                return
            from browser.run_session import handle_user_message
            await handle_user_message(provider, settings, envelope, payload, bridge,
                router_provider=router, routing_memory=memory)
            return
        started = time.monotonic()
        initial = await bridge.control.interruptible(choose_route(payload.text, state, settings.routing, router))
        decision, usage, source = initial
        bridge.control.check()
        if source == "router_failed":
            raise DocumentError(decision.failure or "router_unavailable")
        memory.previous_route = decision.route
        prepared_followup = decision.route == Route.DOCUMENT and request is None and bool(active_ids)
        await bridge.send("routing_status", {"route": decision.route.value, "adapter": state.adapter,
            "router_model": router.model if router else "unavailable", "router_latency_ms": int((time.monotonic()-started)*1000),
            "prepared_followup": prepared_followup, "document_count": len(active_ids) if prepared_followup else 0}, envelope.id)
        if decision.route == Route.DOCUMENT:
            if request is None and not active_ids:
                raise DocumentError("document_not_ready")
            if request and request.mode == "specific":
                # A newly selected file cannot reuse a previous file's retry question.
                if active_ids != (request.document_id,):
                    documents.clear_document_context()
            question = documents.document_question(payload.text)
            if request:
                request = request.model_copy(update={"question": question})
            else:
                request = ResearchRequest(mode="automatic", question=question, page_state=page,
                    allowlist=payload.allowlist, allow_remote_context=payload.allow_remote_context)
            bridge.research = True
            bridge.user_file_action = research
            await run_research(documents, request.model_dump(), bridge, provider,
                prepared_ids=active_ids if prepared_followup else None)
            return
        documents.clear_document_context()
        bridge.research = False
        bridge.user_file_action = False
        await documents.invalidate(bridge)
        from browser.run_session import handle_user_message
        await handle_user_message(provider, settings, envelope, payload, bridge,
            router_provider=router, routing_memory=memory, initial_routing=initial)
    except (DocumentError, ValidationError, ControlInterrupted) as error:
        code = error.code if isinstance(error, DocumentError) else "operation_interrupted" if isinstance(error, ControlInterrupted) else "invalid_document_request"
        if research or bridge.research:
            await bridge.send("document_done", {"status":"failed", "error":code, "answer":None,
                "prepared":[], "research_id":envelope.id})
        else:
            messages = {**ROUTER_FAILURE_MESSAGES,
                "document_not_ready":"Choose Ask about this PDF on a prepared PDF card first.",
                "document_retry_unavailable":"There is no previous question for this PDF. Enter the question to retry."}
            await bridge.send("error", {"code":"bad_request", "message":messages.get(code,"The current document or page scope is unavailable.")}, envelope.id)
        await bridge.close()
