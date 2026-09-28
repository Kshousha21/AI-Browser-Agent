"""Authenticated WebSocket message loop and active agent-run handling."""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import WebSocket
from pydantic import ValidationError

from .agent import run_agent
from app.site_adapters import SiteAdapterRegistry
from .audit import AuditLogger
from .config import Settings
from .build_info import compatible_hello, PROTOCOL_MISMATCH
from .prompting import clamp_page_state
from .product_scope import product_pattern_allowed, product_url_allowed
from .protocol import (
    AssistantDeltaPayload,
    AssistantDonePayload,
    ConfirmationResponsePayload,
    Envelope,
    HelloAckPayload,
    StopPayload,
    SessionControlPayload,
    ToolResultPayload,
    UserMessagePayload,
    error_envelope,
    make_envelope,
)
from .providers import ModelProvider, ProviderError
from .session_bridge import WebSocketBridge
from .tool_routing import RoutingMemory

log = logging.getLogger("browser.server")


async def send(websocket: WebSocket, envelope: Envelope) -> None:
    await websocket.send_text(envelope.model_dump_json())


async def session_loop(
    websocket: WebSocket,
    provider: ModelProvider,
    settings: Settings,
    session_id: str,
    server_name: str,
    document_services_factory=None,
    *, router_provider: ModelProvider | None = None,
) -> None:
    from document_workflow.session import DocumentSession
    documents = DocumentSession(settings, document_services_factory)
    routing_memory = RoutingMemory()
    greeted = False
    run_task: asyncio.Task[None] | None = None
    bridge: WebSocketBridge | None = None
    audit_logger = AuditLogger(settings.security.audit_log_path)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                envelope = Envelope.model_validate_json(raw)
            except ValidationError as exc:
                log.warning("malformed envelope", extra={"detail": str(exc)})
                await send(websocket, error_envelope("protocol_error", "malformed envelope"))
                continue

            if envelope.type == "hello":
                if not compatible_hello(envelope.payload):
                    await send(websocket, error_envelope("protocol_mismatch", PROTOCOL_MISMATCH, envelope.id))
                    await websocket.close(code=1002, reason=PROTOCOL_MISMATCH)
                    return
                greeted = True
                await send(
                    websocket,
                    make_envelope(
                        "hello_ack",
                        HelloAckPayload(
                            server=server_name,
                            session_id=session_id,
                            provider=provider.name,
                            model=provider.model,
                            research_max_pdfs=settings.research.max_pdfs,
                            research_max_visual_pages=settings.research.max_visual_pages,
                        ),
                        corr_id=envelope.id,
                    ),
                )
                continue
            if envelope.type == "ping":
                await send(websocket, make_envelope("pong", {}, corr_id=envelope.id))
                continue
            if envelope.type == "document_context_clear":
                if greeted and envelope.payload == {} and (run_task is None or run_task.done()):
                    routing_memory.clear_navigation()
                    routing_memory.conversation.clear()
                    documents.clear_document_context()
                continue
            if envelope.type == "stop":
                if not _valid_stop(envelope):
                    await send(
                        websocket,
                        error_envelope("bad_request", "invalid stop payload", envelope.id),
                    )
                elif bridge is not None:
                    await documents.invalidate(bridge)
                    routing_memory.clear_navigation()
                    routing_memory.conversation.clear()
                    documents.clear_document_context()
                    bridge.stop()
                    await bridge.status(
                        "Stopping...", 0, settings.agent.max_steps, 0, {}
                    )
                continue
            if envelope.type in {"pause", "resume", "session_event"}:
                try:
                    command = SessionControlPayload.model_validate(envelope.payload)
                except ValidationError:
                    await send(websocket, error_envelope("bad_request", "invalid session control", envelope.id))
                    continue
                if command.run_id not in {None, bridge.run_id if bridge else None}:
                    continue
                if bridge is not None:
                    await documents.invalidate(bridge)
                    routing_memory.clear_navigation()
                    routing_memory.conversation.clear()
                    documents.clear_document_context()
                if bridge is None or run_task is None or run_task.done():
                    continue
                if command.run_id not in {None, bridge.run_id}:
                    continue
                event = envelope.type if envelope.type != "session_event" else command.event
                if event == "resume":
                    bridge.control.resume()
                elif event == "side_panel_closed":
                    bridge.stop()
                    run_task.cancel()
                else:
                    reasons = {"pause": "Paused - You are controlling the browser",
                        "active_target_changed": "Target tab changed",
                        "target_tab_closed": "Target tab closed",
                        "target_document_changed": "Target document changed"}
                    bridge.control.invalidate(reasons.get(event, "Browser state changed"))
                await bridge.session_event(event)
                await bridge.control.notify()
                continue
            if envelope.type == "tool_result":
                try:
                    payload = ToolResultPayload.model_validate(envelope.payload)
                except ValidationError as exc:
                    await send(
                        websocket,
                        error_envelope(
                            "bad_request", f"invalid tool result: {exc}", envelope.id
                        ),
                    )
                    continue
                if bridge is None or not bridge.resolve(envelope.corr_id, payload):
                    await send(
                        websocket,
                        error_envelope(
                            "protocol_error",
                            "unknown tool request correlation",
                            envelope.id,
                        ),
                    )
                continue
            if envelope.type == "confirmation_response":
                try:
                    payload = ConfirmationResponsePayload.model_validate(
                        envelope.payload
                    )
                except ValidationError as exc:
                    await send(
                        websocket,
                        error_envelope(
                            "bad_request",
                            f"invalid confirmation response: {exc}",
                            envelope.id,
                        ),
                    )
                    continue
                if bridge is None or not bridge.resolve_confirmation(
                    envelope.corr_id, payload
                ):
                    await send(
                        websocket,
                        error_envelope(
                            "protocol_error",
                            "unknown confirmation request correlation",
                            envelope.id,
                        ),
                    )
                continue
            if envelope.type in {"document_prepare", "document_question", "document_refresh", "document_file_action", "document_research"}:
                if not greeted or (run_task is not None and not run_task.done()):
                    await send(websocket, error_envelope("bad_request", "Document operation is not available during another run.", envelope.id))
                    continue
                bridge = WebSocketBridge(websocket, str(uuid.uuid4()), envelope.id, session_id, audit_logger, security=settings.security)
                bridge.document_session = documents
                bridge.user_file_action = envelope.type in {"document_file_action", "document_research"}
                if envelope.type == "document_research":
                    from .query_dispatch import dispatch_query
                    bridge.research = True
                    run_task = asyncio.create_task(dispatch_query(provider, settings, envelope, bridge,
                        documents, routing_memory, router_provider, research=True))
                else:
                    run_task = asyncio.create_task(documents.run(envelope.type, envelope.payload, bridge))
                continue
            if envelope.type != "user_message":
                await send(
                    websocket,
                    error_envelope(
                        "protocol_error",
                        f"unsupported message type {envelope.type!r}",
                        envelope.id,
                    ),
                )
                continue
            if not greeted:
                await send(
                    websocket,
                    error_envelope(
                        "protocol_error",
                        "send 'hello' before 'user_message'",
                        envelope.id,
                    ),
                )
                continue
            if run_task is not None and not run_task.done():
                await send(
                    websocket,
                    error_envelope("bad_request", "a run is already active", envelope.id),
                )
                continue
            try:
                payload = UserMessagePayload.model_validate(envelope.payload)
            except ValidationError as exc:
                await send(
                    websocket,
                    error_envelope("bad_request", f"invalid payload: {exc}", envelope.id),
                )
                continue

            bridge = WebSocketBridge(
                websocket,
                str(uuid.uuid4()),
                envelope.id,
                session_id,
                audit_logger,
                security=settings.security,
            )
            await documents.invalidate(bridge)
            bridge.document_session = documents
            from .query_dispatch import dispatch_query
            run_task = asyncio.create_task(dispatch_query(provider, settings, envelope, bridge,
                documents, routing_memory, router_provider))
    finally:
        active_run = run_task is not None and not run_task.done()
        if active_run:
            run_task.cancel()
        try:
            if bridge is not None:
                bridge.stop()
                await bridge.close()
                if active_run:
                    await bridge.session_event("side_panel_closed")
        finally:
            # Native tab-panel removal can race a completed transport failure.
            # Collect both pending and already-finished runs on owner shutdown.
            if run_task is not None:
                await asyncio.gather(run_task, return_exceptions=True)


def _valid_stop(envelope: Envelope) -> bool:
    try:
        StopPayload.model_validate(envelope.payload)
    except ValidationError:
        return False
    return True


async def handle_user_message(
    provider: ModelProvider,
    settings: Settings,
    envelope: Envelope,
    payload: UserMessagePayload,
    bridge: WebSocketBridge,
    *, router_provider: ModelProvider | None = None,
    routing_memory: RoutingMemory | None = None,
    initial_routing=None,
) -> None:
    if not product_url_allowed(payload.page_state.url) or not all(
        product_pattern_allowed(pattern) for pattern in payload.allowlist
    ):
        await bridge.send(
            "error",
            {"code": "bad_request", "message": "Open a normal HTTP or HTTPS website. Chrome and internal pages cannot be controlled."},
            envelope.id,
        )
        await bridge.close()
        return
    bridge.target = payload.page_state
    adapters = SiteAdapterRegistry(settings.site_adapters)
    page = adapters.refine_page_state(
        clamp_page_state(payload.page_state, settings.page)
    )
    log.info(
        "run started",
        extra={
            "run_id": bridge.run_id,
            "turn_id": envelope.id,
            "goal": payload.text,
            "url": page.url,
            "text_chars": len(page.visible_text),
            "elements": len(page.elements),
        },
    )
    try:
        await bridge.status("Reading page...", 0, settings.agent.max_steps, 0, {})
        outcome = await run_agent(
            payload.text,
            page,
            payload.allowlist,
            provider,
            settings.agent,
            bridge,
            security_settings=settings.security,
            read_only=payload.read_only,
            site_adapters=adapters,
            page_settings=settings.page,
            control=bridge.control,
            routing_settings=settings.routing,
            router_provider=router_provider,
            routing_memory=routing_memory,
            initial_routing=initial_routing,
        )
    except ProviderError as exc:
        log.error("provider failed", extra={"turn_id": envelope.id, "detail": str(exc)})
        await bridge.send(
            "error", {"code": "provider_error", "message": str(exc)}, envelope.id
        )
        return
    except Exception as exc:  # noqa: BLE001 - logged and reported at boundary
        log.exception("run failed", extra={"turn_id": envelope.id})
        await bridge.send(
            "error", {"code": "internal_error", "message": str(exc)}, envelope.id
        )
        return

    if routing_memory is not None and bridge.target is not None:
        routing_memory.conversation.adopt(bridge.target)
    await bridge.status(
        "Finished",
        settings.agent.max_steps,
        settings.agent.max_steps,
        outcome.elapsed_ms,
        outcome.usage,
    )
    await bridge.send(
        "assistant_delta", AssistantDeltaPayload(text=outcome.text), envelope.id
    )
    await bridge.send(
        "assistant_done",
        AssistantDonePayload(
            text=outcome.text,
            finish_reason=outcome.finish_reason,
            usage=outcome.usage,
            elapsed_ms=outcome.elapsed_ms,
        ),
        envelope.id,
    )
    log.info(
        "run finished",
        extra={
            "run_id": bridge.run_id,
            "turn_id": envelope.id,
            "answer_chars": len(outcome.text),
            "finish_reason": outcome.finish_reason,
            "usage": outcome.usage,
            "elapsed_ms": outcome.elapsed_ms,
        },
    )
    await bridge.close()
