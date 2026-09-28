"""FastAPI app: GET /health and WS /ws.

Three independent gates guard the socket, and all three must pass:

1. the process binds loopback only (enforced in config and at run time),
2. the handshake's Origin header must be allow-listed,
3. the first WebSocket frame must authenticate with the shared token.

Origin rejection happens before ``accept()``. Token authentication happens
immediately afterwards and has a two-second deadline.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from . import logging_setup
from .config import Settings, load_settings
from .build_info import APPLICATION_VERSION, PROTOCOL_REVISION
from .logging_setup import bind, unbind
from .protocol import (
    PROTOCOL_VERSION,
    AuthPayload,
    Envelope,
)
from .providers import ModelProvider, build_provider
from .run_session import session_loop

log = logging.getLogger("browser.server")

SERVER_NAME = "ai-browser-agent-backend/0.1.0"

# 1008 is "policy violation", which is what a failed auth check is.
WS_POLICY_VIOLATION = 1008
WS_INTERNAL_ERROR = 1011
AUTH_TIMEOUT_SECONDS = 2.0


def origin_allowed(origin: str | None, settings: Settings) -> bool:
    """An absent Origin is never allowed: the header is one of the three gates."""
    if not origin:
        return False
    if origin in settings.server.allowed_origins:
        return True
    if settings.server.allow_any_extension_origin and origin.startswith(
        "chrome-extension://"
    ):
        return True
    return False


def token_valid(presented: str | None, settings: Settings) -> bool:
    if not presented or not settings.auth.token:
        return False
    return hmac.compare_digest(presented, settings.auth.token)


async def authenticate(websocket: WebSocket, settings: Settings) -> bool:
    """Require a valid auth envelope as the socket's first frame."""
    try:
        raw = await asyncio.wait_for(
            websocket.receive_text(), timeout=AUTH_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        log.warning("rejected connection: authentication timeout")
        await websocket.close(
            code=WS_POLICY_VIOLATION, reason="authentication timeout"
        )
        return False
    except WebSocketDisconnect:
        log.info("client disconnected before authentication")
        return False

    try:
        envelope = Envelope.model_validate_json(raw)
    except ValidationError:
        log.warning("rejected connection: first frame is malformed")
        await websocket.close(code=WS_POLICY_VIOLATION, reason="auth frame required")
        return False

    if envelope.type != "auth":
        log.warning(
            "rejected connection: first frame is not auth",
            extra={"message_type": envelope.type},
        )
        await websocket.close(code=WS_POLICY_VIOLATION, reason="auth frame required")
        return False

    try:
        payload = AuthPayload.model_validate(envelope.payload)
    except ValidationError:
        log.warning("rejected connection: malformed auth payload")
        await websocket.close(code=WS_POLICY_VIOLATION, reason="invalid auth payload")
        return False

    if not token_valid(payload.token, settings):
        log.warning("rejected connection: token")
        await websocket.close(code=WS_POLICY_VIOLATION, reason="invalid token")
        return False
    return True


def create_app(settings: Settings, provider: ModelProvider | None = None, *, document_services_factory=None, router_provider: ModelProvider | None = None) -> FastAPI:
    """Build the app.

    ``provider`` is injectable so tests run without a network or an API key.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.provider = provider or build_provider(settings.provider)
        app.state.router_provider = router_provider or (build_provider(settings.routing_provider)
            if settings.routing.enabled and settings.routing_provider is not None else None)
        log.info(
            "backend ready",
            extra={
                "provider": app.state.provider.name,
                "host": settings.server.host,
                "port": settings.server.port,
            },
        )
        try:
            yield
        finally:
            try:
                await app.state.provider.aclose()
            finally:
                if app.state.router_provider is not None:
                    await app.state.router_provider.aclose()

    app = FastAPI(title="AI Browser Agent Backend", lifespan=lifespan)
    app.state.settings = settings

    @app.get("/health")
    async def health() -> dict[str, object]:
        current: ModelProvider | None = getattr(app.state, "provider", None)
        return {
            "status": "ok",
            "service": "ai-browser-agent",
            "backend_version": APPLICATION_VERSION,
            "protocol_revision": PROTOCOL_REVISION,
            "protocol_version": PROTOCOL_VERSION,
            "provider": current.name if current else settings.provider.name,
            "active_provider": current.name if current else settings.provider.name,
            "provider_ready": current is not None,
            "product_scope": "http_https",
        }

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")

        if not origin_allowed(origin, settings):
            log.warning("rejected connection: origin", extra={"origin": origin})
            await websocket.close(code=WS_POLICY_VIOLATION, reason="origin not allowed")
            return

        await websocket.accept()
        if not await authenticate(websocket, settings):
            return

        session_id = str(uuid.uuid4())
        ctx = bind(session_id=session_id)
        log.info("session opened", extra={"origin": origin})
        try:
            await session_loop(
                websocket,
                app.state.provider,
                settings,
                session_id,
                SERVER_NAME,
                document_services_factory,
                router_provider=app.state.router_provider,
            )
        except WebSocketDisconnect:
            log.info("session closed by client")
        except Exception:
            log.exception("session failed")
            try:
                await websocket.close(code=WS_INTERNAL_ERROR)
            except RuntimeError:
                pass  # already closed
        finally:
            unbind(ctx)

    return app


def main() -> None:
    """Console entry point: validate config, then serve."""
    import sys

    import uvicorn

    from .config import ConfigError

    try:
        settings = load_settings()
    except ConfigError as exc:
        print("Configuration error. Run scripts/doctor.ps1 or python -m browser.config_check for safe diagnostics.", file=sys.stderr)
        raise SystemExit(2) from exc

    logging_setup.configure_logging(settings.logging.level, settings.logging.format)
    uvicorn.run(
        create_app(settings),
        host=settings.server.host,
        port=settings.server.port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
