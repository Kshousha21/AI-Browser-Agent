"""Backend-only SDK/HTTP transport settings and safe error normalization."""

from __future__ import annotations

import logging
import ssl
from typing import Any

from ..provider_settings import ProviderSettings
from .base import ProviderError


async def no_api_key() -> str:
    """An empty SDK credential supplier; explicit omit headers select no auth."""
    return ""


def protect_sdk_logs() -> None:
    """Do not let SDK debug request dumps bypass central application redaction."""
    for name in ("openai", "openai._base_client", "anthropic", "anthropic._base_client", "httpx", "httpx2", "httpcore", "httpcore2"):
        logging.getLogger(name).setLevel(logging.WARNING)


def client_options(settings: ProviderSettings) -> dict[str, Any]:
    """Build verified TLS and environment-backed headers, with no redirect replay."""
    protect_sdk_logs()
    options: dict[str, Any] = {"follow_redirects": False, "timeout": settings.timeout_seconds,
                               "trust_env": settings.trust_env}
    if settings.ca_bundle or settings.client_cert:
        try:
            context = ssl.create_default_context(cafile=settings.ca_bundle)
            if settings.client_cert:
                context.load_cert_chain(settings.client_cert, settings.client_key)
        except (OSError, ssl.SSLError):
            raise ProviderError("provider TLS certificate/key configuration is invalid", code="configuration_error") from None
        options["verify"] = context

    async def apply_auth(request: Any) -> None:
        # Never transport cookies. Compatible no-auth endpoints do not receive
        # fabricated SDK keys or inherited vendor/enterprise credentials.
        request.headers.pop("cookie", None)
        if settings.auth_mode != "native":
            for name in ("authorization", "x-api-key", "api-key"):
                request.headers.pop(name, None)
        for name, value in settings.headers.items():
            request.headers[name] = value

    options["event_hooks"] = {"request": [apply_auth]}
    return options


def provider_error(exc: Exception, family: str) -> ProviderError:
    """Never copy an upstream error body, request headers, or exception text."""
    if isinstance(exc, ProviderError):
        return exc
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    timeout = "timeout" in type(exc).__name__.lower()
    connection = any(part in type(exc).__name__.lower() for part in ("connection", "connecterror", "readerror", "remoteprotocolerror"))
    retryable = timeout or connection or (
        isinstance(status, int) and (status in {408, 409, 429} or status >= 500)
    )
    code = "request_failed"
    hint = "request failed"
    if timeout:
        code, hint = "timeout", "request timed out"
    elif connection:
        code, hint = "connection_error", "connection failed"
    elif status in {401, 403}:
        code, hint = "authentication_error", "authentication/authorization failed; check the active profile environment"
    elif status == 429:
        code, hint = "rate_limited", "rate limit reached"
    elif status in {400, 404, 422}:
        code, hint = "unsupported_request", "request rejected; check model, endpoint, native tool support, and generation settings"
    elif isinstance(status, int) and status >= 500:
        code, hint = "unavailable", "service unavailable"
    elif status is None:
        code, hint = "malformed_response", "response could not be parsed"
    suffix = f" (HTTP {status})" if isinstance(status, int) else ""
    return ProviderError(f"{family} {hint}{suffix}", retryable=retryable, code=code, status_code=status)
