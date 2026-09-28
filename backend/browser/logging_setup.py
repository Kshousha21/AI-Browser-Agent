"""Structured logging.

One JSON object per line on stdout: timestamp, level, logger, message, plus any
safe metadata attached to the record. Context that should ride along with
every line in a request (session id, connection id) is held in a ContextVar and
merged in by the formatter, so call sites do not have to thread it through.

Persistence uses closed event/metadata categories. Free text, paths, URLs,
credentials and traceback bodies are omitted; correlation IDs are hashed.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from .audit import redact as redact_value, safe_metadata

_LOG_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)

_SECRET_HINTS = ("token", "api_key", "apikey", "secret", "password", "authorization")

# Attributes LogRecord always carries; anything else was passed as `extra`.
_STANDARD_ATTRS = frozenset(
    """args asctime created exc_info exc_text filename funcName levelname levelno
    lineno module msecs message msg name pathname process processName
    relativeCreated stack_info thread threadName taskName""".split()
)


def bind(**kwargs: Any) -> contextvars.Token[dict[str, Any]]:
    """Add fields to every log line emitted in the current context."""
    merged = {**_LOG_CONTEXT.get(), **kwargs}
    return _LOG_CONTEXT.set(merged)


def unbind(token: contextvars.Token[dict[str, Any]]) -> None:
    _LOG_CONTEXT.reset(token)


def redact(key: str, value: Any) -> Any:
    lowered = key.lower()
    if any(hint in lowered for hint in _SECRET_HINTS):
        return "***"
    return redact_value(value, key)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name if record.name in _LOGGERS else "external",
            "msg": record.msg if isinstance(record.msg, str) and record.msg in _MESSAGES else "diagnostic event",
        }
        payload.update(safe_metadata({**_LOG_CONTEXT.get(), **{
            key: value for key, value in record.__dict__.items()
            if key not in _STANDARD_ATTRS and not key.startswith("_")}}))
        if record.exc_info:
            payload["exception"] = True  # Tracebacks include paths and arbitrary values.
        try:
            return json.dumps(payload, default=str)
        except (TypeError, ValueError):
            return json.dumps({**payload, "msg": str(payload["msg"])}, default=repr)


class TextFormatter(logging.Formatter):
    """Human-readable fallback for interactive use."""

    def format(self, record: logging.LogRecord) -> str:
        payload = json.loads(JsonFormatter().format(record))
        return " ".join(f"{key}={value}" for key, value in payload.items())


_LOGGERS = {"browser.routing", "browser.server", "browser.run_session", "browser.agent",
            "browser.provider_resolution", "uvicorn", "uvicorn.error", "uvicorn.access"}
_MESSAGES = {"route selected", "route transition", "model decision", "backend ready", "session opened", "session closed by client", "session failed",
    "rejected connection: authentication timeout", "client disconnected before authentication",
    "rejected connection: first frame is malformed", "rejected connection: first frame is not auth",
    "rejected connection: malformed auth payload", "rejected connection: token",
    "rejected connection: origin", "malformed envelope", "run started", "run finished",
    "provider failed", "run failed", "agent step", "agent loop aborted",
    "provider.name is deprecated; use provider.active only"}


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            TextFormatter("%(asctime)s %(levelname)-7s %(name)s  %(message)s")
        )

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    # uvicorn installs its own colourised handlers; route them through ours so
    # the output stays one stream of parseable lines.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
