"""Append-only, redacted security audit records."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .actions import ActionResult, ARGUMENT_MODELS, ToolError
from typing import get_args
from .providers import ToolCall

_SENSITIVE_KEYS = (
    "api_key",
    "apikey",
    "token",
    "cookie",
    "authorization",
    "password",
    "secret",
    "environment",
    "env",
)
_INLINE_SECRET = re.compile(
    r"(?i)(authorization\s*:\s*(?:bearer|basic)\s+|"
    r"(?:api[_ -]?key|token|cookie|password|secret)\s*[:=]\s*)\S+"
)
_KEY_SHAPE = re.compile(r"\b(?:sk-ant-|sk-proj-|sk-)[A-Za-z0-9_-]{12,}\b")

_EVENTS = {"pause", "resume", "active_target_changed", "target_tab_closed",
           "target_document_changed", "side_panel_closed", "stale_action_rejected"}
_TOOLS = set(ARGUMENT_MODELS) | _EVENTS | {
    "get_page_files", "acquire_attachment",
    "download_attachment", "open_attachment_preview", "inspect_attachment_preview", "return_research_results"}
_CONFIRMATIONS = {"policy_allow_once", "allow_once", "allow_run", "cancel", "not_required", "not_executed",
    "profile_rejected", "session_control", "read_only_blocked", "loop_blocked", "loop_aborted",
    "preapproved_for_run", "target_invalidated", "user_click", "user_research"}
_CONFIRMATIONS |= {item + "_invalidated" for item in tuple(_CONFIRMATIONS)}
_NUMBERS = {"step", "max_steps", "text_chars", "elements", "answer_chars", "elapsed_ms",
    "router_latency_ms", "main_model_latency_ms", "tools_exposed", "duration_ms", "loop_repeat", "tab_id", "control_epoch", "limit", "offset", "port"}
_FLAGS = {"rerouted", "success", "ok", "page_changed", "reresolved", "loop_equivalent_state"}
_CATEGORIES = {
    "route": {"READ", "SEARCH", "ACTION", "NAVIGATION", "DOCUMENT", "IMAGE", "GENERAL", "UNSUPPORTED", "ALL"},
    "router_failure": {"router_timeout", "router_unavailable", "router_invalid_response"},
    "route_source": {"deterministic", "local_router", "router_failed", "orchestrator", "main_request", "disabled"},
    "tool": _TOOLS, "executed_tool": _TOOLS,
    "error": set(get_args(ToolError)), "confirmation": _CONFIRMATIONS,
    "risk": {"low", "medium", "high", "unclassified"},
    "provider": {"mock", "anthropic", "openai", "azure_openai", "ollama",
                 "openai_compatible", "anthropic_compatible"},
    "finish_reason": {"stop", "stopped", "max_steps", "budget", "timeout", "error"},
}


def safe_metadata(values: dict[str, Any]) -> dict[str, Any]:
    """Closed persistence schema: no free text, paths, URLs or arbitrary keys."""
    output = {}
    for key, value in values.items():
        if key in _NUMBERS and type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 1e15:
            output[key] = value
        elif key in _FLAGS and type(value) is bool:
            output[key] = value
        elif key in _CATEGORIES:
            output[key] = value if isinstance(value, str) and value in _CATEGORIES[key] else None
        elif key in {"session_id", "run_id", "turn_id", "snapshot_id", "acquisition_id"} and isinstance(value, str):
            output[key] = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
        elif key == "usage" and isinstance(value, dict):
            output[key] = {name: count for name, count in value.items()
                if name in {"prompt_tokens", "completion_tokens", "total_tokens"}
                and type(count) is int and 0 <= count <= 1e15}
    return output


def redact(value: Any, key: str = "") -> Any:
    """Recursively redact credentials, including secrets embedded in URLs/text."""
    lowered = key.casefold()
    if any(hint in lowered for hint in _SENSITIVE_KEYS):
        return "***"
    if isinstance(value, dict):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        cleaned = _INLINE_SECRET.sub(r"\1***", value)
        cleaned = _KEY_SHAPE.sub("***", cleaned)
        if key.casefold() == "url":
            cleaned = _redact_url(cleaned)
        return cleaned
    return value


class AuditLogger:
    """Write one append-only JSON record for every attempted browser action."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def record_event(self, *, session_id: str, run_id: str, event: str,
                     tab_id: int | None, epoch: int) -> None:
        allowed = {"pause", "resume", "active_target_changed", "target_tab_closed",
            "target_document_changed", "side_panel_closed", "stale_action_rejected"}
        if event not in allowed:
            return
        self.record(session_id=session_id, run_id=run_id,
            call=ToolCall(tool=event, arguments={"tab_id": tab_id, "control_epoch": epoch},
                snapshot_id="", reason="Browser session lifecycle"),
            result=ActionResult(success=True, action=event, snapshot_id="", message=event),
            confirmation="session_control")

    def record(
        self,
        *,
        session_id: str,
        run_id: str,
        call: ToolCall,
        result: ActionResult,
        confirmation: str,
        password_target: bool = False,
    ) -> None:
        record = {
            "audit_version": 2,
            "ts": datetime.now(timezone.utc).isoformat(),
            **safe_metadata({"session_id": session_id, "run_id": run_id,
                             "tool": call.tool, "snapshot_id": call.snapshot_id,
                             "confirmation": confirmation}),
            "arguments": safe_metadata({key: value for key, value in call.arguments.items()
                if key in {"tab_id", "control_epoch", "limit", "offset", "acquisition_id"}}),
            "result": {
                "success": result.success,
                "error": result.error,
                "page_changed": result.page_changed,
                "reresolved": result.reresolved,
                "duration_ms": result.duration_ms,
            },
        }
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(descriptor, payload.encode("utf-8"))
            finally:
                os.close(descriptor)


def _redact_url(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    query = []
    for key, item in parse_qsl(parts.query, keep_blank_values=True):
        query.append(
            (
                key,
                "***"
                if any(hint in key.casefold() for hint in _SENSITIVE_KEYS)
                else item,
            )
        )
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )
