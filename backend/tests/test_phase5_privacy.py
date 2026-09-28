"""Synthetic production-readiness regressions for persisted diagnostics."""
import json
import logging

import pytest

from browser.actions import ActionResult
from browser.audit import AuditLogger
from browser.logging_setup import JsonFormatter, TextFormatter, bind, unbind
from browser.providers import ToolCall

CANARY = "SYNTHETIC_PRIVATE_CONTENT_5"


@pytest.mark.parametrize("tool,args", [
    ("get_page_files", {"query": CANARY}),
    ("navigate", {"url": "https://example.invalid/" + CANARY}),
    ("type_text", {"text": CANARY, "element_id": "f0:e-001"}),
    ("acquire_attachment", {"filename": "C:/" + CANARY + ".pdf"}),
])
def test_audit_keeps_decisions_without_content(tmp_path, tool, args):
    audit = AuditLogger(tmp_path / "audit.jsonl")
    call = ToolCall(tool=tool, arguments=args, snapshot_id=CANARY, reason=CANARY)
    result = ActionResult(success=False, action=tool, snapshot_id=CANARY,
                          error="permission_denied", message=CANARY,
                          element_name=CANARY, data={"chunks": CANARY})
    for choice in ("cancel", "allow_once"):
        audit.record(session_id="session-1", run_id="run-1", call=call,
                     result=result, confirmation=choice)
    serialized = audit.path.read_text(encoding="utf-8")
    assert CANARY not in serialized
    rows = [json.loads(line) for line in serialized.splitlines()]
    assert [row["confirmation"] for row in rows] == ["cancel", "allow_once"]
    assert all(row["tool"] == tool and row["result"]["error"] == "permission_denied" for row in rows)
    assert rows[0]["session_id"] == rows[1]["session_id"]


@pytest.mark.parametrize("formatter", [JsonFormatter, TextFormatter])
@pytest.mark.parametrize("kind", ["run", "malformed", "exception", "access", "unknown"])
def test_log_formats_keep_only_safe_diagnostics(formatter, kind):
    logger = logging.getLogger("uvicorn.access" if kind == "access" else "browser.run_session")
    record = logger.makeRecord(logger.name, logging.ERROR, "C:/" + CANARY, 1,
        "run started" if kind == "run" else CANARY, (),
        (RuntimeError, RuntimeError(CANARY), None) if kind == "exception" else None,
        extra={"goal": CANARY, "url": "https://example.invalid/" + CANARY,
               "detail": CANARY, "element_name": CANARY, "unknown": CANARY,
               "turn_id": CANARY, "text_chars": 25})
    context = bind(session_id="session-1", private_context=CANARY)
    try:
        rendered = formatter().format(record)
    finally:
        unbind(context)
    assert CANARY not in rendered
    assert "25" in rendered
    assert "ERROR" in rendered


def test_audit_unknown_names_cannot_persist_content(tmp_path):
    audit = AuditLogger(tmp_path / "audit.jsonl")
    call = ToolCall(tool=CANARY, arguments={CANARY: CANARY}, snapshot_id="s", reason="")
    audit.record(session_id="session-1", run_id="run-1", call=call,
                 result=ActionResult(success=True, action=CANARY, snapshot_id="s"),
                 confirmation=CANARY)
    assert CANARY not in audit.path.read_text(encoding="utf-8")
