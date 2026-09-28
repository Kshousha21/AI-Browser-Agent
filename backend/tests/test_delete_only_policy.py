"""User-selected deletion-only confirmations retain execution boundaries."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from browser.actions import ActionResult
from browser.agent_authorization import authorize_action
from browser.audit import AuditLogger
from browser.config import AgentSettings, SecuritySettings
from browser.protocol import ConfirmationResponsePayload
from browser.security import ActionClassifier, PermissionManager, RiskAssessment
from browser.session_bridge import WebSocketBridge
from browser.session_control import ControlInterrupted
from .test_security import call, page


def make_bridge(tmp_path, policy="delete_only"):
    return WebSocketBridge(AsyncMock(), "run", "turn", "session",
        AuditLogger(tmp_path / "audit.jsonl"),
        security=SecuritySettings(confirmation_policy=policy))


async def pending_confirmation(bridge, assessment):
    pending = asyncio.create_task(bridge.confirm(call("click", {"element_id": "f0:e-001"}), assessment, 1))
    async with asyncio.timeout(2):
        while not bridge._confirmations:
            await asyncio.sleep(0)
    return pending, next(iter(bridge._confirmations))


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["submit", "send", "upload", "download", "preview_navigation", "close_tab", "untrusted_argument"])
async def test_non_delete_authorization_needs_no_modal_and_is_audited(tmp_path, reason):
    bridge = make_bridge(tmp_path)
    operation = call("click", {"element_id": "f0:e-001"})
    decision = await bridge.confirm(operation, RiskAssessment("high", (reason,), "scope", "Target"), 1)
    assert decision == "policy_allow_once"
    bridge.websocket.send_text.assert_not_called()
    await bridge.audit(operation, ActionResult(success=True, action="click", snapshot_id="s-1"), decision)
    record = json.loads(bridge.audit_logger.path.read_text().splitlines()[-1])
    assert record["confirmation"] == "policy_allow_once"


@pytest.mark.asyncio
@pytest.mark.parametrize("decision,expected", [("cancel", "cancel"), ("allow_once", "allow_once"), ("allow_run", "allow_once")])
async def test_delete_always_waits_for_its_own_human_decision(tmp_path, decision, expected):
    bridge = make_bridge(tmp_path)
    assessment = RiskAssessment("high", ("submit", "delete"), "scope", "Delete")
    pending, key = await pending_confirmation(bridge, assessment)
    assert not pending.done()
    wire = json.loads(bridge.websocket.send_text.call_args.args[0])
    assert wire["type"] == "confirmation_request"
    assert "delete" in wire["payload"]["risk_reasons"]
    bridge.resolve_confirmation(key, ConfirmationResponsePayload(decision=decision))
    assert await pending == expected
    pending, key = await pending_confirmation(bridge, assessment)
    assert not pending.done()
    bridge.resolve_confirmation(key, ConfirmationResponsePayload(decision="cancel"))
    assert await pending == "cancel"


@pytest.mark.asyncio
async def test_legacy_consequential_setting_does_not_prompt_for_submit(tmp_path):
    bridge = make_bridge(tmp_path, "consequential")
    result = await bridge.confirm(call("click", {}), RiskAssessment("high", ("submit",), "scope", "Submit"), 1)
    assert result == "policy_allow_once"
    bridge.websocket.send_text.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["paused", "stopped", "waiting"])
async def test_automatic_authorization_never_bypasses_control(tmp_path, state):
    bridge = make_bridge(tmp_path)
    bridge.control.invalidate("Interrupted", state=state)
    with pytest.raises(ControlInterrupted):
        await bridge.confirm(call("click", {}), RiskAssessment("high", ("submit",), "scope", "Submit"), 1)
    bridge.websocket.send_text.assert_not_called()


@pytest.mark.asyncio
async def test_paused_delete_rejects_late_allow(tmp_path):
    bridge = make_bridge(tmp_path)
    pending, key = await pending_confirmation(bridge, RiskAssessment("high", ("delete",), "scope", "Delete"))
    bridge.control.invalidate("Paused")
    bridge.resolve_confirmation(key, ConfirmationResponsePayload(decision="allow_once"))
    with pytest.raises(ControlInterrupted):
        await pending


def test_deletion_cannot_be_disabled_by_required_list_or_run_grant():
    manager = PermissionManager(SecuritySettings(confirmation_policy="delete_only", require_confirmation_for=[]), False)
    assessment = RiskAssessment("high", ("delete",), "scope", "Delete")
    assert manager.check(assessment) == "confirm"
    assert manager.apply(assessment, "allow_run")
    assert not manager.has_run_grant(assessment)
    assert manager.check(assessment) == "confirm"
    assert not manager.apply(assessment, "policy_allow_once")


@pytest.mark.parametrize("name", ["Delete page", "Remove attachment", "Erase", "Seite l\u00f6schen", "Datei loeschen", "Entfernen"])
@pytest.mark.parametrize("tool,args", [("click", {}), ("press_key", {"key": "Enter"})])
def test_deletion_labels_in_english_and_german(name, tool, args):
    current = page()
    current.elements[0].name = name
    classifier = ActionClassifier("Complete the task", current, ["https://portal.example/*"])
    risk = classifier.classify(call(tool, {"element_id": "f0:e-001", **args}), current)
    assert "delete" in risk.reasons


@pytest.mark.parametrize("suffix", ["/pages/removepage.action?pageId=123", "/removeattachment.action", "/api/content/123/delete", "/content?action=delete"])
def test_deletion_endpoint_cannot_be_disguised_as_ordinary_navigation(suffix):
    current = page()
    url = "https://portal.example" + suffix
    current.elements[0].href = url
    current.elements[0].name = "Continue"
    classifier = ActionClassifier("Open " + url, current, ["https://portal.example/*"])
    assert "delete" in classifier.classify(call("navigate", {"url": url}), current).reasons
    assert "delete" in classifier.classify(call("click", {"element_id": "f0:e-001"}), current).reasons


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True])
async def test_ordinary_submit_has_no_approval_roundtrip(tmp_path, changed):
    current = page()
    fresh = current.model_copy(deep=True)
    fresh.snapshot_id = "fresh"
    if changed:
        fresh.elements[0].form_action = "https://portal.example/other-target"
    bridge = make_bridge(tmp_path)
    bridge.execute = AsyncMock(return_value=ActionResult(success=True, action="get_page_state", snapshot_id="fresh", data={"page_state": fresh.model_dump()}))
    classifier = ActionClassifier("Submit request", current, ["https://portal.example/*"])
    result = await authorize_action(call("click", {"element_id": "f0:e-001"}), current, 0,
        ["https://portal.example/*"], classifier,
        PermissionManager(SecuritySettings(confirmation_policy="delete_only"), False),
        bridge, AgentSettings(), 1, 0, {})
    bridge.execute.assert_not_called()
    bridge.websocket.send_text.assert_not_called()
    assert result.result is None
    assert result.confirmation == "not_required"


def test_policy_is_closed_configuration_and_cannot_be_a_client_decision():
    with pytest.raises(ValidationError):
        SecuritySettings(confirmation_policy="automatic")
    with pytest.raises(ValidationError):
        ConfirmationResponsePayload(decision="policy_allow_once")


@pytest.mark.asyncio
async def test_internal_navigation_is_rejected(tmp_path):
    bridge = make_bridge(tmp_path)
    for url in ["chrome://settings/", "file:///private.pdf"]:
        result = await bridge.execute(call("navigate", {"url": url}), 1, 3)
        assert result.error == "navigation_blocked"
    bridge.websocket.send_text.assert_not_called()


def test_delete_modal_disables_run_grant_in_built_panel():
    from .panel_clear_support import run
    result = run(r"""
const ws=sockets[0];ws.frame('document_choices',choices);prepareChoice();await drain();
ws.frame('confirmation_request',{tool:'click',arguments:{element_id:'e'},risk_reasons:['delete']});
const buttons=el('confirmation-dialog').querySelectorAll('button[value=allow_once],button[value=allow_run]');
console.log(JSON.stringify({open:el('confirmation-dialog').open,
 once:buttons.find(b=>b.value==='allow_once').disabled,
 run:buttons.find(b=>b.value==='allow_run').disabled}));el('clear').click();
""")
    assert result == {"open": True, "once": False, "run": True}
