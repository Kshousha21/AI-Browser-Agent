"""Session interruption, tab authorization, lineage, and privacy regressions."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from browser.actions import ActionResult, validate_tool_call
from browser.agent import run_agent
from browser.audit import AuditLogger
from browser.config import AgentSettings, SecuritySettings
from browser.context_control import page_observation
from browser.prompting import model_safe_result
from browser.protocol import ConfirmationResponsePayload, PageState, ToolResultPayload
from browser.providers import MockProvider, ToolCall
from browser.security import ActionClassifier, PermissionManager, action_identity
from browser.session_bridge import WebSocketBridge
from browser.session_control import ControlInterrupted, RunControl
from browser.tab_control import ApprovedTabClose, authorize_close, safe_tabs
from .test_agent import PAGE, FakeBridge, tool

A = PAGE.model_copy(update={"tab_id": 1, "document_id": "document-A", "snapshot_id": "snapshot-A", "visible_text": "OLD_PAGE_BODY_CANARY"})
B = PAGE.model_copy(update={"tab_id": 2, "document_id": "document-B", "snapshot_id": "snapshot-B", "visible_text": "NEW_PAGE_BODY_CANARY", "url": "http://127.0.0.1:8080/b.html"})


def observed(page):
    return ActionResult(success=True, action="get_page_state", snapshot_id=page.snapshot_id, data={"page_state": page.model_dump()})


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("name,args", [("get_tabs", {}), ("switch_tab", {"tab_id": 2}), ("open_tab", {"url": B.url}), ("close_tab", {"tab_id": 2})])
def test_tab_tools_have_strict_arguments(name, args):
    validate_tool_call(tool(name, args))
    with pytest.raises((ValidationError, ValueError)):
        validate_tool_call(tool(name, {**args, "extra": "not accepted"}))


@pytest.mark.parametrize("value", [-1, True, "2", 2.5])
def test_tab_ids_are_exact_nonnegative_integers(value):
    with pytest.raises(ValidationError):
        validate_tool_call(tool("switch_tab", {"tab_id": value}))


def test_unrelated_tabs_and_extra_browser_fields_are_redacted_before_model_context():
    rows = [{"tab_id": 1, "active": True, "allowed": True, "title": "Fixture A", "location": A.url, "revision": "revision-1", "opened_by_agent": True},
        {"tab_id": 8, "active": False, "allowed": False, "title": "UNRELATED_TITLE", "url": "https://private.example/path?secret=value", "content": "UNRELATED_CONTENT"},
        {"tab_id": 9, "active": False, "allowed": True, "title": "PORTAL_TITLE", "location": "chrome://settings"}]
    result = model_safe_result(ActionResult(success=True, action="get_tabs", snapshot_id=A.snapshot_id, data={"tabs": rows}), A)
    serialized = result.model_dump_json()
    assert all(canary not in serialized for canary in ["UNRELATED", "private.example", "PORTAL_TITLE", "revision-1"])
    assert result.data["tabs"][1] == {"tab_id": 8, "active": False, "allowed": False, "details": "redacted"}
    assert result.data["tabs"][0]["opened_by_agent"] is True
    assert len(safe_tabs(rows * 500)) <= 100


@pytest.mark.parametrize("field,value", [("tab_id", 2), ("document_id", "replacement-document")])
def test_approval_identity_does_not_cross_tabs_or_documents(field, value):
    call = tool("click", {"element_id": "f0:e-003"}, A.snapshot_id)
    assert action_identity(call, A) != action_identity(call, A.model_copy(update={field: value}))


def test_identical_document_labels_in_different_tabs_still_force_full_state():
    current = A.model_copy(update={"tab_id": 2, "snapshot_id": "new-snapshot"})
    observation = page_observation(A, current)
    assert observation["type"] == "page_state_full"
    assert observation["full_snapshot_reason"] == "tab_changed"
    assert observation["tab_id"] == 2


def tab_result(revision="r-1", allowed=True):
    return ActionResult(success=True, action="get_tabs", snapshot_id=A.snapshot_id,
        data={"tabs": [{"tab_id": 2, "active": False, "allowed": allowed, "opened_by_agent": True,
            "title": "Fixture B", "location": B.url, "revision": revision}]})


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["allow_once", "allow_run"])
async def test_close_uses_one_shot_exact_tab_approval(choice):
    bridge = FakeBridge([tab_result(), tab_result()], confirmation_choices=[choice])
    call, failure, confirmation = await authorize_close(tool("close_tab", {"tab_id": 2}), A,
        PermissionManager(SecuritySettings(), False), bridge, AgentSettings(), 2)
    assert failure is None and isinstance(call, ApprovedTabClose)
    assert call.arguments == {"tab_id": 2} and call.tab_revision == "r-1"
    assert confirmation == "allow_once"
    assert bridge.confirmation_scopes == ["close_tab:2"]
    assert [call.tool for call in bridge.calls] == ["get_tabs", "get_tabs"]


@pytest.mark.asyncio
async def test_close_cancel_never_produces_an_executable_approval():
    bridge = FakeBridge([tab_result()], confirmation_choices=["cancel"])
    call, failure, _ = await authorize_close(tool("close_tab", {"tab_id": 2}), A,
        PermissionManager(SecuritySettings(), False), bridge, AgentSettings(), 2)
    assert failure is not None and not failure.success
    assert not isinstance(call, ApprovedTabClose)
    assert len(bridge.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("second", [tab_result("changed-revision"), tab_result(allowed=False)])
async def test_changed_or_unsupported_tab_invalidates_close_approval(second):
    bridge = FakeBridge([tab_result(), second], confirmation_choices=["allow_once"])
    call, failure, _ = await authorize_close(tool("close_tab", {"tab_id": 2}), A,
        PermissionManager(SecuritySettings(), False), bridge, AgentSettings(), 2)
    assert failure.error == "stale_snapshot" and not isinstance(call, ApprovedTabClose)


@pytest.mark.asyncio
async def test_close_cannot_bypass_exact_tab_validation(tmp_path):
    bridge = WebSocketBridge(AsyncMock(), "run", "turn", "session", AuditLogger(tmp_path / "audit.jsonl"))
    result = await bridge.execute(tool("close_tab", {"tab_id": 2}), 2, 30)
    assert result.error == "permission_denied"
    bridge.websocket.send_text.assert_not_awaited()
    fake = FakeBridge([])
    _call, result, _ = await authorize_close(tool("close_tab", {"tab_id": 2}), A,
        None, fake, AgentSettings(), 2)
    assert result.error == "permission_denied" and fake.calls == []


@pytest.mark.asyncio
async def test_paused_run_has_no_provider_or_browser_work_until_resume():
    control = RunControl(); control.invalidate("Paused")
    provider = MockProvider(scripted=["Done"])
    bridge = FakeBridge([observed(B)])
    task = asyncio.create_task(run_agent("Read current page", A, ["http://127.0.0.1:8080/*"], provider, AgentSettings(), bridge, control=control))
    await asyncio.sleep(0.05)
    assert provider.calls == [] and bridge.calls == []
    control.resume()
    outcome = await asyncio.wait_for(task, 3)
    assert outcome.text == "Done"
    assert bridge.calls[0].tool == "get_page_state" and bridge.calls[0].arguments == {"full": True}
    assert "NEW_PAGE_BODY_CANARY" in str(provider.calls[0])
    assert "OLD_PAGE_BODY_CANARY" not in str(provider.calls[0])


@pytest.mark.asyncio
async def test_provider_in_flight_is_cancelled_and_old_decision_never_executes():
    entered, cancelled = asyncio.Event(), asyncio.Event()
    class SlowProvider(MockProvider):
        async def generate(self, messages, tools=None, response_schema=None):
            if not self.calls:
                self.calls.append(messages)
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set(); raise
            return await super().generate(messages, tools, response_schema)
    provider = SlowProvider(scripted=["Resumed answer"])
    control = RunControl(); bridge = FakeBridge([observed(A), observed(B)])
    task = asyncio.create_task(run_agent("Original goal", A, ["http://127.0.0.1:8080/*"], provider, AgentSettings(), bridge, control=control))
    await entered.wait(); control.invalidate("Paused")
    await cancelled.wait(); await asyncio.sleep(0.05)
    assert len(provider.calls) == 1 and len(bridge.calls) == 1
    control.resume()
    outcome = await asyncio.wait_for(task, 3)
    assert outcome.text == "Resumed answer"
    assert [call.tool for call in bridge.calls] == ["get_page_state", "get_page_state"]
    assert bridge.calls[-1].arguments == {"full": True}
    assert "OLD_PAGE_BODY_CANARY" not in str(provider.calls[-1])
    assert "NEW_PAGE_BODY_CANARY" in str(provider.calls[-1])
    assert "Original goal" in str(provider.calls[-1])


@pytest.mark.asyncio
async def test_pause_during_confirmation_invalidates_late_allow(tmp_path):
    socket = AsyncMock()
    bridge = WebSocketBridge(socket, "run", "turn", "session", AuditLogger(tmp_path / "audit.jsonl"))
    from browser.security import RiskAssessment
    pending = asyncio.create_task(bridge.confirm(tool("click", {"element_id": "f0:e-003"}), RiskAssessment("high", ("delete",), "target", "Delete"), 2))
    await until(lambda: bool(bridge._confirmations))
    key = next(iter(bridge._confirmations))
    bridge.control.invalidate("Paused")
    assert bridge.resolve_confirmation(key, ConfirmationResponsePayload(decision="allow_once"))
    with pytest.raises(ControlInterrupted): await pending
    bridge.control.resume()
    assert bridge.resolve_confirmation(key, ConfirmationResponsePayload(decision="allow_once"))
    assert bridge._confirmations == {}
    with pytest.raises(ControlInterrupted):
        await bridge.execute(tool("click", {"element_id": "f0:e-003"}), 2, 30)


@pytest.mark.asyncio
async def test_target_closure_interrupts_pending_request_without_waiting_for_timeout(tmp_path):
    bridge = WebSocketBridge(AsyncMock(), "run", "turn", "session", AuditLogger(tmp_path / "audit.jsonl"))
    bridge.target = A
    pending = asyncio.create_task(bridge.execute(tool("click", {"element_id": "f0:e-003"}), 2, 30))
    await until(lambda: bool(bridge._pending))
    key = next(iter(bridge._pending))
    bridge.resolve(key, ToolResultPayload(result=ActionResult(success=False, action="click", snapshot_id=A.snapshot_id,
        error="target_tab_closed", message="Target tab closed")))
    with pytest.raises(ControlInterrupted): await asyncio.wait_for(pending, 1)
    assert bridge.control.state == "paused" and not bridge._pending
    records = [json.loads(line) for line in bridge.audit_logger.path.read_text().splitlines()]
    assert [row["tool"] for row in records] == ["click", "stale_action_rejected"]


@pytest.mark.asyncio
async def test_dispatch_binding_uses_target_tab_and_document(tmp_path):
    socket = AsyncMock()
    bridge = WebSocketBridge(socket, "run", "turn", "session", AuditLogger(tmp_path / "audit.jsonl")); bridge.target = A
    pending = asyncio.create_task(bridge.execute(tool("get_page_state", {}, A.snapshot_id), 2, 30))
    await until(lambda: bool(socket.send_text.await_args))
    frame = json.loads(socket.send_text.await_args.args[0])
    assert frame["payload"]["tab_id"] == 1 and frame["payload"]["document_id"] == "document-A"
    bridge.resolve(frame["id"], ToolResultPayload(result=observed(B)))
    with pytest.raises(ControlInterrupted):
        await pending
    assert bridge.control.state == "paused"
    assert bridge.target == A


@pytest.mark.asyncio
async def test_resume_outside_scope_waits_without_provider_or_repeated_browser_reads():
    control = RunControl(); control.invalidate("Paused"); control.resume()
    denied = ActionResult(success=False, action="get_page_state", snapshot_id=A.snapshot_id, error="permission_denied", message="Outside scope")
    bridge = FakeBridge([denied, observed(B)]); provider = MockProvider(scripted=["Done"])
    task = asyncio.create_task(run_agent("Read", A, [], provider, AgentSettings(), bridge, control=control))
    await until(lambda: control.state == "waiting")
    await asyncio.sleep(0.05)
    assert len(bridge.calls) == 1 and provider.calls == []
    control.resume()
    assert (await asyncio.wait_for(task, 3)).text == "Done"


@pytest.mark.asyncio
async def test_provider_ignoring_cancellation_cannot_delay_pause_or_return_stale_decision():
    control = RunControl()
    entered, release = asyncio.Event(), asyncio.Event()
    async def ignores_cancellation():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return "stale decision"
    pending = asyncio.create_task(control.interruptible(ignores_cancellation()))
    await entered.wait()
    control.invalidate("Paused")
    with pytest.raises(ControlInterrupted):
        await asyncio.wait_for(pending, 1)
    assert control.state == "paused"
    control.resume()
    await control.complete_refresh()
    assert await control.interruptible(asyncio.sleep(0, result="fresh decision")) == "fresh decision"
    release.set()
    await until(lambda: not control._discarded)


@pytest.mark.asyncio
async def test_pause_between_authorization_and_dispatch_rejects_queued_action(tmp_path):
    socket = AsyncMock()
    bridge = WebSocketBridge(socket, "run", "turn", "session", AuditLogger(tmp_path / "audit.jsonl"))
    bridge.target = A
    await bridge._send_lock.acquire()
    pending = asyncio.create_task(bridge.execute(tool("click", {"element_id": "f0:e-003"}), 2, 30))
    await until(lambda: bool(bridge._pending))
    bridge.control.invalidate("Paused")
    bridge._send_lock.release()
    with pytest.raises(ControlInterrupted):
        await asyncio.wait_for(pending, 1)
    socket.send_text.assert_not_awaited()
    assert not bridge._pending
