"""Empty searches and expired observations must not silently wait for Resume."""
import asyncio
import json
from unittest.mock import AsyncMock
import pytest
from browser.actions import ActionResult
from browser.agent import run_agent
from browser.audit import AuditLogger
from browser.config import AgentSettings
from browser.protocol import ToolResultPayload
from browser.providers import MockProvider, ToolCall
from browser.session_bridge import WebSocketBridge
from browser.session_control import ControlInterrupted
from browser.search_recovery import empty_search, validate_empty_search_retry
from .test_agent import PAGE, FakeBridge
from .test_session_control import observed


@pytest.mark.parametrize("error,research,interrupt", [
    ("stale_snapshot", False, False), ("stale_snapshot", True, True),
    ("target_document_changed", False, True), ("target_tab_changed", False, True),
    ("run_paused", False, True)])
async def test_websocket_only_ordinary_staleness_is_recoverable(tmp_path, error, research, interrupt):
    socket = AsyncMock()
    bridge = WebSocketBridge(socket, "run", "turn", "session", AuditLogger(tmp_path / "audit.jsonl"))
    bridge.target = PAGE; bridge.research = research
    async def reply(text):
        envelope = json.loads(text)
        if envelope["type"] == "tool_request":
            bridge.resolve(envelope["id"], ToolResultPayload(result=ActionResult(success=False,
                action="read_page", snapshot_id=PAGE.snapshot_id, error=error)))
    socket.send_text.side_effect = reply
    call = ToolCall(reason="Fixture recovery", tool="read_page", arguments={}, snapshot_id=PAGE.snapshot_id)
    if interrupt:
        with pytest.raises(ControlInterrupted):
            await asyncio.wait_for(bridge.execute(call, 2, 10), 2)
        assert bridge.control.state == "paused"
    else:
        result = await asyncio.wait_for(bridge.execute(call, 2, 10), 2)
        assert result.error == "stale_snapshot"
        assert bridge.control.state == "running" and bridge.control.epoch == 0


async def test_agent_refreshes_stale_observation_without_another_model_call_or_replaying_action():
    fresh = PAGE.model_copy(update={"snapshot_id":"fresh", "visible_text":"FRESH_SOURCE_DETAILS"})
    bridge = FakeBridge([observed(PAGE), ActionResult(success=False, action="read_page",
        snapshot_id=PAGE.snapshot_id, error="stale_snapshot"), observed(fresh)])
    provider = MockProvider(scripted=[ToolCall(reason="Fixture recovery", tool="read_page", arguments={}, snapshot_id=PAGE.snapshot_id),
        "FRESH_SOURCE_DETAILS"])
    result = await run_agent("Read the procedure", PAGE, ["http://127.0.0.1:8080/*"], provider, AgentSettings(), bridge)
    assert result.text == "FRESH_SOURCE_DETAILS"
    assert [c.tool for c in bridge.calls] == ["get_page_state", "read_page", "get_page_state"]
    assert len(provider.calls) == 2
    assert "fresh" in repr(provider.calls[-1]) and "page_state_full" in repr(provider.calls[-1])
    assert "[RETIRED PAGE CONTEXT]" in repr(provider.calls[-1])


@pytest.mark.parametrize("failure", ["unavailable", "owner_changed"])
async def test_failed_stale_refresh_stops_without_another_decision(failure):
    refresh = (ActionResult(success=False, action="get_page_state", snapshot_id=PAGE.snapshot_id, error="injection_failed")
        if failure == "unavailable" else observed(PAGE.model_copy(update={"document_id":"different"})))
    bridge = FakeBridge([observed(PAGE), ActionResult(success=False, action="read_page", snapshot_id=PAGE.snapshot_id, error="stale_snapshot"), refresh])
    provider = MockProvider(scripted=[ToolCall(reason="Fixture recovery", tool="read_page", arguments={}, snapshot_id=PAGE.snapshot_id)])
    result = await run_agent("Read the procedure", PAGE, ["http://127.0.0.1:8080/*"], provider, AgentSettings(), bridge)
    assert result.finish_reason == "stale_snapshot" and len(provider.calls) == 1






async def test_repeated_expired_snapshots_have_two_refresh_limit():
    pages = [PAGE.model_copy(update={"snapshot_id":f"s-{i}"}) for i in range(3)]
    results = [observed(pages[0])]
    calls = []
    for i, page in enumerate(pages):
        calls.append(ToolCall(tool="read_page", arguments={}, snapshot_id=page.snapshot_id, reason="Read current source"))
        results.append(ActionResult(success=False, action="read_page", snapshot_id=page.snapshot_id, error="stale_snapshot"))
        if i < 2:
            results.append(observed(pages[i+1]))
    bridge = FakeBridge(results)
    provider = MockProvider(scripted=calls)
    outcome = await run_agent("Read the procedure", pages[0], ["http://127.0.0.1:8080/*"], provider, AgentSettings(), bridge)
    assert outcome.finish_reason == "stale_snapshot"
    assert len(provider.calls) == 3 and len(bridge.calls) == 6


async def test_same_click_can_execute_after_stale_refresh_on_unchanged_page():
    from browser.config import SecuritySettings
    from browser.protocol import PageState, PageElement
    original=PageState(url="https://example.org/",tab_id=1,document_id="d",snapshot_id="s",
        elements=[PageElement(agent_id="f0:e-001",tag="a",role="link",name="Guide",href="https://example.org/guide")])
    destination=original.model_copy(update={"url":"https://example.org/guide","document_id":"guide","snapshot_id":"next"})
    refreshed = original.model_copy(update={"snapshot_id":"fresh-click-snapshot"})
    click = ToolCall(tool="click", arguments={"element_id":original.elements[0].agent_id},
        snapshot_id=original.snapshot_id, reason="Open the observed source link")
    bridge = FakeBridge([observed(original),
        ActionResult(success=False, action="click", snapshot_id=original.snapshot_id, error="stale_snapshot"),
        observed(refreshed),
        ActionResult(success=True, action="click", snapshot_id=destination.snapshot_id,
            page_changed=True, data={"page_state":destination.model_dump()})])
    provider = MockProvider(scripted=[click, click.model_copy(update={"snapshot_id":refreshed.snapshot_id}), "Source opened."])
    outcome = await run_agent("Open the observed source", original, [],
        provider, AgentSettings(), bridge, security_settings=SecuritySettings())
    assert outcome.finish_reason == "stop"
    assert [c.tool for c in bridge.calls] == ["get_page_state", "click", "get_page_state", "click"]
    assert bridge.calls[-1].snapshot_id == refreshed.snapshot_id
    assert not any(confirmation == "loop_blocked" for _, confirmation in bridge.audit_records)
    assert not bridge.confirmation_scopes


def test_discarding_one_rejected_attempt_preserves_other_loop_history():
    from browser.agent_support import LoopDetector
    detector = LoopDetector()
    rejected = ToolCall(tool="click", arguments={"element_id":"f0:e-003"}, snapshot_id=PAGE.snapshot_id, reason="Read source")
    prior = rejected.model_copy(update={"arguments":{"element_id":"f0:e-004"}})
    assert detector.record(prior, PAGE) == 1
    assert detector.record(rejected, PAGE) == 1
    detector.discard_rejected_attempt(detector.fingerprint(rejected, PAGE), detector.semantic_fingerprint(rejected))
    assert detector.record(rejected, PAGE) == 1
    assert detector.record(prior, PAGE) == 2
    assert detector.record(prior, PAGE) == 3
