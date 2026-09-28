"""Cross-provider acceptance through real loopback HTTP/SSE, without API keys."""

from __future__ import annotations

import asyncio
import json

import pytest

from browser.actions import ActionResult, tool_schemas
from browser.agent import run_agent
from browser.config import AgentSettings, SecuritySettings
from browser.providers import CompletionChunk, CompletionResult, FinalAnswer, Message, MockProvider, ProviderError, ToolCall
from tests.provider_wire import CASES, PROTOCOLS, Reply, answer, make_provider, protocol, running_server, streamed_answer
from tests.test_agent import FakeBridge, PAGE, tool

pytestmark = pytest.mark.asyncio
MESSAGES = [Message(role="system", content="Use the synthetic local search."),
            Message(role="user", content="Search for photonics")]
META = {"snapshot_id": "s-start", "reason": "Use synthetic search"}


@pytest.fixture(scope="module")
def server():
    with running_server() as instance:
        yield instance


@pytest.fixture
def wire(server):
    server.reset()
    return server


@pytest.mark.parametrize("case", (*CASES, "mock"))
async def test_same_agent_contract_type_click_final(case, wire):
    steps = [("type_text", {"element_id": "f0:e-002", "text": "photonics"}),
             ("click", {"element_id": "f0:e-003"})]
    if case == "mock":
        provider = MockProvider(scripted=[*(tool(name, args) for name, args in steps), "Found synthetic results."])
    else:
        provider, _ = make_provider(case, wire)
        wire.replies.extend([*(answer(case, name=name, arguments={**args, **META}) for name, args in steps), answer(case)])
    bridge = FakeBridge(results=[
        ActionResult(success=True, action="get_page_state", snapshot_id=PAGE.snapshot_id, data={"page_state": PAGE.model_dump()}),
        *(ActionResult(success=True, action=name, snapshot_id=PAGE.snapshot_id) for name, _ in steps),
        ActionResult(success=True, action="wait_for_stable", snapshot_id=PAGE.snapshot_id),
        ActionResult(success=True, action="get_page_state", snapshot_id=PAGE.snapshot_id, data={"page_state": PAGE.model_dump()}),
    ])
    try:
        outcome = await run_agent("Search for photonics", PAGE, ["http://127.0.0.1:8080/*"], provider,
                                  AgentSettings(), bridge, security_settings=SecuritySettings(), read_only=False)
        assert outcome.text == "Found synthetic results."
        assert [(call.tool, call.arguments) for call in bridge.calls[1:3]] == steps
        assert [call.tool for call in bridge.calls] == ["get_page_state", "type_text", "click", "wait_for_stable", "get_page_state"]
        assert not bridge.confirmation_scopes
        if case == "mock":
            return
        assert len(wire.received) == 3
        assert outcome.usage == {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45}
        request = wire.received[1]["body"]
        first = wire.received[0]["body"]
        if protocol(case) == "anthropic":
            assert first["system"]
            assert first["max_tokens"] == 128
            assert first["tools"][0]["strict"] is True
            assert any(block.get("type") == "tool_result" for msg in request["messages"]
                       if isinstance(msg["content"], list) for block in msg["content"])
        elif protocol(case) == "responses":
            assert first["store"] is False
            assert first["max_output_tokens"] == 128
            assert first["tools"][0]["strict"] is True
            assert any(item.get("type") == "function_call_output" and item["call_id"] == "call-fixture" for item in request["input"])
            assert any(item.get("encrypted_content") == "synthetic-opaque" for item in request["input"])
        elif protocol(case) == "ollama":
            assert first["options"]["num_predict"] == 128
            assert first["stream"] is False
            assert any(msg.get("tool_name") == "type_text" and msg["role"] == "tool" for msg in request["messages"])
            assert any(msg.get("thinking") == "synthetic planning" for msg in request["messages"])
        else:
            assert first["max_completion_tokens"] == 128
            assert first["tools"][0]["function"]["strict"] is True
            assert first["parallel_tool_calls"] is False
            assert any(msg.get("tool_call_id") == "call-fixture" for msg in request["messages"])
        path = wire.received[0]["path"]
        if case == "azure_versioned":
            assert path == "/openai/deployments/synthetic-deployment/chat/completions"
            assert "api-version=2024-10-21" in wire.received[0]["query"]
            assert wire.received[0]["headers"]["api-key"] == "synthetic-secret-marker"
        elif case.startswith("azure"):
            assert path.startswith("/openai/v1/")
            assert wire.received[0]["query"] == ""
        elif protocol(case) == "ollama":
            assert path == "/api/chat"
        elif protocol(case) == "responses":
            assert path == "/v1/responses"
    finally:
        await provider.aclose()


@pytest.mark.parametrize("case", CASES)
async def test_answer_stream_and_usage(case, wire):
    provider, _ = make_provider(case, wire)
    wire.replies.append(streamed_answer(case))
    try:
        events = [event async for event in provider.stream(MESSAGES)]
        assert isinstance(events[-1], CompletionResult)
        assert all(isinstance(event, CompletionChunk) for event in events[:-1])
        assert "".join(event.text for event in events[:-1]) == "Synthetic answer."
        assert events[-1].text == "Synthetic answer."
        assert events[-1].usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        assert events[-1].finish_reason in {"stop", "end_turn"}
    finally:
        await provider.aclose()


@pytest.mark.parametrize("case", PROTOCOLS)
async def test_truncated_stream_is_error(case, wire):
    provider, _ = make_provider(case, wire)
    wire.replies.append(streamed_answer(case, terminal=False))
    try:
        with pytest.raises(ProviderError):
            await provider.complete(MESSAGES)
    finally:
        await provider.aclose()


@pytest.mark.parametrize("case", PROTOCOLS)
@pytest.mark.parametrize("arguments", [
    {**META, "element_id": "f0:e-003", "unexpected": "synthetic-sensitive-argument"},
    {**META, "element_id": None}, {"element_id": "f0:e-003"}, [],
])
async def test_malformed_tool_never_reaches_browser(case, arguments, wire):
    provider, _ = make_provider(case, wire)
    wire.replies.append(answer(case, name="click", arguments=arguments))
    bridge = FakeBridge(results=[ActionResult(success=True, action="get_page_state", snapshot_id=PAGE.snapshot_id,
                                             data={"page_state": PAGE.model_dump()})])
    try:
        with pytest.raises(ProviderError) as error:
            await run_agent("Search for photonics", PAGE, ["http://127.0.0.1:8080/*"], provider,
                            AgentSettings(), bridge, read_only=False)
        assert error.value.code in {"invalid_arguments", "malformed_response"}
        assert "synthetic-sensitive-argument" not in str(error.value)
        assert [call.tool for call in bridge.calls] == ["get_page_state"]
    finally:
        await provider.aclose()


@pytest.mark.parametrize("case", PROTOCOLS)
async def test_empty_normalized_arguments_and_nullable_fields(case, wire):
    from browser.providers import ToolSchema
    empty = ToolSchema(name="empty", input_schema={"type": "object", "properties": {
        "snapshot_id": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["snapshot_id", "reason"], "additionalProperties": False})
    nullable = ToolSchema(name="nullable", input_schema={"type": "object", "properties": {
        **empty.input_schema["properties"], "query": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "required": ["snapshot_id", "reason", "query"], "additionalProperties": False})
    provider, _ = make_provider(case, wire)
    wire.replies.extend([answer(case, name="empty", arguments=META), answer(case, name="nullable", arguments={**META, "query": None})])
    try:
        assert (await provider.generate(MESSAGES, [empty])).decision.arguments == {}
        assert (await provider.generate(MESSAGES, [nullable])).decision.arguments == {"query": None}
        request = wire.received[-1]["body"]["tools"][0]
        schema = (request["input_schema"] if protocol(case) == "anthropic" else
                  request["parameters"] if protocol(case) == "responses" else request["function"]["parameters"])
        assert {"type": "null"} in schema["properties"]["query"]["anyOf"]
    finally:
        await provider.aclose()


@pytest.mark.parametrize("case", PROTOCOLS)
async def test_unknown_usage_and_no_prose_action_fallback(case, wire):
    provider, _ = make_provider(case, wire)
    prose = json.dumps({"type": "tool", "tool": "click", "arguments": {"element_id": "f0:e-003"}, **META})
    wire.replies.append(answer(case, text=prose, usage=False))
    try:
        result = await provider.generate(MESSAGES, tool_schemas())
        assert isinstance(result.decision, FinalAnswer)
        assert result.usage == {}
    finally:
        await provider.aclose()


@pytest.mark.parametrize("case", PROTOCOLS)
@pytest.mark.parametrize("status,retryable,attempts", [(400, False, 1), (401, False, 1), (429, True, 2), (503, True, 2), (529, True, 2)])
async def test_http_errors_retries_and_secret_redaction(case, status, retryable, attempts, wire, caplog):
    provider, settings = make_provider(case, wire, max_retries=1)
    wire.replies.extend([Reply({"error": {"message": "synthetic-secret-marker", "type": "api_error"}}, status=status,
                               headers={"Retry-After": "0.01", "Set-Cookie": "fixture=synthetic-secret-marker"}) for _ in range(2)])
    try:
        with pytest.raises(ProviderError) as error:
            await provider.generate(MESSAGES)
        assert error.value.retryable is retryable
        assert error.value.status_code == status
        assert len(wire.received) == attempts
        if retryable:
            assert wire.received[1]["time"] - wire.received[0]["time"] >= 0.009
        assert all("cookie" not in request["headers"] for request in wire.received)
        assert "synthetic-secret-marker" not in str(error.value) + repr(settings) + settings.model_dump_json() + caplog.text
        assert error.value.__cause__ is None
    finally:
        await provider.aclose()


@pytest.mark.parametrize("case", PROTOCOLS)
async def test_transient_error_then_success(case, wire):
    provider, _ = make_provider(case, wire, max_retries=1)
    wire.replies.extend([Reply({"error": "busy"}, status=429, headers={"Retry-After": "0.01"}), answer(case)])
    try:
        assert isinstance((await provider.generate(MESSAGES)).decision, FinalAnswer)
        assert len(wire.received) == 2
    finally:
        await provider.aclose()


@pytest.mark.parametrize("case", PROTOCOLS)
async def test_timeout_and_outer_cancellation(case, wire):
    provider, _ = make_provider(case, wire, timeout_seconds=0.04)
    reply = answer(case)
    reply.delay = 0.3
    wire.replies.append(reply)
    try:
        with pytest.raises(ProviderError) as error:
            await provider.generate(MESSAGES)
        assert error.value.code == "timeout"
        assert error.value.retryable
        wire.replies.append(reply)
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01):
                await provider.generate(MESSAGES)
    finally:
        await provider.aclose()


@pytest.mark.parametrize("case", PROTOCOLS)
async def test_malformed_wire_body_is_normalized_error(case, wire):
    provider, _ = make_provider(case, wire)
    wire.replies.append(Reply("not-json"))
    try:
        with pytest.raises(ProviderError):
            await provider.generate(MESSAGES)
    finally:
        await provider.aclose()
