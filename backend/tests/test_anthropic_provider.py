"""The Anthropic provider against a local server speaking the real wire format.

This exercises the actual `anthropic` SDK over a real SSE stream, so the
streaming parse, the system/messages split, the usage mapping and the error
handling are covered without an API key and without leaving loopback. What it
cannot cover is the behaviour of Anthropic's own service.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from browser.actions import tool_schemas
from browser.providers import (
    AnthropicProvider,
    CompletionChunk,
    CompletionResult,
    FinalAnswer,
    Message,
    ProviderResponse,
    ProviderError,
    ToolCall,
    ToolSchema,
)

MESSAGES = [
    Message(role="system", content="You are a page assistant."),
    Message(role="user", content="Summarize this page"),
]

#: Requests the fake server saw, so tests can assert on what was actually sent.
RECEIVED: list[dict[str, Any]] = []


def sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def message_stream(
    pieces: list[str], stop_reason: str = "end_turn", stop_details: Any = None
) -> Iterator[str]:
    yield sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 42, "output_tokens": 0},
            },
        },
    )
    yield sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    for piece in pieces:
        yield sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": piece},
            },
        )
    yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})

    delta: dict[str, Any] = {"stop_reason": stop_reason, "stop_sequence": None}
    if stop_details is not None:
        delta["stop_details"] = stop_details
    yield sse(
        "message_delta",
        {"type": "message_delta", "delta": delta, "usage": {"output_tokens": 7}},
    )
    yield sse("message_stop", {"type": "message_stop"})


def tool_stream() -> Iterator[str]:
    yield sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_tool",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 52, "output_tokens": 0},
            },
        },
    )
    yield sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_test",
                "name": "search_documents",
                "input": {},
            },
        },
    )
    for piece in [
        '{"query":"photonics",',
        '"snapshot_id":"s-1",',
        '"reason":"Search the current website"}',
    ]:
        yield sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": piece},
            },
        )
    yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 11},
        },
    )
    yield sse("message_stop", {"type": "message_stop"})


def build_fake_anthropic() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/messages")
    async def messages(request: Request) -> object:
        body = await request.json()
        RECEIVED.append({"body": body, "headers": dict(request.headers)})
        prompt = body["messages"][-1]["content"]

        if body.get("tools"):
            return StreamingResponse(tool_stream(), media_type="text/event-stream")
        if body.get("output_config", {}).get("format"):
            return StreamingResponse(
                message_stream(['{"type":"final","answer":"Structured answer"}']),
                media_type="text/event-stream",
            )
        if prompt == "boom-400":
            return JSONResponse(
                {"type": "error", "error": {"type": "invalid_request_error",
                                            "message": "you asked for it"}},
                status_code=400,
            )
        if prompt == "boom-429":
            return JSONResponse(
                {"type": "error", "error": {"type": "rate_limit_error",
                                            "message": "slow down"}},
                status_code=429,
            )
        if prompt == "refuse-silently":
            return StreamingResponse(
                message_stream(
                    [],
                    stop_reason="refusal",
                    stop_details={"type": "refusal", "category": "cyber",
                                  "explanation": "declined"},
                ),
                media_type="text/event-stream",
            )
        if prompt == "hit-the-cap":
            return StreamingResponse(
                message_stream(["cut off here"], stop_reason="max_tokens"),
                media_type="text/event-stream",
            )

        return StreamingResponse(
            message_stream(["Northwind ", "covers ", "travel ", "policy."]),
            media_type="text/event-stream",
        )

    return app


@pytest.fixture(scope="module")
def fake_anthropic_url() -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    config = uvicorn.Config(
        build_fake_anthropic(), host="127.0.0.1", port=port, log_level="error"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(200):
        if server.started:
            break
        threading.Event().wait(0.05)
    else:
        raise RuntimeError("fake Anthropic server did not start")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def clear_received() -> None:
    RECEIVED.clear()


@pytest.fixture
def provider(fake_anthropic_url: str) -> AnthropicProvider:
    return AnthropicProvider(
        api_key="sk-ant-not-a-real-key",
        model="claude-opus-5",
        base_url=fake_anthropic_url,
        timeout_seconds=10.0,
    )


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


async def test_streams_chunks_then_a_result(provider: AnthropicProvider) -> None:
    items = [item async for item in provider.stream(MESSAGES)]

    assert isinstance(items[-1], CompletionResult)
    assert all(isinstance(i, CompletionChunk) for i in items[:-1])
    assert [i.text for i in items[:-1]] == [
        "Northwind ",
        "covers ",
        "travel ",
        "policy.",
    ]


async def test_final_text_is_the_reassembled_stream(
    provider: AnthropicProvider,
) -> None:
    result = await provider.complete(MESSAGES)

    assert result.text == "Northwind covers travel policy."
    assert result.finish_reason == "end_turn"


async def test_usage_is_normalised_to_the_shared_key_names(
    provider: AnthropicProvider,
) -> None:
    """input_tokens/output_tokens become prompt_/completion_ like the other provider."""
    result = await provider.complete(MESSAGES)

    assert result.usage["prompt_tokens"] == 42
    assert result.usage["completion_tokens"] == 7
    assert result.usage["total_tokens"] == 49


async def test_max_tokens_stop_reason_is_preserved(
    provider: AnthropicProvider,
) -> None:
    result = await provider.complete([Message(role="user", content="hit-the-cap")])

    assert result.finish_reason == "max_tokens"
    assert result.text == "cut off here"


# --------------------------------------------------------------------------
# Section 14 normalized generation
# --------------------------------------------------------------------------


async def test_generate_returns_a_normalized_final_answer(
    provider: AnthropicProvider,
) -> None:
    response = await provider.generate(MESSAGES)

    assert isinstance(response, ProviderResponse)
    assert isinstance(response.decision, FinalAnswer)
    assert response.decision.answer == "Northwind covers travel policy."
    assert response.raw_text == response.decision.answer
    assert response.usage["total_tokens"] == 49


async def test_native_tool_call_is_normalized_from_anthropic_sse(
    provider: AnthropicProvider,
) -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "snapshot_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["query", "snapshot_id", "reason"],
        "additionalProperties": False,
    }
    response = await provider.generate(
        MESSAGES,
        tools=[
            ToolSchema(
                name="search_documents",
                description="Search the current site",
                input_schema=schema,
            )
        ],
    )

    assert isinstance(response.decision, ToolCall)
    assert response.decision.tool == "search_documents"
    assert response.decision.arguments == {"query": "photonics"}
    assert response.decision.snapshot_id == "s-1"
    assert response.decision.reason == "Search the current website"
    assert response.finish_reason == "tool_use"
    assert RECEIVED[0]["body"]["tools"] == [
        {
            "name": "search_documents",
            "description": "Search the current site",
            "input_schema": schema,
            "strict": True,
        }
    ]


def test_get_page_state_is_emitted_as_a_strict_anthropic_tool(
    provider: AnthropicProvider,
) -> None:
    request = provider._build_request(MESSAGES, tools=tool_schemas())
    emitted = next(
        item
        for item in request["tools"]
        if item["name"] == "get_page_state"
    )
    schema = emitted["input_schema"]

    assert emitted["strict"] is True
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"full", "snapshot_id", "reason"}
    assert set(schema["required"]) == {"snapshot_id", "reason"}


async def test_response_schema_uses_native_structured_output(
    provider: AnthropicProvider,
) -> None:
    response = await provider.generate(MESSAGES, response_schema=FinalAnswer)

    assert response.decision == FinalAnswer(answer="Structured answer")
    output_format = RECEIVED[0]["body"]["output_config"]["format"]
    assert output_format["type"] == "json_schema"


# --------------------------------------------------------------------------
# Request shape
# --------------------------------------------------------------------------


async def test_system_is_hoisted_out_of_the_message_list(
    provider: AnthropicProvider,
) -> None:
    """Anthropic takes the system prompt as its own argument, not a message."""
    await provider.complete(MESSAGES)

    body = RECEIVED[0]["body"]
    assert body["system"] == "You are a page assistant."
    assert [m["role"] for m in body["messages"]] == ["user"]


async def test_temperature_is_never_sent(provider: AnthropicProvider) -> None:
    """Current Claude models reject temperature with a 400."""
    await provider.complete(MESSAGES, temperature=0.7)

    assert "temperature" not in RECEIVED[0]["body"]


async def test_max_tokens_is_sent(provider: AnthropicProvider) -> None:
    await provider.complete(MESSAGES)

    assert RECEIVED[0]["body"]["max_tokens"] == 16000


async def test_per_call_max_output_tokens_overrides_the_default(
    provider: AnthropicProvider,
) -> None:
    await provider.complete(MESSAGES, max_output_tokens=256)

    assert RECEIVED[0]["body"]["max_tokens"] == 256


async def test_effort_is_sent_as_output_config(fake_anthropic_url: str) -> None:
    provider = AnthropicProvider(
        api_key="sk-ant-not-a-real-key",
        model="claude-opus-5",
        base_url=fake_anthropic_url,
        effort="low",
    )

    await provider.complete(MESSAGES)

    assert RECEIVED[0]["body"]["output_config"] == {"effort": "low"}
    await provider.aclose()


async def test_effort_is_omitted_when_unset(provider: AnthropicProvider) -> None:
    await provider.complete(MESSAGES)

    assert "output_config" not in RECEIVED[0]["body"]


async def test_refusal_fallback_sends_the_beta_and_the_parameter(
    provider: AnthropicProvider,
) -> None:
    await provider.complete(MESSAGES)

    assert RECEIVED[0]["body"]["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in RECEIVED[0]["headers"]["anthropic-beta"]


async def test_refusal_fallback_can_be_turned_off(fake_anthropic_url: str) -> None:
    provider = AnthropicProvider(
        api_key="sk-ant-not-a-real-key",
        model="claude-opus-5",
        base_url=fake_anthropic_url,
        refusal_fallback=False,
    )

    await provider.complete(MESSAGES)

    assert "fallbacks" not in RECEIVED[0]["body"]
    assert "anthropic-beta" not in RECEIVED[0]["headers"]
    await provider.aclose()


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------


async def test_a_silent_refusal_becomes_an_error_not_a_blank_answer(
    provider: AnthropicProvider,
) -> None:
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete([Message(role="user", content="refuse-silently")])

    assert "declined" in str(excinfo.value)


async def test_a_client_error_is_not_retryable(provider: AnthropicProvider) -> None:
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete([Message(role="user", content="boom-400")])

    assert excinfo.value.retryable is False
    assert "400" in str(excinfo.value)


async def test_a_rate_limit_is_retryable(provider: AnthropicProvider) -> None:
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete([Message(role="user", content="boom-429")])

    assert excinfo.value.retryable is True


async def test_an_unreachable_backend_is_retryable() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]

    provider = AnthropicProvider(
        api_key="sk-ant-not-a-real-key",
        model="claude-opus-5",
        base_url=f"http://127.0.0.1:{dead_port}",
        timeout_seconds=2.0,
    )

    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(MESSAGES)

    assert excinfo.value.retryable is True
    await provider.aclose()


async def test_a_system_only_message_list_is_rejected(
    provider: AnthropicProvider,
) -> None:
    """Anthropic needs at least one real turn; fail clearly rather than 400."""
    with pytest.raises(ProviderError, match="non-system"):
        await provider.complete([Message(role="system", content="only this")])


async def test_no_api_key_is_rejected_at_construction() -> None:
    with pytest.raises(ProviderError, match="API key"):
        AnthropicProvider(api_key="", model="claude-opus-5")
