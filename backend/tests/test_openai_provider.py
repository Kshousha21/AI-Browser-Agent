"""The OpenAI provider against a local server that speaks the real wire format.

This exercises the actual `openai` SDK against a real HTTP SSE stream, so the
streaming parse, the usage frame and the error mapping are covered without an
API key and without leaving loopback. What it cannot cover is the behaviour of
OpenAI's own service.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from browser.providers import (
    CompletionChunk,
    CompletionResult,
    Message,
    OpenAIProvider,
    ProviderError,
)
from browser.providers.openai_provider import (
    _restore_optional_defaults,
    _strict_function_parameters,
)

MESSAGES = [Message(role="user", content="hello")]


def test_strict_tool_parameters_require_every_nullable_optional_field() -> None:
    source = {
        "type": "object",
        "properties": {
            "query": {
                "anyOf": [
                    {"type": "string", "maxLength": 500},
                    {"type": "null"},
                ],
                "default": None,
            },
            "page": {"type": "integer", "default": 1, "title": "Page"},
            "reason": {"type": "string"},
        },
        "required": ["reason"],
        "additionalProperties": False,
    }

    strict = _strict_function_parameters(source)

    assert strict["required"] == ["query", "page", "reason"]
    assert strict["additionalProperties"] is False
    assert {"type": "null"} in strict["properties"]["query"]["anyOf"]
    assert {"type": "null"} in strict["properties"]["page"]["anyOf"]
    assert "default" not in json.dumps(strict)
    assert "title" not in json.dumps(strict)
    assert "maxLength" not in json.dumps(strict)
    assert source["properties"]["query"]["default"] is None
    assert source["required"] == ["reason"]


def test_openai_transport_nulls_restore_local_optional_defaults() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "page": {"type": "integer", "default": 1},
            "reason": {"type": "string"},
        },
        "required": ["reason"],
    }
    restored = _restore_optional_defaults(
        {"query": None, "page": None, "reason": "Search."}, schema
    )
    assert restored == {"reason": "Search."}


def sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def chunk(text: str | None = None, finish: str | None = None) -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "delta": {} if text is None else {"content": text},
                "finish_reason": finish,
            }
        ],
    }


def build_fake_openai() -> FastAPI:
    """An OpenAI-compatible endpoint, driven by the prompt it receives."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> object:
        body = await request.json()
        prompt = body["messages"][-1]["content"]

        if prompt == "boom-400":
            return JSONResponse(
                {"error": {"message": "you asked for it", "type": "invalid_request"}},
                status_code=400,
            )
        if prompt == "boom-429":
            return JSONResponse(
                {"error": {"message": "slow down", "type": "rate_limit"}},
                status_code=429,
            )

        def stream() -> Iterator[str]:
            if prompt == "no-usage":
                yield sse(chunk("only "))
                yield sse(chunk("text", finish="stop"))
            elif prompt == "length-capped":
                yield sse(chunk("cut off here", finish="length"))
            else:
                for piece in ("Northwind ", "covers ", "travel ", "policy."):
                    yield sse(chunk(piece))
                yield sse(chunk(finish="stop"))
                # The trailing usage frame carries no choices at all.
                yield sse(
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "fake-model",
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 4,
                            "total_tokens": 15,
                        },
                    }
                )
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


@pytest.fixture(scope="module")
def fake_openai_url() -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    config = uvicorn.Config(
        build_fake_openai(), host="127.0.0.1", port=port, log_level="error"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # started is set once the socket is actually accepting.
    for _ in range(200):
        if server.started:
            break
        threading.Event().wait(0.05)
    else:
        raise RuntimeError("fake OpenAI server did not start")

    yield f"http://127.0.0.1:{port}/v1"

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def provider(fake_openai_url: str) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="sk-not-a-real-key",
        model="fake-model",
        base_url=fake_openai_url,
        timeout_seconds=10.0,
    )


async def test_streams_chunks_then_a_result(provider: OpenAIProvider) -> None:
    items = [item async for item in provider.stream(MESSAGES)]

    assert isinstance(items[-1], CompletionResult)
    assert all(isinstance(i, CompletionChunk) for i in items[:-1])
    assert [i.text for i in items[:-1]] == [
        "Northwind ",
        "covers ",
        "travel ",
        "policy.",
    ]


async def test_final_text_is_the_reassembled_stream(provider: OpenAIProvider) -> None:
    result = await provider.complete(MESSAGES)

    assert result.text == "Northwind covers travel policy."
    assert result.finish_reason == "stop"


async def test_usage_is_read_from_the_trailing_frame(provider: OpenAIProvider) -> None:
    """The last SSE frame has an empty `choices`; it must not be dropped."""
    result = await provider.complete(MESSAGES)

    assert result.usage == {
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
    }


async def test_a_stream_without_a_usage_frame_still_completes(
    provider: OpenAIProvider,
) -> None:
    result = await provider.complete([Message(role="user", content="no-usage")])

    assert result.text == "only text"
    assert result.usage == {}


async def test_finish_reason_is_preserved(provider: OpenAIProvider) -> None:
    result = await provider.complete([Message(role="user", content="length-capped")])

    assert result.finish_reason == "length"
    assert result.text == "cut off here"


async def test_a_client_error_is_not_retryable(provider: OpenAIProvider) -> None:
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete([Message(role="user", content="boom-400")])

    assert excinfo.value.retryable is False
    assert "400" in str(excinfo.value)


async def test_a_rate_limit_is_retryable(provider: OpenAIProvider) -> None:
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete([Message(role="user", content="boom-429")])

    assert excinfo.value.retryable is True


async def test_an_unreachable_backend_is_retryable() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]

    provider = OpenAIProvider(
        api_key="sk-not-a-real-key",
        model="fake-model",
        base_url=f"http://127.0.0.1:{dead_port}/v1",
        timeout_seconds=2.0,
    )

    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(MESSAGES)

    assert excinfo.value.retryable is True
    await provider.aclose()


async def test_aclose_releases_the_client(provider: OpenAIProvider) -> None:
    await provider.complete(MESSAGES)
    await provider.aclose()
