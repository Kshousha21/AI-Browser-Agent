"""Synthetic loopback fixtures exercising the actual vendor SDK HTTP boundary."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from browser.provider_resolution import resolve_provider
from browser.providers import build_provider

CASES = (
    "anthropic", "openai", "azure_openai", "ollama",
    "openai_compatible", "anthropic_compatible", "azure_versioned",
    "openai_responses", "azure_responses", "internal_responses",
)
PROTOCOLS = ("anthropic", "openai", "ollama", "openai_responses")
MESSAGES_TEXT = "Search for photonics in this synthetic local fixture."


def protocol(case: str) -> str:
    if "responses" in case:
        return "responses"
    if "anthropic" in case:
        return "anthropic"
    return "ollama" if case == "ollama" else "openai"


@dataclass
class Reply:
    body: Any
    status: int = 200
    media: str = "application/json"
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0


class WireServer:
    def __init__(self) -> None:
        self.replies: deque[Reply] = deque()
        self.received: list[dict[str, Any]] = []
        self.url = ""
        self.app = FastAPI()

        @self.app.post("/{path:path}")
        async def receive(request: Request) -> Response:
            self.received.append({"path": request.url.path, "query": str(request.url.query),
                                  "headers": dict(request.headers), "body": await request.json(),
                                  "time": time.monotonic()})
            if not self.replies:
                return JSONResponse({"error": "fixture exhausted"}, status_code=400)
            reply = self.replies.popleft()
            if reply.delay:
                await asyncio.sleep(reply.delay)
            if isinstance(reply.body, (dict, list)):
                return JSONResponse(reply.body, status_code=reply.status, headers=reply.headers)
            return Response(reply.body, status_code=reply.status, media_type=reply.media, headers=reply.headers)

    def reset(self) -> None:
        self.replies.clear()
        self.received.clear()


@contextmanager
def running_server():
    fixture = WireServer()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        fixture.url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(fixture.app, log_level="error"))
        thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
        thread.start()
        for _ in range(200):
            if server.started:
                break
            threading.Event().wait(0.01)
        else:
            raise RuntimeError("loopback provider fixture failed to start")
        try:
            yield fixture
        finally:
            server.should_exit = True
            thread.join(timeout=5)


def make_provider(case: str, server: WireServer, **overrides: Any):
    family = {"azure_versioned": "azure_openai", "azure_responses": "azure_openai",
              "openai_responses": "openai", "internal_responses": "openai_compatible"}.get(case, case)
    profile = {"type": family, "model": "synthetic-model", "base_url": server.url,
               "api_key_env": "FIXTURE_KEY", "max_retries": 0, "timeout_seconds": 2,
               "max_output_tokens": 128, "refusal_fallback": False}
    environment = {"FIXTURE_KEY": "synthetic-secret-marker", "ENDPOINT": server.url,
                   "DEPLOYMENT": "synthetic-deployment", "VERSION": "2024-10-21"}
    if family == "azure_openai":
        profile.update(endpoint_env="ENDPOINT", deployment_env="DEPLOYMENT")
        profile.pop("base_url")
        if case == "azure_versioned":
            profile["api_version_env"] = "VERSION"
    elif protocol(case) == "openai" or protocol(case) == "responses":
        profile["base_url"] += "/v1"
    if protocol(case) == "responses":
        profile["api"] = "responses"
    profile.update(overrides)
    settings = resolve_provider({"active": "test"}, {"test": profile}, environment)
    return build_provider(settings), settings


def sse(event: str, data: Any) -> str:
    return (f"event: {event}\n" if event else "") + f"data: {json.dumps(data)}\n\n"


def answer(case: str, *, name: str | None = None, arguments: Any = None,
           text: str = "Found synthetic results.", usage: bool = True) -> Reply:
    kind = protocol(case)
    if kind == "anthropic":
        block = ({"type": "tool_use", "id": "call-fixture", "name": name, "input": {}}
                 if name else {"type": "text", "text": ""})
        start = {"id": "msg-fixture", "type": "message", "role": "assistant",
                 "model": "synthetic-model", "content": [], "stop_reason": None,
                 "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 0} if usage else {}}
        pieces = [sse("message_start", {"type": "message_start", "message": start}),
                  sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": block})]
        delta = ({"type": "input_json_delta", "partial_json": json.dumps(arguments)}
                 if name else {"type": "text_delta", "text": text})
        pieces += [sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
                   sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
                   sse("message_delta", {"type": "message_delta", "delta": {
                       "stop_reason": "tool_use" if name else "end_turn", "stop_sequence": None},
                       "usage": {"output_tokens": 5} if usage else {}}),
                   sse("message_stop", {"type": "message_stop"})]
        return Reply("".join(pieces), media="text/event-stream")
    if kind == "ollama":
        message: dict[str, Any] = {"role": "assistant", "content": "" if name else text}
        if name:
            message.update(tool_calls=[{"function": {"name": name, "arguments": arguments}}], thinking="synthetic planning")
        body = {"model": "synthetic-model", "message": message, "done": True, "done_reason": "stop"}
        if usage:
            body.update(prompt_eval_count=10, eval_count=5)
        return Reply(body)
    if kind == "responses":
        output = ([{"type": "reasoning", "id": "rs-fixture", "summary": [], "encrypted_content": "synthetic-opaque"},
                   {"type": "function_call", "id": "fc-fixture", "call_id": "call-fixture", "name": name,
                    "arguments": json.dumps(arguments), "status": "completed"}]
                  if name else [{"type": "message", "id": "msg-fixture", "role": "assistant", "status": "completed",
                                 "content": [{"type": "output_text", "text": text, "annotations": []}]}])
        return Reply({"id": "resp-fixture", "object": "response", "created_at": 1,
                      "model": "synthetic-model", "status": "completed", "error": None,
                      "output": output, "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15} if usage else None})
    message = {"role": "assistant", "content": None if name else text}
    if name:
        message["tool_calls"] = [{"id": "call-fixture", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments)}}]
    return Reply({"id": "chat-fixture", "object": "chat.completion", "created": 1,
                  "model": "synthetic-model", "choices": [{"index": 0, "message": message,
                  "finish_reason": "tool_calls" if name else "stop"}],
                  "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15} if usage else None})


def streamed_answer(case: str, *, terminal: bool = True) -> Reply:
    kind = protocol(case)
    if kind == "anthropic":
        result = answer(case, text="Synthetic answer.")
        if not terminal:
            result.body = result.body.split("event: message_delta")[0]
        return result
    if kind == "ollama":
        pieces = [json.dumps({"message": {"role": "assistant", "content": part}, "done": False})
                  for part in ("Synthetic ", "answer.")]
        if terminal:
            pieces.append(json.dumps(answer(case, text="").body))
        return Reply("\n".join(pieces) + "\n", media="application/x-ndjson")
    if kind == "responses":
        pieces = [sse("response.output_text.delta", {"type": "response.output_text.delta", "delta": part,
                   "item_id": "msg-fixture", "output_index": 0, "content_index": 0, "sequence_number": index})
                  for index, part in enumerate(("Synthetic ", "answer."))]
        if terminal:
            pieces.append(sse("response.completed", {"type": "response.completed", "sequence_number": 2,
                          "response": answer(case, text="Synthetic answer.").body}))
        return Reply("".join(pieces), media="text/event-stream")
    def chunk(delta: dict, finish=None, usage=None):
        return {"id": "chat-fixture", "object": "chat.completion.chunk", "created": 1, "model": "synthetic-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], "usage": usage}
    pieces = [sse("", chunk({"content": part})) for part in ("Synthetic ", "answer.")]
    if terminal:
        pieces += [sse("", chunk({}, "stop", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})), "data: [DONE]\n\n"]
    return Reply("".join(pieces), media="text/event-stream")
