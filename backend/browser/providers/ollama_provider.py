"""Ollama's native /api/chat protocol, with native tools and NDJSON streaming."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from pydantic import BaseModel

from .base import (
    CompletionChunk, CompletionResult, Message, ModelProvider, ProviderError,
    ProviderResponse, ToolSchema, final_decision,
)
from .transport import protect_sdk_logs, provider_error
from .validation import usage_counts, validated_tool


class OllamaProvider(ModelProvider):
    name = "ollama"

    def __init__(
        self, *, model: str, base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 60, max_output_tokens: int = 16000,
        max_retries: int = 2, temperature: float | None = None,
        top_p: float | None = None, http_client: httpx.AsyncClient | None = None,
        num_ctx: int | None = None, think: bool | None = None,
        tool_mode: str = "native",
    ) -> None:
        self._model = model
        self._url = base_url.rstrip("/") + "/api/chat"
        self._timeout = timeout_seconds
        self._maximum = max_output_tokens
        self._retries = max_retries
        self._temperature = temperature
        self._top_p = top_p
        self._num_ctx = num_ctx
        self._think = think
        if tool_mode not in {"native", "structured"}:
            raise ProviderError("unsupported Ollama tool_mode", code="configuration_error")
        self._tool_mode = tool_mode
        self.prefers_response_schema = tool_mode == "structured"
        self._thinking: OrderedDict[str, str] = OrderedDict()
        protect_sdk_logs()
        self._client = http_client or httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=False)

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self, messages: list[Message], tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> ProviderResponse:
        if tools and response_schema:
            raise ProviderError("tools and response_schema cannot be used together")
        structured = bool(tools) and self._tool_mode == "structured"
        reroute_routes = ()
        if structured:
            from .ollama_dispatch import dispatch_messages, routing_choices
            reroute_routes = routing_choices(messages)
            messages = dispatch_messages(messages, tools, reroute_routes)
        request = self._request(messages, stream=False)
        if structured:
            from .ollama_dispatch import generation_schema
            request["format"] = generation_schema(tools, reroute_routes)
        elif tools:
            request["tools"] = [
                {"type": "function", "function": {
                    "name": tool.name, "description": tool.description,
                    "parameters": tool.input_schema,
                }} for tool in tools
            ]
        if response_schema:
            request["format"] = response_schema.model_json_schema()
        try:
            async with asyncio.timeout(self._timeout):
                response = await self._send(request)
                try:
                    data = response.json()
                finally:
                    await response.aclose()
                return self._normalize(data, tools, response_schema, reroute_routes)
        except Exception as exc:
            raise provider_error(exc, "Ollama") from None

    async def stream(
        self, messages: Sequence[Message], *, temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[CompletionChunk | CompletionResult]:
        request = self._request(messages, stream=True)
        if temperature is not None:
            request["options"]["temperature"] = temperature
        if max_output_tokens is not None:
            request["options"]["num_predict"] = max_output_tokens
        parts: list[str] = []
        final: dict[str, Any] | None = None
        try:
            async with asyncio.timeout(self._timeout):
                response = await self._send(request, stream=True)
                try:
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        message = self._checked_message(data, terminal=False)
                        if message.get("tool_calls"):
                            raise ProviderError("unexpected tool call in answer-only stream", code="invalid_response")
                        text = message.get("content", "")
                        if text:
                            parts.append(text)
                            yield CompletionChunk(text=text)
                        if data.get("done") is True:
                            final = data
                            break
                finally:
                    await response.aclose()
        except Exception as exc:
            raise provider_error(exc, "Ollama") from None
        if final is None:
            raise ProviderError("Ollama stream ended before its terminal frame", code="malformed_response")
        yield CompletionResult(
            text="".join(parts), finish_reason=final.get("done_reason") or "stop",
            usage=usage_counts(final.get("prompt_eval_count"), final.get("eval_count")),
        )

    def _request(self, messages: Sequence[Message], *, stream: bool) -> dict[str, Any]:
        turns: list[dict[str, Any]] = []
        tool_names: dict[str, str] = {}
        for message in messages:
            content = message.content
            images = []
            if isinstance(content, list):
                from .images import translate
                blocks = translate(content, "anthropic", message.role)
                images = [block["source"]["data"] for block in blocks if block["type"] == "image"]
                content = "\n".join(block["text"] for block in blocks if block["type"] == "text")
            item: dict[str, Any] = {"role": message.role, "content": content}
            if images:
                item["images"] = images
            if message.role == "assistant" and message.tool_name:
                if message.tool_arguments is None or not message.tool_call_id:
                    raise ProviderError("Ollama assistant tools require arguments and a normalized call ID")
                item["tool_calls"] = [{"function": {"name": message.tool_name, "arguments": message.tool_arguments}}]
                tool_names[message.tool_call_id] = message.tool_name
                if message.tool_call_id in self._thinking:
                    item["thinking"] = self._thinking[message.tool_call_id]
            if message.role == "tool":
                name = message.tool_name or tool_names.get(message.tool_call_id or "")
                if not name:
                    raise ProviderError("Ollama tool results require a paired native tool call")
                item["tool_name"] = name
            turns.append(item)
        options: dict[str, Any] = {"num_predict": self._maximum}
        if self._temperature is not None:
            options["temperature"] = self._temperature
        if self._top_p is not None:
            options["top_p"] = self._top_p
        if self._num_ctx is not None:
            options["num_ctx"] = self._num_ctx
        request = {"model": self._model, "messages": turns, "stream": stream, "options": options}
        if self._think is not None:
            request["think"] = self._think
        return request

    def _normalize(
        self, data: Any, tools: list[ToolSchema] | None, schema: type[BaseModel] | None,
        reroute_routes=(),
    ) -> ProviderResponse:
        message = self._checked_message(data, terminal=True)
        text = message.get("content", "")
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list) or len(calls) > 1:
            raise ProviderError("Ollama must return at most one native tool per turn", code="invalid_response")
        call_id = None
        if tools and self._tool_mode == "structured":
            if calls or data.get("done_reason") not in {None, "stop"}:
                raise ProviderError("structured dispatch response was incomplete or mixed", code="invalid_response")
            from .ollama_dispatch import dispatch_decision
            decision = dispatch_decision(text, tools, reroute_routes)
            if decision.type == "tool":
                call_id = "ollama-" + str(uuid.uuid4())
        elif calls:
            try:
                function = calls[0]["function"]
                decision = validated_tool(function["name"], function["arguments"], tools)
            except (KeyError, TypeError):
                raise ProviderError("Ollama returned a malformed native tool call", code="invalid_arguments") from None
            if data.get("done_reason") not in {None, "stop", "tool_calls"}:
                raise ProviderError("Ollama native tool response was incomplete", code="invalid_response")
            call_id = "ollama-" + str(uuid.uuid4())
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking:
                self._thinking[call_id] = thinking
                while len(self._thinking) > 64:
                    self._thinking.popitem(last=False)
        else:
            if data.get("done_reason") not in {None, "stop"}:
                raise ProviderError("Ollama final response was incomplete", code="invalid_response")
            if not text.strip():
                raise ProviderError("Ollama returned neither text nor a native tool; select a tool-capable model", code="invalid_response")
            decision = final_decision(text, schema)
        return ProviderResponse(
            decision=decision, raw_text=text, tool_call_id=call_id,
            finish_reason=data.get("done_reason") or "stop",
            usage=usage_counts(data.get("prompt_eval_count"), data.get("eval_count")),
        )

    @staticmethod
    def _checked_message(data: Any, *, terminal: bool) -> dict[str, Any]:
        if not isinstance(data, dict) or data.get("error"):
            raise ProviderError("Ollama returned an error response", code="invalid_response")
        message = data.get("message")
        if (not isinstance(message, dict) or message.get("role") != "assistant"
                or not isinstance(message.get("content", ""), str)
                or (terminal and data.get("done") is not True)):
            raise ProviderError("Ollama returned a malformed chat response", code="malformed_response")
        return message

    async def _send(self, payload: dict[str, Any], *, stream: bool = False) -> httpx.Response:
        for attempt in range(self._retries + 1):
            response: httpx.Response | None = None
            try:
                request = self._client.build_request("POST", self._url, json=payload)
                response = await self._client.send(request, stream=stream)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                delay = _retry_delay(response, attempt)
                if response is not None:
                    await response.aclose()
                error = provider_error(exc, "Ollama")
                if not error.retryable or attempt == self._retries:
                    raise error from None
                await asyncio.sleep(delay)
        raise ProviderError("Ollama retry limit reached", retryable=True)

    async def aclose(self) -> None:
        self._thinking.clear()
        await self._client.aclose()


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    value = response.headers.get("Retry-After") if response is not None else None
    if value:
        try:
            return max(0, min(float(value), 60))
        except ValueError:
            try:
                return max(0, min((parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds(), 60))
            except (ValueError, TypeError):
                pass
    return min(0.5 * 2 ** attempt, 8)
