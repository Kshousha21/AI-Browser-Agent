"""OpenAI Responses wire translation, reused by OpenAI/compatible/Azure clients."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel

from .base import (
    CompletionChunk, CompletionResult, Message, ProviderError, ProviderResponse,
    ToolSchema, final_decision,
)
from .openai_schema import _restore_optional_defaults, _strict_function_parameters
from .transport import provider_error
from .validation import usage_counts, validated_tool
from .images import translate


class ResponsesProtocol:
    """Stateless server requests; bounded continuation items stay backend-only."""

    def __init__(self, client: Any, model: str, family: str, headers: dict[str, Any]) -> None:
        self.client = client
        self.model = model
        self.family = family
        self.headers = headers
        self._continuations: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def request(self, messages: Sequence[Message], generation: dict[str, Any]) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "assistant" and message.tool_name:
                if not message.tool_call_id or message.tool_arguments is None:
                    raise ProviderError("Responses native tools require a call ID and arguments")
                items.extend(self._continuations.get(message.tool_call_id, []))
                items.append({"type": "function_call", "call_id": message.tool_call_id,
                              "name": message.tool_name, "arguments": json.dumps(message.tool_arguments)})
            elif message.role == "tool":
                if not message.tool_call_id:
                    raise ProviderError("Responses tool results require tool_call_id")
                content = message.content if isinstance(message.content, str) else json.dumps(message.content)
                items.append({"type": "function_call_output", "call_id": message.tool_call_id, "output": content})
            else:
                content: Any = message.content
                if isinstance(content, list):
                    content = ("\n".join(item["text"] for item in content)
                        if all(isinstance(item, dict) and item.get("type") == "text"
                               and isinstance(item.get("text"), str) for item in content)
                        else translate(content, "responses", message.role))
                items.append({"role": message.role, "content": content})
        options = dict(generation)
        options["max_output_tokens"] = options.pop("max_completion_tokens")
        effort = options.pop("reasoning_effort", None)
        if effort is not None:
            options["reasoning"] = {"effort": effort}
        return {
            "model": self.model, "input": items, "store": False,
            "extra_headers": self.headers,
            "include": ["reasoning.encrypted_content"], **options,
        }

    async def generate(
        self, messages: list[Message], tools: list[ToolSchema] | None,
        response_schema: type[BaseModel] | None, generation: dict[str, Any],
    ) -> ProviderResponse:
        if tools and response_schema:
            raise ProviderError("tools and response_schema cannot be used together")
        request = self.request(messages, generation)
        if tools:
            request["parallel_tool_calls"] = False
            request["tools"] = [{
                "type": "function", "name": tool.name, "description": tool.description,
                "parameters": _strict_function_parameters(tool.input_schema), "strict": True,
            } for tool in tools]
        if response_schema:
            request["text"] = {"format": {
                "type": "json_schema", "name": response_schema.__name__, "strict": True,
                "schema": _strict_function_parameters(response_schema.model_json_schema()),
            }}
        try:
            response = await self.client.responses.create(**request)
            return self.normalize(response, tools, response_schema)
        except Exception as exc:
            raise provider_error(exc, self.family) from None

    def normalize(self, response: Any, tools: list[ToolSchema] | None, schema: type[BaseModel] | None) -> ProviderResponse:
        if response.status not in {"completed", "incomplete"} or response.error:
            raise ProviderError("Responses request did not complete successfully", code="invalid_response")
        text_parts: list[str] = []
        calls: list[Any] = []
        reasoning: list[dict[str, Any]] = []
        for item in response.output:
            if item.type == "function_call":
                calls.append(item)
            elif item.type == "reasoning":
                reasoning.append(item.model_dump(mode="json", exclude_none=True))
            elif item.type == "message":
                for block in item.content:
                    if block.type == "refusal":
                        raise ProviderError("OpenAI declined the request", code="refusal")
                    if block.type == "output_text":
                        text_parts.append(block.text)
        if len(calls) > 1:
            raise ProviderError("provider returned multiple tools for a single-action turn", code="invalid_response")
        text = "".join(text_parts)
        call_id = None
        if calls:
            call = calls[0]
            if response.status != "completed" or not call.call_id:
                raise ProviderError("Responses native tool response was incomplete", code="invalid_response")
            try:
                arguments = json.loads(call.arguments)
            except (TypeError, json.JSONDecodeError):
                raise ProviderError("Responses native tool returned invalid JSON", code="invalid_arguments") from None
            tool_schema = next((tool.input_schema for tool in tools or [] if tool.name == call.name), {})
            decision = validated_tool(call.name, _restore_optional_defaults(arguments, tool_schema), tools)
            call_id = call.call_id
            self._continuations[call_id] = reasoning
            while len(self._continuations) > 256:
                self._continuations.popitem(last=False)
        else:
            if not text.strip():
                raise ProviderError("Responses returned neither text nor a native tool call", code="invalid_response")
            decision = final_decision(text, schema)
        return ProviderResponse(
            decision=decision, raw_text=text, tool_call_id=call_id,
            finish_reason="tool_use" if calls else ("stop" if response.status == "completed" else "length"),
            usage=_usage(response.usage),
        )

    async def stream(self, messages: Sequence[Message], generation: dict[str, Any]) -> AsyncIterator[CompletionChunk | CompletionResult]:
        request = self.request(messages, generation)
        terminal = None
        parts: list[str] = []
        try:
            stream = await self.client.responses.create(**request, stream=True)
            async with stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        parts.append(event.delta)
                        yield CompletionChunk(text=event.delta)
                    elif event.type in {"response.completed", "response.incomplete"}:
                        terminal = event.response
                    elif event.type in {"response.failed", "error", "response.refusal.delta"}:
                        raise ProviderError("Responses answer stream failed or was refused", code="invalid_response")
        except Exception as exc:
            raise provider_error(exc, self.family) from None
        if terminal is None:
            raise ProviderError("Responses stream ended without a terminal response", code="malformed_response")
        yield CompletionResult(text="".join(parts),
                               finish_reason="stop" if terminal.status == "completed" else "length",
                               usage=_usage(terminal.usage))

    def clear(self) -> None:
        self._continuations.clear()


def _usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    return usage_counts(getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None),
                        getattr(usage, "total_tokens", None))
