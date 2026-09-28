"""OpenAI implementation of :class:`ModelProvider`.

Supports Chat Completions and Responses. Compatible gateways reuse the
same translation paths with backend-only transport configuration.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel

from .base import (
    CompletionChunk,
    CompletionResult,
    Message,
    ModelProvider,
    ProviderError,
    ProviderResponse,
    ToolSchema,
    final_decision,
)
from .transport import no_api_key, protect_sdk_logs, provider_error
from .validation import usage_counts, validated_tool
from .openai_schema import _restore_optional_defaults, _strict_function_parameters
from .openai_responses import ResponsesProtocol
from .images import translate

class OpenAIProvider(ModelProvider):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        temperature: float | None = 0.2,
        max_output_tokens: int = 1024,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        top_p: float | None = None,
        effort: str | None = None,
        generation_model: str | None = None,
        auth_mode: str = "native",
        http_client: Any = None,
        client: Any = None,
        api: str = "chat_completions",
    ) -> None:
        if not api_key and auth_mode == "native" and client is None:
            raise ProviderError("OpenAI provider requires an API key")
        # Imported here so the package is only needed when this provider is used;
        # the test suite runs against the mock without it installed.
        try:
            from openai import AsyncOpenAI, DefaultAsyncHttpxClient, omit
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ProviderError(
                "the 'openai' package is not installed; pip install openai"
            ) from exc

        self._model = model
        self._api = api
        self._generation_model = generation_model or model
        self._temperature = temperature
        self._top_p = top_p
        self._effort = effort
        self._max_output_tokens = max_output_tokens
        self._request_headers = {} if auth_mode == "native" else {"Authorization": omit}
        protect_sdk_logs()
        auth_options: dict[str, Any] = {}
        if auth_mode != "native":
            auth_options = {"default_headers": {"Authorization": omit},
                            "admin_api_key": "", "organization": "", "project": ""}
        self._client = client or AsyncOpenAI(
            api_key=api_key if auth_mode == "native" else no_api_key,
            base_url=base_url or None,
            timeout=timeout_seconds,
            max_retries=max_retries,
            http_client=http_client or DefaultAsyncHttpxClient(follow_redirects=False),
            **auth_options,
        )
        self._responses = ResponsesProtocol(self._client, model, self.name, self._request_headers)

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> ProviderResponse:
        """Generate a normalized final answer or native OpenAI tool call."""
        if self._api == "responses":
            return await self._responses.generate(messages, tools, response_schema, self._generation_parameters())
        if tools and self._generation_model.lower().startswith("gpt-6"):
            raise ProviderError("this model requires api=responses for native tool calling", code="unsupported_request")
        if tools and response_schema:
            raise ProviderError("tools and response_schema cannot be used together")
        request: dict[str, Any] = {
            "model": self._model,
            "messages": _message_payload(messages),
            "extra_headers": self._request_headers,
            **self._generation_parameters(),
        }
        if tools:
            request["parallel_tool_calls"] = False
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": _strict_function_parameters(
                            tool.input_schema
                        ),
                        "strict": True,
                    },
                }
                for tool in tools
            ]
        if response_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "strict": True,
                    "schema": _strict_function_parameters(
                        response_schema.model_json_schema()
                    ),
                },
            }

        try:
            completion = await self._client.chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise _as_provider_error(exc) from None

        try:
            choice = completion.choices[0]
            raw_text = choice.message.content or ""
            calls = choice.message.tool_calls or []
            if not isinstance(raw_text, str):
                raise ValueError("non-text response")
        except (AttributeError, IndexError, TypeError, ValueError):
            raise ProviderError("OpenAI returned a malformed completion", code="malformed_response") from None
        if getattr(choice.message, "refusal", None):
            raise ProviderError("OpenAI declined the request", code="refusal")
        if choice.finish_reason not in {"stop", "length", "tool_calls"}:
            raise ProviderError("OpenAI completion omitted a supported finish reason", code="invalid_response")
        if len(calls) > 1:
            raise ProviderError("provider returned multiple tools for a single-action turn", code="invalid_response")
        if calls:
            call = calls[0]
            try:
                arguments = json.loads(call.function.arguments)
            except (AttributeError, TypeError, json.JSONDecodeError):
                raise ProviderError("OpenAI native tool returned invalid JSON arguments", code="invalid_arguments") from None
            arguments = _restore_optional_defaults(
                arguments,
                next(
                    (
                        tool.input_schema
                        for tool in tools or []
                        if tool.name == call.function.name
                    ),
                    {},
                ),
            )
            decision = validated_tool(call.function.name, arguments, tools)
            if choice.finish_reason not in {"tool_calls", "stop"}:
                raise ProviderError("OpenAI native tool response was incomplete", code="invalid_response")
            tool_call_id = call.id
            if not tool_call_id:
                raise ProviderError("OpenAI native tool omitted its call ID", code="invalid_response")
        else:
            if not raw_text.strip():
                raise ProviderError("OpenAI returned neither text nor a native tool call", code="invalid_response")
            decision = final_decision(raw_text, response_schema)
            tool_call_id = None
        return ProviderResponse(
            decision=decision,
            raw_text=raw_text,
            finish_reason=choice.finish_reason or "stop",
            usage=_usage_dict(completion.usage),
            tool_call_id=tool_call_id,
        )
    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[CompletionChunk | CompletionResult]:
        if self._api == "responses":
            async for item in self._responses.stream(messages, self._generation_parameters(temperature, max_output_tokens)):
                yield item
            return
        payload = _message_payload(messages)
        parts: list[str] = []
        finish_reason = "stop"
        usage: dict[str, int] = {}
        finished = False

        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=payload,  # type: ignore[arg-type]
                **self._generation_parameters(temperature, max_output_tokens),
                stream=True,
                stream_options={"include_usage": True},
                extra_headers=self._request_headers,
            )
            async with stream:
                async for chunk in stream:
                    # The final usage frame carries no choices, so this is not
                    # defensive padding: it is the documented shape.
                    if getattr(chunk, "usage", None):
                        usage = _usage_dict(chunk.usage)
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                        finished = True
                    text = getattr(choice.delta, "content", None)
                    if text:
                        parts.append(text)
                        yield CompletionChunk(text=text)
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise _as_provider_error(exc) from None

        if not finished:
            raise ProviderError("OpenAI stream ended without a finish reason", code="malformed_response")
        yield CompletionResult(
            text="".join(parts), finish_reason=finish_reason, usage=usage
        )

    async def aclose(self) -> None:
        self._responses.clear()
        await self._client.close()

    def _generation_parameters(
        self, temperature: float | None = None, maximum: int | None = None,
    ) -> dict[str, Any]:
        """Keep sampling/reasoning differences inside the OpenAI protocol path."""
        params: dict[str, Any] = {
            "max_completion_tokens": self._max_output_tokens if maximum is None else maximum,
        }
        family = self._generation_model.lower()
        reasoning = family.startswith(("o1", "o3", "o4", "gpt-5", "gpt-6"))
        if family.startswith("gpt-6") and self._effort in {"none", "minimal"}:
            raise ProviderError("this model requires reasoning effort low or higher", code="unsupported_request")
        if reasoning and self._effort is not None:
            params["reasoning_effort"] = self._effort
        sampling = not reasoning or (
            family.startswith(("gpt-5.1", "gpt-5.2")) and self._effort == "none"
        )
        chosen = self._temperature if temperature is None else temperature
        if sampling and chosen is not None:
            params["temperature"] = chosen
        if sampling and self._top_p is not None:
            params["top_p"] = self._top_p
        return params


def _message_payload(messages: Sequence[Message]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {"role": message.role, "content": translate(message.content, "chat", message.role)}
        if message.role == "assistant" and message.tool_name:
            if not message.tool_call_id or message.tool_arguments is None:
                raise ProviderError("OpenAI assistant tool calls require id and arguments")
            item["tool_calls"] = [
                {
                    "id": message.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": message.tool_name,
                        "arguments": json.dumps(message.tool_arguments),
                    },
                }
            ]
        if message.role == "tool":
            if not message.tool_call_id:
                raise ProviderError("OpenAI tool results require tool_call_id")
            item["tool_call_id"] = message.tool_call_id
        payload.append(item)
    return payload


def _usage_dict(usage: Any) -> dict[str, int]:
    out = usage_counts(getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None))
    total = getattr(usage, "total_tokens", None)
    if type(total) is int and total >= 0:
        out["total_tokens"] = total
    return out


def _as_provider_error(exc: Exception) -> ProviderError:
    return provider_error(exc, "OpenAI")
