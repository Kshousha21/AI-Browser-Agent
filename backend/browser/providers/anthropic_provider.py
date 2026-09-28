"""Anthropic implementation of :class:`ModelProvider`.

Uses the official ``anthropic`` SDK's streaming helper.

Two shape differences from the OpenAI provider are worth knowing about, because
they are properties of the API rather than choices made here:

* The system prompt is a top-level ``system`` argument, not a message with
  ``role="system"``. Our :class:`Message` list is split accordingly.
* ``temperature`` is **rejected** by current Claude models (Opus 5, Sonnet 5,
  Fable 5, and the 4.6+ family return a 400). The interface still accepts the
  argument for the OpenAI provider's sake; this one ignores it and uses
  ``output_config.effort`` instead, which is the current depth/cost control.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel

from .anthropic_continuation import Continuations
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
from .transport import protect_sdk_logs, provider_error
from .validation import usage_counts, validated_tool
from .images import translate

log = logging.getLogger(__name__)

# Anthropic's own beta flag for server-side refusal fallbacks. Paired with the
# scalar `fallbacks="default"` form; the array form uses a different flag and
# mixing the two is a 400.
REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class AnthropicProvider(ModelProvider):
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        max_output_tokens: int = 16000,
        timeout_seconds: float = 60.0,
        effort: str | None = None,
        refusal_fallback: bool = True,
        max_retries: int = 2,
        auth_mode: str = "native",
        http_client: Any = None,
    ) -> None:
        if not api_key and auth_mode == "native":
            raise ProviderError("Anthropic provider requires an API key")
        # Imported here so the package is only needed when this provider is
        # used; the test suite runs against the mock without it installed.
        try:
            from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient, omit, transform_schema
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ProviderError(
                "the 'anthropic' package is not installed; pip install anthropic"
            ) from exc

        self._model = model
        self._max_output_tokens = max_output_tokens
        self._effort = effort
        self._validate_generation(max_output_tokens)
        self._continuations = Continuations()
        self._refusal_fallback = refusal_fallback
        self._transform_schema = transform_schema
        protect_sdk_logs()
        auth_options: dict[str, Any] = {}
        if auth_mode != "native":
            auth_options = {"auth_token": "", "default_headers": {"X-Api-Key": omit, "Authorization": omit}}
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout_seconds,
            max_retries=max_retries,
            http_client=http_client or DefaultAsyncHttpxClient(follow_redirects=False),
            **auth_options,
        )

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> ProviderResponse:
        """Generate a normalized final answer or native Anthropic tool call."""
        if tools and response_schema:
            raise ProviderError("tools and response_schema cannot be used together")
        request = self._build_request(messages, tools=tools)
        if response_schema is not None:
            request["output_format"] = response_schema

        parts: list[str] = []
        try:
            async with self._open_stream(request) as stream:
                async for text in _completed_text(stream):
                    if text:
                        parts.append(text)
                final = await stream.get_final_message()
        except Exception as exc:  # noqa: BLE001 - normalised below
            if isinstance(exc, ProviderError):
                raise
            raise _as_provider_error(exc) from None

        text = "".join(parts)
        stop_reason = _stop_reason(final)
        if stop_reason == "refusal" and not tools and response_schema is None:
            # No-tools consumers need a typed, body-free refusal signal. Do not
            # reinterpret a contradictory tool block as a successful abstention.
            if any(getattr(block, "type", None) == "tool_use" for block in final.content):
                raise ProviderError("unadvertised tool in refusal", code="invalid_tool")
            raise ProviderError(_refusal_message(final), code="refusal")
        if stop_reason == "refusal" and not text.strip():
            raise ProviderError(_refusal_message(final))

        tool_call_id: str | None = None
        calls = [block for block in final.content if getattr(block, "type", None) == "tool_use"]
        if len(calls) > 1:
            raise ProviderError("provider returned multiple tools for a single-action turn", code="invalid_response")
        if calls:
            block = calls[0]
            decision = validated_tool(block.name, block.input, tools)
            tool_call_id = block.id
            if not tool_call_id or stop_reason != "tool_use":
                raise ProviderError("Anthropic native tool response was incomplete", code="invalid_response")
            # Preserve the COMPLETE assistant turn, including signed thinking.
            # The neutral agent sees only an opaque handle, never those blocks.
            tool_call_id = self._continuations.remember(final.content, tool_call_id)
        else:
            if not text.strip():
                raise ProviderError("Anthropic returned neither text nor a native tool call", code="invalid_response")
            decision = final_decision(text, response_schema)

        return ProviderResponse(
            decision=decision,
            raw_text=text,
            finish_reason=stop_reason,
            usage=_usage_dict(final.usage),
            tool_call_id=tool_call_id,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[CompletionChunk | CompletionResult]:
        # temperature is deliberately dropped: current Claude models reject it.
        del temperature

        request = self._build_request(
            messages, max_output_tokens=max_output_tokens
        )

        parts: list[str] = []
        try:
            async with self._open_stream(request) as stream:
                async for text in _completed_text(stream):
                    if text:
                        parts.append(text)
                        yield CompletionChunk(text=text)
                final = await stream.get_final_message()
        except Exception as exc:  # noqa: BLE001 - normalised below
            raise _as_provider_error(exc) from None

        text = "".join(parts)
        stop_reason = _stop_reason(final)

        # A refusal with nothing streamed would otherwise surface as a blank
        # answer bubble, which reads as a bug rather than a decision.
        if stop_reason == "refusal" and not text.strip():
            raise ProviderError(_refusal_message(final))

        yield CompletionResult(
            text=text,
            finish_reason=stop_reason,
            usage=_usage_dict(final.usage),
        )

    def _build_request(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int | None = None,
        tools: list[ToolSchema] | None = None,
    ) -> dict[str, Any]:
        limit = self._max_output_tokens if max_output_tokens is None else max_output_tokens
        self._validate_generation(limit)
        system, turns = _split_system(messages)
        self._continuations.restore(list(messages), turns)
        if not turns:
            raise ProviderError("Anthropic requires at least one non-system message")
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": limit,
            "messages": turns,
        }
        if system:
            request["system"] = system
        if self._effort:
            request["output_config"] = {"effort": self._effort}
        if tools:
            request["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
            request["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": self._transform_schema(tool.input_schema),
                    # Approved Opus 5 grammar mitigation ONLY. Compatible
                    # providers and canonical schemas remain unchanged. Every
                    # response still passes validated_tool before dispatch.
                    "strict": True,
                }
                for tool in tools
            ]
        return request

    def _validate_generation(self, limit: int) -> None:
        if type(limit) is not int or limit <= 0:
            raise ProviderError("Anthropic max_output_tokens must be a positive integer", code="configuration_error")
        if self.name == "anthropic" and self._model == "claude-opus-5":
            if self._effort not in {None, "low", "medium", "high", "xhigh", "max"}:
                raise ProviderError("Anthropic Opus 5 effort must be low, medium, high, xhigh or max", code="configuration_error")

    def _open_stream(self, request: dict[str, Any]) -> Any:
        """Open the streaming context, with refusal fallback when enabled.

        On a policy decline the API re-runs the request on a fallback model
        inside the same call, rather than returning nothing.
        """
        if self._refusal_fallback:
            return self._client.beta.messages.stream(
                betas=[REFUSAL_FALLBACK_BETA],
                fallbacks="default",
                **request,
            )
        return self._client.messages.stream(**request)

    async def aclose(self) -> None:
        self._continuations.clear()
        await self._client.close()


def _split_system(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Hoist system messages out of the turn list.

    Anthropic takes the system prompt as its own argument; sending one as a
    message role is a different feature with its own placement rules.
    """
    system_parts = [
        m.content for m in messages if m.role == "system" and isinstance(m.content, str)
    ]
    turns: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            if not isinstance(message.content, str):
                raise ProviderError("Anthropic system messages must contain text")
            continue
        if message.role == "tool":
            if not message.tool_call_id:
                raise ProviderError("Anthropic tool results require tool_call_id")
            turns.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content,
                        }
                    ],
                }
            )
            continue
        if message.role == "assistant" and message.tool_name:
            if not message.tool_call_id or message.tool_arguments is None:
                raise ProviderError("Anthropic assistant tool calls require id and arguments")
            turns.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": message.tool_call_id,
                            "name": message.tool_name,
                            "input": message.tool_arguments,
                        }
                    ],
                }
            )
            continue
        turns.append({"role": message.role, "content": translate(message.content, "anthropic", message.role)})
    return "\n\n".join(system_parts), turns


def _usage_dict(usage: Any) -> dict[str, int]:
    """Normalise to the same key names the OpenAI provider reports."""
    if usage is None:
        return {}
    prompt = getattr(usage, "input_tokens", None)
    completion = getattr(usage, "output_tokens", None)
    if type(prompt) is int:
        for key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            value = getattr(usage, key, None)
            if type(value) is int and value >= 0:
                prompt += value
    return usage_counts(prompt, completion)


def _refusal_message(final: Any) -> str:
    return "the model declined this request; no tool was executed"


def _as_provider_error(exc: Exception) -> ProviderError:
    # Recognize only a fixed diagnostic category; never expose upstream text.
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if (getattr(exc, "status_code", None) == 400 and isinstance(error, dict)
            and error.get("type") == "invalid_request_error"
            and isinstance(message, str) and len(message) <= 4096
            and "compiled grammar" in message.lower() and "too large" in message.lower()):
        return ProviderError("Anthropic combined strict-tool grammar exceeds the API complexity limit (HTTP 400)",
                             code="schema_too_complex", status_code=400)
    return provider_error(exc, "Anthropic")


async def _completed_text(stream: Any) -> AsyncIterator[str]:
    completed = False
    async for event in stream:
        if event.type == "content_block_delta" and event.delta.type == "text_delta":
            yield event.delta.text
        elif event.type == "message_stop":
            completed = True
    if not completed:
        raise ProviderError("Anthropic stream ended without message_stop", code="malformed_response")


def _stop_reason(final: Any) -> str:
    if not final.stop_reason:
        raise ProviderError("Anthropic response omitted its stop reason", code="malformed_response")
    return final.stop_reason
