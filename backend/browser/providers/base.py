"""The model provider interface.

Everything above this layer talks in terms of :class:`ModelProvider`. Adding a
second backend (Azure OpenAI, Anthropic, a local server) means implementing this
one class; no caller changes.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str | list[dict[str, Any]]
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None


class ToolSchema(BaseModel):
    """Provider-neutral native tool definition."""

    name: str
    description: str = ""
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    type: Literal["tool"] = "tool"
    tool: str
    arguments: dict[str, Any]
    snapshot_id: str
    reason: str


class FinalAnswer(BaseModel):
    type: Literal["final"] = "final"
    answer: str


Decision = Annotated[ToolCall | FinalAnswer, Field(discriminator="type")]


class ProviderResponse(BaseModel):
    """The normalized result returned by every model vendor."""

    decision: Decision
    raw_text: str
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str = "stop"
    tool_call_id: str | None = None


@dataclass(frozen=True)
class CompletionChunk:
    """One piece of a streamed answer."""

    text: str


@dataclass(frozen=True)
class CompletionResult:
    """Terminal value of a stream."""

    text: str
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """Any failure reaching or parsing a model backend.

    Carries ``retryable`` so callers can distinguish a rate limit from a bad
    request without matching on strings.
    """

    def __init__(
        self, message: str, *, retryable: bool = False,
        code: str = "provider_error", status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.code = code
        self.status_code = status_code


class ModelProvider(abc.ABC):
    """Provider-neutral generation API, with streaming retained for the UI."""

    #: Stable identifier used in logs and in the hello_ack payload.
    name: str = "unset"
    # Opt-in structured answer generation; existing vendor profiles are unchanged.
    prefers_response_schema: bool = False

    @property
    @abc.abstractmethod
    def model(self) -> str:
        """The concrete model/deployment this instance talks to."""

    @abc.abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> ProviderResponse:
        """Generate one normalized final answer or native tool call."""

    def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[CompletionChunk | CompletionResult]:
        """Stream a completion.

        Providers may override this to expose real vendor streaming. The
        fallback calls :meth:`generate` and emits the final text as one chunk.

        Implementations are async generators, so this is declared as a normal
        method returning an async iterator rather than as ``async def``.
        """
        return self._stream_from_generate(
            messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    async def _stream_from_generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[CompletionChunk | CompletionResult]:
        del temperature, max_output_tokens
        response = await self.generate(list(messages))
        if response.raw_text:
            yield CompletionChunk(text=response.raw_text)
        yield CompletionResult(
            text=response.raw_text,
            finish_reason=response.finish_reason,
            usage=response.usage,
        )

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> CompletionResult:
        """Non-streaming convenience wrapper over :meth:`stream`."""
        result: CompletionResult | None = None
        parts: list[str] = []
        async for item in self.stream(
            messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ):
            if isinstance(item, CompletionResult):
                result = item
            else:
                parts.append(item.text)
        if result is None:
            raise ProviderError("stream ended without a CompletionResult")
        return result if result.text else CompletionResult(
            text="".join(parts), finish_reason=result.finish_reason, usage=result.usage
        )

    async def aclose(self) -> None:
        """Release any held network resources. Safe to call more than once."""


def final_decision(
    text: str, response_schema: type[BaseModel] | None = None
) -> FinalAnswer | ToolCall:
    """Validate structured output when requested, otherwise return plain text."""
    if response_schema is None:
        return FinalAnswer(answer=text)
    try:
        parsed = response_schema.model_validate_json(text)
    except ValidationError:
        raise ProviderError("structured response validation failed", code="invalid_response") from None
    if isinstance(parsed, ToolCall):
        raise ProviderError("browser actions require native tool calling, not a text response", code="invalid_response")
    if isinstance(parsed, FinalAnswer):
        return parsed
    return FinalAnswer(answer=text)


def tool_decision(name: str, raw_arguments: Any) -> ToolCall:
    """Normalize a vendor tool input into the strict agent ToolCall shape."""
    if not isinstance(raw_arguments, dict):
        raise ProviderError("native tool returned non-object arguments", code="invalid_arguments")
    arguments = dict(raw_arguments)
    snapshot_id = arguments.pop("snapshot_id", None)
    reason = arguments.pop("reason", None)
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ProviderError("native tool omitted snapshot_id", code="invalid_arguments")
    if not isinstance(reason, str) or not reason:
        raise ProviderError("native tool omitted reason", code="invalid_arguments")
    return ToolCall(
        tool=name,
        arguments=arguments,
        snapshot_id=snapshot_id,
        reason=reason,
    )
