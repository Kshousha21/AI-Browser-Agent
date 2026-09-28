"""In-process provider used by the test suite and by `provider.name: mock`.

Requires no API key, so the whole suite runs offline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from pydantic import BaseModel

from .base import (
    CompletionChunk,
    CompletionResult,
    FinalAnswer,
    Message,
    ModelProvider,
    ProviderError,
    ProviderResponse,
    ToolSchema,
    ToolCall,
)


class MockProvider(ModelProvider):
    name = "mock"

    def __init__(
        self,
        *,
        model: str = "mock-1",
        scripted: list[str | ToolCall | ProviderResponse] | None = None,
        fail_with: str | None = None,
    ) -> None:
        self._model = model
        # Replies are consumed in order; the last one repeats once exhausted.
        self._scripted = scripted or ["This is a mock answer."]
        self._fail_with = fail_with
        self.calls: list[list[Message]] = []
        self.closed = False

    @property
    def model(self) -> str:
        return self._model

    def _next_reply(
        self, messages: Sequence[Message]
    ) -> str | ToolCall | ProviderResponse:
        self.calls.append(list(messages))
        if self._fail_with:
            raise ProviderError(self._fail_with)
        index = min(len(self.calls) - 1, len(self._scripted) - 1)
        return self._scripted[index]

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> ProviderResponse:
        del tools, response_schema
        reply = self._next_reply(messages)
        if isinstance(reply, ProviderResponse):
            return reply
        if isinstance(reply, ToolCall):
            return ProviderResponse(
                decision=reply,
                raw_text="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                finish_reason="tool_use",
                tool_call_id=f"mock-call-{len(self.calls)}",
            )
        return ProviderResponse(
            decision=FinalAnswer(answer=reply),
            raw_text=reply,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[CompletionChunk | CompletionResult]:
        reply = self._next_reply(messages)
        if not isinstance(reply, str):
            raise ProviderError("MockProvider.stream only supports scripted text")

        # Split on whitespace so tests can assert that reassembly is lossless.
        for word in reply.split(" "):
            yield CompletionChunk(text=word + " ")
        yield CompletionResult(
            text=reply,
            finish_reason="stop",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    async def aclose(self) -> None:
        self.closed = True
