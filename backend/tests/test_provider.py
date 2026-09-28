"""The ModelProvider contract, exercised against the mock.

Nothing here touches the network or needs an API key. Vendor wire behaviour is
covered by the local OpenAI and Anthropic SSE harnesses.
"""

from __future__ import annotations

import inspect

import pytest

from browser.config import ProviderSettings
from browser.providers import (
    AnthropicProvider,
    CompletionChunk,
    CompletionResult,
    FinalAnswer,
    Message,
    MockProvider,
    ModelProvider,
    OpenAIProvider,
    ProviderError,
    ProviderResponse,
    build_provider,
)

MESSAGES = [
    Message(role="system", content="You are a test."),
    Message(role="user", content="Say something."),
]


async def test_stream_yields_chunks_then_exactly_one_result() -> None:
    provider = MockProvider(scripted=["alpha beta gamma"])

    items = [item async for item in provider.stream(MESSAGES)]

    assert isinstance(items[-1], CompletionResult)
    assert all(isinstance(i, CompletionChunk) for i in items[:-1])
    assert sum(isinstance(i, CompletionResult) for i in items) == 1


async def test_concatenated_chunks_match_the_final_text() -> None:
    provider = MockProvider(scripted=["alpha beta gamma"])

    chunks: list[str] = []
    final: CompletionResult | None = None
    async for item in provider.stream(MESSAGES):
        if isinstance(item, CompletionResult):
            final = item
        else:
            chunks.append(item.text)

    assert final is not None
    assert "".join(chunks).strip() == final.text


async def test_complete_wraps_stream() -> None:
    provider = MockProvider(scripted=["the whole answer"])

    result = await provider.complete(MESSAGES)

    assert isinstance(result, CompletionResult)
    assert result.text == "the whole answer"
    assert result.finish_reason == "stop"


async def test_generate_returns_the_normalized_section_14_shape() -> None:
    provider = MockProvider(scripted=["the whole answer"])

    response = await provider.generate(MESSAGES)

    assert isinstance(response, ProviderResponse)
    assert response.decision == FinalAnswer(answer="the whole answer")
    assert response.raw_text == "the whole answer"
    assert response.usage["total_tokens"] == 0


async def test_messages_reach_the_provider_unchanged() -> None:
    provider = MockProvider()

    await provider.complete(MESSAGES)

    assert provider.calls == [MESSAGES]


async def test_scripted_replies_advance_per_call() -> None:
    provider = MockProvider(scripted=["first", "second"])

    assert (await provider.complete(MESSAGES)).text == "first"
    assert (await provider.complete(MESSAGES)).text == "second"
    # The last reply repeats rather than running out.
    assert (await provider.complete(MESSAGES)).text == "second"


async def test_failures_surface_as_provider_error() -> None:
    provider = MockProvider(fail_with="upstream exploded")

    with pytest.raises(ProviderError, match="upstream exploded"):
        await provider.complete(MESSAGES)


async def test_aclose_is_idempotent() -> None:
    provider = MockProvider()

    await provider.aclose()
    await provider.aclose()

    assert provider.closed is True


def test_provider_error_carries_a_retry_signal() -> None:
    assert ProviderError("rate limited", retryable=True).retryable is True
    assert ProviderError("bad request").retryable is False


# --------------------------------------------------------------------------
# Interface conformance
# --------------------------------------------------------------------------


def test_the_interface_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        ModelProvider()  # type: ignore[abstract]


def test_a_subclass_missing_generate_cannot_be_instantiated() -> None:
    class Incomplete(ModelProvider):
        name = "incomplete"

        @property
        def model(self) -> str:
            return "x"

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "implementation", [MockProvider, OpenAIProvider, AnthropicProvider]
)
def test_implementations_satisfy_the_interface(implementation: type) -> None:
    """All registered providers expose the section 14 generate signature."""
    assert issubclass(implementation, ModelProvider)

    expected = inspect.signature(ModelProvider.generate)
    actual = inspect.signature(implementation.generate)

    assert list(actual.parameters) == list(expected.parameters)


def test_openai_provider_refuses_to_start_without_a_key() -> None:
    with pytest.raises(ProviderError, match="API key"):
        OpenAIProvider(api_key="", model="gpt-4o-mini")


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def test_factory_builds_the_mock() -> None:
    provider = build_provider(ProviderSettings(name="mock", model="mock-1"))

    assert isinstance(provider, MockProvider)
    assert provider.name == "mock"
    assert provider.model == "mock-1"


def test_factory_builds_openai_without_calling_out() -> None:
    """Construction must not perform I/O; only a later stream() would."""
    provider = build_provider(
        ProviderSettings(name="openai", model="gpt-4o-mini", api_key="sk-not-real")
    )

    assert isinstance(provider, OpenAIProvider)
    assert provider.name == "openai"
    assert provider.model == "gpt-4o-mini"


def test_factory_builds_anthropic_without_calling_out() -> None:
    provider = build_provider(
        ProviderSettings(
            name="anthropic", model="claude-opus-5", api_key="sk-ant-not-real"
        )
    )

    assert isinstance(provider, AnthropicProvider)
    assert provider.name == "anthropic"
    assert provider.model == "claude-opus-5"
