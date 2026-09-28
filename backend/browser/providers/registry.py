"""Closed, configuration-driven provider registry. No arbitrary module imports."""

from __future__ import annotations

from collections.abc import Callable

from ..provider_settings import ProviderSettings, SUPPORTED_TYPES
from .anthropic_provider import AnthropicProvider
from .azure_openai import AzureOpenAIProvider
from .base import ModelProvider, ProviderError
from .mock import MockProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .transport import client_options


class OpenAICompatibleProvider(OpenAIProvider):
    name = "openai_compatible"


class AnthropicCompatibleProvider(AnthropicProvider):
    name = "anthropic_compatible"


def _openai(settings: ProviderSettings) -> ModelProvider:
    from openai import DefaultAsyncHttpxClient
    implementation = OpenAICompatibleProvider if settings.name == "openai_compatible" else OpenAIProvider
    return implementation(
        api_key=settings.api_key, model=settings.model, base_url=settings.base_url,
        temperature=settings.temperature, top_p=settings.top_p, effort=settings.effort,
        max_output_tokens=settings.max_output_tokens, timeout_seconds=settings.timeout_seconds,
        max_retries=settings.max_retries, auth_mode=settings.auth_mode, api=settings.api,
        http_client=DefaultAsyncHttpxClient(**client_options(settings)),
    )


def _anthropic(settings: ProviderSettings) -> ModelProvider:
    from anthropic import DefaultAsyncHttpxClient
    implementation = AnthropicCompatibleProvider if settings.name == "anthropic_compatible" else AnthropicProvider
    return implementation(
        api_key=settings.api_key, model=settings.model, base_url=settings.base_url,
        max_output_tokens=settings.max_output_tokens, timeout_seconds=settings.timeout_seconds,
        effort=settings.effort, refusal_fallback=settings.refusal_fallback,
        max_retries=settings.max_retries, auth_mode=settings.auth_mode,
        http_client=DefaultAsyncHttpxClient(**client_options(settings)),
    )


def _ollama(settings: ProviderSettings) -> ModelProvider:
    import httpx
    return OllamaProvider(
        model=settings.model, base_url=settings.base_url or "http://127.0.0.1:11434",
        timeout_seconds=settings.timeout_seconds, max_output_tokens=settings.max_output_tokens,
        max_retries=settings.max_retries, temperature=settings.temperature, top_p=settings.top_p,
        num_ctx=settings.num_ctx, think=settings.think, tool_mode=settings.tool_mode,
        http_client=httpx.AsyncClient(**client_options(settings)),
    )


FACTORIES: dict[str, Callable[[ProviderSettings], ModelProvider]] = {
    "mock": lambda settings: MockProvider(model=settings.model or "mock-1"),
    "anthropic": _anthropic,
    "openai": _openai,
    "azure_openai": AzureOpenAIProvider,
    "ollama": _ollama,
    "openai_compatible": _openai,
    "anthropic_compatible": _anthropic,
}


def build_provider(settings: ProviderSettings) -> ModelProvider:
    """Construct the already validated active profile without any network call."""
    if settings.name == "internal_custom":
        raise ProviderError("internal_custom = pending endpoint contract", code="configuration_error")
    factory = FACTORIES.get(settings.name)
    if factory is None:
        raise ProviderError("unknown provider type; supported: " + ", ".join(SUPPORTED_TYPES), code="configuration_error")
    return factory(settings)
