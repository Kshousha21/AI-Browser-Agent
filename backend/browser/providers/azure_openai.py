"""Azure endpoint/deployment translation reusing the OpenAI message contract."""

from __future__ import annotations

from typing import Any

from ..provider_settings import ProviderSettings
from .openai_provider import OpenAIProvider
from .transport import client_options, no_api_key


class AzureOpenAIProvider(OpenAIProvider):
    name = "azure_openai"

    def __init__(self, settings: ProviderSettings) -> None:
        from openai import AsyncAzureOpenAI, AsyncOpenAI, DefaultAsyncHttpxClient, omit

        options: dict[str, Any] = {
            "api_key": settings.api_key,
            "timeout": settings.timeout_seconds,
            "max_retries": settings.max_retries,
            "http_client": DefaultAsyncHttpxClient(**client_options(settings)),
        }
        if settings.auth_mode != "native":
            options.update(api_key=no_api_key, admin_api_key="", organization="", project="",
                           default_headers={"Authorization": omit})
        if settings.api_version:
            client = AsyncAzureOpenAI(
                azure_endpoint=settings.base_url, azure_deployment=settings.model,
                api_version=settings.api_version, **options,
            )
        else:
            # Current Azure v1 shares the OpenAI API path and needs no dated version.
            endpoint = (settings.base_url or "").rstrip("/")
            if not endpoint.endswith("/openai/v1"):
                endpoint += "/openai/v1"
            client = AsyncOpenAI(base_url=endpoint + "/", **options)
        super().__init__(
            client=client, api_key=settings.api_key, model=settings.model,
            generation_model=settings.generation_model,
            temperature=settings.temperature, top_p=settings.top_p, effort=settings.effort,
            max_output_tokens=settings.max_output_tokens,
            api=settings.api,
            auth_mode=settings.auth_mode,
        )
