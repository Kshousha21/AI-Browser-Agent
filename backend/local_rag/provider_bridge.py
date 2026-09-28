"""No-tools use of existing registry clients, with explicit context consent."""
import asyncio
from contextlib import asynccontextmanager
from ipaddress import ip_address
from urllib.parse import urlsplit, urlunsplit
from browser.provider_settings import ProviderSettings, checked_url
from browser.providers import build_provider
from browser.providers.base import FinalAnswer, ProviderError, ToolCall
from .errors import RagError
from .models import RagSettings


class ProviderBridge:
    def __init__(self, settings: ProviderSettings, *, allow_remote_context: bool = False,
                 factory=build_provider):
        self._settings = settings.model_copy(deep=True)
        self._allow_remote = allow_remote_context is True
        self._factory = factory

    @property
    def supports_images(self):
        # Explicit operator capability for the CURRENT configured model. No
        # provider-name/model-name guessing and no fallback model selection.
        return self._settings.supports_images is True and self._settings.name != "mock"

    def settings_for_request(self, limits: RagSettings) -> ProviderSettings:
        settings = self._settings.model_copy(deep=True)
        if settings.name == "mock":
            raise RagError("rag_provider_not_allowed")
        defaults = {"openai": "https://api.openai.com/v1",
                    "anthropic": "https://api.anthropic.com",
                    "ollama": "http://127.0.0.1:11434"}
        try:
            endpoint = checked_url(settings.base_url or defaults.get(settings.name, ""))
            parsed = urlsplit(endpoint)
            host = parsed.hostname
            if host == "localhost":
                # Pin the reserved hostname to loopback; do not trust hosts/DNS.
                netloc = "127.0.0.1" + (f":{parsed.port}" if parsed.port else "")
                endpoint = urlunsplit(parsed._replace(netloc=netloc))
                local = True
            else:
                try:
                    local = ip_address(host).is_loopback
                except ValueError:
                    local = False
        except (ValueError, TypeError):
            raise RagError("rag_configuration_error") from None
        if not local and not self._allow_remote:
            raise RagError("rag_remote_context_denied")
        settings.base_url = endpoint
        if local:
            settings.trust_env = False
        settings.max_retries = 0
        settings.refusal_fallback = False
        settings.timeout_seconds = min(settings.timeout_seconds, limits.timeout_seconds)
        settings.max_output_tokens = min(settings.max_output_tokens, limits.max_output_tokens)
        return settings

    @asynccontextmanager
    async def session(self, limits: RagSettings):
        settings = self.settings_for_request(limits)
        provider = None
        try:
            provider = self._factory(settings)
            yield provider
        except RagError:
            raise
        except Exception:
            raise RagError("rag_provider_failed") from None
        finally:
            if provider is not None:
                try:
                    await asyncio.wait_for(provider.aclose(), timeout=2.0)
                except Exception:
                    pass  # Cleanup must not expose an SDK body or mask failure.


async def generate(provider, messages, limits: RagSettings, *, response_schema=None) -> str | None:
    """None means an explicit native refusal, never guessed from model prose."""
    try:
        response = await asyncio.wait_for(
            provider.generate(messages, tools=[], response_schema=(response_schema
                if getattr(provider, "prefers_response_schema", False) else None)),
            timeout=limits.timeout_seconds)
    except TimeoutError:
        raise RagError("rag_provider_timeout") from None
    except ProviderError as exc:
        if exc.code == "refusal":
            return None
        code = ("rag_unexpected_tool_call" if exc.code == "invalid_tool" else
                "rag_provider_timeout" if exc.code == "timeout" else "rag_provider_failed")
        raise RagError(code) from None
    except Exception:
        raise RagError("rag_provider_failed") from None
    if (isinstance(response.decision, ToolCall) or response.tool_call_id is not None
            or response.finish_reason in {"tool_calls", "tool_use", "function_call"}):
        raise RagError("rag_unexpected_tool_call")
    if response.finish_reason == "refusal":
        return None  # Discard ALL provider prose; no parsing or repair.
    if not isinstance(response.decision, FinalAnswer) or response.finish_reason in {
        "length", "max_tokens", "incomplete", "content_filter"}:
        raise RagError("rag_provider_failed")
    return response.decision.answer
