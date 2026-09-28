"""Provider configuration models; no vendor SDK objects or network access."""

from __future__ import annotations

import re
from ipaddress import ip_address
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SUPPORTED_TYPES = (
    "mock", "anthropic", "openai", "azure_openai", "ollama",
    "openai_compatible", "anthropic_compatible",
)

# Compatibility only: canonical profiles must explicitly select their model.
LEGACY_MODELS = {
    "anthropic": "claude-opus-5", "openai": "gpt-4o-mini", "mock": "mock-1",
}
KEY_ENV_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def checked_url(value: str) -> str:
    """Accept HTTP(S) endpoints without embedded credentials or query secrets."""
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"} and parsed.hostname
            and parsed.username is None and parsed.password is None
            and not parsed.query and not parsed.fragment
            and not any(char.isspace() or ord(char) < 32 for char in value)
        )
        _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("base URL must be HTTP(S), without credentials, query, or fragment")
    return value.rstrip("/")


def checked_local_url(value: str) -> str:
    """Literal loopback only; pin localhost instead of trusting DNS/hosts."""
    parsed = urlsplit(checked_url(value))
    if parsed.hostname == "localhost":
        netloc = "127.0.0.1" + (f":{parsed.port}" if parsed.port else "")
        return urlunsplit(parsed._replace(netloc=netloc))
    try:
        local = ip_address(parsed.hostname).is_loopback
    except ValueError:
        local = False
    if not local:
        raise ValueError("local_only requires a literal loopback endpoint")
    return urlunsplit(parsed)


class AuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    mode: Literal["native", "none", "bearer", "api_key_header"] = "native"
    token_env: str | None = None
    header_name: str | None = None


class TLSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    ca_bundle_env: str | None = None
    client_cert_env: str | None = None
    client_key_env: str | None = None


class ProviderProfile(BaseModel):
    """Canonical profile: secrets are environment references, never literal values."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    type: str
    model: str = ""
    model_env: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    api_key_env: str | None = None
    endpoint_env: str | None = None
    deployment_env: str | None = None
    api_version_env: str | None = None
    api: Literal["chat_completions", "responses"] = "chat_completions"
    max_output_tokens: int = Field(default=16000, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0, le=3600)
    max_retries: int = Field(default=2, ge=0, le=5)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = None
    refusal_fallback: bool = False
    supports_images: bool = False
    local_only: bool = False
    num_ctx: int | None = Field(default=None, ge=2048, le=131072)
    think: bool | None = None
    tool_mode: Literal["native", "structured"] = "native"
    auth: AuthConfig = Field(default_factory=AuthConfig)
    headers_env: dict[str, str] = Field(default_factory=dict)
    tls: TLSConfig = Field(default_factory=TLSConfig)

    @field_validator("type")
    @classmethod
    def supported_type(cls, value: str) -> str:
        if value == "internal_custom":
            raise ValueError("internal_custom = pending endpoint contract; not production-usable")
        if value not in SUPPORTED_TYPES:
            raise ValueError("unknown provider type; supported types: " + ", ".join(SUPPORTED_TYPES))
        return value


class ProviderSettings(BaseModel):
    """Resolved active-provider view retained for existing backend consumers.

    ``active`` identifies the canonical profile. ``name`` is its resolved
    protocol type, not a second independently configurable selection.
    The flat constructor remains a deprecated compatibility interface.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    active: str | None = None
    name: str = "anthropic"
    model: str = ""
    api_key: str = Field(default="", exclude=True, repr=False)
    base_url: str | None = None
    temperature: float | None = 0.2
    top_p: float | None = None
    max_output_tokens: int = Field(default=16000, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0, le=3600)
    max_retries: int = Field(default=2, ge=0, le=5)
    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = None
    refusal_fallback: bool = True
    supports_images: bool = False
    local_only: bool = False
    num_ctx: int | None = Field(default=None, ge=2048, le=131072)
    think: bool | None = None
    tool_mode: Literal["native", "structured"] = "native"
    auth_mode: Literal["native", "none", "bearer", "api_key_header"] = "native"
    headers: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)
    api_version: str | None = None
    api: Literal["chat_completions", "responses"] = "chat_completions"
    generation_model: str | None = None
    ca_bundle: str | None = Field(default=None, exclude=True, repr=False)
    client_cert: str | None = Field(default=None, exclude=True, repr=False)
    client_key: str | None = Field(default=None, exclude=True, repr=False)
    # Runtime transport policy; canonical browser profiles retain the default.
    trust_env: bool = Field(default=True, exclude=True)

    @model_validator(mode="after")
    def legacy_defaults(self) -> "ProviderSettings":
        ProviderProfile.supported_type(self.name)
        if self.active is None and not self.model:
            self.model = LEGACY_MODELS.get(self.name, "")
        if (self.num_ctx is not None or self.think is not None or self.tool_mode != "native") and self.name != "ollama":
            raise ValueError("num_ctx, think and structured tool_mode are available only for Ollama profiles")
        if self.local_only:
            if self.name not in {"ollama", "openai_compatible"}:
                raise ValueError("local_only requires Ollama or an OpenAI-compatible local server")
            self.base_url = checked_local_url(
                self.base_url or ("http://127.0.0.1:11434" if self.name == "ollama" else "")
            )
            if self.auth_mode != "none" or self.headers or self.api_key:
                raise ValueError("local_only requires no provider credentials or custom headers")
            if self.model.lower().endswith((":cloud", "-cloud")):
                raise ValueError("local_only cannot select a cloud model tag")
            self.trust_env = False
        return self


def checked_header(name: str, value: str) -> tuple[str, str]:
    forbidden = {"host", "content-length", "transfer-encoding", "connection", "cookie", "set-cookie"}
    if not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name) or name.lower() in forbidden:
        raise ValueError("invalid or reserved provider header name")
    if not value or any(ord(char) < 32 or ord(char) > 126 for char in value):
        raise ValueError("provider header environment value must be nonempty printable ASCII")
    return name, value


def safe_validation_details(errors: list[dict[str, Any]]) -> str:
    """Report schema locations without values, exception contexts, or endpoints."""
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'profile'}: {error['type']}"
        for error in errors
    )
