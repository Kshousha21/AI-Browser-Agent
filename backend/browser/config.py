"""Configuration: config.yaml layered under .env / process environment.

Precedence, lowest to highest: config.yaml -> .env -> real environment.
Environment keys are prefixed ``BROWSER_`` and use ``__`` to descend into a
nested section, e.g. ``BROWSER_PROVIDER__API_KEY``.

Validation happens once at startup. A missing or malformed field raises
:class:`ConfigError`, whose message lists every problem at once rather than
stopping at the first one found.
"""

from __future__ import annotations

import contextvars
import os
from collections import ChainMap
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from .provider_settings import KEY_ENV_VARS, ProviderSettings
from .provider_resolution import resolve_provider
from .routing_settings import RoutingSettings
from local_visual_retrieval.schemas import VisualSettings, MultimodalSettings

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Set by load_settings() immediately before Settings() is constructed. A
# ContextVar rather than a plain global so concurrent loads cannot cross wires.
_active_config_path: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "active_config_path", default=None
)
_provider_environment: contextvars.ContextVar[Mapping[str, Any] | None] = contextvars.ContextVar(
    "provider_environment", default=None
)


class ConfigError(RuntimeError):
    """Raised at startup when configuration is missing or invalid."""


class ServerSettings(BaseModel):
    # Loopback only. Binding to anything routable would expose a token-guarded
    # LLM proxy to the local network, so the value is validated, not trusted.
    host: str = "127.0.0.1"
    port: int = 8765
    allowed_origins: list[str] = Field(default_factory=list)
    allow_any_extension_origin: bool = True

    @field_validator("host")
    @classmethod
    def _must_be_loopback(cls, v: str) -> str:
        if v not in ("127.0.0.1", "::1", "localhost"):
            raise ValueError(
                "host must be a loopback address (127.0.0.1, ::1, localhost), "
                f"got {v!r}"
            )
        return v

    @field_validator("port")
    @classmethod
    def _valid_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be 1-65535, got {v}")
        return v


class AuthSettings(BaseModel):
    # Shared secret; the side panel sends it in the first WebSocket frame.
    token: str = Field(default="", exclude=True, repr=False)

    @field_validator("token")
    @classmethod
    def _non_trivial(cls, v: str) -> str:
        if v and len(v) < 16:
            raise ValueError("token must be at least 16 characters")
        return v


class PageSettings(BaseModel):
    # Mirrors the extension option of the same name; the backend clamps whatever
    # the client sends so a misconfigured client cannot blow up the context.
    max_visible_text_chars: int = 20000
    max_elements: int = 300


class AgentSettings(BaseModel):
    max_steps: int = Field(default=30, ge=1, le=100)
    step_timeout_seconds: float = Field(default=30.0, ge=1.0, le=3600.0)
    run_timeout_seconds: float = Field(default=180.0, ge=5.0, le=14400.0)
    token_budget: int | None = Field(default=None, ge=1_000)
    history_steps: int = Field(default=12, ge=2, le=50)


class ResearchSettings(BaseModel):
    model_config = {"extra": "forbid"}
    max_pdfs: int = Field(default=3, ge=1, le=5, strict=True)
    max_visual_pages: int = Field(default=3, ge=1, le=5, strict=True)


class SiteAdapterEntrySettings(BaseModel):
    enabled: bool = False
    domains: list[str] = Field(default_factory=list)

    @field_validator("domains")
    @classmethod
    def _exact_hostnames_only(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            host = value.strip().casefold().rstrip(".")
            if not host or any(token in host for token in ("*", "/", ":")):
                raise ValueError("site adapter domains must be exact hostnames")
            if host not in cleaned:
                cleaned.append(host)
        return cleaned


class SiteAdaptersSettings(BaseModel):
    """Generic DOM adapter; no domain configuration required."""
    model_config = {"extra": "forbid"}


class SecuritySettings(BaseModel):
    confirmation_policy: Literal["consequential", "delete_only"] = "delete_only"
    read_only_mode: bool = Field(default=False, exclude=True)  # Legacy configuration only.
    require_confirmation_for: list[str] = Field(
        default_factory=lambda: [
            "submit",
            "delete",
            "upload",
            "download",
            "send",
            "execute_javascript",
            "navigate_external",
            "untrusted_argument",
            "cross_origin_transfer",
        ]
    )
    audit_log_path: Path = Path("audit/actions.jsonl")


class LoggingSettings(BaseModel):
    level: str = "INFO"
    format: Literal["json", "text"] = "json"


class _YamlSource(PydanticBaseSettingsSource):
    """Lowest-priority source: the contents of config.yaml."""

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        raise NotImplementedError  # __call__ supplies everything at once

    def __call__(self) -> dict[str, Any]:
        path = _active_config_path.get()
        if path is None:
            return {}
        return _read_yaml(path)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BROWSER_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        hide_input_in_errors=True,
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    providers: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)
    page: PageSettings = Field(default_factory=PageSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    routing_provider: ProviderSettings | None = Field(default=None, exclude=True, repr=False)
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    visual_retrieval: VisualSettings = Field(default_factory=VisualSettings)
    multimodal_retrieval: MultimodalSettings = Field(default_factory=MultimodalSettings)
    site_adapters: SiteAdaptersSettings = Field(default_factory=SiteAdaptersSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @model_validator(mode="before")
    @classmethod
    def _resolve_active_provider(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        try:
            data["provider"] = resolve_provider(
                data.get("provider"), data.get("providers", {}),
                _provider_environment.get() or os.environ,
            )
        except ValueError as exc:
            auth = data.get("auth", {})
            if isinstance(auth, dict) and not auth.get("token"):
                raise ValueError(str(exc) + "; auth.token is required (BROWSER_AUTH__TOKEN)") from None
            raise
        routing = RoutingSettings.model_validate(data.get("routing", {}))
        # Resolve the second profile only when used; API selection stays independent.
        data["routing_provider"] = None
        if routing.enabled:
            router = resolve_provider({"active": routing.fallback_provider}, data.get("providers", {}),
                                      _provider_environment.get() or os.environ)
            if router.name != "ollama" or not router.local_only or router.auth_mode != "none":
                raise ValueError("routing fallback requires an unauthenticated local_only Ollama profile")
            data["routing_provider"] = router
        return data

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Highest priority first.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls),
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return raw


def _required_field_problems(s: Settings) -> list[str]:
    """Cross-field requirements, which cannot live in a single-field validator."""
    problems: list[str] = []
    if not s.auth.token:
        problems.append(
            "auth.token is required (set it in config.yaml or BROWSER_AUTH__TOKEN)"
        )
    if s.provider.auth_mode == "native" and s.provider.name != "mock" and not s.provider.api_key:
        env_name = KEY_ENV_VARS.get(s.provider.name, "BROWSER_PROVIDER__API_KEY")
        problems.append(
            f"provider.api_key is required for provider {s.provider.name!r} "
            f"(set {env_name} or BROWSER_PROVIDER__API_KEY)"
        )
    return problems


def load_settings(config_path: Path | None = None) -> Settings:
    """Load, layer and validate settings.

    Raises :class:`ConfigError` listing every problem found.
    """
    settings, path = _load_layered_settings(config_path)
    if not settings.security.audit_log_path.is_absolute():
        settings.security.audit_log_path = (
            path.parent / settings.security.audit_log_path
        ).resolve()
    problems = _required_field_problems(settings)
    if problems:
        raise ConfigError(
            "Incomplete configuration ({}):\n{}".format(
                path, "\n".join(f"  - {p}" for p in problems)
            )
        )
    return settings


def load_provider_settings(config_path: Path | None = None,
                           profile: str | None = None) -> ProviderSettings:
    """Reuse normal config layering without requiring a browser session token."""
    settings, _ = _load_layered_settings(config_path, profile)
    return settings.provider


def _load_layered_settings(config_path: Path | None, profile: str | None = None) -> tuple[Settings, Path]:
    # Coerced rather than trusted: a str here would otherwise fail deep inside
    # _read_yaml with an unhelpful AttributeError.
    path = Path(config_path or os.environ.get("BROWSER_CONFIG", DEFAULT_CONFIG_PATH))
    token = _active_config_path.set(path)
    environment_token = _provider_environment.set(ChainMap(os.environ, dotenv_values(path.parent / ".env")))
    try:
        overrides = {"provider": {"active": profile}} if profile is not None else {}
        settings = Settings(_env_file=path.parent / ".env", **overrides)
    except ValidationError as exc:
        lines = [
            f"  - {'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        ]
        raise ConfigError(
            "Invalid configuration in {}:\n{}".format(path, "\n".join(lines))
        ) from exc
    finally:
        _active_config_path.reset(token)
        _provider_environment.reset(environment_token)

    return settings, path


def load_retrieval_settings(config_path: Path | None = None):
    """Operator CLI retrieval settings without requiring a browser auth token."""
    from types import SimpleNamespace
    path = Path(config_path or os.environ.get("BROWSER_CONFIG", DEFAULT_CONFIG_PATH))
    raw = _read_yaml(path)
    env = {**dotenv_values(path.parent / ".env"), **os.environ}
    sections = {}
    for name, cls in (("visual_retrieval", VisualSettings), ("multimodal_retrieval", MultimodalSettings)):
        data = dict(raw.get(name, {}))
        prefix = "BROWSER_" + name.upper() + "__"
        for key, value in env.items():
            if key.startswith(prefix) and value:
                field = key[len(prefix):].lower()
                try:
                    import json
                    parsed = json.loads(value)
                except (ValueError, TypeError): parsed = value
                data[field] = parsed
        sections[name] = cls.model_validate(data)
    return SimpleNamespace(**sections)
