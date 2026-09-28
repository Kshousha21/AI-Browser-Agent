"""Resolve one canonical provider profile and its environment-backed secrets."""

from __future__ import annotations

import logging
from collections import ChainMap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .provider_settings import (
    KEY_ENV_VARS, ProviderProfile, ProviderSettings, checked_header,
    checked_url, safe_validation_details,
)

log = logging.getLogger(__name__)


def resolve_provider(
    selection: Any, profiles: Mapping[str, Any], environment: Mapping[str, Any],
) -> ProviderSettings:
    """Validate only the selected profile, preserving the legacy runtime view."""
    if isinstance(selection, ProviderSettings):
        return selection
    selector = dict(selection or {})
    if not isinstance(profiles, Mapping):
        raise ValueError("providers must be a mapping of named profiles")
    if "active" not in selector:
        if profiles:
            raise ValueError("provider.active is required when providers profiles are configured")
        return _legacy(selector, environment)

    active = selector["active"]
    if not isinstance(active, str) or not active.strip():
        raise ValueError("provider.active must name a configured profile")
    if "name" in selector:
        if selector["name"] != active:
            raise ValueError("conflicting provider.name and provider.active; remove the legacy selector")
        log.warning("provider.name is deprecated; use provider.active only")
    if set(selector) - {"active", "name"}:
        raise ValueError("legacy provider fields conflict with canonical configuration; move them into providers.<active>")
    if active == "internal_custom":
        raise ValueError("internal_custom = pending endpoint contract; not registered")
    if active not in profiles:
        raise ValueError("provider.active does not name a configured providers profile")
    try:
        profile = ProviderProfile.model_validate(profiles[active])
    except ValidationError as exc:
        # A useful supported-type error without copying arbitrary user input.
        raw = profiles[active]
        if isinstance(raw, dict) and isinstance(raw.get("type"), str):
            ProviderProfile.supported_type(raw["type"])
        raise ValueError("invalid active provider profile: " + safe_validation_details(exc.errors())) from None

    problems: list[str] = []

    def env(name: str | None, field: str, *, required: bool = True) -> str:
        if not name:
            if required:
                problems.append(f"{field} must name an environment variable")
            return ""
        # Restrict reference syntax so error messages cannot contain secret values.
        import re
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            problems.append(f"{field} has an invalid environment variable name")
            return ""
        value = environment.get(name)
        if not isinstance(value, str) or not value.strip():
            if required:
                problems.append(f"{field}: required environment variable {name} is missing")
            return ""
        return value

    model = env(profile.model_env, "model_env") if profile.model_env else profile.model
    url = env(profile.base_url_env, "base_url_env") if profile.base_url_env else profile.base_url
    if profile.base_url and profile.base_url_env:
        problems.append("choose base_url or base_url_env, not both")
    generation_model = model or None
    api_version = None
    if profile.type == "azure_openai":
        url = env(profile.endpoint_env, "endpoint_env")
        model = env(profile.deployment_env, "deployment_env")
        if profile.api_version_env:
            api_version = env(profile.api_version_env, "api_version_env")
            if profile.api == "responses":
                problems.append("Azure Responses requires the v1 endpoint; omit api_version_env")
    if profile.api == "responses" and profile.type not in {"openai", "openai_compatible", "azure_openai"}:
        problems.append("api=responses is available only for OpenAI protocol profiles")
    if not model.strip():
        problems.append("model is required (or deployment_env for Azure)")
    if not url:
        url = {
            "anthropic": "https://api.anthropic.com",
            "openai": "https://api.openai.com/v1",
            "ollama": "http://127.0.0.1:11434",
        }.get(profile.type)
    if profile.type != "mock" and not url:
        problems.append("base_url or base_url_env is required")
    if url:
        try:
            url = checked_url(url)
        except ValueError as exc:
            problems.append(str(exc))

    auth_mode = profile.auth.mode
    key = ""
    headers: dict[str, str] = {}
    if profile.type in {"mock", "ollama"} and auth_mode == "native":
        auth_mode = "none"
    if auth_mode == "native":
        reference = profile.api_key_env or KEY_ENV_VARS.get(profile.type)
        if profile.type == "azure_openai":
            reference = reference or "AZURE_OPENAI_API_KEY"
        key = env(reference, "api_key_env")
    elif auth_mode != "none":
        value = env(profile.auth.token_env or profile.api_key_env, "auth.token_env")
        name = "Authorization" if auth_mode == "bearer" else profile.auth.header_name
        if not name:
            problems.append("auth.header_name is required for api_key_header")
        elif value:
            try:
                header, content = checked_header(name, f"Bearer {value}" if auth_mode == "bearer" else value)
                headers[header] = content
            except ValueError as exc:
                problems.append(str(exc))
    for name, reference in profile.headers_env.items():
        value = env(reference, "headers_env")
        try:
            header, content = checked_header(name, value)
            if name.lower() in {key.lower() for key in headers}:
                raise ValueError("duplicate authentication/header configuration")
            if name.lower() in {"authorization", "api-key", "x-api-key"} and auth_mode == "native":
                raise ValueError("native authentication headers cannot be overridden; select an explicit auth mode")
            headers[header] = content
        except ValueError as exc:
            problems.append(str(exc))

    paths: dict[str, str | None] = {}
    for field in ("ca_bundle", "client_cert", "client_key"):
        reference = getattr(profile.tls, field + "_env")
        value = env(reference, "tls." + field + "_env") if reference else None
        if value and not Path(value).is_file():
            problems.append(f"tls.{field}_env must reference a readable certificate/key file path")
        paths[field] = value
    if bool(paths["client_cert"]) != bool(paths["client_key"]):
        problems.append("mTLS requires both client_cert_env and client_key_env")
    if problems:
        raise ValueError("active provider configuration: " + "; ".join(problems))

    return ProviderSettings(
        active=active, name=profile.type, model=model, generation_model=generation_model,
        api_key=key, base_url=url, api_version=api_version, api=profile.api, auth_mode=auth_mode,
        headers=headers, temperature=profile.temperature, top_p=profile.top_p,
        effort=profile.effort, refusal_fallback=profile.refusal_fallback,
        supports_images=profile.supports_images,
        local_only=profile.local_only, num_ctx=profile.num_ctx, think=profile.think,
        tool_mode=profile.tool_mode,
        max_output_tokens=profile.max_output_tokens,
        timeout_seconds=profile.timeout_seconds, max_retries=profile.max_retries,
        **paths,
    )


def _legacy(selector: dict[str, Any], environment: Mapping[str, Any]) -> ProviderSettings:
    log.warning(
        "Legacy provider.name/flat configuration is deprecated; migrate to "
        "provider.active and providers.<profile>. Literal provider keys are deprecated."
    )
    result = ProviderSettings.model_validate(selector)
    bare = KEY_ENV_VARS.get(result.name, "")
    layers = environment.maps if isinstance(environment, ChainMap) else [environment]
    selected = next(
        (layer.get("BROWSER_PROVIDER__API_KEY") or layer.get(bare)
         for layer in layers if layer.get("BROWSER_PROVIDER__API_KEY") or layer.get(bare)),
        result.api_key,
    )
    if selected:
        result.api_key = selected
    if result.base_url:
        result.base_url = checked_url(result.base_url)
    return result
