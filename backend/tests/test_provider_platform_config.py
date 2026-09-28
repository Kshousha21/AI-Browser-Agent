"""Active-profile validation, migration, environment layering and secret hygiene."""

from __future__ import annotations

import json
import ssl

import pytest
import yaml

from browser.config import ConfigError, Settings, load_settings
from browser.provider_resolution import resolve_provider
from browser.provider_settings import SUPPORTED_TYPES
from browser.providers import ProviderError, build_provider
from browser.providers.transport import client_options

SECRET = "synthetic-sensitive-config-marker"


def profile(kind):
    result = {"type": kind, "model": "synthetic-model", "api_key_env": "FIXTURE_KEY"}
    if kind.endswith("_compatible"):
        result["base_url_env"] = "FIXTURE_URL"
    if kind == "azure_openai":
        result.update(endpoint_env="FIXTURE_URL", deployment_env="FIXTURE_DEPLOYMENT", api_version_env="FIXTURE_VERSION")
    return result


def environment():
    return {"FIXTURE_KEY": SECRET, "FIXTURE_URL": "http://127.0.0.1:45678", "FIXTURE_DEPLOYMENT": "fixture-deployment",
            "FIXTURE_VERSION": "2024-10-21"}


@pytest.mark.parametrize("kind", SUPPORTED_TYPES)
async def test_each_registered_type_validates_and_builds_without_network(kind):
    resolved = resolve_provider({"active": "chosen"}, {"chosen": profile(kind)}, environment())
    instance = build_provider(resolved)
    try:
        assert instance.name == kind
        assert instance.model == ("fixture-deployment" if kind == "azure_openai" else "synthetic-model")
        assert resolved.active == "chosen"
    finally:
        await instance.aclose()


def test_inactive_invalid_profiles_and_missing_credentials_do_not_block_startup(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"auth": {"token": "synthetic-backend-token-0123456789"},
                    "provider": {"active": "mock"}, "providers": {"mock": {"type": "mock", "model": "mock-1"},
                    "anthropic": {"type": "anthropic", "api_key_env": "MISSING"},
                    "future": {"type": "not-implemented"}, "bad": "not-a-profile"}}), encoding="utf-8")
    assert load_settings(path).provider.name == "mock"


@pytest.mark.parametrize("selection,profiles,match", [
    ({"active": "test"}, {"test": {"type": "missing-type"}}, "supported types"),
    ({"active": "test"}, {"test": {"type": "openai"}}, "model is required"),
    ({"active": "test"}, {"test": {"type": "openai", "model": "test"}}, "OPENAI_API_KEY"),
    ({"active": "absent"}, {"test": {"type": "mock"}}, "does not name"),
    ({"active": "test", "name": "other"}, {"test": {"type": "mock"}}, "conflicting"),
    ({"active": "test", "model": "legacy"}, {"test": {"type": "mock"}}, "legacy provider fields conflict"),
    ({"active": "internal_custom"}, {}, "pending endpoint contract"),
    ({"active": "test"}, {"test": {"type": "internal_custom"}}, "pending endpoint contract"),
    ({}, {"test": {"type": "mock"}}, "provider.active is required"),
    ({"active": "test"}, {"test": {"type": "mock", "model": "test", "capabilities": {"tools": True}}}, "extra_forbidden"),
    ({"active": "test"}, {"test": {"type": "mock", "model": "test", "api_key": SECRET}}, "extra_forbidden"),
])
def test_invalid_active_configuration_fails_safely(selection, profiles, match):
    with pytest.raises(ValueError, match=match) as error:
        resolve_provider(selection, profiles, {})
    assert SECRET not in str(error.value)


@pytest.mark.parametrize("url", ["ftp://example.test", "relative/path", "https://user:secret@example.test",
                                     "https://example.test?token=" + SECRET, "https://example.test/#fragment",
                                     "https://example.test:bad", "https://example.test/\r\nheader"])
def test_invalid_internal_urls_never_echo_input(url):
    with pytest.raises(ValueError, match="base URL") as error:
        resolve_provider({"active": "test"}, {"test": {"type": "openai_compatible", "model": "test",
                         "base_url": url, "auth": {"mode": "none"}}}, {})
    assert SECRET not in str(error.value)


@pytest.mark.parametrize("kind", ["openai", "anthropic", "openai_compatible", "anthropic_compatible", "ollama"])
def test_no_auth_requires_no_key(kind):
    entry = {**profile(kind), "auth": {"mode": "none"}}
    resolved = resolve_provider({"active": "test"}, {"test": entry}, {"FIXTURE_URL": "http://127.0.0.1:45678"})
    assert resolved.api_key == ""
    assert resolved.auth_mode == "none"


def test_legacy_warns_and_agreeing_selector_is_not_a_second_source(caplog):
    legacy = resolve_provider({"name": "mock"}, {}, {})
    assert legacy.model == "mock-1"
    assert "deprecated" in caplog.text
    caplog.clear()
    canonical = resolve_provider({"active": "local", "name": "local"}, {"local": {"type": "mock", "model": "chosen"}}, {})
    assert canonical.name == "mock" and canonical.active == "local"
    assert "deprecated" in caplog.text


def test_canonical_environment_overrides_and_empty_example_selector(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"auth": {"token": "synthetic-backend-token-0123456789"}, "provider": {"active": "internal"},
                    "providers": {"internal": {**profile("openai_compatible"), "model_env": "FIXTURE_MODEL"},
                                  "local": {"type": "mock", "model": "yaml-model"}}}), encoding="utf-8")
    (tmp_path / ".env").write_text("BROWSER_PROVIDER__ACTIVE=\nFIXTURE_KEY=dotenv-sensitive-marker\n"
                                    "FIXTURE_URL=http://127.0.0.1:45678\nFIXTURE_MODEL=dotenv-model\n", encoding="utf-8")
    monkeypatch.setenv("FIXTURE_KEY", SECRET)
    current = load_settings(path)
    assert current.provider.api_key == SECRET
    assert current.provider.model == "dotenv-model"
    assert SECRET not in current.model_dump_json() + repr(current)
    assert "dotenv-sensitive-marker" not in current.model_dump_json()
    monkeypatch.setenv("BROWSER_PROVIDER__ACTIVE", "local")
    monkeypatch.setenv("BROWSER_PROVIDERS__LOCAL__MODEL", "environment-model")
    assert load_settings(path).provider.model == "environment-model"


@pytest.mark.parametrize("field", ["api_key", "headers", "client_key", "client_cert", "ca_bundle"])
def test_resolved_secrets_are_excluded_from_public_settings(field):
    resolved = resolve_provider({"active": "test"}, {"test": {"type": "mock", "model": "mock-1"}}, {})
    setattr(resolved, field, {"X-Secret": SECRET} if field == "headers" else SECRET)
    settings = Settings(provider=resolved, auth={"token": SECRET})
    assert SECRET not in settings.model_dump_json() + repr(settings)


@pytest.mark.parametrize("entry,match", [
    ({"auth": {"mode": "bearer", "token_env": "MISSING"}}, "MISSING"),
    ({"auth": {"mode": "api_key_header", "token_env": "FIXTURE_KEY"}}, "header_name"),
    ({"auth": {"mode": "api_key_header", "header_name": "Cookie", "token_env": "FIXTURE_KEY"}}, "reserved"),
    ({"headers_env": {"Authorization": "FIXTURE_KEY"}}, "cannot be overridden"),
    ({"headers_env": {"X-Custom": "bad reference"}}, "invalid environment variable name"),
    ({"max_retries": 100}, "less_than_equal"),
    ({"timeout_seconds": 0}, "greater_than"),
    ({"api": "responses", "type": "ollama"}, "OpenAI protocol"),
])
def test_invalid_auth_transport_and_bounds(entry, match):
    with pytest.raises(ValueError, match=match):
        resolve_provider({"active": "test"}, {"test": {**profile("openai"), **entry}}, environment())


def test_azure_responses_requires_v1():
    with pytest.raises(ValueError, match="v1 endpoint"):
        resolve_provider({"active": "test"}, {"test": {**profile("azure_openai"), "api": "responses"}}, environment())


def test_mtls_paths_and_verified_ssl_context(tmp_path, monkeypatch):
    certificate = tmp_path / "fixture-cert.pem"
    key = tmp_path / "fixture-key.pem"
    ca = tmp_path / "fixture-ca.pem"
    for path in (certificate, key, ca):
        path.write_text("synthetic-invalid-pem", encoding="utf-8")
    entry = {**profile("openai"), "tls": {"client_cert_env": "CERT", "client_key_env": "KEY", "ca_bundle_env": "CA"}}
    env = {**environment(), "CERT": str(certificate), "KEY": str(key), "CA": str(ca)}
    resolved = resolve_provider({"active": "test"}, {"test": entry}, env)
    with pytest.raises(ProviderError, match="TLS") as error:
        client_options(resolved)
    assert str(key) not in str(error.value)
    context = ssl.create_default_context()
    calls = []
    class Context:
        def load_cert_chain(self, cert, private):
            calls.append((cert, private))
    def create(*, cafile):
        assert cafile == str(ca)
        assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
        return Context()
    monkeypatch.setattr("browser.providers.transport.ssl.create_default_context", create)
    assert isinstance(client_options(resolved)["verify"], Context)
    assert calls == [(str(certificate), str(key))]
    env.pop("KEY")
    with pytest.raises(ValueError, match="mTLS requires both"):
        resolve_provider({"active": "test"}, {"test": entry}, env)


def test_shipped_example_contains_only_empty_environment_values():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / ".env.example"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            assert line.partition("=")[2] == ""
