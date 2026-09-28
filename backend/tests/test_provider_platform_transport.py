"""Enterprise auth and OpenAI request normalization through loopback HTTP."""

from __future__ import annotations

import pytest

from browser.provider_resolution import resolve_provider
from browser.providers import Message, ProviderError, build_provider
from browser.actions import tool_schemas
from tests.provider_wire import PROTOCOLS, answer, make_provider, running_server

pytestmark = pytest.mark.asyncio
MESSAGES = [Message(role="user", content="Synthetic fixture only")]


@pytest.fixture(scope="module")
def server():
    with running_server() as instance:
        yield instance


@pytest.fixture
def wire(server):
    server.reset()
    return server


@pytest.mark.parametrize("case", ["openai_compatible", "anthropic_compatible", "ollama"])
@pytest.mark.parametrize("mode", ["none", "bearer", "api_key_header", "custom"])
async def test_environment_auth_headers_reach_only_configured_endpoint(case, mode, wire):
    auth = {"mode": "none" if mode == "custom" else mode}
    if mode in {"bearer", "api_key_header"}:
        auth["token_env"] = "ENTERPRISE_TOKEN"
    if mode == "api_key_header":
        auth["header_name"] = "X-Enterprise-Key"
    entry = {"type": case, "model": "synthetic-model", "base_url": wire.url,
             "auth": auth, "headers_env": {"X-Enterprise-Context": "EXTRA"}, "max_retries": 0,
             "refusal_fallback": False}
    if mode == "custom":
        entry["headers_env"]["Authorization"] = "AUTH_HEADER"
    env = {"ENTERPRISE_TOKEN": "synthetic-enterprise-token", "EXTRA": "synthetic-context", "AUTH_HEADER": "Bearer synthetic-custom"}
    settings = resolve_provider({"active": "test"}, {"test": entry}, env)
    provider = build_provider(settings)
    wire.replies.append(answer(case))
    try:
        await provider.generate(MESSAGES)
        headers = wire.received[0]["headers"]
        assert headers["x-enterprise-context"] == "synthetic-context"
        assert "x-api-key" not in headers and "api-key" not in headers
        assert "cookie" not in headers
        if mode == "bearer":
            assert headers["authorization"] == "Bearer synthetic-enterprise-token"
        elif mode == "api_key_header":
            assert headers["x-enterprise-key"] == "synthetic-enterprise-token"
            assert "authorization" not in headers
        elif mode == "custom":
            assert headers["authorization"] == "Bearer synthetic-custom"
        else:
            assert "authorization" not in headers
        assert settings.api_key == ""
    finally:
        await provider.aclose()


@pytest.mark.parametrize("case", PROTOCOLS)
async def test_redirect_is_not_followed(case, wire):
    from tests.provider_wire import Reply
    provider, _ = make_provider(case, wire)
    wire.replies.extend([Reply({}, status=307, headers={"Location": wire.url + "/other"}), answer(case)])
    try:
        with pytest.raises(ProviderError):
            await provider.generate(MESSAGES)
        assert len(wire.received) == 1
    finally:
        await provider.aclose()


@pytest.mark.parametrize("case", ["openai_responses", "azure_responses", "azure_openai", "azure_versioned"])
@pytest.mark.parametrize("mode", ["none", "bearer"])
async def test_openai_family_explicit_auth(case, mode, wire):
    provider, _ = make_provider(case, wire, auth={"mode": mode, "token_env": "FIXTURE_KEY"})
    wire.replies.append(answer(case))
    try:
        await provider.generate(MESSAGES)
        headers = wire.received[0]["headers"]
        if mode == "bearer":
            assert headers["authorization"] == "Bearer synthetic-secret-marker"
        else:
            assert "authorization" not in headers
        assert "api-key" not in headers and "x-api-key" not in headers
    finally:
        await provider.aclose()


@pytest.mark.parametrize("case,model,effort,sampling", [
    ("openai", "gpt-4o-mini", None, True), ("openai", "o3", "high", False),
    ("openai", "gpt-5", "medium", False), ("openai", "gpt-5.2", "none", True),
    ("openai_responses", "gpt-6-astra", "low", False),
    ("azure_responses", "gpt-6-astra", "low", False),
])
async def test_generation_settings_follow_protocol_and_underlying_model(case, model, effort, sampling, wire):
    provider, _ = make_provider(case, wire, model=model, effort=effort, temperature=0.2, top_p=0.9)
    wire.replies.append(answer(case))
    try:
        await provider.generate(MESSAGES, tool_schemas())
        body = wire.received[0]["body"]
        assert ("temperature" in body) is sampling
        assert ("top_p" in body) is sampling
        if "responses" in case:
            assert body["reasoning"] == {"effort": effort}
            assert "max_tokens" not in body and "max_completion_tokens" not in body
        elif effort:
            assert body["reasoning_effort"] == effort
    finally:
        await provider.aclose()


@pytest.mark.parametrize("api,effort,match", [("chat_completions", "low", "api=responses"),
                                              ("responses", "none", "low or higher"),
                                              ("responses", "minimal", "low or higher")])
async def test_invalid_gpt6_api_or_effort_fails_before_network(api, effort, match, wire):
    provider, _ = make_provider("openai", wire, model="gpt-6-astra", api=api, effort=effort)
    try:
        with pytest.raises(ProviderError, match=match):
            await provider.generate(MESSAGES, tool_schemas())
        assert not wire.received
    finally:
        await provider.aclose()
