"""Actual registry/SDK wire requests to loopback fixtures, never live endpoints."""
import json
import logging
import pytest
from browser.provider_settings import ProviderSettings
from browser.providers.transport import client_options
from local_rag import ProviderBridge, RagError, RagService, RagSettings
from .provider_wire import CASES, answer, make_provider, running_server
from .rag_fixtures import corpus, INJECTION


@pytest.fixture(scope="module")
def server():
    with running_server() as server:
        yield server


@pytest.fixture(autouse=True)
def reset(server):
    server.reset()


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("flow", ["valid", "repair", "unexpected_tool"])
async def test_registry_wire_no_tools_trust_and_local_validator(tmp_path, server, case, flow, caplog):
    caplog.set_level(logging.DEBUG)
    retrieval, _, docs = corpus(tmp_path)
    unused, settings = make_provider(case, server)
    await unused.aclose()
    good = json.dumps({"status": "answered", "answer": "The manual has an instruction-like sentence.",
        "citations": [{"source_id": "S1", "evidence": INJECTION}]})
    if flow == "unexpected_tool":
        server.replies.append(answer(case, name="navigate", arguments={"url": "https://invalid.example"}))
    elif flow == "repair":
        server.replies.extend([answer(case, text="private-invalid-wire-body"), answer(case, text=good)])
    else:
        server.replies.append(answer(case, text=good))
    service = RagService(retrieval, ProviderBridge(settings))
    if flow == "unexpected_tool":
        with pytest.raises(RagError, match="^rag_unexpected_tool_call$"):
            await service.answer("manual", document_ids=[docs["injection"].document_id])
    else:
        result = await service.answer("manual", document_ids=[docs["injection"].document_id])
        assert result.citations[0].document_id == docs["injection"].document_id
        assert result.attempts == (2 if flow == "repair" else 1)
    assert len(server.received) == (2 if flow == "repair" else 1)
    for request in server.received:
        body = request["body"]
        assert not body.get("tools")
        messages = body.get("messages", body.get("input", []))
        trusted = json.dumps(body.get("system", "")) + json.dumps([
            item for item in messages if item.get("role") in {"system", "developer"}])
        untrusted = json.dumps([item for item in messages if item.get("role") == "user"])
        assert "No tools are available" in trusted and INJECTION not in trusted
        assert INJECTION in untrusted and "untrusted_document_data" in untrusted
        assert "private-invalid-wire-body" not in json.dumps(body)
        assert "cookie" not in request["headers"]
    for private in (INJECTION, "private-invalid-wire-body", "synthetic-secret-marker"):
        assert private not in caplog.text


@pytest.mark.parametrize("endpoint", ["http://127.0.0.1:8123/v1", "http://127.1.2.3:8123", "http://[::1]:8123", "http://localhost:8123", "http://[::ffff:127.0.0.1]"])
def test_loopback_policy_and_proxy_disabled(endpoint):
    original = ProviderSettings(name="openai_compatible", model="synthetic", base_url=endpoint,
                                auth_mode="none", max_retries=3)
    result = ProviderBridge(original).settings_for_request(RagSettings())
    assert result.trust_env is False and result.max_retries == 0 and result.refusal_fallback is False
    assert result.max_output_tokens <= 2048 and result.timeout_seconds <= 30
    assert client_options(result)["trust_env"] is False
    assert original.trust_env is True and original.max_retries == 3
    assert client_options(original)["trust_env"] is True
    if "localhost" in endpoint:
        assert "127.0.0.1" in result.base_url


@pytest.mark.parametrize("endpoint", ["https://approved.example/v1", "http://192.168.1.2:8123", "http://localhost.invalid:8123", "http://127.0.0.1.invalid", "http://2130706433", "http://0.0.0.0"])
def test_nonloopback_requires_independent_explicit_consent(endpoint):
    settings = ProviderSettings(name="openai_compatible", model="synthetic", base_url=endpoint, auth_mode="none")
    with pytest.raises(RagError, match="^rag_remote_context_denied$"):
        ProviderBridge(settings).settings_for_request(RagSettings())
    allowed = ProviderBridge(settings, allow_remote_context=True).settings_for_request(RagSettings())
    assert allowed.base_url == endpoint and allowed.trust_env is True


@pytest.mark.parametrize("name", ["openai", "anthropic"])
def test_implicit_vendor_default_endpoint_is_remote(name):
    with pytest.raises(RagError, match="^rag_remote_context_denied$"):
        ProviderBridge(ProviderSettings(name=name, api_key="synthetic")).settings_for_request(RagSettings())


def test_browser_mock_is_not_production_rag_provider():
    with pytest.raises(RagError, match="^rag_provider_not_allowed$"):
        ProviderBridge(ProviderSettings(name="mock")).settings_for_request(RagSettings())


async def test_environment_proxy_cannot_receive_loopback_context(tmp_path, server, monkeypatch):
    with running_server() as proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            monkeypatch.setenv(key, proxy.url)
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")
        retrieval, _, docs = corpus(tmp_path)
        settings = ProviderSettings(name="ollama", model="synthetic", base_url=server.url, auth_mode="none")
        server.replies.append(answer("ollama", text=json.dumps({"status": "answered", "answer": "12 W",
            "citations": [{"source_id": "S1", "evidence": "12 W"}]})))
        result = await RagService(retrieval, ProviderBridge(settings)).answer("12 W Lumen", top_k=1, mode="lexical")
        assert result.answer == "12 W" and len(server.received) == 1 and proxy.received == []


async def test_redirect_does_not_replay_context(tmp_path, server):
    from .provider_wire import Reply
    with running_server() as redirected:
        retrieval, _, _ = corpus(tmp_path)
        server.replies.append(Reply({}, status=307, headers={"Location": redirected.url}))
        settings = ProviderSettings(name="ollama", model="synthetic", base_url=server.url, auth_mode="none")
        with pytest.raises(RagError, match="^rag_provider_failed$"):
            await RagService(retrieval, ProviderBridge(settings)).answer("Lumen")
        assert len(server.received) == 1 and not redirected.received
