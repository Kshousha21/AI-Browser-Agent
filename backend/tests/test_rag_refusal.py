"""Explicit refusals abstain; unsafe/malformed/transport results still fail."""
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from browser.providers.base import ProviderError, ProviderResponse, FinalAnswer, ToolCall
from local_rag import ProviderBridge, RagError, RagService, RagSettings
from local_rag.models import INSUFFICIENT
from .provider_wire import answer, make_provider, running_server, sse, Reply
from .rag_fixtures import corpus, rag, DeterministicRagProvider, INJECTION, grounded
from .retrieval_fixtures import TestEmbedder


def refusal_reply(text="", *, tool=False):
    if tool or text:
        reply = answer("anthropic", name="navigate" if tool else None,
                       arguments={"url": "https://invalid.example"}, text=text)
        reply.body = reply.body.replace('"stop_reason": "tool_use"' if tool else '"stop_reason": "end_turn"',
                                        '"stop_reason": "refusal"')
        return reply
    start = {"id":"synthetic", "type":"message", "role":"assistant", "model":"synthetic",
             "content":[], "stop_reason":None, "stop_sequence":None,
             "usage":{"input_tokens":10,"output_tokens":0}}
    return Reply(sse("message_start", {"type":"message_start","message":start}) +
        sse("message_delta", {"type":"message_delta","delta":{"stop_reason":"refusal",
            "stop_details":{"type":"refusal","category":"general_harms","explanation":INJECTION}},
            "usage":{"output_tokens":0}}) +
        sse("message_stop", {"type":"message_stop"}), media="text/event-stream")


@pytest.mark.parametrize("text", ["", "RAW_PRIVATE_REFUSAL " + INJECTION,
    '{"status":"answered","answer":"untrusted","citations":[{"source_id":"S99","evidence":"bad"}]}'])
async def test_native_refusal_wire_is_fixed_local_abstention(tmp_path, text, caplog):
    caplog.set_level(logging.DEBUG)
    retrieval, store, docs = corpus(tmp_path)
    before = {p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with running_server() as wire:
        unused, settings = make_provider("anthropic", wire, model="claude-opus-5")
        await unused.aclose()
        wire.replies.append(refusal_reply(text))
        result = await RagService(retrieval, ProviderBridge(settings)).answer(
            "Describe the manual as text", document_ids=[docs["injection"].document_id])
        assert len(wire.received) == result.attempts == 1
        body = wire.received[0]["body"]
        assert not body.get("tools") and not body.get("fallbacks")
        assert INJECTION not in body["system"]
        assert INJECTION in json.dumps(body["messages"])
    assert result.status == "insufficient_evidence" and result.answer == INSUFFICIENT
    assert result.citations == [] and result.measurements["provider_refusal"] == 1
    assert before == {p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert INJECTION not in caplog.text + repr(result) + json.dumps(result.safe_metadata())


async def test_refusal_with_unadvertised_tool_still_fails_no_repair(tmp_path):
    retrieval, _, _ = corpus(tmp_path)
    with running_server() as wire:
        unused, settings = make_provider("anthropic", wire)
        await unused.aclose()
        wire.replies.append(refusal_reply(tool=True))
        with pytest.raises(RagError, match="^rag_unexpected_tool_call$"):
            await RagService(retrieval, ProviderBridge(settings)).answer("Lumen")
        assert len(wire.received) == 1


@pytest.mark.parametrize("response", [
    ProviderError("RAW_PRIVATE_REFUSAL", code="refusal"),
    ProviderResponse(decision=FinalAnswer(answer="RAW_PRIVATE_REFUSAL"),
                     raw_text="RAW_PRIVATE_REFUSAL", finish_reason="refusal"),
])
@pytest.mark.parametrize("repair_first", [False, True])
async def test_general_refusal_not_fixture_specific_and_never_repaired(tmp_path, response, repair_first):
    retrieval, _, _ = corpus(tmp_path)
    replies = ["invalid-json", response, grounded] if repair_first else [response, grounded]
    service, provider = rag(retrieval, DeterministicRagProvider(*replies))
    result = await service.answer("Ordinary Lumen question")
    assert result.answer == INSUFFICIENT and not result.citations
    assert result.attempts == len(provider.calls) == 1 + int(repair_first)
    assert provider.closed and result.measurements["provider_refusal"] == 1
    assert "RAW_PRIVATE_REFUSAL" not in str(provider.calls)


@pytest.mark.parametrize("response,code", [
    ("I refuse to answer.", "rag_invalid_answer"),
    (ProviderError("refusal"), "rag_provider_failed"),
    (ProviderError("refusal",code="authentication_error"), "rag_provider_failed"),
    (ProviderError("refusal",code="timeout"), "rag_provider_timeout"),
    (ProviderResponse(decision=FinalAnswer(answer="refusal"),raw_text="",finish_reason="max_tokens"), "rag_provider_failed"),
    (ProviderResponse(decision=ToolCall(tool="navigate",arguments={},reason="",snapshot_id="s"),
                      raw_text="",finish_reason="refusal"), "rag_unexpected_tool_call"),
    (ProviderResponse(decision=FinalAnswer(answer="refusal"),raw_text="",finish_reason="refusal",
                      tool_call_id="unadvertised"), "rag_unexpected_tool_call"),
])
async def test_no_prose_heuristic_or_blanket_failure_to_pass(tmp_path, response, code):
    retrieval, _, _ = corpus(tmp_path)
    service, provider = rag(retrieval, DeterministicRagProvider(response), RagSettings(max_attempts=1))
    with pytest.raises(RagError, match="^" + code + "$"):
        await service.answer("Lumen")
    assert len(provider.calls) == 1


@pytest.mark.parametrize("kind", ["refusal", "error", "fake_abstention", "cited_abstention"])
async def test_acceptance_requires_exact_canary_outcome(tmp_path, monkeypatch, kind):
    spec = importlib.util.spec_from_file_location("rag_check",
        Path(__file__).resolve().parents[2] / "tools/rag_acceptance.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setattr(helper, "OnnxEmbedder", lambda *a, **kw:TestEmbedder())
    response = ProviderError("synthetic", code="refusal" if kind=="refusal" else "provider_error")
    if kind == "fake_abstention":
        response = json.dumps({"status":"insufficient_evidence","answer":"fake","citations":[]})
    if kind == "cited_abstention":
        response = json.dumps({"status":"insufficient_evidence","answer":"fake",
                               "citations":[{"source_id":"S1","evidence":"Synthetic"}]})
    class Factory:
        def __call__(self, settings):
            return DeterministicRagProvider(response, response)
    from browser.provider_settings import ProviderSettings
    bridge = ProviderBridge(ProviderSettings(name="ollama", model="synthetic", auth_mode="none"),
                            factory=Factory())
    outcomes = await helper.acceptance(tmp_path, bridge, SimpleNamespace(model_directory=tmp_path,device="cpu"))
    assert len(outcomes) == 7 and outcomes[-1]["passed"] == (kind == "refusal")
    # Refusal does not make the five normal answered contracts pass.
    assert not outcomes[0]["passed"]
