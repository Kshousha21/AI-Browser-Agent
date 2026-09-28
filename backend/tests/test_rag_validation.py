"""Strict output boundary, bounded repair, typed failures and request isolation."""
import asyncio
import json
import pytest
from browser.providers.base import ProviderResponse, FinalAnswer, ToolCall, ProviderError
from local_rag import RagError, RagSettings
from local_rag.citations import validate_answer
from local_rag.context import build_context
from .rag_fixtures import corpus, rag, grounded, DeterministicRagProvider


@pytest.fixture
def retrieval(tmp_path):
    return corpus(tmp_path)[0]


def valid():
    return {"status": "answered", "answer": "Synthetic answer", "citations": [{"source_id": "S1", "evidence": "Synthetic"}]}


def invalid_cases():
    yield "invalid JSON", "not json", "rag_invalid_answer"
    yield "fence", "```json\n" + json.dumps(valid()) + "\n```", "rag_invalid_answer"
    yield "duplicate", '{"status":"answered","status":"insufficient_evidence","answer":"x","citations":[]}', "rag_invalid_answer"
    for key, value in [("reasoning", "private"), ("page_start", 9)]:
        yield key, json.dumps({**valid(), key: value}), "rag_invalid_answer"
    for answer in ["", " ", "x" * 4001, 3, None]:
        yield "answer", json.dumps({**valid(), "answer": answer}), "rag_invalid_answer"
    for citations in [[], None, valid()["citations"] * 21]:
        yield "citations", json.dumps({**valid(), "citations": citations}), "rag_invalid_answer"
    for citation, code in [
        ({"source_id": "S10", "evidence": "Synthetic"}, "rag_unknown_source"),
        ({"source_id": "S99", "evidence": "Synthetic"}, "rag_invalid_answer"),
        ({"source_id": "S1", "evidence": "not present"}, "rag_invalid_evidence"),
        ({"source_id": "S1", "evidence": "synthetic"}, "rag_invalid_evidence"),
        ({"source_id": "S1", "evidence": ""}, "rag_invalid_answer"),
        ({"source_id": "S1", "evidence": " "}, "rag_invalid_evidence"),
        ({"source_id": "S1", "evidence": "x" * 201}, "rag_invalid_answer"),
        ({"source_id": "S1", "evidence": 1}, "rag_invalid_answer"),
        ({"source_id": "S1", "evidence": "Synthetic", "document_id": "a" * 64}, "rag_invalid_answer"),
    ]:
        yield "citation", json.dumps({**valid(), "citations": [citation]}), code
    yield "abstention", json.dumps({**valid(), "status": "insufficient_evidence"}), "rag_invalid_answer"
    yield "raw limit", "x" * 16001, "rag_invalid_answer"
    yield "list", "[]", "rag_invalid_answer"


@pytest.mark.parametrize("label,raw,code", list(invalid_cases()))
async def test_invalid_output_fails_closed_without_repair(retrieval, label, raw, code, caplog):
    service, provider = rag(retrieval, DeterministicRagProvider(raw), RagSettings(max_attempts=1))
    with pytest.raises(RagError, match="^" + code + "$") as caught:
        await service.answer("Lumen", top_k=1)
    assert len(provider.calls) == 1 and provider.closed
    assert "private" not in str(caught.value) and raw not in caplog.text


async def test_one_repair_reuses_context_and_only_safe_correction(retrieval):
    bad = "private-raw-output-marker"
    service, provider = rag(retrieval, DeterministicRagProvider(bad, grounded))
    result = await service.answer("Lumen")
    assert result.attempts == len(provider.calls) == 2
    first, second = provider.calls[0][0], provider.calls[1][0]
    assert second[:3] == first
    assert second[-1].role == "system" and "rag_invalid_answer" in second[-1].content
    assert bad not in str(second) and result.status == "answered"


async def test_second_invalid_response_stops(retrieval):
    service, provider = rag(retrieval, DeterministicRagProvider("bad1", "bad2", grounded))
    with pytest.raises(RagError, match="^rag_invalid_answer$"):
        await service.answer("Lumen")
    assert len(provider.calls) == 2 and provider.closed


@pytest.mark.parametrize("response", [
    ProviderResponse(decision=ToolCall(tool="navigate", arguments={}, snapshot_id="s", reason="bad"), raw_text="private-tool"),
    ProviderResponse(decision=FinalAnswer(answer="private-tool"), raw_text="", tool_call_id="call"),
    ProviderError("private-sdk-body", code="invalid_tool"),
])
async def test_unexpected_tools_never_repaired(retrieval, response):
    service, provider = rag(retrieval, DeterministicRagProvider(response, grounded))
    with pytest.raises(RagError, match="^rag_unexpected_tool_call$"):
        await service.answer("Lumen")
    assert len(provider.calls) == 1 and provider.closed


@pytest.mark.parametrize("error,code", [(RuntimeError("private-body"), "rag_provider_failed"),
    (ProviderError("private-body", code="timeout"), "rag_provider_timeout"),
    (ProviderError("private-body", code="authentication_error"), "rag_provider_failed")])
async def test_safe_errors_no_repair_or_leak(retrieval, error, code, caplog):
    service, provider = rag(retrieval, DeterministicRagProvider(error))
    with pytest.raises(RagError, match="^" + code + "$") as caught:
        await service.answer("private-question Lumen")
    assert "private" not in str(caught.value) + caplog.text and len(provider.calls) == 1


async def test_timeout_cancels_and_closes_provider(retrieval):
    class Slow(DeterministicRagProvider):
        cancelled = False
        async def generate(self, *args, **kwargs):
            try:
                await asyncio.sleep(10)
            finally:
                self.cancelled = True
    provider = Slow()
    service, _ = rag(retrieval, provider, RagSettings(timeout_seconds=0.02))
    with pytest.raises(RagError, match="^rag_provider_timeout$"):
        await service.answer("Lumen")
    assert provider.closed and provider.cancelled


@pytest.mark.parametrize("reason", ["length", "max_tokens", "incomplete", "content_filter"])
async def test_truncated_response_not_presented_as_complete(retrieval, reason):
    response = ProviderResponse(decision=FinalAnswer(answer=json.dumps(valid())), raw_text="", finish_reason=reason)
    service, provider = rag(retrieval, DeterministicRagProvider(response))
    with pytest.raises(RagError, match="^rag_provider_failed$"):
        await service.answer("Lumen")
    assert len(provider.calls) == 1


def test_source_ids_are_bound_to_current_request(retrieval):
    a = build_context("Lumen", retrieval.search("Lumen", 1, mode="lexical"), RagSettings())
    b = build_context("Vega", retrieval.search("Vega", 1, mode="lexical"), RagSettings())
    raw = grounded(a.messages)
    with pytest.raises(RagError, match="^rag_invalid_evidence$"):
        validate_answer(raw, b.sources, RagSettings())


async def test_outer_cancellation_propagates_and_closes(retrieval):
    started = asyncio.Event()
    class Slow(DeterministicRagProvider):
        async def generate(self, *args, **kwargs):
            started.set()
            await asyncio.sleep(10)
    provider = Slow()
    service, _ = rag(retrieval, provider)
    task = asyncio.create_task(service.answer("Lumen"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.closed


async def test_whitespace_changed_quote_is_rejected_then_repaired_from_same_sources(retrieval):
    from .rag_fixtures import entries
    invalid = None
    def changed_whitespace(messages):
        nonlocal invalid
        entry = entries(messages)[0]
        exact = entry["text"][:100]
        changed = exact.replace(" ", "  ")
        assert changed != exact and changed not in entry["text"]
        invalid = json.dumps({"status": "answered", "answer": "Synthetic answer marker.",
            "citations": [{"source_id": entry["source_id"], "evidence": changed}]})
        return invalid
    def repair(messages):
        instruction = messages[-1].content
        assert "rag_invalid_evidence" in instruction
        assert "preserving its exact spaces" in instruction
        assert "Do not replace line breaks with spaces" in instruction
        assert invalid not in str(messages)
        hint_message = messages[-2]
        hints = json.loads(hint_message.content)
        assert hint_message.role == "user" and hints["trust"] == "untrusted_document_data"
        hint = hints["citation_copy_hints"][0]
        source = next(item for item in entries(messages) if item["source_id"] == hint["source_id"])
        assert hint["evidence"] in source["text"]
        return json.dumps({"status": "answered", "answer": "Synthetic answer.", "citations": [hint]})
    service, provider = rag(retrieval, DeterministicRagProvider(changed_whitespace, repair))
    result = await service.answer("Lumen", top_k=1)
    assert result.status == "answered" and result.attempts == 2
    assert provider.calls[1][0][:-2] == provider.calls[0][0]
    assert len(result.citations) == 1 and provider.closed


def test_copy_hints_do_not_repair_changed_words_or_unknown_sources(retrieval):
    from local_rag.citations import citation_copy_hints
    context = build_context("Lumen", retrieval.search("Lumen", 1, mode="lexical"), RagSettings())
    for citation in [{"source_id": "S1", "evidence": "invented private provider claim"},
                     {"source_id": "S15", "evidence": "Synthetic"},
                     {"source_id": "S1", "evidence_type": "visual"}]:
        raw = json.dumps({"status": "answered", "answer": "private response", "citations": [citation]})
        assert citation_copy_hints(raw, context.sources, RagSettings()) == []
        with pytest.raises(RagError):
            validate_answer(raw, context.sources, RagSettings())


def test_structured_repair_allows_only_verified_source_quote_pairs(retrieval):
    from pydantic import ValidationError
    from local_rag.citations import citation_repair_schema, citation_copy_hints
    context = build_context("Lumen", retrieval.search("Lumen", 1, mode="lexical"), RagSettings())
    hit = context.sources["S1"]
    raw = json.dumps({"status": "answered", "answer": "Synthetic answer.",
        "citations": [{"source_id": "S1", "evidence": hit.text[:100].replace(" ", "  ")}]})
    schema = citation_repair_schema(raw, context.sources, RagSettings())
    assert schema is not None
    # The original whitespace-altered quote remains invalid even during repair.
    with pytest.raises(ValidationError):
        schema.model_validate_json(raw)
    quote = citation_copy_hints(raw, context.sources, RagSettings())[0]
    repaired = {"status": "answered", "answer": "Synthetic answer.", "citations": [quote]}
    schema.model_validate(repaired)
    assert validate_answer(json.dumps(repaired), context.sources, RagSettings())[0] == "answered"
    for invalid in [{**quote, "source_id": "S15"}, {**quote, "evidence": "invented claim"},
                    {"source_id": "S1", "evidence_type": "visual"}]:
        with pytest.raises(ValidationError):
            schema.model_validate({**repaired, "citations": [invalid]})
    schema.model_validate({"status": "insufficient_evidence", "answer": "Unknown", "citations": []})
