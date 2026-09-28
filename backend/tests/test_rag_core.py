"""Phase 4C -> 4D -> 4E acceptance using only invented PDFs and test vectors."""
from contextlib import closing
import json
import sqlite3
import pytest
from local_rag import RagError, RagSettings
from local_rag.context import build_context, check_request_budget
from local_rag.models import INSUFFICIENT
from .rag_fixtures import corpus, rag, entries, grounded, DeterministicRagProvider, INJECTION


@pytest.fixture
def data(tmp_path):
    return corpus(tmp_path)


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
async def test_single_source_and_numeric_unit_provenance(data, mode):
    retrieval, store, documents = data
    provider = DeterministicRagProvider(lambda m: grounded(m, ["12 W"], "The Lumen package uses 12 W."))
    service, _ = rag(retrieval, provider)
    result = await service.answer("Lumen power", mode=mode, document_ids=[documents["lumen"].document_id])
    assert result.status == "answered" and result.answer.endswith("12 W.")
    citation = result.citations[0]
    assert citation.document_id == documents["lumen"].document_id
    assert citation.page_start == citation.page_end == 1
    hit = next(h for h in retrieval.search("Lumen power", mode=mode).results if h.chunk_id == citation.chunk_id)
    for key in ("chunk_index", "char_start", "char_end", "trust"):
        assert getattr(citation, key) == getattr(hit, key)
    assert provider.closed and result.attempts == 1
    assert "12 W" not in repr(result) and "answer" not in result.safe_metadata()
    # Evidence TYPE is safe provenance; exact private evidence substrings are
    # still excluded from normalized results.
    assert all("evidence" not in citation for citation in result.model_dump()["citations"])


async def test_multi_page_answer(data):
    retrieval, _, docs = data
    provider = DeterministicRagProvider(lambda m: grounded(m,
        ["copper plate", "ceramic housing"], "Lumen uses a copper plate and ceramic housing."))
    service, _ = rag(retrieval, provider)
    result = await service.answer("Lumen materials", document_ids=[docs["lumen"].document_id])
    assert {c.page_start for c in result.citations} == {1, 2}
    assert {c.document_id for c in result.citations} == {docs["lumen"].document_id}
    assert len({c.chunk_id for c in result.citations}) == 2


async def test_conflict_contract_and_locally_mapped_sources(data):
    retrieval, _, docs = data
    provider = DeterministicRagProvider(lambda m: grounded(m, ["8 W", "11 W"],
        "The sources conflict: Vega is listed at both 8 W and 11 W."))
    service, _ = rag(retrieval, provider)
    result = await service.answer("Vega maximum power", document_ids=[docs[k].document_id for k in ("conflict_a", "conflict_b")])
    assert "conflict" in result.answer
    assert {c.document_id for c in result.citations} == {docs[k].document_id for k in ("conflict_a", "conflict_b")}
    assert "cite both" in provider.calls[0][0][0].content


@pytest.mark.parametrize("question,document,evidence,answer", [
    ("Wie viel Leistung verwendet Lumen?", "lumen", "12 W", "Lumen verwendet 12 W."),
    ("How much power does Nara use?", "german", "6 W", "Nara uses 6 W."),
])
async def test_multilingual_fixture_flow(data, question, document, evidence, answer):
    retrieval, _, docs = data
    provider = DeterministicRagProvider(lambda m: grounded(m, [evidence], answer))
    service, _ = rag(retrieval, provider)
    result = await service.answer(question, document_ids=[docs[document].document_id])
    assert result.answer == answer and result.citations[0].document_id == docs[document].document_id


async def test_irrelevant_question_abstains_without_claiming_semantic_threshold(data):
    service, provider = rag(data[0], DeterministicRagProvider(json.dumps({
        "status": "insufficient_evidence", "answer": "Fabricated fact should be discarded.", "citations": []})))
    result = await service.answer("How many moons does invented planet Wug have?")
    assert result.context_count == 5 and result.answer == INSUFFICIENT
    assert result.status == "insufficient_evidence" and result.citations == []
    assert len(provider.calls) == 1


async def test_zero_lexical_results_never_construct_provider(data):
    service, provider = rag(data[0])
    result = await service.answer("zzunfindabletermqq", mode="lexical")
    assert result.status == "insufficient_evidence" and result.attempts == 0
    assert result.retrieved_count == result.context_count == 0 and provider.calls == []


@pytest.mark.parametrize("kwargs", [
    {"question": ""}, {"question": "   "}, {"question": "x" * 1001}, {"question": "\x00"},
    {"question": 12}, {"top_k": 0}, {"top_k": 11}, {"top_k": True}, {"top_k": "5"},
    {"mode": "other"}, {"document_ids": []}, {"document_ids": ["private-path"]},
    {"document_ids": ["a" * 64, "a" * 64]},
])
async def test_invalid_requests_are_safe_and_precede_provider(data, kwargs):
    service, provider = rag(data[0])
    with pytest.raises(RagError, match="^rag_invalid_request$"):
        await service.answer(**{"question": "Lumen", **kwargs})
    assert not provider.calls


async def test_unknown_document_filter_fails_closed(data):
    service, provider = rag(data[0])
    with pytest.raises(RagError, match="^rag_retrieval_failed$"):
        await service.answer("Lumen", document_ids=["a" * 64])
    assert not provider.calls


@pytest.mark.parametrize("top_k", [3, 5, 10])
async def test_top_k_rank_filter_and_no_full_corpus(tmp_path, top_k):
    pages = {f"doc{i}": [f"Synthetic thermal chunk {i}. " + "bounded evidence " * 75] for i in range(20)}
    retrieval, _, docs = corpus(tmp_path, pages=pages)
    service, provider = rag(retrieval)
    expected = retrieval.search("thermal", top_k=top_k)
    result = await service.answer("thermal", top_k=top_k)
    visible = entries(provider.calls[0][0])
    assert [x["document_id"] for x in visible] == [h.document_id for h in expected.results[:len(visible)]]
    assert result.retrieved_count == top_k and result.context_count <= top_k
    assert result.dropped_due_budget == top_k - len(visible)
    assert [x["source_id"] for x in visible] == [f"S{i+1}" for i in range(len(visible))]
    assert all(set(x) == {"source_id", "document_id", "page_start", "page_end", "trust", "text"} for x in visible)
    assert len(visible) < len(docs) and result.measurements["context_chars"] <= 14000
    serialized = json.dumps([m.content for m in provider.calls[0][0]])
    assert str(tmp_path) not in serialized and ".pdf" not in serialized and "vector" not in serialized


async def test_small_context_keeps_whole_chunks_and_reports_drops(data):
    retrieval = data[0]
    result = retrieval.search("thermal", 5)
    settings = RagSettings(max_context_chars=750)
    context = build_context("thermal", result, settings)
    assert 0 < len(context.sources) < 5
    assert context.context_chars <= 750
    selected = entries(context.messages)
    assert [item["text"] for item in selected] == [h.text for h in context.sources.values()]
    service, _ = rag(retrieval, settings=settings)
    answer = await service.answer("thermal")
    assert answer.context_count + answer.dropped_due_budget == 5


def test_oversized_first_chunk_skipped_without_cutting_provenance(data):
    found = data[0].search("thermal", 3)
    first = found.results[0].model_copy(update={"text": "x" * 3000, "char_end": 3000})
    found = found.model_copy(update={"results": [first, *found.results[1:]]})
    context = build_context("thermal", found, RagSettings(max_context_chars=1000))
    assert "S1" not in context.sources and "S2" in context.sources
    assert context.sources["S2"] is found.results[1]


def test_no_chunk_fits_is_typed(data):
    found = data[0].search("thermal", 1)
    first = found.results[0].model_copy(update={"text": "x" * 3000, "char_end": 3000})
    with pytest.raises(RagError, match="^rag_context_budget$"):
        build_context("thermal", found.model_copy(update={"results": [first]}), RagSettings(max_context_chars=512))


def test_request_bound_includes_question_json_escaping(data):
    found = data[0].search("thermal", 1)
    with pytest.raises(RagError, match="^rag_context_budget$"):
        build_context("\u2603" * 1000, found, RagSettings(max_request_chars=2048))


async def test_injection_is_untrusted_unchanged_and_no_tools(data):
    retrieval, _, docs = data
    service, provider = rag(retrieval, DeterministicRagProvider(lambda m: grounded(m, [INJECTION], "The manual contains an instruction-like sentence.")))
    result = await service.answer("What instruction-like sentence is in the manual?", document_ids=[docs["injection"].document_id])
    messages, tools, schema = provider.calls[0]
    assert tools == [] and schema is None
    assert INJECTION not in messages[0].content
    assert entries(messages)[0]["text"].endswith(INJECTION)
    assert entries(messages)[0]["trust"] == result.citations[0].trust == "untrusted_document_data"
    assert "No tools are available" in messages[0].content
    assert result.context_count == 1


def test_random_boundaries_and_collision_rejection(data, monkeypatch):
    found = data[0].search("Lumen", 1)
    a = build_context("Lumen", found, RagSettings())
    b = build_context("Lumen", found, RagSettings())
    assert a.messages[0].content != b.messages[0].content
    monkeypatch.setattr("local_rag.context.secrets.token_hex", lambda _: "z" * 48)
    with pytest.raises(RagError, match="^rag_context_budget$"):
        build_context("RAG_" + "z" * 48, found, RagSettings())


@pytest.mark.parametrize("mutation", ["delete", "stale"])
@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
async def test_authoritative_source_excludes_deleted_and_stale(data, mutation, mode):
    retrieval, store, docs = data
    excluded = docs["lumen"].document_id
    if mutation == "delete":
        store.delete(excluded)
    else:
        with closing(sqlite3.connect(store.path)) as db:
            db.execute("UPDATE documents SET parser_version=? WHERE document_id=?", ("stale-fixture", excluded))
            db.commit()
    service, provider = rag(retrieval)
    result = await service.answer("Lumen thermal package", mode=mode)
    assert all(c.document_id != excluded for c in result.citations)
    assert all(x["document_id"] != excluded for x in entries(provider.calls[0][0]))


async def test_answering_preserves_index_and_creates_no_history(data, tmp_path, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    retrieval, store, _ = data
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    provider = DeterministicRagProvider(lambda m: grounded(m, ["12 W"], "private-answer-marker"))
    service, _ = rag(retrieval, provider)
    result = await service.answer("private-question-marker Lumen")
    after = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert before == after
    assert set(vars(service)) == {"retrieval", "provider", "settings", "renderer", "multimodal"}
    assert service.multimodal is None  # Optional collaborator, never conversation history.
    assert service.renderer is None  # No request, page image or answer history is retained.
    for private in ("private-question-marker", "private-answer-marker", "12 W", "copper plate", str(tmp_path)):
        assert private not in caplog.text and private not in repr(result)
        assert private not in json.dumps(result.safe_metadata())


async def test_full_synthetic_chain_needs_no_network_or_external_process(tmp_path, monkeypatch):
    import socket
    import subprocess
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected network or process")
    for name in ("getaddrinfo", "create_connection"):
        monkeypatch.setattr(socket, name, forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    retrieval, _, docs = corpus(tmp_path)
    service, _ = rag(retrieval)
    result = await service.answer("Lumen", document_ids=[docs["lumen"].document_id])
    assert result.status == "answered" and result.citations
