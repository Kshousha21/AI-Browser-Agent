"""Conditional multimodal contract; mock semantic outputs, never paid quality claims."""
import json
import pytest
from browser.provider_settings import ProviderSettings
from local_pdf import DocumentStore, PdfIngestionService
from local_pdf.rendering import LocalPageRenderer
from local_rag import ProviderBridge, RagService, RagSettings, RagError
from local_rag.context import build_context
from local_rag.citations import validate_answer
from local_rag.models import INSUFFICIENT
from local_rag.visual import VisualSource, select_pages
from local_retrieval import RetrievalService
from .retrieval_fixtures import TestEmbedder
from .rag_fixtures import DeterministicRagProvider, grounded
from .visual_pdf_fixtures import research_pdf


def setup(tmp_path, kind="chart", *, supports_images=True, remote=False, consent=False, reply=None):
    path = tmp_path / (kind + ".pdf")
    path.write_bytes(research_pdf(kind))
    store = DocumentStore(tmp_path / "documents.sqlite3")
    doc = PdfIngestionService(store).ingest_pdf(path).document
    retrieval = RetrievalService(TestEmbedder(), source_path=store.path, index_path=tmp_path / "retrieval.sqlite3")
    retrieval.index_all()
    provider = DeterministicRagProvider(reply or visual_reply)
    settings = ProviderSettings(name="ollama", model="synthetic-configured", supports_images=supports_images,
        base_url="https://provider.invalid" if remote else "http://127.0.0.1:11434", auth_mode="none")
    bridge = ProviderBridge(settings, allow_remote_context=consent, factory=lambda _: provider)
    renderer = LocalPageRenderer(store, lambda identifier: path if identifier == doc.document_id else None)
    return RagService(retrieval, bridge, renderer=renderer), provider, doc, renderer


def visual_reply(messages):
    block = next(block for block in messages[-1].content if block["type"] == "text")
    source = json.loads(block["text"])
    return json.dumps({"status": "answered", "answer": "The synthetic chart has a readable 6 W label.",
        "citations": [{"source_id": source["source_id"], "evidence_type": "visual"}]})


@pytest.mark.parametrize("question,page", [("Explain the chart on page 4.", 4),
    ("Explain the architecture diagram on page 2.", 2), ("Read the table on page 3.", 3)])
async def test_explicit_page_visual_contract_and_zero_tools(tmp_path, question, page):
    service, provider, doc, _ = setup(tmp_path)
    stages = []
    result = await service.answer(question, document_ids=[doc.document_id],
        progress=lambda stage, **metadata: stages.append((stage, metadata)))
    assert result.status == "answered" and result.measurements["visual_pages"] == 1
    assert result.citations[0].document_id == doc.document_id
    assert result.citations[0].page_start == result.citations[0].page_end == page
    assert result.citations[0].evidence_type == "visual" and result.citations[0].chunk_id is None
    assert len(provider.calls) == 1 and provider.calls[0][1] == [] and provider.closed
    assert any(stage == "rendering" and metadata["page"] == page for stage, metadata in stages)


async def test_pure_text_path_avoids_render_and_preserves_exact_spans(tmp_path, monkeypatch):
    service, provider, doc, renderer = setup(tmp_path, reply=lambda m: grounded(m, ["ordinary searchable text"]))
    def forbidden(*args):
        pytest.fail("pure text must not render")
    monkeypatch.setattr(renderer, "render", forbidden)
    result = await service.answer("ordinary searchable text introduction", mode="lexical", top_k=1, document_ids=[doc.document_id])
    assert result.measurements["visual_pages"] == 0 and result.citations[0].evidence_type == "text"


async def test_image_only_page_uses_visual_not_fabricated_text(tmp_path):
    service, provider, doc, _ = setup(tmp_path, "image_only")
    result = await service.answer("What does this selected report show?", document_ids=[doc.document_id])
    assert result.retrieved_count == 0 and result.citations[0].evidence_type == "visual"
    assert doc.extracted_chars == 0 and result.measurements["visual_pages"] == 1


async def test_illegible_figure_abstains(tmp_path):
    reply = json.dumps({"status": "insufficient_evidence", "answer": "Unreadable labels; do not invent numbers.", "citations": []})
    service, _, doc, _ = setup(tmp_path, "illegible", reply=reply)
    result = await service.answer("Explain figure on page 4", document_ids=[doc.document_id])
    assert result.answer == INSUFFICIENT and not result.citations


@pytest.mark.parametrize("supports,consent", [(True, False), (False, False), (True, True)])
async def test_remote_consent_precedes_factory_text_and_images(tmp_path, supports, consent):
    service, provider, doc, _ = setup(tmp_path, supports_images=supports, remote=True, consent=consent)
    factories = []
    service.provider._factory = lambda settings: factories.append(settings.model) or provider
    if not consent:
        with pytest.raises(RagError, match="rag_remote_context_denied"):
            await service.answer("Explain chart page 4", document_ids=[doc.document_id])
        assert not factories and not provider.calls
    else:
        result = await service.answer("Explain chart page 4", document_ids=[doc.document_id])
        assert factories == ["synthetic-configured"] and result.measurements["visual_pages"] == 1
        assert len([block for block in provider.calls[0][0][-1].content if block["type"] == "image"]) == 1


async def test_nonvision_keeps_configured_model_and_reports_limitation(tmp_path, monkeypatch):
    service, provider, doc, renderer = setup(tmp_path, supports_images=False, reply=grounded)
    monkeypatch.setattr(renderer, "render", lambda *args: pytest.fail("nonvision must avoid needless render"))
    result = await service.answer("Explain chart page 4", document_ids=[doc.document_id])
    assert result.status == "answered" and result.notices and result.measurements["visual_pages"] == 0
    assert len(provider.calls) == 1 and "configured model" in result.notices[0]


@pytest.mark.parametrize("pages", [[{"document_id": "a"*64, "page": 1}], [{"document_id": "CURRENT", "page": 5}],
    [{"document_id": "CURRENT", "page": True}], [{"document_id": "CURRENT", "page": 1, "filename": "wrong.pdf"}],
    [{"document_id": "CURRENT", "page": p} for p in range(1, 5)]])
async def test_visual_page_cannot_authorize_other_document_or_outofbounds(tmp_path, pages):
    service, provider, doc, _ = setup(tmp_path)
    pages = [{key: doc.document_id if value == "CURRENT" else value for key, value in item.items()} for item in pages]
    with pytest.raises(RagError, match="rag_invalid_page"):
        await service.answer("figure", document_ids=[doc.document_id], explicit_pages=pages)
    assert not provider.calls


async def test_duplicate_pages_render_once_and_hard_limit(tmp_path):
    service, _, doc, _ = setup(tmp_path)
    result = await service.answer("figure", document_ids=[doc.document_id],
        explicit_pages=[{"document_id": doc.document_id, "page": 4}] * 2)
    assert result.measurements["visual_pages"] == 1
    assert RagSettings().max_visual_pages == 3
    assert RagSettings(max_visual_pages=5).max_visual_pages == 5
    with pytest.raises(ValueError):
        RagSettings(max_visual_pages=6)


@pytest.mark.parametrize("citation", [
    {"source_id": "S2", "evidence": "invented"},
    {"source_id": "S1", "evidence_type": "visual"},
    {"source_id": "S2", "evidence_type": "visual", "page": 99},
    {"source_id": "S2", "evidence_type": "visual", "document_id": "a"*64},
    {"source_id": "S3", "evidence_type": "visual"},
])
def test_visual_citation_requires_request_local_source_and_no_model_provenance(tmp_path, citation):
    service, _, doc, _ = setup(tmp_path)
    context = build_context("CPO", service.retrieval.search("CPO", 1, [doc.document_id]), RagSettings())
    sources = {**context.sources, "S2": VisualSource(doc.document_id, 4, "b"*64)}
    raw = json.dumps({"status": "answered", "answer": "Synthetic claim.", "citations": [citation]})
    with pytest.raises(RagError):
        validate_answer(raw, sources, RagSettings())


def test_conflicting_text_and_visual_support_keeps_both_provenances(tmp_path):
    service, _, doc, _ = setup(tmp_path, "conflict")
    context = build_context("CPO", service.retrieval.search("CPO", 1, [doc.document_id]), RagSettings())
    sources = {**context.sources, "S2": VisualSource(doc.document_id, 4, "b"*64)}
    raw = json.dumps({"status": "answered", "answer": "The sources conflict: text says 10 W while chart label says 6 W.",
        "citations": [{"source_id": "S1", "evidence": "10 W"}, {"source_id": "S2", "evidence_type": "visual"}]})
    status, text, citations = validate_answer(raw, sources, RagSettings())
    assert status == "answered" and "conflict" in text
    assert [citation.evidence_type for citation in citations] == ["both", "visual"]
    assert all(citation.page_start == 4 for citation in citations)


async def test_schema_capable_local_provider_gets_visual_only_citations(tmp_path):
    from local_rag.models import VisualProviderAnswer
    service, provider, doc, _ = setup(tmp_path, "image_only")
    provider.prefers_response_schema = True
    original = provider.generate
    received = []
    async def constrained(messages, tools=None, response_schema=None):
        received.append(response_schema)
        return await original(messages, tools, None)
    provider.generate = constrained
    result = await service.answer("Read this chart on page 1", document_ids=[doc.document_id])
    assert received == [VisualProviderAnswer]
    assert result.citations[0].evidence_type == "visual"
