"""Synthetic-only indexing, fusion, offline, scope and cancellation contracts."""
import asyncio
from contextlib import asynccontextmanager, closing
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import pytest
from local_retrieval.errors import RetrievalError
from local_retrieval.ranking import reciprocal_rank_scores
from local_visual_retrieval.model import discover, local_snapshot, NomicEmbedder
from local_visual_retrieval.schemas import VisualSettings, VisualError, MultimodalSettings
from local_visual_retrieval.index_store import vector_bytes, vector_read
from local_rag import RagService, RagSettings, ProviderBridge, RagError
from browser.provider_settings import ProviderSettings
from .rag_fixtures import DeterministicRagProvider
from .visual_retrieval_support import setup

@pytest.mark.parametrize("missing", ["adapter", "base", "weights", "tokenizer"])
def test_local_detection_never_downloads(tmp_path, monkeypatch, missing):
    import socket
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("network forbidden"))
    adapter, base = tmp_path / "adapter", tmp_path / "base"
    adapter.mkdir(); base.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path":"Qwen/Qwen2.5-VL-3B-Instruct"}))
    (base / "config.json").write_text('{"hidden_size":4}')
    for directory, name in [(adapter,"adapter_model.safetensors"),(base,"model.safetensors"),(adapter,"tokenizer.json"),(adapter,"preprocessor_config.json")]:
        (directory / name).write_text("synthetic")
    paths = {"adapter":adapter / "adapter_config.json", "base":base / "config.json", "weights":base / "model.safetensors", "tokenizer":adapter / "tokenizer.json"}
    settings = VisualSettings(model_directory=str(adapter), base_model_directory=str(base))
    assert discover(settings)[2] == 4
    paths[missing].unlink()
    with pytest.raises(VisualError, match="visual_embedding_model_unavailable"): discover(settings)

def test_missing_cache_is_read_only(tmp_path, monkeypatch):
    from huggingface_hub import constants
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    with pytest.raises(VisualError, match="visual_embedding_model_unavailable"): local_snapshot("nomic-ai/nomic-embed-multimodal-3b")
    assert not list(tmp_path.iterdir())

async def test_index_render_search_and_reuse(tmp_path):
    fused, doc, model, _ = setup(tmp_path)
    before = fused.text.index.path.read_bytes()
    result = await fused.visual.index_documents([doc.document_id])
    assert result["embedded_pages"] == doc.page_count == model.images == 4
    assert model.opened == model.closed == 1
    assert not list(tmp_path.glob("*.png"))
    result = await fused.visual.index_documents([doc.document_id])
    assert result["embedded_pages"] == 0 and model.images == 4
    hits = await fused.visual.search("temperature", 2, [doc.document_id])
    assert [hit.page for hit in hits] == [4, 1]
    assert hits[0].source_hash == doc.content_hash
    assert hits[0].prepared_identity == fused.visual.index.read()[1][doc.document_id]["fingerprint"]
    assert model.opened == model.closed == 2
    assert before == fused.text.index.path.read_bytes()

@pytest.mark.parametrize("field,value", [("model","new-model"),("revision","2"),("processor","new"),("runtime","new"),("artifact_fingerprint","b"*64),("renderer","new")])
async def test_incompatible_requires_explicit_rebuild(tmp_path, field, value):
    fused, doc, model, _ = setup(tmp_path)
    await fused.visual.index_documents([doc.document_id])
    before = fused.visual.index.path.read_bytes()
    model.identity = model.identity.model_copy(update={field:value})
    with pytest.raises(VisualError, match="visual_index_incompatible"):
        await fused.visual.search("temperature", document_ids=[doc.document_id])
    with pytest.raises(VisualError, match="visual_index_incompatible"):
        await fused.visual.index_documents([doc.document_id])
    assert before == fused.visual.index.path.read_bytes()
    await fused.visual.index_documents(rebuild=True)
    assert len(await fused.visual.search("temperature", document_ids=[doc.document_id])) == 4

@pytest.mark.parametrize("mutation", ["DELETE FROM pages WHERE page=2", "UPDATE pages SET vector=X'00000000' WHERE page=1", "UPDATE pages SET rendered_id='invented' WHERE page=1", "UPDATE metadata SET documents='{}'", "UPDATE metadata SET generation='corrupt'"])
async def test_corrupt_index_closed(tmp_path, mutation):
    fused, doc, _, _ = setup(tmp_path)
    await fused.visual.index_documents([doc.document_id])
    with closing(sqlite3.connect(fused.visual.index.path)) as c, c: c.execute(mutation)
    with pytest.raises(VisualError, match="visual_index_corrupt"):
        await fused.visual.search("question", document_ids=[doc.document_id])

async def test_unknown_document_and_empty_scope_rejected(tmp_path):
    fused, doc, _, _ = setup(tmp_path)
    await fused.visual.index_documents([doc.document_id])
    for ids in ([], ["b"*64]):
        with pytest.raises(VisualError): await fused.visual.search("question", document_ids=ids)
        with pytest.raises(VisualError): await fused.search("question", ids)

async def test_only_selected_document_ranked(tmp_path):
    from local_pdf import PdfIngestionService
    from .pdf_fixtures import pdf_bytes
    fused, doc, _, _ = setup(tmp_path)
    other = tmp_path / "other.pdf"; other.write_bytes(pdf_bytes(["Unrelated temperature report."]))
    PdfIngestionService(fused.visual.renderer.store).ingest_pdf(other)
    await fused.visual.index_documents([doc.document_id])
    result = await fused.search("temperature", [doc.document_id])
    assert all(p.document_id == doc.document_id for p in result.pages)

async def test_render_identity_change_cannot_commit(tmp_path):
    fused, doc, _, path = setup(tmp_path)
    path.write_bytes(b"not original")
    with pytest.raises(VisualError, match="visual_render_failed"):
        await fused.visual.index_documents([doc.document_id])
    assert not fused.visual.index.path.exists()

async def test_index_cancellation_releases_runtime_and_no_commit(tmp_path):
    fused, doc, model, _ = setup(tmp_path)
    entered = asyncio.Event()
    async def image(png):
        entered.set(); await asyncio.Event().wait()
    model.image = image
    task = asyncio.create_task(fused.visual.index_documents([doc.document_id]))
    await entered.wait(); task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert model.opened == model.closed == 1
    assert not fused.visual.index.path.exists()

async def test_query_cancellation_releases_runtime(tmp_path):
    fused, doc, model, _ = setup(tmp_path)
    await fused.visual.index_documents([doc.document_id])
    entered = asyncio.Event()
    async def query(q): entered.set(); await asyncio.Event().wait()
    model.query = query
    task = asyncio.create_task(fused.search("question", [doc.document_id]))
    await entered.wait(); task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert model.opened == model.closed

@pytest.mark.parametrize("question", ["What does the report say about thermal resistance?", "What temperature results are reported?", "Summarize the report", "What does the graph show?", "What is the system architecture?", "What differences are visible between the two approaches?", "What does the table report about performance?"])
async def test_every_question_runs_both_and_deduplicates_pages(tmp_path, question):
    fused, doc, model, _ = setup(tmp_path)
    await fused.visual.index_documents([doc.document_id])
    text_calls = len(fused.text.embedder.calls)
    result = await fused.search(question, [doc.document_id])
    assert model.queries == 1 and len(fused.text.embedder.calls) > text_calls
    assert len(result.pages) == len({(p.document_id,p.page) for p in result.pages}) <= 5
    assert any(p.evidence_type == "text_and_visual" for p in result.pages)
    assert all(p.fusion_score > 0 for p in result.pages)
    assert all("text" not in p.safe_metadata() and "text_chunks" in p.safe_metadata() for p in result.pages)

def test_rank_fusion_is_exact_and_duplicate_page_cannot_boost():
    score = reciprocal_rank_scores([("A",6),("A",6),("A",11)],[("A",7),("A",6)],rrf_k=60)
    assert score[("A",6)] == 1/61 + 1/62
    assert score[("A",11)] == 1/62
    assert score[("A",7)] == 1/61

async def test_visual_failure_continues_text(tmp_path):
    fused, doc, model, _ = setup(tmp_path)
    result = await fused.search("temperature", [doc.document_id])
    assert result.status == {"text":"ok","visual":"visual_index_unavailable"}
    assert result.text_result.results and model.queries == 0

async def test_text_failure_uses_visual_pages_and_authoritative_text(tmp_path):
    fused, doc, _, _ = setup(tmp_path)
    await fused.visual.index_documents([doc.document_id])
    def fail(*args): raise RetrievalError("index_unavailable")
    fused.text.search = fail
    result = await fused.search("temperature", [doc.document_id])
    assert result.status == {"text":"text_retrieval_unavailable","visual":"ok"}
    assert result.pages[0].page == 4 and result.text_result.results

async def test_image_only_pdf_visual_evidence(tmp_path):
    fused, doc, _, _ = setup(tmp_path, "image_only")
    await fused.visual.index_documents([doc.document_id])
    result = await fused.search("What are the temperature results?", [doc.document_id])
    assert result.pages[0].evidence_type == "visual_only" and not result.text_result.results

async def test_limits_and_raw_scores_do_not_affect_fusion(tmp_path):
    fused, doc, _, _ = setup(tmp_path)
    await fused.visual.index_documents([doc.document_id])
    fused.settings = MultimodalSettings(text_top_k=2, visual_top_k=3, fused_top_k=2, max_visual_pages=1)
    result = await fused.search("temperature", [doc.document_id])
    assert len(result.pages) == 2 and len(result.text_result.results) <= 2
    search = fused.visual.search
    async def exaggerated(*a): return [h.model_copy(update={"score":h.score*1e9}) for h in await search(*a)]
    fused.visual.search = exaggerated
    same = await fused.search("temperature", [doc.document_id])
    assert [p.safe_metadata() for p in result.pages] == [p.safe_metadata() for p in same.pages]

async def test_missing_model_no_cloud_fallback(tmp_path):
    fused, doc, _, _ = setup(tmp_path)
    fused.visual.model = NomicEmbedder(VisualSettings(model_directory=str(tmp_path/"absent")))
    result = await fused.search("temperature", [doc.document_id])
    assert result.status["visual"] == "visual_index_unavailable" and result.text_result.results
    with pytest.raises(VisualError, match="visual_embedding_model_unavailable"):
        await fused.visual.index_documents([doc.document_id])

async def test_real_runtime_process_offline_and_reaped():
    model = NomicEmbedder()
    async with model.worker() as worker:
        process = worker.process
        # This deliberately requests a nonexistent local model; it must never call a hub.
        with pytest.raises(VisualError): await worker.call({"op":"load", "adapter":"missing-local-adapter", "base":"missing-local-base", "device":"cpu"})
    assert process.returncode is not None

@pytest.mark.parametrize("bad", [[float('nan')]*4,[0.]*4,[1.,2.]])
def test_invalid_vectors_rejected(bad):
    with pytest.raises(VisualError): vector_bytes(bad,4)

async def test_rag_bounded_images_default_question_and_current_provider(tmp_path):
    fused, doc, _, _ = setup(tmp_path)
    await fused.visual.index_documents([doc.document_id])
    fused.settings = MultimodalSettings(max_visual_pages=2)
    def reply(messages):
        blocks = messages[-1].content
        source = json.loads(blocks[0]["text"])["source_id"]
        return json.dumps({"status":"answered", "answer":"Synthetic visual answer.", "citations":[{"source_id":source,"evidence_type":"visual"}]})
    provider = DeterministicRagProvider(reply)
    bridge = ProviderBridge(ProviderSettings(name="ollama", model="synthetic", auth_mode="none", supports_images=True), factory=lambda _:provider)
    service = RagService(fused.text, bridge, renderer=fused.visual.renderer, multimodal=fused)
    result = await service.answer("What temperature results are reported?",document_ids=[doc.document_id])
    assert result.status == "answered" and result.measurements["visual_pages"] == 2
    assert result.citations[0].document_id == doc.document_id and result.citations[0].evidence_type == "visual"
    assert result.retrieval_status["visual_interpretation"] == "performed"

async def test_no_image_support_still_retrieves_visual_and_keeps_consent(tmp_path):
    fused, doc, model, _ = setup(tmp_path)
    await fused.visual.index_documents([doc.document_id])
    provider = DeterministicRagProvider()
    config = ProviderSettings(name="ollama", model="synthetic", auth_mode="none", supports_images=False, base_url="https://remote.invalid")
    service = RagService(fused.text, ProviderBridge(config,factory=lambda _:provider), renderer=fused.visual.renderer, multimodal=fused)
    with pytest.raises(RagError, match="rag_remote_context_denied"):
        await service.answer("What does the report say?",document_ids=[doc.document_id])
    assert model.queries == 1 and not provider.calls
    service.provider = ProviderBridge(config,allow_remote_context=True,factory=lambda _:provider)
    result = await service.answer("What does the report say?",document_ids=[doc.document_id])
    assert model.queries == 2 and result.measurements["visual_pages"] == 0
    assert result.retrieval_status["visual_interpretation"] == "not_performed"


async def test_explicit_rebuild_repairs_damaged_rows(tmp_path):
    fused, doc, _, _ = setup(tmp_path)
    await fused.visual.index_documents([doc.document_id])
    with closing(sqlite3.connect(fused.visual.index.path)) as c, c: c.execute("DELETE FROM pages WHERE page=2")
    await fused.visual.index_documents(rebuild=True)
    assert len(await fused.visual.search("question", document_ids=[doc.document_id])) == 4

async def test_prepared_document_answers_automatically_use_both_paths(tmp_path):
    from .public_document_support import services
    from .visual_retrieval_support import SyntheticVisualModel
    service, provider = services(tmp_path / "workflow")
    service.visual_retrieval.model = SyntheticVisualModel()
    fused, doc, model, path = setup(tmp_path)
    await fused.visual.index_documents([doc.document_id])
    service.retrieval = fused.text
    service.renderer = fused.visual.renderer
    service.multimodal = fused
    for answer in (service.answer, service.research_answer):
        await answer("What does the report say?", [doc.document_id])
    assert model.queries == 2


async def test_existing_bounded_rendering_survives_missing_visual_index(tmp_path):
    from .test_rag_visual import visual_reply
    fused, doc, model, _ = setup(tmp_path)
    provider = DeterministicRagProvider(visual_reply)
    bridge = ProviderBridge(ProviderSettings(name="ollama", model="synthetic", auth_mode="none", supports_images=True), factory=lambda _:provider)
    service = RagService(fused.text, bridge, renderer=fused.visual.renderer, multimodal=fused)
    result = await service.answer("Read the chart on page 4",document_ids=[doc.document_id])
    assert result.status == "answered" and result.measurements["visual_pages"] == 1
    assert result.retrieval_status["visual"] == "visual_index_unavailable"
    assert model.queries == 0


@pytest.mark.parametrize("field,value,expected", [("VISUAL_RETRIEVAL__ENABLED","false",False), ("MULTIMODAL_RETRIEVAL__TEXT_TOP_K","3",3), ("VISUAL_RETRIEVAL__TIMEOUT_SECONDS","120",120.0)])
def test_configuration_environment_overrides(monkeypatch, field, value, expected):
    from browser.config import Settings
    monkeypatch.setenv("BROWSER_"+field,value)
    config = Settings(_env_file=None, provider={"active":"mock"}, providers={"mock":{"type":"mock","model":"mock"}})
    section, leaf = field.lower().split("__")
    assert getattr(getattr(config,section),leaf) == expected

@pytest.mark.parametrize("values", [{"always_use_text":False}, {"always_use_visual":False}, {"fusion":"raw_similarity"}, {"max_visual_pages":6}])
def test_invalid_multimodal_modes_rejected(values):
    with pytest.raises(ValueError): MultimodalSettings(**values)

async def test_no_socket_during_visual_index_or_search(tmp_path, monkeypatch):
    import socket
    fused, doc, _, _ = setup(tmp_path)
    def forbidden(*args,**kwargs): pytest.fail("visual retrieval attempted network access")
    monkeypatch.setattr(socket.socket,"connect",forbidden)
    await fused.visual.index_documents([doc.document_id])
    assert await fused.visual.search("temperature", document_ids=[doc.document_id])


async def test_representation_format_change_requires_rebuild(tmp_path):
    fused, doc, model, _ = setup(tmp_path)
    await fused.visual.index_documents([doc.document_id])
    model.identity = model.identity.model_copy(update={"representation":"future-format"})
    with pytest.raises(VisualError,match="visual_index_incompatible"):
        await fused.visual.search("question",document_ids=[doc.document_id])

async def test_stale_source_excluded_without_rebuild(tmp_path):
    fused, doc, _, _ = setup(tmp_path)
    await fused.visual.index_documents([doc.document_id])
    # Ingested chunks are authoritative; an index claim cannot override changed content.
    with closing(sqlite3.connect(fused.text.source.path)) as c, c:
        c.execute("UPDATE chunks SET text=replace(text,'Synthetic','Different') WHERE document_id=?", (doc.document_id,))
    with pytest.raises(VisualError):
        await fused.visual.search("question",document_ids=[doc.document_id])


async def test_explicit_page_and_requested_page_limit_preserved(tmp_path):
    fused, doc, _, _ = setup(tmp_path)
    await fused.visual.index_documents([doc.document_id])
    seen = []
    def reply(messages):
        labels=[json.loads(b["text"]) for b in messages[-1].content if b["type"]=="text"]
        seen.extend(label["page"] for label in labels)
        return json.dumps({"status":"answered","answer":"Synthetic page.","citations":[{"source_id":labels[0]["source_id"],"evidence_type":"visual"}]})
    provider=DeterministicRagProvider(reply)
    bridge=ProviderBridge(ProviderSettings(name="ollama",model="synthetic",auth_mode="none",supports_images=True),factory=lambda _:provider)
    service=RagService(fused.text,bridge,renderer=fused.visual.renderer,multimodal=fused)
    result=await service.answer("Explain this report",top_k=1,document_ids=[doc.document_id],
                               explicit_pages=[{"document_id":doc.document_id,"page":3}])
    assert seen==[3] and result.citations[0].page_start==3
    with pytest.raises(RagError,match="rag_invalid_page"):
        await service.answer("Explain",document_ids=[doc.document_id],explicit_pages=[{"document_id":"b"*64,"page":1}])
