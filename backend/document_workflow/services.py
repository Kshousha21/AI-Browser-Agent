"""Compose existing ingestion/index/RAG APIs without changing their semantics."""
from pydantic import ValidationError
from local_pdf import PdfIngestionService, PdfIngestError
from local_retrieval import RetrievalService, RetrievalError
from local_retrieval.local_model import OnnxEmbedder
from local_rag import ProviderBridge, RagService
from .models import CODES, DocumentError, DownloadReceipt
from .storage import AcquisitionStorage


class DocumentServices:
    def __init__(self, provider_settings, *, ingestion=None, retrieval=None, storage=None, rag_factory=None,
                 visual_settings=None, multimodal_settings=None, visual_retrieval=None):
        self.ingestion = ingestion or PdfIngestionService()
        self.retrieval = retrieval or RetrievalService(OnnxEmbedder())
        self.storage = storage or AcquisitionStorage()
        self.provider_settings = provider_settings
        self.rag_factory = rag_factory
        from local_pdf.rendering import LocalPageRenderer
        from local_visual_retrieval import VisualRetrievalService
        from local_visual_retrieval.fusion import MultimodalRetrieval
        self.renderer = LocalPageRenderer(self.ingestion.store, self.storage.prepared_path)
        self.visual_retrieval = visual_retrieval or VisualRetrievalService(self.renderer,
            source_path=self.ingestion.store.path,
            index_path=self.retrieval.index.path.with_name("visual_retrieval.sqlite3"), settings=visual_settings)
        self.multimodal = MultimodalRetrieval(self.retrieval, self.visual_retrieval, multimodal_settings)
        self.visual_index_status = "not_attempted"

    def prepare(self, raw_receipt, acquisition_id, selected, status, *, check=None):
        # Native parsing/index transactions are not forcibly interrupted. An
        # admitted file operation can revoke all subsequent stages when the
        # current bounded native call returns; legacy callers remain unchanged.
        checkpoint = check if check is not None else lambda: None
        checkpoint()
        try:
            receipt = DownloadReceipt.model_validate(raw_receipt)
        except ValidationError:
            raise DocumentError("download_incomplete") from None
        with self.storage.stage(receipt, acquisition_id) as path:
            status("validating")
            try:
                status("ingesting")
                ingested = self.ingestion.ingest_pdf(path)
            except PdfIngestError as error:
                code = error.code if error.code in CODES else "ingestion_failed"
                raise DocumentError(code) from None
            checkpoint()
            identifier = ingested.document.document_id
            self.storage.retain_pdf(path, identifier)
            checkpoint()
            self.storage.record(acquisition_id, identifier, selected.source_fingerprint, ingested.status, "pending")
            self.visual_index_status = "not_attempted"
            text_error = None
            try:
                status("indexing")
                self.retrieval.index_document(identifier)
            except RetrievalError as error:
                text_error = error
            checkpoint()
            if self.multimodal.settings.enabled:
                import asyncio
                from local_visual_retrieval import VisualError
                try:
                    async def index_with_cancellation():
                        task = asyncio.create_task(self.visual_retrieval.index_documents([identifier], check=checkpoint))
                        try:
                            while not task.done():
                                checkpoint()
                                await asyncio.wait({task}, timeout=.1)
                            return await task
                        finally:
                            if not task.done():
                                task.cancel()
                                await asyncio.gather(task, return_exceptions=True)
                    asyncio.run(index_with_cancellation())
                    self.visual_index_status = "ready"
                except (VisualError, RetrievalError) as error:
                    self.visual_index_status = error.code if isinstance(error, VisualError) else "visual_source_changed"
            if text_error is not None and self.visual_index_status != "ready":
                self.storage.record(acquisition_id, identifier, selected.source_fingerprint, ingested.status, "failed")
                code = "index_incompatible" if text_error.code == "index_model_mismatch" else "index_failed"
                raise DocumentError(code) from None
            checkpoint()
            self.storage.record(acquisition_id, identifier, selected.source_fingerprint, ingested.status, "ready")
            return {"document_id": identifier, "acquisition_id": acquisition_id,
                    "title": selected.metadata.get("attachment_filename") or "Selected PDF",
                    "source_fingerprint": selected.source_fingerprint, "result_ref": selected.result_ref,
                    "ingestion_status": ingested.status, "index_status": "ready"}

    async def answer(self, question, document_ids, allow_remote_context=False):
        bridge = ProviderBridge(self.provider_settings, allow_remote_context=allow_remote_context)
        service = (self.rag_factory(self.retrieval, bridge) if self.rag_factory else RagService(self.retrieval, bridge))
        service.renderer = self.renderer
        service.multimodal = self.multimodal
        return await service.answer(question, top_k=self.multimodal.settings.fused_top_k, document_ids=sorted(set(document_ids)))

    async def research_answer(self, question, document_ids, allow_remote_context=False,
                              max_visual_pages=3, progress=None):
        from local_rag.models import RagSettings
        bridge = ProviderBridge(self.provider_settings, allow_remote_context=allow_remote_context)
        service = (self.rag_factory(self.retrieval, bridge) if self.rag_factory else
                   RagService(self.retrieval, bridge))
        service.settings = RagSettings(max_visual_pages=max_visual_pages)
        service.renderer = self.renderer
        service.multimodal = self.multimodal
        return await service.answer(question, top_k=self.multimodal.settings.fused_top_k, document_ids=sorted(set(document_ids)),
                                    visual="auto", progress=progress)
