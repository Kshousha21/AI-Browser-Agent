"""Page-level rank fusion. Raw text and visual similarities never meet."""
import asyncio
from dataclasses import dataclass, field
from time import perf_counter
from local_retrieval import RetrievalError
from local_retrieval.models import SearchHit, SearchResult
from local_retrieval.ranking import reciprocal_rank_scores
from .schemas import MultimodalSettings, VisualError

@dataclass
class PageEvidence:
    document_id: str
    page: int
    text_chunks: list = field(default_factory=list, repr=False)
    has_visual_evidence: bool = False
    retrieval_sources: list[str] = field(default_factory=list)
    fusion_score: float = 0.
    @property
    def evidence_type(self):
        return "text_and_visual" if self.text_chunks and self.has_visual_evidence else "visual_only" if self.has_visual_evidence else "text_only"
    def safe_metadata(self):
        return {"document_id": self.document_id, "page": self.page, "retrieval_sources": self.retrieval_sources,
                "evidence_type": self.evidence_type, "fusion_score": self.fusion_score, "text_chunks": len(self.text_chunks)}

@dataclass
class FusedEvidence:
    pages: list[PageEvidence]
    text_result: SearchResult
    status: dict[str, str]
    measurements: dict[str, float]
    visual_results: list = field(default_factory=list)
    text_results_count: int = 0

def empty_result(top_k=5, mode="hybrid"):
    return SearchResult(mode=mode, top_k=top_k, returned_count=0, index_generation="0"*64,
                        candidate_count=0, excluded_stale_documents=0, unindexed_documents=0, results=[])

class MultimodalRetrieval:
    def __init__(self, text, visual, settings=None):
        self.text, self.visual = text, visual
        self.settings = settings or MultimodalSettings()
    async def search(self, question, document_ids, top_k=None, mode="hybrid"):
        # Validate before either backend; an empty list is never an all-documents query.
        from local_retrieval.models import SearchRequest
        if not document_ids: raise VisualError("visual_invalid_request")
        request = SearchRequest(query=question, document_ids=document_ids, top_k=top_k or self.settings.fused_top_k, mode=mode)
        with self.text.source.snapshot() as source:
            if not set(request.document_ids).issubset(source): raise VisualError("visual_invalid_request")
        start = perf_counter()
        status = {"text": "ok", "visual": "ok"}
        try:
            text = await asyncio.to_thread(self.text.search, request.query, self.settings.text_top_k, request.document_ids, request.mode)
        except RetrievalError:
            status["text"] = "text_retrieval_unavailable"
            text = empty_result(self.settings.text_top_k, request.mode)
        text_ms = (perf_counter() - start) * 1000
        visual_start = perf_counter()
        try:
            visual = await self.visual.search(request.query, self.settings.visual_top_k, request.document_ids)
        except (VisualError, RetrievalError) as error:
            status["visual"] = error.code if isinstance(error, VisualError) else "visual_source_changed"
            visual = []
        visual_ms = (perf_counter() - visual_start) * 1000
        fusion_start = perf_counter()
        pages = {}
        text_ranking, visual_ranking = [], []
        for hit in text.results:
            if hit.document_id not in request.document_ids: raise VisualError("visual_source_changed")
            key = (hit.document_id, hit.page_start)
            item = pages.setdefault(key, PageEvidence(*key))
            item.text_chunks.append(hit)
            if key not in text_ranking:
                text_ranking.append(key)
                item.retrieval_sources.append("text")
        for hit in visual:
            if hit.document_id not in request.document_ids: raise VisualError("visual_source_changed")
            key = (hit.document_id, hit.page)
            item = pages.setdefault(key, PageEvidence(*key))
            item.has_visual_evidence = True
            if key not in visual_ranking:
                visual_ranking.append(key)
                item.retrieval_sources.append("visual")
        scores = reciprocal_rank_scores(text_ranking, visual_ranking, rrf_k=self.text.settings.rrf_k)
        ordered = sorted(pages, key=lambda key: (-scores[key], key))[:min(request.top_k, self.settings.fused_top_k)]
        selected = [pages[key] for key in ordered]
        # Pages found only visually can supply their existing exact authoritative chunks
        # even with a text-only answer model or an unavailable semantic index.
        if any(item.has_visual_evidence and not item.text_chunks for item in selected):
            with self.text.source.snapshot() as source:
                for item in selected:
                    if item.has_visual_evidence and not item.text_chunks:
                        if item.document_id not in source: raise VisualError("visual_source_changed")
                        item.text_chunks = [SearchHit(**chunk.model_dump(), rank=1) for chunk in source[item.document_id].chunks
                                            if chunk.page_start == item.page][:self.settings.text_top_k]
        chunks = []
        for item in selected:
            item.fusion_score = scores[(item.document_id, item.page)]
            for hit in item.text_chunks:
                if len(chunks) < self.settings.text_top_k:
                    chunks.append(hit.model_copy(update={"rank": len(chunks) + 1}))
        text_results_count = text.returned_count
        text = text.model_copy(update={"results": chunks, "returned_count": len(chunks), "top_k": self.settings.text_top_k})
        return FusedEvidence(selected, text, status, {"text_retrieval_ms": round(text_ms, 3),
            "visual_retrieval_ms": round(visual_ms, 3), "fusion_ms": round((perf_counter()-fusion_start)*1000, 3),
            "end_to_end_retrieval_ms": round((perf_counter()-start)*1000, 3)}, visual_results=visual, text_results_count=text_results_count)
