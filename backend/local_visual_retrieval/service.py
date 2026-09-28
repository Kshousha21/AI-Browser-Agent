"""Authorized source snapshots, ephemeral rendering and local page ranking."""
import asyncio
from contextlib import asynccontextmanager
from threading import Lock
from time import perf_counter
import numpy as np
from pydantic import ValidationError
from local_pdf.rendering import RenderError
from local_retrieval.models import SearchRequest, fingerprint
from local_retrieval.source import SourceCatalog
from local_retrieval.errors import RetrievalError
from .schemas import VisualError, VisualHit, VisualSettings
from .index_store import VisualIndex, vector_bytes, vector_read, row_digest
from .model import NomicEmbedder

_OPERATION = Lock()
@asynccontextmanager
async def operation():
    # Shared across services and the preparation worker's event loops.
    while not _OPERATION.acquire(blocking=False): await asyncio.sleep(.05)
    try: yield
    finally: _OPERATION.release()

class VisualRetrievalService:
    def __init__(self, renderer, *, source_path=None, index_path=None, settings=None, model=None):
        self.settings = settings or VisualSettings()
        self.renderer = renderer
        self.source = SourceCatalog(source_path)
        self.index = VisualIndex(index_path, self.settings.max_index_pages)
        if self.index.path == self.source.path or (self.index.path.exists() and self.source.path.exists() and self.index.path.samefile(self.source.path)): raise VisualError("visual_invalid_request")
        render_identity = fingerprint(renderer.settings.model_dump()) if hasattr(renderer, "settings") else "synthetic-renderer"
        try:
            import importlib.metadata
            render_identity += ":pdfium=" + importlib.metadata.version("pypdfium2")
        except importlib.metadata.PackageNotFoundError: pass
        self.model = model or NomicEmbedder(self.settings, render_identity)
        self.measurements = {}
    def _request(self, query, top_k, document_ids):
        if not self.settings.enabled: raise VisualError("visual_disabled")
        try: return SearchRequest(query=query, top_k=top_k, document_ids=document_ids)
        except ValidationError: raise VisualError("visual_invalid_request") from None
    async def status(self):
        description = await self.model.describe()
        try:
            identity, docs, rows, generation = self.index.read(description[0])
            return {"status": "ready", "documents": len(docs), "pages": len(rows), "identity": identity.model_dump(), "generation": generation}
        except VisualError as error:
            if error.code != "visual_index_unavailable": raise
            return {"status": "not_indexed", "identity": description[0].model_dump()}
    async def index_documents(self, document_ids=None, *, rebuild=False, check=None):
        self._request("index", 5, document_ids)
        checkpoint = check or (lambda: None)
        started = perf_counter()
        async with operation():
            checkpoint()
            description = await self.model.describe()
            identity = description[0]
            try:
                if rebuild:
                    documents, rows, before = {}, [], self.index.rebuild_generation()
                else:
                    _, documents, rows, before = self.index.read(identity)
            except VisualError as error:
                if error.code != "visual_index_unavailable": raise
                documents, rows, before = {}, [], None
            if rebuild: documents, rows = {}, []
            with self.source.snapshot() as source:
                requested = set(document_ids) if document_ids else set(source)
                if not requested.issubset(source): raise VisualError("visual_invalid_request")
                wanted = {key: {"fingerprint": source[key].fingerprint, "page_count": source[key].metadata.page_count,
                                "source_hash": source[key].metadata.content_hash} for key in requested}
            changed = {key: value for key, value in wanted.items() if documents.get(key) != value}
            keep = set(documents) - set(changed)
            if document_ids is None: keep &= requested
            rows = [row for row in rows if row[0] in keep]
            documents = {key: documents[key] for key in keep}
            if len(rows) + sum(doc["page_count"] for doc in changed.values()) > self.settings.max_index_pages:
                raise VisualError("visual_capacity_exceeded")
            if changed:
                async with self.model.session(description) as runtime:
                    for key, document in sorted(changed.items()):
                        for page in range(1, document["page_count"] + 1):
                            checkpoint()
                            try: rendered = await self.renderer.render(key, page, sorted(requested))
                            except RenderError: raise VisualError("visual_render_failed") from None
                            if rendered.document_id != key or rendered.page != page: raise VisualError("visual_source_changed")
                            blob = vector_bytes(await runtime.image(rendered.png), identity.dimension)
                            rows.append((key, page, rendered.rendered_page_id, blob, row_digest(key, page, rendered.rendered_page_id, blob)))
                            del rendered
                            checkpoint()
            documents.update(wanted)
            checkpoint()
            with self.source.snapshot() as current:
                if any(key not in current or current[key].fingerprint != doc["fingerprint"] for key, doc in wanted.items()):
                    raise VisualError("visual_source_changed")
                generation = self.index.write(identity, documents, rows, before)
            self.measurements = {**getattr(self.model, "measurements", {}), "index_ms": round((perf_counter()-started)*1000, 3)}
            return {"status": "indexed", "documents": len(documents), "pages": len(rows), "embedded_pages": sum(x["page_count"] for x in changed.values()), "generation": generation}
    async def search(self, query, top_k=5, document_ids=None):
        request = self._request(query, top_k, document_ids)
        self.measurements = {}
        started = perf_counter()
        async with operation():
            # Missing/corrupt indexes do not require loading or hashing model weights.
            self.index.read()
            description = await self.model.describe()
            identity, documents, rows, generation = self.index.read(description[0])
            with self.source.snapshot() as source:
                requested = set(request.document_ids) if request.document_ids else set(source)
                if not requested.issubset(source): raise VisualError("visual_invalid_request")
                eligible = {key for key in requested if key in documents and documents[key]["fingerprint"] == source[key].fingerprint
                            and documents[key]["source_hash"] == source[key].metadata.content_hash
                            and documents[key]["page_count"] == source[key].metadata.page_count}
                selected = [row for row in rows if row[0] in eligible]
                if not selected: raise VisualError("visual_index_unavailable")
                async with self.model.session(description) as runtime:
                    query_vector = vector_read(vector_bytes(await runtime.query(request.query), identity.dimension), identity.dimension)
                # Dense BiQwen vectors, not ColNomic late-interaction patch vectors.
                scored = [(float(vector_read(row[3], identity.dimension) @ query_vector), row) for row in selected]
                scored.sort(key=lambda pair: (-pair[0], pair[1][0], pair[1][1]))
                result = [VisualHit(document_id=row[0], source_hash=documents[row[0]]["source_hash"],
                    prepared_identity=documents[row[0]]["fingerprint"], page=row[1], rendered_page_id=row[2], rank=rank, score=score)
                    for rank, (score, row) in enumerate(scored[:top_k], 1)]
            self.measurements = {**getattr(self.model, "measurements", {}), "query_ms": round((perf_counter()-started)*1000, 3)}
            return result
