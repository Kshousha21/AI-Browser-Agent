"""Explicit local indexing and ranked-chunk retrieval; no model-facing integration."""
import logging
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from .errors import RetrievalError
from .index_store import IndexStore
from .models import RetrievalSettings, SearchRequest, SearchResult
from .ranking import fuse, lexical, semantic
from .source import SourceCatalog, chunk_fingerprint
from .vectors import Embedder, encode_batch, unpack

_LOG = logging.getLogger("local_retrieval")


class RetrievalService:
    def __init__(self, embedder: Embedder, *, source_path: Path | str | None = None,
                 index_path: Path | str | None = None, settings: RetrievalSettings | None = None):
        self.settings = settings or RetrievalSettings()
        self.source = SourceCatalog(source_path, self.settings)
        self.index = IndexStore(index_path, self.settings)
        if (self.source.path == self.index.path
                or self.source.path.exists() and self.index.path.exists()
                and self.source.path.samefile(self.index.path)):
            raise RetrievalError("invalid_local_path")
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5, document_ids: list[str] | None = None,
               mode: str = "hybrid") -> SearchResult:
        started = perf_counter()
        try:
            request = SearchRequest(query=query, top_k=top_k, document_ids=document_ids, mode=mode)
        except ValidationError:
            raise RetrievalError("invalid_arguments") from None
        with self.source.snapshot() as source, self.index.connection() as connection:
            if request.document_ids and not set(request.document_ids).issubset(source):
                raise RetrievalError("unknown_document")
            if connection is None:
                raise RetrievalError("index_unavailable")
            indexed = self.index.read(connection, self.embedder.identity)
            requested = set(request.document_ids) if request.document_ids else set(source)
            eligible = {key for key in requested if indexed.documents.get(key) == source[key].fingerprint}
            valid = {key: item for key, item in indexed.chunks.items() if item.chunk.document_id in eligible}
            # Compare every candidate to authoritative chunk identity/content,
            # not just an indexed document-level assertion.
            authoritative = {chunk.chunk_id: chunk for key in eligible for chunk in source[key].chunks}
            if (set(valid) != set(authoritative)
                    or any(item.fingerprint != chunk_fingerprint(authoritative[key]) for key, item in valid.items())):
                raise RetrievalError("index_corrupt")
            lexical_ids = lexical(connection, request.query, eligible, self.settings.candidate_pool) if mode != "semantic" else []
            semantic_ids, scores = [], {}
            if mode != "lexical" and valid:
                vector = unpack(encode_batch(self.embedder, [request.query])[0], self.embedder.identity.dimension)
                semantic_ids, scores = semantic(valid, vector, self.settings.candidate_pool)
            if not set(lexical_ids).issubset(valid):
                raise RetrievalError("index_corrupt")
            hits = fuse(valid, lexical_ids, semantic_ids, scores, mode=mode, top_k=top_k, rrf_k=self.settings.rrf_k)
            stale = sum(key not in source or source[key].fingerprint != value for key, value in indexed.documents.items())
            result = SearchResult(mode=mode, top_k=top_k, returned_count=len(hits), results=hits,
                                  index_generation=indexed.generation,
                                  candidate_count=len(set(lexical_ids) | set(semantic_ids)),
                                  excluded_stale_documents=stale,
                                  unindexed_documents=len(requested - set(indexed.documents)))
            _LOG.info("retrieval status=ok mode=%s candidate_count=%d returned_count=%d duration_ms=%d generation=%s",
                      mode, result.candidate_count, len(hits), int((perf_counter() - started) * 1000), result.index_generation)
            return result

    def index_document(self, document_id: str) -> dict[str, object]:
        return self._index([document_id])

    def index_all(self, *, rebuild: bool = False) -> dict[str, object]:
        """Sync all authoritative documents, including removals; explicit rebuild changes model."""
        return self._index(None, rebuild=rebuild)

    def sync(self) -> dict[str, object]:
        return self.index_all()

    def _index(self, document_ids: list[str] | None, *, rebuild: bool = False) -> dict[str, object]:
        try:
            SearchRequest(query="index", document_ids=document_ids)
        except ValidationError:
            raise RetrievalError("invalid_arguments") from None
        with self.source.snapshot() as source, self.index.connection() as connection:
            if document_ids and not set(document_ids).issubset(source):
                raise RetrievalError("unknown_document")
            before = self.index.read(connection, self.embedder.identity, allow_model_change=rebuild)
            initial = {key: document.fingerprint for key, document in source.items()}
            requested = set(document_ids) if document_ids else set(source)
            changed = {key: source[key] for key in requested if rebuild or before.documents.get(key) != source[key].fingerprint}
            removals = set(before.documents) - set(source) if document_ids is None else set()
            exists = connection is not None
        vectors = {}
        needed = []
        for document in changed.values():
            for chunk in document.chunks:
                old = before.chunks.get(chunk.chunk_id)
                if not rebuild and old and old.fingerprint == chunk_fingerprint(chunk):
                    vectors[chunk.chunk_id] = old.vector
                else:
                    needed.append(chunk)
        for start in range(0, len(needed), self.settings.embedding_batch_size):
            batch = needed[start:start + self.settings.embedding_batch_size]
            for chunk, vector in zip(batch, encode_batch(self.embedder, [item.text for item in batch]), strict=True):
                vectors[chunk.chunk_id] = vector
        with self.source.snapshot() as current:
            if {key: document.fingerprint for key, document in current.items()} != initial:
                raise RetrievalError("source_changed")
            if changed or removals or rebuild or not exists:
                current_generation = self.index.apply(before, self.embedder.identity, changed, vectors, removals, rebuild=rebuild)
            else:
                current_generation = before.generation
        result = {"status": "indexed" if changed or removals or rebuild or not exists else "already_indexed",
                  "indexed_documents": len(changed), "removed_documents": len(removals),
                  "embedded_chunks": len(needed), "reused_chunks": len(vectors) - len(needed),
                  "index_generation": current_generation}
        _LOG.info("index status=%s indexed_documents=%d embedded_chunks=%d generation=%s",
                  result["status"], len(changed), len(needed), current_generation)
        return result

    def remove(self, document_id: str) -> dict[str, object]:
        try:
            SearchRequest(query="remove", document_ids=[document_id])
        except ValidationError:
            raise RetrievalError("invalid_arguments") from None
        with self.index.connection() as connection:
            before = self.index.read(connection, self.embedder.identity)
        if document_id not in before.documents:
            return {"status": "not_indexed", "index_generation": before.generation}
        result = self.index.apply(before, self.embedder.identity, {}, {}, {document_id})
        return {"status": "removed", "index_generation": result}

    def status(self) -> dict[str, object]:
        with self.source.snapshot() as source, self.index.connection() as connection:
            indexed = self.index.read(connection, self.embedder.identity)
            return {"status": "ready" if connection is not None else "not_indexed",
                    "index_generation": indexed.generation, "embedding": indexed.identity.model_dump(),
                    "documents": len(indexed.documents), "chunks": len(indexed.chunks),
                    "stale_documents": sum(key not in source or source[key].fingerprint != value for key, value in indexed.documents.items()),
                    "unindexed_documents": len(set(source) - set(indexed.documents))}
