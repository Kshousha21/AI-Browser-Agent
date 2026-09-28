"""Literal FTS5 query construction, local cosine and deterministic reciprocal ranks."""
import re
import sqlite3

import numpy as np
from local_pdf.models import DocumentChunk

from .errors import RetrievalError
from .index_store import IndexedChunk
from .models import SearchHit
from .vectors import Vector, unpack


def lexical(connection: sqlite3.Connection, query: str, documents: set[str], pool: int) -> list[str]:
    terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query.lower())))
    if len(terms) > 128:
        raise RetrievalError("invalid_arguments")
    if not terms or not documents:
        return []
    # User punctuation/operators cannot become FTS grammar. OR supports partial
    # keyword matches; BM25 rewards multiple and rare terms using corpus stats.
    expression = " OR ".join('"' + term + '"' for term in terms)
    placeholders = ",".join("?" for _ in documents)
    rows = connection.execute(
        "SELECT c.chunk_id FROM chunk_fts JOIN indexed_chunks c ON c.id=chunk_fts.rowid "
        f"WHERE chunk_fts MATCH ? AND c.document_id IN ({placeholders}) "
        "ORDER BY bm25(chunk_fts),c.document_id,c.chunk_index,c.chunk_id LIMIT ?",
        (expression, *sorted(documents), pool)).fetchall()
    return [row[0] for row in rows]


def semantic(chunks: dict[str, IndexedChunk], query: Vector, pool: int) -> tuple[list[str], dict[str, float]]:
    if not chunks:
        return [], {}
    ordered = sorted(chunks, key=lambda key: stable_key(chunks[key].chunk))
    matrix = np.stack([unpack(chunks[key].vector, len(query)) for key in ordered])
    scores = np.clip(matrix @ query, -1, 1)
    if not np.isfinite(scores).all():
        raise RetrievalError("invalid_embedding")
    positions = sorted(range(len(ordered)), key=lambda i: (-float(scores[i]), stable_key(chunks[ordered[i]].chunk)))[:pool]
    return [ordered[i] for i in positions], {ordered[i]: float(scores[i]) for i in positions}


def stable_key(chunk: DocumentChunk) -> tuple[str, int, str]:
    return chunk.document_id, chunk.chunk_index, chunk.chunk_id


def reciprocal_rank_scores(*rankings, rrf_k=60):
    ranks = [{key: rank for rank, key in enumerate(dict.fromkeys(ranking), 1)} for ranking in rankings]
    return {key: sum(1.0 / (rrf_k + source[key]) for source in ranks if key in source)
            for key in set().union(*(set(source) for source in ranks))}


def fuse(chunks: dict[str, IndexedChunk], lexical_ids: list[str], semantic_ids: list[str],
         cosine: dict[str, float], *, mode: str, top_k: int, rrf_k: int) -> list[SearchHit]:
    bm25_ranks = {key: rank for rank, key in enumerate(lexical_ids, 1)}
    semantic_ranks = {key: rank for rank, key in enumerate(semantic_ids, 1)}
    candidates = set(lexical_ids) | set(semantic_ids)
    scores = reciprocal_rank_scores(lexical_ids, semantic_ids, rrf_k=rrf_k)
    if mode == "hybrid":
        ordered = sorted(candidates, key=lambda key: (-scores[key], min(bm25_ranks.get(key, 1000000), semantic_ranks.get(key, 1000000)), stable_key(chunks[key].chunk)))
    else:
        ordered = lexical_ids if mode == "lexical" else semantic_ids
    return [SearchHit(**chunks[key].chunk.model_dump(), rank=rank,
                      bm25_rank=bm25_ranks.get(key), semantic_rank=semantic_ranks.get(key),
                      hybrid_score=scores[key] if mode == "hybrid" else None, cosine_score=cosine.get(key))
            for rank, key in enumerate(ordered[:top_k], 1)]
