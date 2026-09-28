"""Strict local query, index identity and untrusted result contracts."""
from typing import Annotated, Literal, Self
import hashlib
import json

from pydantic import Field, field_validator, model_validator
from local_pdf.models import DocumentChunk, LocalModel

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Mode = Literal["lexical", "semantic", "hybrid"]
PIPELINE = "fts5-cosine-rrf-v1"


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True).encode("utf-8")).hexdigest()


class RetrievalSettings(LocalModel):
    candidate_pool: int = Field(default=50, ge=20, le=200)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    max_documents: int = Field(default=1000, ge=1, le=10000)
    max_chunks: int = Field(default=10000, ge=1, le=100000)
    max_text_chars: int = Field(default=20000000, ge=1, le=100000000)
    embedding_batch_size: int = Field(default=8, ge=1, le=32)


class SearchRequest(LocalModel):
    query: str = Field(min_length=1, max_length=1000, repr=False)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[Digest] | None = Field(default=None, min_length=1, max_length=100)
    mode: Mode = "hybrid"

    @field_validator("query")
    @classmethod
    def trim_query(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(char) < 32 and char not in "\n\t\r" for char in value):
            raise ValueError("invalid local query")
        return value

    @field_validator("document_ids")
    @classmethod
    def unique_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("duplicate document filter")
        return sorted(value) if value else value


class EmbeddingIdentity(LocalModel):
    model_id: str = Field(min_length=1, max_length=200)
    revision: str = Field(min_length=1, max_length=100)
    artifact_fingerprint: Digest
    dimension: int = Field(ge=1, le=2048)
    backend: str = Field(min_length=1, max_length=200)
    embedding_pipeline: str = Field(min_length=1, max_length=100)
    normalization: Literal["float32-l2"] = "float32-l2"
    execution_artifact: str | None = Field(default=None, max_length=100)
    artifact_sha256: Digest | None = None
    precision: Literal["quint8", "fp16", "fp32"] | None = None
    execution_provider: Literal["CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"] | None = None

    @property
    def identity(self) -> str:
        return fingerprint(self.model_dump(exclude_none=True))


class SearchHit(DocumentChunk):
    rank: int = Field(ge=1, le=20)
    bm25_rank: int | None = Field(default=None, ge=1, le=200)
    semantic_rank: int | None = Field(default=None, ge=1, le=200)
    hybrid_score: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cosine_score: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)

    def safe_metadata(self) -> dict[str, object]:
        return self.model_dump(exclude={"text"})


class SearchResult(LocalModel):
    mode: Mode
    top_k: int = Field(ge=1, le=20)
    returned_count: int = Field(ge=0, le=20)
    index_generation: Digest
    candidate_count: int = Field(ge=0, le=400)
    excluded_stale_documents: int = Field(ge=0)
    unindexed_documents: int = Field(ge=0)
    results: list[SearchHit] = Field(max_length=20, repr=False)

    @model_validator(mode="after")
    def consistent_count(self) -> Self:
        if self.returned_count != len(self.results) or self.returned_count > self.top_k:
            raise ValueError("invalid retrieval count")
        return self

    def safe_metadata(self) -> dict[str, object]:
        return {**self.model_dump(exclude={"results"}), "results": [hit.safe_metadata() for hit in self.results]}
