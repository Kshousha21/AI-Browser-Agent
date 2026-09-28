"""Strict bounded contracts. Private text is excluded from representations."""
from typing import Literal
from pydantic import Field
from local_pdf.models import LocalModel
from local_retrieval.models import SearchRequest, Digest, Mode

INSUFFICIENT = "The supplied evidence is insufficient to answer this question."


class RagRequest(SearchRequest):
    top_k: int = Field(default=5, ge=1, le=10)


class RagSettings(LocalModel):
    max_visual_pages: int = Field(default=3, ge=1, le=5)
    max_context_chars: int = Field(default=14000, ge=512, le=16000)
    max_request_chars: int = Field(default=20000, ge=2048, le=24000)
    max_answer_chars: int = Field(default=4000, ge=1, le=4000)
    max_output_chars: int = Field(default=16000, ge=512, le=16000)
    max_attempts: int = Field(default=2, ge=1, le=2)
    max_output_tokens: int = Field(default=2048, ge=1, le=4096)
    timeout_seconds: float = Field(default=30.0, ge=0.01, le=120.0)


class EvidenceCitation(LocalModel):
    source_id: str = Field(pattern=r"^S(?:[1-9]|1[0-5])$")
    evidence: str = Field(min_length=1, max_length=200, repr=False)
    evidence_type: Literal["text"] = "text"


class VisualEvidenceCitation(LocalModel):
    source_id: str = Field(pattern=r"^S(?:[1-9]|1[0-5])$")
    evidence_type: Literal["visual"]


class ProviderAnswer(LocalModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: str = Field(min_length=1, max_length=4000, repr=False)
    citations: list[EvidenceCitation | VisualEvidenceCitation] = Field(max_length=20, repr=False)


class VisualProviderAnswer(ProviderAnswer):
    """Image-only evidence cannot be cited as extracted text."""
    citations: list[VisualEvidenceCitation] = Field(max_length=20, repr=False)


class Citation(LocalModel):
    source_id: str
    document_id: Digest
    chunk_id: Digest | None = None
    page_start: int
    page_end: int
    chunk_index: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    evidence_type: Literal["text", "visual", "both"] = "text"
    rendered_page_id: Digest | None = None
    trust: Literal["untrusted_document_data"] = "untrusted_document_data"


class RagResult(LocalModel):
    retrieval_status: dict[str, str] = Field(default_factory=dict)
    notices: list[str] = Field(default_factory=list)
    status: Literal["answered", "insufficient_evidence"]
    answer: str = Field(repr=False)
    citations: list[Citation]
    mode: Mode
    top_k: int
    retrieved_count: int
    context_count: int
    dropped_due_budget: int
    index_generation: Digest
    attempts: int
    measurements: dict[str, int | float]

    def safe_metadata(self) -> dict:
        return self.model_dump(exclude={"answer"})
