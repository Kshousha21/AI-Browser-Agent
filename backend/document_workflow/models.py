"""Strict user-only document requests, safe failures and download receipts."""
from typing import Annotated, Literal
from pydantic import Field, field_validator, model_validator
from browser.protocol import PageState
from local_pdf.models import LocalModel
from browser.providers import ToolCall

Ref = Annotated[str, Field(pattern=r"^result-[a-f0-9]{24}$")]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Nonce = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
MAX_BYTES = 25 * 1024 * 1024
CODES = frozenset({"selection_required", "stale_result_ref", "stale_snapshot", "attachment_unavailable",
    "attachment_preview_only", "unsupported_attachment_type", "no_current_results",
    "preview_selection_limit", "preview_control_missing", "preview_ambiguous", "preview_identity_mismatch", "preview_unexpected_download",
    "read_only_blocked", "confirmation_cancelled", "download_failed", "download_incomplete",
    "download_too_large", "invalid_download_path", "invalid_pdf", "ingestion_failed",
    "pdf_not_found", "pdf_too_large", "pdf_encrypted", "pdf_parse_failed",
    "pdf_limit_exceeded", "pdf_read_failed", "pdf_crypto_unavailable", "pdf_extraction_not_allowed",
    "index_failed", "index_incompatible", "provenance_failed", "document_not_ready",
    "operation_interrupted", "invalid_document_request", "document_operation_failed",
    "document_retry_unavailable", "router_unavailable", "router_timeout", "router_invalid_response", "research_no_relevant_documents", "research_search_incomplete", "research_scope_changed"})


class DocumentError(RuntimeError):
    def __init__(self, code: str):
        self.code = code if code in CODES else "document_operation_failed"
        super().__init__(self.code)


class PrepareRequest(LocalModel):
    offer_id: Nonce
    result_refs: list[Ref] = Field(min_length=1, max_length=2)
    read_only: bool = Field(default=False, exclude=True)  # Legacy preparation clients only.
    question: str | None = Field(default=None, min_length=1, max_length=1000, repr=False)
    allow_remote_context: bool = False

    @field_validator("result_refs")
    @classmethod
    def unique_refs(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("duplicate selection")
        return value


class QuestionRequest(LocalModel):
    document_ids: list[Digest] = Field(min_length=1, max_length=2)
    question: str = Field(min_length=1, max_length=1000, repr=False)
    allow_remote_context: bool = False

    @field_validator("document_ids")
    @classmethod
    def unique_ids(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("duplicate selection")
        return value


class RefreshRequest(LocalModel):
    """Explicit user observation only; no URL, query, refs or approval accepted."""


class FileActionRequest(LocalModel):
    """One human click, one current result; not a model tool or bulk request."""
    offer_id: Nonce
    result_ref: Ref
    action: Literal["preview", "download", "prepare"]


class UserDownloadReceipt(LocalModel):
    """Browser-owned item only; file paths are neither needed nor accepted."""
    download_id: int = Field(ge=0)
    state: Literal["complete"]


class ResearchRequest(LocalModel):
    mode: Literal["automatic", "specific"]
    question: str = Field(min_length=1, max_length=1000, repr=False)
    page_state: PageState = Field(repr=False)
    allowlist: list[str] = Field(default_factory=list, max_length=20, repr=False)
    allow_remote_context: bool = False
    offer_id: Nonce | None = None
    result_ref: Ref | None = None
    document_id: Digest | None = None

    @model_validator(mode="after")
    def exact_scope(self):
        if not self.question.strip():
            raise ValueError("question required")
        if self.page_state.tab_id is None or not self.page_state.document_id or not self.page_state.snapshot_id:
            raise ValueError("current owner/document authority required")
        if self.mode == "automatic":
            if self.offer_id or self.result_ref or self.document_id:
                raise ValueError("automatic scope comes only from current bounded results")
        elif not ((self.document_id is not None and self.offer_id is None and self.result_ref is None)
                  or (self.document_id is None and self.offer_id is not None and self.result_ref is not None)):
            raise ValueError("select exactly one current or prepared PDF")
        return self


class DownloadReceipt(LocalModel):
    acquisition_id: Nonce
    download_id: int = Field(ge=0)
    state: Literal["complete"]
    filename: str = Field(min_length=1, max_length=4096, repr=False)
    size: int = Field(gt=0, le=MAX_BYTES)


class ApprovedAcquisition(ToolCall):
    """Only the user workflow constructs this after exact one-shot confirmation.

    This class is not a model tool/schema and is never registered with providers.
    """


class ApprovedPreview(ToolCall):
    """Internal user-only preview navigation/inspection; not a native model tool."""


class ApprovedUserDownload(ToolCall):
    """Exact one-shot UI download; never registered with a model provider."""


class ApprovedResearchReturn(ToolCall):
    """Return only to this research task's previously observed results page."""
