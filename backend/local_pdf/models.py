"""Validated local configuration and explicitly untrusted stored records."""
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

PIPELINE_VERSION = "page-text-v1"
Trust = Literal["untrusted_document_data"]


class LocalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)


class IngestionSettings(LocalModel):
    """Independent of browser/provider settings; no environment or .env loading."""

    max_file_bytes: int = Field(default=25 * 1024 * 1024, ge=1, le=100 * 1024 * 1024)
    max_pages: int = Field(default=2000, ge=1, le=10000)
    max_page_stream_bytes: int = Field(default=8 * 1024 * 1024, ge=1, le=32 * 1024 * 1024)
    max_page_chars: int = Field(default=250000, ge=1, le=1000000)
    max_total_chars: int = Field(default=10000000, ge=1, le=50000000)
    max_chunks: int = Field(default=20000, ge=1, le=100000)
    max_chunk_chars: int = Field(default=1600, ge=128, le=16000)
    overlap_chars: int = Field(default=200, ge=0, le=2000)

    @model_validator(mode="after")
    def bounded_overlap(self) -> Self:
        if self.overlap_chars > self.max_chunk_chars // 2:
            raise ValueError("overlap must not exceed half the maximum chunk size")
        return self


class DocumentMetadata(LocalModel):
    document_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    filename: str = Field(max_length=200, repr=False)
    file_size: int = Field(ge=1)
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=0)
    extracted_chars: int = Field(ge=0)
    ingested_at: str = Field(max_length=40)
    parser_name: Literal["pypdf"] = "pypdf"
    parser_version: str = Field(max_length=32)
    source_type: Literal["local_pdf"] = "local_pdf"
    trust: Trust = "untrusted_document_data"
    pipeline_version: Literal["page-text-v1"] = PIPELINE_VERSION
    max_chunk_chars: int = Field(ge=128, le=16000)
    overlap_chars: int = Field(ge=0, le=2000)

    def safe_metadata(self) -> dict[str, object]:
        """Bounded CLI output: omit even the local filename, not just body text."""
        return self.model_dump(exclude={"filename"})


class DocumentPage(LocalModel):
    document_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_number: int = Field(ge=1)
    text: str = Field(repr=False)
    text_status: Literal["text", "empty"]
    trust: Trust = "untrusted_document_data"

    @model_validator(mode="after")
    def consistent_status(self) -> Self:
        if (self.text_status == "empty") != (self.text == ""):
            raise ValueError("page text status must match its text")
        return self


class DocumentChunk(LocalModel):
    chunk_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    document_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    chunk_index: int = Field(ge=0)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=1)
    text: str = Field(min_length=1, repr=False)
    trust: Trust = "untrusted_document_data"

    @model_validator(mode="after")
    def consistent_provenance(self) -> Self:
        if self.page_start != self.page_end or self.char_end - self.char_start != len(self.text):
            raise ValueError("chunk must be a contiguous slice of one page")
        return self


class IngestionResult(LocalModel):
    status: Literal["ingested", "already_ingested"]
    document: DocumentMetadata
