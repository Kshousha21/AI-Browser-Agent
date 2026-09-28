"""Closed, content-free errors safe for a local CLI or application log."""
from typing import Literal

ErrorCode = Literal[
    "pdf_not_found", "invalid_pdf", "pdf_too_large", "pdf_encrypted",
    "pdf_parse_failed", "pdf_limit_exceeded", "pdf_read_failed",
    "pdf_crypto_unavailable", "pdf_extraction_not_allowed",
    "storage_failed", "unsupported_store", "invalid_store_path",
    "network_storage_unsupported",
    "document_not_found", "invalid_document_id", "invalid_arguments",
]


class PdfIngestError(RuntimeError):
    """Never attach a parser message, document text, filename or local path."""

    def __init__(self, code: ErrorCode) -> None:
        self.code = code
        super().__init__(code)
