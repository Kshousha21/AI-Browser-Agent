"""Explicit local ingestion entry point, independent of browser/provider config."""
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import re
from time import perf_counter

from .errors import PdfIngestError
from .models import DocumentMetadata, IngestionResult, IngestionSettings
from .parser import PARSER_VERSION, extract_pages, read_pdf
from .store import DocumentStore
from .text import chunk_pages

_LOG = logging.getLogger("local_pdf")


class PdfIngestionService:
    def __init__(self, store: DocumentStore | None = None, settings: IngestionSettings | None = None):
        self.store = store if store is not None else DocumentStore()
        self.settings = settings if settings is not None else IngestionSettings()

    def ingest_pdf(self, local_path: Path | str) -> IngestionResult:
        """Accept only an operator's local path. Never exposed as a model tool."""
        started = perf_counter()
        try:
            path = Path(local_path)
            data = read_pdf(path, self.settings)
            document_id = hashlib.sha256(data).hexdigest()
            existing = self.store.get(document_id)
            if existing is not None:
                result = IngestionResult(status="already_ingested", document=existing)
            else:
                pages = extract_pages(data, document_id, self.settings)
                chunks = chunk_pages(pages, self.settings)
                document = DocumentMetadata(
                    document_id=document_id, content_hash=document_id,
                    filename=re.sub(r"[\x00-\x1f\x7f-\x9f]", "", path.name)[:200],
                    file_size=len(data), page_count=len(pages), chunk_count=len(chunks),
                    extracted_chars=sum(len(page.text) for page in pages),
                    ingested_at=datetime.now(timezone.utc).isoformat(), parser_version=PARSER_VERSION,
                    max_chunk_chars=self.settings.max_chunk_chars, overlap_chars=self.settings.overlap_chars,
                )
                result = self.store.put(document, pages, chunks)
            _LOG.info("pdf_ingestion status=%s document_id=%s page_count=%d chunk_count=%d duration_ms=%d",
                      result.status, result.document.document_id, result.document.page_count,
                      result.document.chunk_count, int((perf_counter() - started) * 1000))
            return result
        except PdfIngestError as error:
            _LOG.warning("pdf_ingestion status=%s", error.code)
            raise
