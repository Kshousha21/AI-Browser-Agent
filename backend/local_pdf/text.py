"""Conservative text normalization and deterministic, page-local chunking."""
from collections.abc import Sequence
import hashlib
import re

from .errors import PdfIngestError
from .models import DocumentChunk, DocumentPage, IngestionSettings, PIPELINE_VERSION


def normalize_text(text: str) -> str:
    """Normalize line endings/spacing only; never rewrite words, units or numbers."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_pages(pages: Sequence[DocumentPage], settings: IngestionSettings) -> list[DocumentChunk]:
    """Keep exact page slices/overlap; prefer existing paragraph, then word breaks.

    Positions are character offsets in normalized page text. Chunk indices are
    zero-based document-wide; page numbers are one-based. Empty pages emit none.
    """
    chunks: list[DocumentChunk] = []
    for page in pages:
        start = 0
        while start < len(page.text):
            end = min(start + settings.max_chunk_chars, len(page.text))
            if end < len(page.text):
                lower = start + max(settings.overlap_chars + 1, settings.max_chunk_chars // 2)
                boundary = page.text.rfind("\n\n", lower, end)
                if boundary >= lower:
                    end = boundary + 2
                else:
                    boundary = page.text.rfind(" ", lower, end)
                    if boundary >= lower:
                        end = boundary + 1
            if len(chunks) >= settings.max_chunks:
                raise PdfIngestError("pdf_limit_exceeded")
            position = (f"{page.document_id}:{PIPELINE_VERSION}:{settings.max_chunk_chars}:"
                        f"{settings.overlap_chars}:{page.page_number}:{start}:{end}:{len(chunks)}")
            chunks.append(DocumentChunk(
                chunk_id=hashlib.sha256(position.encode("ascii")).hexdigest(),
                document_id=page.document_id, chunk_index=len(chunks),
                page_start=page.page_number, page_end=page.page_number,
                char_start=start, char_end=end, text=page.text[start:end],
            ))
            if end == len(page.text):
                break
            start = end - settings.overlap_chars
    return chunks
