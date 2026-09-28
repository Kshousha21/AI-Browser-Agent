"""Consistent read-only snapshots of the authoritative Phase 4C schema."""
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3

from pydantic import ValidationError
from local_pdf.models import DocumentChunk, DocumentMetadata
from local_pdf.store import APPLICATION_ID, SCHEMA_VERSION, LOCAL_DIRECTORY, ROOT
from local_storage import StorageError, database_path, connect_database

from .errors import RetrievalError
from .models import RetrievalSettings, fingerprint


def local_path(value: Path | str) -> Path:
    """Only operator-supplied paths; checkout-local data belongs in .local."""
    try:
        path = Path(value).expanduser().resolve()
        if path.is_relative_to(ROOT) and not path.is_relative_to(LOCAL_DIRECTORY):
            raise RetrievalError("invalid_local_path")
        return path
    except (OSError, ValueError, TypeError):
        raise RetrievalError("invalid_local_path") from None


def chunk_fingerprint(chunk: DocumentChunk) -> str:
    return fingerprint(chunk.model_dump())


def store_path(value: Path | str | None, filename: str) -> Path:
    try:
        return database_path(value, filename)
    except StorageError as error:
        raise RetrievalError("network_storage_unsupported" if error.code == "network_storage_unsupported"
                             else "invalid_local_path") from None


@dataclass(frozen=True)
class SourceDocument:
    metadata: DocumentMetadata = field(repr=False)
    chunks: tuple[DocumentChunk, ...] = field(repr=False)
    fingerprint: str


class SourceCatalog:
    def __init__(self, path: Path | str | None = None, settings: RetrievalSettings | None = None):
        self.path = store_path(path, "documents.sqlite3")
        self.settings = settings or RetrievalSettings()

    @contextmanager
    def snapshot(self) -> Iterator[dict[str, SourceDocument]]:
        """Hold a source read transaction through result construction/index commit.

        Phase 4C uses SQLite rollback journals. Reject a manually enabled WAL
        source: a read snapshot there would not prevent concurrent deletion.
        No PRAGMA or schema write is issued against the ingestion database.
        """
        if not self.path.exists():
            yield {}
            return
        connection = None
        try:
            connection = connect_database(self.path)
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            identity = connection.execute("PRAGMA application_id").fetchone()[0]
            if (version, identity) == (0, 0) and not connection.execute("SELECT name FROM sqlite_master").fetchone():
                yield {}
                return
            if version != SCHEMA_VERSION or identity != APPLICATION_ID:
                raise RetrievalError("source_unavailable")
            if connection.execute("PRAGMA journal_mode").fetchone()[0] not in {"delete", "truncate", "persist"}:
                raise RetrievalError("source_unavailable")
            documents = connection.execute("SELECT * FROM documents ORDER BY document_id LIMIT ?",
                                           (self.settings.max_documents + 1,)).fetchall()
            rows = connection.execute("SELECT * FROM chunks ORDER BY document_id,chunk_index LIMIT ?",
                                      (self.settings.max_chunks + 1,)).fetchall()
            if len(documents) > self.settings.max_documents or len(rows) > self.settings.max_chunks:
                raise RetrievalError("index_capacity_exceeded")
            grouped: dict[str, list[DocumentChunk]] = {}
            chars = 0
            for row in rows:
                chunk = DocumentChunk.model_validate(dict(row))
                chars += len(chunk.text)
                if chars > self.settings.max_text_chars:
                    raise RetrievalError("index_capacity_exceeded")
                grouped.setdefault(chunk.document_id, []).append(chunk)
            result = {}
            for row in documents:
                metadata = DocumentMetadata.model_validate(dict(row))
                chunks = grouped.pop(metadata.document_id, [])
                if len(chunks) != metadata.chunk_count or [c.chunk_index for c in chunks] != list(range(len(chunks))):
                    raise RetrievalError("source_unavailable")
                signature = fingerprint({
                    "document_id": metadata.document_id, "parser": metadata.parser_version,
                    "pipeline": metadata.pipeline_version, "chunk_chars": metadata.max_chunk_chars,
                    "overlap": metadata.overlap_chars, "chunks": [chunk_fingerprint(c) for c in chunks],
                })
                result[metadata.document_id] = SourceDocument(metadata, tuple(chunks), signature)
            if grouped:
                raise RetrievalError("source_unavailable")
            yield result
        except (sqlite3.Error, OSError, ValueError, ValidationError, StorageError):
            raise RetrievalError("source_unavailable") from None
        finally:
            if connection is not None:
                connection.close()
