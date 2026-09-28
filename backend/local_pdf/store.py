"""Versioned local SQLite records. No source paths, PDFs, credentials or APIs."""
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3
from typing import TypeVar

from pydantic import ValidationError
from local_storage import ROOT, StorageError, database_path, connect_database

from .errors import PdfIngestError
from .models import DocumentChunk, DocumentMetadata, DocumentPage, IngestionResult, LocalModel

Record = TypeVar("Record", DocumentPage, DocumentChunk)

LOCAL_DIRECTORY = ROOT / ".local"
SCHEMA_VERSION = 1
APPLICATION_ID = 0x50444631
_SCHEMA = (
    """CREATE TABLE documents (
        document_id TEXT PRIMARY KEY CHECK(document_id = content_hash),
        content_hash TEXT NOT NULL UNIQUE, filename TEXT NOT NULL,
        file_size INTEGER NOT NULL, page_count INTEGER NOT NULL,
        chunk_count INTEGER NOT NULL, extracted_chars INTEGER NOT NULL,
        ingested_at TEXT NOT NULL, parser_name TEXT NOT NULL,
        parser_version TEXT NOT NULL, source_type TEXT NOT NULL,
        trust TEXT NOT NULL, pipeline_version TEXT NOT NULL,
        max_chunk_chars INTEGER NOT NULL, overlap_chars INTEGER NOT NULL)""",
    """CREATE TABLE pages (
        document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
        page_number INTEGER NOT NULL CHECK(page_number >= 1), text TEXT NOT NULL,
        text_status TEXT NOT NULL CHECK(text_status IN ('text', 'empty')), trust TEXT NOT NULL,
        PRIMARY KEY(document_id, page_number))""",
    """CREATE TABLE chunks (
        chunk_id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL CHECK(chunk_index >= 0),
        page_start INTEGER NOT NULL, page_end INTEGER NOT NULL CHECK(page_end = page_start),
        char_start INTEGER NOT NULL CHECK(char_start >= 0),
        char_end INTEGER NOT NULL CHECK(char_end > char_start),
        text TEXT NOT NULL, trust TEXT NOT NULL,
        UNIQUE(document_id, chunk_index),
        FOREIGN KEY(document_id, page_start) REFERENCES pages(document_id, page_number)
            ON DELETE CASCADE)""",
)


def validate_id(document_id: str) -> None:
    if not isinstance(document_id, str) or not re.fullmatch(r"[a-f0-9]{64}", document_id):
        raise PdfIngestError("invalid_document_id")


class DocumentStore:
    """Operator-selected store; paths inside this checkout must be in .local.

    A private external directory may be explicitly selected for local use/tests.
    This is an application store, not an interface for opening arbitrary SQLite.
    """

    def __init__(self, path: Path | str | None = None):
        try:
            self.path = database_path(path, "documents.sqlite3")
        except StorageError as error:
            raise PdfIngestError("network_storage_unsupported" if error.code == "network_storage_unsupported"
                                 else "invalid_store_path") from None

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection | None]:
        connection = None
        try:
            connection = connect_database(self.path, write=write)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            if write:
                connection.execute("PRAGMA secure_delete = ON")
                connection.execute("PRAGMA synchronous = FULL")
                connection.execute("BEGIN IMMEDIATE")
            initialized = self._schema(connection, create=write)
            yield connection if initialized else None
            if write:
                connection.commit()
        except PdfIngestError:
            if connection is not None and connection.in_transaction:
                connection.rollback()
            raise
        except (sqlite3.Error, OSError, ValueError, ValidationError, StorageError):
            if connection is not None and connection.in_transaction:
                connection.rollback()
            raise PdfIngestError("storage_failed") from None
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _schema(connection: sqlite3.Connection, *, create: bool) -> bool:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        identity = connection.execute("PRAGMA application_id").fetchone()[0]
        if version == SCHEMA_VERSION and identity == APPLICATION_ID:
            return True
        tables = connection.execute("SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall()
        if version or identity or tables:
            raise PdfIngestError("unsupported_store")
        if not create:
            # A rolled-back first write can leave an empty SQLite file. Reads
            # see an empty store; the next write may initialize it atomically.
            return False
        for statement in _SCHEMA:
            connection.execute(statement)
        connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return True

    def get(self, document_id: str) -> DocumentMetadata | None:
        validate_id(document_id)
        if not self.path.exists():
            return None
        with self._connection() as connection:
            if connection is None:
                return None
            row = connection.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
            return DocumentMetadata.model_validate(dict(row)) if row else None

    @staticmethod
    def _bounds(limit: int, offset: int) -> None:
        if type(limit) is not int or type(offset) is not int or not 1 <= limit <= 100 or not 0 <= offset <= 1000000:
            raise PdfIngestError("invalid_arguments")

    def list_documents(self, *, limit: int = 20, offset: int = 0) -> list[DocumentMetadata]:
        self._bounds(limit, offset)
        if not self.path.exists():
            return []
        with self._connection() as connection:
            if connection is None:
                return []
            rows = connection.execute("SELECT * FROM documents ORDER BY ingested_at, document_id LIMIT ? OFFSET ?",
                                      (limit, offset)).fetchall()
            return [DocumentMetadata.model_validate(dict(row)) for row in rows]

    def pages(self, document_id: str, *, limit: int = 100, offset: int = 0) -> list[DocumentPage]:
        """Local data access only; callers must retain the untrusted-data marker."""
        return self._records(document_id, "pages", "page_number", DocumentPage, limit, offset)

    def chunks(self, document_id: str, *, limit: int = 100, offset: int = 0) -> list[DocumentChunk]:
        """Ordered page slices; never inserted into model-facing context here."""
        return self._records(document_id, "chunks", "chunk_index", DocumentChunk, limit, offset)

    def _records(self, document_id: str, table: str, order: str, model: type[Record],
                 limit: int, offset: int) -> list[Record]:
        validate_id(document_id)
        self._bounds(limit, offset)
        if not self.path.exists():
            return []
        with self._connection() as connection:
            if connection is None:
                return []
            # Table/order come only from the two fixed methods above.
            rows = connection.execute(f"SELECT * FROM {table} WHERE document_id = ? ORDER BY {order} LIMIT ? OFFSET ?",
                                      (document_id, limit, offset)).fetchall()
            return [model.model_validate(dict(row)) for row in rows]

    @staticmethod
    def _insert_records(connection: sqlite3.Connection, table: str, records: Sequence[LocalModel]) -> None:
        for record in records:
            values = record.model_dump()
            columns = ",".join(values)
            placeholders = ",".join("?" for _ in values)
            connection.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(values.values()))

    def put(self, document: DocumentMetadata, pages: Sequence[DocumentPage],
            chunks: Sequence[DocumentChunk]) -> IngestionResult:
        """Insert all rows atomically; recheck identity under the write lock."""
        with self._connection(write=True) as connection:
            assert connection is not None
            row = connection.execute("SELECT * FROM documents WHERE content_hash = ?",
                                     (document.content_hash,)).fetchone()
            if row:
                return IngestionResult(status="already_ingested", document=DocumentMetadata.model_validate(dict(row)))
            self._validate_records(document, pages, chunks)
            self._insert_records(connection, "documents", [document])
            self._insert_records(connection, "pages", pages)
            self._insert_records(connection, "chunks", chunks)
            return IngestionResult(status="ingested", document=document)

    @staticmethod
    def _validate_records(document: DocumentMetadata, pages: Sequence[DocumentPage],
                          chunks: Sequence[DocumentChunk]) -> None:
        if (document.document_id != document.content_hash or len(pages) != document.page_count
                or len(chunks) != document.chunk_count or sum(len(page.text) for page in pages) != document.extracted_chars):
            raise PdfIngestError("storage_failed")
        for number, page in enumerate(pages, start=1):
            if page.document_id != document.document_id or page.page_number != number:
                raise PdfIngestError("storage_failed")
        for index, chunk in enumerate(chunks):
            if (chunk.document_id != document.document_id or chunk.chunk_index != index
                    or chunk.page_start > len(pages) or len(chunk.text) > document.max_chunk_chars
                    or chunk.text != pages[chunk.page_start - 1].text[chunk.char_start:chunk.char_end]):
                raise PdfIngestError("storage_failed")

    def delete(self, document_id: str) -> bool:
        """Delete the document and dependent rows in one transaction. No unlink."""
        validate_id(document_id)
        if not self.path.exists():
            return False
        with self._connection(write=True) as connection:
            assert connection is not None
            return connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,)).rowcount == 1
