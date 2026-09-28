"""Atomic local index generations, metadata, FTS5 rows and float32 vectors."""
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import sqlite3

from pydantic import ValidationError
from local_pdf.models import DocumentChunk, PIPELINE_VERSION
from local_storage import StorageError, connect_database

from .errors import RetrievalError
from .models import EmbeddingIdentity, PIPELINE, RetrievalSettings, fingerprint
from .schema import APPLICATION_ID, DDL, SCHEMA_VERSION
from .source import SourceDocument, chunk_fingerprint, store_path
from .vectors import unpack
from .embedding_identity import compatible_identity

def generation(identity: EmbeddingIdentity, documents: dict[str, str]) -> str:
    return fingerprint({"embedding": identity.model_dump(exclude_none=True), "pipeline": PIPELINE,
                        "source_pipeline": PIPELINE_VERSION, "documents": documents})


@dataclass(frozen=True)
class IndexedChunk:
    chunk: DocumentChunk = field(repr=False)
    fingerprint: str
    vector: bytes = field(repr=False)


@dataclass(frozen=True)
class IndexSnapshot:
    identity: EmbeddingIdentity
    generation: str
    documents: dict[str, str]
    chunks: dict[str, IndexedChunk] = field(repr=False)


class IndexStore:
    def __init__(self, path: Path | str | None = None, settings: RetrievalSettings | None = None):
        self.path = store_path(path, "retrieval.sqlite3")
        self.settings = settings or RetrievalSettings()

    @contextmanager
    def connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection | None]:
        if not write and not self.path.exists():
            yield None
            return
        connection = None
        try:
            connection = connect_database(self.path, write=write)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            if write:
                connection.execute("PRAGMA secure_delete=ON")
                connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            ready = self._schema(connection, write)
            yield connection if ready else None
            if write:
                connection.commit()
        except RetrievalError:
            raise
        except (sqlite3.Error, OSError, ValueError, TypeError, ValidationError, StorageError):
            raise RetrievalError("index_write_failed" if write else "index_corrupt") from None
        finally:
            if connection is not None:
                connection.close()  # Uncommitted work rolls back, including new schema.

    @staticmethod
    def _schema(connection: sqlite3.Connection, create: bool) -> bool:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        identity = connection.execute("PRAGMA application_id").fetchone()[0]
        if (version, identity) == (SCHEMA_VERSION, APPLICATION_ID):
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RetrievalError("index_corrupt")
            return True
        if version or identity or connection.execute("SELECT name FROM sqlite_master").fetchone():
            raise RetrievalError("unsupported_index")
        if not create:
            return False
        try:
            for statement in DDL:
                connection.execute(statement)
        except sqlite3.OperationalError as error:
            if "no such module: fts5" in str(error):
                raise RetrievalError("fts5_unavailable") from None
            raise
        connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        return True

    def read(self, connection: sqlite3.Connection | None, expected: EmbeddingIdentity,
             *, allow_model_change: bool = False) -> IndexSnapshot:
        if connection is None:
            return IndexSnapshot(expected, generation(expected, {}), {}, {})
        metadata = connection.execute("SELECT * FROM index_metadata WHERE singleton=1").fetchone()
        if metadata is None:
            if any(connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
                   for table in ("indexed_documents", "indexed_chunks", "chunk_embeddings")):
                raise RetrievalError("index_corrupt")
            return IndexSnapshot(expected, generation(expected, {}), {}, {})
        identity = EmbeddingIdentity.model_validate_json(metadata["identity_json"])
        if metadata["pipeline"] != PIPELINE or metadata["source_pipeline"] != PIPELINE_VERSION:
            raise RetrievalError("unsupported_index")
        if not allow_model_change and not compatible_identity(identity, expected):
            raise RetrievalError("index_model_mismatch")
        documents = dict(connection.execute("SELECT document_id,fingerprint FROM indexed_documents LIMIT ?",
                                            (self.settings.max_documents + 1,)).fetchall())
        rows = connection.execute("SELECT c.*,e.vector FROM indexed_chunks c LEFT JOIN chunk_embeddings e USING(chunk_id) ORDER BY c.document_id,c.chunk_index LIMIT ?",
                                  (self.settings.max_chunks + 1,)).fetchall()
        if len(documents) > self.settings.max_documents or len(rows) > self.settings.max_chunks:
            raise RetrievalError("index_capacity_exceeded")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RetrievalError("index_corrupt")
        chunks = {}
        chars = 0
        for row in rows:
            chunk = DocumentChunk.model_validate({name: row[name] for name in DocumentChunk.model_fields})
            chars += len(chunk.text)
            if chars > self.settings.max_text_chars:
                raise RetrievalError("index_capacity_exceeded")
            if (row["text_hash"] != hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
                    or row["fingerprint"] != chunk_fingerprint(chunk)):
                raise RetrievalError("index_corrupt")
            unpack(row["vector"], identity.dimension)
            chunks[chunk.chunk_id] = IndexedChunk(chunk, row["fingerprint"], row["vector"])
        if generation(identity, documents) != metadata["generation"]:
            raise RetrievalError("index_corrupt")
        return IndexSnapshot(identity, generation(identity, documents), documents, chunks)

    @staticmethod
    def insert_chunk(connection: sqlite3.Connection, chunk: DocumentChunk, vector: bytes) -> None:
        values = {**chunk.model_dump(), "text_hash": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                  "fingerprint": chunk_fingerprint(chunk)}
        connection.execute(f"INSERT INTO indexed_chunks ({','.join(values)}) VALUES ({','.join('?' for _ in values)})", tuple(values.values()))
        connection.execute("INSERT INTO chunk_embeddings VALUES (?,?)", (chunk.chunk_id, vector))

    def apply(self, before: IndexSnapshot, identity: EmbeddingIdentity,
              documents: dict[str, SourceDocument], vectors: dict[str, bytes],
              removals: set[str], *, rebuild: bool = False) -> str:
        """Publish one generation or roll back everything; source lock is held by caller."""
        with self.connection(write=True) as connection:
            assert connection is not None
            current = self.read(connection, before.identity)
            if current.generation != before.generation:
                raise RetrievalError("source_changed")
            if rebuild:
                connection.execute("DELETE FROM indexed_documents")
            for identifier in sorted(removals | set(documents)):
                connection.execute("DELETE FROM indexed_documents WHERE document_id=?", (identifier,))
            for identifier, document in sorted(documents.items()):
                connection.execute("INSERT INTO indexed_documents VALUES (?,?)", (identifier, document.fingerprint))
                for chunk in document.chunks:
                    self.insert_chunk(connection, chunk, vectors[chunk.chunk_id])
            identities = dict(connection.execute("SELECT document_id,fingerprint FROM indexed_documents").fetchall())
            if len(identities) > self.settings.max_documents or connection.execute("SELECT count(*) FROM indexed_chunks").fetchone()[0] > self.settings.max_chunks:
                raise RetrievalError("index_capacity_exceeded")
            result = generation(identity, identities)
            connection.execute("INSERT OR REPLACE INTO index_metadata VALUES (1,?,?,?,?)",
                               (identity.model_dump_json(), PIPELINE, PIPELINE_VERSION, result))
            connection.execute("INSERT INTO chunk_fts(chunk_fts,rank) VALUES('integrity-check',1)")
            return result
