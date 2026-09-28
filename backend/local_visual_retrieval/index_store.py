"""Separate versioned SQLite page vectors; atomic explicit index/rebuild only."""
from contextlib import contextmanager
from hashlib import sha256
import json
import sqlite3
import numpy as np
from local_storage import database_path, connect_database, StorageError
from local_retrieval.models import fingerprint
from .schemas import VisualError, VisualIdentity, VisualHit

APP_ID = 0x56524731
SCHEMA = 1

def vector_bytes(vector, dimension):
    try:
        values = np.asarray(vector, dtype="<f4")
        if values.shape != (dimension,) or not np.isfinite(values).all(): raise ValueError()
        norm = float(np.linalg.norm(values))
        if norm < 1e-8: raise ValueError()
        return (values / norm).astype("<f4").tobytes()
    except (ValueError, TypeError, OverflowError):
        raise VisualError("visual_embedding_failed") from None

def vector_read(blob, dimension):
    if not isinstance(blob, bytes) or len(blob) != dimension * 4:
        raise VisualError("visual_index_corrupt")
    v = np.frombuffer(blob, dtype="<f4")
    if not np.isfinite(v).all() or abs(float(np.linalg.norm(v)) - 1) > .001:
        raise VisualError("visual_index_corrupt")
    return v

def row_digest(document_id, page, rendered_id, blob):
    return fingerprint([document_id, page, rendered_id, sha256(blob).hexdigest()])

class VisualIndex:
    def __init__(self, path=None, max_pages=10000):
        try: self.path = database_path(path, "visual_retrieval.sqlite3")
        except StorageError: raise VisualError("visual_invalid_request") from None
        self.max_pages = max_pages
    @contextmanager
    def connection(self, write=False):
        c = None
        try:
            if not write and not self.path.exists():
                raise VisualError("visual_index_unavailable")
            c = connect_database(self.path, write=write)
            c.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            pair = (c.execute("PRAGMA application_id").fetchone()[0], c.execute("PRAGMA user_version").fetchone()[0])
            if pair == (0, 0) and write and not c.execute("SELECT name FROM sqlite_master").fetchone():
                c.execute("CREATE TABLE metadata (singleton INTEGER PRIMARY KEY CHECK(singleton=1), identity TEXT NOT NULL, documents TEXT NOT NULL, generation TEXT NOT NULL)")
                c.execute("CREATE TABLE pages (document_id TEXT NOT NULL, page INTEGER NOT NULL, rendered_id TEXT NOT NULL, vector BLOB NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(document_id,page))")
                c.execute(f"PRAGMA application_id={APP_ID}")
                c.execute(f"PRAGMA user_version={SCHEMA}")
            elif pair != (APP_ID, SCHEMA): raise VisualError("visual_index_incompatible")
            if c.execute("PRAGMA quick_check").fetchone()[0] != "ok": raise VisualError("visual_index_corrupt")
            yield c
            if write: c.commit()
        except VisualError: raise
        except (sqlite3.Error, OSError, ValueError, TypeError, StorageError):
            raise VisualError("visual_index_write_failed" if write else "visual_index_corrupt") from None
        finally:
            if c is not None: c.close()
    def read(self, expected=None):
        with self.connection() as c:
            meta = c.execute("SELECT identity,documents,generation FROM metadata WHERE singleton=1").fetchone()
            if not meta: raise VisualError("visual_index_corrupt")
            try:
                identity = VisualIdentity.model_validate_json(meta[0])
                documents = json.loads(meta[1])
                if not isinstance(documents, dict) or len(documents) > self.max_pages: raise ValueError()
                for key, doc in documents.items():
                    if set(doc) != {"fingerprint", "page_count", "source_hash"}: raise ValueError()
                    VisualHit(document_id=key, source_hash=doc["source_hash"], prepared_identity=doc["fingerprint"],
                              page=doc["page_count"], rendered_page_id="0"*64, rank=1, score=0.)
                    if doc["source_hash"] != key or not 1 <= doc["page_count"] <= self.max_pages: raise ValueError()
                if sum(doc["page_count"] for doc in documents.values()) > self.max_pages: raise ValueError()
                if c.execute("SELECT 1 FROM pages WHERE length(vector) != ? OR length(rendered_id) != 64 OR page < 1 LIMIT 1", (identity.dimension * 4,)).fetchone():
                    raise VisualError("visual_index_corrupt")
                rows = c.execute("SELECT document_id,page,rendered_id,vector,digest FROM pages ORDER BY document_id,page LIMIT ?", (self.max_pages + 1,)).fetchall()
                if len(rows) > self.max_pages: raise VisualError("visual_capacity_exceeded")
                grouped = {key: [] for key in documents}
                for key, page, rendered, blob, digest in rows:
                    VisualHit(document_id=key, source_hash=key, prepared_identity=documents[key]["fingerprint"],
                              page=page, rendered_page_id=rendered, rank=1, score=0.)
                    vector_read(blob, identity.dimension)
                    if row_digest(key, page, rendered, blob) != digest: raise ValueError()
                    grouped[key].append(page)
                if any(pages != list(range(1, documents[key]["page_count"] + 1)) for key, pages in grouped.items()): raise ValueError()
                generation = fingerprint([identity.model_dump(), documents, [row[4] for row in rows]])
                if generation != meta[2]: raise ValueError()
            except (ValueError, TypeError, KeyError):
                raise VisualError("visual_index_corrupt") from None
            if expected is not None and identity != expected:
                raise VisualError("visual_index_incompatible")
            return identity, documents, rows, generation
    def rebuild_generation(self):
        # Explicit rebuild can repair damaged rows in this recognized schema.
        with self.connection() as c:
            row = c.execute("SELECT generation FROM metadata WHERE singleton=1").fetchone()
            return row[0] if row else None

    def write(self, identity, documents, rows, before):
        if len(rows) > self.max_pages: raise VisualError("visual_capacity_exceeded")
        rows = sorted(rows, key=lambda r: (r[0], r[1]))
        generation = fingerprint([identity.model_dump(), documents, [row[4] for row in rows]])
        with self.connection(write=True) as c:
            current = c.execute("SELECT generation FROM metadata WHERE singleton=1").fetchone()
            if (current[0] if current else None) != before: raise VisualError("visual_source_changed")
            c.execute("DELETE FROM pages")
            c.executemany("INSERT INTO pages VALUES (?,?,?,?,?)", rows)
            c.execute("INSERT OR REPLACE INTO metadata VALUES (1,?,?,?)", (identity.model_dump_json(), json.dumps(documents, sort_keys=True), generation))
        return generation
