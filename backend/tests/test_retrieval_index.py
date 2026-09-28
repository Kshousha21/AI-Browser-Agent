"""Index lifecycle, authority, incompatible models/vectors and transaction safety."""
from contextlib import closing
import hashlib
import sqlite3

import numpy as np
import pytest

from local_pdf import PdfIngestionService, IngestionSettings
from local_retrieval import RetrievalService, RetrievalError
from local_retrieval.vectors import normalized
from .retrieval_fixtures import fixture_service, TestEmbedder, ingest_texts


def test_idempotence_generation_and_persistence(tmp_path):
    service, store, _, embedder = fixture_service(tmp_path)
    source_bytes = store.path.read_bytes()
    first = service.index_all()
    assert first["embedded_chunks"] == 5
    embedder.calls.clear()
    again = service.index_all()
    assert again["status"] == "already_indexed" and not embedder.calls
    assert first["index_generation"] == again["index_generation"]
    reopened = RetrievalService(TestEmbedder(), source_path=store.path, index_path=service.index.path)
    assert reopened.search("thermal packaging") == service.search("thermal packaging")
    assert store.path.read_bytes() == source_bytes
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1


@pytest.mark.parametrize("mode", ["lexical", "semantic", "hybrid"])
def test_deleted_document_never_returned_before_sync(tmp_path, mode):
    service, store, documents, _ = fixture_service(tmp_path)
    service.index_all()
    deleted = documents["optics"].document_id
    assert store.delete(deleted)
    result = service.search("xylophotonic9 optical", mode=mode)
    assert all(hit.document_id != deleted for hit in result.results)
    assert result.excluded_stale_documents == 1
    synced = service.sync()
    assert synced["removed_documents"] == 1 and synced["embedded_chunks"] == 0
    with service.index.connection() as connection:
        assert not connection.execute("SELECT 1 FROM indexed_documents WHERE document_id=?", (deleted,)).fetchone()
        assert not connection.execute("SELECT 1 FROM indexed_chunks WHERE document_id=?", (deleted,)).fetchone()
        assert connection.execute("SELECT count(*) FROM chunk_embeddings").fetchone()[0] == 4
        assert not connection.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'xylophotonic9'").fetchone()


def test_changed_chunk_representation_is_excluded_then_rebuilt(tmp_path):
    text = "Synthetic thermal packaging and cooling processors. " * 30
    service, store, documents, embedder = fixture_service(tmp_path, {"long": text})
    service.index_all()
    identifier = documents["long"].document_id
    store.delete(identifier)
    changed = PdfIngestionService(store, IngestionSettings(max_chunk_chars=256, overlap_chars=32)).ingest_pdf(tmp_path / "long.pdf")
    assert changed.document.document_id == identifier
    assert service.search("thermal").returned_count == 0
    assert service.status()["stale_documents"] == 1
    result = service.sync()
    assert result["embedded_chunks"] == changed.document.chunk_count
    assert service.search("thermal").returned_count > 0


def test_remove_only_changes_index_and_sync_restores_source(tmp_path):
    service, store, documents, _ = fixture_service(tmp_path)
    service.index_all()
    identifier = documents["optics"].document_id
    before = store.path.read_bytes()
    assert service.remove(identifier)["status"] == "removed"
    assert service.search("xylophotonic9", mode="lexical").returned_count == 0
    assert store.path.read_bytes() == before and store.get(identifier) is not None
    assert service.remove(identifier)["status"] == "not_indexed"
    assert service.sync()["embedded_chunks"] == 1


@pytest.mark.parametrize("dimension,revision", [(32, "changed"), (33, "test-v1")])
def test_model_or_dimension_mismatch_requires_explicit_rebuild(tmp_path, dimension, revision):
    service, store, _, _ = fixture_service(tmp_path)
    service.index_all()
    other = RetrievalService(TestEmbedder(dimension=dimension, revision=revision), source_path=store.path, index_path=service.index.path)
    with pytest.raises(RetrievalError, match="^index_model_mismatch$"):
        other.search("thermal")
    with pytest.raises(RetrievalError, match="^index_model_mismatch$"):
        other.index_all()
    assert other.index_all(rebuild=True)["embedded_chunks"] == 5
    assert other.search("thermal").returned_count > 0


@pytest.mark.parametrize("bad", [[0.0]*32, [float('nan')]*32, [float('inf')]*32, [1.0]*31, [1e309]*32])
def test_invalid_embeddings_fail_before_index_commit(tmp_path, bad, monkeypatch):
    service, _, _, embedder = fixture_service(tmp_path)
    monkeypatch.setattr(embedder, "encode", lambda texts: [bad for _ in texts])
    with pytest.raises(RetrievalError, match="invalid_embedding|embedding_dimension_mismatch"):
        service.index_all()
    assert not service.index.path.exists()


def test_vector_contract_is_float32_normalized_cosine():
    vector = normalized([3.0, 4.0], 2)
    assert vector.dtype == np.float32
    assert vector.tolist() == pytest.approx([0.6, 0.8])
    assert float(vector @ vector) == pytest.approx(1.0)


@pytest.mark.parametrize("existing", [False, True])
def test_index_transaction_rollback_and_retry(tmp_path, monkeypatch, existing):
    service, _, documents, _ = fixture_service(tmp_path)
    if existing:
        service.index_document(documents["optics"].document_id)
    before = service.status()
    insert = service.index.insert_chunk
    def fail_after_insert(connection, chunk, vector):
        insert(connection, chunk, vector)
        raise sqlite3.OperationalError("synthetic-private-chunk-body")
    monkeypatch.setattr(service.index, "insert_chunk", fail_after_insert)
    with pytest.raises(RetrievalError, match="^index_write_failed$"):
        service.sync()
    assert service.status() == before
    monkeypatch.setattr(service.index, "insert_chunk", insert)
    assert service.sync()["embedded_chunks"] == (4 if existing else 5)


@pytest.mark.parametrize("kind", ["bytes", "schema", "vector", "orphan", "fingerprint"])
def test_corrupt_or_unsupported_index_fails_safely(tmp_path, kind):
    service, _, _, _ = fixture_service(tmp_path)
    service.index_all()
    if kind == "bytes":
        service.index.path.write_bytes(b"not a database: synthetic content")
    else:
        with closing(sqlite3.connect(service.index.path)) as connection:
            if kind == "schema":
                connection.execute("PRAGMA user_version=99")
            elif kind == "vector":
                connection.execute("UPDATE chunk_embeddings SET vector=?", (b"bad",))
            elif kind == "orphan":
                connection.execute("DELETE FROM indexed_documents")
            else:
                connection.execute("UPDATE indexed_chunks SET fingerprint=?", ("a"*64,))
            connection.commit()
    with pytest.raises(RetrievalError, match="index_corrupt|unsupported_index"):
        service.search("optical")


def test_authoritative_change_during_embedding_cannot_publish(tmp_path, monkeypatch):
    service, store, documents, embedder = fixture_service(tmp_path)
    original = embedder.encode
    def remove_source(texts):
        result = original(texts)
        store.delete(documents["optics"].document_id)
        return result
    monkeypatch.setattr(embedder, "encode", remove_source)
    with pytest.raises(RetrievalError, match="^source_changed$"):
        service.sync()
    assert not service.index.path.exists()


def test_empty_source_and_empty_pdf_index_without_fake_chunks(tmp_path):
    service, _, _, embedder = fixture_service(tmp_path, {"blank": ""})
    result = service.index_all()
    assert result["embedded_chunks"] == 0 and not embedder.calls
    assert service.search("synthetic").returned_count == 0


def test_source_and_index_must_be_different(tmp_path):
    with pytest.raises(RetrievalError, match="^invalid_local_path$"):
        RetrievalService(TestEmbedder(), source_path=tmp_path/"same.sqlite3", index_path=tmp_path/"same.sqlite3")
