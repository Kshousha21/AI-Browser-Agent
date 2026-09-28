"""Phase 4C extraction, identity, limits and atomic persistence acceptance."""
import hashlib
from contextlib import closing
from io import BytesIO
from pathlib import Path
import sqlite3

from pypdf import PdfReader
from pypdf._page import PageObject
import pytest

from local_pdf import DocumentStore, IngestionSettings, PdfIngestError, PdfIngestionService
from local_pdf.store import APPLICATION_ID, SCHEMA_VERSION, ROOT
from .pdf_fixtures import pdf_bytes, SYNTHETIC_TEXT


@pytest.fixture
def store(tmp_path):
    return DocumentStore(tmp_path / "private" / "documents.sqlite3")


@pytest.mark.parametrize('wrong_xref', [False, True])
def test_object_header_whitespace_keeps_strict_reference_validation(tmp_path, store, wrong_xref, caplog):
    from .pdf_fixtures import spaced_object_headers
    data = spaced_object_headers(pdf_bytes([SYNTHETIC_TEXT]), wrong_xref=wrong_xref)
    path = tmp_path / 'synthetic-whitespace.pdf'
    path.write_bytes(data)
    if wrong_xref:
        with pytest.raises(PdfIngestError, match='pdf_parse_failed'):
            PdfIngestionService(store).ingest_pdf(path)
        assert not store.path.exists()
    else:
        result = PdfIngestionService(store).ingest_pdf(path)
        assert result.document.document_id == hashlib.sha256(data).hexdigest()
        assert store.pages(result.document.document_id)[0].text == SYNTHETIC_TEXT
    assert 'Superfluous whitespace' not in caplog.text and 'ref repaired' not in caplog.text


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "invented.pdf"
    path.write_bytes(pdf_bytes([SYNTHETIC_TEXT]))
    return path


def counts(store):
    if not store.path.exists():
        return (0, 0, 0)
    with closing(sqlite3.connect(store.path)) as connection:
        if not connection.execute("SELECT name FROM sqlite_master WHERE name='documents'").fetchone():
            return (0, 0, 0)
        return tuple(connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                     for name in ("documents", "pages", "chunks"))


def test_one_page_persistent_metadata_and_exact_bytes_identity(source, store):
    result = PdfIngestionService(store).ingest_pdf(source)
    document = result.document
    assert result.status == "ingested"
    assert document.document_id == document.content_hash == hashlib.sha256(source.read_bytes()).hexdigest()
    assert document.filename == source.name and document.file_size == source.stat().st_size
    assert document.page_count == document.chunk_count == 1
    assert document.source_type == "local_pdf" and document.parser_name == "pypdf"
    assert document.parser_version == "6.17.0" and document.ingested_at.endswith("+00:00")
    reopened = DocumentStore(store.path)
    assert reopened.get(document.document_id) == document
    assert reopened.pages(document.document_id)[0].text == SYNTHETIC_TEXT
    assert reopened.chunks(document.document_id)[0].text == SYNTHETIC_TEXT
    assert counts(reopened) == (1, 1, 1)
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert connection.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID
        columns = [row[1] for row in connection.execute("PRAGMA table_info(documents)")]
    assert not any("path" in column or "password" in column or "credential" in column for column in columns)


def test_multiple_pages_preserve_empty_page_and_numbering(tmp_path, store):
    path = tmp_path / "pages.PDF"
    path.write_bytes(pdf_bytes(["Market Example A", "", "Technology Example B"]))
    document = PdfIngestionService(store).ingest_pdf(path).document
    pages = store.pages(document.document_id)
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert [p.text_status for p in pages] == ["text", "empty", "text"]
    assert pages[1].text == "" and document.page_count == 3
    assert [c.page_start for c in store.chunks(document.document_id)] == [1, 3]
    assert all(p.document_id == document.document_id for p in pages)


def test_all_empty_pdf_is_valid_without_invented_chunks(tmp_path, store):
    path = tmp_path / "blank.pdf"
    path.write_bytes(pdf_bytes(["", ""]))
    document = PdfIngestionService(store).ingest_pdf(path).document
    assert document.extracted_chars == document.chunk_count == 0
    assert counts(store) == (1, 2, 0)


def test_duplicate_and_renamed_bytes_keep_existing_representation(source, store, monkeypatch):
    first = PdfIngestionService(store).ingest_pdf(source)
    renamed = source.with_name("renamed.pdf")
    renamed.write_bytes(source.read_bytes())
    def forbidden(*args, **kwargs):
        pytest.fail("duplicate must not extract again")
    monkeypatch.setattr("local_pdf.service.extract_pages", forbidden)
    changed_settings = IngestionSettings(max_chunk_chars=256, overlap_chars=32)
    service = PdfIngestionService(store, changed_settings)
    for path in (source, renamed):
        again = service.ingest_pdf(path)
        assert again.status == "already_ingested" and again.document == first.document
    assert counts(store) == (1, 1, 1)
    # The transaction also rechecks duplicates (a caller can race the early read).
    assert store.put(first.document, [], []).status == "already_ingested"


def test_changed_bytes_same_filename_are_a_new_document(source, store):
    service = PdfIngestionService(store)
    first = service.ingest_pdf(source)
    source.write_bytes(pdf_bytes(["Different invented example"]))
    second = service.ingest_pdf(source)
    assert first.document.document_id != second.document.document_id
    assert counts(store) == (2, 2, 2)


@pytest.mark.parametrize("kind,code", [
    ("missing", "pdf_not_found"), ("directory", "invalid_pdf"),
    ("extension", "invalid_pdf"), ("fake", "invalid_pdf"),
    ("empty", "invalid_pdf"), ("truncated", "pdf_parse_failed"),
    ("malformed", "pdf_parse_failed"), ("zero_pages", "invalid_pdf"),
])
def test_invalid_inputs_fail_without_rows(tmp_path, store, kind, code):
    path = tmp_path / "input.pdf"
    if kind == "directory":
        path.mkdir()
    elif kind == "extension":
        path = path.with_suffix(".txt")
        path.write_bytes(pdf_bytes(["Synthetic"]))
    elif kind == "fake":
        path.write_bytes(b"This is not a PDF.")
    elif kind == "empty":
        path.write_bytes(b"")
    elif kind == "truncated":
        path.write_bytes(pdf_bytes(["Synthetic"])[:-12])
    elif kind == "malformed":
        path.write_bytes(b"%PDF-1.7\nnot objects\n%%EOF")
    elif kind == "zero_pages":
        path.write_bytes(pdf_bytes([]))
    with pytest.raises(PdfIngestError, match=f"^{code}$"):
        PdfIngestionService(store).ingest_pdf(path)
    assert counts(store) == (0, 0, 0) and not store.path.exists()


@pytest.mark.parametrize("algorithm", ["RC4-128", "AES-128", "AES-256"])
@pytest.mark.parametrize("password", ["", "synthetic-opening-password"])
def test_encrypted_pdf_empty_opening_password_only(tmp_path, store, algorithm, password, monkeypatch, caplog):
    from pypdf._encryption import Encryption
    from .pdf_fixtures import encrypted_pdf_bytes
    path = tmp_path / "encrypted.pdf"
    data = encrypted_pdf_bytes([SYNTHETIC_TEXT, "Second synthetic page"],
                               password=password, algorithm=algorithm)
    path.write_bytes(data)
    attempts = []
    verify = Encryption.verify
    def track(self, candidate, **kwargs):
        attempts.append(candidate)
        return verify(self, candidate, **kwargs)
    monkeypatch.setattr(Encryption, "verify", track)
    if password:
        with pytest.raises(PdfIngestError, match="^pdf_encrypted$"):
            PdfIngestionService(store).ingest_pdf(path)
        assert not store.path.exists()
    else:
        result = PdfIngestionService(store).ingest_pdf(path)
        assert result.document.document_id == hashlib.sha256(data).hexdigest()
        assert result.document.page_count == 2
        assert store.pages(result.document.document_id)[0].text == SYNTHETIC_TEXT
    assert attempts == [b""]
    assert path.read_bytes() == data
    assert SYNTHETIC_TEXT not in caplog.text and "synthetic-opening-password" not in caplog.text


def test_encrypted_pdf_extraction_permission_required(tmp_path, store):
    from .pdf_fixtures import encrypted_pdf_bytes
    path = tmp_path / "restricted.pdf"
    path.write_bytes(encrypted_pdf_bytes([SYNTHETIC_TEXT], allow_extraction=False))
    with pytest.raises(PdfIngestError, match="^pdf_extraction_not_allowed$"):
        PdfIngestionService(store).ingest_pdf(path)
    assert not store.path.exists()


def test_missing_crypto_dependency_reports_safe_actionable_error(tmp_path, store, monkeypatch, caplog):
    from pypdf.errors import DependencyError
    from .pdf_fixtures import encrypted_pdf_bytes
    path = tmp_path / "encrypted.pdf"
    path.write_bytes(encrypted_pdf_bytes([SYNTHETIC_TEXT]))
    def missing(*args, **kwargs):
        raise DependencyError("synthetic-private-dependency-detail")
    monkeypatch.setattr(PdfReader, "_handle_encryption", missing)
    with pytest.raises(PdfIngestError, match="^pdf_crypto_unavailable$"):
        PdfIngestionService(store).ingest_pdf(path)
    assert not store.path.exists()
    assert "synthetic-private-dependency-detail" not in caplog.text


def test_invalid_encryption_permissions_are_not_accepted(tmp_path, store, monkeypatch):
    from .pdf_fixtures import encrypted_pdf_bytes
    path = tmp_path / "encrypted.pdf"
    path.write_bytes(encrypted_pdf_bytes([SYNTHETIC_TEXT]))
    monkeypatch.setattr(PdfReader, "are_permissions_valid", property(lambda self: False))
    with pytest.raises(PdfIngestError, match="^pdf_parse_failed$"):
        PdfIngestionService(store).ingest_pdf(path)
    assert not store.path.exists()


@pytest.mark.parametrize("setting,value,code", [
    ("max_file_bytes", 10, "pdf_too_large"), ("max_pages", 1, "pdf_limit_exceeded"),
    ("max_page_stream_bytes", 10, "pdf_limit_exceeded"),
    ("max_page_chars", 5, "pdf_limit_exceeded"), ("max_total_chars", 5, "pdf_limit_exceeded"),
    ("max_chunks", 1, "pdf_limit_exceeded"),
])
def test_resource_limits_fail_before_storage(tmp_path, store, setting, value, code):
    source = tmp_path / "bounded.pdf"
    source.write_bytes(pdf_bytes([SYNTHETIC_TEXT, SYNTHETIC_TEXT]))
    settings = IngestionSettings(**{setting: value})
    with pytest.raises(PdfIngestError, match=f"^{code}$"):
        PdfIngestionService(store, settings).ingest_pdf(source)
    assert not store.path.exists()


def test_second_page_extraction_failure_keeps_existing_store_atomic(source, store, tmp_path, monkeypatch):
    original = PdfIngestionService(store).ingest_pdf(source)
    other = tmp_path / "fails.pdf"
    other.write_bytes(pdf_bytes(["First synthetic page", "Second synthetic page"]))
    extract = PageObject.extract_text
    calls = []
    def fail_second(page, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic-private-body")
        return extract(page, *args, **kwargs)
    monkeypatch.setattr(PageObject, "extract_text", fail_second)
    with pytest.raises(PdfIngestError, match="^pdf_parse_failed$"):
        PdfIngestionService(store).ingest_pdf(other)
    assert len(calls) == 2 and counts(store) == (1, 1, 1)
    assert store.get(original.document.document_id) == original.document
    assert store.get(hashlib.sha256(other.read_bytes()).hexdigest()) is None


@pytest.mark.parametrize("existing", [False, True])
def test_storage_failure_rolls_back_documents_pages_and_chunks(source, store, monkeypatch, existing):
    service = PdfIngestionService(store)
    if existing:
        service.ingest_pdf(source)
    source.write_bytes(pdf_bytes(["Another synthetic example"]))
    insert = store._insert_records
    def fail_after_chunk(connection, table, records):
        insert(connection, table, records)
        if table == "chunks":
            raise sqlite3.OperationalError("synthetic-private-storage-detail")
    monkeypatch.setattr(store, "_insert_records", fail_after_chunk)
    with pytest.raises(PdfIngestError, match="^storage_failed$"):
        service.ingest_pdf(source)
    assert counts(store) == ((1, 1, 1) if existing else (0, 0, 0))
    identifier = hashlib.sha256(source.read_bytes()).hexdigest()
    assert store.get(identifier) is None
    assert store.pages(identifier) == [] and store.chunks(identifier) == []
    assert len(store.list_documents()) == int(existing)
    monkeypatch.setattr(store, "_insert_records", insert)
    assert service.ingest_pdf(source).status == "ingested"
    assert counts(store) == ((2, 2, 2) if existing else (1, 1, 1))


def test_delete_cascades_and_retains_original_pdf(source, store):
    original = source.read_bytes()
    document = PdfIngestionService(store).ingest_pdf(source).document
    assert store.delete(document.document_id)
    assert counts(store) == (0, 0, 0) and store.get(document.document_id) is None
    assert not store.delete(document.document_id)
    assert source.read_bytes() == original


def test_delete_failure_rolls_back_cascade(source, store):
    document = PdfIngestionService(store).ingest_pdf(source).document
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute("CREATE TRIGGER reject_delete BEFORE DELETE ON chunks BEGIN SELECT RAISE(ABORT, 'synthetic'); END")
    with pytest.raises(PdfIngestError, match="^storage_failed$"):
        store.delete(document.document_id)
    assert counts(store) == (1, 1, 1)


@pytest.mark.parametrize("kind", ["future", "foreign", "corrupt"])
def test_unrecognized_store_is_not_replaced(source, store, kind):
    store.path.parent.mkdir(parents=True)
    if kind == "corrupt":
        store.path.write_bytes(b"synthetic non-database")
    else:
        with closing(sqlite3.connect(store.path)) as connection:
            connection.execute("PRAGMA user_version=99" if kind == "future" else "CREATE TABLE unrelated(value TEXT)")
    before = store.path.read_bytes()
    with pytest.raises(PdfIngestError):
        PdfIngestionService(store).ingest_pdf(source)
    assert store.path.read_bytes() == before


def test_bounded_listing_and_data_access(source, store):
    document = PdfIngestionService(store).ingest_pdf(source).document
    assert store.list_documents(limit=1) == [document]
    assert store.list_documents(limit=1, offset=1) == []
    assert store.pages(document.document_id, offset=1) == []
    assert store.chunks(document.document_id, offset=1) == []
    for kwargs in ({"limit": 101}, {"limit": 0}, {"offset": -1}, {"limit": True}):
        with pytest.raises(PdfIngestError, match="invalid_arguments"):
            store.list_documents(**kwargs)


def test_reject_store_in_source_tree_and_path_shaped_document_id(store):
    with pytest.raises(PdfIngestError, match="invalid_store_path"):
        DocumentStore(ROOT / "backend" / "documents.sqlite3")
    for operation in (store.get, store.pages, store.chunks, store.delete):
        with pytest.raises(PdfIngestError, match="invalid_document_id"):
            operation("../../invented.pdf")


def test_fixture_generation_is_deterministic_and_sanitized():
    data = pdf_bytes([SYNTHETIC_TEXT, ""])
    assert data == pdf_bytes([SYNTHETIC_TEXT, ""])
    reader = PdfReader(BytesIO(data), strict=True)
    assert len(reader.pages) == 2 and not reader.is_encrypted
    assert reader.pages[0].extract_text() == SYNTHETIC_TEXT
    assert set(reader.metadata) == {"/Producer"}
    for forbidden in (b"/URI", b"/JS", b"/JavaScript", b"/EmbeddedFile", b"/Author", b"/Title", b"private-source.example", b"/Launch"):
        assert forbidden not in data
