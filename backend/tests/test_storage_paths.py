"""Synthetic local storage, network rejection and transaction safety."""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import stat
from unittest.mock import Mock

import pytest
import local_storage as paths
from local_pdf import DocumentStore, PdfIngestionService, PdfIngestError
from local_retrieval import RetrievalService, RetrievalError
from local_retrieval.source import SourceCatalog
from local_retrieval.index_store import IndexStore
from document_workflow.storage import AcquisitionStorage
from .pdf_fixtures import pdf_bytes
from .retrieval_fixtures import TestEmbedder


@pytest.fixture(autouse=True)
def no_operator_override(monkeypatch):
    monkeypatch.delenv("BROWSER_DATA_ROOT", raising=False)


@pytest.mark.parametrize("name", ["ordinary", "spaces in directory", "unicode-\u00fc-\u03bb", "percent%23#"])
@pytest.mark.parametrize("existing", [False, True])
def test_local_filenames_and_readonly_uri_roundtrip(tmp_path, name, existing):
    target = tmp_path / name / "documents.sqlite3"
    store = DocumentStore(target)
    source = tmp_path / "synthetic.pdf"
    source.write_bytes(pdf_bytes(["Synthetic thermal power 12 W."]))
    service = PdfIngestionService(store)
    if existing:
        service.ingest_pdf(source)
    result = service.ingest_pdf(source)
    assert result.status == ("already_ingested" if existing else "ingested")
    assert store.pages(result.document.document_id)[0].text == "Synthetic thermal power 12 W."
    with closing(paths.connect_database(target)) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("CREATE TABLE forbidden(x)")
    uri = paths.readonly_uri(target)
    assert uri.endswith("?mode=ro") and uri.count("?") == 1
    assert "#" not in uri and " " not in uri


def test_local_checkout_keeps_ignored_default_and_does_not_create_it(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    monkeypatch.setattr(paths, "ROOT", repo)
    assert paths.data_root() == repo / ".local"
    assert DocumentStore().path == repo / ".local/documents.sqlite3"
    assert not (repo / ".local").exists()
    monkeypatch.chdir(repo)
    assert DocumentStore(".local/relative.sqlite3").path == repo / ".local/relative.sqlite3"
    with pytest.raises(PdfIngestError, match="^invalid_store_path$"):
        DocumentStore(repo / "backend/documents.sqlite3")


@pytest.mark.parametrize("root_kind", ["mapped", "UNC"])
def test_network_checkout_uses_local_user_runtime_root(tmp_path, monkeypatch, root_kind):
    repo = tmp_path / root_kind
    repo.mkdir()
    old = repo / ".local"
    old.mkdir()
    sentinel = old / "documents.sqlite3"
    sentinel.write_bytes(b"private old store sentinel - must not open or migrate")
    appdata = tmp_path / "user-owned"
    appdata.mkdir()
    monkeypatch.setattr(paths, "ROOT", repo)
    original = paths.filesystem_kind
    monkeypatch.setattr(paths, "filesystem_kind", lambda p:"network" if p.is_relative_to(repo) else original(p))
    monkeypatch.setenv("LOCALAPPDATA", str(appdata))
    monkeypatch.setenv("XDG_DATA_HOME", str(appdata))
    root = paths.data_root()
    assert root.is_relative_to(appdata) and not root.is_relative_to(repo)
    assert DocumentStore().path == root / "documents.sqlite3"
    assert SourceCatalog().path == root / "documents.sqlite3"
    assert IndexStore().path == root / "retrieval.sqlite3"
    assert AcquisitionStorage().root == root / "acquired"
    assert not root.exists() and sentinel.read_bytes().startswith(b"private old")


def test_production_defaults_compose_pdf_retrieval_and_workflow_locally(tmp_path, monkeypatch):
    monkeypatch.setenv("BROWSER_DATA_ROOT", str(tmp_path / "private data"))
    source = tmp_path / "synthetic.pdf"
    source.write_bytes(pdf_bytes(["Synthetic thermal power 12 W."]))
    ingestion = PdfIngestionService()
    doc = ingestion.ingest_pdf(source).document
    retrieval = RetrievalService(TestEmbedder())
    retrieval.index_document(doc.document_id)
    result = retrieval.search("thermal power", document_ids=[doc.document_id])
    assert result.results and all(h.document_id == doc.document_id for h in result.results)
    root = paths.data_root()
    assert ingestion.store.path == retrieval.source.path == root / "documents.sqlite3"
    assert retrieval.index.path == root / "retrieval.sqlite3"
    assert AcquisitionStorage().directory() == root / "acquired"
    assert source.exists()


@pytest.mark.parametrize("constructor", [DocumentStore, SourceCatalog, IndexStore])
def test_explicit_network_rejected_before_any_sqlite_or_mkdir(tmp_path, monkeypatch, constructor):
    target = tmp_path / "network" / "private.sqlite3"
    monkeypatch.setattr(paths, "filesystem_kind", lambda p:"network")
    connect = Mock(side_effect=AssertionError("must not connect"))
    monkeypatch.setattr(sqlite3, "connect", connect)
    with pytest.raises((PdfIngestError, RetrievalError), match="^network_storage_unsupported$"):
        constructor(target)
    connect.assert_not_called()
    assert not target.parent.exists()


@pytest.mark.parametrize("value", ["//server.invalid/share/private.sqlite3", r"\\server.invalid\share\private.sqlite3"])
def test_explicit_unc_fails_before_resolution(value, monkeypatch):
    resolve = Mock(side_effect=AssertionError("must not resolve remote authority"))
    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(paths.StorageError, match="^network_storage_unsupported$"):
        paths.database_path(value, "documents.sqlite3")
    resolve.assert_not_called()


@pytest.mark.parametrize("value", ["", "relative/data", "file:///C:/private/data", "//server.invalid/share/data"])
def test_bad_override_never_falls_back_or_creates(value, tmp_path, monkeypatch):
    monkeypatch.setenv("BROWSER_DATA_ROOT", value)
    with pytest.raises(paths.StorageError):
        paths.data_root()
    assert paths.readiness()["ready"] is False
    assert not list(tmp_path.iterdir())


def test_explicit_local_database_is_not_relocated_by_bad_root(tmp_path, monkeypatch):
    monkeypatch.setenv("BROWSER_DATA_ROOT", "//server.invalid/share/data")
    assert DocumentStore(tmp_path / "chosen.sqlite3").path == tmp_path / "chosen.sqlite3"


@pytest.mark.parametrize("kind", ["same", "alias", "hardlink"])
def test_document_and_index_never_alias_or_overwrite(tmp_path, kind):
    source = tmp_path / "source.sqlite3"
    with closing(sqlite3.connect(source)) as connection:
        connection.execute("CREATE TABLE preserve(x)")
    index = source if kind=="same" else tmp_path / "." / source.name
    if kind=="hardlink":
        index = tmp_path / "alias.sqlite3"
        os.link(source, index)
    before = source.read_bytes()
    with pytest.raises(RetrievalError, match="^invalid_local_path$"):
        RetrievalService(TestEmbedder(), source_path=source, index_path=index)
    assert source.read_bytes() == before


@pytest.mark.parametrize("kind", ["invalid_parent", "readonly", "permission_denied"])
def test_failed_initialization_is_typed_and_non_destructive(tmp_path, monkeypatch, kind):
    target = tmp_path / "parent" / "new.sqlite3"
    if kind == "invalid_parent":
        target.parent.write_bytes(b"preserve parent")
    elif kind == "readonly":
        target.parent.mkdir()
        with closing(sqlite3.connect(target)) as connection:
            connection.execute("CREATE TABLE preserved(x)")
        target.chmod(stat.S_IREAD)
    else:
        monkeypatch.setattr(paths, "connect_database", Mock(side_effect=PermissionError("PRIVATE_PATH")))
        # Modules use a directly imported shared connector.
        import local_pdf.store as module
        monkeypatch.setattr(module, "connect_database", paths.connect_database)
    store = DocumentStore(target)
    before = target.read_bytes() if target.exists() else None
    try:
        with pytest.raises(PdfIngestError):
            with store._connection(write=True):
                pytest.fail("failed initialization must not expose a connection")
        assert (target.read_bytes() if target.exists() else None) == before
    finally:
        if kind == "readonly":
            target.chmod(stat.S_IREAD | stat.S_IWRITE)


@pytest.mark.parametrize("store_type", ["pdf", "index"])
def test_initial_transaction_failure_leaves_empty_recoverable_database(tmp_path, monkeypatch, store_type):
    store = DocumentStore(tmp_path / "db.sqlite3") if store_type=="pdf" else IndexStore(tmp_path / "db.sqlite3")
    connection = store._connection if store_type=="pdf" else store.connection
    with pytest.raises((PdfIngestError, RetrievalError)):
        with connection(write=True) as db:
            db.execute("CREATE TABLE partial_private(x)")
            raise sqlite3.OperationalError("PRIVATE_BODY")
    with closing(sqlite3.connect(store.path)) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("PRAGMA user_version").fetchone()[0] == 0
        assert db.execute("SELECT name FROM sqlite_master").fetchall() == []
    with connection(write=True) as db:
        assert db is not None


def test_source_read_transaction_blocks_writer_commit_and_preserves_consistency(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(pdf_bytes(["Synthetic thermal power 12 W."]))
    store = DocumentStore(tmp_path / "source.sqlite3")
    doc = PdfIngestionService(store).ingest_pdf(source).document
    with SourceCatalog(store.path).snapshot() as snapshot:
        with closing(sqlite3.connect(store.path, timeout=0.01)) as writer:
            writer.execute("DELETE FROM documents")
            with pytest.raises(sqlite3.OperationalError):
                writer.commit()
            writer.rollback()
        assert doc.document_id in snapshot
    assert store.get(doc.document_id) == doc


def test_doctor_storage_readiness_never_opens_database_or_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("BROWSER_DATA_ROOT", str(tmp_path / "not-created"))
    monkeypatch.setattr(sqlite3, "connect", Mock(side_effect=AssertionError("no database inspection")))
    before = list(tmp_path.iterdir())
    result = paths.readiness()
    assert result == {"category":"local","ready":True,"state":"not_created","source":"override"}
    assert str(tmp_path) not in json.dumps(result) and before == list(tmp_path.iterdir())


def test_mapped_checkout_cli_default_uses_explicit_local_root_across_processes(tmp_path):
    import subprocess
    import sys
    root = Path(__file__).absolute().parents[2]
    data = tmp_path / "user data"
    source = tmp_path / "synthetic.pdf"
    source.write_bytes(pdf_bytes(["Synthetic CLI thermal 12 W."]))
    env = {**os.environ, "BROWSER_DATA_ROOT":str(data)}
    outputs = []
    for args in (["ingest",str(source)], ["list"]):
        run = subprocess.run([sys.executable,"-B",str(root/"tools/pdf_ingest.py"),*args],
                             cwd=root,env=env,capture_output=True,text=True,timeout=20)
        assert run.returncode == 0 and not run.stderr
        assert str(data) not in run.stdout and "Synthetic CLI" not in run.stdout
        outputs.append(json.loads(run.stdout))
    assert outputs[0]["document"]["document_id"] == outputs[1]["documents"][0]["document_id"]
    assert (data/"documents.sqlite3").is_file() and source.exists()


def test_document_service_default_storage_no_acquisition_authority_change(tmp_path, monkeypatch):
    import document_workflow.services as module
    from .public_document_support import settings, offer, request, receipt, ALLOWLIST
    from document_workflow.models import PrepareRequest
    monkeypatch.setenv("BROWSER_DATA_ROOT", str(tmp_path/"runtime"))
    monkeypatch.setattr(module,"OnnxEmbedder",lambda:TestEmbedder())
    service = module.DocumentServices(settings().provider)
    selection, _, payload = offer()
    _, [selected] = selection.consume(PrepareRequest(**request(payload)), ALLOWLIST)
    raw = receipt(tmp_path/"downloads","a"*32)
    stages = []
    result = service.prepare(raw,"a"*32,selected,stages.append)
    assert result["index_status"] == "ready"
    assert service.ingestion.store.path == service.retrieval.source.path
    assert service.ingestion.store.path.parent == service.retrieval.index.path.parent == paths.data_root()
    assert service.storage.root == paths.data_root()/"acquired"
    assert Path(raw["filename"]).exists() and not list(service.storage.root.glob("*.pdf"))
    assert stages == ["validating","ingesting","indexing"]
