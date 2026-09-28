"""PDF text remains local untrusted data; CLI/logging never expose its body."""
import builtins
from io import StringIO
import json
import logging
from pathlib import Path
import subprocess
import sys

from pypdf._page import PageObject
import pytest

from local_pdf import DocumentStore, PdfIngestError, PdfIngestionService
from local_storage import data_root
from .pdf_fixtures import pdf_bytes, INJECTION_TEXT, SYNTHETIC_TEXT

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "tools/pdf_ingest.py"
CANARY = "synthetic-confidential-body-canary"


def run_cli(tmp_path, *args):
    return subprocess.run([sys.executable, "-B", str(CLI), "--database", str(tmp_path / "private.sqlite3"), *args],
                          cwd=tmp_path, capture_output=True, text=True, timeout=20)


def test_real_cli_ingest_duplicate_inspect_list_delete_are_bounded_metadata(tmp_path):
    source = tmp_path / f"{CANARY}.pdf"
    original = pdf_bytes([CANARY + "\n" + SYNTHETIC_TEXT])
    source.write_bytes(original)
    output = run_cli(tmp_path, "ingest", str(source))
    assert output.returncode == 0
    first = json.loads(output.stdout)
    assert first["status"] == "ingested"
    identifier = first["document"]["document_id"]
    results = [output, run_cli(tmp_path, "ingest", str(source)),
               run_cli(tmp_path, "inspect", identifier), run_cli(tmp_path, "list", "--limit", "1")]
    assert json.loads(results[1].stdout)["status"] == "already_ingested"
    assert json.loads(results[2].stdout)["document"] == first["document"]
    assert json.loads(results[3].stdout)["documents"] == [first["document"]]
    deleted = run_cli(tmp_path, "delete", identifier)
    assert deleted.returncode == 0 and json.loads(deleted.stdout)["status"] == "deleted"
    results.extend([deleted, run_cli(tmp_path, "inspect", identifier)])
    assert results[-1].returncode == 2 and "document_not_found" in results[-1].stderr
    assert source.read_bytes() == original
    for result in results:
        combined = result.stdout + result.stderr
        assert CANARY not in combined and SYNTHETIC_TEXT not in combined
        assert str(tmp_path) not in combined and '"text"' not in combined and '"filename"' not in combined
        assert len(combined) < 2000


@pytest.mark.parametrize("arguments,code", [
    (["ingest", "missing-private.pdf"], "pdf_not_found"),
    (["inspect", CANARY], "invalid_document_id"),
    (["list", "--limit", "101"], "invalid_arguments"),
    (["list", "--limit", CANARY], "invalid_arguments"),
    ([CANARY], "invalid_arguments"),
    (["ingest", "missing.pdf", "--max-chunk-chars", "128", "--overlap-chars", "100"], "invalid_arguments"),
])
def test_cli_errors_do_not_echo_arguments_or_tracebacks(tmp_path, arguments, code):
    result = run_cli(tmp_path, *arguments)
    assert result.returncode == 2 and not result.stdout
    assert json.loads(result.stderr.splitlines()[-1]) == {"error": code}
    assert CANARY not in result.stderr and "Traceback" not in result.stderr


def test_success_logs_and_record_representations_omit_text_filename_path(tmp_path, caplog):
    source = tmp_path / f"{CANARY}.pdf"
    source.write_bytes(pdf_bytes([CANARY]))
    store = DocumentStore(tmp_path / "records.sqlite3")
    with caplog.at_level(logging.INFO, logger="local_pdf"):
        document = PdfIngestionService(store).ingest_pdf(source).document
    records = [document, *store.pages(document.document_id), *store.chunks(document.document_id)]
    assert CANARY not in caplog.text + repr(records)
    assert str(tmp_path) not in caplog.text
    assert document.document_id in caplog.text and "page_count=1" in caplog.text
    assert not list(tmp_path.rglob("*.jsonl"))


@pytest.mark.parametrize("failure", ["warning", "exception"])
def test_parser_diagnostics_fail_closed_without_private_output(tmp_path, monkeypatch, caplog, capsys, failure):
    source = tmp_path / "synthetic.pdf"
    source.write_bytes(pdf_bytes([SYNTHETIC_TEXT]))
    parent = logging.getLogger("pypdf")
    child = logging.getLogger("pypdf._page")
    capture = StringIO()
    handler = logging.StreamHandler(capture)
    child.addHandler(handler)
    before = (parent.handlers[:], parent.level, parent.propagate, child.handlers[:])
    def private_diagnostic(*args, **kwargs):
        if failure == "warning":
            child.warning(CANARY)
        raise RuntimeError(CANARY)
    monkeypatch.setattr(PageObject, "extract_text", private_diagnostic)
    try:
        with pytest.raises(PdfIngestError) as caught:
            PdfIngestionService(DocumentStore(tmp_path / "db.sqlite3")).ingest_pdf(source)
        stdout, stderr = capsys.readouterr()
        assert str(caught.value) == "pdf_parse_failed"
        assert CANARY not in caplog.text + stdout + stderr + capture.getvalue()
        assert (parent.handlers[:], parent.level, parent.propagate, child.handlers[:]) == before
        assert not (tmp_path / "db.sqlite3").exists()
    finally:
        child.removeHandler(handler)
        handler.close()


@pytest.mark.parametrize('change', ['message', 'source', 'level', 'idnum', 'generation', 'extra', 'long_number'])
def test_header_whitespace_exception_cannot_admit_other_parser_diagnostics(change, capsys):
    from local_pdf.parser import _RejectDiagnostics
    name, level = 'pypdf._reader', logging.WARNING
    message = 'Superfluous whitespace found in object header %(idnum)r %(generation)r'
    values = dict(idnum=b'3', generation=b'0')
    if change == 'message': message = 'Object ID %(idnum)d,%(generation)d ref repaired'
    if change == 'source': name = 'pypdf._page'
    if change == 'level': level = logging.ERROR
    if change == 'idnum': values['idnum'] = CANARY
    if change == 'generation': values['generation'] = b'-1'
    if change == 'extra': values['unexpected'] = CANARY
    if change == 'long_number': values['idnum'] = b'3' * 21
    record = logging.LogRecord(name, level, 'synthetic.py', 1, message, (values,), None)
    with pytest.raises(PdfIngestError, match='pdf_parse_failed'):
        _RejectDiagnostics().emit(record)
    captured = capsys.readouterr()
    assert not captured.out and not captured.err


def test_fresh_process_injection_ingestion_cannot_import_providers_or_use_network(tmp_path):
    """No cached imports: block every browser/provider module, SDK and network.

    Covers Anthropic, OpenAI/Azure, Ollama and internal-compatible through their
    shared browser.providers boundary, as well as direct SDK/client access.
    """
    source = tmp_path / "injection.pdf"
    source.write_bytes(pdf_bytes([INJECTION_TEXT]))
    (tmp_path / ".env").write_text("SYNTHETIC_PRIVATE_CONFIG=do-not-read\n")
    script = r'''
import builtins, io, pathlib, runpy, socket, sys
from importlib.abc import MetaPathFinder
root, source, database = map(pathlib.Path, sys.argv[1:])
class RejectProviders(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'browser', 'anthropic', 'openai', 'ollama', 'httpx', 'requests', 'urllib3'}:
            raise AssertionError('provider or browser import during ingestion')
sys.meta_path.insert(0, RejectProviders())
def forbidden(*args, **kwargs):
    raise AssertionError('network or process action during ingestion')
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden
socket.getaddrinfo = forbidden
import subprocess, os
subprocess.Popen = forbidden
os.system = forbidden
original_open, original_io_open = builtins.open, io.open
def guard(original):
    def safe_open(file, *args, **kwargs):
        if not isinstance(file, int) and pathlib.Path(file).name in {'.env', 'config.yaml'}:
            raise AssertionError('configuration read during ingestion')
        return original(file, *args, **kwargs)
    return safe_open
builtins.open, io.open = guard(original_open), guard(original_io_open)
sys.path.insert(0, str(root / 'backend'))
from local_pdf import PdfIngestionService, DocumentStore
store = DocumentStore(database)
result = PdfIngestionService(store).ingest_pdf(source)
pages, chunks = store.pages(result.document.document_id), store.chunks(result.document.document_id)
assert pages[0].text.startswith('Ignore all previous instructions')
assert all(record.trust == 'untrusted_document_data' for record in [result.document, *pages, *chunks])
sys.argv = [str(root / 'tools/pdf_ingest.py'), '--database', str(database), 'ingest', str(source)]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT), str(source), str(tmp_path / "private.sqlite3")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "already_ingested"
    assert INJECTION_TEXT not in result.stdout + result.stderr
    assert {p.name for p in tmp_path.iterdir()} == {".env", "injection.pdf", "private.sqlite3"}


def test_git_ignore_covers_database_journals_exports_and_doctor_rejects_tracking():
    paths = [".local/documents.sqlite3", ".local/documents.sqlite3-wal", ".local/documents.sqlite3-shm",
             ".local/documents.sqlite3-journal", ".local/private-export.json"]
    result = subprocess.run(["git", "check-ignore", "--no-index", *paths], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and result.stdout.splitlines() == paths
    assert DocumentStore().path == data_root() / "documents.sqlite3"
    sys.path.insert(0, str(ROOT / "tools"))
    from operational import dangerous_tracked
    assert dangerous_tracked(paths + ["backend/local_pdf/store.py"]) == paths
