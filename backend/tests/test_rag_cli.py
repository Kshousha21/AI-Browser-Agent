"""Real CLI subprocess acceptance: synthetic local PDFs and loopback SDK wire."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from browser.config import load_provider_settings
from local_retrieval.embedding_identity import model_identity
from .retrieval_fixtures import TestEmbedder
from .rag_fixtures import corpus
from .provider_wire import answer, running_server

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cli_data(tmp_path):
    # Lexical subprocess search requires production index identity, but never
    # consumes these test-only vectors or loads/downloads model weights.
    embedder = TestEmbedder(dimension=384)
    embedder._identity = model_identity()
    retrieval, store, docs = corpus(tmp_path / "corpus", embedder=embedder)
    return retrieval, store, docs


def run_cli(tmp_path, data, endpoint, *extra, question="Lumen 12 W"):
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"provider": {"active": "local"}, "providers": {
        "local": {"type": "ollama", "model": "synthetic", "base_url": endpoint,
                  "auth": {"mode": "none"}, "timeout_seconds": 2, "max_retries": 0}}}), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("BROWSER_")}
    return subprocess.run([sys.executable, "-B", str(ROOT / "tools/local_rag.py"), "ask", question,
        "--config", str(config), "--database", str(data[1].path), "--index-database", str(data[0].index.path),
        "--device", "cpu", "--mode", "lexical", "--top-k", "1", *extra],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)


def test_real_cli_bounded_answer_citation_and_no_context_dump(tmp_path, cli_data):
    with running_server() as server:
        server.replies.append(answer("ollama", text=json.dumps({"status": "answered", "answer": "Lumen uses 12 W.",
            "citations": [{"source_id": "S1", "evidence": "12 W"}]})))
        result = run_cli(tmp_path, cli_data, server.url)
    assert result.returncode == 0 and not result.stderr
    output = json.loads(result.stdout)
    assert output["answer"] == "Lumen uses 12 W." and len(output["citations"]) == 1
    assert output["citations"][0]["document_id"] == cli_data[2]["lumen"].document_id
    assert output["citations"][0]["page_start"] == 1
    assert all("evidence" not in citation for citation in output["citations"])
    for forbidden in ("copper plate", '"evidence":', "synthetic-secret", str(tmp_path), ".pdf", "BEGIN_RAG"):
        assert forbidden not in result.stdout + result.stderr


@pytest.mark.parametrize("kind,code", [("remote", "rag_remote_context_denied"), ("invalid", "rag_invalid_answer"), ("badargs", "rag_invalid_request"), ("config", "rag_configuration_error")])
def test_real_cli_typed_failures_exit_codes_and_privacy(tmp_path, cli_data, kind, code):
    with running_server() as server:
        server.replies.extend([answer("ollama", text="private-raw-response") for _ in range(2)])
        endpoint = "https://not-contacted.invalid" if kind == "remote" else server.url
        if kind == "config":
            endpoint = "http://private-user:private-secret@127.0.0.1"
        extra = ["--top-k", "11"] if kind == "badargs" else []
        result = run_cli(tmp_path, cli_data, endpoint, *extra)
        assert len(server.received) == (2 if kind == "invalid" else 0)
    assert result.returncode == 2 and not result.stdout
    assert json.loads(result.stderr) == {"error": code}
    assert "private-raw-response" not in result.stderr and str(tmp_path) not in result.stderr
    assert "private-secret" not in result.stdout + result.stderr


def test_real_cli_zero_results_does_not_contact_remote(tmp_path, cli_data):
    result = run_cli(tmp_path, cli_data, "https://not-contacted.invalid", question="zzunfindableqq")
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["status"] == "insufficient_evidence" and output["attempts"] == 0


def test_provider_profile_override_uses_existing_layering_without_browser_auth(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"provider": {"active": "missing"}, "providers": {
        "chosen": {"type": "ollama", "model": "synthetic", "base_url_env": "RAG_TEST_ENDPOINT",
                   "auth": {"mode": "none"}}}}), encoding="utf-8")
    (tmp_path / ".env").write_text("RAG_TEST_ENDPOINT=http://127.0.0.1:1111\n", encoding="utf-8")
    monkeypatch.setenv("RAG_TEST_ENDPOINT", "http://127.0.0.1:2222")
    monkeypatch.delenv("BROWSER_PROVIDER__ACTIVE", raising=False)
    settings = load_provider_settings(config, "chosen")
    assert settings.active == "chosen" and settings.base_url == "http://127.0.0.1:2222"


@pytest.mark.parametrize("extra", [[], ["--local-temp"]])
def test_real_acceptance_not_claimed_without_explicit_run(extra):
    result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/rag_acceptance.py"), *extra],
                            cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 3 and not result.stderr
    assert json.loads(result.stdout)["status"] == "REAL RAG PROVIDER ACCEPTANCE NOT VERIFIED"
