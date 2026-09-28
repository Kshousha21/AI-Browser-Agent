"""GPU-independent selection, representation compatibility and failure isolation."""
import hashlib
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from local_retrieval import RetrievalError, RetrievalService, EmbeddingIdentity
from local_retrieval.artifacts import artifact_manifest
from local_retrieval.embedding_identity import model_identity, compatible_identity, REPRESENTATION_FIELDS
from local_retrieval.execution import resolve_execution_target, target, capabilities
from local_retrieval.index_store import generation
from local_retrieval.local_model import OnnxEmbedder
from .retrieval_fixtures import fixture_service, TestEmbedder
from .test_retrieval_model import FakeSession, tokenizer
from .test_retrieval_isolation import load_tool

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
DML = "DmlExecutionProvider"


@pytest.mark.parametrize("providers,invalid,selected", [
    ((CUDA, DML, CPU), (), "cuda"), ((DML, CPU), (), "directml"),
    ((CPU,), (), "cpu"), ((CUDA, DML, CPU), ("cuda",), "directml"),
    ((CUDA, DML, CPU), ("cuda", "directml"), "cpu"),
    ((CUDA, CPU), ("cuda",), "cpu"),
])
def test_auto_priority_requires_successful_probe(providers, invalid, selected):
    probes = []
    def probe(candidate):
        probes.append(candidate.selected_device)
        if candidate.selected_device in invalid:
            raise RuntimeError("PRIVATE_DRIVER_DUMP")
    result = resolve_execution_target("auto", probe, providers=providers)
    assert result.selected.selected_device == selected
    assert "cpu" not in probes
    assert "PRIVATE" not in json.dumps(result.metadata())
    if selected != "cpu":
        assert probes[-1] == selected


@pytest.mark.parametrize("mode", ["cuda", "directml"])
@pytest.mark.parametrize("failure", ["absent", "probe", "missing_model"])
def test_explicit_gpu_never_silently_falls_back(mode, failure):
    calls = []
    def probe(candidate):
        calls.append(candidate.selected_device)
        raise RetrievalError("embedding_model_unavailable" if failure == "missing_model" else "embedding_failed")
    providers = (CPU,) if failure == "absent" else (CUDA, DML, CPU)
    with pytest.raises(RetrievalError, match="^embedding_accelerator_unavailable$"):
        resolve_execution_target(mode, probe, providers=providers)
    assert "cpu" not in calls


def test_cpu_is_forced_without_any_gpu_probe():
    result = resolve_execution_target("cpu", lambda _: pytest.fail("CPU invoked GPU probe"), providers=(CUDA, DML, CPU))
    assert result.selected == target("cpu")
    info = artifact_manifest()
    assert info["files"][0]["sha256"] == "98a01d88b7de996cdea58c32ca71208c09968d143798814b2ea09d3439dc334f"
    assert info["files"][0]["bytes"] == 118453870
    assert info["embedding_pipeline"] == "masked-mean-disjoint-128-weighted-l2-v1"


@pytest.mark.parametrize("mode", ["auto", "cuda"])
def test_missing_gpu_model_never_downloads(tmp_path, monkeypatch, mode):
    monkeypatch.setattr("local_retrieval.execution.available_providers", lambda: (CUDA, CPU))
    guard = load_tool("retrieval_offline_guard")
    with guard.offline_guard() as attempts:
        model = OnnxEmbedder(device=mode, gpu_directory=tmp_path / "absent")
        if mode == "auto":
            assert model.execution_metadata()["selected_device"] == "cpu"
            assert model.execution_metadata()["initialization_fallbacks"][0]["reason"] == "artifact_unavailable"
        else:
            with pytest.raises(RetrievalError, match="^embedding_accelerator_unavailable$"):
                model.execution_metadata()
        assert attempts == []
    assert not (tmp_path / "absent").exists()


def mock_cuda(tmp_path, monkeypatch):
    model = OnnxEmbedder(device="cuda", gpu_directory=tmp_path)
    session = FakeSession()
    monkeypatch.setattr("local_retrieval.execution.available_providers", lambda: (CUDA, CPU))
    monkeypatch.setattr(model, "_construct", lambda selected, **kwargs: (session, tokenizer()))
    return model, session


def test_valid_probe_metadata_and_normalized_vectors(tmp_path, monkeypatch):
    model, session = mock_cuda(tmp_path, monkeypatch)
    output = model.encode(["optical cooling"])
    assert len(output[0]) == 384 and session.calls == 2
    metadata = model.execution_metadata()
    assert metadata["requested_device"] == metadata["selected_device"] == "cuda"
    assert metadata["artifact_kind"] == "gpu-o4" and metadata["precision"] == "fp16"
    assert metadata["execution_provider"] == CUDA and metadata["gpu_probe_passed"]
    assert metadata["cpu_ep_fallback_disabled"]


@pytest.mark.parametrize("mode", ["auto", "cuda"])
@pytest.mark.parametrize("existing", [False, True])
def test_mid_index_oom_does_not_switch_device_or_publish_partial_data(tmp_path, monkeypatch, mode, existing):
    model, session = mock_cuda(tmp_path, monkeypatch)
    model.device = mode
    original = session.run
    def fail_later(outputs, inputs):
        if session.calls >= 2:
            raise RuntimeError("CUDA OOM PRIVATE_BODY_OR_DRIVER")
        return original(outputs, inputs)
    monkeypatch.setattr(session, "run", fail_later)
    service, store, documents, _ = fixture_service(tmp_path, {f"pdf{i}": f"optical cooling {i}" for i in range(20)}, embedder=model)
    if existing:
        service.index_document(documents["pdf0"].document_id)
        session.calls = 0
    index_before = service.index.path.read_bytes() if existing else None
    before = store.path.read_bytes()
    with pytest.raises(RetrievalError, match="^embedding_failed$"):
        service.index_all()
    assert (service.index.path.read_bytes() if service.index.path.exists() else None) == index_before
    assert store.path.read_bytes() == before
    assert model.execution_metadata()["selected_device"] == "cuda"


@pytest.mark.parametrize("field,value", [
    ("execution_artifact", "gpu-o4"), ("artifact_sha256", "a" * 64),
    ("precision", "fp16"), ("execution_provider", CUDA),
])
def test_changed_representation_cannot_reuse_an_index(tmp_path, field, value):
    embedder = TestEmbedder()
    embedder._identity = embedder.identity.model_copy(update={"execution_artifact": "cpu-avx2", "artifact_sha256": "b" * 64,
                                                            "precision": "quint8", "execution_provider": CPU})
    service, store, _, _ = fixture_service(tmp_path, embedder=embedder)
    service.index_all()
    before = service.index.path.read_bytes()
    other = TestEmbedder()
    other._identity = embedder.identity.model_copy(update={field: value})
    changed = RetrievalService(other, source_path=store.path, index_path=service.index.path)
    with pytest.raises(RetrievalError, match="^index_model_mismatch$"):
        changed.search("optical")
    assert service.index.path.read_bytes() == before
    assert changed.index_all(rebuild=True)["embedded_chunks"] == 5


def test_original_cpu_index_identity_stays_readable_without_rewrite(tmp_path):
    # Synthetic vectors isolate metadata compatibility; this is not model-quality evidence.
    embedder = TestEmbedder(dimension=384)
    expected = model_identity(target("cpu"))
    embedder._identity = expected
    service, store, _, _ = fixture_service(tmp_path, embedder=embedder)
    service.index_all()
    legacy = EmbeddingIdentity.model_validate(expected.model_dump(exclude=REPRESENTATION_FIELDS))
    with closing(sqlite3.connect(service.index.path)) as connection:
        docs = dict(connection.execute("SELECT document_id,fingerprint FROM indexed_documents").fetchall())
        connection.execute("UPDATE index_metadata SET identity_json=?,generation=?",
                           (json.dumps(legacy.model_dump(exclude_none=True)), generation(legacy, docs)))
        connection.commit()
    before = service.index.path.read_bytes()
    assert compatible_identity(legacy, expected)
    assert not compatible_identity(legacy, model_identity(target("cuda")))
    assert service.search("optical").returned_count > 0
    assert service.index.path.read_bytes() == before


def test_doctor_discovery_never_initializes_or_claims_inference(monkeypatch):
    monkeypatch.setattr("local_retrieval.execution.available_providers", lambda: (DML, CPU))
    monkeypatch.setattr(OnnxEmbedder, "_load", lambda _: pytest.fail("Doctor initialized model"))
    result = capabilities()
    assert result["available_devices"] == ["directml"]
    assert result["probe_performed"] is False
    assert "NOT VERIFIED" in result["status"]


@pytest.mark.parametrize("arguments", [["--device", "cpu", "status"], ["status", "--device", "cpu"]])
def test_cli_device_before_or_after_command(tmp_path, monkeypatch, capsys, arguments):
    cli = load_tool("local_retrieval")
    embedder = TestEmbedder()
    captured = []
    def factory(directory, **options):
        captured.append(options["device"])
        return embedder
    monkeypatch.setattr(cli, "OnnxEmbedder", factory)
    assert cli.main(["--database", str(tmp_path / "source.sqlite3"), "--index-database", str(tmp_path / "index.sqlite3"), *arguments]) == 0
    assert captured == ["cpu"] and "text" not in capsys.readouterr().out
