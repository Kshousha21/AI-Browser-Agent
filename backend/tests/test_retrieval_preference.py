"""Synthetic preference policy, stale identity handling, privacy and offline use."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from local_retrieval import RetrievalError
from local_retrieval.embedding_identity import model_identity
from local_retrieval.execution import resolve_execution_target, target
from local_retrieval import preference as pref
from .test_retrieval_isolation import load_tool, ROOT

CPU, CUDA, DML = "CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"
ALL = (CUDA, DML, CPU)


def profile(providers=ALL, values=None):
    values = values or {"cpu": (10., 300.), "cuda": (20., 600.), "directml": (30., 900.)}
    return pref.Preference(context=pref.context_identity(providers), measurements=[
        pref.Measurement(device=device, identity=model_identity(target(device)).identity, verified=True,
                         query_ms=times[0], batch_ms=times[1]) for device, times in values.items()])


def write_raw(raw):
    path = pref.preference_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


@pytest.mark.parametrize("fastest", ["cpu", "cuda", "directml"])
def test_auto_uses_fastest_verified_representation_not_provider_priority(fastest):
    values = {device: (30., 900.) for device in pref.ORDER}
    values[fastest] = (10., 300.)
    pref.save_preference(profile(values=values), ALL)
    before = pref.preference_path().read_bytes()
    calls = []
    result = resolve_execution_target("auto", lambda item: calls.append(item.selected_device), providers=ALL)
    assert result.selected.selected_device == fastest and calls == [fastest]
    assert result.metadata()["selection_basis"] == "performance_profile"
    assert result.metadata()["performance_profile"] == "valid"
    assert pref.preference_path().read_bytes() == before


def test_balanced_score_and_stable_tie_policy():
    values = {"cpu": (10., 900.), "directml": (30., 300.)}
    pref.save_preference(profile(values=values), ALL)
    assert pref.load_preference(ALL) == (("directml", "cpu"), "valid")


@pytest.mark.parametrize("mode", ["cpu", "cuda", "directml"])
def test_explicit_modes_do_not_read_preferences(mode, monkeypatch):
    monkeypatch.setattr(pref, "load_preference", lambda _: pytest.fail("explicit read preference"))
    calls = []
    result = resolve_execution_target(mode, lambda item: calls.append(item.selected_device), providers=ALL)
    assert result.selected.selected_device == mode
    assert result.selection_basis == "explicit" and result.performance_profile == "not_used"
    assert calls == ([] if mode == "cpu" else [mode])


@pytest.mark.parametrize("failure", ["probe", "missing_artifact"])
def test_failed_preferred_candidate_uses_next_measured_without_repeating(failure):
    pref.save_preference(profile(), ALL)
    calls = []
    def probe(item):
        calls.append(item.selected_device)
        if item.selected_device == "cpu":
            raise RetrievalError("embedding_model_unavailable" if failure == "missing_artifact" else "embedding_failed")
    result = resolve_execution_target("auto", probe, providers=ALL)
    assert result.selected.selected_device == "cuda" and calls == ["cpu", "cuda"]
    assert result.selection_basis == "performance_profile"


def test_all_measured_candidates_fail_then_remaining_capability_is_probed():
    pref.save_preference(profile(values={"cpu": (10., 300.)}), ALL)
    calls = []
    def probe(item):
        calls.append(item.selected_device)
        if item.selected_device == "cpu":
            raise RetrievalError("embedding_model_invalid")
    result = resolve_execution_target("auto", probe, providers=ALL)
    assert result.selected.selected_device == "cuda" and calls == ["cpu", "cuda"]
    assert result.selection_basis == "capability_fallback"


@pytest.mark.parametrize("kind", ["missing", "corrupt", "oversize", "unknown_field", "unverified", "duplicate",
                                  "nan", "zero", "negative", "string_time", "unknown_device", "wrong_policy"])
def test_invalid_or_missing_preferences_retain_safe_capability_fallback(kind):
    data = profile().model_dump()
    if kind == "unknown_field": data["private_query"] = "PRIVATE_QUERY_PATH_VECTOR_CANARY"
    if kind == "unverified": data["measurements"][0]["verified"] = False
    if kind == "duplicate": data["measurements"][1] = data["measurements"][0]
    if kind == "nan": data["measurements"][0]["query_ms"] = float("nan")
    if kind == "zero": data["measurements"][0]["query_ms"] = 0
    if kind == "negative": data["measurements"][0]["batch_ms"] = -1
    if kind == "string_time": data["measurements"][0]["query_ms"] = "1"
    if kind == "unknown_device": data["measurements"][0]["device"] = "remote"
    if kind == "wrong_policy": data["policy"] = "unknown"
    if kind != "missing":
        write_raw(b"{" if kind == "corrupt" else b"x" * (pref.MAX_BYTES + 1) if kind == "oversize" else json.dumps(data).encode())
    result = resolve_execution_target("auto", lambda _: None, providers=ALL)
    assert result.selected.selected_device == "cuda" and result.selection_basis == "capability_fallback"
    assert result.performance_profile == ("missing" if kind == "missing" else "invalid")
    assert "PRIVATE" not in json.dumps(result.metadata())


@pytest.mark.parametrize("change", ["providers", "runtime", "artifact", "precision", "pipeline", "policy", "python"])
def test_relevant_context_changes_invalidate_without_rewriting(change, monkeypatch):
    pref.save_preference(profile(), ALL)
    before = pref.preference_path().read_bytes()
    providers = ALL
    if change == "providers": providers = (DML, CPU)
    elif change == "policy": monkeypatch.setattr(pref, "POLICY", "next-synthetic-policy")
    elif change == "python": monkeypatch.setattr(pref.platform, "python_version", lambda: "different")
    else:
        field, value = {"runtime": ("backend", "changed-runtime"), "artifact": ("artifact_sha256", "a" * 64),
                        "precision": ("precision", "fp16"), "pipeline": ("embedding_pipeline", "new-pipeline")}[change]
        original = pref.model_identity
        monkeypatch.setattr(pref, "model_identity", lambda item: original(item).model_copy(update={field: value}))
    assert pref.load_preference(providers) == ((), "stale")
    assert pref.preference_path().read_bytes() == before
    assert resolve_execution_target("auto", lambda _: None, providers=providers).selection_basis == "capability_fallback"


def test_measurement_identity_tampering_is_rejected():
    data = profile().model_dump()
    data["measurements"][0]["identity"] = "b" * 64
    write_raw(json.dumps(data).encode())
    assert pref.load_preference(ALL) == ((), "stale")


def test_atomic_write_failure_preserves_previous_profile_and_removes_temp(monkeypatch):
    pref.save_preference(profile(), ALL)
    path = pref.preference_path()
    before = path.read_bytes()
    def fail(*args): raise OSError("PRIVATE_PATH")
    monkeypatch.setattr(pref.os, "replace", fail)
    with pytest.raises(OSError): pref.save_preference(profile(), ALL)
    assert path.read_bytes() == before
    assert list(path.parent.glob("*.tmp")) == []


def test_symlink_profile_is_ignored_without_read_or_write(monkeypatch):
    path = write_raw(profile().model_dump_json().encode())
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda item: item == path or original(item))
    assert pref.load_preference(ALL) == ((), "invalid")
    with pytest.raises(ValueError): pref.save_preference(profile(), ALL)


def test_profiles_are_separate_for_runtime_packages(monkeypatch):
    cpu = pref.preference_path()
    monkeypatch.setattr(pref, "runtime_distribution", lambda: ("onnxruntime-directml", "1.24.4"))
    assert pref.preference_path() != cpu and pref.preference_path().parent == cpu.parent


def test_calibration_recomputes_only_synthetic_verified_measurements(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    tool = load_tool("retrieval_preference")
    monkeypatch.setattr(tool, "available_providers", lambda: ALL)
    seen = []
    def measure(device):
        seen.append(device)
        if device == "cuda": raise RetrievalError("embedding_accelerator_unavailable")
        return profile().measurements[0 if device == "cpu" else 2]
    monkeypatch.setattr(tool, "measure", measure)
    result = tool.calibrate()
    assert seen == list(pref.ORDER) and result["preferred_device"] == "cpu"
    assert result["python_network_or_provider_attempts"] == 0
    assert pref.load_preference(ALL) == (("cpu", "directml"), "valid")
    persisted = json.loads(pref.preference_path().read_text())
    assert set(persisted) == {"schema_version", "policy", "context", "measurements"}
    assert all(set(item) == {"device", "identity", "verified", "query_ms", "batch_ms"} for item in persisted["measurements"])


def test_failed_calibration_cannot_overwrite_existing_preference(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    pref.save_preference(profile(), ALL)
    before = pref.preference_path().read_bytes()
    tool = load_tool("retrieval_preference")
    monkeypatch.setattr(tool, "available_providers", lambda: ALL)
    def fail(_): raise RetrievalError("embedding_model_unavailable")
    monkeypatch.setattr(tool, "measure", fail)
    with pytest.raises(ValueError): tool.calibrate()
    assert pref.preference_path().read_bytes() == before


@pytest.mark.parametrize("invalid", [False, True])
def test_calibration_measure_uses_only_fixed_synthetic_inputs_and_validates_vectors(monkeypatch, invalid):
    import numpy as np
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    tool = load_tool("retrieval_preference")
    calls = []
    class Model:
        identity = model_identity(target("cpu"))
        def __init__(self, *, device): assert device == "cpu"
        def encode(self, texts):
            calls.append(texts)
            assert texts == [tool.QUERY] or texts == tool.BATCH
            value = np.zeros((len(texts), 384))
            value[:, 0] = 0 if invalid else 1
            return value.tolist()
    monkeypatch.setattr(tool, "OnnxEmbedder", Model)
    if invalid:
        with pytest.raises(ValueError): tool.measure("cpu")
    else:
        result = tool.measure("cpu")
        assert result.verified and result.query_ms > 0 and result.batch_ms > 0
        assert len(calls) == 15


def test_calibration_rejects_query_arguments_without_echoing_them():
    result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/retrieval_preference.py"),
                             "--query", "PRIVATE_QUERY_CANARY"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "PRIVATE_QUERY" not in result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "PERFORMANCE PREFERENCE NOT VERIFIED"


def test_auto_preference_fresh_process_never_benchmarks_private_data(tmp_path):
    script = r'''
import json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(sys.argv[1])/'tools'))
sys.path.insert(0,str(Path(sys.argv[1])/'backend'))
from retrieval_offline_guard import offline_guard
with offline_guard() as attempts:
    from local_retrieval import preference as p
    from local_retrieval.local_model import OnnxEmbedder
    from local_retrieval.embedding_identity import model_identity
    from local_retrieval.execution import target
    from tests.test_retrieval_model import FakeSession,tokenizer
    from tests.retrieval_fixtures import fixture_service
    p.LOCAL_DIRECTORY=Path(sys.argv[2])/'.local'
    providers=('CUDAExecutionProvider','CPUExecutionProvider')
    p.save_preference(p.Preference(context=p.context_identity(providers),measurements=[
        p.Measurement(device='cpu',identity=model_identity(target('cpu')).identity,verified=True,query_ms=1.,batch_ms=30.)]),providers)
    before=p.preference_path().read_bytes()
    with patch('local_retrieval.execution.available_providers',return_value=providers):
        model=OnnxEmbedder(device='auto')
        with patch.object(model,'_construct',return_value=(FakeSession(),tokenizer())):
            service,_,_,_=fixture_service(Path(sys.argv[2]),{'private':'PRIVATE_PDF_CANARY optical cooling'},embedder=model)
            service.index_all()
            result=service.search('PRIVATE_QUERY_CANARY optical')
            assert result.returned_count==1 and not attempts
            assert model.execution_metadata()['selection_basis']=='performance_profile'
            assert model.execution_metadata()['selected_device']=='cpu'
    assert p.preference_path().read_bytes()==before
    assert 'retrieval_preference' not in sys.modules
    print(json.dumps({'attempts':len(attempts),'device':'cpu','count':result.returned_count}))
'''
    result = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT), str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"attempts": 0, "device": "cpu", "count": 1}
    assert "PRIVATE" not in result.stdout + result.stderr


def test_preferred_cpu_cannot_reinterpret_or_rewrite_gpu_index(tmp_path, monkeypatch):
    from local_retrieval import RetrievalService
    from local_retrieval.local_model import OnnxEmbedder
    from .retrieval_fixtures import fixture_service, TestEmbedder
    from .test_retrieval_model import FakeSession, tokenizer
    previous = TestEmbedder(dimension=384)
    previous._identity = model_identity(target("directml"))
    service, store, _, _ = fixture_service(tmp_path, embedder=previous)
    service.index_all()
    before = service.index.path.read_bytes()
    pref.save_preference(profile(), ALL)
    monkeypatch.setattr("local_retrieval.execution.available_providers", lambda: ALL)
    model = OnnxEmbedder(device="auto")
    monkeypatch.setattr(model, "_construct", lambda *args, **kwargs: (FakeSession(), tokenizer()))
    current = RetrievalService(model, source_path=store.path, index_path=service.index.path)
    with pytest.raises(RetrievalError, match="^index_model_mismatch$"):
        current.search("synthetic optical")
    assert model.execution_metadata()["selected_device"] == "cpu"
    assert model.execution_metadata()["selection_basis"] == "performance_profile"
    assert service.index.path.read_bytes() == before


def test_gpu_acceptance_reports_auto_cpu_without_denying_gpu_availability(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    tool = load_tool("retrieval_gpu_acceptance")
    monkeypatch.setattr(tool, "capabilities", lambda: {"available_devices": ["directml"]})
    class Model:
        def __init__(self, *, device): assert device == "auto"
        def execution_metadata(self):
            return {"selected_device": "cpu", "selection_basis": "performance_profile"}
    monkeypatch.setattr(tool, "OnnxEmbedder", Model)
    monkeypatch.setattr(tool, "compare", lambda *args: pytest.fail("GPU benchmark ran after auto selected CPU"))
    result = tool.run("auto", 100)
    assert result["status"] == "GPU ACCEPTANCE NOT RUN; AUTO SELECTED CPU"
    assert result["execution"]["selected_device"] == "cpu"
