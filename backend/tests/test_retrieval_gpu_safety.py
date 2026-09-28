"""Graph-placement proof, conflicting packages, and fresh-process GPU-path isolation."""
import json
from pathlib import Path
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from local_retrieval import RetrievalError
from local_retrieval.gpu_profile import validate_profile
from local_retrieval.execution import runtime_distribution, target
from local_retrieval.session_runtime import create_session
from .test_retrieval_isolation import load_tool, ROOT


def event(provider, operation, size=4):
    return {"cat": "Node", "args": {"provider": provider, "op_name": operation,
            "input_type_shape": [{"int64": [size]}], "output_type_shape": [{"int64": [size]}]}}


@pytest.mark.parametrize("problem", [None, "cpu_matmul", "large_cpu_tensor", "unknown_provider", "no_shapes", "missing_gpu_work"])
def test_directml_accepts_only_proven_gpu_compute_and_bounded_cpu_shapes(tmp_path, problem):
    events = [event("DmlExecutionProvider", "MatMul", 384)] * 192 + [event("CPUExecutionProvider", "Gather")]
    if problem == "cpu_matmul": events.append(event("CPUExecutionProvider", "MatMul", 384))
    if problem == "large_cpu_tensor": events.append(event("CPUExecutionProvider", "Gather", 5))
    if problem == "unknown_provider": events.append(event("AzureExecutionProvider", "MatMul"))
    if problem == "no_shapes": events.append({"cat": "Node", "args": {"provider": "CPUExecutionProvider", "op_name": "Gather"}})
    if problem == "missing_gpu_work": events = events[-1:]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(events))
    if problem:
        with pytest.raises(RetrievalError, match="^embedding_accelerator_unavailable$"):
            validate_profile(path, tmp_path)
    else:
        result = validate_profile(path, tmp_path)
        assert result["gpu_matmul_events"] == 192 and result["cpu_max_tensor_elements"] == 4
        assert result["fully_gpu_graph"] is False and result["profiling_stopped"]


@pytest.mark.parametrize("packages,valid", [([], True), (["onnxruntime"], True),
    (["onnxruntime-gpu"], False), (["onnxruntime-directml"], False),
    (["onnxruntime", "onnxruntime-gpu"], False), (["onnxruntime", "onnxruntime-directml"], False)])
def test_base_setup_rejects_overlapping_runtime_variants(packages, valid):
    assert load_tool("check_base_runtime").safe_base(packages) is valid


@pytest.mark.parametrize("device,package", [("cuda", "onnxruntime-gpu==1.29.0"), ("directml", "onnxruntime-directml==1.24.4")])
def test_optional_setup_selects_exactly_one_runtime(device, package):
    lines = load_tool("setup_accelerator").requirement_lines(device)
    assert [line for line in lines if line.startswith("onnxruntime")] == [package]
    assert "numpy==2.4.6" in lines and "tokenizers==0.23.2" in lines


@pytest.mark.parametrize("installed", [[], ["onnxruntime", "onnxruntime-gpu"], ["onnxruntime-directml", "onnxruntime-gpu"]])
def test_runtime_rejects_missing_or_overlapping_distributions(monkeypatch, installed):
    def version(name):
        if name not in installed:
            raise PackageNotFoundError(name)
        return "1.29.0"
    monkeypatch.setattr("local_retrieval.execution.version", version)
    with pytest.raises(RetrievalError, match="^embedding_runtime_(unavailable|conflict)$"):
        runtime_distribution()


@pytest.mark.parametrize("device", ["cpu", "cuda", "directml"])
def test_session_options_enforce_claimed_execution_and_disable_retry(monkeypatch, tmp_path, device):
    captured = {}
    class Options:
        def __init__(self): self.entries = {}
        def add_session_config_entry(self, key, value): self.entries[key] = value
        def get_session_config_entry(self, key): return self.entries[key]
    class Session:
        def __init__(self, data, *, sess_options, providers):
            captured.update(options=sess_options, providers=providers, retries_disabled=False)
        def disable_fallback(self): captured["retries_disabled"] = True
        def get_providers(self): return captured["providers"]
        def get_session_options(self): return captured["options"]
        def get_inputs(self): return [SimpleNamespace(name=name) for name in ("input_ids", "attention_mask", "token_type_ids")]
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(disable_telemetry_events=lambda: None,
        SessionOptions=Options, InferenceSession=Session, ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential")))
    selected = target(device)
    create_session(b"synthetic", selected, profile_prefix=str(tmp_path / "probe") if device == "directml" else None)
    assert captured["providers"] == [selected.execution_provider] and captured["retries_disabled"]
    if device == "cuda":
        assert captured["options"].entries["session.disable_cpu_ep_fallback"] == "1"
    if device == "directml":
        assert captured["options"].enable_mem_pattern is False and captured["options"].enable_profiling is True


def test_mock_gpu_index_and_query_in_fresh_process_have_no_remote_access(tmp_path):
    script = r'''
import json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(sys.argv[1])/'tools'))
from retrieval_offline_guard import offline_guard
with offline_guard() as attempts:
    sys.path.insert(0,str(Path(sys.argv[1])/'backend'))
    from local_retrieval.local_model import OnnxEmbedder
    from tests.test_retrieval_model import FakeSession,tokenizer
    from tests.retrieval_fixtures import fixture_service
    with patch('local_retrieval.execution.available_providers',return_value=('CUDAExecutionProvider','CPUExecutionProvider')):
        model=OnnxEmbedder(device='cuda')
        with patch.object(model,'_construct',return_value=(FakeSession(),tokenizer())):
            service,_,_,_=fixture_service(Path(sys.argv[2]),embedder=model)
            service.index_all()
            result=service.search('PRIVATE_QUERY_CANARY optical')
            assert result.returned_count>0 and not attempts
            assert model.execution_metadata()['selected_device']=='cuda'
            print(json.dumps({'attempts':len(attempts),'count':result.returned_count,'device':'cuda'}))
'''
    result = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT), str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"attempts": 0, "count": 5, "device": "cuda"}
    assert "PRIVATE_QUERY" not in result.stdout + result.stderr
