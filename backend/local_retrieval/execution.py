"""Explicit local execution selection; availability alone never proves GPU usability."""
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from .artifacts import artifact_manifest
from .errors import RetrievalError

Device = Literal["auto", "cpu", "cuda", "directml"]
PROVIDERS = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "directml": "DmlExecutionProvider"}
RUNTIME_PINS = {"onnxruntime": "1.29.0", "onnxruntime-gpu": "1.29.0", "onnxruntime-directml": "1.24.4"}


def runtime_distribution() -> tuple[str, str]:
    installed = []
    for name in RUNTIME_PINS:
        try:
            installed.append((name, version(name)))
        except PackageNotFoundError:
            pass
    if len(installed) != 1:
        raise RetrievalError("embedding_runtime_conflict" if installed else "embedding_runtime_unavailable")
    return installed[0]


def available_providers() -> tuple[str, ...]:
    runtime_distribution()
    try:
        import onnxruntime as ort
        return tuple(provider for provider in PROVIDERS.values() if provider in ort.get_available_providers())
    except Exception:
        raise RetrievalError("embedding_runtime_unavailable") from None


@dataclass(frozen=True)
class EmbeddingExecutionTarget:
    selected_device: str
    execution_provider: str
    artifact_kind: str
    precision: str
    window_batch: int = 8


def target(device: str) -> EmbeddingExecutionTarget:
    if device not in PROVIDERS:
        raise RetrievalError("invalid_arguments")
    # O4 uses fused CUDA-oriented operators. DirectML uses the generic graph.
    kind = {"cpu": "cpu-avx2", "cuda": "gpu-o4", "directml": "gpu-fp32"}[device]
    return EmbeddingExecutionTarget(device, PROVIDERS[device], kind,
                                    "quint8" if device == "cpu" else artifact_manifest(kind)["precision"])


@dataclass(frozen=True)
class ExecutionSelection:
    requested_device: str
    selected: EmbeddingExecutionTarget
    reasons: tuple[tuple[str, str], ...] = ()
    selection_basis: str = "explicit"
    performance_profile: str = "not_used"

    def metadata(self) -> dict[str, object]:
        return {"requested_device": self.requested_device,
                "selected_device": self.selected.selected_device,
                "execution_provider": self.selected.execution_provider,
                "artifact_kind": self.selected.artifact_kind, "precision": self.selected.precision,
                "window_batch": self.selected.window_batch,
                "selection_basis": self.selection_basis, "performance_profile": self.performance_profile,
                "initialization_fallbacks": [{"device": device, "reason": reason} for device, reason in self.reasons]}


def resolve_execution_target(mode: str, probe: Callable[[EmbeddingExecutionTarget], None],
                             *, providers: tuple[str, ...] | None = None) -> ExecutionSelection:
    """Resolve auto preferences or capability fallback; explicit CPU remains lazy."""
    if mode not in ("auto", "cpu", "cuda", "directml"):
        raise RetrievalError("invalid_arguments")
    available = available_providers() if providers is None else providers
    choices = ("cuda", "directml", "cpu") if mode == "auto" else (mode,)
    preferred, profile_status = (), "not_used"
    if mode == "auto":
        from .preference import load_preference
        preferred, profile_status = load_preference(available)
    # Verified measured candidates first; if none is still usable, retain the
    # capability fallback for remaining candidates. Never probe a failure twice.
    candidates = [(device, True) for device in preferred]
    candidates += [(device, False) for device in choices if device not in preferred]
    reasons = []
    for device, measured in candidates:
        selected = target(device)
        basis = "performance_profile" if measured else ("capability_fallback" if mode == "auto" else "explicit")
        if selected.execution_provider not in available:
            reasons.append((device, "provider_absent"))
            continue
        if device == "cpu" and not measured:
            return ExecutionSelection(mode, selected, tuple(reasons), basis, profile_status)
        try:
            probe(selected)
            return ExecutionSelection(mode, selected, tuple(reasons), basis, profile_status)
        except RetrievalError as error:
            category = "artifact_unavailable" if error.code == "embedding_model_unavailable" else "probe_failed"
            reasons.append((device, category))
        except Exception:
            reasons.append((device, "probe_failed"))
    raise RetrievalError("embedding_accelerator_unavailable" if mode != "cpu" else "embedding_runtime_unavailable")


def capabilities() -> dict[str, object]:
    """Lightweight diagnostics only. Listed GPU providers still require a real probe."""
    available = available_providers()
    devices = [device for device in ("cuda", "directml") if PROVIDERS[device] in available]
    return {"status": "GPU PROVIDER PRESENT; INFERENCE NOT VERIFIED" if devices else "CPU ONLY",
            "available_devices": devices, "cpu_available": PROVIDERS["cpu"] in available,
            "probe_performed": False}
