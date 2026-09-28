"""Explicit representation identity, including safe compatibility with original CPU indexes."""
from importlib.metadata import version

from .artifacts import artifact_manifest, CPU_KIND
from .execution import EmbeddingExecutionTarget, runtime_distribution, target
from .models import EmbeddingIdentity, fingerprint

REPRESENTATION_FIELDS = {"execution_artifact", "artifact_sha256", "precision", "execution_provider"}


def model_identity(selected: EmbeddingExecutionTarget | None = None) -> EmbeddingIdentity:
    selected = selected or target("cpu")
    info = artifact_manifest(selected.artifact_kind)
    package, installed = runtime_distribution()
    backend = ";".join([f"{package}/{installed}", *[f"{name}/{version(name)}" for name in ("tokenizers", "numpy")]])
    suffix = {"cpu": "cpu-sequential-threads2", "cuda": "cuda-sequential-no-cpu-fallback-batch8",
              "directml": "directml-gpu-compute-cpu-shapes4-v1-batch8"}[selected.selected_device]
    return EmbeddingIdentity(model_id=info["model_id"], revision=info["revision"],
                             dimension=info["dimension"], artifact_fingerprint=fingerprint(info),
                             backend=backend + ";" + suffix, embedding_pipeline=info["embedding_pipeline"],
                             execution_artifact=selected.artifact_kind, artifact_sha256=info["files"][0]["sha256"],
                             precision=selected.precision, execution_provider=selected.execution_provider)


def compatible_identity(stored: EmbeddingIdentity, expected: EmbeddingIdentity) -> bool:
    if stored == expected:
        return True
    # The original CPU descriptor already fingerprinted the complete pinned
    # artifact manifest. Preserve that exact known representation read-only.
    if any(getattr(stored, name) is not None for name in REPRESENTATION_FIELDS):
        return False
    cpu = artifact_manifest(CPU_KIND)
    return (expected.execution_artifact == CPU_KIND
            and expected.artifact_sha256 == cpu["files"][0]["sha256"]
            and expected.artifact_fingerprint == fingerprint(cpu)
            and expected.precision == "quint8" and expected.execution_provider == "CPUExecutionProvider"
            and stored.model_dump(exclude=REPRESENTATION_FIELDS) == expected.model_dump(exclude=REPRESENTATION_FIELDS))
