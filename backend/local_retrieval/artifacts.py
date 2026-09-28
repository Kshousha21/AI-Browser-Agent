"""Pinned CPU and separate accelerator artifacts; no network/model resolution."""
import json
from pathlib import Path

from local_pdf.store import LOCAL_DIRECTORY
from .errors import RetrievalError

CPU_KIND = "cpu-avx2"
DEFAULT_MODEL_DIRECTORY = LOCAL_DIRECTORY / "models" / "minilm-l12-v2-e8f8c211226b"


def artifact_manifest(kind: str = CPU_KIND) -> dict:
    base = json.loads(Path(__file__).with_name("model_manifest.json").read_text(encoding="utf-8"))
    if kind == CPU_KIND:
        return base  # Keep the accepted CPU manifest/fingerprint unchanged.
    gpu = json.loads(Path(__file__).with_name("gpu_manifest.json").read_text(encoding="utf-8"))
    if kind not in gpu["artifacts"] or any(base[name] != gpu[name] for name in ("model_id", "revision", "license")):
        raise RetrievalError("embedding_model_invalid")
    spec = dict(gpu["artifacts"][kind])
    precision = spec.pop("precision")
    return {**base, "artifact_kind": kind, "precision": precision, "files": [spec, *base["files"][1:]]}


def artifact_directory(kind: str) -> Path:
    artifact_manifest(kind)
    return DEFAULT_MODEL_DIRECTORY if kind == CPU_KIND else DEFAULT_MODEL_DIRECTORY.with_name(DEFAULT_MODEL_DIRECTORY.name + "-" + kind)
