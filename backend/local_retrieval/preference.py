"""Bounded, local synthetic performance preferences; never a model or data resolver."""
import os
import platform
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import Field, ValidationError, model_validator
from local_pdf.models import LocalModel
from local_pdf.store import LOCAL_DIRECTORY
from .embedding_identity import model_identity
from .execution import PROVIDERS, runtime_distribution, target
from .models import Digest, fingerprint

POLICY = "synthetic-query-batch30-balanced-v1"
MAX_BYTES = 8192
ORDER = ("cuda", "directml", "cpu")


class Measurement(LocalModel):
    device: Literal["cpu", "cuda", "directml"]
    identity: Digest
    verified: Literal[True]
    query_ms: float = Field(gt=0, le=3_600_000, allow_inf_nan=False)
    batch_ms: float = Field(gt=0, le=3_600_000, allow_inf_nan=False)

    @property
    def score_ms(self) -> float:
        # Equal weight to warm single-query and batch per-text embedding latency.
        return (self.query_ms + self.batch_ms / 30) / 2


class Preference(LocalModel):
    schema_version: Literal[1] = 1
    policy: Literal["synthetic-query-batch30-balanced-v1"] = POLICY
    context: Digest
    measurements: list[Measurement] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique_devices(self):
        if len({item.device for item in self.measurements}) != len(self.measurements):
            raise ValueError("duplicate_device")
        return self


def context_identity(providers: tuple[str, ...]) -> str:
    """Only a digest is persisted; no host name, path, query or hardware serial."""
    return fingerprint({"policy": POLICY, "providers": sorted(set(providers)),
                        "representations": {device: model_identity(target(device)).identity for device in ORDER},
                        "python": platform.python_version(), "system": platform.system(),
                        "release": platform.release(), "machine": platform.machine(),
                        "logical_cpus": os.cpu_count()})


def preference_path() -> Path:
    package, _ = runtime_distribution()
    # A CPU environment must not overwrite a different runtime's measurements.
    path = LOCAL_DIRECTORY / ("embedding-preference-" + package + ".json")
    if LOCAL_DIRECTORY.resolve() != LOCAL_DIRECTORY.absolute() or path.is_symlink():
        raise ValueError("invalid_preference_location")
    return path


def validate_preference(profile: Preference, providers: tuple[str, ...]) -> bool:
    return profile.context == context_identity(providers) and all(
        PROVIDERS[item.device] in providers and item.identity == model_identity(target(item.device)).identity
        for item in profile.measurements)


def load_preference(providers: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
    """Read-only. Invalid/stale files are ignored, never repaired during retrieval."""
    try:
        path = preference_path()
        if not path.exists():
            return (), "missing"
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            return (), "invalid"
        with path.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            return (), "invalid"
        profile = Preference.model_validate_json(raw)
        if not validate_preference(profile, providers):
            return (), "stale"
        ranked = sorted(profile.measurements, key=lambda item: (item.score_ms, ORDER.index(item.device)))
        return tuple(item.device for item in ranked), "valid"
    except (OSError, ValueError, ValidationError, RecursionError):
        return (), "invalid"


def save_preference(profile: Preference, providers: tuple[str, ...]) -> None:
    """Explicit calibration only; validate before an atomic helper-owned write."""
    profile = Preference.model_validate_json(profile.model_dump_json())
    if not validate_preference(profile, providers):
        raise ValueError("stale_preference")
    data = profile.model_dump_json().encode("utf-8")
    if len(data) > MAX_BYTES:
        raise ValueError("invalid_preference")
    path = preference_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".embedding-preference-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if preference_path() != path:
            raise ValueError("invalid_preference_location")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
