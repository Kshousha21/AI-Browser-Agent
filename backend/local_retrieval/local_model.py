"""Verified local ONNX/tokenizer files; optional accelerators never download."""
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from threading import RLock
from tempfile import TemporaryDirectory

from .errors import RetrievalError
from .models import EmbeddingIdentity
from .source import local_path
from .artifacts import artifact_manifest, artifact_directory, DEFAULT_MODEL_DIRECTORY
from .embedding_identity import model_identity
from .execution import EmbeddingExecutionTarget, resolve_execution_target
from .session_runtime import create_session, encode_local
from .gpu_profile import validate_profile

MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")


def manifest() -> dict:
    """Original CPU artifact manifest remains byte-for-byte unchanged."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def verified_files(directory: Path | str, artifact_kind: str = "cpu-avx2") -> dict[str, bytes]:
    """Verify the bytes that will actually be loaded, including bounded sizes."""
    folder = local_path(directory)
    files = {}
    try:
        for spec in artifact_manifest(artifact_kind)["files"]:
            path = folder / spec["local"]
            if not path.is_file():
                raise RetrievalError("embedding_model_unavailable")
            if path.is_symlink() or path.stat().st_size != spec["bytes"]:
                raise RetrievalError("embedding_model_invalid")
            with path.open("rb") as stream:
                data = stream.read(spec["bytes"] + 1)
            if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise RetrievalError("embedding_model_invalid")
            files[spec["local"]] = data
        return files
    except OSError:
        raise RetrievalError("embedding_model_unavailable") from None


class OnnxEmbedder:
    """Local CPU/CUDA/DirectML embeddings with selection frozen before indexing.

    Only auto initialization can fall back. Once selected, failures including
    OOM are returned safely without changing providers or the representation.
    """

    def __init__(self, directory: Path | str = DEFAULT_MODEL_DIRECTORY, *,
                 device: str = "auto", gpu_directory: Path | str | None = None):
        if device not in {"auto", "cpu", "cuda", "directml"}:
            raise RetrievalError("invalid_arguments")
        self.directory = local_path(directory)
        self.gpu_directory = local_path(gpu_directory) if gpu_directory is not None else None
        self.device = device
        self._identity = model_identity()
        self._selection = None
        self._session = None
        self._tokenizer = None
        self._probe_evidence = None
        self._lock = RLock()

    @property
    def identity(self) -> EmbeddingIdentity:
        with self._lock:
            self._resolve()
            return self._identity

    def execution_metadata(self) -> dict[str, object]:
        with self._lock:
            self._resolve()
            return {**self._selection.metadata(), "dimension": 384,
                    "gpu_probe_passed": self._selection.selected.selected_device != "cpu",
                    "cpu_ep_fallback_disabled": self._selection.selected.selected_device == "cuda",
                    "graph_execution": self._probe_evidence}

    def _resolve(self) -> None:
        if self._selection is None:
            selection = resolve_execution_target(self.device, self._probe)
            self._identity = model_identity(selection.selected)
            self._selection = selection

    def _construct(self, selected: EmbeddingExecutionTarget, *, profile_prefix: str | None = None):
        directory = self.directory if selected.selected_device == "cpu" else (
            self.gpu_directory or artifact_directory(selected.artifact_kind))
        data = verified_files(directory, selected.artifact_kind)
        try:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_str(data["tokenizer.json"].decode("utf-8"))
            tokenizer.no_truncation()
            tokenizer.no_padding()
            session = create_session(data["model.onnx"], selected, profile_prefix=profile_prefix)
            return session, tokenizer
        except RetrievalError:
            raise
        except Exception:
            raise RetrievalError("embedding_model_invalid") from None

    def _probe(self, selected: EmbeddingExecutionTarget) -> None:
        if selected.selected_device == "directml":
            with TemporaryDirectory(prefix="local-embedding-probe-") as temporary:
                directory = Path(temporary).resolve()
                session, tokenizer = self._construct(selected, profile_prefix=str(directory / "synthetic"))
                try:
                    for count in (1, 8):
                        encode_local(session, tokenizer, ["Synthetic local optical connections."] * count, 384, 8)
                    profile = Path(session.end_profiling())  # Stop before any private text can run.
                    evidence = validate_profile(profile, directory)
                except Exception:
                    session.end_profiling()
                    raise RetrievalError("embedding_accelerator_unavailable") from None
            self._session, self._tokenizer, self._probe_evidence = session, tokenizer, evidence
            return
        session, tokenizer = self._construct(selected)
        # One bounded, content-free compatibility probe before exposing identity.
        encode_local(session, tokenizer, ["Synthetic local optical connections."], 384, selected.window_batch)
        self._session, self._tokenizer = session, tokenizer

    def _load(self) -> None:
        self._resolve()
        if self._session is None:
            self._session, self._tokenizer = self._construct(self._selection.selected)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not 1 <= len(texts) <= 32 or any(not isinstance(text, str) or not text.strip() or len(text) > 16000 for text in texts):
            raise RetrievalError("invalid_arguments")
        with self._lock:
            self._load()
            return encode_local(self._session, self._tokenizer, list(texts),
                                self._identity.dimension, self._selection.selected.window_batch)
