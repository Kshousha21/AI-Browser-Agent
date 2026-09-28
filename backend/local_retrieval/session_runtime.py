"""Bounded local ONNX sessions and the unchanged Phase 4D pooling contract."""
from contextlib import contextmanager
import os
from threading import RLock

import numpy as np
from .errors import RetrievalError
from .execution import EmbeddingExecutionTarget
from .token_windows import windows as token_windows
from .vectors import normalized

_NATIVE_DIAGNOSTICS = RLock()


@contextmanager
def quiet_native_initialization():
    """Discard native initialization diagnostics rather than logging paths/driver dumps."""
    with _NATIVE_DIAGNOSTICS:
        copies = []
        try:
            with open(os.devnull, "wb") as sink:
                for descriptor in (1, 2):
                    copies.append((descriptor, os.dup(descriptor)))
                    os.dup2(sink.fileno(), descriptor)
                yield
        finally:
            for descriptor, copy in copies:
                os.dup2(copy, descriptor)
                os.close(copy)


def create_session(data: bytes, selected: EmbeddingExecutionTarget, *, profile_prefix: str | None = None):
    import onnxruntime as ort
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    gpu = selected.selected_device != "cpu"
    if selected.selected_device == "cuda":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if selected.selected_device == "directml":
        if profile_prefix is None:
            raise RetrievalError("embedding_accelerator_unavailable")
        options.enable_mem_pattern = False
        options.enable_profiling = True
        options.profile_file_prefix = profile_prefix
    try:
        with quiet_native_initialization():
            session = ort.InferenceSession(data, sess_options=options,
                                           providers=[selected.execution_provider])
            session.disable_fallback()  # Also disable Python's EP-error retry path.
        actual = session.get_providers()
        if selected.execution_provider not in actual or any(provider not in {selected.execution_provider, "CPUExecutionProvider"} for provider in actual):
            raise RetrievalError("embedding_accelerator_unavailable" if gpu else "embedding_model_invalid")
        if selected.selected_device == "cuda" and session.get_session_options().get_session_config_entry("session.disable_cpu_ep_fallback") != "1":
            raise RetrievalError("embedding_accelerator_unavailable")
        if set(item.name for item in session.get_inputs()) != {"input_ids", "attention_mask", "token_type_ids"}:
            raise RetrievalError("embedding_model_invalid")
        return session
    except RetrievalError:
        raise
    except Exception:
        raise RetrievalError("embedding_accelerator_unavailable" if gpu else "embedding_model_invalid") from None


def encode_local(session, tokenizer, texts: list[str], dimension: int, window_batch: int = 8) -> list[list[float]]:
    """Same CPU token windows, float64 masked sums and final float32 normalization."""
    try:
        windows = token_windows(tokenizer, texts, 128)
        totals = np.zeros((len(texts), dimension), dtype=np.float64)
        counts = np.zeros(len(texts), dtype=np.float64)
        for start in range(0, len(windows), window_batch):
            batch = windows[start:start + window_batch]
            inputs = {"input_ids": np.asarray([part.ids for _, part in batch], dtype=np.int64),
                      "attention_mask": np.asarray([part.attention_mask for _, part in batch], dtype=np.int64),
                      "token_type_ids": np.asarray([part.type_ids for _, part in batch], dtype=np.int64)}
            hidden = session.run(None, inputs)[0]
            expected = (*inputs["input_ids"].shape, dimension)
            if hidden.shape != expected or not np.isfinite(hidden).all():
                raise RetrievalError("invalid_embedding")
            mask = inputs["attention_mask"][..., None]
            sums = (hidden.astype(np.float64) * mask).sum(axis=1)
            for index, (owner, _) in enumerate(batch):
                totals[owner] += sums[index]
                counts[owner] += mask[index].sum()
        if (counts <= 0).any():
            raise RetrievalError("invalid_embedding")
        return [normalized(total / count, dimension).tolist()
                for total, count in zip(totals, counts, strict=True)]
    except RetrievalError:
        raise
    except Exception:
        # Includes device OOM. Never retry another provider after selection.
        raise RetrievalError("embedding_failed") from None
