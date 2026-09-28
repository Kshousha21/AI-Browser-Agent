"""Finite float32 unit vectors and transparent bounded cosine ranking."""
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .errors import RetrievalError
from .models import EmbeddingIdentity

Vector = NDArray[np.float32]


class Embedder(Protocol):
    @property
    def identity(self) -> EmbeddingIdentity: ...

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


def normalized(vector: Sequence[float], dimension: int) -> Vector:
    """Reject bad dimensions/nonfinite/zero values; normalize before float32 storage."""
    try:
        with np.errstate(all="ignore"):
            raw = np.asarray(vector, dtype=np.float64)
            if raw.shape != (dimension,):
                raise RetrievalError("embedding_dimension_mismatch")
            length = np.linalg.norm(raw)
            if not np.isfinite(raw).all() or not np.isfinite(length) or length <= 1e-12:
                raise RetrievalError("invalid_embedding")
            result = np.asarray(raw / length, dtype="<f4")
        if not np.isfinite(result).all():
            raise RetrievalError("invalid_embedding")
        return result
    except (ValueError, TypeError, OverflowError):
        raise RetrievalError("invalid_embedding") from None


def unpack(blob: bytes, dimension: int) -> Vector:
    """Stored vectors must already be finite unit vectors, not silently repaired."""
    if not isinstance(blob, bytes) or len(blob) != dimension * 4:
        raise RetrievalError("index_corrupt")
    vector = np.frombuffer(blob, dtype="<f4")
    if not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector)) - 1.0) > 1e-4:
        raise RetrievalError("index_corrupt")
    return vector


def encode_batch(embedder: Embedder, texts: Sequence[str]) -> list[bytes]:
    try:
        encoded = embedder.encode(texts)
        if len(encoded) != len(texts):
            raise RetrievalError("invalid_embedding")
        return [normalized(vector, embedder.identity.dimension).tobytes() for vector in encoded]
    except RetrievalError:
        raise
    except Exception:
        raise RetrievalError("embedding_failed") from None
