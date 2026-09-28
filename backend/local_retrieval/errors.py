"""Closed, content-free local retrieval errors; never attach upstream details."""
from typing import Literal

Code = Literal[
    "invalid_arguments", "unknown_document", "source_unavailable", "source_changed",
    "index_unavailable", "index_corrupt", "unsupported_index", "index_model_mismatch",
    "index_capacity_exceeded", "index_write_failed", "invalid_local_path",
    "embedding_model_unavailable", "embedding_model_invalid", "embedding_failed",
    "embedding_dimension_mismatch", "invalid_embedding", "model_prepare_failed",
    "fts5_unavailable", "embedding_accelerator_unavailable", "embedding_runtime_conflict",
    "embedding_runtime_unavailable",
    "network_storage_unsupported",
]


class RetrievalError(RuntimeError):
    def __init__(self, code: Code):
        self.code = code
        super().__init__(code)
