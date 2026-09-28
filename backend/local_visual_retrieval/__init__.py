"""Offline visual PDF page retrieval, separate from the MiniLM text index."""
from .schemas import VisualError, VisualSettings, MultimodalSettings

__all__ = ["VisualError", "VisualSettings", "MultimodalSettings", "VisualRetrievalService"]

def __getattr__(name):
    if name == "VisualRetrievalService":
        from .service import VisualRetrievalService
        return VisualRetrievalService
    raise AttributeError(name)
