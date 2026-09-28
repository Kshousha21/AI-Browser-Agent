"""Local ranked document data only. No provider, browser, or answering integration."""
from .errors import RetrievalError
from .models import EmbeddingIdentity, RetrievalSettings, SearchResult
from .service import RetrievalService

__all__ = ["RetrievalError", "EmbeddingIdentity", "RetrievalSettings", "SearchResult", "RetrievalService"]
