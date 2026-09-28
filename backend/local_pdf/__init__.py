"""Local, operator-controlled PDF ingestion. No browser or model integration."""

from .errors import PdfIngestError
from .models import IngestionSettings
from .service import PdfIngestionService
from .store import DocumentStore

__all__ = ["DocumentStore", "IngestionSettings", "PdfIngestError", "PdfIngestionService"]
