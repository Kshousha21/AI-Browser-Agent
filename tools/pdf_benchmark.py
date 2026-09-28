"""Repeatable synthetic local ingestion measurements; never uses private PDFs."""
from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from local_pdf import DocumentStore, PdfIngestionService
from tests.pdf_fixtures import pdf_bytes


def main() -> None:
    artifacts = ROOT / ".local"
    artifacts.mkdir(exist_ok=True)
    # This helper creates and cleans only its own synthetic temporary directory.
    with TemporaryDirectory(prefix="pdf-benchmark-", dir=artifacts) as directory:
        folder = Path(directory)
        text = "Market Example A: 12.5 mm. Technology Example B: 300 K.\n" * 60
        source = folder / "synthetic.pdf"
        source.write_bytes(pdf_bytes([text] * 40))
        store = DocumentStore(folder / "documents.sqlite3")
        started = perf_counter()
        document = PdfIngestionService(store).ingest_pdf(source).document
        elapsed = perf_counter() - started
        print(json.dumps({"fixture": "synthetic-40-pages", "file_bytes": document.file_size,
                          "page_count": document.page_count, "extracted_characters": document.extracted_chars,
                          "chunk_count": document.chunk_count, "elapsed_seconds": round(elapsed, 4),
                          "database_bytes": store.path.stat().st_size}))


if __name__ == "__main__":
    main()
