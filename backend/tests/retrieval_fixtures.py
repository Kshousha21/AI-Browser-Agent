"""Invented retrieval corpus and deterministic test-only embeddings."""
import hashlib
from pathlib import Path
import re

from local_pdf import DocumentStore, IngestionSettings, PdfIngestionService
from local_retrieval import EmbeddingIdentity, RetrievalService
from local_retrieval.models import fingerprint
from .pdf_fixtures import pdf_bytes

CORPUS = {
    "optics": "Photonics packaging uses optical interconnects to carry data with light. Co-packaged optics reduces power in AI datacenter links. Synthetic rare term: xylophotonic9.",
    "cooling": "Thermal packaging removes heat from processors. Efficient cooling lowers chip temperature and improves reliability.",
    "lidar": "Automotive lidar sensors measure distance with laser pulses. Autonomous vehicles detect obstacles using reflected light.",
    "agriculture": "Farmers grow wheat and barley in agricultural fields. Irrigation supplies water to crops during dry weather.",
    "german": "Optische Verbindungen transportieren Daten mit Licht. Die photonische Verpackung reduziert den Stromverbrauch in Rechenzentren.",
}


class TestEmbedder:
    """Synthetic feature space only; it is never registered with the production CLI.

    Shared concept features make ranking tests interpretable. Remaining tokens
    use stable hashes; no full query, expected rank or corpus ID is hard-coded.
    This tests mechanics, not learned semantic or multilingual quality.
    """
    __test__ = False
    groups = [
        {"photonics", "optics", "optical", "interconnects", "light", "optische", "licht", "photonische"},
        {"thermal", "heat", "cooling", "temperature", "hot", "warm", "temperatur"},
        {"agriculture", "agricultural", "farmers", "wheat", "barley", "crops", "irrigation"},
        {"lidar", "automotive", "vehicles", "distance", "obstacles"},
    ]

    def __init__(self, *, revision="test-v1", dimension=32):
        self._identity = EmbeddingIdentity(model_id="test-only/concept-hash", revision=revision,
            dimension=dimension, artifact_fingerprint=fingerprint("synthetic-features-v1"),
            backend="test-only-python", embedding_pipeline="test-features-v1")
        self.calls = []

    @property
    def identity(self):
        return self._identity

    def encode(self, texts):
        self.calls.extend(texts)
        results = []
        for text in texts:
            vector = [0.0] * self.identity.dimension
            for term in re.findall(r"[^\W_]+", text.lower()):
                found = False
                for index, group in enumerate(self.groups):
                    if term in group:
                        vector[index] += 2.0
                        found = True
                if not found:
                    index = 4 + int(hashlib.sha256(term.encode()).hexdigest()[:8], 16) % (len(vector) - 4)
                    vector[index] += 0.1
            if not any(vector):
                vector[-1] = 1.0
            results.append(vector)
        return results


def ingest_texts(folder: Path, texts: dict[str, str], *, max_chunk_chars=1600, overlap_chars=200):
    folder.mkdir(parents=True, exist_ok=True)
    store = DocumentStore(folder / "documents.sqlite3")
    service = PdfIngestionService(store, IngestionSettings(max_chunk_chars=max_chunk_chars, overlap_chars=overlap_chars))
    documents = {}
    for key, text in texts.items():
        source = folder / (key + ".pdf")
        source.write_bytes(pdf_bytes([text]))
        documents[key] = service.ingest_pdf(source).document
    return store, documents


def fixture_service(folder: Path, texts=None, *, embedder=None, **kwargs):
    store, documents = ingest_texts(folder, CORPUS if texts is None else texts)
    embedder = embedder or TestEmbedder()
    service = RetrievalService(embedder, source_path=store.path, index_path=folder / "retrieval.sqlite3", **kwargs)
    return service, store, documents, embedder
