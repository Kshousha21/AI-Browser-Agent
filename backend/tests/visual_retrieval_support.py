"""Synthetic page embeddings exercise mechanics, not Nomic relevance."""
from contextlib import asynccontextmanager
import numpy as np
from local_pdf import DocumentStore, PdfIngestionService
from local_pdf.rendering import LocalPageRenderer
from local_retrieval import RetrievalService
from local_visual_retrieval.schemas import VisualIdentity
from local_visual_retrieval.service import VisualRetrievalService
from local_visual_retrieval.fusion import MultimodalRetrieval
from .retrieval_fixtures import TestEmbedder
from .visual_pdf_fixtures import research_pdf

class SyntheticVisualModel:
    def __init__(self):
        self.identity = VisualIdentity(model="synthetic-only", revision="1", artifact_fingerprint="a"*64,
            processor="synthetic", runtime="synthetic", dimension=4, renderer="synthetic")
        self.images = self.queries = self.opened = self.closed = 0
        self.measurements = {}
    async def describe(self): return (self.identity, None, None)
    @asynccontextmanager
    async def session(self, description):
        self.opened += 1
        try: yield self
        finally: self.closed += 1
    async def image(self, png):
        vector = np.zeros(4); vector[self.images % 4] = 1
        self.images += 1
        return vector.tolist()
    async def query(self, query):
        self.queries += 1
        return [0., 0., 0., 1.]

def setup(folder, kind="chart"):
    path = folder / "synthetic.pdf"
    path.write_bytes(research_pdf(kind))
    store = DocumentStore(folder / "documents.sqlite3")
    doc = PdfIngestionService(store).ingest_pdf(path).document
    text = RetrievalService(TestEmbedder(), source_path=store.path, index_path=folder / "retrieval.sqlite3")
    text.index_all()
    renderer = LocalPageRenderer(store, lambda identifier: path if identifier == doc.document_id else None)
    model = SyntheticVisualModel()
    visual = VisualRetrievalService(renderer, source_path=store.path, index_path=folder / "visual.sqlite3", model=model)
    return MultimodalRetrieval(text, visual), doc, model, path
