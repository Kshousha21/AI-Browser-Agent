"""Shared composition for operator-only local retrieval commands."""
import argparse
from pathlib import Path
from .schemas import VisualError

class SafeParser(argparse.ArgumentParser):
    def error(self, message): raise VisualError("visual_invalid_request")

def components(args):
    from browser.config import load_retrieval_settings
    from local_pdf import DocumentStore
    from local_pdf.rendering import LocalPageRenderer
    from document_workflow.storage import AcquisitionStorage
    from local_retrieval import RetrievalService
    from local_retrieval.local_model import OnnxEmbedder
    from local_visual_retrieval.service import VisualRetrievalService
    from local_visual_retrieval.fusion import MultimodalRetrieval
    settings = load_retrieval_settings(args.config)
    store = DocumentStore(args.database)
    storage = AcquisitionStorage(args.prepared_root)
    renderer = LocalPageRenderer(store, storage.prepared_path)
    visual = VisualRetrievalService(renderer, source_path=store.path, index_path=args.visual_index,
                                    settings=settings.visual_retrieval)
    text = RetrievalService(OnnxEmbedder(), source_path=store.path, index_path=args.text_index)
    fused = MultimodalRetrieval(text, visual, settings.multimodal_retrieval)
    return visual, fused, renderer

def options(parser):
    parser.add_argument("--config", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--visual-index", type=Path)
    parser.add_argument("--text-index", type=Path)
    parser.add_argument("--prepared-root", type=Path)

