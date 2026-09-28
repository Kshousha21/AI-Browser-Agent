"""Answer from explicitly ingested local PDFs; remote context requires consent."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from browser.config import load_provider_settings
from local_retrieval import RetrievalService
from local_retrieval.local_model import DEFAULT_MODEL_DIRECTORY, OnnxEmbedder
from local_rag import ProviderBridge, RagError, RagService


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise RagError("rag_invalid_request")


def main(argv=None) -> int:
    try:
        parser = SafeParser(description=__doc__)
        ask = parser.add_subparsers(dest="command", required=True).add_parser("ask")
        ask.add_argument("question")
        ask.add_argument("--top-k", type=int, default=5)
        ask.add_argument("--mode", choices=["lexical", "semantic", "hybrid"], default="hybrid")
        ask.add_argument("--document-id", action="append", dest="document_ids")
        ask.add_argument("--provider-profile")
        ask.add_argument("--config", type=Path)
        ask.add_argument("--allow-remote-context", action="store_true")
        ask.add_argument("--database", type=Path)
        ask.add_argument("--index-database", type=Path)
        ask.add_argument("--visual-index", type=Path)
        ask.add_argument("--prepared-root", type=Path)
        ask.add_argument("--model-directory", type=Path, default=DEFAULT_MODEL_DIRECTORY)
        ask.add_argument("--gpu-model-directory", type=Path)
        ask.add_argument("--device", choices=["auto", "cpu", "cuda", "directml"], default="auto")
        args = parser.parse_args(argv)
        try:
            provider = load_provider_settings(args.config, args.provider_profile)
        except Exception:
            raise RagError("rag_configuration_error") from None
        embedder = OnnxEmbedder(args.model_directory, device=args.device, gpu_directory=args.gpu_model_directory)
        retrieval = RetrievalService(embedder, source_path=args.database, index_path=args.index_database)
        from local_visual_retrieval.cli import components
        args.text_index = args.index_database
        _, fused, renderer = components(args)
        fused.text = retrieval
        if not args.document_ids:
            with retrieval.source.snapshot() as source: args.document_ids = sorted(source)
        service = RagService(retrieval, ProviderBridge(provider, allow_remote_context=args.allow_remote_context),
                             renderer=renderer, multimodal=fused)
        result = asyncio.run(service.answer(args.question, args.top_k, args.mode, args.document_ids))
        print(json.dumps(result.model_dump(), ensure_ascii=True))
        return 0
    except RagError as exc:
        print(json.dumps({"error": exc.code}), file=sys.stderr)
        return 2
    except Exception:
        print(json.dumps({"error": "rag_retrieval_failed"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
