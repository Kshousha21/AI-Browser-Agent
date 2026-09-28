"""Local ranked-chunk CLI. No query, chunk text, filename or vector output."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from pydantic import ValidationError
from local_retrieval import RetrievalError, RetrievalService, RetrievalSettings
from local_retrieval.local_model import DEFAULT_MODEL_DIRECTORY, OnnxEmbedder


class SafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise RetrievalError("invalid_arguments")


def main(argv: list[str] | None = None) -> int:
    try:
        parser = SafeParser(description=__doc__)
        parser.add_argument("--database", type=Path)
        parser.add_argument("--index-database", type=Path)
        parser.add_argument("--model-directory", type=Path)
        parser.add_argument("--gpu-model-directory", type=Path)
        parser.add_argument("--device", choices=["auto", "cpu", "cuda", "directml"])
        parser.add_argument("--candidate-pool", type=int, default=50)
        parser.add_argument("--rrf-k", type=int, default=60)
        commands = parser.add_subparsers(dest="command", required=True)
        commands.add_parser("prepare-model")
        commands.add_parser("status")
        commands.add_parser("sync")
        all_command = commands.add_parser("index-all")
        all_command.add_argument("--rebuild", action="store_true")
        commands.add_parser("index").add_argument("document_id")
        commands.add_parser("remove").add_argument("document_id")
        search = commands.add_parser("search")
        search.add_argument("query")
        search.add_argument("--mode", choices=["lexical", "semantic", "hybrid"], default="hybrid")
        search.add_argument("--top-k", type=int, default=5)
        search.add_argument("--document-id", action="append", dest="document_ids")
        for command in commands.choices.values():
            command.add_argument("--device", choices=["auto", "cpu", "cuda", "directml"], default=argparse.SUPPRESS)
        args = parser.parse_args(argv)
        settings = RetrievalSettings(candidate_pool=args.candidate_pool, rrf_k=args.rrf_k)
        if args.command == "prepare-model":
            from local_retrieval.bootstrap import prepare_model
            from local_retrieval.execution import available_providers, target, PROVIDERS
            requested = args.device or "cpu"  # Original preparation still downloads CPU only.
            if requested == "auto":
                available = available_providers()
                requested = next((device for device in ("cuda", "directml", "cpu") if PROVIDERS[device] in available), "cpu")
            selected = target(requested)
            output = prepare_model(args.model_directory, artifact_kind=selected.artifact_kind)
        else:
            options = {}
            if args.device is not None:
                options["device"] = args.device
            if args.gpu_model_directory is not None:
                options["gpu_directory"] = args.gpu_model_directory
            embedder = OnnxEmbedder(args.model_directory or DEFAULT_MODEL_DIRECTORY, **options)
            service = RetrievalService(embedder, source_path=args.database,
                                       index_path=args.index_database, settings=settings)
            if args.command == "search":
                output = service.search(args.query, args.top_k, args.document_ids, args.mode).safe_metadata()
            elif args.command == "index":
                output = service.index_document(args.document_id)
            elif args.command == "index-all":
                output = service.index_all(rebuild=args.rebuild)
            elif args.command == "sync":
                output = service.sync()
            elif args.command == "remove":
                output = service.remove(args.document_id)
            else:
                output = service.status()
            if hasattr(embedder, "execution_metadata"):
                output["execution"] = embedder.execution_metadata()
        print(json.dumps(output, ensure_ascii=True))
        return 0
    except (RetrievalError, ValidationError, OSError, ValueError):
        error = sys.exc_info()[1]
        code = error.code if isinstance(error, RetrievalError) else "invalid_arguments"
        print(json.dumps({"error": code}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
