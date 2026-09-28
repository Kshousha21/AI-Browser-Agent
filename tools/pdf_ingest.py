"""Operator-only PDF management CLI. JSON output never contains document text."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from pydantic import ValidationError
from local_pdf import DocumentStore, IngestionSettings, PdfIngestError, PdfIngestionService


class _SafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise PdfIngestError("invalid_arguments")


def main(argv: list[str] | None = None) -> int:
    """No .env, provider settings, browser, audit payloads or network clients."""
    try:
        parser = _SafeParser(description=__doc__)
        parser.add_argument("--database", type=Path,
                            help="Private local store; inside this checkout use .local/")
        commands = parser.add_subparsers(dest="command", required=True)
        ingest = commands.add_parser("ingest")
        ingest.add_argument("pdf_path", type=Path)
        for name in IngestionSettings.model_fields:
            ingest.add_argument("--" + name.replace("_", "-"), type=int, default=None)
        for name in ("inspect", "delete"):
            commands.add_parser(name).add_argument("document_id")
        listing = commands.add_parser("list")
        listing.add_argument("--limit", type=int, default=20)
        listing.add_argument("--offset", type=int, default=0)
        args = parser.parse_args(argv)
        store = DocumentStore(args.database)
        if args.command == "ingest":
            values = {name: getattr(args, name) for name in IngestionSettings.model_fields
                      if getattr(args, name) is not None}
            result = PdfIngestionService(store, IngestionSettings(**values)).ingest_pdf(args.pdf_path)
            output = {"status": result.status, "document": result.document.safe_metadata()}
        elif args.command == "list":
            output = {"documents": [item.safe_metadata() for item in store.list_documents(limit=args.limit, offset=args.offset)]}
        elif args.command == "inspect":
            document = store.get(args.document_id)
            if document is None:
                raise PdfIngestError("document_not_found")
            output = {"document": document.safe_metadata()}
        else:
            if not store.delete(args.document_id):
                raise PdfIngestError("document_not_found")
            output = {"status": "deleted", "document_id": args.document_id}
        print(json.dumps(output, ensure_ascii=True))
        return 0
    except (PdfIngestError, ValidationError, OSError, ValueError):
        error = sys.exc_info()[1]
        code = error.code if isinstance(error, PdfIngestError) else "invalid_arguments"
        print(json.dumps({"error": code}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
