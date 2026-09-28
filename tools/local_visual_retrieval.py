"""Explicit offline page-index operations. Output contains metadata only."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from local_visual_retrieval.schemas import VisualError

from local_visual_retrieval.cli import SafeParser, components, options

async def execute(args):
    visual, _, _ = components(args)
    if args.command == "status": return await visual.status()
    if args.command == "search":
        hits = await visual.search(args.query, args.top_k, args.document_ids)
        return {"results": [hit.model_dump() for hit in hits], "measurements": visual.measurements}
    return await visual.index_documents([args.document_id] if args.command == "index" else None,
                                         rebuild=args.command == "rebuild")

def main(argv=None):
    try:
        parser = SafeParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        for name in ("status", "index-all", "index", "search", "rebuild"):
            command = commands.add_parser(name)
            options(command)
            if name == "index": command.add_argument("document_id")
            if name == "search":
                command.add_argument("query")
                command.add_argument("--top-k", type=int, default=5)
                command.add_argument("--document-id", action="append", dest="document_ids")
        result = asyncio.run(execute(parser.parse_args(argv)))
        print(json.dumps(result))
        return 0
    except VisualError as error:
        print(json.dumps({"error": error.code})); return 2
    except Exception:
        print(json.dumps({"error": "visual_operation_unavailable"})); return 2

if __name__ == "__main__": raise SystemExit(main())
