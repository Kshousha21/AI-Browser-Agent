"""Search both indexes and print fused page metadata, never document body text."""
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from local_visual_retrieval.cli import SafeParser, options, components
from local_visual_retrieval.schemas import VisualError

async def execute(args):
    _, fused, _ = components(args)
    # This operator CLI historically permits searching all explicitly ingested PDFs.
    ids = args.document_ids
    if not ids:
        with fused.text.source.snapshot() as source: ids = sorted(source)
    result = await fused.search(args.query, ids, args.top_k)
    return {"status": result.status, "measurements": result.measurements,
            "results": [{"rank": rank, **page.safe_metadata()} for rank, page in enumerate(result.pages, 1)]}

def main(argv=None):
    try:
        parser = SafeParser(description=__doc__)
        command = parser.add_subparsers(dest="command", required=True).add_parser("search")
        options(command)
        command.add_argument("query")
        command.add_argument("--top-k", type=int, default=5)
        command.add_argument("--document-id", action="append", dest="document_ids")
        print(json.dumps(asyncio.run(execute(parser.parse_args(argv)))))
        return 0
    except VisualError as error:
        print(json.dumps({"error": error.code})); return 2
    except Exception:
        print(json.dumps({"error": "multimodal_retrieval_unavailable"})); return 2

if __name__ == "__main__": raise SystemExit(main())
