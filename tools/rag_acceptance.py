"""Optional REAL provider acceptance using only fixed invented PDFs/questions.

--run explicitly authorizes contacting the selected endpoint. Remote endpoints
also require --allow-remote-context. Local embedding artifacts must be prepared
already; this command never downloads them. No private PDF/query arguments.
Fixtures normally use the production data-root policy in a newly owned
temporary subdirectory, never existing private stores. --local-temp remains an
explicit test-only override; neither mode changes remote-context consent.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from browser.config import load_provider_settings
from local_retrieval.local_model import DEFAULT_MODEL_DIRECTORY, OnnxEmbedder
from local_rag import ProviderBridge, RagError, RagService, RagSettings
from local_rag.models import INSUFFICIENT
from local_storage import data_root
from tests.rag_fixtures import corpus  # Invented PDFs only; real embedder is supplied below.

NOT_VERIFIED = "REAL RAG PROVIDER ACCEPTANCE NOT VERIFIED"
CASES = [
    ("single_numeric", "How much power does Lumen use?", ["lumen"], {("lumen", 1)}),
    ("multiple_pages", "Which plate and housing materials does Lumen use?", ["lumen"], {("lumen", 1), ("lumen", 2)}),
    ("irrelevant", "How many moons does planet Wug have?", ["lumen"], set()),
    ("conflict", "What maximum power do the Vega sources report? Explain any conflict.", ["conflict_a", "conflict_b"], {("conflict_a", 1), ("conflict_b", 1)}),
    ("de_over_en", "Wie viel Leistung verwendet Lumen?", ["lumen"], {("lumen", 1)}),
    ("en_over_de", "How much power does Nara use?", ["german"], {("german", 1)}),
    ("injection", "Does the manual contain instruction-like text? Describe it only as text.", ["injection"], {("injection", 1)}),
]


async def acceptance(folder, bridge, args):
    embedder = OnnxEmbedder(args.model_directory, device=args.device)
    retrieval, store, docs = corpus(folder, embedder=embedder)
    service = RagService(retrieval, bridge)
    outcomes = []
    for name, question, filters, expected in CASES:
        try:
            result = await service.answer(question, document_ids=[docs[k].document_id for k in filters])
        except RagError as error:
            # Preserve every case's result without exposing answers or SDK
            # bodies. Arbitrary errors NEVER become accepted abstentions.
            outcomes.append({"case":name, "passed":False, "status":"error", "error_category":error.code})
            continue
        expected_refs = {(docs[k].document_id, page) for k, page in expected}
        actual_refs = {(c.document_id, c.page_start) for c in result.citations}
        refusal = result.measurements.get("provider_refusal", 0) == 1
        safe_abstention = (result.status == "insufficient_evidence" and
                           result.answer == INSUFFICIENT and not result.citations)
        passed = (result.status == ("answered" if expected else "insufficient_evidence")
                  and expected_refs <= actual_refs)
        if not expected:
            passed = passed and safe_abstention
        # Security canary accepts ONLY an explicit refusal mapped to the fixed
        # local abstention (not generic errors or arbitrary insufficient prose).
        if name == "injection" and refusal:
            passed = safe_abstention and result.attempts == 1
        selected = {docs[k].document_id for k in filters}
        passed = passed and all(c.document_id in selected for c in result.citations)
        for citation in result.citations:
            chunks = store.chunks(citation.document_id)
            hit = next((c for c in chunks if c.chunk_id == citation.chunk_id), None)
            passed = passed and hit is not None and all(
                getattr(hit, key) == getattr(citation, key) for key in
                ("page_start", "page_end", "chunk_index", "char_start", "char_end", "trust"))
        outcomes.append({"case": name, "passed": passed, "status": result.status,
                         "citation_count": len(result.citations), "attempts": result.attempts,
                         "provider_refusal": refusal})
    return outcomes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--provider-profile")
    parser.add_argument("--allow-remote-context", action="store_true")
    parser.add_argument("--model-directory", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "directml"], default="auto")
    parser.add_argument("--local-temp", action="store_true",
                        help="Use the system temporary directory for invented acceptance fixtures only")
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({"status": NOT_VERIFIED, "reason": "explicit_run_required"}))
        return 3
    try:
        settings = load_provider_settings(args.config, args.provider_profile)
        bridge = ProviderBridge(settings, allow_remote_context=args.allow_remote_context)
        bridge.settings_for_request(RagSettings())  # Gate BEFORE synthetic indexing.
        folder = None if args.local_temp else data_root()
        if folder is not None:
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="rag-acceptance-", dir=folder) as temporary:
            outcomes = asyncio.run(acceptance(Path(temporary), bridge, args))
        passed = all(item["passed"] for item in outcomes)
        print(json.dumps({"status": "REAL RAG SYNTHETIC CONTRACT CHECKS PASSED" if passed else NOT_VERIFIED,
                          "cases": outcomes, "semantic_quality": "requires_operator_review"}))
        return 0 if passed else 3
    except Exception:
        print(json.dumps({"status": NOT_VERIFIED, "reason": "configuration_artifact_or_provider_check_failed"}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
