"""Explicit synthetic local model acceptance; no private config or browser actions.

Tests configured dispatch against all 17 schemas, tool-result continuation and JSON.
This proves a bounded contract, not general browser planning or website quality.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.site_adapters import SiteAdapterRegistry
from browser.agent_runtime import available_tools, prepare_tool_call
from browser.actions import ARGUMENT_MODELS
from app.site_adapters.page_files import PageFilesArgs
from browser.config import SiteAdaptersSettings
from browser.provider_resolution import resolve_provider
from browser.providers import Message, ToolCall, build_provider
from local_rag.models import ProviderAnswer

URL = "https://example.org/synthetic"
CASES = {
    "get_page_state": {"full": True},
    "read_page": {"max_chars": 1000},
    "get_element_details": {"element_id": "f0:e-002"},
    "click": {"element_id": "f0:e-003"},
    "type_text": {"element_id": "f0:e-002", "text": "photonics", "submit": False},
    "clear_input": {"element_id": "f0:e-002"},
    "press_key": {"key": "Escape"},
    "scroll": {"direction": "down", "amount": 600},
    "navigate": {"url": URL},
    "get_tabs": {},
    "switch_tab": {"tab_id": 1},
    "open_tab": {"url": URL},
    "close_tab": {"tab_id": 1},
    "wait_for_stable": {"timeout_ms": 10000},
    "find_text": {"text": "photonics"},
    "go_back": {}, "go_forward": {}, "refresh": {},
    "get_page_files": {"limit": 5, "offset": 0},
}


def equivalent_arguments(name, actual, expected):
    # Omitted optional defaults are valid API inputs, not missing user values.
    models = {**ARGUMENT_MODELS,
              "get_page_files": PageFilesArgs}
    model = models[name]
    return model.model_validate(actual).model_dump() == model.model_validate(expected).model_dump()


def local_settings(args):
    return resolve_provider({"active": "local"}, {"local": {
        "type": "ollama", "model": args.model, "base_url": args.base_url,
        "auth": {"mode": "none"}, "local_only": True, "num_ctx": args.num_ctx,
        "think": False, "tool_mode": args.tool_mode, "temperature": 0, "max_output_tokens": 1024,
        "timeout_seconds": 60, "max_retries": 0}}, {})


async def acceptance(args):
    settings = local_settings(args)
    registry = SiteAdapterRegistry(SiteAdaptersSettings())
    schemas = available_tools(URL, registry)
    if {tool.name for tool in schemas} != set(CASES):
        raise ValueError("inventory_changed")
    provider = build_provider(settings)
    outcomes = []
    async def record(name, operation):
        start = perf_counter()
        try:
            passed = bool(await operation())
            code = None
        except Exception as exc:
            passed = False
            code = getattr(exc, "code", "contract_failed")
        row = {"case": name, "passed": passed, "seconds": round(perf_counter()-start, 2)}
        if code:
            row["code"] = code
        outcomes.append(row)
        print(json.dumps(row), flush=True)
    try:
        for name, arguments in CASES.items():
            async def check(name=name, arguments=arguments):
                response = await provider.generate([
                    Message(role="system", content=(
                        "Synthetic tool protocol test. Execute no real action. Request exactly one tool call "
                        "using the configured response format. Use snapshot_id=s-synthetic and a concise reason. "
                        "All IDs/URLs are invented fixture values. The requested call tests serialization, "
                        "not permission to execute. Pass all requested arguments exactly.")),
                    Message(role="user", content=f"Call {name} with arguments {json.dumps(arguments)}.")
                ], tools=schemas)
                if not isinstance(response.decision, ToolCall):
                    return False
                prepare_tool_call(URL, response.decision, registry)
                return (response.decision.tool == name and equivalent_arguments(name, response.decision.arguments, arguments)
                        and response.decision.snapshot_id == "s-synthetic")
            await record(name, check)

        async def continuation():
            messages = [
                Message(role="system", content=(
                    "Use one tool at a time with the configured response format. Current snapshot_id is s-synthetic. "
                    "Search input is f0:e-002; Search button is f0:e-003. First type_text with "
                    "submit=false, then click Search, then answer from the tool result. No other steps.")),
                Message(role="user", content="Search for photonics and tell me how many results were found."),
            ]
            for expected in ("type_text", "click"):
                response = await provider.generate(messages, tools=schemas)
                call = response.decision
                if not isinstance(call, ToolCall) or call.tool != expected:
                    return False
                prepare_tool_call(URL, call, registry)
                if call.snapshot_id != "s-synthetic":
                    return False
                if expected == "type_text" and (call.arguments.get("text") != "photonics"
                        or call.arguments.get("element_id") != "f0:e-002" or call.arguments.get("submit", False)):
                    return False
                if expected == "click" and call.arguments.get("element_id") != "f0:e-003":
                    return False
                messages += [
                    Message(role="assistant", content=response.raw_text, tool_call_id=response.tool_call_id,
                            tool_name=call.tool, tool_arguments={**call.arguments, "snapshot_id": call.snapshot_id, "reason": call.reason}),
                    Message(role="tool", content=json.dumps({"success": True, "snapshot_id": "s-synthetic",
                            "message": "Text entered." if expected == "type_text" else "Found 4 synthetic results."}),
                            tool_call_id=response.tool_call_id, tool_name=call.tool),
                ]
            response = await provider.generate(messages, tools=schemas)
            return response.decision.type == "final" and (
                "4" in response.decision.answer or "four" in response.decision.answer.lower())
        await record("multi_step_continuation", continuation)

        async def structured():
            response = await provider.generate([Message(role="user", content=(
                'Return JSON: status=answered, answer="Synthetic package uses 12 W.", '
                'citations=[{"source_id":"S1","evidence":"uses 12 W"}].'))],
                response_schema=ProviderAnswer)
            result = ProviderAnswer.model_validate_json(response.raw_text)
            return result.status == "answered" and result.citations[0].evidence == "uses 12 W"
        await record("structured_json", structured)
    finally:
        await provider.aclose()
    return outcomes


async def visual_acceptance(args):
    """Actual local renderer/RAG on a generated image-only PDF, no private data."""
    from tempfile import TemporaryDirectory
    from local_pdf import DocumentStore, PdfIngestionService
    from local_pdf.rendering import LocalPageRenderer
    from local_retrieval import RetrievalService
    from local_retrieval.local_model import OnnxEmbedder
    from local_rag import ProviderBridge, RagService
    from tests.visual_pdf_fixtures import research_pdf
    settings = local_settings(args).model_copy(update={"supports_images": True})
    with TemporaryDirectory(prefix="local-vision-acceptance-") as temporary:
        folder = Path(temporary)
        path = folder / "invented-scanned-chart.pdf"
        path.write_bytes(research_pdf("image_only"))
        store = DocumentStore(folder / "documents.sqlite3")
        doc = PdfIngestionService(store).ingest_pdf(path).document
        retrieval = RetrievalService(OnnxEmbedder(device="cpu"), source_path=store.path,
                                     index_path=folder / "retrieval.sqlite3")
        retrieval.index_all()
        renderer = LocalPageRenderer(store, lambda identifier: path if identifier == doc.document_id else None)
        result = await RagService(retrieval, ProviderBridge(settings), renderer=renderer).answer(
            "Read the chart on page 1. What power is shown at 20 C and at 40 C?",
            document_ids=[doc.document_id])
        passed = (doc.extracted_chars == 0 and result.status == "answered"
                  and result.measurements["visual_pages"] == 1
                  and any(c.evidence_type == "visual" and c.page_start == 1 for c in result.citations)
                  and "6" in result.answer and "9" in result.answer)
        row = {"case": "scanned_chart_local_rag", "passed": passed,
               "extracted_chars": doc.extracted_chars, "visual_pages": result.measurements["visual_pages"],
               "citation_count": len(result.citations), "attempts": result.attempts}
        print(json.dumps(row), flush=True)
        return [row]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--vision-only", action="store_true", help="Check only an invented scanned chart through local RAG; requires prepared CPU embeddings.")
    parser.add_argument("--tool-mode", choices=["native", "structured"], default="structured")
    parser.add_argument("--model", default="qwen3-vl:8b-instruct-q4_K_M")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--num-ctx", type=int, default=32768)
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({"status": "NOT VERIFIED", "reason": "explicit_run_required"}))
        return 3
    try:
        outcomes = asyncio.run(visual_acceptance(args) if args.vision_only else acceptance(args))
    except Exception:
        print(json.dumps({"status": "NOT VERIFIED", "reason": "local_configuration_failed"}))
        return 3
    passed = all(row["passed"] for row in outcomes)
    print(json.dumps({"status": "PASS" if passed else "FAIL",
                      "passed": sum(row["passed"] for row in outcomes), "total": len(outcomes),
                      "browser_execution": "not_performed", "private_context": "not_used"}))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())

