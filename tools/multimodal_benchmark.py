"""Synthetic-only local measurement; real models by default, no downloads.

--synthetic-runtime measures the deterministic contract harness, NOT model quality.
All PDFs/stores are temporary; only invented questions and safe metadata are emitted.
"""
import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from browser.config import load_retrieval_settings
from local_pdf import DocumentStore, PdfIngestionService
from local_pdf.rendering import LocalPageRenderer
from local_retrieval import RetrievalService
from local_retrieval.local_model import OnnxEmbedder
from local_visual_retrieval import VisualRetrievalService, VisualError
from local_visual_retrieval.fusion import MultimodalRetrieval
from tests.visual_retrieval_support import SyntheticVisualModel
from tests.retrieval_fixtures import TestEmbedder
from tests.visual_pdf_fixtures import research_pdf

async def benchmark(args, folder):
    # Reuse the existing invented chart/table/architecture PDF renderer fixture.
    from io import BytesIO
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject, NameObject
    reader = PdfReader(BytesIO(research_pdf()))
    writer = PdfWriter()
    for page in reader.pages: writer.add_page(page)
    # Add explicit invented thermal results for the manual relevance labels.
    for index in (0, 3):
        page = writer.pages[index]
        content = page.get_contents().get_data()
        content += b"\nBT /F1 12 Tf 35 700 Td (Synthetic thermal resistance: design A 2 K/W; design B 3 K/W. Temperatures 20 C and 40 C.) Tj ET"
        stream = DecodedStreamObject(); stream.set_data(content)
        page[NameObject("/Contents")] = writer._add_object(stream)
    path = folder / "invented-report.pdf"
    with path.open("wb") as stream: writer.write(stream)
    store = DocumentStore(folder / "documents.sqlite3")
    doc = PdfIngestionService(store).ingest_pdf(path).document
    text = RetrievalService(TestEmbedder() if args.synthetic_runtime else OnnxEmbedder(),
                            source_path=store.path, index_path=folder / "retrieval.sqlite3")
    text.index_all()
    renderer = LocalPageRenderer(store, lambda key: path if key == doc.document_id else None)
    settings = load_retrieval_settings()
    visual = VisualRetrievalService(renderer, source_path=store.path, index_path=folder / "visual.sqlite3",
        settings=settings.visual_retrieval, model=SyntheticVisualModel() if args.synthetic_runtime else None)
    report = {"mode":"synthetic-contract" if args.synthetic_runtime else "real-local-models",
              "synthetic_documents":1,"pages":4, "quality_claim":"No semantic quality claim from latency."}
    started = perf_counter()
    try:
        report["index"] = await visual.index_documents([doc.document_id])
        report["visual_runtime"] = visual.measurements
    except VisualError as error:
        report["visual_status"] = error.code
        report["visual_runtime"] = {"load_ms":None,"gpu_before_bytes":None,"gpu_after_load_bytes":None,"gpu_peak_bytes":None}
    report["visual_index_attempt_ms"] = round((perf_counter()-started)*1000,3)
    fused = MultimodalRetrieval(text, visual, settings.multimodal_retrieval)
    dataset = json.loads((Path(__file__).resolve().parents[1]/"resources/multimodal_relevance.json").read_text())
    cases = []
    for item in dataset:
        q = item["question"]; expected = set(item["expected_pages"])
        t = perf_counter(); text_result = text.search(q, 5, [doc.document_id]); text_ms = (perf_counter()-t)*1000
        result = await fused.search(q,[doc.document_id])
        text_pages = list(dict.fromkeys(hit.page_start for hit in text_result.results))
        visual_pages = [hit.page for hit in result.visual_results]
        fused_pages = [page.page for page in result.pages]
        available = result.status["visual"] == "ok"
        def first_rank(pages):
            return next((rank for rank, page in enumerate(pages,1) if page in expected),None)
        cases.append({"first_relevant_rank":{"text":first_rank(text_pages), "visual":first_rank(visual_pages),
                                              "fused":first_rank(fused_pages)}, **item,"text_pages":text_pages,"visual_pages":visual_pages,"fused_pages":fused_pages,
            "text_label_hit": bool(expected & set(text_pages)),
            "visual_label_hit": bool(expected & set(visual_pages)) if available else None,
            "fusion_label_coverage_gain":len(expected & set(fused_pages))-len(expected & set(text_pages)) if available else None,
            "manual_usefulness_review":"pending", "standalone_text_ms":round(text_ms,3),
            "status":result.status,"measurements":result.measurements, "visual_query_runtime":dict(visual.measurements)})
    report["cases"] = cases
    if args.answer:
        from browser.config import load_settings
        from local_rag import RagService, RagSettings, ProviderBridge, RagError
        provider = load_settings().provider
        if provider.name != "ollama" or not provider.local_only:
            raise ValueError("local benchmark provider required")
        service = RagService(text, ProviderBridge(provider), RagSettings(timeout_seconds=120.0),
                             renderer=renderer, multimodal=fused)
        try:
            answer = await service.answer(dataset[0]["question"], document_ids=[doc.document_id])
            report["local_rag"] = {"status":answer.status, "answer_chars":len(answer.answer),
                "citations":[{"page":c.page_start,"evidence_type":c.evidence_type} for c in answer.citations],
                "retrieval_status":answer.retrieval_status,"measurements":answer.measurements}
        except RagError as error:
            report["local_rag"] = {"error":error.code}
    report["limitations"] = ["Small invented corpus; label coverage is not expert relevance judgment.",
                             "Optional --answer uses only the configured local Ollama generator."]
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-runtime",action="store_true")
    parser.add_argument("--answer",action="store_true",help="Also validate one synthetic answer using the configured local-only Ollama provider")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(prefix="multimodal-synthetic-") as folder:
            result=asyncio.run(benchmark(args,Path(folder)))
        data=json.dumps(result,indent=2)
        if args.output: args.output.write_text(data,encoding="utf-8")
        print(data)
        return 0
    except Exception:
        print(json.dumps({"error":"synthetic_benchmark_unavailable"}));return 2

if __name__=="__main__":raise SystemExit(main())
