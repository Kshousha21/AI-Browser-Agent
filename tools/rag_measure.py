"""Offline synthetic RAG measurements, explicitly using TEST vectors/provider.

This acceptance helper is not imported by production code. No real LLM timing
or learned multilingual quality is measured. It accepts no private input files.
"""
import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from tests.rag_fixtures import corpus, rag


async def measure(folder):
    pages = {f"synthetic{i}": [f"Synthetic thermal package {i} uses 12 W. " +
                              "Invented copper evidence for this bounded benchmark. " * 22]
             for i in range(20)}
    retrieval, _, documents = corpus(folder, pages=pages)
    measurements = []
    for top_k in (3, 5, 10):
        service, _ = rag(retrieval)
        result = await service.answer("thermal package power", top_k=top_k)
        measurements.append({"top_k": top_k, "retrieved_count": result.retrieved_count,
            "context_count": result.context_count, "dropped_due_budget": result.dropped_due_budget,
            **result.measurements})
    return {"status": "OFFLINE SYNTHETIC FIXTURE ONLY", "corpus_documents": len(documents),
            "corpus_chars": sum(doc.extracted_chars for doc in documents.values()), "measurements": measurements}


def main():
    folder = ROOT / ".local"
    folder.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="rag-measure-", dir=folder) as temporary:
        output = asyncio.run(measure(Path(temporary)))
    print(json.dumps(output))


if __name__ == "__main__":
    main()
