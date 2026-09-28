"""Measure one explicit local device in its own runtime for cross-environment comparisons."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from retrieval_acceptance import timed
from retrieval_offline_guard import offline_guard
from local_retrieval import RetrievalError, RetrievalService
from local_retrieval.local_model import OnnxEmbedder


def measure(device: str, maximum: int) -> dict[str, object]:
    from tests.retrieval_fixtures import CORPUS, ingest_texts
    with offline_guard() as attempts:
        model = OnnxEmbedder(device=device)
        start = perf_counter()
        model._load()
        cold = (perf_counter() - start) * 1000
        _, query = timed(lambda: model.encode(["overheating mitigation"]))
        _, batch = timed(lambda: model.encode(list(CORPUS.values()) * 6), repeats=3)
        result = {"embedding": model.identity.model_dump(), "execution": model.execution_metadata(),
                  "cold_load_including_probe_ms": round(cold, 3), "one_query_median_ms": query,
                  "embedding_batch_inputs": 30, "batch_median_ms": batch, "indexes": []}
        with TemporaryDirectory(prefix="local-device-benchmark-") as temporary:
            folder = Path(temporary)
            for count in (100, 1000):
                if count > maximum:
                    continue
                texts = {f"synthetic{index}": list(CORPUS.values())[index % len(CORPUS)] + f" Synthetic record {index}." for index in range(count)}
                store, _ = ingest_texts(folder, texts)
                service = RetrievalService(model, source_path=store.path, index_path=folder / f"index-{count}.sqlite3")
                _, indexing = timed(service.index_all, repeats=1)
                assert service.status()["chunks"] == count
                _, semantic = timed(lambda: service.search("thermal packaging", mode="semantic"), repeats=3)
                _, hybrid = timed(lambda: service.search("thermal packaging", mode="hybrid"), repeats=3)
                result["indexes"].append({"documents": count, "chunks": count, "index_ms": indexing,
                    "semantic_median_ms": semantic, "hybrid_median_ms": hybrid,
                    "database_bytes": service.index.path.stat().st_size})
                print(json.dumps({"progress": "synthetic_index_measured", "device": device, "chunks": count}), file=sys.stderr)
        assert attempts == []
        return {"status": "passed", **result, "python_network_or_provider_attempts": len(attempts)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "directml"], required=True)
    parser.add_argument("--max-chunks", type=int, choices=[100, 1000], default=1000)
    args = parser.parse_args()
    try:
        print(json.dumps(measure(args.device, args.max_chunks)))
    except RetrievalError as error:
        print(json.dumps({"status": "not_verified", "error": error.code}))
        raise SystemExit(2)
