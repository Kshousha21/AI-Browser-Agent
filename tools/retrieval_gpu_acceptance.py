"""Explicit real GPU acceptance/CPU comparison; never a mandatory pytest skip."""
import argparse
import gc
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from retrieval_acceptance import acceptance, timed
from retrieval_offline_guard import offline_guard
from local_retrieval import RetrievalError, RetrievalService
from local_retrieval.execution import capabilities
from local_retrieval.local_model import OnnxEmbedder, DEFAULT_MODEL_DIRECTORY

NOT_AVAILABLE = "GPU ACCELERATION NOT AVAILABLE ON THIS WORKSTATION"


def compare(device: str, maximum: int) -> dict[str, object]:
    from tests.retrieval_fixtures import CORPUS, ingest_texts
    results = {}
    models = {}
    for selected in ("cpu", device):
        gc.collect()
        model = OnnxEmbedder(device=selected)
        start = perf_counter()
        model._load()  # Includes local checksums; GPU also includes its required compatibility probe.
        cold = (perf_counter() - start) * 1000
        _, query = timed(lambda: model.encode(["overheating mitigation"]))
        _, batch = timed(lambda: model.encode(list(CORPUS.values()) * 6), repeats=3)
        results[selected] = {"cold_load_including_probe_ms": round(cold, 3), "one_query_median_ms": query,
                             "embedding_batch_inputs": 30, "batch_median_ms": batch,
                             "execution": model.execution_metadata(), "indexes": []}
        models[selected] = model
    with TemporaryDirectory(prefix="gpu-retrieval-benchmark-") as temporary:
        folder = Path(temporary)
        for count in (100, 1000):
            if count > maximum:
                continue
            texts = {f"synthetic{index}": list(CORPUS.values())[index % len(CORPUS)] + f" Synthetic record {index}." for index in range(count)}
            store, _ = ingest_texts(folder, texts)
            for selected in ("cpu", device):
                service = RetrievalService(models[selected], source_path=store.path, index_path=folder / f"{selected}-{count}.sqlite3")
                _, indexing = timed(service.index_all, repeats=1)
                status = service.status()
                assert status["chunks"] == count
                _, semantic = timed(lambda: service.search("thermal packaging", mode="semantic"), repeats=3)
                _, hybrid = timed(lambda: service.search("thermal packaging", mode="hybrid"), repeats=3)
                results[selected]["indexes"].append({"documents": count, "chunks": count, "index_ms": indexing,
                    "semantic_median_ms": semantic, "hybrid_median_ms": hybrid,
                    "database_bytes": service.index.path.stat().st_size})
                print(json.dumps({"progress": "synthetic_index_measured", "device": selected, "chunks": count}), file=sys.stderr)
    ratios = {key: round(results["cpu"][key] / results[device][key], 4) for key in
              ("cold_load_including_probe_ms", "one_query_median_ms", "batch_median_ms")}
    ratios["indexes"] = [{"chunks": cpu["chunks"], **{key: round(cpu[key] / gpu[key], 4) for key in
                         ("index_ms", "semantic_median_ms", "hybrid_median_ms")}}
                         for cpu, gpu in zip(results["cpu"]["indexes"], results[device]["indexes"], strict=True)]
    return {"measurements": results, "speedup_cpu_divided_by_gpu": ratios,
            "cpu_reference_runtime": "CPUExecutionProvider in the same optional runtime; base CPU evidence remains separate"}


def run(device: str, maximum: int) -> dict[str, object]:
    with offline_guard() as attempts:
        discovered = capabilities()
        if not discovered["available_devices"]:
            return {"status": NOT_AVAILABLE, "capabilities": discovered}
        selection = OnnxEmbedder(device=device)
        metadata = selection.execution_metadata()
        selected = metadata["selected_device"]
        if selected == "cpu":
            return {"status": "GPU ACCEPTANCE NOT RUN; AUTO SELECTED CPU", "execution": metadata}
        del selection
        gc.collect()
        verified = acceptance(DEFAULT_MODEL_DIRECTORY, selected)
        gc.collect()
        comparison = compare(selected, maximum)
        assert attempts == []
        return {"status": "GPU ACCEPTANCE PASSED", "execution": metadata,
                "synthetic_acceptance": verified, "comparison": comparison,
                "python_network_or_provider_attempts": len(attempts)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cuda", "directml"], default="auto")
    parser.add_argument("--max-chunks", type=int, choices=[100, 1000], default=1000)
    args = parser.parse_args()
    try:
        result = run(args.device, args.max_chunks)
        print(json.dumps(result))
        raise SystemExit(0 if result["status"] == "GPU ACCEPTANCE PASSED" else 2)
    except RetrievalError as error:
        print(json.dumps({"status": "GPU ACCEPTANCE NOT VERIFIED", "error": error.code}))
        raise SystemExit(2)
    except Exception:
        print(json.dumps({"status": "GPU ACCEPTANCE FAILED", "error": "synthetic_acceptance_failed"}))
        raise SystemExit(2)
