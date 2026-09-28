"""Explicit real-model acceptance and bounded synthetic timing; never a pytest skip."""
import argparse
import json
from pathlib import Path
from statistics import median
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from retrieval_offline_guard import offline_guard


def timed(operation, repeats=5):
    elapsed = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = operation()
        elapsed.append((perf_counter() - start) * 1000)
    return result, round(median(elapsed), 3)


def vector_scale(original, dimension):
    """Vector scan only: excludes PDF/index I/O and learned embedding generation."""
    import numpy as np
    from local_retrieval.index_store import IndexedChunk
    from local_retrieval.ranking import semantic
    from local_retrieval.vectors import normalized
    rng = np.random.default_rng(404)
    output = []
    for count in (100, 1000, 10000):
        matrix = rng.normal(size=(count, dimension)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        chunks = {}
        for index, vector in enumerate(matrix):
            identifier = f"{index:064x}"
            chunk = original.model_copy(update={"chunk_id": identifier, "document_id": identifier, "chunk_index": 0})
            chunks[identifier] = IndexedChunk(chunk, identifier, vector.astype("<f4").tobytes())
        query = normalized(rng.normal(size=dimension), dimension)
        result, elapsed = timed(lambda: semantic(chunks, query, 50))
        assert len(result[0]) == 50
        output.append({"vectors": count, "dimension": dimension, "vector_bytes": count * dimension * 4,
                       "scan_median_ms": elapsed, "repeats": 5})
    return output


def acceptance(directory, device="cpu"):
    with offline_guard() as attempts:
        import numpy as np
        from local_retrieval.local_model import OnnxEmbedder
        from tests.retrieval_fixtures import fixture_service
        with TemporaryDirectory(prefix="phase4d-synthetic-") as temporary:
            folder = Path(temporary)
            embedder = OnnxEmbedder(directory, device=device)
            service, store, documents, _ = fixture_service(folder, embedder=embedder)
            _, index_ms = timed(service.index_all, repeats=1)
            status = service.status()
            checks = []

            def first(query, topic, mode="semantic", document_ids=None):
                result = service.search(query, mode=mode, document_ids=document_ids)
                assert result.results[0].document_id == documents[topic].document_id
                checks.append({"case": topic + "_" + mode, "passed": True})
                return result

            first("xylophotonic9", "optics", "lexical")
            assert service.search("overheating mitigation", mode="lexical").returned_count == 0
            first("overheating mitigation", "cooling")
            first("thermal packaging cooling", "cooling", "hybrid")
            selected = [documents[key].document_id for key in ("german", "agriculture", "lidar")]
            first("Optical connections carry information using photons in data centers", "german", document_ids=selected)
            first("Wie kann man die Temperatur von Prozessoren senken und Waerme abfuehren?", "cooling")
            result = first("photonics packaging", "optics", document_ids=[documents["optics"].document_id])
            assert result.returned_count == 1
            repeated = service.search("overheating mitigation")
            assert repeated == service.search("overheating mitigation")
            vectors = np.asarray(embedder.encode(["Synthetic optical links", "Synthetische optische Verbindungen"]), dtype=np.float32)
            assert vectors.shape == (2, 384) and np.isfinite(vectors).all()
            assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
            # Overflow windows contribute, rather than silently discarding the tail.
            prefix = "Optical interconnect packaging. " * 70
            long_vectors = np.asarray(embedder.encode([prefix + "Agricultural crops irrigation. " * 70,
                                                       prefix + "Thermal cooling processor heat. " * 70]))
            assert np.linalg.norm(long_vectors[0] - long_vectors[1]) > 0.01
            times = {}
            for mode in ("lexical", "semantic", "hybrid"):
                _, times[mode + "_median_ms"] = timed(lambda: service.search("thermal packaging", mode=mode))
            scale = vector_scale(store.chunks(documents["optics"].document_id)[0], 384)
            assert not attempts
            output = {"status": "passed", "embedding": embedder.identity.model_dump(), "execution": embedder.execution_metadata(),
                      "checks": checks, "finite_unit_vectors": True, "overflow_windows_verified": True,
                      "repeat_ranking_stable": True, "python_network_or_provider_attempts": len(attempts),
                      "documents": status["documents"], "chunks": status["chunks"],
                      "indexing_including_cold_model_load_ms": index_ms,
                      "retrieval_database_bytes": service.index.path.stat().st_size,
                      "source_database_bytes": store.path.stat().st_size,
                      "query_repeats": 5, "timings": times, "vector_scan_only": scale}
            return output


def main():
    from local_retrieval import RetrievalError
    from local_retrieval.local_model import DEFAULT_MODEL_DIRECTORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-directory", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda", "directml"], default="cpu")
    args = parser.parse_args()
    try:
        print(json.dumps(acceptance(args.model_directory, args.device)))
        return 0
    except RetrievalError as error:
        print(json.dumps({"status": "not_verified", "error": error.code}))
        return 2
    except Exception:
        print(json.dumps({"status": "failed", "error": "synthetic_acceptance_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
