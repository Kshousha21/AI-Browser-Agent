"""Synthetic-only release/store drills and capacity measurements; no live inputs.

No PDF/query/provider/URL inputs, downloads, migrations or production-store access.
seed/check accept only a test-owned clone beneath .test-artifacts/. measure creates
and cleans its own temporary synthetic folder beneath ignored .local/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["seed", "check", "measure"])
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args()
    source = args.source_root.resolve()
    if source != ROOT and not source.is_relative_to(ROOT / ".test-artifacts"):
        parser.error("source root must be an isolated checkout under .test-artifacts")
    if args.mode != "measure" and source == ROOT:
        parser.error("release drills require an isolated source checkout")
    sys.path.insert(0, str(source / "backend"))
    from local_pdf import DocumentStore, PdfIngestionService
    from local_retrieval import RetrievalService, RetrievalError
    from tests.pdf_fixtures import pdf_bytes
    from tests.retrieval_fixtures import TestEmbedder
    from retrieval_offline_guard import offline_guard

    def service(folder, embedder=None):
        return RetrievalService(embedder or TestEmbedder(dimension=384),
            source_path=folder / "documents.sqlite3", index_path=folder / "retrieval.sqlite3")

    def populate(folder, count, pages=2):
        store = DocumentStore(folder / "documents.sqlite3")
        ingest = PdfIngestionService(store)
        total, chunks = 0, 0
        for n in range(count):
            path = folder / ("synthetic-%04d.pdf" % n)
            text = ("Synthetic photonics calibration sample %d: 12.5 mm and 300 K.\n" % n) * 20
            path.write_bytes(pdf_bytes([text] * pages))
            doc = ingest.ingest_pdf(path).document
            total += path.stat().st_size
            chunks += doc.chunk_count
        return {"documents": count, "pages_per_document": pages, "chunks": chunks,
                "browser_original_bytes": total}

    with offline_guard() as attempts:
        if args.mode in {"seed", "check"}:
            folder = source / ".local" / "phase5-release-drill"
            marker = folder / "synthetic-owner.json"
            if args.mode == "seed":
                folder.mkdir(parents=True, exist_ok=False)
                populate(folder, 2)
                service(folder).index_all()
                (folder / "provenance.json").write_text('{"synthetic":true}', encoding="utf-8")
                (folder / "model-cache-sentinel").write_bytes(b"synthetic sentinel, not model weights")
                paths = [p for p in folder.iterdir() if p.is_file()]
                marker.write_text(json.dumps({"owner": "phase5-synthetic-only",
                    "hashes": {p.name: digest(p) for p in paths}}), encoding="utf-8")
                print(json.dumps({"release_store_seed": "PASS", "documents": 2}))
            else:
                saved = json.loads(marker.read_text(encoding="utf-8"))
                assert saved["owner"] == "phase5-synthetic-only"
                hashes = saved["hashes"]
                assert all(Path(name).name == name for name in hashes)
                assert all(digest(folder / name) == value for name, value in hashes.items())
                for name in ("documents.sqlite3", "retrieval.sqlite3"):
                    with sqlite3.connect((folder / name).as_uri() + "?mode=ro", uri=True) as db:
                        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                found = service(folder).search("synthetic photonics", mode="hybrid")
                assert found.returned_count > 0
                incompatible = service(folder, TestEmbedder(revision="incompatible-drill", dimension=384))
                try:
                    incompatible.search("synthetic photonics")
                except RetrievalError as error:
                    assert error.code == "index_model_mismatch"
                else:
                    raise AssertionError("incompatible index accepted")
                assert all(digest(folder / name) == value for name, value in hashes.items())
                print(json.dumps({"release_store_check": "PASS", "sqlite_integrity": "ok",
                    "compatible_search": "PASS", "incompatible_index": "REJECTED",
                    "source_index_originals_provenance_sentinel_bytes": "UNCHANGED"}))
        else:
            local = ROOT / ".local"
            local.mkdir(exist_ok=True)
            with TemporaryDirectory(prefix="phase5-synthetic-", dir=local) as temporary:
                root = Path(temporary)
                capacity = []
                for count in (10, 100, 1000):
                    folder = root / str(count)
                    folder.mkdir()
                    row = populate(folder, count)
                    service(folder).index_all()
                    row.update(pdf_db_bytes=(folder / "documents.sqlite3").stat().st_size,
                               index_bytes=(folder / "retrieval.sqlite3").stat().st_size)
                    capacity.append(row)
                # Actual accepted CPU embedder, fixed generated inputs only.
                from local_retrieval.local_model import OnnxEmbedder
                timings = []
                for label, pages in (("small", 2), ("medium", 40)):
                    folder = root / label
                    folder.mkdir()
                    start = perf_counter()
                    row = populate(folder, 1, pages=pages)
                    ingestion = perf_counter() - start
                    engine = service(folder, OnnxEmbedder(device="cpu"))
                    start = perf_counter()
                    engine.index_all()
                    indexing = perf_counter() - start
                    start = perf_counter()
                    result = engine.search("synthetic photonics temperature")
                    retrieval = perf_counter() - start
                    timings.append({"fixture": label, **row,
                        "generation_and_ingestion_seconds": round(ingestion, 4),
                        "initialization_and_index_seconds": round(indexing, 4),
                        "warm_retrieval_seconds": round(retrieval, 4),
                        "returned_count": result.returned_count})
                print(json.dumps({"capacity": capacity, "cpu_synthetic_timings": timings,
                    "capacity_embeddings": "deterministic 384-dimensional fixtures",
                    "timing_embeddings": "real accepted CPU model", "private_inputs": 0}))
        assert not attempts
    print(json.dumps({"python_network_provider_attempts": len(attempts)}))


if __name__ == "__main__":
    main()
