"""Explicit synthetic-only calibration for auto; accepts no documents or queries."""
import argparse
import gc
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from retrieval_offline_guard import offline_guard
from local_retrieval import RetrievalError
from local_retrieval.execution import available_providers, PROVIDERS
from local_retrieval.local_model import OnnxEmbedder
from local_retrieval.preference import Measurement, Preference, context_identity, save_preference, ORDER

# Fixed invented text only. No source database, PDF parser or query argument.
QUERY = "Synthetic thermal packaging and optical connections."
BATCH = [
    "Synthetic optical links connect components using photons.",
    "Synthetic cooling removes processor heat through a heat sink.",
    "Synthetische optische Verbindungen transportieren Informationen.",
    "Synthetic fields need irrigation for agricultural crops.",
    "Synthetische Kuehlung senkt die Temperatur von Prozessoren.",
] * 6


class SafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError("invalid_arguments")


def measure(device: str) -> Measurement:
    import numpy as np
    model = OnnxEmbedder(device=device)  # Explicit modes cannot consume preferences.
    reference = np.asarray(model.encode([QUERY]), dtype=np.float32)
    batch = np.asarray(model.encode(BATCH), dtype=np.float32)
    for value, count in ((reference, 1), (batch, 30)):
        if value.shape != (count, 384) or not np.isfinite(value).all() or not np.allclose(np.linalg.norm(value, axis=1), 1, atol=1e-5):
            raise ValueError("invalid_synthetic_vectors")
    if not np.allclose(reference, model.encode([QUERY]), rtol=1e-5, atol=1e-6):
        raise ValueError("unstable_synthetic_vectors")
    timings = []
    for texts, repeats in (([QUERY], 7), (BATCH, 5)):
        elapsed = []
        for _ in range(repeats):
            start = perf_counter()
            model.encode(texts)
            elapsed.append((perf_counter() - start) * 1000)
        timings.append(median(elapsed))
    return Measurement(device=device, identity=model.identity.identity, verified=True,
                       query_ms=timings[0], batch_ms=timings[1])


def calibrate() -> dict[str, object]:
    with offline_guard() as attempts:
        providers = available_providers()
        context = context_identity(providers)
        measurements, failures = [], []
        for device in ORDER:
            if PROVIDERS[device] not in providers:
                continue
            try:
                measurements.append(measure(device))
            except (RetrievalError, ValueError):
                failures.append({"device": device, "reason": "verification_failed"})
            finally:
                gc.collect()
        if attempts or not measurements:
            raise ValueError("calibration_not_verified")
        profile = Preference(context=context, measurements=measurements)
        # Re-check identities/providers before publishing, never retain stale results.
        save_preference(profile, available_providers())
        ranked = sorted(measurements, key=lambda item: (item.score_ms, ORDER.index(item.device)))
        return {"status": "PERFORMANCE PREFERENCE VERIFIED", "policy": profile.policy,
                "preferred_device": ranked[0].device,
                "measurements": [{"device": item.device, "query_ms": round(item.query_ms, 3),
                    "batch_ms": round(item.batch_ms, 3), "score_ms": round(item.score_ms, 3)} for item in ranked],
                "failures": failures, "python_network_or_provider_attempts": len(attempts)}


if __name__ == "__main__":
    try:
        SafeParser(description=__doc__).parse_args()
        print(json.dumps(calibrate()))
    except Exception:
        print(json.dumps({"status": "PERFORMANCE PREFERENCE NOT VERIFIED"}))
        raise SystemExit(2)
