"""Private stdio worker. No network, document logging, or rendered-file persistence."""
import base64
import contextlib
import importlib.metadata
import io
import json
import socket
import sys
import time

# Enforced in this dedicated process, including third-party model/runtime code.
def deny_network(*args, **kwargs):
    raise OSError("offline visual runtime")
socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
socket.create_connection = deny_network

model = processor = torch = None
metrics = {}

def dispatch(request):
    global model, processor, torch, metrics
    op = request["op"]
    if op == "describe":
        packages = ("colpali-engine", "transformers", "peft", "torch", "torchvision", "tokenizers", "Pillow")
        return {"runtime": ";".join(p + "=" + importlib.metadata.version(p) for p in packages)}
    if op == "load":
        from pathlib import Path
        if not (Path(request["base"]) / "config.json").is_file() or not (Path(request["adapter"]) / "adapter_config.json").is_file():
            raise FileNotFoundError()
        started = time.perf_counter()
        import torch as torch_module
        from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
        from peft import PeftModel
        torch = torch_module
        device = "cuda" if request["device"] == "auto" and torch.cuda.is_available() else request["device"]
        if device == "auto": device = "cpu"
        if device == "cuda" and not torch.cuda.is_available(): raise RuntimeError()
        metrics = {"device": device, "gpu_before_bytes": torch.cuda.memory_allocated() if device == "cuda" else 0}
        if device == "cuda": torch.cuda.reset_peak_memory_stats()
        # Resolve the base explicitly: the adapter's remote base name is never passed to a loader.
        model, loading = BiQwen2_5.from_pretrained(request["base"], local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa", output_loading_info=True, key_mapping={r"^model\.": "language_model."})
        if loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise ValueError("incomplete local base weights")
        model = model.to(device)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Found missing adapter keys.*")
            model = PeftModel.from_pretrained(model, request["adapter"], local_files_only=True,
                                              key_mapping={r"^model\.": "language_model."}).eval()
        image_config = json.loads((Path(request["adapter"]) / "preprocessor_config.json").read_text())
        # New Transformers releases use size rather than the older max_pixels key.
        # Preserve the adapter's trained page resolution and bound visual activations.
        bounds = {"shortest_edge": image_config["min_pixels"], "longest_edge": image_config["max_pixels"]}
        processor = BiQwen2_5_Processor.from_pretrained(request["adapter"], local_files_only=True,
                                                     trust_remote_code=False, size=bounds)
        metrics.update(load_ms=round((time.perf_counter() - started) * 1000, 3),
                       gpu_after_load_bytes=torch.cuda.memory_allocated() if device == "cuda" else 0)
        return dict(metrics)
    if op == "metrics":
        return {**metrics, "gpu_peak_bytes": torch.cuda.max_memory_allocated() if metrics["device"] == "cuda" else 0}
    if model is None: raise RuntimeError()
    if op == "image":
        from PIL import Image
        with Image.open(io.BytesIO(base64.b64decode(request["png"], validate=True))) as image:
            rgb = image.convert("RGB")
            try: batch = processor.process_images([rgb]).to(model.device)
            finally: rgb.close()
    elif op == "query":
        batch = processor.process_queries([request["query"]]).to(model.device)
    else: raise ValueError()
    with torch.inference_mode():
        vector = model(**batch).float().cpu()
    if vector.ndim != 2 or vector.shape[0] != 1: raise ValueError()
    return {"vector": vector[0].tolist()}

if __name__ == "__main__":
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            # Library informational messages cannot enter the private JSON protocol.
            with contextlib.redirect_stdout(sys.stderr):
                result = dispatch(request)
        except Exception:
            result = {"error": "visual_embedding_model_unavailable" if request.get("op") in {"describe", "load"} else "visual_embedding_failed"}
        print(json.dumps(result), flush=True)
