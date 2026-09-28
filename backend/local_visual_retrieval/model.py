"""Local cache detection and an operation-scoped, cancellable runtime process.

No hub download APIs. A child process owns Torch/CUDA and exits before generation.
This also permits an already installed ColPali environment outside the backend venv.
"""
import asyncio
import base64
from contextlib import asynccontextmanager
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from local_retrieval.models import fingerprint
from .schemas import VisualError, VisualIdentity, VisualSettings

@lru_cache(maxsize=256)
def _digest(path, size, mtime):
    h = sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def local_snapshot(model, revision="main", directory=None):
    if directory:
        root = Path(directory).expanduser()
    else:
        from huggingface_hub.constants import HF_HUB_CACHE
        cache = Path(HF_HUB_CACHE) / ("models--" + model.replace("/", "--"))
        ref = cache / "refs" / revision
        revision = ref.read_text().strip() if ref.is_file() else revision
        root = cache / "snapshots" / revision
    if not root.is_dir():
        raise VisualError("visual_embedding_model_unavailable")
    return root.resolve()

def discover(settings):
    try:
        adapter = local_snapshot(settings.model, settings.revision, settings.model_directory)
        config = json.loads((adapter / "adapter_config.json").read_text())
        base_name = config["base_model_name_or_path"]
        if base_name != "Qwen/Qwen2.5-VL-3B-Instruct":
            raise ValueError()
        base = local_snapshot(base_name, config.get("revision") or "main", settings.base_model_directory)
        base_config = json.loads((base / "config.json").read_text())
        dimension = base_config.get("text_config", base_config).get("hidden_size")
        required = [(adapter, "adapter_model.safetensors"), (adapter, "tokenizer.json"),
                    (adapter, "preprocessor_config.json"), (adapter, "adapter_config.json"), (base, "config.json")]
        index = base / "model.safetensors.index.json"
        weights = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
        if any(Path(name).name != name for name in weights):
            raise ValueError()
        required += [(base, name) for name in weights]
        required += [(adapter, p.name) for p in adapter.iterdir() if p.suffix in {".json", ".txt", ".yml"} if (adapter, p.name) not in required]
        hashes = []
        for root, name in required:
            p = root / name
            stat = p.stat()
            if stat.st_size < 1: raise ValueError()
            hashes.append(("adapter/" if root == adapter else "base/") + name + ":" + _digest(str(p), stat.st_size, stat.st_mtime_ns))
        return adapter, base, dimension, fingerprint(hashes)
    except VisualError:
        raise
    except (OSError, ValueError, KeyError, TypeError):
        raise VisualError("visual_embedding_model_unavailable") from None

class Runtime:
    def __init__(self, process, timeout):
        self.process, self.timeout = process, timeout
    async def call(self, request):
        try:
            self.process.stdin.write(json.dumps(request).encode() + b"\n")
            await self.process.stdin.drain()
            line = await asyncio.wait_for(self.process.stdout.readline(), self.timeout)
            result = json.loads(line)
            if "error" in result: raise VisualError(result["error"])
            return result
        except VisualError:
            raise
        except (OSError, ValueError, TimeoutError, ConnectionError):
            raise VisualError("visual_embedding_failed") from None
    async def image(self, png):
        return (await self.call({"op": "image", "png": base64.b64encode(png).decode()}))["vector"]
    async def query(self, query):
        return (await self.call({"op": "query", "query": query}))["vector"]

class NomicEmbedder:
    def __init__(self, settings=None, renderer_identity="pdfium-default-v1"):
        self.settings = settings or VisualSettings()
        self.renderer_identity = renderer_identity
        self.measurements = {}
    @asynccontextmanager
    async def worker(self):
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                   HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false")
        process = None
        try:
            process = await asyncio.create_subprocess_exec(self.settings.python_executable or sys.executable,
                "-B", str(Path(__file__).with_name("runtime.py")), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, env=env,
                limit=256 * 1024, **({"creationflags": 0x08000000} if sys.platform == "win32" else {}))
            yield Runtime(process, self.settings.timeout_seconds)
        except (OSError, NotImplementedError):
            raise VisualError("visual_embedding_model_unavailable") from None
        finally:
            if process is not None:
                if process.returncode is None:
                    try: process.terminate()
                    except ProcessLookupError: pass
                await process.wait()
    async def describe(self):
        adapter, base, dimension, artifact = await asyncio.to_thread(discover, self.settings)
        async with self.worker() as worker:
            versions = await worker.call({"op": "describe"})
        identity = VisualIdentity(model=self.settings.model, revision=adapter.name + ":" + base.name,
            artifact_fingerprint=artifact, processor="BiQwen2_5_Processor:" + versions["runtime"],
            runtime=versions["runtime"], dimension=dimension, renderer=self.renderer_identity, device_policy=self.settings.device)
        return identity, adapter, base
    @asynccontextmanager
    async def session(self, description):
        identity, adapter, base = description
        async with self.worker() as worker:
            self.measurements = await worker.call({"op": "load", "adapter": str(adapter), "base": str(base),
                                                   "device": self.settings.device})
            yield worker
            self.measurements.update(await worker.call({"op": "metrics"}))
