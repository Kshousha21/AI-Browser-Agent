"""Strict DirectML acceptance of GPU tensor compute plus tiny CPU shape bookkeeping."""
import json
import math
from pathlib import Path

from .errors import RetrievalError

CPU_SHAPE_OPS = {"Gather", "Unsqueeze", "Concat", "Slice", "Cast", "Sqrt", "Div", "Reshape", "Equal", "Where"}


def validate_profile(path: Path, directory: Path) -> dict[str, object]:
    """Only fixed-model synthetic probes are profiled; no user input is recorded."""
    try:
        if path.resolve().parent != directory.resolve() or path.is_symlink() or path.stat().st_size > 4_000_000:
            raise ValueError("invalid_profile")
        events = json.loads(path.read_text(encoding="utf-8"))
        cpu_count = gpu_count = gpu_matmul = cpu_max = 0
        if not isinstance(events, list) or len(events) > 20000:
            raise ValueError("invalid_profile")
        for event in events:
            args = event.get("args", {})
            provider = args.get("provider")
            if event.get("cat") != "Node" or not provider:
                continue
            if provider == "DmlExecutionProvider":
                gpu_count += 1
                gpu_matmul += args.get("op_name") == "MatMul"
            elif provider == "CPUExecutionProvider":
                if args.get("op_name") not in CPU_SHAPE_OPS:
                    raise ValueError("unexpected_cpu_compute")
                tensors = args.get("input_type_shape", []) + args.get("output_type_shape", [])
                if not tensors:
                    raise ValueError("missing_shape_evidence")
                sizes = []
                for tensor in tensors:
                    for shape in tensor.values():
                        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
                            raise ValueError("invalid_tensor_shape")
                        sizes.append(math.prod(shape))
                largest = max(sizes, default=0)
                if largest > 4:
                    raise ValueError("unexpected_cpu_tensor")
                cpu_max = max(cpu_max, largest)
                cpu_count += 1
            else:
                raise ValueError("unexpected_provider")
        # Pinned 12-layer graph has 96 GPU MatMuls per forward, probed at batches 1 and 8.
        if gpu_matmul != 192 or gpu_count < gpu_matmul or cpu_count > 400:
            raise ValueError("insufficient_gpu_evidence")
        return {"policy": "directml-gpu-compute-cpu-shapes4-v1", "gpu_node_events": gpu_count,
                "gpu_matmul_events": gpu_matmul, "cpu_shape_events": cpu_count,
                "cpu_max_tensor_elements": cpu_max, "profiling_stopped": True,
                "fully_gpu_graph": False}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        raise RetrievalError("embedding_accelerator_unavailable") from None
