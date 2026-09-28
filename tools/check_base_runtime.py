"""Prevent normal CPU setup from overlapping incompatible ONNX Runtime packages."""
from importlib.metadata import distributions


def safe_base(names: list[str]) -> bool:
    variants = [name.lower().replace("_", "-") for name in names
                if name.lower().replace("_", "-") in {"onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"}]
    return not variants or variants == ["onnxruntime"]


if __name__ == "__main__":
    if not safe_base([item.metadata["Name"] for item in distributions()]):
        print("CPU setup refused: conflicting accelerator runtime. Use a separate optional accelerator environment.")
        raise SystemExit(2)
    print("CPU runtime package preflight passed.")
