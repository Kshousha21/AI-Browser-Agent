"""Explicit optional runtime installation in a separate ignored virtual environment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
PINS = {"cuda": ("onnxruntime-gpu", "1.29.0"), "directml": ("onnxruntime-directml", "1.24.4")}


def requirement_lines(device: str) -> list[str]:
    name, version = PINS[device]
    lines = (ROOT / "backend/requirements.txt").read_text().splitlines()
    return [line for line in lines if not line.strip().startswith("onnxruntime==")] + [name + "==" + version]


def setup(device: str) -> dict[str, object]:
    if device not in PINS or device == "directml" and sys.platform != "win32":
        raise ValueError("unsupported_accelerator_setup")
    # Optional setup never uninstalls/replaces the base project's CPU runtime.
    parent = (ROOT / ".local/accelerators").resolve()
    target = (parent / device).resolve()
    if not parent.is_relative_to(ROOT / ".local") or target.parent != parent or target.is_symlink():
        raise ValueError("invalid_accelerator_directory")
    python = target / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not target.exists():
        venv.create(target, with_pip=True)
    if not python.is_file():
        raise ValueError("invalid_accelerator_environment")
    probe = "import importlib.metadata as m,json,sys; print(json.dumps({'prefix':sys.prefix,'variants':[d.metadata['Name'].lower().replace('_','-') for d in m.distributions() if d.metadata['Name'].lower().replace('_','-') in ('onnxruntime','onnxruntime-gpu','onnxruntime-directml')]}))"
    result = subprocess.run([str(python), "-B", "-c", probe], check=True, capture_output=True, text=True)
    state = json.loads(result.stdout)
    if Path(state["prefix"]).resolve() != target:
        raise ValueError("invalid_accelerator_environment")
    variants = state["variants"]
    if len(variants) > 1 or variants and variants != [PINS[device][0]]:
        raise ValueError("conflicting_onnxruntime_environment")
    requirements = target / "accelerator-requirements.txt"
    requirements.write_text("\n".join(requirement_lines(device)) + "\n")
    subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    return {"status": "optional_runtime_installed", "device": device, "runtime": PINS[device][0],
            "version": PINS[device][1], "inference_verified": False, "model_prepared": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=list(PINS))
    try:
        print(json.dumps(setup(parser.parse_args().device)))
    except (ValueError, OSError, subprocess.SubprocessError):
        print(json.dumps({"error": "accelerator_setup_failed"}), file=sys.stderr)
        raise SystemExit(2)
