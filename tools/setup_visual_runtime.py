"""Explicit package-only setup for a local-disk Nomic worker; never downloads models."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import venv
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from local_storage import data_root, private_path, require_local

def setup(device):
    target = require_local(private_path(Path.home() / ".venvs" / "ai-browser-visual"))
    if target.is_symlink(): raise ValueError("invalid runtime directory")
    python = target / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file(): venv.EnvBuilder(with_pip=True).create(target)
    probe = subprocess.run([str(python), "-B", "-c", "import sys;print(sys.prefix)"], capture_output=True, text=True, check=True)
    if Path(probe.stdout.strip()).resolve() != target.resolve(): raise ValueError("invalid runtime prefix")
    command = [str(python), "-m", "pip", "install", "--no-compile", "--disable-pip-version-check", "-r", str(ROOT / "backend/requirements.txt")]
    if device == "cuda":
        command += ["-r", str(ROOT / "backend/requirements-visual-cuda.txt")]
    subprocess.run(command, check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    return {"status":"runtime_installed", "device":device, "python_executable":str(python), "model_downloaded":False}

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    try: print(json.dumps(setup(parser.parse_args().device)))
    except Exception:
        print(json.dumps({"error":"visual_runtime_setup_failed"}));raise SystemExit(2)
