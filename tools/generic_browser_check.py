"""Run the neutral real-Chrome transition regression from any working directory."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    backend = Path(__file__).resolve().parents[1] / "backend"
    raise SystemExit(subprocess.call(
        [sys.executable, "-B", "-m", "pytest", "-q", "-ra", "tests/test_generic_browser_transitions.py"],
        cwd=backend,
    ))
