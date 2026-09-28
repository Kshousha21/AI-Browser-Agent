"""Read-only provider discovery; never creates an inference session or downloads."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from local_retrieval.execution import capabilities, runtime_distribution
from local_retrieval.errors import RetrievalError

if __name__ == "__main__":
    try:
        name, version = runtime_distribution()
        print(json.dumps({**capabilities(), "runtime": name, "version": version}))
    except RetrievalError as error:
        print(json.dumps({"status": "UNAVAILABLE", "error": error.code}))
        raise SystemExit(2)
