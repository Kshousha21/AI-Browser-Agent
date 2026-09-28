"""Read-only dependency/FTS5/model-artifact check; no inference or download."""
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def check():
    from local_storage import readiness
    from local_retrieval import RetrievalError
    from local_retrieval.local_model import DEFAULT_MODEL_DIRECTORY, model_identity, verified_files
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
        fts5 = True
    except sqlite3.Error:
        fts5 = False
    finally:
        connection.close()
    try:
        verified_files(DEFAULT_MODEL_DIRECTORY)
        model = "verified"
    except RetrievalError as error:
        model = error.code
    from local_retrieval.execution import capabilities
    return {"fts5": fts5, "model_artifacts": model, "dimension": model_identity().dimension,
            "acceleration": capabilities(), "runtime_storage": readiness()}


if __name__ == "__main__":
    print(json.dumps(check()))
