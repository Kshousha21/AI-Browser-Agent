from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# The package lives at backend/browser; tests should run from anywhere.
BACKEND_ROOT = Path(__file__).absolute().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from browser.config import Settings  # noqa: E402

TEST_TOKEN = "test-token-0123456789abcdef"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the developer's real environment out of the suite.

    An exported OPENAI_API_KEY or BROWSER_* variable would otherwise leak into
    settings, and a real backend/.env would be picked up from the working
    directory, so both are removed and the cwd is moved somewhere empty.
    """
    for key in list(os.environ):
        if key.startswith("BROWSER_") or key in {
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        }:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    # Operator performance preferences must not affect deterministic test policy.
    monkeypatch.setattr("local_retrieval.preference.LOCAL_DIRECTORY", tmp_path / ".local")
    # Never consume a workstation's Nomic/base weights in deterministic tests.
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "empty-model-cache"))


@pytest.fixture
def settings() -> Settings:
    """Valid settings pointing at the mock provider, so no API key is needed."""
    return Settings(
        auth={"token": TEST_TOKEN},
        provider={"active": "mock"},
        providers={"mock": {"type": "mock", "model": "mock-1"}},
    )
