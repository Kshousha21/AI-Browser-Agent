"""The first-run smoke check must exercise the actual adapter-specific tools."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from browser.providers import FinalAnswer, MockProvider, ProviderResponse, ToolCall
class Router(MockProvider):
    def __init__(self, reply): super().__init__(scripted=[reply])

_spec = importlib.util.spec_from_file_location(
    "local_routing_check", Path(__file__).parents[2] / "tools/local_routing_check.py"
)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


@pytest.mark.parametrize("searches", [True, False])
async def test_smoke_uses_adapter_tools_and_requires_search(monkeypatch, searches):
    class Main(MockProvider):
        async def generate(self, messages, tools=None, **kwargs):
            names = {tool.name for tool in tools}
            assert {"click", "type_text", "press_key"} <= names
            decision = (ToolCall(tool="type_text", arguments={"element_id":"f0:e-001", "text": "AI"},
                                snapshot_id="synthetic-snapshot", reason="Synthetic search")
                        if searches else FinalAnswer(answer="I could search."))
            return ProviderResponse(decision=decision, raw_text="")

    providers = iter([Router('{"route":"SEARCH"}'), Main()])
    monkeypatch.setattr("browser.config.load_provider_settings", lambda *args:
                        SimpleNamespace(name="ollama", local_only=True))
    monkeypatch.setattr("browser.providers.build_provider", lambda settings: next(providers))
    args = SimpleNamespace(config=Path("unused"), router_profile="router", main_profile="main")
    assert await module.check(args) is searches
