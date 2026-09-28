"""Slow local inference can use longer, finite timeout budgets."""

import pytest
import yaml
from pydantic import ValidationError

from browser.config import AgentSettings, load_provider_settings, load_settings
from browser.provider_settings import ProviderProfile, ProviderSettings
from browser.routing_settings import RoutingSettings


def test_long_local_timeouts_load_and_environment_overrides(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "auth": {"token": "synthetic-timeout-test-token"},
        "provider": {"active": "local_agent"},
        "providers": {
            name: {"type": "ollama", "model": "fixture-model",
                   "base_url": "http://127.0.0.1:11434",
                   "auth": {"mode": "none"}, "local_only": True,
                   "timeout_seconds": timeout}
            for name, timeout in (("local_agent", 900), ("local_router", 300))
        },
        "routing": {"enabled": True, "fallback_timeout_seconds": 300},
        "agent": {"step_timeout_seconds": 1200, "run_timeout_seconds": 3600},
    }), encoding="utf-8")
    settings = load_settings(path)
    assert settings.provider.timeout_seconds == 900
    assert settings.routing.fallback_timeout_seconds == 300
    assert settings.agent.step_timeout_seconds == 1200
    assert settings.agent.run_timeout_seconds == 3600
    assert load_provider_settings(path, profile="local_router").timeout_seconds == 300
    monkeypatch.setenv("BROWSER_PROVIDERS__LOCAL_AGENT__TIMEOUT_SECONDS", "1800")
    monkeypatch.setenv("BROWSER_AGENT__STEP_TIMEOUT_SECONDS", "2400")
    settings = load_settings(path)
    assert settings.provider.timeout_seconds == 1800
    assert settings.agent.step_timeout_seconds == 2400


@pytest.mark.parametrize("model,field,extra,maximum", [
    (AgentSettings, "step_timeout_seconds", {}, 3600),
    (AgentSettings, "run_timeout_seconds", {}, 14400),
    (ProviderProfile, "timeout_seconds", {"type": "mock"}, 3600),
    (ProviderSettings, "timeout_seconds", {"name": "mock"}, 3600),
    (RoutingSettings, "fallback_timeout_seconds", {}, 3600),
])
def test_timeout_limits_remain_positive_and_finite(model, field, extra, maximum):
    assert getattr(model(**extra, **{field: maximum}), field) == maximum
    for invalid in (0, -1, maximum + 1, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            model(**extra, **{field: invalid})
