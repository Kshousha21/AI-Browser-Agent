"""Config loading, layering and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from browser.config import ConfigError, Settings, load_settings

GOOD_YAML = """
server:
  host: 127.0.0.1
  port: 9001
auth:
  token: yaml-token-0123456789abcdef
provider:
  name: openai
  model: gpt-4o-mini
  api_key: yaml-key
page:
  max_visible_text_chars: 1234
"""


def write(tmp_path: Path, text: str, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_values_from_yaml(tmp_path: Path) -> None:
    settings = load_settings(write(tmp_path, GOOD_YAML))

    assert settings.server.port == 9001
    assert settings.auth.token == "yaml-token-0123456789abcdef"
    assert settings.provider.model == "gpt-4o-mini"
    assert settings.page.max_visible_text_chars == 1234


def test_defaults_apply_when_yaml_omits_a_section(tmp_path: Path) -> None:
    yaml = "auth:\n  token: only-the-token-0123456789\nprovider:\n  name: mock\n"
    settings = load_settings(write(tmp_path, yaml))

    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 8765
    assert settings.logging.format == "json"


def test_environment_overrides_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BROWSER_SERVER__PORT", "7777")
    monkeypatch.setenv("BROWSER_PROVIDER__MODEL", "gpt-4o")

    settings = load_settings(write(tmp_path, GOOD_YAML))

    assert settings.server.port == 7777
    assert settings.provider.model == "gpt-4o"


def test_bare_openai_api_key_is_accepted_as_an_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml = "auth:\n  token: yaml-token-0123456789abcdef\nprovider:\n  name: openai\n"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-environment")

    settings = load_settings(write(tmp_path, yaml))

    assert settings.provider.api_key == "sk-from-environment"


def test_bare_anthropic_key_loads_from_sibling_dotenv(tmp_path: Path) -> None:
    config = write(
        tmp_path,
        "auth:\n  token: ''\nprovider:\n  name: anthropic\n  model: claude-opus-5\n",
    )
    (tmp_path / ".env").write_text(
        "BROWSER_AUTH__TOKEN=dotenv-token-0123456789abcdef\n"
        "ANTHROPIC_API_KEY=sk-ant-from-dotenv\n",
        encoding="utf-8",
    )

    settings = load_settings(config)

    assert settings.auth.token == "dotenv-token-0123456789abcdef"
    assert settings.provider.api_key == "sk-ant-from-dotenv"


def test_prefixed_key_beats_the_bare_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml = "auth:\n  token: yaml-token-0123456789abcdef\nprovider:\n  name: openai\n"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-bare")
    monkeypatch.setenv("BROWSER_PROVIDER__API_KEY", "sk-prefixed")

    settings = load_settings(write(tmp_path, yaml))

    assert settings.provider.api_key == "sk-prefixed"


def test_missing_token_is_reported(tmp_path: Path) -> None:
    yaml = "provider:\n  name: mock\n"

    with pytest.raises(ConfigError) as excinfo:
        load_settings(write(tmp_path, yaml))

    assert "auth.token" in str(excinfo.value)


def test_missing_api_key_for_openai_is_reported(tmp_path: Path) -> None:
    yaml = "auth:\n  token: yaml-token-0123456789abcdef\nprovider:\n  name: openai\n"

    with pytest.raises(ConfigError) as excinfo:
        load_settings(write(tmp_path, yaml))

    assert "provider.api_key" in str(excinfo.value)


def test_every_missing_field_is_listed_at_once(tmp_path: Path) -> None:
    """The error names all the problems, not just the first one found."""
    yaml = "provider:\n  name: openai\n"

    with pytest.raises(ConfigError) as excinfo:
        load_settings(write(tmp_path, yaml))

    message = str(excinfo.value)
    assert "auth.token" in message
    assert "provider.api_key" in message


def test_non_loopback_host_is_rejected(tmp_path: Path) -> None:
    yaml = (
        "server:\n  host: 0.0.0.0\n"
        "auth:\n  token: yaml-token-0123456789abcdef\n"
        "provider:\n  name: mock\n"
    )

    with pytest.raises(ConfigError) as excinfo:
        load_settings(write(tmp_path, yaml))

    assert "loopback" in str(excinfo.value)


def test_short_token_is_rejected(tmp_path: Path) -> None:
    yaml = "auth:\n  token: short\nprovider:\n  name: mock\n"

    with pytest.raises(ConfigError) as excinfo:
        load_settings(write(tmp_path, yaml))

    assert "16 characters" in str(excinfo.value)


def test_out_of_range_port_is_rejected(tmp_path: Path) -> None:
    yaml = (
        "server:\n  port: 99999\n"
        "auth:\n  token: yaml-token-0123456789abcdef\n"
        "provider:\n  name: mock\n"
    )

    with pytest.raises(ConfigError) as excinfo:
        load_settings(write(tmp_path, yaml))

    assert "port" in str(excinfo.value)


def test_malformed_yaml_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_settings(write(tmp_path, "auth:\n  token: [unclosed\n"))

    assert "not valid YAML" in str(excinfo.value)


def test_yaml_that_is_not_a_mapping_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_settings(write(tmp_path, "- a\n- b\n"))

    assert "mapping" in str(excinfo.value)


def test_missing_file_falls_back_to_defaults_and_still_validates(
    tmp_path: Path,
) -> None:
    """No config.yaml is not itself an error; the missing fields are."""
    with pytest.raises(ConfigError) as excinfo:
        load_settings(tmp_path / "does-not-exist.yaml")

    assert "auth.token" in str(excinfo.value)


def test_shipped_config_yaml_parses(tmp_path: Path) -> None:
    """The checked-in config.yaml is well-formed; it is only missing secrets."""
    shipped = Path(__file__).resolve().parent.parent / "config.yaml"
    isolated = write(tmp_path, shipped.read_text(encoding="utf-8"))

    with pytest.raises(ConfigError) as excinfo:
        load_settings(isolated)

    message = str(excinfo.value)
    assert "not valid YAML" not in message
    assert "auth.token" in message


def test_settings_can_be_built_directly_for_tests() -> None:
    settings = Settings(auth={"token": "x" * 20}, provider={"name": "mock"})

    assert settings.provider.name == "mock"
    assert settings.server.host == "127.0.0.1"


def test_site_adapter_config_rejects_obsolete_host_configuration() -> None:
    with pytest.raises(ValueError):
        Settings(
            auth={"token": "x" * 20},
            provider={"name": "mock"},
            site_adapters={
                "obsolete_adapter": {"enabled": True, "domains": ["*.example.org"]}
            },
        )


# --------------------------------------------------------------------------
# Provider selection
# --------------------------------------------------------------------------


def test_anthropic_is_the_default_provider(tmp_path: Path) -> None:
    yaml = "auth:\n  token: yaml-token-0123456789abcdef\nprovider:\n  api_key: k\n"

    settings = load_settings(write(tmp_path, yaml))

    assert settings.provider.name == "anthropic"
    assert settings.provider.model == "claude-opus-5"


def test_model_defaults_per_provider_when_left_blank(tmp_path: Path) -> None:
    yaml = (
        "auth:\n  token: yaml-token-0123456789abcdef\n"
        "provider:\n  name: openai\n  api_key: k\n"
    )

    assert load_settings(write(tmp_path, yaml)).provider.model == "gpt-4o-mini"


def test_an_explicit_model_is_not_overridden(tmp_path: Path) -> None:
    yaml = (
        "auth:\n  token: yaml-token-0123456789abcdef\n"
        "provider:\n  name: anthropic\n  model: claude-sonnet-5\n  api_key: k\n"
    )

    assert load_settings(write(tmp_path, yaml)).provider.model == "claude-sonnet-5"


def test_bare_anthropic_api_key_is_accepted_as_an_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-environment")
    yaml = "auth:\n  token: yaml-token-0123456789abcdef\nprovider:\n  name: anthropic\n"

    assert load_settings(write(tmp_path, yaml)).provider.api_key == "sk-ant-from-environment"


def test_the_wrong_providers_key_is_not_picked_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exported OPENAI_API_KEY must not be handed to Anthropic."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    yaml = "auth:\n  token: yaml-token-0123456789abcdef\nprovider:\n  name: anthropic\n"

    with pytest.raises(ConfigError) as excinfo:
        load_settings(write(tmp_path, yaml))

    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


def test_bare_process_env_api_key_beats_config_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-environment")
    yaml = (
        "auth:\n  token: yaml-token-0123456789abcdef\n"
        "provider:\n  name: anthropic\n  api_key: sk-ant-from-yaml\n"
    )

    assert load_settings(write(tmp_path, yaml)).provider.api_key == "sk-ant-from-environment"


def test_an_invalid_effort_level_is_rejected(tmp_path: Path) -> None:
    yaml = (
        "auth:\n  token: yaml-token-0123456789abcdef\n"
        "provider:\n  name: anthropic\n  api_key: k\n  effort: turbo\n"
    )

    with pytest.raises(ConfigError) as excinfo:
        load_settings(write(tmp_path, yaml))

    assert "effort" in str(excinfo.value)


def test_max_output_tokens_leaves_room_for_thinking(tmp_path: Path) -> None:
    """Thinking tokens count against max_tokens, so the default is generous."""
    yaml = "auth:\n  token: yaml-token-0123456789abcdef\nprovider:\n  api_key: k\n"

    assert load_settings(write(tmp_path, yaml)).provider.max_output_tokens >= 8192
