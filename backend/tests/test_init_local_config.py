"""First-run configuration must not disclose secrets or overwrite user settings."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "init_local_config", Path(__file__).parents[2] / "tools/init_local_config.py"
)
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


def test_creates_unique_tokens_and_preserves_other_settings(tmp_path):
    template = tmp_path / "example"
    template.write_text("# comment\nBROWSER_AUTH__TOKEN=\nBROWSER_PROVIDER__ACTIVE=\n")
    tokens = []
    for name in ("first", "second"):
        target = tmp_path / name
        assert module.initialize(template, target)
        lines = target.read_text().splitlines()
        tokens.append(lines[1].split("=", 1)[1])
        assert lines[0] == "# comment"
        assert lines[2] == "BROWSER_PROVIDER__ACTIVE=local_agent"
    assert all(len(token) >= 32 for token in tokens)
    assert tokens[0] != tokens[1]


def test_preserves_existing_configuration_without_reading_template(tmp_path):
    target = tmp_path / ".env"
    original = b"private existing configuration\r\n"
    target.write_bytes(original)
    assert not module.initialize(tmp_path / "missing-template", target)
    assert target.read_bytes() == original


@pytest.mark.parametrize("content", ["no token", "BROWSER_AUTH__TOKEN=\nBROWSER_AUTH__TOKEN=\n"])
def test_rejects_invalid_template_without_creating_config(tmp_path, content):
    template = tmp_path / "example"
    template.write_text(content)
    target = tmp_path / ".env"
    with pytest.raises(ValueError, match="exactly one"):
        module.initialize(template, target)
    assert not target.exists()


def test_cli_never_displays_generated_token(tmp_path, monkeypatch, capsys):
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / ".env.example").write_text("BROWSER_AUTH__TOKEN=\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["init_local_config.py"])
    module.main()
    token = (backend / ".env").read_text().strip().split("=", 1)[1]
    assert token not in capsys.readouterr().out


def test_exclusive_create_preserves_concurrent_writer(tmp_path, monkeypatch):
    template = tmp_path / "example"
    template.write_text("BROWSER_AUTH__TOKEN=\n")
    target = tmp_path / ".env"
    original_open = Path.open

    def racing_open(path, mode="r", *args, **kwargs):
        if path == target and mode == "x":
            with original_open(target, "w") as stream:
                stream.write("other writer")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    assert not module.initialize(template, target)
    assert target.read_text() == "other writer"
