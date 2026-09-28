"""Root identity must survive Windows aliases without redirecting execution."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import operational

OVERRIDES = ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR")
ENGINES = ("python", "powershell")


@pytest.fixture(autouse=True)
def clean_git_environment(monkeypatch):
    for name in OVERRIDES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "Project With Spaces"
    for name in ("backend", "extension", "shared", "scripts", "tools"):
        (root / name).mkdir(parents=True)
    for name in ("common.ps1", "build.ps1"):
        shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
    for name in ("extension/src", "extension/public", "extension/dist/content"):
        (root / name).mkdir(parents=True)
    for name in ("package.json", "package-lock.json", "tsconfig.json",
                 "vite.config.ts", "vite.content.config.ts"):
        (root / "extension" / name).write_text("{}", encoding="utf-8")
    (root / "extension/dist/manifest.json").write_text("{}", encoding="utf-8")
    (root / "extension/dist/content/extract.js").touch()
    shutil.copy2(ROOT / "tools/build-metadata.mjs", root / "tools/build-metadata.mjs")
    shutil.copy2(ROOT / "shared/build-contract.json", root / "shared/build-contract.json")
    return root


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True, timeout=15)


def ps_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def powershell(command, *, env=None, cwd=None):
    executable = shutil.which("powershell.exe")
    assert executable, "Windows PowerShell is required for operational acceptance"
    return subprocess.run([executable, "-NoProfile", "-ExecutionPolicy", "RemoteSigned", "-Command", command],
                          env=env, cwd=cwd, capture_output=True, text=True, timeout=30)


def check_root(engine, root, monkeypatch, *, identity=None, reported=None,
               returncode=0, override=None):
    """Exercise production guards, mocking only the external Git boundary."""
    env = dict(os.environ)
    if override:
        env[override] = str(root.parent / "unrelated")
    if engine == "python":
        if override:
            monkeypatch.setenv(override, env[override])
        if identity is not None:
            def command(args, **kwargs):
                assert args[1:3] == ["-C", str(root.absolute())]
                output = reported if args[-1] == "--show-toplevel" else identity
                return SimpleNamespace(returncode=returncode, stdout=output)
            monkeypatch.setattr(operational, "command", command)
        try:
            return operational.repository_root(root) == root.absolute()
        except ValueError:
            return False
    if engine == "powershell":
        command = ""
        if identity is not None:
            command += (
                "function git { $global:LASTEXITCODE=" + str(returncode) + "; "
                "if ($args -contains '--show-toplevel') { " + ps_literal(reported) + " } "
                "else { " + ps_literal(identity.rstrip("\n")) + ".Split([char]10); "
                # Emit the final empty prefix as a separate PowerShell line.
                + ("''; " if identity.endswith("\n\n") else "") + "} }; "
            )
        command += (
            ". " + ps_literal(root / "scripts/common.ps1") + "; "
            "try { $before=(Get-Location).Path; $value=Get-ProjectRoot; "
            "if ($value -ne " + ps_literal(str(root.absolute())) + " -or "
            "$before -cne (Get-Location).Path) { exit 2 }; 'ACCEPTED' } catch { exit 3 }"
        )
        result = powershell(command, env=env)
        return result.returncode == 0 and "ACCEPTED" in result.stdout
    args = [shutil.which("node")]
    if identity is not None:
        # Native module synchronization makes the mock visible to the unchanged
        # build-metadata module's named execFileSync import.
        preload = root / "tools/git-fixture.mjs"
        preload.write_text(
            'import cp from "node:child_process";\n'
            'import {syncBuiltinESMExports} from "node:module";\n'
            'cp.execFileSync=(_exe,args)=>{\n'
            + ("throw Error('Git failed');\n" if returncode else "")
            + 'if(args.includes("--show-toplevel")) return ' + json.dumps(reported) + ';\n'
            + 'if(args.includes("--short=12")) return "0123456789ab\\n";\n'
            + 'return ' + json.dumps(identity) + '; };\nsyncBuiltinESMExports();\n',
            encoding="utf-8",
        )
        args += ["--import", preload.as_uri()]
    result = subprocess.run([*args, str(root / "tools/build-metadata.mjs"), "--write"],
                            env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    saved = json.loads((root / "extension/dist/build-info.json").read_text())
    return saved["git_commit"] is not None


@pytest.mark.parametrize("engine", ENGINES)
def test_source_snapshot_without_git(project, monkeypatch, engine):
    assert not (project / ".git").exists()
    assert check_root(engine, project, monkeypatch)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("override", OVERRIDES)
def test_git_environment_cannot_redirect_execution(project, monkeypatch, engine, override):
    assert check_root(engine, project, monkeypatch, override=override)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("marker", ("backend", "extension", "shared/build-contract.json"))
@pytest.mark.parametrize("problem", ("missing", "wrong_type"))
def test_structural_markers_are_required(tmp_path, monkeypatch, engine, marker, problem):
    root = tmp_path / "Incomplete"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/common.ps1", root / "scripts/common.ps1")
    for name in ("backend", "extension", "shared/build-contract.json"):
        target = root / name
        if name == marker and problem == "missing": continue
        target.parent.mkdir(parents=True, exist_ok=True)
        directory = name in {"backend", "extension"}
        if name == marker: directory = not directory
        if directory: target.mkdir()
        else: target.write_text('{}')
    assert not check_root(engine, root, monkeypatch)


@pytest.mark.parametrize("engine", ENGINES)
def test_parent_git_does_not_change_snapshot_root(project, monkeypatch, engine):
    git(project.parent, "init", "-q")
    assert check_root(engine, project, monkeypatch)
    assert not (project / ".git").exists()


@pytest.mark.parametrize("engine", ENGINES)
def test_real_windows_case_and_separator_alias(project, monkeypatch, engine):
    alias = Path(str(project).swapcase().replace("\\", "/"))
    assert check_root(engine, alias, monkeypatch)


@pytest.mark.parametrize("identity,code", (("true\nnested/\n", 0), ("false\n\n", 0), ("", 1)))
def test_optional_metadata_ignores_invalid_git(project, monkeypatch, identity, code):
    git(project, "init", "-q")
    assert not check_root("metadata", project, monkeypatch, identity=identity, reported=str(project), returncode=code)


@contextmanager
def drive_alias(target):
    """Create only an unused local SUBST alias, never a workstation share mapping."""
    drive = next(letter + ":" for letter in reversed("DEFGHIJKLMNOPQRSTUVWXYZ")
                 if not os.path.exists(letter + ":\\"))
    result = subprocess.run(["subst", drive, str(target)], capture_output=True, text=True)
    assert result.returncode == 0, "Cannot create isolated local drive fixture"
    try:
        yield Path(drive + "\\")
    finally:
        # Verify this is still our owned fixture before removing its drive alias.
        assert os.path.samefile(drive + "\\", target)
        subprocess.run(["subst", drive, "/D"], check=True, capture_output=True)


def test_build_npm_and_node_keep_original_drive(project):
    (project / "extension/package.json").write_text(json.dumps({
        "name": "root-fixture", "private": True, "scripts": {"build": "node probe.cjs"},
    }), encoding="utf-8")
    (project / "extension/probe.cjs").write_text(
        'require("node:fs").writeFileSync("dist/build-info.json", JSON.stringify({'
        'cwd:process.cwd(), initial:process.env.INIT_CWD}));\n', encoding="utf-8",
    )
    with drive_alias(project.parent) as mapped:
        candidate = mapped / project.name
        assert operational.repository_root(candidate) == candidate
        command = (
            # Execute unmodified production scripts and real npm.cmd; the
            # fixture Node script records both npm's initial cwd and its own.
            "& " + ps_literal(candidate / "scripts/build.ps1") + "; "
            "if ((Get-Location).Path -cne " + ps_literal(candidate) + ") { exit 2 }"
        )
        result = powershell(command, cwd=candidate)
        assert result.returncode == 0, result.stdout + result.stderr
        probe = json.loads((project / "extension/dist/build-info.json").read_text())
        assert probe["cwd"].casefold() == str(candidate / "extension").casefold()
        assert probe["initial"].casefold() == str(candidate / "extension").casefold()
        assert not probe["cwd"].startswith(("\\\\", "//"))
        assert "UNC paths are not supported" not in result.stdout + result.stderr


def test_python_preserves_alias_without_filesystem_canonicalization(project, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Repository execution root must not use resolve()")
    monkeypatch.setattr(Path, "resolve", forbidden)
    assert operational.repository_root(project) == project
