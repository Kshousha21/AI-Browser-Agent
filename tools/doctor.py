"""Read-only Windows/local developer diagnostics; no live service requests."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from operational import command, dangerous_tracked, executable_version, json_result, port_status, repository_root

ROOT = Path(__file__).absolute().parents[1]


def diagnose(root: Path = ROOT) -> dict:
    checks = []
    def add(name, ok, detail, *, warning=False):
        checks.append(dict(name=name, status="OK" if ok else "WARN" if warning else "FAIL", detail=detail))
    try:
        root = repository_root(root)
    except ValueError:
        add("Repository", False, "Missing project structure. Expected backend/, extension/ and shared/build-contract.json.")
        return dict(result="NOT READY", checks=checks)
    add("Repository", True, "Project structure found")
    parent_repo = any((parent / ".git").exists() for parent in root.parents)
    if parent_repo:
        add("Parent repository", False, "Nested/parent Git repository detected; it is not used or modified.", warning=True)
    system_python = shutil.which("python") or sys.executable
    ok, version = executable_version(system_python, (3, 11))
    add("Python", ok and version.startswith("3."), version + " (requires Python 3.11+)")
    python = root / "backend/.venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    add("Virtual environment", python.is_file(), "PRESENT" if python.is_file() else "MISSING; run scripts/setup.ps1")
    if python.is_file():
        ok, version = executable_version(str(python), (3, 11))
        add("Venv Python", ok and version.startswith("3."), version)
        pip = command([str(python), "-B", "-m", "pip", "check"], cwd=root, timeout=60)
        add("Requirements", bool(pip and pip.returncode == 0), "pip check passed" if pip and pip.returncode == 0 else "Dependency check failed; run scripts/setup.ps1")
        installed = json_result(command([str(python), "-B", str(root / "tools/check_requirements.py")], cwd=root))
        add("Required package versions", installed.get("ready") is True, "Project requirements satisfied" if installed.get("ready") is True else "Required packages missing/incompatible; run scripts/setup.ps1")
        retrieval = json_result(command([str(python), "-B", str(root / "tools/check_local_retrieval.py")], cwd=root))
        storage = retrieval.get("runtime_storage", {})
        add("Private runtime storage", storage.get("ready") is True,
            "Local filesystem; " + storage.get("state", "unavailable") + "; no database opened or probe write")
        add("Local retrieval FTS5", retrieval.get("fts5") is True, "Available" if retrieval.get("fts5") is True else "Unavailable; run setup with an FTS5-enabled Python/SQLite runtime")
        acceleration = retrieval.get("acceleration", {})
        add("Embedding acceleration", bool(acceleration.get("cpu_available")), acceleration.get("status", "CPU runtime unavailable"))
        for device in ("cuda", "directml"):
            optional = root / ".local/accelerators" / device / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            if optional.is_file():
                capability = json_result(command([str(optional), "-B", str(root / "tools/accelerator_capabilities.py")], cwd=root))
                add("Optional " + device + " runtime", False, capability.get("status", "UNAVAILABLE") + "; discovery only, real acceptance is separate", warning=True)
        add("Local embedding artifacts", retrieval.get("model_artifacts") == "verified",
            "Pinned model checksums verified; 384 dimensions; no inference/network request" if retrieval.get("model_artifacts") == "verified"
            else "Local model absent/invalid; explicit tools/local_retrieval.py prepare-model required for semantic indexing/search. Offline core tests remain available.", warning=True)
    node = shutil.which("node")
    ok, version = executable_version(node or "node", (22, 12))
    add("Node", ok, version + " (requires Node 22.12+)")
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    npm_result = command([npm, "--version"]) if npm else None
    add("npm", bool(npm_result and npm_result.returncode == 0), "PRESENT" if npm_result and npm_result.returncode == 0 else "MISSING")
    add("node_modules", (root / "extension/node_modules/typescript/package.json").is_file(), "Run scripts/setup.ps1 if missing")
    manifest = root / "extension/dist/manifest.json"
    add("Extension build", manifest.is_file(), "extension/dist/manifest.json exists" if manifest.is_file() else "MISSING; run scripts/build.ps1")
    if manifest.is_file():
        try:
            built = json.loads(manifest.read_text(encoding="utf-8"))
            source = json.loads((root / "extension/public/manifest.json").read_text(encoding="utf-8"))
            add("Manifest", built == source, "Source/build manifest match; version " + source["version"] if built == source else "Build differs; rebuild/reload extension/dist")
        except (OSError, ValueError, KeyError):
            add("Manifest", False, "Manifest could not be parsed")
    if node:
        metadata = json_result(command([node, str(root / "tools/build-metadata.mjs"), "--check"], cwd=root))
        add("Build freshness", metadata.get("fresh") is True, "Current" if metadata.get("fresh") is True else "Stale/missing; run scripts/build.ps1, then reload extension/dist")
        add("Protocol", metadata.get("compatible") is True, "Backend/extension compatible" if metadata.get("compatible") is True else "Mismatch/missing build metadata; rebuild/reload extension and restart backend")
        contract = json.loads((root / "shared/build-contract.json").read_text(encoding="utf-8"))
        add("Backend protocol (source)", True, str(contract["protocol_version"]) + "/" + str(contract["protocol_revision"]))
        add("Extension protocol (dist)", metadata.get("compatible") is True, str(metadata.get("protocol_version", "unknown")) + "/" + str(metadata.get("protocol_revision", "unknown")))
        parity = json_result(command([node, str(root / "tools/build-metadata.mjs"), "--scope"], cwd=root))
        add("Extension product scope", parity.get("scope_parity") is True, "Normal HTTP(S) URLs; internal browser pages rejected" if parity.get("scope_parity") is True else "Scope parity failed or dependencies missing")
    git_available = ((root / ".git").exists() and shutil.which("git") is not None
        and not any(name in os.environ for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR")))
    if git_available:
        tracked = command(["git", "-C", str(root), "ls-files", "-z"])
        paths = tracked.stdout.split("\0") if tracked and tracked.returncode == 0 else []
        dangerous = dangerous_tracked(paths)
        add("Tracked private files", tracked is not None and tracked.returncode == 0 and not dangerous, f"{len(dangerous)} dangerous tracked files; contents NEVER displayed")
        ignored = command(["git", "-C", str(root), "check-ignore", "--no-index", "backend/.env", ".env", ".agent_token", "local-private.pem", "local-private.key", "local-private.pfx", "recon/local.html", "backend/audit/local.jsonl", "backend/.venv/local", ".local/documents.sqlite3", ".local/retrieval.sqlite3", ".local/models/model.onnx"])
        ignored_count = len(ignored.stdout.splitlines()) if ignored else 0
        add("Secret/cache ignores", ignored_count == 12, "Sensitive local files ignored" if ignored_count == 12 else "Missing ignore protection; inspect .gitignore")
        untracked = command(["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"])
        source_count = len([x for x in (untracked.stdout.split("\0") if untracked else []) if x])
        if source_count:
            add("Untracked project files", False, f"{source_count} untracked files; review and include required source before moving to another machine.", warning=True)
    else:
        add("Git publication checks", False, "Unavailable for this source snapshot; review private files and ignore rules before publishing.", warning=True)
    if python.is_file():
        result = command([str(python), "-B", "-m", "browser.config_check", "--json"], cwd=root / "backend")
        config = json_result(result)
        add("Backend config", config.get("valid") is True, config.get("message", "Config checker unavailable; run setup"))
        add("Product scope / host permissions", config.get("product_scope") is True, "Generic URL validation and optional site grants checked")
        add("Backend connection token", config.get("backend_token") == "PRESENT", config.get("backend_token", "MISSING") + "; values NEVER printed")
        add("Active provider", config.get("valid") is True, config.get("active_provider", "UNAVAILABLE") + " (configuration only; no live request)")
        server = config.get("server", {})
        if server:
            add("Loopback binding", server["host"] in {"127.0.0.1", "localhost", "::1"}, "Loopback only")
            state = port_status(server["host"], server["port"])
            add("Backend port", state == "free", f"Port {server['port']}: {state}; no processes terminated", warning=True)
        if config.get("active_provider") == "mock":
            add("Live provider", False, "LIVE PROVIDER NOT CONFIGURED (mock selected); live services are not required for local development.", warning=True)
        else:
            add("Live provider", False, "LIVE VERIFICATION PENDING; not contacted by doctor.", warning=True)
    return dict(result="LOCAL READY" if all(x["status"] != "FAIL" for x in checks) else "NOT READY", checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    if args.port is not None:
        state = port_status(args.host, args.port)
        print(state)
        raise SystemExit(0 if state == "free" else 2)
    try:
        result = diagnose()
    except Exception:
        result = dict(result="NOT READY", checks=[dict(name="Diagnostics", status="FAIL", detail="Missing or malformed operational inputs. Run setup; values are never printed.")])
    if args.json:
        print(json.dumps(result))
    else:
        print("AI Browser Agent Doctor")
        for check in result["checks"]:
            print(f"{check['name']}: {check['status']} - {check['detail']}")
        print("Result: " + result["result"])
    raise SystemExit(0 if result["result"] == "LOCAL READY" else 2)


if __name__ == "__main__":
    main()
