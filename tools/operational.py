"""Standard-library operational checks. No secret contents in diagnostic output."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

DIRECTORY_MARKERS = ("backend", "extension")
FILE_MARKER = "shared/build-contract.json"


def command(args, *, cwd=None, timeout=30):
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None


def repository_root(candidate: Path) -> Path:
    # Keep the caller's drive spelling; resolve() can turn it into a UNC path.
    root = Path(os.path.abspath(candidate))
    if not all((root / marker).is_dir() for marker in DIRECTORY_MARKERS) or not (root / FILE_MARKER).is_file():
        raise ValueError("Not the project root. Expected backend/, extension/ and shared/build-contract.json.")
    # Source snapshots work without Git; never discover or redirect to a parent.
    return root


def dangerous_tracked(paths: list[str]) -> list[str]:
    found = []
    for name in paths:
        path = Path(name.replace("\\", "/"))
        base = path.name.lower()
        parts = {part.lower() for part in path.parts}
        secret_env = (base == ".env" or base.startswith(".env.")) and base != ".env.example"
        if secret_env or base == ".agent_token" or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx", ".crt", ".cer"} or parts.intersection({"recon", "audit", "logs", ".local"}):
            found.append(name)
    return found


def port_status(host: str, port: int) -> str:
    if host not in {"localhost", "127.0.0.1", "::1"}:
        return "non_loopback"
    sockets = []
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        seen = set()
        for family, kind, protocol, _, address in addresses:
            if (family, address) in seen:
                continue
            seen.add((family, address))
            probe = socket.socket(family, kind, protocol)
            sockets.append(probe)
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(address)
        return "free"
    except OSError:
        return "occupied_or_unavailable"
    finally:
        for probe in sockets:
            probe.close()


def json_result(result):
    try:
        return json.loads(result.stdout) if result else {}
    except (ValueError, TypeError):
        return {}


def executable_version(executable: str, minimum: tuple[int, int]):
    import re
    result = command([executable, "--version"])
    match = re.search(r"(?:Python )?v?(\d+)\.(\d+)\.(\d+)", result.stdout if result and not result.returncode else "")
    if not match:
        return False, "MISSING"
    version = tuple(int(x) for x in match.groups())
    return version[:2] >= minimum, ".".join(str(x) for x in version)
