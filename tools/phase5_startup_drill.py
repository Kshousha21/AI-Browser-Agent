"""Start/stop only an isolated Phase 5 clone with transient synthetic settings."""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.request

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    root = args.source_root.resolve()
    assert root.is_relative_to(ROOT / ".test-artifacts")
    assert (root / ".local/phase5-release-drill/synthetic-owner.json").is_file()
    assert not (root / "backend/.env").exists()
    env = {k: v for k, v in os.environ.items() if not k.startswith("BROWSER_")
           and k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}}
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    token = secrets.token_hex(24)
    origin = "chrome-extension://" + "a" * 32
    env.update(BROWSER_CONFIG=str(root / "backend/config.yaml"),
        BROWSER_PROVIDER__ACTIVE="mock", BROWSER_AUTH__TOKEN=token,
        BROWSER_SERVER__PORT=str(port), BROWSER_SERVER__HOST="127.0.0.1",
        BROWSER_SERVER__ALLOW_ANY_EXTENSION_ORIGIN="false",
        BROWSER_SERVER__ALLOWED_ORIGINS=json.dumps([origin]))
    command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "RemoteSigned",
               "-File", str(root / "scripts/start-backend.ps1")]
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def health():
        with http.open(f"http://127.0.0.1:{port}/health", timeout=1) as response:
            return json.load(response)
    def frame(kind, payload):
        return json.dumps({"type": kind, "payload": payload})
    with tempfile.TemporaryFile() as output:
        started = time.perf_counter()
        child = subprocess.Popen(command, cwd=root, env=env, stdout=output,
            stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.monotonic() + 20
            while True:
                assert child.poll() is None, "test backend exited before readiness"
                try:
                    report = health()
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise AssertionError("test backend readiness deadline") from None
                    time.sleep(0.1)
            startup = time.perf_counter() - started
            assert report["active_provider"] == "mock" and "read_only_default" not in report
            for revision in (2, 999):
                with connect(f"ws://127.0.0.1:{port}/ws", origin=origin, proxy=None,
                             open_timeout=3, close_timeout=2) as ws:
                    ws.send(frame("auth", {"token": token}))
                    ws.send(frame("hello", {"client": "synthetic-release-drill",
                        "protocol_version": 1, "protocol_revision": revision}))
                    answer = json.loads(ws.recv(timeout=3))
                    assert answer["type"] == ("hello_ack" if revision == 2 else "error")
                    if revision != 2:
                        assert answer["payload"]["code"] == "protocol_mismatch"
            occupied = subprocess.run(command, cwd=root, env=env, capture_output=True,
                text=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
            assert occupied.returncode != 0
            assert "occupied or unavailable" in " ".join((occupied.stdout + occupied.stderr).split())
            assert health()["status"] == "ok"
        finally:
            # Fault drill: stop only this newly spawned wrapper and its children.
            # No pre-existing backend/process is discovered or terminated.
            if child.poll() is None:
                subprocess.run(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"],
                    capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            child.wait(timeout=10)
            output.seek(0)
            assert token.encode() not in output.read()
    with socket.socket() as probe:
        probe.settimeout(1)
        assert probe.connect_ex(("127.0.0.1", port)) != 0
    print(json.dumps({"startup_seconds": round(startup, 4), "health": "PASS",
        "exact_origin_authenticated_handshake": "PASS", "protocol_mismatch": "REJECTED",
        "occupied_port": "REJECTED_EXISTING_LISTENER_PRESERVED",
        "owned_backend_stopped": True, "test_token_logged": False,
        "provider_website_calls": 0}))


if __name__ == "__main__":
    main()
