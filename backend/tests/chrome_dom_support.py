"""Neutral isolated Chrome DOM harness; synthetic pages only."""

import base64

import json

import os

import re

import shutil

import socket

import subprocess

import tempfile

import time

from http.client import HTTPConnection

from pathlib import Path

from typing import Any, Callable

from urllib.parse import quote

import pytest

from browser.config import PageSettings

from browser.context_control import full_page_payload

from browser.prompting import clamp_page_state, model_safe_page

from browser.protocol import PageState

class _Cdp:
    def __init__(self, websocket: Any) -> None:
        self.websocket = websocket
        self.sequence = 0
        self.events: list[dict[str, Any]] = []

    def call(self, method: str, **params: object) -> dict[str, Any]:
        self.sequence += 1
        request_id = self.sequence
        self.websocket.send(
            json.dumps({"id": request_id, "method": method, "params": params})
        )
        while True:
            message = json.loads(self.websocket.recv(timeout=15))
            if message.get("id") != request_id:
                if "method" in message:
                    self.events.append(message)
                continue
            if "error" in message:
                raise AssertionError(f"CDP {method} failed: {message['error']}")
            return message.get("result", {})

    def event(self, method: str) -> dict[str, Any]:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            for index, message in enumerate(self.events):
                if message.get("method") == method:
                    return self.events.pop(index).get("params", {})
            message = json.loads(
                self.websocket.recv(timeout=max(0.1, deadline - time.monotonic()))
            )
            if message.get("method") == method:
                return message.get("params", {})
            if "method" in message:
                self.events.append(message)
        raise AssertionError(f"CDP event did not arrive: {method}")

    def navigate_with_html(self, url: str, html: bytes) -> None:
        self.sequence += 1
        navigation_id = self.sequence
        self.websocket.send(
            json.dumps(
                {
                    "id": navigation_id,
                    "method": "Page.navigate",
                    "params": {"url": url},
                }
            )
        )
        fulfillment_id: int | None = None
        navigation_done = False
        fulfillment_done = False
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            message = json.loads(
                self.websocket.recv(timeout=max(0.1, deadline - time.monotonic()))
            )
            if message.get("method") == "Fetch.requestPaused":
                paused = message.get("params", {})
                request = paused.get("request", {})
                assert request.get("url") == url
                self.sequence += 1
                fulfillment_id = self.sequence
                self.websocket.send(
                    json.dumps(
                        {
                            "id": fulfillment_id,
                            "method": "Fetch.fulfillRequest",
                            "params": {
                                "requestId": paused["requestId"],
                                "responseCode": 200,
                                "responseHeaders": [
                                    {
                                        "name": "Content-Type",
                                        "value": "text/html; charset=utf-8",
                                    },
                                    {
                                        "name": "Cache-Control",
                                        "value": "no-store",
                                    },
                                ],
                                "body": base64.b64encode(html).decode("ascii"),
                            },
                        }
                    )
                )
                continue
            if message.get("id") == navigation_id:
                if "error" in message:
                    raise AssertionError(
                        f"CDP Page.navigate failed: {message['error']}"
                    )
                navigation_done = True
            elif fulfillment_id is not None and message.get("id") == fulfillment_id:
                if "error" in message:
                    raise AssertionError(
                        f"CDP Fetch.fulfillRequest failed: {message['error']}"
                    )
                fulfillment_done = True
            elif "method" in message:
                self.events.append(message)
            if navigation_done and fulfillment_done:
                return
        raise AssertionError("intercepted fixture navigation did not complete")

    def evaluate(self, expression: str) -> Any:
        result = self.call(
            "Runtime.evaluate",
            expression=expression,
            awaitPromise=True,
            returnByValue=True,
        )
        if "exceptionDetails" in result:
            raise AssertionError(
                f"browser evaluation failed: {result['exceptionDetails']}"
            )
        return result.get("result", {}).get("value")

@pytest.fixture(scope="module")
def chrome_debug_port() -> int:
    if not BUNDLE.is_file():
        pytest.skip(f"built extractor unavailable: {BUNDLE}")
    bundle_text = BUNDLE.read_text(encoding="utf-8")
    if (
        "__aiBrowserAgentContent" not in bundle_text
        or "adapter_payload" not in bundle_text
    ):
        pytest.skip("built generic extractor bundle is unavailable or stale")
    chrome = _find_chrome()
    if chrome is None:
        pytest.skip("installed Google Chrome was not found")
    port = _free_loopback_port()
    profile = Path(tempfile.mkdtemp(prefix="ai-browser-dom-smoke-"))
    command = [
        str(chrome),
        "--headless=new",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-default-apps",
        "--disable-sync",
        "--metrics-recording-only",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-pings",
        "--proxy-server=direct://",
        "--proxy-bypass-list=*",
        "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1",
        "--remote-debugging-address=127.0.0.1",
        f"--remote-debugging-port={port}",
        "--remote-allow-origins=*",
        f"--user-data-dir={profile}",
        "about:blank",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        _wait_for_chrome(port, process)
        yield port
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        shutil.rmtree(profile, ignore_errors=True)

def _extract_html(
    port: int,
    html: str,
    url: str,
) -> tuple[dict[str, Any], int]:
    target = _json_request(port, f"/json/new?{quote('about:blank', safe='')}", "PUT")
    target_id = target["id"]
    try:
        with websocket_client.connect(
            target["webSocketDebuggerUrl"],
            origin="http://127.0.0.1",
            open_timeout=10,
            close_timeout=2,
        ) as websocket:
            cdp = _Cdp(websocket)
            cdp.call("Page.enable")
            cdp.call("Runtime.enable")
            cdp.call(
                "Fetch.enable",
                patterns=[{"urlPattern": url, "requestStage": "Request"}],
            )
            cdp.navigate_with_html(url, html.encode("utf-8"))
            _wait_for_document(cdp, url)
            cdp.call("Fetch.disable")
            cdp.evaluate(BUNDLE.read_text(encoding="utf-8"))
            raw = cdp.evaluate(
                "JSON.stringify(__aiBrowserAgentContent.extractPageState("
                "{maxVisibleTextChars:20000,maxElements:300}))"
            )
            assert isinstance(raw, str)
            value = json.loads(raw)
            assert isinstance(value, dict)
            dom_count = cdp.evaluate("document.querySelectorAll('*').length")
            assert isinstance(dom_count, int)
            return value, dom_count
    finally:
        _close_target(port, target_id)

def _wait_for_document(cdp: _Cdp, expected_url: str) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        ready = cdp.evaluate(
            "JSON.stringify({ready:document.readyState,url:location.href})"
        )
        if isinstance(ready, str):
            state = json.loads(ready)
            if state["ready"] == "complete" and state["url"] == expected_url:
                return
        time.sleep(0.05)
    raise AssertionError(f"fixture did not finish loading: {expected_url}")

def _json_request(
    port: int, path: str, method: str = "GET"
) -> dict[str, Any]:
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        value = json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()
    assert isinstance(value, dict)
    return value

def _close_target(port: int, target_id: str) -> None:
    connection = HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", f"/json/close/{target_id}")
        response = connection.getresponse()
        response.read()
    except OSError:
        pass
    finally:
        connection.close()

def _wait_for_chrome(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.skip("Google Chrome exited before DevTools became available")
        try:
            _json_request(port, "/json/version")
            return
        except OSError:
            time.sleep(0.05)
    pytest.skip("Google Chrome DevTools did not become available")

def _free_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])

def _find_chrome() -> Path | None:
    names = ("chrome", "google-chrome", "google-chrome-stable")
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
    candidates: list[Path] = []
    for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        root = os.environ.get(variable)
        if root:
            candidates.append(
                Path(root) / "Google" / "Chrome" / "Application" / "chrome.exe"
            )
    return next((path for path in candidates if path.is_file()), None)

def _assert_no_raw_html(value: object) -> None:
    if isinstance(value, dict):
        assert not (RAW_HTML_KEYS & {str(key).casefold() for key in value})
        for item in value.values():
            _assert_no_raw_html(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_raw_html(item)

def _assert_key_absent(value: object, forbidden: str) -> None:
    if isinstance(value, dict):
        assert forbidden not in value
        for item in value.values():
            _assert_key_absent(item, forbidden)
    elif isinstance(value, list):
        for item in value:
            _assert_key_absent(item, forbidden)


REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "extension/dist/content/extract.js"
EXTRACTOR_SOURCE = REPO / "extension/src/content/page.ts"
websocket_client = pytest.importorskip("websockets.sync.client")
