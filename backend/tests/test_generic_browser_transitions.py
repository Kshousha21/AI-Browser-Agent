"""Neutral real-Chrome regression for occlusion recovery and document lineage.

Uses the existing local CDP harness; all HTML is served from loopback.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from browser.context_control import page_observation, serialized_size
from browser.protocol import PageState
from .chrome_dom_support import (
    BUNDLE, _Cdp, _close_target, _json_request, _wait_for_document,
    chrome_debug_port, websocket_client,
)

OPTIONS = {"maxVisibleTextChars": 20000, "maxElements": 300}


@pytest.fixture(scope="module")
def transitions(chrome_debug_port: int):
    html = (Path(__file__).parent / "test_site" / "document-search.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?", 1)[0] != "/document-search.html":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/document-search.html"
    target = _json_request(chrome_debug_port, "/json/new?about:blank", "PUT")
    try:
        with websocket_client.connect(target["webSocketDebuggerUrl"], origin="http://127.0.0.1", close_timeout=2) as websocket:
            cdp = _Cdp(websocket)
            cdp.call("Page.enable")
            cdp.call("Runtime.enable")
            cdp.call("Page.navigate", url=url)
            _wait_for_document(cdp, url)

            def inject():
                cdp.evaluate((BUNDLE.parent / "instrument.js").read_text(encoding="utf-8"))
                cdp.evaluate(BUNDLE.read_text(encoding="utf-8"))

            def extract():
                return PageState.model_validate(cdp.evaluate(
                    "__aiBrowserAgentContent.extractPageState(" + json.dumps(OPTIONS) + ")"
                ))

            def execute(tool, arguments, snapshot):
                call = {"tool": tool, "arguments": arguments, "snapshotId": snapshot}
                return cdp.evaluate("__aiBrowserAgentContent.executeTool(" + json.dumps(call) + "," + json.dumps(OPTIONS) + ")")

            inject()
            initial = extract()
            search = next(item for item in initial.elements if item.name == "Search documents")
            button = next(item for item in initial.elements if item.name == "Search")
            typed = execute("type_text", {"element_id": search.agent_id, "text": "Sample notes"}, initial.snapshot_id)
            blocked = execute("click", {"element_id": button.agent_id}, initial.snapshot_id)
            observed = cdp.evaluate("document.getElementById('query').dataset.observedValue")
            stable = execute("wait_for_stable", {"timeout_ms": 10000}, initial.snapshot_id)
            fresh = extract()
            fresh_button = next(item for item in fresh.elements if item.name == "Search")
            try:
                retry = execute("click", {"element_id": fresh_button.agent_id}, fresh.snapshot_id)
            except AssertionError as error:
                # Navigation may destroy the evaluation context after the click commits.
                if "context" not in str(error).lower():
                    raise
                retry = {"context_replaced": True}
            _wait_for_document(cdp, url + "?q=Sample+notes")
            inject()
            results = extract()
            first = next(item for item in results.elements if item.name == "Document 1")
            opened = execute("click", {"element_id": first.agent_id}, results.snapshot_id)
            settled = execute("wait_for_stable", {"timeout_ms": 10000}, results.snapshot_id)
            detail = extract()
            yield locals()
    finally:
        _close_target(chrome_debug_port, target["id"])
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_occluded_control_requires_fresh_snapshot_and_reacquisition(transitions):
    run = transitions
    assert run["typed"]["success"] is True
    assert run["observed"] == "Sample notes"
    assert run["blocked"]["success"] is False
    assert run["blocked"]["error"] == "element_occluded"
    assert run["stable"]["success"] is True
    assert run["fresh"].snapshot_id != run["initial"].snapshot_id
    assert run["fresh_button"].occluded is False
    assert run["retry"].get("success") or run["retry"].get("context_replaced")
    assert run["results"].url.endswith("?q=Sample+notes")


def test_search_reload_replaces_document_and_forces_full_observation(transitions):
    initial, results = transitions["initial"], transitions["results"]
    assert results.document_id != initial.document_id
    # History navigation counters restart inside a new document runtime.
    assert results.navigation_id == 0
    assert results.snapshot_id != initial.snapshot_id
    assert page_observation(initial, results)["type"] == "page_state_full"
    assert results.site_context is None


def test_same_document_panel_mutation_keeps_lineage_and_emits_bounded_diff(transitions):
    run = transitions
    results, detail = run["results"], run["detail"]
    assert run["opened"]["success"] is True
    assert run["settled"]["success"] is True
    assert (detail.url, detail.document_id, detail.navigation_id) == (results.url, results.document_id, results.navigation_id)
    assert detail.snapshot_id != results.snapshot_id
    assert "Details inserted into the same top-level DOM." in detail.visible_text
    assert "HIDDEN_FIXTURE_CANARY" not in detail.visible_text
    assert len({item.agent_id for item in detail.elements}) == len(detail.elements)
    assert any(region.data_key == "document-1" for region in detail.semantic_regions)
    assert next(item for item in detail.elements if item.name == "Document 1").selected
    observation = page_observation(results, detail)
    assert observation["type"] == "page_state_diff"
    assert serialized_size(observation) < serialized_size(page_observation(None, detail))
