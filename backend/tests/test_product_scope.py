"""Product admission and persisted settings must not expand the shipped URL scope."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from browser.product_scope import product_pattern_allowed, product_url_allowed
from browser.protocol import PageState, UserMessagePayload, make_envelope
from browser.providers import MockProvider, ToolCall
from browser.run_session import handle_user_message
from browser.session_bridge import WebSocketBridge

ORIGIN = "https://example.org"


def node_check(script, data):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for extension scope/settings regression")
    extension = Path(__file__).resolve().parents[2] / "extension"
    bootstrap = r"""
const {buildSync} = require('esbuild');
const data = JSON.parse(require('fs').readFileSync(0, 'utf8'));
function load(entry) {
  const code = buildSync({entryPoints:[entry],bundle:true,write:false,format:'cjs',platform:'node'}).outputFiles[0].text;
  const module = {exports:{}};
  new Function('module','exports','require',code)(module,module.exports,require);
  return module.exports;
}
"""
    result = subprocess.run([node, "-e", bootstrap + script], cwd=extension,
        input=json.dumps(data), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)









def test_python_and_extension_accept_websites_and_reject_internal_pages():
    urls=["https://example.org/docs","http://news.example:8080/a","https://example.org/portal/","chrome://settings","file:///private.pdf","https://user:pass@example.org"]
    expected=[True,True,True,False,False,False]
    assert [product_url_allowed(u) for u in urls]==expected
    actual=node_check("const scope=load('src/lib/product-scope.ts'); console.log(JSON.stringify(data.map(scope.isProductUrl)));",urls)
    assert actual==expected

async def test_internal_pages_rejected_before_provider(settings):
    provider=MockProvider(scripted=["must not run"])
    for url in ["chrome://settings","file:///private.pdf","about:blank"]:
        payload=UserMessagePayload(text="Read",page_state=PageState(url=url))
        bridge=AsyncMock()
        await handle_user_message(provider,settings,make_envelope("user_message",payload),payload,bridge)
        assert bridge.send.await_args.args[1]["code"]=="bad_request"
    assert not provider.calls
