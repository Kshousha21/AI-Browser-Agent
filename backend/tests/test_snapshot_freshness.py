"""Test the real runtime's freshness using deterministic platform primitives."""
import pytest

from .test_product_scope import node_check


DOM = r"""
// Only platform DOM/observer primitives are synthetic, never runtime decisions.
let clock = 100;
Object.defineProperty(globalThis, 'performance', {configurable: true, value: {now: () => clock}});
const observers = [];
global.MutationObserver = class {
  constructor(callback) { this.callback = callback; this.records = []; this.queued = false; }
  observe() { observers.push(this); }
  takeRecords() { return this.records.splice(0); }
  disconnect() { this.records.length = 0; }
  enqueue(record) {
    this.records.push(record);
    if (this.queued) return;
    this.queued = true;
    queueMicrotask(() => {
      this.queued = false;
      const records = this.takeRecords();
      if (records.length) this.callback(records, this);
    });
  }
};
function mutation(record) { for (const observer of observers) observer.enqueue(record); }
global.Node = { ELEMENT_NODE: 1, TEXT_NODE: 3 };
global.HTMLElement = class {
  constructor(tag) {
    this.tagName = tag; this.nodeType = Node.ELEMENT_NODE;
    this.attributes = new Map(); this.childNodes = []; this.parentElement = null;
    this.hidden = false; this.isContentEditable = false; this.scrollHeight = 800;
  }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  hasAttribute(name) { return this.attributes.has(name); }
  setAttribute(name, value) {
    const oldValue = this.getAttribute(name);
    this.attributes.set(name, String(value));
    mutation({type: 'attributes', target: this, attributeName: name, oldValue});
  }
  checkVisibility() { return true; }
  getBoundingClientRect() { return {x: 0, y: 0, left: 0, top: 0, right: 600, bottom: 800, width: 600, height: 800}; }
  get textContent() { return this.childNodes.map(node => node.textContent).join(''); }
  set textContent(text) {
    this.childNodes = [{nodeType: Node.TEXT_NODE, textContent: text}];
    mutation({type: 'childList', target: this});
  }
};
global.Element = HTMLElement;
global.SVGElement = class {};
global.HTMLInputElement = class extends HTMLElement {};
global.getComputedStyle = () => ({display: 'block', visibility: 'visible', opacity: '1',
  fontSize: '16px', position: 'static', color: 'rgb(0,0,0)', backgroundColor: 'rgb(255,255,255)', cursor: 'auto'});
global.innerWidth = 1000; global.innerHeight = 800;
global.location = new URL('http://127.0.0.1:8080/synthetic-selection.html');
const html = new HTMLElement('HTML'), body = new HTMLElement('BODY');
body.parentElement = html; body.textContent = 'Before mutation';
global.document = {documentElement: html, body, title: 'Synthetic selection', location,
  readyState: 'complete', querySelector: () => null,
  querySelectorAll: selector => selector === '*' ? [html, body] : []};
global.fetch = undefined;
global.XMLHttpRequest = class { send() {} };
global.history = {
  pushState(state, title, url) { if (url !== undefined) location.href = new URL(url, location.href).href; },
  replaceState(state, title, url) { if (url !== undefined) location.href = new URL(url, location.href).href; }
};
global.addEventListener = () => {};
global.dispatchEvent = () => true;
global.CustomEvent = class { constructor(type) { this.type = type; } };
new Function(require('fs').readFileSync('public/content/instrument.js', 'utf8'))();
const {BrowserRuntime} = load('src/content/executor.ts');
const runtime = new BrowserRuntime();
runtime.bindTab(1);
const options = {maxVisibleTextChars: 2000, maxElements: 20};
const first = runtime.extract(options);
"""


@pytest.mark.parametrize("change", ["dom", "navigation"])
def test_observation_covers_queued_pre_extraction_mutation(change):
    actual = node_check(DOM + r"""
(async () => {
  clock = 200;
  if (data.change === 'dom') body.textContent = 'Covered by fresh observation';
  else history.pushState({}, '', '?generation=2');
  const refreshed = runtime.extract(options);
  const beforeDelivery = runtime.identity();
  await Promise.resolve();
  const afterDelivery = runtime.identity();
  console.log(JSON.stringify({beforeDelivery, afterDelivery,
    changedSnapshot: first.snapshot_id !== refreshed.snapshot_id,
    capturedText: refreshed.visible_text, navigation: refreshed.navigation_id,
    accountedMutations: runtime.domVersion, lastMutationAt: runtime.lastMutationAt}));
})().catch(error => {console.error(error); process.exit(1);});
""", {"change": change})
    assert actual["changedSnapshot"] is True
    assert actual["beforeDelivery"]["unchanged"] is True
    assert actual["afterDelivery"] == actual["beforeDelivery"]
    assert actual["capturedText"] == (
        "Covered by fresh observation" if change == "dom" else "Before mutation"
    )
    assert actual["navigation"] == (1 if change == "navigation" else 0)
    assert actual["accountedMutations"] == 1
    assert actual["lastMutationAt"] == 200


@pytest.mark.parametrize("change", ["dom", "navigation", "same_url_navigation", "invalidate"])
def test_genuine_post_observation_changes_remain_stale(change):
    actual = node_check(DOM + r"""
(async () => {
  // First queued mutation is covered. The second is after extraction but
  // before the same observer-delivery microtask: it must NOT be discarded.
  clock = 200;
  body.textContent = 'Covered by fresh observation';
  const refreshed = runtime.extract(options);
  clock = 300;
  if (data.change === 'dom') body.textContent = 'Changed after observation';
  if (data.change === 'navigation') history.pushState({}, '', '?generation=3');
  if (data.change === 'same_url_navigation') history.replaceState({}, '', location.href);
  if (data.change === 'invalidate') runtime.invalidate();
  await Promise.resolve();
  console.log(JSON.stringify({identity: runtime.identity(),
    documentUnchanged: runtime.identity().document_id === refreshed.document_id,
    snapshotUnchanged: runtime.identity().snapshot_id === refreshed.snapshot_id,
    accountedMutations: runtime.domVersion, lastMutationAt: runtime.lastMutationAt}));
})().catch(error => {console.error(error); process.exit(1);});
""", {"change": change})
    assert actual["identity"]["unchanged"] is False
    assert actual["documentUnchanged"] is True
    assert actual["snapshotUnchanged"] is (change != "invalidate")
    assert actual["accountedMutations"] == (1 if change == "invalidate" else 2)
    assert actual["lastMutationAt"] == (200 if change == "invalidate" else 300)
    if change == "invalidate":
        assert actual["identity"]["snapshot_id"] == ""


def test_post_observation_agent_decoration_does_not_hide_real_mutation():
    actual = node_check(DOM + r"""
(async () => {
  body.setAttribute('data-agent-id', 'f0:e-001');
  body.setAttribute('data-agent-highlight', 'true');
  await Promise.resolve();
  const decorated = runtime.identity();
  body.setAttribute('data-agent-highlight', 'false');
  body.textContent = 'Real mutation alongside decoration';
  await Promise.resolve();
  console.log(JSON.stringify({decorated, mutated: runtime.identity()}));
})().catch(error => {console.error(error); process.exit(1);});
""", {})
    assert actual["decorated"]["unchanged"] is True
    assert actual["mutated"]["unchanged"] is False
    assert actual["mutated"]["snapshot_id"] == actual["decorated"]["snapshot_id"]


@pytest.mark.parametrize("change", ["navigation", "same_url_navigation"])
def test_real_preview_dispatch_rejects_navigation_after_fresh_observation(change):
    actual = node_check(DOM + r"""
(async () => {
  const tools = load('src/sidepanel/browser-tools.ts');
  let navigations = 0, leases = 0;
  global.__aiBrowserAgentContent = {identity: () => runtime.identity()};
  global.chrome = {
    tabs: {
      get: async () => ({id: 1, windowId: 2, active: true, status: 'complete', url: location.href}),
      update: async () => { navigations++; throw Error('Stale action reached navigation'); }
    },
    scripting: {executeScript: async request => [{documentId: 'chrome-synthetic',
      result: await request.func(...(request.args ?? []))}]},
    runtime: {connect: () => {leases++; throw Error('Stale action acquired lease');}}
  };
  tools.browserSession.bindOwner(1, 2);
  tools.browserSession.begin();
  tools.browserSession.runId = 'synthetic-run';
  const selected = {...first, tab_id: 1};
  tools.browserSession.adopt(selected);
  // Exercise the actual content guard even before a Chrome tab-update event:
  // same-URL navigation must still retire the old snapshot's authority.
  history.pushState({}, '', data.change === 'navigation' ? '?generation=2' : location.href);
  await Promise.resolve();
  const response = await tools.executeBrowserTool({
    tool: 'open_attachment_preview', run_id: 'synthetic-run', control_epoch: 0,
    tab_id: 1, document_id: selected.document_id, snapshot_id: selected.snapshot_id,
    arguments: {preview_id: 'a'.repeat(32),
      url: 'http://127.0.0.1:8080/portal/spaces/DEMO/pages/100/Synthetic?preview=%2F100%2F200%2Fsample.pdf',
      attachment: {parent_id: '100', attachment_id: '200', filename: 'sample.pdf', version: null}}
  }, {allowlist: ['http://127.0.0.1:8080/*']});
  console.log(JSON.stringify({success: response.success, error: response.error,
    navigations, leases, sameSnapshot: runtime.identity().snapshot_id === selected.snapshot_id,
    sameDocument: runtime.identity().document_id === selected.document_id}));
})().catch(error => {console.error(error); process.exit(1);});
""", {"change": change})
    assert actual == {
        "success": False, "error": "stale_snapshot", "navigations": 0,
        "leases": 0, "sameSnapshot": True, "sameDocument": True,
    }
