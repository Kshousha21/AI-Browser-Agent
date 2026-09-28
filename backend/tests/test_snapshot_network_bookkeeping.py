"""Real instrumentation/runtime: counters are not document/target changes."""
import pytest

from .test_product_scope import node_check
from .test_snapshot_freshness import DOM


NETWORK_DOM = DOM.replace("global.fetch = undefined;", """
let finishFetch;
global.fetch = () => new Promise(resolve => { finishFetch = () => resolve({ok: true}); });
""").replace("global.XMLHttpRequest = class { send() {} };", """
global.XMLHttpRequest = class {
  callbacks=[]; send() {} addEventListener(name, callback) {if(name==='loadend')this.callbacks.push(callback);}
  complete() {this.callbacks.splice(0).forEach(callback=>callback());}
};
""")


@pytest.mark.parametrize("stage", ["start", "completion", "roundtrip"])
def test_xhr_bookkeeping_alone_preserves_exact_snapshot(stage):
    actual = node_check(NETWORK_DOM + r"""
(async()=>{const xhr=new XMLHttpRequest();xhr.send();
  if(data.stage==='completion')runtime.extract(options);
  const before=runtime.identity();if(data.stage!=='start')xhr.complete();await Promise.resolve();
  const after=runtime.identity();console.log(JSON.stringify({sameDocument:before.document_id===after.document_id,
    sameSnapshot:before.snapshot_id===after.snapshot_id,unchanged:after.unchanged}));})();
""", dict(stage=stage))
    assert actual == dict(sameDocument=True, sameSnapshot=True, unchanged=True)


@pytest.mark.parametrize("change", ["navigation", "dialog", "malformed", "extra_field"])
def test_transient_instrument_change_then_restore_still_invalidates(change):
    actual = node_check(NETWORK_DOM + r"""
const before=html.getAttribute('data-ai-browser-agent-runtime'), changed=JSON.parse(before);
if(data.change==='navigation')changed.navigation++;
if(data.change==='dialog')changed.dialogs.push({type:'alert',message:'Synthetic',dismissed:true});
if(data.change==='extra_field')changed.unexpected=true;
html.setAttribute('data-ai-browser-agent-runtime',data.change==='malformed'?'invalid':JSON.stringify(changed));
html.setAttribute('data-ai-browser-agent-runtime',before);
// identity() must account for pending records even before observer delivery.
console.log(JSON.stringify({unchanged:runtime.identity().unchanged}));
""", dict(change=change))
    assert actual == dict(unchanged=False)


@pytest.mark.parametrize("stage", ["start", "completion", "roundtrip"])
def test_fetch_bookkeeping_alone_preserves_exact_snapshot(stage):
    actual = node_check(NETWORK_DOM + r"""
(async () => {
  const pending = fetch('http://127.0.0.1:8080/synthetic-count-only');
  if (data.stage === 'completion') runtime.extract(options);
  const before = runtime.identity();
  if (data.stage !== 'start') { finishFetch(); await pending; }
  await Promise.resolve();
  const after = runtime.identity();
  console.log(JSON.stringify({sameDocument: before.document_id === after.document_id,
    sameSnapshot: before.snapshot_id === after.snapshot_id, unchanged: after.unchanged}));
})();
""", dict(stage=stage))
    assert actual == dict(sameDocument=True, sameSnapshot=True, unchanged=True)


@pytest.mark.parametrize("change", ["dom", "navigation", "same_url_navigation", "dialog", "malformed", "other_target"])
def test_counter_bookkeeping_never_masks_security_relevant_changes(change):
    actual = node_check(NETWORK_DOM + r"""
(async () => {
  const pending = fetch('http://127.0.0.1:8080/synthetic-count-only');
  runtime.extract(options);
  if(data.change==='dom')body.textContent='Changed target evidence';
  if(data.change==='navigation')history.pushState({},'', '?changed=1');
  if(data.change==='same_url_navigation')history.replaceState({},'',location.href);
  if(data.change==='dialog')alert('Synthetic dialog');
  finishFetch(); await pending;
  if(data.change==='malformed')html.setAttribute('data-ai-browser-agent-runtime','invalid');
  if(data.change==='other_target')body.setAttribute('data-ai-browser-agent-runtime',html.getAttribute('data-ai-browser-agent-runtime'));
  await Promise.resolve();
  console.log(JSON.stringify({unchanged:runtime.identity().unchanged}));
})();
""", dict(change=change))
    assert actual == dict(unchanged=False)


@pytest.mark.parametrize("stage", ["before_open", "during_lease"])
def test_real_preview_dispatch_survives_only_network_bookkeeping(stage):
    actual = node_check(NETWORK_DOM + r"""
(async () => {
  global.window=global;
  const tools=load('src/sidepanel/browser-tools.ts');let navigations=0,leases=0;
  const listeners=()=>{const values=new Set();return{addListener:f=>values.add(f),removeListener:f=>values.delete(f),emit:x=>[...values].forEach(f=>f(x))}};
  const messages=listeners(),disconnected=listeners();
  const pending=fetch('http://127.0.0.1:8080/synthetic-count-only');
  const selected={...runtime.extract(options),tab_id:1};
  if(data.stage==='before_open'){finishFetch();await pending;}
  global.__aiBrowserAgentContent={identity:()=>runtime.identity()};
  global.chrome={tabs:{get:async()=>({id:1,windowId:2,active:true,status:'complete',url:location.href}),
    update:async()=>{navigations++;throw Error('Synthetic navigation boundary');}},
    scripting:{executeScript:async request=>[{documentId:'chrome-synthetic',result:await request.func(...(request.args??[]))}]},
    runtime:{connect:()=>{leases++;return{onMessage:messages,onDisconnect:disconnected,
      disconnect(){disconnected.emit({})},postMessage(message){if(message.tabId)queueMicrotask(async()=>{
        if(data.stage==='during_lease'){finishFetch();await pending;}messages.emit({ready:true});
      })}}}}};
  tools.browserSession.bindOwner(1,2);tools.browserSession.begin();tools.browserSession.runId='synthetic-run';tools.browserSession.adopt(selected);
  const response=await tools.executeBrowserTool({tool:'open_attachment_preview',run_id:'synthetic-run',control_epoch:0,
    tab_id:1,document_id:selected.document_id,snapshot_id:selected.snapshot_id,arguments:{preview_id:'a'.repeat(32),
      url:'http://127.0.0.1:8080/portal/spaces/DEMO/pages/100/Synthetic?preview=%2F100%2F200%2Fsample.pdf',
      attachment:{parent_id:'100',attachment_id:'200',filename:'sample.pdf',version:null}}},{allowlist:['http://127.0.0.1:8080/*']});
  tools.browserSession.stop();
  console.log(JSON.stringify({navigations,leases,stale:response.error==='stale_snapshot'}));
})().catch(error=>{console.error(error);process.exit(1)});
""", dict(stage=stage))
    assert actual == dict(navigations=1, leases=1, stale=False)
