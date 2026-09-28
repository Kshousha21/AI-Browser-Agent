"""Native panel configuration and immutable owner boundaries, without Chrome/network."""
import pytest

from .test_product_scope import node_check
from .test_browser_control import run

PANEL_SETUP = r"""
const calls=[];
const origin='https://example.org';
const tabs=new Map([[1,{id:1,windowId:1,url:origin+'/portal/a'}],
 [2,{id:2,windowId:1,url:'chrome://settings/'}]]);
const saved=new Map();
let allowlist=[origin+'/portal/*','http://127.0.0.1/*'];
global.chrome={runtime:{id:'a'.repeat(32),getURL:p=>'chrome-extension://'+'a'.repeat(32)+'/'+p},
 storage:{local:{get:async()=>({settings:{allowlist}})}},
 tabs:{get:async id=>{if(!tabs.has(id))throw Error('closed');return {...tabs.get(id)};},query:async()=>[...tabs.values()]},
 sidePanel:{setOptions:async value=>{calls.push(['options',value]);saved.set(value.tabId??'default',{...saved.get(value.tabId??'default'),...value});},
 getOptions:async ({tabId})=>({...saved.get('default'),...saved.get(tabId),tabId}),
 setPanelBehavior:async value=>calls.push(['behavior',value]),open:async value=>calls.push(['open',value])}};
const scope=load('src/lib/panel-scope.ts');
const {TabPanels}=load('src/background/tab-panels.ts');
"""


def panel_run(script, data=None):
    return node_check(PANEL_SETUP + '\n(async()=>{\n' + script +
        '\n})().catch(e=>{console.error(e);process.exit(1);});', data or {})


@pytest.mark.parametrize('suffix', ['', '?tabId=-1', '?tabId=01', '?tabId=1.2',
    '?tabId=1&extra=1', '?tabId=1&tabId=2', '?tabId=9007199254740992', '?tabId=1#extra'])
def test_panel_identity_rejects_unbound_or_ambiguous_url(suffix):
    assert panel_run("console.log(JSON.stringify(scope.panelTabId(chrome.runtime.getURL('sidepanel/index.html'+data.suffix))));",
                     dict(suffix=suffix)) is None


def test_panel_identity_exact_extension_resource_and_safe_tab_id():
    actual = panel_run(r"""
console.log(JSON.stringify({zero:scope.panelTabId(chrome.runtime.getURL(scope.panelPath(0))),
 owner:scope.panelTabId(chrome.runtime.getURL(scope.panelPath(31))),
 foreign:scope.panelTabId('chrome-extension://'+'b'.repeat(32)+'/sidepanel/index.html?tabId=31'),
 other:scope.panelTabId(chrome.runtime.getURL('options/index.html?tabId=31'))}));
""")
    assert actual == dict(zero=0, owner=31, foreign=None, other=None)


@pytest.mark.parametrize('case', ['ordinary_tab', 'global', 'wrong_path', 'disabled', 'missing_context', 'closed_owner'])
def test_panel_boot_rejects_non_native_or_unavailable_owner(case):
    actual = panel_run(r"""
chrome.runtime.ContextType={SIDE_PANEL:'SIDE_PANEL'};
chrome.runtime.getContexts=async()=>data.case==='missing_context'?[]:[{contextType:'SIDE_PANEL'}];
chrome.tabs.getCurrent=async()=>data.case==='ordinary_tab'?{id:99}:undefined;
saved.set(1,{enabled:data.case!=='disabled',path:scope.panelPath(data.case==='wrong_path'?2:1)});
if(data.case==='closed_owner')tabs.delete(1);
let rejected=false;
try{await scope.resolvePanelOwner(chrome.runtime.getURL(data.case==='global'?'sidepanel/index.html':scope.panelPath(1)));}
catch{rejected=true;}
console.log(JSON.stringify({rejected}));
""", dict(case=case))
    assert actual['rejected']


def test_panel_boot_accepts_exact_native_owner_only():
    actual = panel_run(r"""
chrome.runtime.ContextType={SIDE_PANEL:'SIDE_PANEL'};
chrome.runtime.getContexts=async filter=>{calls.push(['contexts',filter]);return [{contextType:'SIDE_PANEL'}];};
chrome.tabs.getCurrent=async()=>undefined;
saved.set(1,{enabled:true,path:scope.panelPath(1)});
console.log(JSON.stringify({owner:await scope.resolvePanelOwner(chrome.runtime.getURL(scope.panelPath(1))),calls}));
""")
    assert actual['owner'] == 1
    assert actual['calls'][0][1]['contextTypes'] == ['SIDE_PANEL']


def test_cold_toolbar_gesture_does_not_require_initialized_worker_cache():
    actual = panel_run(r"""
const panels=new TabPanels();await panels.open(tabs.get(1));
console.log(JSON.stringify(calls.filter(x=>x[0]==='open')));
""")
    assert actual == [['open', {'tabId': 1}]]


def test_cold_toolbar_ignores_retired_allowlist_and_needs_no_storage():
    actual = panel_run(r"""
allowlist=[];
const prepared=new TabPanels();await prepared.initialize();
const before=calls.length;
chrome.storage.local.get=()=>{throw Error('No storage work inside gesture');};
chrome.sidePanel.open=async ({tabId})=>{if(!saved.get(tabId)?.enabled)throw Error('disabled');calls.push(['accepted',{tabId}]);};
let rejected=false;try{await new TabPanels().open(tabs.get(1));}catch{rejected=true;}
console.log(JSON.stringify({rejected,enabled:(await chrome.sidePanel.getOptions({tabId:1})).enabled,calls:calls.slice(before)}));
""")
    assert actual == dict(rejected=False, enabled=True, calls=[['accepted', {'tabId': 1}]])


def test_toolbar_requests_open_synchronously_without_settings_or_options_writes():
    actual = panel_run(r"""
chrome.storage.local.get=()=>{throw Error('Gesture crossed storage boundary');};
const pending=new TabPanels().open(tabs.get(1));
const immediate=calls.slice();await pending;
console.log(JSON.stringify(immediate));
""")
    assert actual == [['open', {'tabId': 1}]]


def test_native_panel_disables_global_default_and_only_opens_explicit_supported_tab():
    actual = panel_run(r"""
const panels=new TabPanels();await panels.initialize();
if(await panels.open(tabs.get(2))!==false)throw Error('Unsupported click accepted');
if(await panels.open(tabs.get(1))!==true)throw Error('Supported click not opened');
console.log(JSON.stringify({calls,settings:[...saved.entries()]}));
""")
    assert actual['calls'][0] == ['options', {'enabled': False}]
    assert ['behavior', {'openPanelOnActionClick': False}] in actual['calls']
    assert [x for x in actual['calls'] if x[0] == 'open'] == [['open', {'tabId': 1}]]
    assert dict(actual['settings'])[1] == dict(tabId=1, enabled=True, path='sidepanel/index.html?tabId=1')
    assert all('windowId' not in row[1] for row in actual['calls'])


@pytest.mark.parametrize('destination', ['chrome://settings/', 'file:///private.pdf', 'about:blank'])
def test_supported_navigation_keeps_instance_outside_scope_disables_without_auto_open(destination):
    actual = panel_run(r"""
const panels=new TabPanels();await panels.initialize();const before=calls.length;
tabs.get(1).url=origin+'/portal/b';await panels.refresh(1);
const stable=calls.length===before;
tabs.get(1).pendingUrl=data.destination;await panels.refresh(1);const disabled=!saved.get(1).enabled;
delete tabs.get(1).pendingUrl;await panels.refresh(1);
allowlist=[];await panels.refreshAll();
console.log(JSON.stringify({stable,disabled,final:saved.get(1),opens:calls.filter(x=>x[0]==='open')}));
""", dict(destination=destination))
    assert actual['stable'] and actual['disabled']
    assert actual['final']['enabled'] and actual['opens'] == []


def test_queued_scope_updates_cannot_publish_stale_enable():
    actual = panel_run(r"""
const panels=new TabPanels();await panels.initialize();
let release;const gate=new Promise(resolve=>release=resolve);
const original=chrome.sidePanel.getOptions;let entered=false;
chrome.sidePanel.getOptions=async options=>{entered=true;await gate;return original(options);};
const first=panels.refresh(1);while(!entered)await Promise.resolve();
tabs.get(1).url='chrome://settings/';const second=panels.refresh(1);release();
await Promise.all([first,second]);
console.log(JSON.stringify({enabled:saved.get(1).enabled,opens:calls.filter(x=>x[0]==='open')}));
""")
    assert actual == dict(enabled=False, opens=[])


@pytest.mark.parametrize('tool,args', [('open_tab', {'url': 'http://127.0.0.1:8080/c.html'}),
    ('switch_tab', {'tab_id': 2}), ('close_tab', {'tab_id': 2})])
def test_bound_session_cannot_reassign_or_close_other_panel_target(tool, args):
    # Reuse the actual shipped BrowserSession harness; bind it as native boot does.
    import json
    actual = run("session.bindOwner(1,1);session.begin();await module.extractPageState(1,settings,true);"
        + "const result=await module.executeBrowserTool(request(" + json.dumps(tool) + ',' + json.dumps(args) + "),settings);"
        + "console.log(JSON.stringify({result,calls,owner:session.ownerTabId,target:session.target.tab_id}));")
    assert actual['result']['error'] == 'permission_denied'
    assert actual['calls'] == [] and actual['owner'] == actual['target'] == 1


def test_bound_resume_and_enumeration_never_read_another_tabs_metadata_or_dom():
    actual = run(r"""
session.bindOwner(1,1);session.begin();await module.extractPageState(1,settings,true);
const rows=await session.tabs(settings);session.runId='run';activate(2);
session.resume();const refresh=await module.executeBrowserTool(request('get_page_state',{}, {reset_target:true}),settings);
let rejected=false;try{session.adopt({...session.target,tab_id:2});}catch{rejected=true;}
let rebound=false;try{session.bindOwner(2,1);}catch{rebound=true;}
console.log(JSON.stringify({rows,refresh,rejected,rebound,target:session.target.tab_id,calls}));
""")
    assert all(row == dict(tab_id=row['tab_id'], active=False, allowed=False, details='redacted') for row in actual['rows'][1:])
    assert actual['refresh']['error'] == 'permission_denied'
    assert actual['rejected'] and actual['rebound'] and actual['target'] == 1
    assert actual['calls'] == []


def test_other_window_events_do_not_rebind_and_owner_close_revokes_leases():
    actual = run(r"""
session.bindOwner(1,1);session.begin();await module.extractPageState(1,settings,true);session.runId='run';
let disconnects=0;session.ports.add({disconnect:()=>disconnects++});
session.activated(2,2);const unchanged=session.mode==='running';
session.removed(1);
console.log(JSON.stringify({unchanged,disconnects,mode:session.mode,target:session.target,owner:session.ownerTabId}));
""")
    assert actual == dict(unchanged=True, disconnects=1, mode='stopped', target=None, owner=1)

def test_panel_prepares_unknown_tab_then_toolbar_grants_public_origin():
    result=panel_run("delete tabs.get(1).url; const p=new TabPanels();await p.initialize();const prepared=saved.get(1);tabs.get(1).url='https://any.example/docs';await p.open(tabs.get(1));console.log(JSON.stringify({prepared,opens:calls.filter(x=>x[0]==='open')}));")
    assert result['prepared']['enabled'] is True
    assert result['prepared']['path']=='sidepanel/index.html?tabId=1'
    assert result['opens']==[['open',{'tabId':1}]]
