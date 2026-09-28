"""Exercise the shipped TypeScript tab boundary with a deterministic Chrome API double."""
from .test_product_scope import node_check

SETUP = r"""
let module;
const calls = [];
let serial = 0;
const tabs = new Map([
 [1,{id:1,windowId:1,active:true,status:'complete',title:'Fixture A',url:'http://127.0.0.1:8080/a.html'}],
 [2,{id:2,windowId:1,active:false,status:'complete',title:'Fixture B',url:'http://127.0.0.1:8080/b.html'}],
 [3,{id:3,windowId:1,active:false,status:'complete',title:'UNRELATED_TITLE',url:undefined}],
 [4,{id:4,windowId:1,active:false,status:'complete',title:'PORTAL_TITLE',url:undefined}]
]);
function activate(id) {for(const tab of tabs.values()) tab.active=tab.id===id; module?.browserSession.activated(id);}
global.chrome={tabs:{
 query:async q=>[...tabs.values()].filter(t=>(!q.active||t.active)&&(q.windowId===undefined||t.windowId===q.windowId)),
 get:async id=>{if(!tabs.has(id))throw Error('closed');return {...tabs.get(id)};},
 create:async ({url})=>{const id=10+serial++;const tab={id,windowId:1,active:true,status:'complete',title:'Opened fixture',url};tabs.set(id,tab);activate(id);calls.push(['create',id]);return {...tab};},
 update:async(id,changes)=>{Object.assign(tabs.get(id),changes);if(changes.active)activate(id);calls.push(['update',id]);return {...tabs.get(id)};},
 remove:async id=>{tabs.delete(id);calls.push(['remove',id]);}
},windows:{update:async()=>{}},scripting:{executeScript:async config=>{
 const tab=tabs.get(config.target.tabId);if(!tab)throw Error('closed');
 if(config.files)return [{documentId:'chrome-'+tab.id}];
 global.location={href:tab.url};
 global.__aiBrowserAgentContent={bindTab:()=>{},invalidate:()=>{},extractPageState:()=>({tab_id:tab.id,document_id:'doc-'+tab.id,navigation_id:0,snapshot_id:'snap-'+(++serial),url:tab.url,title:tab.title,visible_text:'Fixture body',elements:[],headings:[],frames:[],truncated:{text:false,elements:false},opaque_regions:[],warnings:[],dialogs:[],looks_like_login:false,semantic_regions:[]})};
 return [{documentId:'chrome-'+tab.id,result:await config.func(...config.args)}];
}}};
module=load('src/sidepanel/browser-tools.ts');
const session=module.browserSession;
const settings={allowlist:['http://127.0.0.1:8080/*','https://portal.example/portal/*'],maxVisibleTextChars:20000};
function request(tool,arguments_={},extra={}) {return {run_id:'run',step:1,max_steps:30,tool,arguments:arguments_,snapshot_id:session.target?.snapshot_id??'initial',tab_id:session.target?.tab_id??1,document_id:session.target?.document_id??'',control_epoch:session.epoch,reason:'Fixture verification',...extra};}
"""


def run(script):
    return node_check(SETUP + "\n(async()=>{\n" + script + "\n})().catch(e=>{console.error(e);process.exit(1);});", {})


def test_enumeration_redacts_unrelated_tabs_and_tracks_agent_opened_tabs():
    result = run(r"""
session.begin();await module.extractPageState(1,settings,true);
const first=await module.executeBrowserTool(request('get_tabs'),settings);
const opened=await module.executeBrowserTool(request('open_tab',{url:'http://127.0.0.1:8080/c.html'}),settings);
const second=await module.executeBrowserTool(request('get_tabs'),settings);
console.log(JSON.stringify({first,opened,second,calls}));
""")
    rows = result["first"]["data"]["tabs"]
    assert rows[0]["tab_id"] == 1 and rows[0]["active"]
    assert rows[2] == {"tab_id": 3, "active": False, "allowed": False, "details": "redacted"}
    assert "UNRELATED_TITLE" not in str(result) and "private.example" not in str(result)
    assert "PORTAL_TITLE" not in str(result)
    assert result["opened"]["success"]
    assert result["opened"]["data"]["page_state"]["tab_id"] != 1
    assert result["second"]["data"]["tabs"][0]["opened_by_agent"]


def test_switch_reobserves_target_and_old_tab_snapshot_cannot_execute():
    result = run(r"""
session.begin();const a=await module.extractPageState(1,settings,true);
const switched=await module.executeBrowserTool(request('switch_tab',{tab_id:2}),settings);
const old=await module.executeBrowserTool(request('click',{element_id:'f0:e-001'},{tab_id:1,document_id:a.document_id,snapshot_id:a.snapshot_id}),settings);
const back=await module.executeBrowserTool(request('switch_tab',{tab_id:1}),settings);
const stale=await module.executeBrowserTool(request('click',{element_id:'f0:e-001'},{snapshot_id:a.snapshot_id}),settings);
console.log(JSON.stringify({switched,old,back,stale,calls}));
""")
    assert result["switched"]["data"]["page_state"]["tab_id"] == 2
    assert result["old"]["error"] == "target_tab_changed"
    assert result["back"]["data"]["page_state"]["tab_id"] == 1
    assert result["stale"]["error"] == "stale_snapshot"


def test_tab_tools_cannot_expand_product_scope():
    result = run(r"""
session.begin();await module.extractPageState(1,settings,true);
const portal=await module.executeBrowserTool(request('open_tab',{url:'chrome://settings'}),settings);
const external=await module.executeBrowserTool(request('open_tab',{url:'file:///private.pdf'}),settings);
const switched=await module.executeBrowserTool(request('switch_tab',{tab_id:3}),settings);
console.log(JSON.stringify({portal,external,switched,calls}));
""")
    assert result["portal"]["error"] == result["external"]["error"] == "navigation_blocked"
    assert result["switched"]["error"] == "permission_denied"
    assert result["calls"] == []


def test_close_checks_revision_exact_id_and_last_tab_window_guard():
    result = run(r"""
session.begin();await module.extractPageState(1,settings,true);
const rows=await session.tabs(settings);const revision=rows.find(t=>t.tab_id===2).revision;
const absent=await module.executeBrowserTool(request('close_tab',{tab_id:2}),settings);
tabs.get(2).title='Changed target';
const stale=await module.executeBrowserTool(request('close_tab',{tab_id:2},{tab_revision:revision}),settings);
const fresh=(await session.tabs(settings)).find(t=>t.tab_id===2).revision;
const closed=await module.executeBrowserTool(request('close_tab',{tab_id:2},{tab_revision:fresh}),settings);
tabs.delete(3);tabs.delete(4);
const only=(await session.tabs(settings))[0].revision;
const last=await module.executeBrowserTool(request('close_tab',{tab_id:1},{tab_revision:only}),settings);
console.log(JSON.stringify({absent,stale,closed,last,calls,remaining:[...tabs.keys()]}));
""")
    assert result["absent"]["error"] == result["stale"]["error"] == "stale_snapshot"
    assert result["closed"]["success"] and result["closed"]["data"]["closed_tab_id"] == 2
    assert result["last"]["error"] == "permission_denied"
    assert result["remaining"] == [1]
    assert [call for call in result["calls"] if call[0] == "remove"] == [["remove", 2]]


def test_pause_closes_leases_and_rejects_queued_or_old_epoch_requests():
    result = run(r"""
session.begin();await module.extractPageState(1,settings,true);
const queued=request('open_tab',{url:'http://127.0.0.1:8080/c.html'});
let disconnects=0;session.ports.add({disconnect:()=>disconnects++});session.pause();
const paused=await module.executeBrowserTool(queued,settings);
session.control({run_id:'run',epoch:1,state:'paused',reason:'Paused'});session.resume();
const fresh=await module.executeBrowserTool(request('get_page_state',{}, {reset_target:true}),settings);
session.control({run_id:'run',epoch:1,state:'running',reason:'Ready'});
const old=await module.executeBrowserTool(queued,settings);
console.log(JSON.stringify({paused,fresh,old,disconnects,calls}));
""")
    assert result["paused"]["error"] == result["old"]["error"] == "run_paused"
    assert result["fresh"]["success"] and result["fresh"]["page_changed"]
    assert result["disconnects"] == 1 and result["calls"] == []
