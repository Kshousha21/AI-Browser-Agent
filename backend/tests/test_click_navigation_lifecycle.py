"""Exercise delayed full-document link navigation through the actual TS bridge."""
import pytest
from .test_browser_control import run


SCRIPT = r"""
const scenario=SCENARIO;
session.bindOwner(1,1);session.begin();session.runId='run';
await module.extractPageState(1,settings,true);
const marketing='http://127.0.0.1:8080/marketing.html';
const templates='http://127.0.0.1:8080/templates.html';
function element(href,name){return {agent_id:'f0:link',tag:'a',role:'link',name,href,visible:true,enabled:true,occluded:false};}
session.target.elements=[element(marketing,'Marketing & PR')];
const original=chrome.scripting.executeScript;
let clicks=0, snapshots=0, invalidations=0;
session.onInvalidation=()=>invalidations++;
chrome.tabs.connect=()=>({onMessage:{addListener:fn=>queueMicrotask(()=>fn({ready:true}))},
 onDisconnect:{addListener:()=>{}},disconnect:()=>{}});
chrome.scripting.executeScript=async config=>{
 const operation=config.args?.[1];
 if(operation?.tool==='click'){
   clicks++;
   const destination=clicks===1?marketing:templates;
   if(scenario==='denied')return [{result:{success:false,action:'click',error:'permission_denied',page_changed:false,data:{}}}];
   if(scenario!=='timeout')setTimeout(()=>{
     if(scenario==='pause')return;
     const url=scenario==='redirect'?'http://127.0.0.1:8080/other.html':destination;
     Object.assign(tabs.get(1),{url,status:'loading'});session.updated(1,{url,status:'loading'});
     setTimeout(()=>{tabs.get(1).status='complete';session.updated(1,{status:'complete'});},40);
   },300);
   if(scenario==='pause')setTimeout(()=>session.pause(),30);
   if(scenario==='lost')return [];
   if(scenario==='destroyed')throw Error('Execution context destroyed');
   return [{result:{success:true,action:'click',snapshot_id:operation.snapshot_id,page_changed:false,data:{}}}];
 }
 const results=await original(config);
 for(const entry of results){
   if(entry.result?.snapshot_id){
     snapshots++;
     entry.result.document_id='doc:'+entry.result.url;
     entry.result.elements=entry.result.url===marketing?[element(templates,'Example Templates')]:[];
   }
 }
 return results;
};
if(scenario==='timeout'){
 let now=0;Object.defineProperty(global,'performance',{value:{now:()=>now}});
 const get=chrome.tabs.get;chrome.tabs.get=async id=>{now+=5000;return get(id);};
}
const before=session.target.snapshot_id;
const first=await module.executeBrowserTool(request('click',{element_id:'f0:link'}),settings);
let second=null,oldSnapshotRetained=session.snapshots.has(before);
if(first.success){
 second=await module.executeBrowserTool(request('click',{element_id:'f0:link'}),settings);
}
const mode=session.mode;
if(second?.success)session.updated(1,{url:'http://127.0.0.1:8080/manual.html'});
console.log(JSON.stringify({first,second,clicks,snapshots,invalidations,mode,
 afterManual:session.mode,oldSnapshotRetained,linkReleased:session.linkNavigation===null}));
"""


@pytest.mark.parametrize("scenario", ["normal", "lost", "destroyed"])
def test_delayed_link_load_returns_fresh_destination_and_continues_to_templates(scenario):
    result = run(SCRIPT.replace("SCENARIO", repr(scenario)))
    assert result["first"]["success"] and result["first"]["page_changed"]
    assert result["first"]["data"]["page_state"]["url"].endswith("/marketing.html")
    assert result["second"]["success"]
    assert result["second"]["data"]["page_state"]["url"].endswith("/templates.html")
    assert result["clicks"] == result["snapshots"] == 2
    assert result["mode"] == "running" and result["afterManual"] == "paused"
    assert result["invalidations"] == 1  # only the later manual navigation
    assert not result["oldSnapshotRetained"] and result["linkReleased"]


@pytest.mark.parametrize("scenario", ["pause", "redirect"])
def test_pause_or_unexpected_destination_still_interrupts_click_navigation(scenario):
    result = run(SCRIPT.replace("SCENARIO", repr(scenario)))
    assert not result["first"]["success"]
    assert result["first"]["error"] == "run_paused"
    assert result["second"] is None and result["clicks"] == 1
    assert result["snapshots"] == 0 and result["mode"] == "paused"
    assert result["linkReleased"]


@pytest.mark.parametrize("scenario,error", [("timeout", "timeout"), ("denied", "permission_denied")])
def test_link_that_does_not_navigate_or_is_denied_never_reports_success(scenario, error):
    result = run(SCRIPT.replace("SCENARIO", repr(scenario)))
    assert not result["first"]["success"] and result["first"]["error"] == error
    assert result["second"] is None and result["clicks"] == 1
    assert result["snapshots"] == 0 and result["linkReleased"]
