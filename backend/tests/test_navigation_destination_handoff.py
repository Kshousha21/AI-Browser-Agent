"""Reproduce direct navigation and exact observed link handoff races in the TS bridge."""
import pytest
from .test_browser_control import run


SCRIPT = r"""
const scenario=SCENARIO;
const origin='https://example.org';
const source=origin+'/home';
const requested=origin+'/guide';
const destination=origin+(scenario==='wrong_page'?'/other':'/guide');
tabs.get(1).url=source;
session.bindOwner(1,1);session.begin();session.runId='run';
await module.extractPageState(1,settings,true);
session.target.elements=[{agent_id:'f0:link',tag:'a',role:'link',name:'Marketing & PR',href:requested,visible:true,enabled:true}];
let navigations=0,invalidations=0;
session.onInvalidation=()=>invalidations++;
function loadDestination(){
 navigations++;
 setTimeout(()=>{
   Object.assign(tabs.get(1),{url:destination,status:'loading'});
   session.updated(1,{url:destination,status:'loading'});
   setTimeout(()=>{tabs.get(1).status='complete';session.updated(1,{status:'complete'});},40);
 },300);
}
chrome.tabs.update=async()=>{loadDestination();return {...tabs.get(1)};};
chrome.tabs.connect=()=>({onMessage:{addListener:fn=>queueMicrotask(()=>fn({ready:true}))},onDisconnect:{addListener:()=>{}},disconnect:()=>{}});
const original=chrome.scripting.executeScript;
chrome.scripting.executeScript=async config=>{
 if(config.args?.[1]?.tool==='click'){
  loadDestination();return [{result:{success:true,action:'click',snapshot_id:session.target.snapshot_id,page_changed:false,data:{}}}];
 }
 const entries=await original(config);
 for(const entry of entries)if(entry.result?.snapshot_id)entry.result.document_id='doc:'+entry.result.url;
 return entries;
};
const tool=scenario==='click_alias'?'click':'navigate';
const response=await module.executeBrowserTool(request(tool,tool==='click'?{element_id:'f0:link'}:{url:requested}),settings);
await new Promise(resolve=>setTimeout(resolve,400));
console.log(JSON.stringify({response,mode:session.mode,invalidations,navigations,target:session.target.url}));
"""


@pytest.mark.parametrize("scenario", ["navigate_alias", "navigate_exact", "click_alias"])
def test_delayed_generic_destination_arrives_before_operation_finishes(scenario):
    result = run(SCRIPT.replace("SCENARIO", repr(scenario)))
    assert result["mode"] == "running"
    assert result["invalidations"] == 0
    assert result["response"]["success"] and result["response"]["page_changed"]
    assert result["response"]["data"]["page_state"]["url"].endswith('/guide')
    assert result["target"].endswith('/guide')
    assert result["navigations"] == 1


def test_redirect_to_different_page_id_interrupts():
    result = run(SCRIPT.replace("SCENARIO", repr("wrong_page")))
    assert result["mode"] == "paused" and result["invalidations"] == 1
    assert not result["response"]["success"]


@pytest.mark.parametrize("actual,expected,match", [
    ("/guide", "/guide", True), ("/guide?id=1", "/guide?id=2", False),
    ("/old", "/new", False), ("https://other.example/guide", "/guide", False),
    ("/guide#part", "/guide", False)])
def test_only_exact_generic_destinations_match(actual, expected, match):
    from .test_product_scope import node_check
    origin = "https://example.org"
    result = node_check("const m=load('src/lib/navigation-target.ts');console.log(JSON.stringify(m.sameNavigationDestination(data.actual,data.expected)));",
        {"actual": actual if actual.startswith("https:") else origin + actual, "expected": origin + expected})
    assert result is match


def test_paused_panel_no_longer_claims_to_be_working():
    from .panel_clear_support import run as panel_run
    result = panel_run(r"""
const ws=sockets[0];ws.frame('document_choices',choices);prepareChoice();await drain();
ws.frame('control_state',{run_id:'test-run',state:'paused',epoch:1,reason:'Target document changed'});await drain();
console.log(JSON.stringify({label:el('send').textContent,resumeDisabled:el('resume').disabled}));el('clear').click();
""")
    assert result == {"label": "Paused", "resumeDisabled": False}
