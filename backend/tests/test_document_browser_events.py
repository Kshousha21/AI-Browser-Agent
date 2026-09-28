"""Content-free event readiness/cancellation and bounded research-grant regressions."""
import pytest

from .test_product_scope import node_check


EVENTS = r"""
global.window=global;
const event=()=>{const handlers=new Set();return {addListener:f=>handlers.add(f),removeListener:f=>handlers.delete(f),
 emit:(...args)=>[...handlers].forEach(f=>f(...args)),size:()=>handlers.size}};
const timers={timeout:0,interval:0};global.setTimeout=()=>{timers.timeout++;throw Error('normal timer forbidden')};
global.setInterval=()=>{timers.interval++;return 1};global.clearInterval=()=>{};
const drain=async()=>{for(let n=0;n<15;n++)await Promise.resolve()};
"""


@pytest.mark.parametrize("case", ["immediate", "mutation", "load", "stop", "clear", "pagehide", "identity"])
def test_preview_wait_is_immediate_or_event_driven_and_cancellation_is_terminal(case):
    result = node_check(EVENTS + r"""
const windowEvents=new Map(),docEvents=new Map();let active=0,notify;
global.addEventListener=(name,f)=>windowEvents.set(name,f);global.removeEventListener=name=>windowEvents.delete(name);
global.document={readyState:'loading',addEventListener:(name,f)=>docEvents.set(name,f),removeEventListener:name=>docEvents.delete(name)};
global.MutationObserver=class {constructor(f){notify=f}observe(){active++}disconnect(){if(active)active--}};
(async()=>{const m=load('src/content/attachment-preview.ts'),abort=new AbortController();let ready=data.case==='immediate',reads=0,settled=false;
const p=m.waitForPreviewState(stage=>{reads++;stage('dialog');if(data.case==='identity')throw new m.PreviewFailure('preview_identity_mismatch');
 if(!ready)throw new m.PreviewFailure('preview_control_missing');return {preview:{url:'safe synthetic result'}}},true,abort.signal).then(x=>{settled=true;return x});
await drain();const immediate=settled;
if(data.case==='mutation'){ready=true;notify([])}
if(data.case==='load'){ready=true;docEvents.get('load')()}
if(['stop','clear'].includes(data.case))abort.abort();
if(data.case==='pagehide')windowEvents.get('pagehide')();
const value=await p;const beforeLate=reads;ready=true;notify([]);await drain();
console.log(JSON.stringify({immediate,error:value.error||null,ready:!!value.preview,active,events:docEvents.size+windowEvents.size,
 lateIgnored:beforeLate===reads,timers:timers.timeout}));})().catch(()=>process.exit(1));
""", dict(case=case))
    assert result == dict(immediate=case in ("immediate", "identity"),
        error="preview_identity_mismatch" if case == "identity" else
              "operation_interrupted" if case in ("stop", "clear", "pagehide") else None,
        ready=case in ("immediate", "mutation", "load"), active=0, events=0,
        lateIgnored=True, timers=0)


@pytest.mark.parametrize("case", ["immediate", "complete_event", "unrelated_event", "stop", "disconnect", "redirect", "interrupted"])
def test_download_completion_uses_events_with_exact_target_and_no_periodic_search(case):
    result = node_check(EVENTS + r"""
const url='https://portal.example/portal/download/attachments/100/synthetic.pdf?version=1&download=true';
const created=event(),changed=event(),naming=event(),messages=event(),disconnected=event(),abort=new AbortController();
let item={id:7,url,finalUrl:url,state:data.case==='immediate'?'complete':'in_progress',exists:true,danger:'safe',
 bytesReceived:12,totalBytes:12,fileSize:12,filename:'synthetic-download-directory/synthetic.pdf',startTime:new Date().toISOString()};
let searches=0,navigations=0,cancellations=0,closed=false,done=false;
const port={onMessage:messages,onDisconnect:disconnected,postMessage:x=>{if(x.tabId)messages.emit({ready:true})},
 disconnect(){if(!closed){closed=true;disconnected.emit()}}};
global.chrome={runtime:{connect:()=>port},tabs:{update:async()=>{navigations++;created.emit(item)}},
 downloads:{onCreated:created,onChanged:changed,onDeterminingFilename:naming,search:async()=>{searches++;return [{...item}]},cancel:async()=>{cancellations++}}};
(async()=>{const {acquire}=load('src/sidepanel/acquisition.ts');const p=acquire({tool:'download_attachment',tab_id:1,
 arguments:{url,user_action_id:'a'.repeat(32),download_action_id:'b'.repeat(32),filename:'synthetic.pdf'}},
 {allowlist:['https://portal.example/portal/*']},url,()=>{},async()=>{},undefined,true,abort.signal)
 .then(receipt=>{done=true;return {receipt}},()=>{done=true;return {error:true}});
await drain();const firstSearches=searches,early=done;await drain();const noPoll=searches===firstSearches;
if(data.case==='stop')abort.abort();else if(data.case==='disconnect')port.disconnect();else if(data.case!=='immediate'){
 if(data.case==='unrelated_event'){changed.emit({id:99});await drain();if(searches!==firstSearches)throw Error('unrelated event searched')}
 item={...item,state:data.case==='interrupted'?'interrupted':'complete',finalUrl:data.case==='redirect'?'https://unrelated.invalid/file.pdf':url};changed.emit({id:7,state:{current:item.state}})}
const value=await p;const finalSearches=searches;changed.emit({id:7});created.emit(item);await drain();
console.log(JSON.stringify({ok:!!value.receipt,receipt:value.receipt||null,early,noPoll,navigations,lateIgnored:finalSearches===searches,
 listeners:created.size()+changed.size()+naming.size()+messages.size()+disconnected.size(),closed,timers:timers.timeout}));})().catch(()=>process.exit(1));
""", dict(case=case))
    ok = case in ("immediate", "complete_event", "unrelated_event")
    assert result == dict(ok=ok, receipt=dict(download_id=7, state="complete") if ok else None,
        early=case == "immediate", noPoll=True, navigations=1, lateIgnored=True, listeners=0,
        closed=True, timers=0)


@pytest.mark.parametrize("case", ["immediate", "commit_event", "stop", "tab_close", "wrong_target"])
def test_document_navigation_is_commit_event_driven_not_a_load_poll(case):
    result = node_check(EVENTS + r"""
const updated=event(),removed=event(),abort=new AbortController(),url='http://127.0.0.1:8080/portal/preview';
let ready=data.case==='immediate',done=false,probes=0;
global.chrome={tabs:{onUpdated:updated,onRemoved:removed,get:async()=>({id:1,url,status:'loading'})},
 scripting:{executeScript:async()=>{probes++;return [{documentId:'owned-doc',result:ready}]}}};
(async()=>{const {waitForDocumentTab}=load('src/sidepanel/browser-tools.ts');const p=waitForDocumentTab(1,url,()=>{},abort.signal)
 .then(()=>{done=true;return true},()=>{done=true;return false});await drain();const early=done,initial=probes;await drain();const noPoll=probes===initial;
if(data.case==='commit_event'){ready=true;updated.emit(1,{status:'loading'},{id:1,url,status:'loading'})}
if(data.case==='stop')abort.abort();if(data.case==='tab_close')removed.emit(1);
if(data.case==='wrong_target')updated.emit(1,{url:'http://localhost:8080/other'},{id:1,url:'http://localhost:8080/other',status:'complete'});
const ok=await p,final=probes;updated.emit(1,{status:'complete'},{id:1,url,status:'complete'});await drain();
console.log(JSON.stringify({ok,early,noPoll,lateIgnored:final===probes,listeners:updated.size()+removed.size(),timers:timers.timeout}));})().catch(()=>process.exit(1));
""", dict(case=case))
    assert result == dict(ok=case in ("immediate", "commit_event"), early=case == "immediate",
        noPoll=True, lateIgnored=True, listeners=0, timers=0)


@pytest.mark.parametrize("case", ["ready", "rejected", "disconnect", "stop"])
def test_document_observer_lease_wait_uses_lifecycle_and_removes_listeners(case):
    result = node_check(EVENTS + r"""
const messages=event(),disconnected=event(),abort=new AbortController();
(async()=>{const {waitForLease}=load('src/sidepanel/browser-tools.ts');const p=waitForLease({onMessage:messages,onDisconnect:disconnected},abort.signal)
 .then(()=>true,()=>false);if(data.case==='stop')abort.abort();else if(data.case==='disconnect')disconnected.emit();
else messages.emit({ready:data.case==='ready'});const ok=await p;
console.log(JSON.stringify({ok,listeners:messages.size()+disconnected.size(),timers:timers.timeout}));})().catch(()=>process.exit(1));
""", dict(case=case))
    assert result == dict(ok=case == "ready", listeners=0, timers=0)


@pytest.mark.parametrize("stage", ["target", "regex", "rules", "PRIVATE_DIAGNOSTIC"])
def test_guard_denial_preserves_only_allowlisted_stage(stage):
    result = node_check(EVENTS + r"""
const messages=event(),disconnected=event(),abort=new AbortController();
(async()=>{const {waitForLease}=load('src/sidepanel/browser-tools.ts');
const pending=waitForLease({onMessage:messages,onDisconnect:disconnected},abort.signal)
 .then(()=>({unexpected:true}),e=>({code:e.code,stage:e.stage}));
messages.emit({ready:false,stage:data.stage});console.log(JSON.stringify(await pending));})();
""", dict(stage=stage))
    assert result == dict(code="operation_interrupted", stage="guard_" + stage if stage != "PRIVATE_DIAGNOSTIC" else "observer_lease")


@pytest.mark.parametrize("case", ["fresh", "over_default", "over_hard", "wrong_source", "wrong_name", "wrong_ref", "duplicate", "elapsed", "stop", "owner", "model_return", "return_target", "duplicate_after_return", "fresh_second_file"])
def test_research_authority_is_bounded_exact_and_state_bound(case):
    result = node_check(EVENTS + r"""
let now=1000;Date.now=()=>now;const {BrowserSession}=load('src/sidepanel/browser-tools.ts'),s=new BrowserSession();
s.bindOwner(1,2);s.begin();const id='a'.repeat(32),url='https://portal.example/portal/search';s.target={url,tab_id:1};
const file={sourceKey:'b'.repeat(64),resultRef:'current',filename:'synthetic.pdf',fileType:'pdf'};
let allowed=false;try{s.beginResearch(id,data.case==='over_hard'?6:3);
const second={...file,sourceKey:'c'.repeat(64),resultRef:'second',filename:'second.pdf'};
const files=data.case==='over_default'?Array.from({length:4},(_,i)=>({...file,sourceKey:String(i).repeat(64),resultRef:String(i)})):
 data.case==='fresh_second_file'?[file,second]:[file];
const legacyDeadline=now+300000;if(data.case==='elapsed')now+=86400000;
s.approveResearchPlan(id,data.case==='wrong_source'?url+'/other':url,files,legacyDeadline);
if(data.case==='stop')s.stop();if(data.case==='owner')s.activated(2,2);
if(data.case==='model_return')s.guard({tool:'return_research_results',control_epoch:0,arguments:{research_id:'model',url}})();
const selection={...file,filename:data.case==='wrong_name'?'wrong.pdf':file.filename,resultRef:data.case==='wrong_ref'?'wrong':file.resultRef};
s.activateResearchFile(id,file.sourceKey,selection);
if(data.case==='duplicate')s.activateResearchFile(id,file.sourceKey,selection);
if(data.case==='return_target')s.researchGrant.returnTarget(url+'/other');
if(['duplicate_after_return','fresh_second_file'].includes(data.case)){
 s.researchGrant.returnTarget(url);s.researchGrant.returnedToResults();
 const next=data.case==='fresh_second_file'?second:selection;
 s.activateResearchFile(id,next.sourceKey,{...next,resultRef:'fresh'});
}
allowed=true}catch{}
console.log(JSON.stringify({allowed}));
""", dict(case=case))
    assert result == dict(allowed=case in ("fresh", "elapsed", "fresh_second_file"))


@pytest.mark.parametrize("case", ["already_source", "different_source", "wrong_owner", "pending_same", "pending_other", "owner_during_observe"])
def test_research_return_observes_existing_source_without_self_invalidating_reload(case):
    result = node_check(EVENTS + r"""
const source='https://portal.example/portal/dosearchsite.action?queryString=synthetic';
const preview='https://portal.example/portal/spaces/DEMO/pages/100/Synthetic?preview=%2F100%2F200%2Fsynthetic.pdf';
const updated=event(),removed=event();let navs=0,extracts=0,probes=0,committed=false;
let tab={id:1,windowId:2,url:data.case==='different_source'?preview:source,active:true,status:'complete'};
if(data.case==='pending_same')tab.pendingUrl=source;if(data.case==='pending_other')tab.pendingUrl=source+'&changed=1';
const page={url:tab.url,tab_id:1,document_id:'initial-document',snapshot_id:'initial-snapshot',navigation_id:1};
let s;
global.chrome={tabs:{get:async()=>({...tab}),query:async()=>[{...tab}],onUpdated:updated,onRemoved:removed,
 update:async(_id,details)=>{navs++;tab={...tab,url:details.url,status:'loading'};return {...tab}}},
 scripting:{executeScript:async options=>{const body=options.func.toString();
   if(body.includes('document.documentElement')){probes++;return [{documentId:'returned-chrome-document',result:committed}]}
   if(body.includes('loaded:')||body.includes('loaded: '))return [{documentId:'owned-chrome-document',result:{same:true,loaded:true}}];
   if(body.includes('extractPageState')){extracts++;if(data.case==='owner_during_observe')s.activated(2,2);
     return [{documentId:'owned-chrome-document',result:{...page,url:source,snapshot_id:'fresh-return-snapshot',
       document_id:data.case==='different_source'?'returned-document':page.document_id}}]}
   throw Error('Unexpected synthetic script');}}};
(async()=>{const m=load('src/sidepanel/browser-tools.ts');s=m.browserSession;s.bindOwner(1,2);s.begin();
s.adopt({...page,url:source});const id='a'.repeat(32),file={sourceKey:'b'.repeat(64),resultRef:'current',filename:'synthetic.pdf',fileType:'pdf'};
s.beginResearch(id);s.approveResearchPlan(id,source,[file],Date.now()+10000);s.activateResearchFile(id,file.sourceKey,file);s.adopt(page);
const call={run_id:'synthetic-run',tool:'return_research_results',arguments:{research_id:id,url:source},
 tab_id:data.case==='wrong_owner'?2:1,document_id:page.document_id,snapshot_id:page.snapshot_id,control_epoch:0};
let finished=false;const pending=m.executeBrowserTool(call,{allowlist:['https://portal.example/portal/*']}).then(result=>{finished=true;return result});
await drain();await drain();const early=finished;
if(data.case==='different_source'){committed=true;tab.status='complete';updated.emit(1,{status:'complete'},{...tab})}
const result=await pending;let nextAllowed=true;try{s.guard({tool:'get_page_state',control_epoch:0,arguments:{}})()}catch{nextAllowed=false}
console.log(JSON.stringify({success:result.success,navs,extracts,early,nextAllowed,fresh:result.data?.page_state?.snapshot_id==='fresh-return-snapshot',
 sameDocument:result.data?.page_state?.document_id===page.document_id}));})().catch(()=>process.exit(1));
""", dict(case=case))
    success = case in ("already_source", "different_source")
    assert result['success'] is success
    assert result['navs'] == int(case == 'different_source')
    assert result['extracts'] == int(success or case == 'owner_during_observe')
    assert result['fresh'] is success
    assert result['sameDocument'] is (case == 'already_source')
    if success:
        assert result['nextAllowed'] is True
        assert result['early'] is (case == 'already_source')
