"""Clock-independent card admission and lifecycle-owned browser document authority."""
import pytest

from .panel_clear_support import run
from .test_product_scope import node_check
from .test_research_panel import SETUP


@pytest.mark.parametrize('stage', ['preview', 'download', 'parsing', 'indexing', 'rendering'])
def test_started_file_operation_survives_elapsed_time_through_each_stage(stage):
    result = node_check(r"""
global.window={};let now=1000;Date.now=()=>now;
const {BrowserSession}=load('src/sidepanel/browser-tools.ts'),s=new BrowserSession();s.bindOwner(1,2);s.begin();
const selected={resultRef:'selected',filename:'synthetic.pdf',fileType:'pdf'},id='a'.repeat(32),preview='b'.repeat(32);
s.beginUserFileAction(id,'prepare',now+300000,selected);
const args={user_action_id:id,user_result_ref:'selected'},request={tool:'get_page_state',control_epoch:0,arguments:args};
const grant=s.userFileGrant;
if(data.stage!=='preview')grant.preview({tool:'open_attachment_preview',arguments:{...args,preview_id:preview,url:'observed-preview',attachment:{filename:'synthetic.pdf'}}});
now+=900001;
s.guard(request)();
if(data.stage==='preview')grant.preview({tool:'open_attachment_preview',arguments:{...args,preview_id:preview,url:'observed-preview',attachment:{filename:'synthetic.pdf'}}});
grant.inspect({tool:'inspect_attachment_preview',arguments:{...args,preview_id:preview}});
grant.download({tool:'acquire_attachment',arguments:{...args,preview_id:preview}},'synthetic.pdf');
s.guard(request)();console.log(JSON.stringify({valid:true,immutable:Object.isFrozen(grant.selection),noTtl:!('expiresAt' in grant)}));
""", dict(stage=stage))
    assert result == dict(valid=True, immutable=True, noTtl=True)


@pytest.mark.parametrize('case', ['forged_admission', 'missing_grant', 'wrong_nonce', 'wrong_ref',
                                 'wrong_file', 'clear', 'stop', 'disconnect', 'owner', 'navigation', 'epoch', 'version'])
def test_forged_or_invalidated_operations_remain_blocked(case):
    result = node_check(r"""
global.window={};let now=1000;Date.now=()=>now;const module=load('src/sidepanel/user-file-action.ts');
const {BrowserSession}=load('src/sidepanel/browser-tools.ts'),s=new BrowserSession();s.bindOwner(1,2);s.begin();
const selection={resultRef:'selected',filename:'synthetic.pdf',fileType:'pdf'},id='a'.repeat(32);
let allowed=false;try{
if(data.case==='forged_admission')new module.UserFileGrant(id,'prepare',now-1,selection,1,s.version,0,'research-operation-admission');
if(data.case!=='missing_grant')s.beginUserFileAction(id,'prepare',now+300000,selection);
now+=900001;
if(['clear','stop','disconnect'].includes(data.case))s.stop();
if(data.case==='owner')s.activated(2,2);
if(data.case==='navigation')s.invalidate('target_document_changed');
if(data.case==='epoch')s.epoch++;if(data.case==='version')s.version++;
const request={tool:'acquire_attachment',control_epoch:0,arguments:{user_action_id:data.case==='wrong_nonce'?'other':id,
 user_result_ref:data.case==='wrong_ref'?'other':'selected'}};
s.guard(request)();s.userFileGrant.download(request,data.case==='wrong_file'?'other.pdf':'synthetic.pdf');allowed=true;}catch{}
console.log(JSON.stringify({allowed,privateAdmission:!Object.keys(module).some(key=>key.includes('ADMISSION'))}));
""", dict(case=case))
    assert result == dict(allowed=False, privateAdmission=True)


@pytest.mark.parametrize('legacy_deadline', [0, 999, 301001])
def test_current_file_operation_admission_ignores_legacy_elapsed_deadlines(legacy_deadline):
    result = node_check(r"""
global.window={};let now=1000;Date.now=()=>now;
const {BrowserSession}=load('src/sidepanel/browser-tools.ts'),s=new BrowserSession();s.bindOwner(1,2);s.begin();
const selection={resultRef:'selected',filename:'synthetic.pdf',fileType:'pdf'},id='a'.repeat(32);
now+=86400000;s.beginUserFileAction(id,'prepare',data.legacy_deadline,selection);
const request={tool:'acquire_attachment',control_epoch:0,arguments:{user_action_id:id,user_result_ref:'selected'}};
s.guard(request)();s.userFileGrant.download(request,'synthetic.pdf');
let duplicateBlocked=false;try{s.userFileGrant.download(request,'synthetic.pdf')}catch{duplicateBlocked=true}
console.log(JSON.stringify({valid:true,duplicateBlocked,noTtl:!('expiresAt' in s.userFileGrant)}));
""", dict(legacy_deadline=legacy_deadline))
    assert result == dict(valid=True, duplicateBlocked=True, noTtl=True)


def test_file_authority_is_an_immutable_copy_not_mutable_result_metadata():
    result = node_check(r"""
const {UserFileGrant}=load('src/sidepanel/user-file-action.ts');const selection={resultRef:'original',filename:'original.pdf',fileType:'pdf'};
const grant=new UserFileGrant('a'.repeat(32),'prepare',Date.now()+300000,selection,1,0,0);
selection.resultRef='other';selection.filename='other.pdf';selection.fileType='zip';
let mutationBlocked=false;try{Object.assign(grant.selection,{filename:'other.pdf'})}catch{mutationBlocked=true}
grant.check({tool:'acquire_attachment',arguments:{user_action_id:grant.id,user_result_ref:'original'}},1,0,0);
console.log(JSON.stringify({selection:grant.selection,mutationBlocked}));
""", {})
    assert result == dict(selection=dict(resultRef='original', filename='original.pdf', fileType='pdf'), mutationBlocked=True)


def test_entire_fresh_research_cohort_is_admitted_once_and_survives_long_operations():
    result = node_check(r"""
global.window={};let now=1000;Date.now=()=>now;const {BrowserSession}=load('src/sidepanel/browser-tools.ts'),s=new BrowserSession();
s.bindOwner(1,2);s.begin();const id='a'.repeat(32),url='https://portal.example/portal/search';s.target={url,tab_id:1};
const files=Array.from({length:3},(_,i)=>({sourceKey:String(i).repeat(64),resultRef:'ref-'+i,filename:'synthetic-'+i+'.pdf',fileType:'pdf'}));
s.beginResearch(id);s.approveResearchPlan(id,url,files,now+300000);
const admitted=files.map(file=>({...file}));files[1].filename='mutated.pdf';let completed=0;
for(const [index,file] of admitted.entries()){
 now+=900001;
 if(index){s.researchGrant.returnTarget(url);s.researchGrant.returnedToResults();}
 const selected={...file,resultRef:index?'new-observation-'+index:file.resultRef};
 s.activateResearchFile(id,file.sourceKey,selected);
 const request={tool:'acquire_attachment',control_epoch:0,arguments:{user_action_id:id,user_result_ref:selected.resultRef}};
 s.guard(request)();s.userFileGrant.download(request,file.filename);completed++;
}
let duplicate=false;try{s.researchGrant.returnTarget(url);s.researchGrant.returnedToResults();s.activateResearchFile(id,admitted[0].sourceKey,admitted[0])}catch{duplicate=true}
console.log(JSON.stringify({completed,duplicate,elapsed:now-1000}));
""", {})
    assert result == dict(completed=3, duplicate=True, elapsed=2700003)


@pytest.mark.parametrize('case', ['elapsed_before_plan', 'long_after_plan', 'mismatched_after_plan', 'cancelled_after_plan'])
def test_actual_panel_research_plan_is_state_bound_not_time_bound(case):
    result = run(SETUP + r"""
submit('Research synthetic packaging');await drain();const id=ws.sent.find(f=>f.type==='document_research').id;
ws.frame('document_choices',{...offer,research_id:id});
if(data.case==='elapsed_before_plan')advance(86400001);
ws.frame('research_plan',{research_id:id,source_url:owner.url,files:[{source_key:key,result_ref:ref,filename:'Synthetic.pdf',file_type:'pdf'}]});
if(data.case!=='elapsed_before_plan')advance(900001);
if(data.case==='cancelled_after_plan')el('stop').click();
ws.frame('research_file',{research_id:id,source_key:key,result_ref:ref,
 filename:data.case==='mismatched_after_plan'?'Other.pdf':'Synthetic.pdf',file_type:'pdf',expires_at_ms:1});
await drain();let allowed=false;try{panelSession.guard({tool:'get_page_state',control_epoch:0,
 arguments:{user_action_id:id,user_result_ref:ref}})();allowed=!!panelSession.userFileGrant;}catch{}
console.log(JSON.stringify({allowed,scripts:calls.filter(c=>c==='script').length,provider:ws.sent.filter(f=>f.type==='user_message').length}));el('clear').click();
""", case=case)
    assert result == dict(allowed=case in ('elapsed_before_plan', 'long_after_plan'), scripts=0, provider=0)


@pytest.mark.parametrize('event', ['none', 'stop', 'clear', 'disconnect', 'owner'])
def test_actual_panel_started_prepare_ignores_time_but_not_cancellation(event):
    result = run(SETUP + r"""
ws.frame('document_choices',offer);button('Prepare locally').click();await drain();
const request=ws.sent.find(f=>f.type==='document_file_action');advance(900001);
if(data.event==='stop')el('stop').click();if(data.event==='clear')el('clear').click();if(data.event==='disconnect')ws.close();
if(data.event==='owner'){owner.active=false;events.activated[0]({tabId:2,windowId:1});}
let valid=false;try{panelSession.guard({tool:'get_page_state',control_epoch:0,
 arguments:{user_action_id:request.id,user_result_ref:ref}})();valid=!!panelSession.userFileGrant;}catch{}
ws.frame('document_done',{user_action_id:request.id,status:'ready',file:{action:'prepare',title:'Synthetic.pdf',file_type:'pdf',document_id:'f'.repeat(64)}},true);
await drain();console.log(JSON.stringify({valid,late:data.event==='none'?false:!!button('Ask about this PDF'),scripts:calls.filter(c=>c==='script').length}));el('clear').click();
""", event=event)
    assert result == dict(valid=event == 'none', late=False, scripts=0)


@pytest.mark.parametrize('case', ['delayed_plan', 'wrong_file', 'wrong_source', 'stop', 'elapsed_before_submit'])
def test_specific_pdf_ask_admits_the_exact_current_file_before_delayed_plan(case):
    result = run(SETUP + r"""
ws.frame('document_choices',offer);button('Ask about this PDF').click();
if(data.case==='elapsed_before_submit')advance(86400001);
submit('Explain the selected report.');await drain();
const request=ws.sent.find(f=>f.type==='document_research');
let valid=false;
if(request){
 const id=request.id;advance(900001);if(data.case==='stop')el('stop').click();
 const file={source_key:key,result_ref:ref,filename:data.case==='wrong_file'?'Other.pdf':'Synthetic.pdf',file_type:'pdf'};
 ws.frame('research_plan',{research_id:id,source_url:data.case==='wrong_source'?owner.url+'?other=1':owner.url,files:[file]});
 ws.frame('research_file',{research_id:id,...file,expires_at_ms:1});await drain();
 try{panelSession.guard({tool:'get_page_state',control_epoch:0,arguments:{user_action_id:id,user_result_ref:ref}})();
  valid=!!panelSession.userFileGrant}catch{}
}
console.log(JSON.stringify({valid,requests:ws.sent.filter(f=>f.type==='document_research').length,
 scripts:calls.filter(c=>c==='script').length}));el('clear').click();
""", case=case)
    assert result == dict(valid=case in ('delayed_plan', 'elapsed_before_submit'), requests=1, scripts=0)
