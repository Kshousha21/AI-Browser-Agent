"""Synthetic local one-click grants, browser-owned downloads, and exact DNR boundaries."""
import pytest
from .test_product_scope import node_check
from .test_tab_panel_acquisition import run as guard_run


@pytest.mark.parametrize('case', ['fresh', 'missing_grant', 'wrong_id', 'wrong_ref', 'other_file',
    'elapsed_after_start', 'owner', 'epoch', 'clear', 'stop', 'disconnect', 'duplicate', 'wrong_action', 'model_action'])
def test_user_download_grant_is_local_one_file_one_shot(case):
    result = node_check("""
global.window={};let now=1000;Date.now=()=>now;
const {BrowserSession}=load('src/sidepanel/browser-tools.ts');const s=new BrowserSession();s.bindOwner(1,2);s.begin();
const id='a'.repeat(32),selection={resultRef:'selected',filename:'synthetic.zip',fileType:'zip'};
if(data.case!=='missing_grant')s.beginUserFileAction(id,'download',now+300000,selection);
const r={tool:'download_attachment',control_epoch:0,arguments:{user_action_id:id,user_result_ref:'selected'}};
if(data.case==='wrong_id')r.arguments.user_action_id='b'.repeat(32);
if(data.case==='wrong_ref')r.arguments.user_result_ref='other';
if(data.case==='elapsed_after_start')now+=300001;
if(data.case==='owner')s.activated(2,2);
if(data.case==='epoch')s.epoch++;
if(['clear','stop','disconnect'].includes(data.case))s.stop();
if(data.case==='wrong_action')r.tool='acquire_attachment';
if(data.case==='model_action')r.tool='navigate';
let allowed=false;try{s.guard(r)();s.userFileGrant.download(r,data.case==='other_file'?'other.zip':'synthetic.zip');
 if(data.case==='duplicate')s.userFileGrant.download(r,'synthetic.zip');allowed=true;}catch{}
console.log(JSON.stringify({allowed}));
""", dict(case=case))
    assert result == dict(allowed=case in {'fresh', 'elapsed_after_start'})


def test_legacy_pdf_identity_stays_default_and_non_pdf_needs_opt_in():
    result = node_check("""
const m=load('src/lib/attachment-preview.ts');const identity={parent_id:'100',attachment_id:'200',filename:'synthetic.zip',version:null};
let legacy=false,human=false;try{m.previewIdentity(identity);legacy=true}catch{}try{m.previewIdentity(identity,true);human=true}catch{}
console.log(JSON.stringify({legacy,human}));
""", {})
    assert result == dict(legacy=False, human=True)


@pytest.mark.parametrize('case', ['matching', 'second_preview', 'different_preview', 'preview_only'])
def test_one_human_grant_cannot_open_or_acquire_another_preview(case):
    result = node_check("""
const {UserFileGrant}=load('src/sidepanel/user-file-action.ts');
const g=new UserFileGrant('a'.repeat(32),data.case==='preview_only'?'preview':'download',Date.now()+10000,
 {resultRef:'selected',filename:'synthetic.zip',fileType:'zip'},1,0,0);
const r={tool:'open_attachment_preview',arguments:{preview_id:'b'.repeat(32),url:'synthetic-preview',attachment:{filename:'synthetic.zip'}}};
let allowed=false;try{g.preview(r);if(data.case==='second_preview')g.preview(r);
 g.download({tool:'download_attachment',arguments:{preview_id:(data.case==='different_preview'?'c':'b').repeat(32)}},'synthetic.zip');allowed=true}catch{}
console.log(JSON.stringify(allowed));
""", dict(case=case))
    assert result == (case == 'matching')


@pytest.mark.parametrize('case', ['exact', 'wrong_name', 'cross_origin', 'query', 'unbound_nonpdf'])
def test_browser_owned_non_pdf_guard_remains_exact_and_scoped(case):
    result = guard_run("""
let target=url.replace('shared.pdf','synthetic.zip')+'?version=1&download=true';
if(data.case==='cross_origin')target=target.replace('portal.example','unrelated.invalid');
if(data.case==='query')target+='&unsupported=true';
const p=port(1);p.value.send({tabId:1,url:target,...(data.case==='unbound_nonpdf'?{}:
 {userActionId:'a'.repeat(32),filename:data.case==='wrong_name'?'other.zip':'synthetic.zip'})});await drain();
const rule=[...rules.values()].find(r=>r.action.type==='allow');
const exact=!!rule&&new RegExp(rule.condition.regexFilter).test(target)&&!new RegExp(rule.condition.regexFilter).test(target+'&other=1');
const ready=p.messages.some(x=>x.ready);p.value.disconnect();await drain();
console.log(JSON.stringify({ready,exact,remaining:rules.size}));
""", dict(case=case))
    assert result == dict(ready=case in {'exact', 'query'}, exact=case in {'exact', 'query'}, remaining=0)


@pytest.mark.parametrize('case', ['complete', 'changed_href', 'redirect', 'wrong_filename'])
def test_browser_download_has_no_path_receipt_and_revalidates_before_navigation(case):
    result = node_check("""
global.window=global;const listeners=()=>{const f=new Set();return {addListener:x=>f.add(x),removeListener:x=>f.delete(x),emit:x=>[...f].forEach(fn=>fn(x))}};
const url='https://portal.example/portal/download/attachments/100/synthetic.zip?version=1&download=true';
const onCreated=listeners(),onDeterminingFilename=listeners(),onMessage=listeners(),onDisconnect=listeners();let navigations=0,closed=0,cancelled=0;
const item={id:7,url,finalUrl:data.case==='redirect'?'https://unrelated.invalid/file.zip':url,state:'complete',exists:true,danger:'safe',
 bytesReceived:12,totalBytes:12,fileSize:12,filename:'synthetic-download-directory/synthetic.zip',startTime:new Date().toISOString()};
const port={onMessage,onDisconnect,postMessage:x=>{if(x.tabId)onMessage.emit({ready:true})},disconnect(){closed++;onDisconnect.emit({})}};
global.chrome={runtime:{connect:()=>port},tabs:{update:async()=>{navigations++;onCreated.emit(item)}},downloads:{onCreated,onChanged:listeners(),onDeterminingFilename,
 search:async()=>[item],cancel:async()=>{cancelled++}}};
(async()=>{const {acquire}=load('src/sidepanel/acquisition.ts');let receipt=null,rejected=false;
try{receipt=await acquire({tool:'download_attachment',tab_id:1,arguments:{url,user_action_id:'a'.repeat(32),download_action_id:'b'.repeat(32),
 filename:data.case==='wrong_filename'?'other.zip':'synthetic.zip'}},{allowlist:['https://portal.example/portal/*']},url,()=>{},
 async()=>{if(data.case==='changed_href')throw Error('changed')},undefined,true);}catch{rejected=true}
console.log(JSON.stringify({receipt,rejected,navigations,closed,cancelled}));})().catch(()=>process.exit(1));
""", dict(case=case))
    assert result['receipt'] == (dict(download_id=7, state='complete') if case == 'complete' else None)
    assert result['rejected'] == (case != 'complete')
    assert result['navigations'] == (1 if case in ('complete', 'redirect') else 0)
    assert result['closed'] == (0 if case == 'wrong_filename' else 1)
