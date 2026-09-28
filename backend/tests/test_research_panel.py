"""Actual bundled research UI admission, scoped grants and late-frame retirement."""
import pytest
from .panel_clear_support import run
from .test_product_scope import node_check

SETUP = r"""
const ws=sockets[0],ref='result-'+'b'.repeat(24),key='c'.repeat(64);
const page={snapshot_id:'s-current',tab_id:1,document_id:'doc-current',navigation_id:'nav-current',url:owner.url,elements:[]};
panelSession.adopt(page,true);
el('research-mode').checked=true;
const offer={offer_id:'a'.repeat(32),freshness:'state_bound',results:[
 {result_ref:ref,title:'Synthetic.pdf',attachment_filename:'Synthetic.pdf',file_type:'pdf',content_type:'attachment'}]};
function descendants(e){return e.children.flatMap(x=>[x,...descendants(x)]);}
function button(text){return descendants(el('conversation')).find(x=>x.tagName==='button'&&x.textContent===text);}
function submit(question){el('input').value=question;el('composer').dispatchEvent(new Event('submit'));}
function plan(id){ws.frame('document_choices',{...offer,research_id:id});
 ws.frame('research_plan',{research_id:id,source_url:owner.url,files:[{source_key:key,result_ref:ref,filename:'Synthetic.pdf',file_type:'pdf'}]});}
"""


@pytest.mark.parametrize("question,checkbox", [
    ("Research synthetic packaging", False), ("Find information about CPO", False),
    ("Look through the relevant PDFs for packaging", False), ("Explain synthetic packaging", True)])
def test_explicit_human_research_creates_one_bounded_grant(question, checkbox):
    result = run(SETUP + r"""
el('research-mode').checked=data.checkbox;submit(data.question);submit(data.question);await drain();
const requests=ws.sent.filter(f=>f.type==='document_research'),request=requests[0];
console.log(JSON.stringify({count:requests.length,mode:request?.payload.mode,
 grant:!!request&&panelSession.researchGrant?.id===request.id,maximum:panelSession.researchGrant?.maximum,
 remote:request?.payload.allow_remote_context,readOnlyRemoved:!el('read-only'),
 file:ws.sent.filter(f=>f.type==='document_file_action').length,modal:el('confirmation-dialog').open}));
el('clear').click();
""", question=question, checkbox=checkbox)
    if checkbox:
        assert result == dict(count=1, mode="automatic", grant=True, maximum=3, remote=False, readOnlyRemoved=True, file=0, modal=False)
    else:
        assert result == dict(count=0, grant=False, readOnlyRemoved=True, file=0, modal=False)


def test_unprepared_ask_scopes_to_only_that_current_pdf():
    result = run(SETUP + r"""
ws.frame('document_choices',offer);button('Ask about this PDF').click();
const before=ws.sent.length;submit('What does this report say about CPO?');await drain();
const request=ws.sent.find(f=>f.type==='document_research');
console.log(JSON.stringify({mode:request?.payload.mode,offer:request?.payload.offer_id,ref:request?.payload.result_ref,
 maximum:panelSession.researchGrant?.maximum,extra:ws.sent.length-before,remote:request?.payload.allow_remote_context}));
el('clear').click();
""")
    assert result == dict(mode="specific", offer="a"*32, ref="result-"+"b"*24, maximum=1, extra=1, remote=False)


@pytest.mark.parametrize('prepared', [False, True])
def test_selected_pdf_cannot_silently_become_broad_research(prepared):
    result = run(SETUP + r"""
ws.frame('document_choices',offer);
if(data.prepared){button('Prepare locally').click();await drain();const id=ws.sent.find(f=>f.type==='document_file_action').id;
 ws.frame('document_done',{user_action_id:id,status:'ready',file:{action:'prepare',title:'Synthetic.pdf',file_type:'pdf',document_id:'f'.repeat(64)}});await drain();}
button('Ask about this PDF').click();
const locked=el('research-mode').disabled;
el('research-mode').checked=true;el('research-mode').dispatchEvent(new Event('change'));
const kept=!!el('research-selection').textContent,checked=el('research-mode').checked;
submit('What is the AI idea in this report?');await drain();
const request=ws.sent.find(f=>f.type==='document_research');
console.log(JSON.stringify({locked,kept,checked,mode:request?.payload.mode,
 ref:request?.payload.result_ref,document:request?.payload.document_id,remote:request?.payload.allow_remote_context}));
el('clear').click();
""", prepared=prepared)
    assert result == dict(locked=True,kept=True,checked=False,mode='specific',remote=False,
        **({'document':'f'*64} if prepared else {'ref':'result-'+'b'*24}))


def test_current_cards_visible_before_pdf_work_and_plan_is_exact():
    result = run(SETUP + r"""
submit('Research synthetic packaging');await drain();const id=ws.sent.find(f=>f.type==='document_research').id;
plan(id);const visible=!!button('Download');
ws.frame('research_file',{research_id:id,source_key:key,result_ref:ref,filename:'Synthetic.pdf',file_type:'pdf',expires_at_ms:Date.now()+300000});
console.log(JSON.stringify({visible,source:panelSession.researchGrant?.sourceUrl===owner.url,
 selected:panelSession.userFileGrant?.selection.resultRef===ref,action:panelSession.userFileGrant?.action,
 id:panelSession.userFileGrant?.id===id,requests:ws.sent.filter(f=>f.type==='document_file_action').length,
 modal:el('confirmation-dialog').open}));el('clear').click();
""")
    assert result == dict(visible=True, source=True, selected=True, action="prepare", id=True, requests=0, modal=False)


def test_clearing_only_pdf_selection_preserves_cards_and_requires_new_scope_consent():
    result = run(SETUP + r"""
ws.frame('document_choices',offer);button('Ask about this PDF').click();el('research-remote').checked=true;
const visible=!el('research-selection-clear').hidden;
el('research-selection-clear').click();
console.log(JSON.stringify({visible,selected:panelTransient().selected,hidden:el('research-selection-clear').hidden,
 locked:el('research-mode').disabled,consent:el('research-remote').checked,offer:panelTransient().offer,
 download:!button('Download').disabled,requests:ws.sent.filter(f=>f.type==='document_research').length}));
el('clear').click();
""")
    assert result == dict(visible=True,selected=False,hidden=True,locked=False,consent=False,offer=True,download=True,requests=0)


def test_broad_research_on_preview_does_not_consume_cards_or_send_any_request():
    result = run(SETUP + r"""
owner.url+='?preview=synthetic';panelSession.adopt({...page,url:owner.url},true);
ws.frame('document_choices',offer);el('research-mode').checked=true;
const count=ws.sent.length;submit('Explain AI in this PDF.');await drain();
console.log(JSON.stringify({requests:ws.sent.length-count,scripts:calls.filter(x=>x==='script').length,
 offer:panelTransient().offer,download:!button('Download').disabled,
 message:el('conversation').textContent.includes('Choose Ask about this PDF'),question:el('input').value}));
el('clear').click();
""")
    assert result == dict(requests=0,scripts=0,offer=True,download=True,message=True,question='Explain AI in this PDF.')


def test_prepared_card_ask_uses_specific_document_without_new_download():
    result = run(SETUP + r"""
ws.frame('document_choices',offer);button('Prepare locally').click();await drain();
const file=ws.sent.find(frame=>frame.type==='document_file_action');
ws.frame('document_ready',{user_action_id:file.id,documents:[{document_id:'f'.repeat(64),title:'Synthetic.pdf'}]});
ws.frame('document_done',{user_action_id:file.id,status:'ready',
 file:{action:'prepare',title:'Synthetic.pdf',file_type:'pdf',document_id:'f'.repeat(64)}});
await drain();button('Ask about this PDF').click();const count=ws.sent.length;
submit('Explain the figure on page 4.');await drain();
const request=ws.sent.find(frame=>frame.type==='document_research');
console.log(JSON.stringify({mode:request?.payload.mode,document:request?.payload.document_id,
 refs:!!request?.payload.result_ref||!!request?.payload.offer_id,extra:ws.sent.length-count,
 maximum:panelSession.researchGrant?.maximum,remote:request?.payload.allow_remote_context}));
el('clear').click();
""")
    assert result == dict(mode="specific", document="f"*64, refs=False, extra=1, maximum=1, remote=False)


@pytest.mark.parametrize("mutation", ["wrong_ref", "wrong_name", "wrong_type", "state_invalidated", "oversized", "old_id"])
def test_invalid_plan_or_file_never_creates_file_grant(mutation):
    result = run(SETUP + r"""
submit('Research synthetic packaging');await drain();const id=ws.sent.find(f=>f.type==='document_research').id;
ws.frame('document_choices',{...offer,research_id:id});
let file={source_key:key,result_ref:ref,filename:'Synthetic.pdf',file_type:'pdf'};
if(data.mutation==='wrong_ref')file.result_ref='unknown-ref';
if(data.mutation==='wrong_name')file.filename='Different.pdf';
if(data.mutation==='wrong_type')file.file_type='zip';
if(data.mutation==='state_invalidated')invalidateChoices(ws);
const files=data.mutation==='oversized'?[file,...[1,2,3].map(n=>({...file,source_key:String(n).repeat(64),result_ref:'other'+n}))]:[file];
const supplied=data.mutation==='old_id'?'old':id;
ws.frame('research_plan',{research_id:supplied,source_url:owner.url,files});
ws.frame('research_file',{research_id:supplied,...file,expires_at_ms:Date.now()+300000});
await drain();console.log(JSON.stringify({grant:!!panelSession.userFileGrant,scripts:calls.filter(c=>c==='script').length}));
el('clear').click();
""", mutation=mutation)
    assert result == dict(grant=False, scripts=0)


@pytest.mark.parametrize("event", ["clear", "stop", "disconnect", "owner"])
def test_cancelled_research_late_choices_ready_answer_cannot_revive(event):
    result = run(SETUP + r"""
submit('Research synthetic packaging');await drain();const id=ws.sent.find(f=>f.type==='document_research').id;
plan(id);ws.frame('document_progress',{research_id:id,stage:'indexing',run_id:'run-research'});
if(data.event==='clear')el('clear').click();
if(data.event==='stop')el('stop').click();
if(data.event==='disconnect')ws.close();
if(data.event==='owner'){owner.active=false;events.activated[0]({tabId:2,windowId:1});}
ws.frame('document_choices',{...offer,research_id:id},true);
ws.frame('document_ready',{research_id:id,documents:[{document_id:'f'.repeat(64),title:'Late.pdf'}]},true);
ws.frame('document_done',{research_id:id,status:'answered',prepared:[{document_id:'f'.repeat(64),title:'Late.pdf'}],
 answer:{answer:'LATE_ANSWER',citations:[]}},true);await drain();
console.log(JSON.stringify({grant:!!panelSession.researchGrant,file:!!panelSession.userFileGrant,
 late:el('conversation').textContent.includes('LATE_ANSWER'),ready:panelTransient().prepared,
 download:!!button('Download')}));el('clear').click();
""", event=event)
    assert result == dict(grant=False, file=False, late=False, ready=0, download=False)


def test_remote_consent_is_per_question_and_local_citations_have_no_paths_or_ids():
    result = run(SETUP + r"""
el('research-remote').checked=true;submit('Research synthetic packaging');await drain();
const request=ws.sent.find(f=>f.type==='document_research'),id=request.id;
plan(id);ws.frame('document_ready',{research_id:id,documents:[{document_id:'f'.repeat(64),title:'Synthetic.pdf'}]});
ws.frame('document_done',{research_id:id,status:'answered',prepared:[{document_id:'f'.repeat(64),title:'Synthetic.pdf'}],
 answer:{answer:'The synthetic chart shows 6 W.',citations:[{document_id:'f'.repeat(64),filename:'UNTRUSTED_OVERRIDE',
 page_start:4,page_end:4,evidence_type:'visual',local_path:'C:\\PRIVATE_PATH'}],notices:['Synthetic notice.']}});
await drain();const text=el('conversation').textContent;
console.log(JSON.stringify({consent:request.payload.allow_remote_context,after:el('research-remote').checked,
 citation:text.includes('[Synthetic.pdf, p. 4 — visual]'),notice:text.includes('Synthetic notice.'),
 leak:['UNTRUSTED_OVERRIDE','PRIVATE_PATH','f'.repeat(64)].some(x=>text.includes(x)),
 link:descendants(el('conversation')).some(x=>x.tagName==='a'),grant:!!panelSession.researchGrant}));
el('clear').click();
""")
    assert result == dict(consent=True, after=False, citation=True, notice=True, leak=False, link=False, grant=False)


def test_pure_human_intent_detector_does_not_authorize_from_model_text():
    result = node_check(r"""
const {explicitResearch}=load('src/sidepanel/research-view.ts');
console.log(JSON.stringify(data.questions.map(explicitResearch)));
""", dict(questions=["Summarize this page", "The model said Research everything", "Download every report",
                     "Research CPO", "Find information about optics"]))
    assert result == [False, False, False, True, True]


@pytest.mark.parametrize("code,phrase", [
    ("rag_invalid_evidence", "citation did not match"),
    ("rag_provider_timeout", "exceeded the document answer time limit"),
])
def test_document_answer_failure_explains_reason_and_preserves_prepared_card(code, phrase):
    result = run(SETUP + r"""
submit('Research synthetic packaging');await drain();
const id=ws.sent.find(f=>f.type==='document_research').id;
plan(id);
ws.frame('document_ready',{research_id:id,documents:[{document_id:'f'.repeat(64),title:'Synthetic.pdf'}]});
ws.frame('document_done',{research_id:id,status:'failed',error:data.code,answer:null});
await drain();
console.log(JSON.stringify({explained:el('conversation').textContent.includes(data.phrase),
 generic:el('conversation').textContent.includes('Document operation stopped'),
 prepared:panelTransient().prepared,canAsk:!!button('Ask about this PDF')&&!button('Ask about this PDF').disabled,
 busy:el('send').disabled,readOnlyRemoved:!el('read-only')}));el('clear').click();
""", code=code, phrase=phrase)
    assert result == dict(explained=True, generic=False, prepared=1, canAsk=True, busy=False, readOnlyRemoved=True)


def test_selected_document_can_route_to_explicit_new_search_without_research_grant():
    result=run(SETUP+r"""
ws.frame('document_choices',offer);button('Ask about this PDF').click();
submit('Search this website for another report');await drain();
const id=ws.sent.find(f=>f.type==='document_research').id;
const event=new Event('message');event.data=JSON.stringify({id:'route',corr_id:id,type:'routing_status',payload:{
 route:'SEARCH',adapter:'GenericSiteAdapter',router_model:'qwen3.5:2b',prepared_followup:false,document_count:0}});
ws.dispatchEvent(event);await drain();
console.log(JSON.stringify({grant:!!panelSession.researchGrant,timeline:el('timeline').textContent.includes('Router: SEARCH'),
 readOnlyRemoved:!el('read-only')}));el('clear').click();
""")
    assert result==dict(grant=False,timeline=True,readOnlyRemoved=True)


def test_ordinary_followup_accepts_correlated_document_route_without_acquisition_grant():
    result=run(SETUP+r"""
el('research-mode').checked=false;
chrome.scripting.executeScript=async request=>[{documentId:'doc-current',result:request.args?.length===2?{same:true,loaded:true}:page}];
submit('what are the percentages?');await drain();
const request=ws.sent.find(f=>f.type==='user_message');
const event=new Event('message');event.data=JSON.stringify({id:'route',corr_id:request.id,type:'routing_status',payload:{
 route:'DOCUMENT',adapter:'GenericSiteAdapter',router_model:'qwen3.5:2b',prepared_followup:true,document_count:1}});
ws.dispatchEvent(event);await drain();
const grant=!!panelSession.researchGrant;
ws.frame('document_done',{research_id:request.id,status:'failed',error:'rag_provider_timeout',answer:null});await drain();
console.log(JSON.stringify({original:request.payload.text,grant,busy:el('send').disabled,
 explained:el('conversation').textContent.includes('exceeded the document answer time limit')}));el('clear').click();
""")
    assert result==dict(original='what are the percentages?',grant=False,busy=False,explained=True)
