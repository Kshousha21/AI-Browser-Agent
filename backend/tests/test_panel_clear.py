"""Deterministic Clear tests execute the real bundled UI; no provider or browser I/O."""
import pytest
from .panel_clear_support import run, NEUTRAL


@pytest.mark.parametrize('state', ['conversation', 'results', 'stale', 'selection', 'consent', 'prepared', 'busy'])
def test_clear_resets_all_transient_state_without_requests(state):
    actual = run(r"""
const ws=sockets[0];
ws.frame('assistant_done',{text:'Synthetic final',usage:{total_tokens:123},elapsed_ms:4000});
ws.frame('timeline_step',{step:1,tool:'get_page_state',reason:'Synthetic',result:{success:true,duration_ms:1}});
if(data.state!=='conversation')ws.frame('document_choices',choices);
if(['selection','consent','busy','stale'].includes(data.state))selectChoice();
if(['consent','prepared','busy'].includes(data.state)){question();el('research-remote').checked=true;}
if(data.state==='prepared'){prepareChoice();await drain();const id=ws.sent.find(f=>f.type==='document_file_action').id;
 ws.frame('document_ready',{...ready,user_action_id:id});ws.frame('document_done',{user_action_id:id,status:'ready',file:{title:'Synthetic.pdf',file_type:'pdf',action:'prepare',document_id:'a'.repeat(64)}});selectChoice();}
if(['busy','stale'].includes(data.state)){prepareChoice();await drain();ws.frame('document_progress',{stage:'validating',run_id:'synthetic-run'});}
if(data.state==='stale')ws.frame('document_done',{status:'failed',error:'stale_result_ref',prepared:[]});
el('input').value='Unsent draft';
panelSession.target={snapshot_id:'old-snapshot'};panelSession.snapshots.add('old-snapshot');
let revoked=0;panelSession.ports.add({disconnect:()=>revoked++});
const before={frames:ws.sent.length,calls:calls.length,sockets:sockets.length};
assert.equal(el('clear').disabled,false);el('clear').click();await drain();
const first=neutral();el('clear').click();await drain();
console.log(JSON.stringify({first,second:neutral(),revoked,closed:ws.closed,
 frames:ws.sent.length-before.frames,calls:calls.length-before.calls,sockets:sockets.length-before.sockets}));
""", state=state)
    assert actual['first'] == actual['second'] == NEUTRAL
    assert actual['revoked'] == actual['closed'] == 1
    assert actual['frames'] == actual['calls'] == actual['sockets'] == 0


def test_clear_cancels_pending_confirmation_and_ignores_all_old_frames():
    actual = run(r"""
const ws=sockets[0];ws.frame('document_choices',choices);prepareChoice();await drain();
ws.frame('confirmation_request',{tool:'acquire_attachment',arguments:{result_ref:'old-ref'},risk_reasons:['download']});
assert.equal(el('confirmation-dialog').open,true);
el('clear').click();await drain();const before=ws.sent.length;
for(const [type,payload] of [['document_choices',choices],['document_ready',ready],
 ['document_progress',{stage:'indexing',run_id:'old'}],['document_done',{status:'failed',error:'stale_result_ref'}],
 ['assistant_delta',{text:'Late'}],['assistant_done',{text:'Late',usage:{},elapsed_ms:1}],
 ['timeline_step',{}],['run_status',{}],['error',{code:'old',message:'Late'}],['tool_request',{}],
 ['confirmation_request',{}],['control_state',{}]])ws.frame(type,payload,true);
await drain();console.log(JSON.stringify({state:neutral(),frames:ws.sent.length-before}));
""")
    assert actual == dict(state=NEUTRAL, frames=0)


@pytest.mark.parametrize('operation', ['send', 'prepare'])
def test_clear_revokes_preflight_awaiting_owner(operation):
    actual = run(r"""
const ws=sockets[0];ws.frame('document_choices',choices);question();el('research-remote').checked=true;
let release;ownerGate=new Promise(resolve=>release=resolve);
if(data.operation==='send'){el('input').value='Pending';el('composer').dispatchEvent(new Event('submit'));}
else prepareChoice();
await drain();const before=ws.sent.length;el('clear').click();release();await drain();
console.log(JSON.stringify({state:neutral(),frames:ws.sent.length-before,scripts:calls.filter(x=>x==='script').length}));
""", operation=operation)
    assert actual == dict(state=NEUTRAL, frames=0, scripts=0)


def test_clear_during_connection_rejects_old_wait_without_reopening():
    actual = run(r"""
el('clear').click();autoHello=false;el('input').value='Pending connect';el('composer').dispatchEvent(new Event('submit'));await drain();
assert.equal(sockets.length,2);const ws=sockets[1];el('clear').click();await drain();
ws.frame('hello_ack',{provider:'mock',...require('../shared/build-contract.json'),read_only_default:true},true);await drain();
console.log(JSON.stringify({state:neutral(),sockets:sockets.length,closed:ws.closed,scripts:calls.filter(x=>x==='script').length}));
""")
    assert actual == dict(state=NEUTRAL, sockets=2, closed=1, scripts=0)


def test_clear_cannot_reenable_a_closed_owner():
    actual = run(r"""
events.removed.forEach(fn=>fn(1));el('clear').click();await drain();
console.log(JSON.stringify({send:el('send').disabled,owner:panelSession.ownerTabId,sockets:sockets.length}));
""")
    assert actual == dict(send=True, owner=1, sockets=1)


def test_clear_during_boot_does_not_automatically_connect():
    actual = run(r"""
console.log(JSON.stringify({state:neutral(),sockets:sockets.length}));
""", clear_on_boot=True)
    assert actual == dict(state=NEUTRAL, sockets=0)


def test_clear_revokes_pending_settings_reconnect():
    actual = run(r"""
let release;ownerGate=new Promise(resolve=>release=resolve);
events.settings.forEach(fn=>fn({settings:{}},'local'));await drain();
el('clear').click();release();await drain();
console.log(JSON.stringify({state:neutral(),sockets:sockets.length}));
""")
    assert actual == dict(state=NEUTRAL, sockets=1)


@pytest.mark.parametrize('operation', ['observation', 'tool_result'])
def test_old_async_completion_cannot_write_into_new_session(operation):
    actual = run(r"""
const old=sockets[0];let release;scriptGate=new Promise(resolve=>release=resolve);
if(data.operation==='observation'){el('input').value='Old observation';el('composer').dispatchEvent(new Event('submit'));}
else {old.frame('document_choices',choices);prepareChoice();await drain();
 old.frame('tool_request',{tool:'get_page_state',arguments:{full:true},reset_target:true,tab_id:1,
   run_id:'synthetic-run',snapshot_id:'synthetic-snapshot',control_epoch:0});}
await drain();assert.equal(calls.filter(x=>x==='script').length,1);
el('clear').click();await drain();assert.deepEqual(neutral().messages,1);
scriptGate=null;el('input').value='New explicit request';el('composer').dispatchEvent(new Event('submit'));await drain();
assert.equal(sockets.length,2);const current=sockets[1];current.frame('document_choices',choices);
const before={messages:el('conversation').children.length,frames:current.sent.length,status:el('run-status').textContent};
release();await drain();
console.log(JSON.stringify({stable:before.messages===el('conversation').children.length&&before.frames===current.sent.length&&before.status===el('run-status').textContent,
 choices:selectableCards(),oldClosed:old.closed,newToolResults:current.sent.filter(x=>x.type==='tool_result').length}));
""", operation=operation)
    assert actual == dict(stable=True, choices=1, oldClosed=1, newToolResults=0)
