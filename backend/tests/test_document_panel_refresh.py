"""Current chat-card refresh and cancellation, with synthetic transport/clock."""
import pytest
from .panel_clear_support import run


def test_elapsed_time_keeps_current_file_actions_without_requests():
    result = run(r"""
const ws=sockets[0];ws.frame('document_choices',choices);selectChoice();
advance(299999);assert.equal(selectableCards(),1);
const before=[calls.length,ws.sent.length];advance(86400001);
console.log(JSON.stringify({rows:selectableCards(),refresh:!!cardButton('Refresh results'),
 expired:el('conversation').textContent.includes('Results expired; refresh results'),unchanged:before[0]===calls.length&&before[1]===ws.sent.length}));el('clear').click();
""")
    assert result == dict(rows=1, refresh=False, expired=False, unchanged=True)


def test_legacy_expiry_metadata_does_not_retire_current_file_choices():
    result = run(r"""
const ws=sockets[0];ws.frame('document_choices',{...choices,expires_at_ms:Date.now()-1});
console.log(JSON.stringify({rows:selectableCards(),refresh:!!cardButton('Refresh results'),
 expired:el('conversation').textContent.includes('Results expired; refresh results')}));el('clear').click();
""")
    assert result == dict(rows=1, refresh=False, expired=False)


def test_owner_invalidation_rejects_late_refresh_choices():
    result = run(r"""
const ws=sockets[0];ws.frame('document_choices',choices);refreshChoices(ws);await drain();
const id=ws.sent.find(f=>f.type==='document_refresh').id;
panelSession.invalidate('active_target_changed');
ws.frame('document_choices',{...choices,refresh_id:id});
ws.frame('document_done',{status:'refreshed',refresh_id:id});
console.log(JSON.stringify({rows:selectableCards()}));el('clear').click();
""")
    assert result == dict(rows=0)


@pytest.mark.parametrize('code', ['protocol_error', 'bad_request', 'protocol_mismatch'])
def test_rejected_refresh_cannot_accept_late_correlated_choices(code):
    result = run(r"""
const ws=sockets[0];ws.frame('document_choices',choices);refreshChoices(ws);await drain();
const id=ws.sent.find(f=>f.type==='document_refresh').id;
ws.frame('error',{code:data.code,message:'Synthetic unsupported refresh'});
ws.frame('document_choices',{...choices,refresh_id:id});
ws.frame('document_done',{status:'refreshed',refresh_id:id});
console.log(JSON.stringify({rows:selectableCards(),busy:el('send').disabled,status:el('run-status').textContent}));el('clear').click();
""", code=code)
    assert result == dict(rows=0, busy=False, status='Error')


def test_unknown_refresh_frame_has_no_choices_or_actions():
    result = run(r"""
const ws=sockets[0];const before=ws.sent.length;ws.frame('document_choices_future',choices);
console.log(JSON.stringify({rows:selectableCards(),unchanged:ws.sent.length===before,
 scripts:calls.filter(c=>c==='script').length}));el('clear').click();
""")
    assert result == dict(rows=0, unchanged=True, scripts=0)


def test_duplicate_prepare_is_reserved_before_async_owner_lookup():
    result = run(r"""
const ws=sockets[0];ws.frame('document_choices',choices);let release;
ownerGate=new Promise(r=>release=r);const button=cardButton('Prepare locally');button.click();button.click();
assert.equal(button.disabled,true);assert.equal(selectableCards(),0);
release();await drain();console.log(JSON.stringify(ws.sent.filter(f=>f.type==='document_file_action').length));el('clear').click();
""")
    assert result == 1


def test_explicit_refresh_publishes_unselected_cards_without_user_message():
    result = run(r"""
const ws=sockets[0];ws.frame('document_choices',choices);selectChoice();invalidateChoices(ws);
const button=cardButton('Refresh results');button.click();button.click();await drain();
const frames=ws.sent.filter(f=>f.type==='document_refresh');assert.equal(frames.length,1);
assert.deepEqual(frames[0].payload,{});
ws.frame('document_choices',{...choices,offer_id:'fresh',refresh_id:frames[0].id});
ws.frame('document_done',{status:'refreshed',refresh_id:frames[0].id});
console.log(JSON.stringify({rows:selectableCards(),selected:panelTransient().selected,
 consent:el('research-remote').checked,messages:ws.sent.filter(f=>f.type==='user_message').length,
 prepareRequests:ws.sent.filter(f=>f.type==='document_file_action').length,scripts:calls.filter(c=>c==='script').length}));el('clear').click();
""")
    assert result == dict(rows=1, selected=False, consent=False, messages=0, prepareRequests=0, scripts=0)


@pytest.mark.parametrize('action', ['clear','stop','disconnect'])
def test_late_refresh_choices_cannot_revive_cancelled_work(action):
    result = run(r"""
const ws=sockets[0];ws.frame('document_choices',choices);refreshChoices(ws);await drain();
const id=ws.sent.find(f=>f.type==='document_refresh').id;
if(data.action==='disconnect')ws.close();else el(data.action).click();
ws.frame('document_choices',{...choices,refresh_id:id},true);
ws.frame('document_done',{status:'refreshed',refresh_id:id},true);await drain();
console.log(JSON.stringify({rows:selectableCards(),consent:el('research-remote').checked}));el('clear').click();
""", action=action)
    assert result == dict(rows=0, consent=False)


@pytest.mark.parametrize('action', ['clear','stop','disconnect'])
def test_refresh_cancelled_during_preflight_never_dispatches(action):
    result = run(r"""
const ws=sockets[0];ws.frame('document_choices',choices);let release;ownerGate=new Promise(r=>release=r);
refreshChoices(ws);if(data.action==='disconnect')ws.close();else el(data.action).click();
release();await drain();console.log(JSON.stringify({requests:ws.sent.filter(f=>f.type==='document_refresh').length,
 scripts:calls.filter(c=>c==='script').length,rows:selectableCards(),busy:el('send').disabled}));el('clear').click();
""", action=action)
    assert result == dict(requests=0, scripts=0, rows=0, busy=False)


def test_file_prepare_does_not_relax_readonly_or_add_blanket_authority():
    result = run(r"""
const ws=sockets[0];ws.frame('document_choices',choices);let release;
ownerGate=new Promise(r=>release=r);prepareChoice();release();await drain();
const frame=ws.sent.find(f=>f.type==='document_file_action');
console.log(JSON.stringify({readOnlyRemoved:!el('read-only'),action:frame.payload.action,
 exactRef:frame.payload.result_ref===choices.results[0].result_ref,
 controls:ws.sent.filter(f=>f.type==='control'||f.type==='resume').length}));el('clear').click();
""")
    assert result == dict(readOnlyRemoved=True, action='prepare', exactRef=True, controls=0)


@pytest.mark.parametrize('code,fragment', [('stale_result_ref','Results changed or were cleared'),
    ('stale_snapshot','The page or selected file state changed'),
    ('attachment_preview_only','Only a preview link'), ('unsupported_attachment_type','Unsupported document type'),
    ('read_only_blocked','This legacy request was blocked'),
    ('attachment_unavailable','No unambiguous authorized preview-PDF or direct-PDF candidate')])
def test_error_states_are_distinct_and_not_actionable(code, fragment):
    result = run(r"""
const ws=sockets[0];ws.frame('document_choices',choices);prepareChoice();await drain();
const id=ws.sent.find(f=>f.type==='document_file_action').id;
ws.frame('document_done',{user_action_id:id,status:'failed',error:data.code,prepared:[]});
console.log(JSON.stringify({message:el('conversation').textContent,rows:selectableCards(),
 status:el('run-status').textContent}));el('clear').click();
""", code=code)
    assert result['rows'] == 0 and result['status'] == 'Error' and fragment in result['message']


@pytest.mark.parametrize('code,fragment', [
    ('invalid_pdf','not a valid PDF'),
    ('pdf_encrypted','requires a password'),
    ('pdf_extraction_not_allowed','does not permit text extraction'),
    ('pdf_crypto_unavailable','Install the updated requirements'),
    ('pdf_parse_failed','parser could not read'),
    ('pdf_too_large','file-size limit'),
    ('pdf_limit_exceeded','preparation limit'),
    ('pdf_not_found','no longer available locally'),
    ('pdf_read_failed','could not read the local PDF')])
def test_pdf_preparation_errors_explain_failure_and_keep_retry_card(code, fragment):
    # A parse/dependency failure does not invalidate the current file selection.
    # Retain the card so the user can retry after fixing the local installation.
    result = run(r"""
const ws=sockets[0];ws.frame('document_choices',choices);prepareChoice();await drain();
const id=ws.sent.find(f=>f.type==='document_file_action').id;
ws.frame('document_done',{user_action_id:id,status:'failed',error:data.code,prepared:[]});
console.log(JSON.stringify({message:el('conversation').textContent,rows:selectableCards(),
 status:el('run-status').textContent}));el('clear').click();
""", code=code)
    assert result['rows'] == 1 and result['status'] == 'Error' and fragment in result['message']
