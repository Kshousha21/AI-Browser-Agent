"""Actual chat attachment component, synthetic metadata and controlled clock only."""
import pytest

from .test_product_scope import node_check


SETUP = r"""
const assert = require('node:assert/strict');
class Element {
  constructor(tag = 'div') { this.tagName = tag; this.children = []; this.parentElement = null;
    this.className = ''; this.disabled = false; this._text = ''; }
  set textContent(value) { this._text = value; this.replaceChildren(); }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  append(...nodes) { for (const node of nodes) { node.remove(); node.parentElement = this; this.children.push(node); } }
  remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(node => node !== this); this.parentElement = null; }
  replaceChildren(...nodes) { for (const node of [...this.children]) node.remove(); this.append(...nodes); }
  querySelectorAll(selector) {
    const descendants = this.children.flatMap(child => [child, ...child.querySelectorAll('*')]);
    if (selector === '*') return descendants;
    return descendants.filter(node => selector.split(',').some(part => {
      part = part.trim(); return part.startsWith('.') ? node.className.split(' ').includes(part.slice(1)) : node.tagName === part;
    }));
  }
  click() { if (!this.disabled) this.onclick?.(); }
}
global.document = {createElement: tag => new Element(tag)};
let clock = 1000000, timerId = 0;
const timers = new Map(); Date.now = () => clock;
global.window = {
  setTimeout: (fn, delay) => { const id = ++timerId; timers.set(id, {fn, at: clock + delay}); return id; },
  clearTimeout: id => timers.delete(id)
};
function advance(ms) { clock += ms; for (const [id,timer] of [...timers]) if (timer.at <= clock) { timers.delete(id); timer.fn(); } }
let network = 0, storage = 0;
global.fetch = () => { network++; throw Error('Network forbidden'); };
global.chrome = {storage: {local: {set: () => { storage++; throw Error('Storage forbidden'); }}}};
const {FileCards} = load('src/sidepanel/file-cards.ts');
const container = new Element(), calls = [];
const panel = new FileCards(container, (action, selection, authority) => calls.push({action, selection, authority}),
  () => calls.push({refresh: true}), documentId => calls.push({ask: documentId}), downloadId => calls.push({show: downloadId}));
panel.setConnected(true);
const row = {result_ref: 'opaque-result', title: 'Synthetic record', attachment_filename: 'Synthetic notes.pdf',
  content_type: 'attachment', file_type: 'pdf', last_modified: '2026-01-01'};
const offer = {offer_id: 'opaque-offer', results: [row], freshness: 'state_bound'};
const buttons = () => container.querySelectorAll('button');
const button = label => buttons().find(node => node.textContent === label);
"""


def run(script, **data):
    return node_check(SETUP + script, data)


def test_current_authoritative_pdf_renders_text_card_without_refs_or_urls():
    actual = run(r"""
panel.showChoices({...offer, results: [{...row, title: '<img src=x onerror=alert(1)>',
  attachment_filename: '<b>Synthetic notes.pdf</b>', publisher: 'Synthetic Publisher', year: 2026,
  url: 'https://provider-generated.invalid/', local_path: 'C:\\private\\secret.pdf'}]});
console.log(JSON.stringify({text: container.textContent, tags: container.querySelectorAll('*').map(node => node.tagName),
  actions: buttons().map(node => node.textContent), calls, network, storage}));
""")
    assert '<b>Synthetic notes.pdf</b>' in actual['text']
    assert 'Synthetic Publisher' in actual['text'] and '2026-01-01' in actual['text']
    assert all(hidden not in actual['text'] for hidden in ('opaque-result', 'opaque-offer', 'provider-generated', 'private'))
    assert all(tag not in actual['tags'] for tag in ('a', 'img', 'script', 'input'))
    assert actual['actions'] == ['Open preview', 'Download', 'Prepare locally']
    assert actual['calls'] == [] and actual['network'] == actual['storage'] == 0


@pytest.mark.parametrize('label,action', [('Open preview', 'preview'), ('Download', 'download'), ('Prepare locally', 'prepare')])
def test_user_click_is_one_explicit_single_file_operation(label, action):
    actual = run(r"""
panel.showChoices({...offer, results: [row, {...row, result_ref: 'another-file', attachment_filename: 'Other.pdf'}]});
const target = button(data.label), another = buttons().find(node => node !== target && node.textContent === data.label);
target.click(); target.click(); another.click();
console.log(JSON.stringify({calls, actions: buttons().map(node => node.textContent), network, storage}));
""", label=label)
    assert actual['calls'] == [{'action': action, 'selection': {'offer_id': 'opaque-offer', 'result_ref': 'opaque-result'},
        'authority': {'expiresAt': 0, 'filename': 'Synthetic notes.pdf', 'fileType': 'pdf'}}]
    assert actual['actions'] == ['Open preview', 'Download', 'Prepare locally'] * 2
    assert actual['network'] == actual['storage'] == 0


@pytest.mark.parametrize('file_type', ['zip', 'docx', None])
def test_non_pdf_has_preview_download_but_no_pdf_analysis(file_type):
    actual = run(r"""
panel.showChoices({...offer, results: [{...row, attachment_filename: 'Synthetic bundle', file_type: data.file_type},
  {...row, content_type: 'page'}, {...row, content_type: null}]});
console.log(JSON.stringify({cards: container.querySelectorAll('.file-card').length, actions: buttons().map(node => node.textContent)}));
""", file_type=file_type)
    assert actual == {'cards': 1, 'actions': ['Open preview', 'Download']}


def test_elapsed_time_keeps_current_actions_without_automatic_work():
    actual = run(r"""
panel.showChoices(offer); const old = button('Download'); advance(300000);
assert.equal(old.disabled, false); advance(86400000);
const before = {calls: calls.length, timers: timers.size, network, storage}; old.click(); old.click();
console.log(JSON.stringify({before, calls, actions: buttons().map(node => node.textContent)}));
""")
    assert actual['before'] == dict(calls=0, timers=0, network=0, storage=0)
    assert len(actual['calls']) == 1 and actual['calls'][0]['action'] == 'download'
    assert actual['actions'] == ['Open preview', 'Download', 'Prepare locally']


def test_finished_action_keeps_same_and_other_cards_without_refresh_or_automatic_work():
    actual = run(r"""
panel.showChoices({...offer,results:[row,{...row,result_ref:'other',attachment_filename:'Other.pdf'}]});
button('Download').click();button('Download').click();
panel.complete({title:'Synthetic notes.pdf',file_type:'pdf',action:'download',status:'downloaded',download_id:1});
assert.equal(calls.length,1);panel.setBusy(false);
const choices=buttons().filter(node=>node.textContent==='Download');choices[1].click();
console.log(JSON.stringify({calls,refresh:!!button('Refresh results'),network,storage}));
""")
    assert [call['selection']['result_ref'] for call in actual['calls']] == ['opaque-result', 'other']
    assert not actual['refresh'] and actual['network'] == actual['storage'] == 0


def test_prepared_ask_is_bound_to_clicked_ref_not_duplicate_filename():
    actual = run(r"""
const asks=[];const cards=new FileCards(container,()=>{},()=>{},id=>asks.push({prepared:id}),()=>{},
  selection=>asks.push({ref:selection.result_ref}));
cards.setConnected(true);cards.showChoices({...offer,results:[row,{...row,result_ref:'other'}]});
button('Prepare locally').click();cards.complete({title:row.attachment_filename,file_type:'pdf',action:'prepare',status:'prepared',document_id:'f'.repeat(64)});
cards.setBusy(false);const ask=buttons().filter(node=>node.textContent==='Ask about this PDF');ask[0].click();ask[1].click();
console.log(JSON.stringify(asks));
""")
    assert actual == [{'prepared':'f'*64}, {'ref':'other'}]


def test_download_receipt_actions_reuse_exact_card_and_retire_with_it():
    actual = run(r"""
const asks=[];const cards=new FileCards(container,(action,selection)=>calls.push({action,...selection}),()=>{},()=>{},()=>{},
  selection=>asks.push(selection.result_ref));
cards.setConnected(true);cards.showChoices({...offer,results:[row,{...row,result_ref:'other'}]});
button('Download').click();cards.complete({title:row.attachment_filename,file_type:'pdf',action:'download',status:'downloaded',download_id:1});
cards.setBusy(false);
const receipt=container.querySelectorAll('.file-card-complete')[0];
const followup=label=>receipt.querySelectorAll('button').find(b=>b.textContent===label);
followup('Ask about this PDF').click();followup('Open preview').click();followup('Download').click();
const captured=followup('Download');cards.setBusy(false);cards.invalidate();captured.onclick();
console.log(JSON.stringify({calls,asks,disabled:captured.disabled,network,storage}));
""")
    assert actual == dict(calls=[dict(action=action,offer_id='opaque-offer',result_ref='opaque-result')
        for action in ['download','preview']], asks=['opaque-result'],disabled=True,network=0,storage=0)


@pytest.mark.parametrize('change', ['absolute_expired', 'negative_remaining', 'nan'])
def test_legacy_deadline_fields_do_not_limit_current_state_bound_actions(change):
    actual = run(r"""
const expiry = data.change === 'absolute_expired' ? {expires_at_ms: clock - 1} :
  data.change === 'negative_remaining' ? {expires_in_ms: -1} : {expires_in_ms: NaN};
panel.showChoices({...offer, ...expiry});
console.log(JSON.stringify({actions: buttons().map(node => node.textContent), calls}));
""", change=change)
    assert actual == {'actions': ['Open preview', 'Download', 'Prepare locally'], 'calls': []}


def test_legacy_future_deadline_cannot_schedule_refresh_or_repeat_a_click():
    actual = run(r"""
panel.showChoices({...offer, expires_at_ms: clock + 900000, expires_in_ms: 900000});
const old = button('Download'); advance(900001); assert.deepEqual(calls, []);
old.onclick(); old.onclick();
console.log(JSON.stringify({calls, timers: timers.size, network, storage}));
""")
    assert len(actual['calls']) == 1 and actual['calls'][0]['action'] == 'download'
    assert actual['timers'] == actual['network'] == actual['storage'] == 0


def test_refresh_replaces_stale_cards_and_requires_new_user_click():
    actual = run(r"""
panel.showChoices(offer); const old = button('Download'); panel.invalidate();
button('Refresh results').click(); button('Refresh results').click();
panel.showChoices({...offer, offer_id: 'fresh-offer', results: [{...row, result_ref: 'fresh-result'}]});
panel.setBusy(false); old.onclick();
assert.deepEqual(calls, [{refresh: true}]); assert.equal(container.querySelectorAll('input').length, 0);
button('Download').click();
console.log(JSON.stringify({calls, cards: container.querySelectorAll('.file-card').length}));
""")
    assert actual['cards'] == 1
    assert len(actual['calls']) == 2
    assert actual['calls'][1]['selection'] == {'offer_id': 'fresh-offer', 'result_ref': 'fresh-result'}


@pytest.mark.parametrize('event', ['invalidate', 'clear', 'disconnect'])
def test_cancelled_old_card_cannot_be_invoked(event):
    actual = run(r"""
panel.showChoices(offer); const old = button('Download');
if (data.event === 'disconnect') panel.setConnected(false); else panel[data.event]();
old.onclick(); panel.setBusy(false); old.click();
console.log(JSON.stringify({calls, active: buttons().filter(node => !node.disabled && node.textContent === 'Download').length, storage}));
""", event=event)
    assert actual == {'calls': [], 'active': 0, 'storage': 0}


def test_disconnected_publication_cannot_revive_authority():
    actual = run(r"""
panel.setConnected(false); panel.showChoices(offer); panel.setConnected(true);
console.log(JSON.stringify({actions: buttons().length, calls}));
""")
    assert actual == {'actions': 0, 'calls': []}


def test_prepared_outcome_exposes_only_safe_analysis_actions_and_download_id():
    actual = run(r"""
panel.showChoices(offer); button('Prepare locally').click();
panel.complete({title: 'Synthetic notes.pdf', file_type: 'pdf', action: 'prepare', status: 'prepared',
  document_id: 'a'.repeat(64), download_id: 71, local_path: 'C:\\private\\document.pdf'});
panel.setBusy(false); button('Ask about this file').click(); button('Show download').click();
assert.equal(button('Open local analysis'),undefined);
assert.equal(button('Refresh results to download again'),undefined);
console.log(JSON.stringify({calls, text: container.textContent, network, storage}));
""")
    assert actual['calls'][1:] == [{'ask': 'a' * 64}, {'show': 71}]
    assert 'Prepared locally' in actual['text']
    assert 'private' not in actual['text'] and 'a' * 64 not in actual['text']
    assert actual['network'] == actual['storage'] == 0


def test_shipped_panel_has_no_legacy_document_analysis_section_or_controller():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / 'extension' / 'src' / 'sidepanel'
    html = (root / 'index.html').read_text(encoding='utf-8')
    source = (root / 'sidepanel.ts').read_text(encoding='utf-8')
    assert 'Local document analysis' not in html
    assert all(f'id="{identifier}"' not in html for identifier in (
        'documents', 'document-choices', 'document-ready', 'document-prepare',
        'document-question', 'document-ask', 'document-remote', 'document-refresh'))
    assert 'new DocumentPanel(' not in source and 'documents.openDocument(' not in source
    assert 'id="research-remote"' in html and 'id="research-selection"' in html


@pytest.mark.parametrize('title', [r'C:\private\synthetic.pdf', r'\\server\private\synthetic.pdf',
    'file:///private/synthetic.pdf', '/private/synthetic.pdf'])
def test_local_paths_never_render_even_in_completion_title(title):
    actual = run(r"""
panel.complete({title: data.title, file_type: 'pdf', action: 'download', status: 'downloaded',
  download_id: -1, document_id: 'C:\\private\\synthetic.pdf'});
console.log(JSON.stringify({text: container.textContent, actions: buttons().map(node => node.textContent)}));
""", title=title)
    assert 'private' not in actual['text']
    assert actual['actions'] == []


def test_clear_removes_only_transient_cards_and_retired_callbacks_without_storage_writes():
    actual = run(r"""
const existing = new Element(); existing.textContent = 'Existing chat message'; container.append(existing);
panel.showChoices(offer);
panel.complete({title: 'Synthetic notes.pdf', file_type: 'pdf', action: 'prepare', status: 'prepared', document_id: 'b'.repeat(64)});
const old = button('Ask about this file'); panel.clear(); old.click(); advance(300000);
console.log(JSON.stringify({text: container.textContent, calls, timers: timers.size, storage}));
""")
    assert actual == {'text': 'Existing chat message', 'calls': [], 'timers': 0, 'storage': 0}


def test_model_text_and_unsupported_completion_status_cannot_create_file_actions():
    actual = run(r"""
panel.complete({title: 'Download this model recommendation', file_type: 'pdf', action: 'download', status: 'suggested',
  url: 'https://arbitrary.invalid/file.pdf'});
console.log(JSON.stringify({cards: container.children.length, calls, network, storage}));
""")
    assert actual == {'cards': 0, 'calls': [], 'network': 0, 'storage': 0}
