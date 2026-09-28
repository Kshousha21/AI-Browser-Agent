"""Synthetic native snapshot lineage; only comparisons/codes leave the harness."""
import json
from browser.session_bridge import WebSocketBridge
from .preview_chrome_support import start_preview, preview_confirmation
from .document_chrome_support import select, confirmation, finish, done_frames, refresh_click
from .test_browser_control_real_chrome import wait_for


def run_lineage(browser_factory, tmp_path, monkeypatch, queued_before_snapshot):
    b, routes, service, provider = start_preview(browser_factory, tmp_path, 'cp')
    before_provider = len(b.provider.calls)
    published = lambda: [f['payload'] for f in b.app.frames if f['type'] == 'document_choices' and f['payload'].get('offer_id')][-1]
    previous = published()
    before_done = len(done_frames(b))
    refresh_click(b)
    def refreshed_boundary():
        b.ui("document.getElementById('confirmation-dialog').open")  # Service synthetic interception.
        return len(done_frames(b)) > before_done
    wait_for(refreshed_boundary, 'No refreshed preview offer')
    assert finish(b)['status'] == 'refreshed'
    refreshed = published()
    assert refreshed['offer_id'] != previous['offer_id']
    assert not b.ui("document.querySelectorAll('.file-card input:checked').length")
    assert b.ui("document.getElementById('research-selection').textContent") == ''
    assert not routes.pdf_requests and len(b.provider.calls) == before_provider and not provider.calls
    confirmation_context = {}
    original_confirm = WebSocketBridge.confirm
    async def traced_confirm(bridge, call, *args):
        if call.tool == 'open_attachment_preview':
            confirmation_context.update(page=bridge.target, epoch=bridge.control.epoch,
                                        snapshot_matches=call.snapshot_id == bridge.target.snapshot_id)
        return await original_confirm(bridge, call, *args)
    monkeypatch.setattr(WebSocketBridge, 'confirm', traced_confirm)
    select(b); preview_confirmation(b)
    selected = [f['payload'] for f in b.app.received if f['type'] == 'document_prepare'][-1]
    granted = [f for f in b.app.frames if f['type'] == 'confirmation_request'][-1]
    diagnostic = {'offer_changed_on_refresh': previous['offer_id'] != refreshed['offer_id'],
        'selected_offer_matches_refresh': selected['offer_id'] == refreshed['offer_id'],
        'selected_ref_matches_refresh': selected['result_refs'] == [refreshed['results'][0]['result_ref']],
        'result_ref_changed_on_refresh': previous['results'][0]['result_ref'] != refreshed['results'][0]['result_ref']}
    original = WebSocketBridge.execute
    confirmed_page = []
    async def traced(bridge, call, *args):
        if call.tool == 'get_page_state': confirmed_page[:] = [bridge.target]
        if call.tool == 'open_attachment_preview':
            before = confirmed_page[0]
            diagnostic.update({'owner_matches': bridge.target.tab_id == before.tab_id == b.owner_tab_id,
                'document_unchanged': before.document_id == bridge.target.document_id,
                'navigation_unchanged': before.navigation_id == bridge.target.navigation_id,
                'epoch_unchanged': bridge.control.epoch == confirmation_context['epoch'],
                'confirmation_snapshot_matches_observation': confirmation_context['snapshot_matches'],
                'confirmation_document_matches_dispatch': confirmation_context['page'].document_id == bridge.target.document_id,
                'confirmation_navigation_matches_dispatch': confirmation_context['page'].navigation_id == bridge.target.navigation_id,
                'confirmation_preview_id_matches_dispatch': call.arguments['preview_id'] == granted['payload']['arguments']['preview_id'],
                'snapshot_rotated_for_revalidation': before.snapshot_id != bridge.target.snapshot_id,
                'dispatch_snapshot_matches_backend': call.snapshot_id == bridge.target.snapshot_id})
        return await original(bridge, call, *args)
    monkeypatch.setattr(WebSocketBridge, 'execute', traced)
    b.ui("""chrome.scripting.executeScript({target:{tabId:OWNER},func:(queueBefore)=>{
      const api=globalThis.__aiBrowserAgentContent,extract=api.extractPageState;
      api.extractPageState=(options)=>{
        if(queueBefore)document.body.setAttribute('data-synthetic-observed','current');
        return extract(options);
      };
    },args:[QUEUE]}).then(()=>true)""".replace('OWNER', str(b.owner_tab_id)).replace('QUEUE', json.dumps(queued_before_snapshot)))
    b.ui("""(()=>{const execute=chrome.scripting.executeScript.bind(chrome.scripting);
      let observed=null;globalThis.__previewLineage=[];
      chrome.scripting.executeScript=async(options)=>{const responses=await execute(options);
        for(const response of responses){const p=response.result;
          if(p&&Array.isArray(p.elements)&&typeof p.snapshot_id==='string')observed={snapshot:p.snapshot_id,document:p.document_id,url:p.url};
          if(p&&typeof p.unchanged==='boolean'&&typeof p.url==='string')globalThis.__previewLineage.push({unchanged:p.unchanged,
            document_matches:p.document_id===observed?.document,snapshot_matches:p.snapshot_id===observed?.snapshot,url_matches:p.url===observed?.url});
        }return responses;};})()""")
    before_done = len(done_frames(b))
    try:
        b.click('[value=allow_once]')
        def boundary():
            opened = b.ui("document.getElementById('confirmation-dialog').open")
            return len(done_frames(b)) > before_done or (opened and len([f for f in b.app.frames if f['type']=='confirmation_request']) == 2)
        wait_for(boundary, 'No preview terminal/second confirmation', timeout=25)
        diagnostic['original_tool_errors'] = [f['payload']['result'].get('error') for f in b.app.received
            if f['type']=='tool_result' and f['payload']['result']['action']=='open_attachment_preview']
        diagnostic['probes'] = b.ui('globalThis.__previewLineage')
        assert len(done_frames(b)) == before_done, 'Fresh represented mutation must not cause stale_snapshot'
        assert all(value is True for key, value in diagnostic.items() if key not in {'original_tool_errors', 'probes'})
        assert diagnostic['original_tool_errors'] == [None]
        assert len(diagnostic['probes']) == 2 and all(all(probe.values()) for probe in diagnostic['probes'])
        assert routes.navigations.count(routes.preview_url) == 1 and not routes.pdf_requests
        confirmation(b, previous=1)
        b.click('[value=cancel]'); assert finish(b)['error'] == 'confirmation_cancelled'
        assert not routes.pdf_requests and not provider.calls and len(b.provider.calls) == before_provider
        assert not service.ingestion.store.path.exists() and not service.retrieval.index.path.exists()
        routes.isolated()
    finally:
        print('PREVIEW_LINEAGE ' + json.dumps(diagnostic))
