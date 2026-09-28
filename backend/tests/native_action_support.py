"""Invoke an isolated test browser toolbar without exposing private page data."""
import json

def native_action(cdp, launcher, tab_id, extension, *, cold=False):
    owner = cdp.evaluate('chrome.tabs.update('+str(tab_id)+',{active:true})',launcher)
    url=cdp.evaluate('chrome.tabs.get('+str(tab_id)+').then(t=>t.url)',launcher)
    rows=cdp.call('Target.getTargets',filter=[dict(type='tab',exclude=False)])['targetInfos']
    matches=[row for row in rows if row.get('url')==url]
    evidence = dict(stage='native_tab_target', matching_targets=len(matches), tab_targets=len(rows),
        blank_target_urls=sum(not row.get('url') for row in rows),
        owner_identity=isinstance(owner,dict) and owner.get('id')==tab_id,
        owner_active=isinstance(owner,dict) and owner.get('active') is True,
        owner_url_unchanged=isinstance(owner,dict) and owner.get('url')==url,
        cold=cold)
    if len(matches) != 1:
        try:
            targets = cdp.call('Target.getTargets')['targetInfos']
            evidence.update(matching_page_targets=sum(t.get('type')=='page' and t.get('url')==url for t in targets),
                owner_panels=sum(t.get('url')==f'chrome-extension://{extension}/sidepanel/index.html?tabId={tab_id}' for t in targets))
        except Exception:
            evidence['diagnostic_transport_available'] = False
    print('NATIVE_TAB_TARGET '+json.dumps(evidence,separators=(',',':')),flush=True)
    assert len(matches)==1, 'Ambiguous synthetic native tab target; refusing wrong-tab action'
    if cold:
        assert not any(t.get('url')==f'chrome-extension://{extension}/service-worker.js'
            for t in cdp.call('Target.getTargets')['targetInfos']), 'Worker warmed before the actual toolbar action'
    # Chromium ExtensionsHandler::TriggerAction invokes the actual toolbar
    # model ExecuteUserAction(kCdp). No JS event or userGesture flag is injected.
    cdp.call('Extensions.triggerAction',id=extension,targetId=matches[0]['targetId'])

