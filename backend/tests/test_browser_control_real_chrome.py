"""Generic acceptance through Chrome's real side panel, using synthetic pages only.

Optional hooks support intercepted synthetic websites.

The provider is scripted in-process; no .env, live credentials, or remote pages.
Run after building extension/dist. Measurements printed with pytest -s.
"""
from __future__ import annotations

import asyncio
from contextlib import ExitStack
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import uvicorn
from websockets.sync.client import connect

from browser.config import Settings
from browser.providers import MockProvider, ToolCall
from browser.providers.base import FinalAnswer, ProviderResponse
from browser.server import create_app
from .conftest import TEST_TOKEN
from .chrome_profile import chrome_arguments, close_test_chrome, phase4b_profile, permission_query, installed_test_extension, wait_for_owned_chrome
from .chrome_startup import ChromeStartup
from .chrome_process import launch_test_chrome
from .chrome_dom_support import _find_chrome, _free_loopback_port

REPO = Path(__file__).absolute().parents[2]


def wait_for(predicate, message, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(message)


def wait_native_panel_layout(cdp, site, panel):
    # Chrome resolves open/starts the renderer before its native opening and
    # window-resize animations finish. Do not start a second tab transition in
    # that interval. Observe actual layout; any size change resets the quiet
    # period. This neither opens the panel nor synthesizes visibility/gestures.
    previous, unchanged_since = None, time.monotonic()
    samples, changes = 0, 0
    def settled():
        nonlocal previous, unchanged_since, samples, changes
        layout = [cdp.evaluate('[innerWidth,innerHeight,document.visibilityState]', session)
                  for session in (site, panel)]
        samples += 1
        if layout != previous:
            changes += 1
            previous, unchanged_since = layout, time.monotonic()
        return (all(row[0] > 0 and row[1] > 0 and row[2] == 'visible' for row in layout)
                and time.monotonic() - unchanged_since >= 0.45)
    try:
        wait_for(settled, 'Native owner/panel layout did not settle')
    except AssertionError:
        state = {}
        try:
            state = cdp.evaluate("""(async()=>{const id=Number(new URL(location.href).searchParams.get('tabId'));
                const tab=await chrome.tabs.get(id), win=await chrome.windows.get(tab.windowId);
                return {owner_active:tab.active,window_focused:win.focused};})()""", panel)
        except Exception:
            pass  # Diagnostics never replace or retry the original failure.
        rows = [dict(nonzero=row[0] > 0 and row[1] > 0, visible=row[2] == 'visible')
                for row in previous] if previous else []
        print('CHROME_LAYOUT ' + json.dumps(dict(samples=samples, changes=changes,
            owner=rows[0] if rows else None, panel=rows[1] if rows else None,
            owner_active=state.get('owner_active') is True,
            window_focused=state.get('window_focused') is True)), flush=True)
        raise


class Cdp:
    def __init__(self, socket):
        self.socket = socket
        self.sequence = 0
        self.responses = {}
        self.event_handlers = []

    def call(self, method, session=None, **params):
        self.last_method = method  # Static command name only, never parameters.
        self.sequence += 1
        request_id = self.sequence
        request = dict(id=request_id, method=method, params=params)
        if session:
            request['sessionId'] = session
        self.socket.send(json.dumps(request))
        while request_id not in self.responses:
            answer = json.loads(self.socket.recv(timeout=15))
            if 'id' in answer:
                self.responses[answer['id']] = answer
            else:
                # A Fetch handler may issue a nested fulfillment command. Keep
                # replies by id so that neither caller loses its response.
                for handler in list(self.event_handlers):
                    handler(answer)
        answer = self.responses.pop(request_id)
        assert 'error' not in answer, (method, answer.get('error'))
        return answer.get('result', {})

    def evaluate(self, expression, session):
        answer = self.call('Runtime.evaluate', session=session, expression=expression,
                           awaitPromise=True, returnByValue=True)
        assert 'exceptionDetails' not in answer, answer.get('exceptionDetails')
        return answer.get('result', {}).get('value')

    def attach(self, target):
        session = self.call('Target.attachToTarget', targetId=target, flatten=True)['sessionId']
        self.call('Runtime.enable', session=session)
        self.call('Page.enable', session=session)
        return session


def page_payloads(messages):
    pages = []
    decoder = json.JSONDecoder()
    def visit(obj):
        if isinstance(obj, dict):
            if obj.get('type') == 'page_state_full':
                pages.append(obj)
            else:
                for value in obj.values():
                    visit(value)
        elif isinstance(obj, list):
            for value in obj:
                visit(value)
    for message in messages:
        text = message.content if isinstance(message.content, str) else json.dumps(message.content)
        pos = 0
        while pos < len(text):
            start = text.find('{', pos)
            if start < 0:
                break
            try:
                obj, length = decoder.raw_decode(text[start:])
                visit(obj)
                pos = start + length
            except ValueError:
                pos = start + 1
    return pages


class ScriptedBrowserProvider(MockProvider):
    def __init__(self, scenario, url):
        super().__init__()
        self.scenario = scenario
        self.url = url
        self.pages = []
        self.cancelled = threading.Event()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.b = None
        self.a = None

    async def generate(self, messages, tools=None, response_schema=None):
        self.calls.append(list(messages))
        pages = page_payloads(messages)
        assert pages, 'A full current PageState must reach the provider'
        page = pages[-1]
        self.pages.append(page)
        n = len(self.calls)
        self.entered.set()
        if self.a is None:
            self.a = page
        if self.scenario == 'lease_close' and n == 1:
            while not self.release.is_set():
                await asyncio.sleep(0.01)
        if self.scenario in {'resume', 'panel_close'} and n == 1:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        decision = FinalAnswer(answer='Local browser-control acceptance complete.')
        args = {}
        tool = None
        snapshot = page['snapshot_id']
        if self.scenario == 'tabs':
            if n == 1:
                tool = 'get_tabs'
            elif n == 2:
                tool, args = 'switch_tab', {'tab_id': self.other_tab_id}
            elif n == 3:
                tool, args = 'open_tab', {'url': self.url + '/b'}
            elif n == 4:
                tool = 'read_page'
            elif n == 5:
                tool = 'get_page_state'
            elif n == 6:
                element = next(x for x in self.a['elements'] if x.get('name') == 'Inspect marker')
                tool, args = 'click', {'element_id': element['agent_id']}
                # Strict tab ownership denies reassignment before a foreign
                # snapshot can be observed. Inject a retired legacy decision.
                snapshot = 'retired-other-tab-snapshot'
        elif self.scenario.startswith('close_'):
            if n == 1:
                tool = 'get_tabs'
            elif n == 2:
                self.b = page
                tool, args = 'close_tab', {'tab_id': page['tab_id']}
        elif self.scenario == 'lease_close' and n == 1:
            element = next(x for x in page['elements'] if x.get('name') == 'Inspect marker')
            tool, args = 'click', {'element_id': element['agent_id']}
        elif self.scenario == 'confirm' and n <= 2:
            element = next(x for x in page['elements'] if x.get('name', '').startswith('Delete'))
            tool, args = 'click', {'element_id': element['agent_id']}
        if tool:
            decision = ToolCall(tool=tool, arguments=args, snapshot_id=snapshot, reason='Synthetic local acceptance')
        return ProviderResponse(decision=decision, raw_text='', tool_call_id=f'local-{n}')


class Capture:
    def __init__(self, app):
        self.app = app
        self.frames = []

    async def __call__(self, scope, receive, send):
        async def recorded_send(message):
            if message['type'] == 'websocket.send' and 'text' in message:
                frame = json.loads(message['text'])
                if frame['type'] in {'tool_request', 'timeline_step', 'confirmation_request', 'control_state', 'assistant_done', 'error'}:
                    self.frames.append(frame)
            await send(message)
        await self.app(scope, receive, recorded_send)


def built_panel_opener(assets):
    """Rollup chooses shared chunk names from the dependency graph, not API names."""
    matches = []
    for bundle in assets.glob('*.js'):
        exported = re.search(r'openPanelForTab as (\w+)', bundle.read_text(encoding='utf-8'))
        if exported:
            matches.append((bundle, exported.group(1)))
    assert len(matches) == 1, 'Exactly one built shared panel opening function is required'
    return matches[0]


def open_tab_panel(cdp, launcher, tab_id, extension):
    """Real user-gesture opening; availability comes from the production worker."""
    path = f'sidepanel/index.html?tabId={tab_id}'
    wait_for(lambda: cdp.evaluate('chrome.sidePanel.getOptions({tabId:' + str(tab_id) + '})', launcher)
             == dict(tabId=tab_id, path=path, enabled=True), 'Owner panel not enabled')
    from .native_action_support import native_action
    native_action(cdp, launcher, tab_id, extension)
    panel_url = f'chrome-extension://{extension}/{path}'
    identifier = wait_for(lambda: next((x['targetId'] for x in cdp.call('Target.getTargets')['targetInfos']
        if x.get('url') == panel_url), None), 'Real tab-specific side panel not created')
    return identifier, cdp.attach(identifier)


class FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        name = 'B' if self.path == '/b' else 'A'
        body = (f'<!doctype html><title>Fixture {name}</title><main><h1>Fixture {name}</h1>'
                f'<p>BODY_CANARY_{name} Local neutral content for browser control.</p>'
                '<button type="button" id="inspect" onclick="window.inspected++">Inspect marker</button>'
                '<button id="save" onclick="window.saved++">Delete note</button></main>'
                '<script>window.saved=0;window.inspected=0;</script>').encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle(self):
        try:
            super().handle()
        except ConnectionResetError:
            pass

    def log_message(self, *args):
        pass


@pytest.fixture
def browser_factory(tmp_path):
    resources = []
    stack = ExitStack()
    def start(scenario, *, provider=None, initial_url=None, allowlist=None,
              read_only=False, interceptor=None, capture_factory=Capture,
              persistent_profile=False, headed=False, site_adapters=None, document_services_factory=None):
        chrome = _find_chrome()
        assert chrome, 'Installed Chrome is required for Phase 3C acceptance'
        assert (REPO / 'extension/dist/manifest.json').is_file(), 'Build extension first'
        site = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler)
        threading.Thread(target=site.serve_forever, daemon=True).start()
        url = f'http://127.0.0.1:{site.server_port}'
        provider = provider if provider is not None else ScriptedBrowserProvider(scenario, url)
        settings = Settings(auth={'token': TEST_TOKEN}, provider={'active': 'mock'},
                            providers={'mock': {'type': 'mock', 'model': 'mock-1'}},
                            site_adapters=site_adapters or {}, security={'audit_log_path': str(tmp_path / 'audit.jsonl')})
        app = capture_factory(create_app(settings, provider=provider, document_services_factory=document_services_factory))
        port = _free_loopback_port()
        server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='error', access_log=False))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        wait_for(lambda: server.started, 'Local backend did not start')
        debug = _free_loopback_port()
        profile = stack.enter_context(phase4b_profile()) if persistent_profile else tmp_path / 'chrome'
        if interceptor is not None and hasattr(interceptor, 'configure_profile'):
            stack.enter_context(interceptor.configure_profile(profile))
        command = chrome_arguments(chrome, profile, debug, headed=headed)
        launch_started = time.monotonic()
        process = launch_test_chrome(command)
        resource = [site, server, thread, process, None]
        resources.append(resource)
        info = wait_for_owned_chrome(debug, process)
        socket = stack.enter_context(connect(info['webSocketDebuggerUrl'], max_size=8_000_000))
        cdp = Cdp(socket)
        startup = ChromeStartup(cdp, process, profile, REPO, started=launch_started, persistent=persistent_profile)
        startup.bind_process()
        resource[4] = cdp  # Browser.close is allowed only after owned-PID proof.
        extension = (installed_test_extension(cdp) if persistent_profile else
                     cdp.call('Extensions.loadUnpacked', path=str(REPO / 'extension/dist'))['id'])
        assert extension, 'Load the built extension into the isolated test profile before persistent-profile tests'
        startup.extension = extension
        startup.clean_restored_pages()
        if persistent_profile:
            # Chrome can cache an unpacked service worker across browser starts.
            # Reload the installed extension normally, preserving the human
            # host grant; never reinstall it or fabricate optional permissions.
            startup.reload(extension)
            assert installed_test_extension(cdp) == extension, 'Installed extension identity changed across reload'
        # Retire any restored pages that appeared during extension initialization.
        startup.clean_restored_pages()
        target = cdp.call('Target.createTarget', url='about:blank' if interceptor else url + '/a')['targetId']
        site_session = cdp.attach(target)
        if interceptor is not None:
            interceptor.install(cdp, site_session)
            cdp.call('Page.navigate', session=site_session, url=initial_url)
            wait_for(lambda: cdp.evaluate('document.readyState', site_session) == 'complete', 'Intercepted fixture not ready')
        launcher_target, launcher = startup.open_options(extension)
        config = dict(backendUrl=f'ws://127.0.0.1:{port}', token=TEST_TOKEN,
                      allowlist=allowlist if allowlist is not None else [url + '/*'], maxVisibleTextChars=20000)
        cdp.evaluate('chrome.storage.local.set({settings:' + json.dumps(config) + '})', launcher)
        if interceptor is not None:
            assert cdp.evaluate(permission_query(initial_url), launcher) is True, (
                'The intercepted fixture origin needs optional site access in the isolated test profile. '
                'Grant it through extension Settings before running this persistent-profile test.')
        owner_url = initial_url if interceptor else url + '/a'
        tab_id = cdp.evaluate('chrome.tabs.query({}).then(tabs=>tabs.find(t=>t.url===' + json.dumps(owner_url) + ')?.id)', launcher)
        assert isinstance(tab_id, int), 'Owning fixture tab missing'
        panel_target, panel = open_tab_panel(cdp, launcher, tab_id, extension)
        cdp.call('Target.closeTarget', targetId=launcher_target)
        cdp.call('Target.activateTarget', targetId=target)
        # A running renderer/socket is not proof that the headed native window
        # completed layout. Establish ordinary visible-window readiness; the
        # unchanged old opener also fails hide/close assertions without this.
        if headed:
            window=cdp.call('Browser.getWindowForTarget',targetId=target)['windowId']
            cdp.call('Browser.setWindowBounds',windowId=window,bounds=dict(windowState='normal',width=1100,height=900))
            # Target activation is not an OS-window focus guarantee on Windows.
            # Focus only this fixture's owning window before checking real visibility.
            cdp.evaluate('chrome.tabs.get(' + str(tab_id) + ').then(t=>chrome.windows.update(t.windowId,{focused:true}))', panel)
            cdp.call('Page.bringToFront',session=site_session)
        wait_for(lambda: cdp.evaluate("document.getElementById('status-text')?.textContent.includes('Backend connected')", panel), 'Panel not connected: ' + str(cdp.evaluate("document.getElementById('status-text')?.textContent", panel)))
        if headed:
            layout_started = time.monotonic()
            wait_native_panel_layout(cdp, site_session, panel)
            startup.timings['panel_layout'] = round(time.monotonic() - layout_started, 4)
        browser = Browser(cdp, panel, panel_target, site_session, target, provider, app, url, tmp_path)
        browser.socket = socket
        browser.extension = extension
        browser.owner_tab_id = tab_id
        startup.timings['panel_ready'] = round(time.monotonic() - launch_started, 4)
        browser.startup = startup.report()
        return browser
    yield start
    for site, server, thread, process, cdp in reversed(resources):
        close_test_chrome(cdp, process)
        server.should_exit = True
        site.shutdown()
        site.server_close()
        thread.join(timeout=5)
    stack.close()


class Browser:
    def __init__(self, cdp, panel, panel_target, site_session, target, provider, app, url, tmp_path):
        self.cdp, self.panel, self.panel_target = cdp, panel, panel_target
        self.site, self.target, self.provider, self.app, self.url, self.tmp_path = site_session, target, provider, app, url, tmp_path

    def ui(self, expression):
        return self.cdp.evaluate(expression, self.panel)

    def click(self, selector):
        self.ui('document.querySelector(' + json.dumps(selector) + ').click()')

    def start(self, goal=None):
        self.ui("document.getElementById('input').value=" + json.dumps(goal if goal is not None else 'Local acceptance with fixture ' + self.url + '/b') + ";document.getElementById('composer').dispatchEvent(new Event('submit',{cancelable:true}))")

    def state(self, value):
        return any(f['type'] == 'control_state' and f['payload']['state'] == value for f in self.app.frames)

    def done(self):
        wait_for(lambda: self.ui("!document.getElementById('send').disabled"), 'Run did not finish: ' + str(self.timeline()))
        assert not [f for f in self.app.frames if f['type'] == 'error'], self.timeline()

    def timeline(self):
        return [dict(tool=f['payload']['tool'], success=f['payload']['result']['success'], error=f['payload']['result'].get('error')) for f in self.app.frames if f['type'] == 'timeline_step']

    def report(self):
        contexts = []
        for messages in self.provider.calls:
            payloads = page_payloads(messages)
            text = json.dumps([dict(role=m.role, content=m.content) for m in messages], ensure_ascii=False, separators=(',', ':'))
            current = payloads[-1]
            old = 'BODY_CANARY_A' if current['url'].endswith('/b') else 'BODY_CANARY_B'
            contexts.append(dict(chars=len(text), full_states=len(payloads), old_body_occurrences=text.count(old), page='B' if current['url'].endswith('/b') else 'A'))
        tabs = [f['payload']['result']['data']['tabs'] for f in self.app.frames if f['type'] == 'timeline_step' and 'tabs' in f['payload']['result'].get('data', {})]
        from browser.tab_control import safe_tabs
        report = dict(scenario=self.provider.scenario, timeline=self.timeline(), contexts=contexts,
                      tab_metadata_chars=[len(json.dumps(safe_tabs(rows), separators=(',', ':'))) for rows in tabs])
        print('PHASE3C_ACCEPTANCE ' + json.dumps(report, separators=(',', ':')))
        assert all(x['old_body_occurrences'] == 0 for x in contexts), report
        assert all(x['full_states'] == 1 for x in contexts), report
        return report


def test_real_chrome_tab_isolation_and_stale_recovery(browser_factory):
    b = browser_factory('tabs')
    b.provider.other_tab_id = b.ui('chrome.tabs.create({url:' + json.dumps(b.url + '/b') + ',active:false}).then(t=>t.id)')
    b.start()
    wait_for(lambda: len(b.provider.calls) >= 6, 'Tab sequence did not reach stale-action check')
    # The retired decision is rejected, then the unchanged owner is refreshed.
    # Same-owner stale recovery no longer requires a manual Resume gesture.
    b.done()
    assert b.cdp.evaluate('window.inspected', b.site) == 0
    assert len(b.provider.calls) == 7
    assert b.provider.pages[-1]['snapshot_id'] != 'retired-other-tab-snapshot'
    assert b.provider.pages[-1]['document_id'] == b.provider.a['document_id']
    assert not any(row['tool'] == 'click' and row['success'] for row in b.timeline())
    assert any(x['error'] in {'stale_snapshot', 'target_tab_changed'} for x in b.timeline())
    assert b.provider.pages[-1]['tab_id'] == b.provider.a['tab_id']
    assert all(page['tab_id'] == b.owner_tab_id for page in b.provider.pages)
    assert sum(x['error'] == 'permission_denied' for x in b.timeline()) == 2
    b.report()


@pytest.mark.parametrize("manual_change", ["navigation", "switch"])
def test_real_chrome_pause_manual_navigation_resume(browser_factory, manual_change):
    b = browser_factory('resume')
    b.start()
    wait_for(b.provider.entered.is_set, 'Provider not entered')
    b.click('#pause')
    wait_for(lambda: b.state('paused'), 'Backend did not pause')
    assert b.provider.cancelled.wait(3)
    count = len(b.provider.calls)
    requests = len([f for f in b.app.frames if f['type'] == 'tool_request'])
    if manual_change == 'navigation':
        b.cdp.call('Page.navigate', session=b.site, url=b.url + '/b')
    else:
        target = b.cdp.call('Target.createTarget', url=b.url + '/b')['targetId']
        b.site = b.cdp.attach(target)
        b.cdp.call('Target.activateTarget', targetId=target)
    wait_for(lambda: b.cdp.evaluate('document.title', b.site) == 'Fixture B', 'Manual navigation failed')
    time.sleep(0.2)
    assert len(b.provider.calls) == count
    assert len([f for f in b.app.frames if f['type'] == 'tool_request']) == requests
    b.click('#resume')
    if manual_change == 'switch':
        wait_for(lambda: b.state('waiting'), 'Hidden owner must not adopt the other tab on Resume')
        assert len(b.provider.calls) == count
        b.cdp.call('Target.activateTarget', targetId=b.target)
        b.click('#resume')
    b.done()
    resumed = [f['payload'] for f in b.app.frames if f['type'] == 'tool_request'][requests:]
    assert resumed[0]['tool'] == 'get_page_state' and resumed[0]['arguments']['full'] is True
    assert resumed[0]['reset_target'] is True
    assert b.provider.pages[-1]['url'].endswith('/b' if manual_change == 'navigation' else '/a')
    assert (b.provider.pages[0]['document_id'] != b.provider.pages[-1]['document_id']) is (manual_change == 'navigation')
    assert b.provider.pages[-1]['tab_id'] == b.owner_tab_id
    assert b.provider.pages[0]['snapshot_id'] != b.provider.pages[-1]['snapshot_id']
    assert b.cdp.evaluate('window.inspected + window.saved', b.site) == 0
    b.report()


def test_real_chrome_pause_confirmation_changed_target(browser_factory):
    b = browser_factory('confirm')
    b.start()
    wait_for(lambda: b.ui("document.getElementById('confirmation-dialog').open"), 'Confirmation not shown')
    original = [f for f in b.app.frames if f['type'] == 'confirmation_request'][-1]
    b.click('#confirmation-pause')
    wait_for(lambda: b.state('paused'), 'Backend did not pause confirmation')
    assert b.ui("document.querySelector('[value=allow_once]').disabled")
    assert not b.ui("document.querySelector('[value=cancel]').disabled")
    b.cdp.evaluate("document.getElementById('save').textContent='Delete revised note'", b.site)
    b.click('[value=allow_once]')
    assert b.cdp.evaluate('window.saved', b.site) == 0
    b.click('#confirmation-resume')
    wait_for(lambda: len([f for f in b.app.frames if f['type'] == 'confirmation_request']) == 2, 'Fresh confirmation not issued')
    fresh = [f for f in b.app.frames if f['type'] == 'confirmation_request'][-1]
    assert fresh['id'] != original['id']
    assert fresh['payload']['arguments']['element_id'] != original['payload']['arguments']['element_id']
    assert 'revised' in fresh['payload']['target']
    b.click('[value=cancel]')
    b.done()
    assert b.cdp.evaluate('window.saved', b.site) == 0
    assert not [f for f in b.app.frames if f['type'] == 'tool_request' and f['payload']['tool'] == 'click']
    b.report()


def test_real_chrome_panel_close_cancels_active_provider(browser_factory):
    b = browser_factory('panel_close')
    b.start()
    wait_for(b.provider.entered.is_set, 'Provider not entered')
    count = len([f for f in b.app.frames if f['type'] == 'tool_request'])
    b.cdp.call('Target.closeTarget', targetId=b.panel_target)
    assert b.provider.cancelled.wait(5), 'Panel closure did not cancel the backend run'
    time.sleep(0.2)
    assert len([f for f in b.app.frames if f['type'] == 'tool_request']) == count
    assert b.cdp.evaluate('window.inspected + window.saved', b.site) == 0
    audit = b.tmp_path / 'audit.jsonl'
    wait_for(lambda: audit.exists() and 'side_panel_closed' in audit.read_text(), 'Closure stop reason was not audited')
    b.report()


def test_real_chrome_close_owner_tab_without_confirmation(browser_factory):
    b = browser_factory('close_automatic')
    b.start()
    wait_for(lambda: not any(x['targetId'] == b.target for x in b.cdp.call('Target.getTargets')['targetInfos']), 'Owner tab did not close')
    assert not [f for f in b.app.frames if f['type'] == 'confirmation_request']
    wait_for(lambda: not any(x['targetId'] == b.panel_target for x in b.cdp.call('Target.getTargets')['targetInfos']), 'Closed owner panel was transferred')
    b.report()


def test_real_chrome_panel_close_revokes_pending_content_click(browser_factory):
    b = browser_factory('lease_close')
    b.start()
    wait_for(b.provider.entered.is_set, 'Provider not entered')
    tab = b.provider.a['tab_id']
    # Delay the existing pre-click animation frame in the extension's isolated
    # world, so panel closure deterministically occurs before pointer dispatch.
    b.ui('chrome.scripting.executeScript({target:{tabId:' + str(tab) + '},func:()=>{globalThis.requestAnimationFrame=cb=>{globalThis.__acceptanceFramePending=true;return setTimeout(()=>cb(performance.now()),700)}}})')
    b.provider.release.set()
    wait_for(lambda: b.ui('chrome.scripting.executeScript({target:{tabId:' + str(tab) + '},func:()=>globalThis.__acceptanceFramePending===true}).then(rows=>rows[0].result)'), 'Click did not reach its pending animation frame')
    b.cdp.call('Target.closeTarget', targetId=b.panel_target)
    time.sleep(0.9)
    assert b.cdp.evaluate('window.inspected', b.site) == 0
    audit = b.tmp_path / 'audit.jsonl'
    wait_for(lambda: audit.exists() and 'side_panel_closed' in audit.read_text(), 'Closure not audited')
    b.report()



def test_real_chrome_resume_outside_scope_waits_without_dom_access(browser_factory):
    b = browser_factory('resume')
    b.start()
    wait_for(b.provider.entered.is_set, 'Provider not entered')
    b.click('#pause')
    wait_for(lambda: b.state('paused'), 'Backend did not pause')
    outside = b.cdp.call('Target.createTarget', url='about:blank')['targetId']
    b.cdp.call('Target.activateTarget', targetId=outside)
    outside_session = b.cdp.attach(outside)
    time.sleep(0.1)
    b.click('#resume')
    wait_for(lambda: b.state('waiting'), 'Outside-scope Resume did not wait')
    assert len(b.provider.calls) == 1
    assert any(x['tool'] == 'get_page_state' and x['error'] == 'permission_denied' for x in b.timeline())
    # Header shows a generic scope message and the product has not installed its
    # DOM registry on the unsupported tab.
    assert b.ui("document.getElementById('page-title').textContent") == 'Fixture A'
    assert b.cdp.evaluate("document.querySelector('[data-agent-id]')===null", outside_session)
    count = len([f for f in b.app.frames if f['type'] == 'tool_request'])
    time.sleep(0.2)
    assert len([f for f in b.app.frames if f['type'] == 'tool_request']) == count
    b.cdp.call('Target.activateTarget', targetId=b.target)
    time.sleep(0.1)
    b.click('#resume')
    b.done()
    assert b.provider.pages[-1]['tab_id'] == b.provider.a['tab_id']
    b.report()


def test_toolbar_grants_arbitrary_public_origin_without_persistent_permission(browser_factory):
    import base64
    b = browser_factory('summary')
    public_url = 'https://public-fixture.example/guide'
    html = b'<html><title>Public guide</title><body><main>The public library opens at nine.</main></body></html>'
    def fulfill(event):
        if event.get('method') == 'Fetch.requestPaused' and event.get('sessionId') == b.site:
            b.cdp.call('Fetch.fulfillRequest', session=b.site, requestId=event['params']['requestId'],
                responseCode=200, responseHeaders=[{'name':'Content-Type','value':'text/html'}],
                body=base64.b64encode(html).decode())
    b.cdp.event_handlers.append(fulfill)
    b.cdp.call('Fetch.enable', session=b.site, patterns=[{'urlPattern':public_url+'*','requestStage':'Request'}])
    b.cdp.call('Page.navigate', session=b.site, url=public_url)
    wait_for(lambda: b.cdp.evaluate('document.readyState', b.site) == 'complete', 'Public fixture not ready')
    assert not b.ui("chrome.permissions.contains({origins:['https://public-fixture.example/*']})")
    targets = b.cdp.call('Target.getTargets', filter=[{'type':'tab','exclude':False}])['targetInfos']
    matches = [row for row in targets if row.get('url') == public_url]
    assert len(matches) == 1
    b.cdp.call('Extensions.triggerAction', id=b.extension, targetId=matches[0]['targetId'])
    assert b.ui('chrome.scripting.executeScript({target:{tabId:'+str(b.owner_tab_id)+'},func:()=>document.title}).then(r=>r[0].result)') == 'Public guide'
    b.start('Summarize the current page')
    b.done()
    assert b.provider.pages[-1]['url'] == public_url
    assert 'public library' in json.dumps(b.provider.pages).lower()
    assert not [f for f in b.app.frames if f['type'] == 'confirmation_request']
