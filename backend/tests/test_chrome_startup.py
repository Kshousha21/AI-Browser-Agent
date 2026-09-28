"""Deterministic startup, identity, privacy and owned-lifecycle regressions."""
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from .chrome_startup import ChromeStartup, OPTIONS, WORKER, options_ready, target_matches, url_class
from .chrome_profile import chrome_arguments, close_test_chrome, installed_test_extension, phase4b_profile
from .chrome_process import launch_test_chrome, wait_owned_tree

EXTENSION = 'a' * 32


@pytest.mark.parametrize('url, origin', [
    ('https://portal.example/page?query=fixture', 'https://portal.example/*'),
    ('https://another.example/docs', 'https://another.example/*'),
    ('http://127.0.0.1:8765/page', 'http://127.0.0.1/*'),
    ('http://[::1]:8765/page', 'http://[::1]/*'),
])
def test_fixture_permission_query_uses_its_own_origin(url, origin):
    from .chrome_profile import permission_query
    query = permission_query(url)
    assert query == 'chrome.permissions.contains(' + json.dumps({'origins': [origin]}) + ')'


@pytest.mark.parametrize('url', ['chrome://settings', 'file:///fixture.html', 'relative/path'])
def test_fixture_permission_query_rejects_non_web_urls(url):
    from .chrome_profile import permission_query
    with pytest.raises(ValueError, match='HTTP'):
        permission_query(url)


def target(identifier, resource=OPTIONS, extension=EXTENSION, kind='page'):
    return dict(targetId=identifier, type=kind, url=f'chrome-extension://{extension}/{resource}')


class Clock:
    now = 0
    def __call__(self): return self.now
    def sleep(self, duration): self.now += duration


class Cdp:
    def __init__(self):
        self.event_handlers = []
        self.rows = [dict(targetId='unrelated', type='page', url='about:blank'),
            target('stale'), target('old-worker', WORKER, kind='service_worker')]
        self.state = dict.fromkeys(('url','runtime','body','complete','form','initialized','storage'), True)
        self.state['error'] = False
        self.commands, self.expressions = [], []
        self.number, self.reload_tick = 0, 0
        self.storage = True
        self.build_ok = True
        self.guard_clean = True
        self.pid = 321

    def call(self, method, **params):
        self.commands.append((method, params))
        if method == 'Target.getTargets':
            if self.reload_tick:
                self.reload_tick += 1
                if self.reload_tick == 4:
                    self.rows = [r for r in self.rows if r['targetId'] != 'old-worker']
                    self.rows.append(target('new-worker', WORKER, kind='service_worker'))
            return dict(targetInfos=list(self.rows))
        if method == 'Target.createTarget':
            self.number += 1
            identifier = f'new-{self.number}'
            self.rows.append(dict(targetId=identifier, type='page', url=params['url']))
            return dict(targetId=identifier)
        if method == 'Target.attachToTarget': return dict(sessionId=params['targetId'])
        if method == 'SystemInfo.getProcessInfo': return dict(processInfo=[dict(type='browser', id=self.pid)])
        if method == 'Browser.getVersion': return dict(product='Chrome/152.0.0.0')
        if method == 'Target.closeTarget':
            self.rows = [r for r in self.rows if r['targetId'] != params['targetId']]
        return {}

    def attach(self, identifier):
        self.commands.append(('attach', dict(targetId=identifier)))
        return identifier

    def evaluate(self, expression, session):
        self.expressions.append((expression, session))
        if expression.startswith('({url:'): return self.state
        if 'setTimeout' in expression:
            self.rows = [r for r in self.rows if r['targetId'] != session]
            self.reload_tick = 1
            return True
        if '__harness_readiness__' in expression: return self.storage
        if 'build-info.json' in expression: return self.build_ok
        if 'getSessionRules' in expression: return self.guard_clean
        return True


@pytest.fixture
def harness(tmp_path):
    dist = tmp_path / 'extension/dist'
    dist.mkdir(parents=True)
    (dist / 'build-info.json').write_text(json.dumps(dict(git_commit='a'*12,
        protocol_version=1, protocol_revision=2, output_fingerprint='synthetic-build')))
    clock, cdp = Clock(), Cdp()
    process = SimpleNamespace(pid=321, poll=lambda: None)
    return ChromeStartup(cdp, process, tmp_path/'profile', tmp_path, clock=clock, pause=clock.sleep)


@pytest.mark.parametrize('row', [dict(type='page',url='about:blank'),
    dict(type='page',url='chrome://extensions/'),
    target('other', extension='b'*32), target('other', resource='sidepanel/index.html'),
    target('other', resource=OPTIONS+'?stale=1'), target('other',kind='service_worker')])
def test_ignores_unrelated_options_targets(row):
    assert not target_matches(row, EXTENSION, OPTIONS, 'page')


def test_new_options_ignores_stale_matching_target(harness):
    identifier, session = harness.open_options(EXTENSION)
    assert identifier == session == 'new-1'
    assert ('attach', {'targetId':'stale'}) not in harness.cdp.commands
    assert any('__harness_readiness__' in e for e, _ in harness.cdp.expressions)
    assert not any('settings' in e for e, _ in harness.cdp.expressions)


def test_reused_target_identifier_rejected(harness, monkeypatch):
    original = harness.cdp.call
    monkeypatch.setattr(harness.cdp, 'call', lambda method, **args:
        {'targetId':'stale'} if method == 'Target.createTarget' else original(method, **args))
    with pytest.raises(AssertionError, match='stale_target_id'):
        harness.open_options(EXTENSION)


@pytest.mark.parametrize('key', ['url','runtime','body','complete','form','initialized','storage'])
def test_options_requires_entire_expected_context(harness, key):
    harness.cdp.state[key] = False
    with pytest.raises(AssertionError, match='stage_timeout'):
        harness.open_options(EXTENSION)
    assert not any('__harness_readiness__' in e for e, _ in harness.cdp.expressions)


def test_error_document_fails_without_blind_retry(harness):
    harness.cdp.state['error'] = True
    with pytest.raises(AssertionError, match='options_navigation_error'):
        harness.open_options(EXTENSION)
    assert harness.clock() == 0


def test_storage_must_resolve_in_expected_context(harness):
    harness.cdp.storage = False
    with pytest.raises(AssertionError, match='storage_context_mismatch'):
        harness.open_options(EXTENSION)


def test_reload_waits_for_new_worker_before_new_options(harness):
    harness.reload(EXTENSION)
    harness.open_options(EXTENSION)
    calls = harness.cdp.commands
    worker = calls.index(('Target.attachToTarget', dict(targetId='new-worker', flatten=True)))
    creations = [i for i, (m, _) in enumerate(calls) if m == 'Target.createTarget']
    assert creations[0] < worker < creations[1]
    assert harness.timings['reload_worker'] > 0
    assert all(args.get('targetId') != 'old-worker' for m,args in calls if m=='Target.attachToTarget')


@pytest.mark.parametrize('flag,stage', [('build_ok','worker_build'),('guard_clean','worker_guard_clean')])
def test_stale_build_or_guard_fails_closed(harness, flag, stage):
    setattr(harness.cdp, flag, False)
    with pytest.raises(AssertionError, match='stage_timeout.*' + stage):
        harness.reload(EXTENSION)


def test_stale_endpoint_rejected_before_target_actions(harness):
    harness.cdp.pid = 999
    with pytest.raises(AssertionError, match='endpoint_not_owned'):
        harness.bind_process()
    assert not any(m == 'Target.createTarget' for m,_ in harness.cdp.commands)


def test_restore_cleanup_only_closes_pages_and_keeps_new_blank(harness):
    harness.bind_process()
    old = harness.clean_restored_pages()
    assert old == {'stale','unrelated'}
    assert {r['targetId'] for r in harness.cdp.rows} == {'old-worker','new-1'}


def test_diagnostics_never_echo_unrecognized_content(harness, capsys, monkeypatch):
    canary = 'SECRET_CONTENT_DO_NOT_PRINT'
    harness.cdp.rows.append(dict(type=canary, url='https://private.invalid/'+canary, targetId=canary))
    harness.options_state = {'content':canary}
    harness.report('synthetic_failure')
    text = capsys.readouterr().out
    assert canary not in text and 'private.invalid' not in text and str(harness.repo) not in text
    assert 'synthetic_failure' in text and 'target_count' in text
    assert url_class('chrome-error://chromewebdata/', EXTENSION) == 'chrome-error'
    harness.failure_reported = False
    original = harness.cdp.call
    def failing(method, **params):
        harness.cdp.last_method = method
        if method == 'Target.closeTarget':
            raise AssertionError((method, dict(code=-32000, message=canary)))
        return original(method, **params)
    monkeypatch.setattr(harness.cdp, 'call', failing)
    with pytest.raises(AssertionError, match='restored_page_cleanup'):
        harness.clean_restored_pages()
    text = capsys.readouterr().out
    assert canary not in text and 'private.invalid' not in text
    assert harness.command_failure == dict(exception='AssertionError', method='Target.closeTarget', code=-32000)


def test_shutdown_waits_for_descendants_after_parent_exit():
    events = []
    remaining = iter([2,1,0])
    job = SimpleNamespace(active=lambda: next(remaining), close=lambda: events.append('tree_closed'))
    process = SimpleNamespace(test_job=job, poll=lambda: 0)
    clock = Clock()
    wait_owned_tree(process, clock=clock, pause=clock.sleep)
    assert clock() == .1 and events == ['tree_closed'] and process.test_job is None


def test_transport_failure_diagnostics_keep_codes_not_reasons(harness, capsys, monkeypatch):
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close
    canary = 'PRIVATE_CLOSE_REASON'
    error = ConnectionClosedError(Close(1001, canary), Close(1002, canary), True)
    error.__cause__ = ConnectionResetError(canary)
    original = harness.cdp.call
    def failing(method, **params):
        harness.cdp.last_method = method
        if method == 'Target.closeTarget':
            raise error
        return original(method, **params)
    monkeypatch.setattr(harness.cdp, 'call', failing)
    with pytest.raises(AssertionError, match='restored_page_cleanup'):
        harness.clean_restored_pages()
    assert harness.command_failure['transport'] == dict(received_code=1001,
        sent_code=1002, cause='ConnectionResetError')
    assert canary not in capsys.readouterr().out


@pytest.mark.parametrize('phase', ['options', 'reload_options'])
def test_options_creation_reset_has_current_stage_and_no_retry(harness, capsys, monkeypatch, phase):
    from websockets.exceptions import ConnectionClosedError
    error = ConnectionClosedError(None, None, None)
    error.__cause__ = ConnectionResetError('PRIVATE_TRANSPORT_DETAIL')
    harness.stage = 'restored_page_cleanup'
    original, attempts = harness.cdp.call, []
    def failing(method, **params):
        harness.cdp.last_method = method
        if method == 'Target.createTarget':
            attempts.append(method)
            raise error
        return original(method, **params)
    monkeypatch.setattr(harness.cdp, 'call', failing)
    with pytest.raises(AssertionError, match=phase + '_create_target'):
        harness.open_options(EXTENSION, phase=phase)
    assert attempts == ['Target.createTarget']
    assert harness.command_failure['method'] == 'Target.createTarget'
    assert harness.command_failure['transport'] == dict(received_code=None, sent_code=None, cause='ConnectionResetError')
    output = capsys.readouterr().out
    assert 'PRIVATE_TRANSPORT_DETAIL' not in output and 'restored_page_cleanup' not in output


@pytest.mark.parametrize('matches', [0, 1, 2])
def test_native_target_diagnostics_preserve_exact_guard_and_privacy(capsys, matches):
    from .native_action_support import native_action
    private_url = 'https://private.invalid/PRIVATE_SYNTHETIC_CANARY'
    calls = []
    class NativeCdp:
        def evaluate(self, expression, session):
            if 'tabs.update' in expression:
                return dict(id=17, url=private_url, active=True, title='PRIVATE_TITLE')
            return private_url
        def call(self, method, **kwargs):
            calls.append(method)
            if method == 'Target.getTargets':
                kind = 'tab' if 'filter' in kwargs else 'page'
                return dict(targetInfos=[dict(targetId=str(i), type=kind, url=private_url) for i in range(matches)])
            return {}
    if matches == 1:
        native_action(NativeCdp(), 'synthetic-session', 17, EXTENSION)
        assert calls.count('Extensions.triggerAction') == 1
    else:
        with pytest.raises(AssertionError, match='Ambiguous synthetic native tab target'):
            native_action(NativeCdp(), 'synthetic-session', 17, EXTENSION)
        assert 'Extensions.triggerAction' not in calls
    output = capsys.readouterr().out
    row = json.loads(output.removeprefix('NATIVE_TAB_TARGET '))
    assert row['matching_targets'] == matches and row['owner_identity'] and row['owner_active']
    assert 'PRIVATE' not in output and 'private.invalid' not in output and 'synthetic-session' not in output


@pytest.mark.parametrize('hidden', ['owner', 'panel'])
def test_layout_diagnostics_preserve_failure_and_hide_untrusted_details(hidden, capsys, monkeypatch):
    from . import test_browser_control_real_chrome as browser
    clock = Clock()
    monkeypatch.setattr(browser.time, 'monotonic', clock)
    monkeypatch.setattr(browser.time, 'sleep', clock.sleep)
    canary = 'PRIVATE_LAYOUT_DETAILS'
    def evaluate(expression, session):
        if expression.startswith('[innerWidth'):
            return [800, 600, canary if session == hidden else 'visible']
        return dict(owner_active=False, window_focused=False, ignored=canary)
    cdp = SimpleNamespace(evaluate=evaluate)
    with pytest.raises(AssertionError, match='Native owner/panel layout did not settle'):
        browser.wait_native_panel_layout(cdp, 'owner', 'panel')
    output = capsys.readouterr().out
    assert canary not in output
    row = json.loads(output.removeprefix('CHROME_LAYOUT '))
    assert row[hidden] == dict(nonzero=True, visible=False)
    assert row['samples'] > 1 and row['changes'] == 1 and clock() >= 15


def test_shutdown_closes_socket_even_when_parent_exited(monkeypatch):
    from . import chrome_profile
    events = []
    cdp = SimpleNamespace(socket=SimpleNamespace(close=lambda: events.append('socket')))
    process = SimpleNamespace(poll=lambda: 0)
    monkeypatch.setattr(chrome_profile, 'wait_owned_tree', lambda p: events.append('tree'))
    close_test_chrome(cdp, process)
    assert events == ['socket','tree']


def test_real_owned_job_cleans_child_but_not_other_process():
    other = launch_test_chrome([sys.executable,'-c','import time; time.sleep(60)'])
    process = launch_test_chrome([sys.executable,'-c',
        'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); time.sleep(.4)'])
    try:
        process.wait(timeout=5)
        # The Windows venv launcher may itself add descendants.
        assert process.test_job.active() >= 1
        wait_owned_tree(process, timeout=.1)
        assert process.test_job is None and other.poll() is None
    finally:
        if process.test_job is not None: wait_owned_tree(process, timeout=.1)
        wait_owned_tree(other, timeout=.1)


def test_profile_lock_prevents_overlap_then_releases(tmp_path, monkeypatch):
    from . import chrome_profile
    subprocess.run(['git','init','-q',str(tmp_path)], check=True)
    (tmp_path/'.gitignore').write_text('.test-artifacts/\n')
    monkeypatch.setattr(chrome_profile,'REPO',tmp_path)
    monkeypatch.setattr(chrome_profile,'ARTIFACTS',tmp_path/'.test-artifacts')
    monkeypatch.setattr(chrome_profile,'PROFILE',tmp_path/'.test-artifacts/phase4b-chrome')
    with phase4b_profile() as first:
        with pytest.raises(RuntimeError, match='already in use'):
            with phase4b_profile(): pass
    with phase4b_profile() as second:
        assert first == second


def test_installed_identity_stays_stable_across_build_metadata_change(tmp_path, monkeypatch):
    from . import chrome_profile
    monkeypatch.setattr(chrome_profile,'REPO',tmp_path)
    dist = tmp_path/'extension/dist'
    dist.mkdir(parents=True)
    cdp = SimpleNamespace(call=lambda _: dict(extensions=[dict(path=str(dist),id=EXTENSION,enabled=True)]))
    assert installed_test_extension(cdp) == EXTENSION
    (dist/'build-info.json').write_text('{"output_fingerprint":"rebuilt"}')
    assert installed_test_extension(cdp) == EXTENSION


def test_canonical_normal_fixture_has_no_legacy_warning(settings, caplog):
    from .public_document_support import settings as document_settings
    assert settings.provider.name == 'mock'
    assert document_settings().provider.name == 'ollama'
    assert not any('Legacy provider' in r.getMessage() for r in caplog.records)


def test_chrome_paths_do_not_change_execution_directory():
    from .chrome_profile import REPO
    before = Path.cwd()
    profile = REPO/'.test-artifacts/phase4b-chrome'
    command = chrome_arguments(Path('chrome.exe'), profile, 12345)
    assert f'--user-data-dir={profile}' in command and Path.cwd() == before
    assert '--remote-debugging-address=127.0.0.1' in command
    assert not any('load-extension' in value for value in command)
    assert '--disable-backgrounding-occluded-windows' not in command
    headed = chrome_arguments(Path('chrome.exe'), profile, 12345, headed=True)
    assert '--disable-backgrounding-occluded-windows' in headed and '--headless=new' not in headed
    assert f'--user-data-dir={profile}' in headed and '--remote-debugging-address=127.0.0.1' in headed
    assert not any('disable-web-security' in value or 'disable-gesture' in value or 'load-extension' in value for value in headed)


def test_cleanup_captures_pages_restored_while_guard_is_created(harness):
    original = harness.cdp.call

    def late_restore(method, **params):
        result = original(method, **params)
        if method == 'Target.createTarget':
            harness.cdp.rows.append(dict(targetId='late-restored', type='page', url='about:blank'))
        return result

    harness.cdp.call = late_restore
    closed = harness.clean_restored_pages()
    assert {'unrelated', 'stale', 'late-restored'} <= closed
    pages = [row for row in harness.cdp.rows if row['type'] == 'page']
    assert len(pages) == 1 and pages[0]['targetId'] not in closed
    calls = harness.cdp.commands
    assert next(params for method, params in calls if method == 'Target.createTarget')['newWindow'] is True
    assert any('document.readyState' in expression for expression, _ in harness.cdp.expressions)
