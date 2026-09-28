"""Bounded, identity-checked startup for the owned synthetic Chrome harness.

Diagnostics deliberately contain categories, booleans and timings, never page
text, arbitrary URLs, Chrome settings, credentials or exception messages.
"""
from collections import Counter
import json
import hashlib
from functools import wraps
import os
import re
import time

OPTIONS = 'options/index.html'
WORKER = 'service-worker.js'


def diagnosed(function):
    @wraps(function)
    def checked(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        except Exception as error:
            if self.failure_reported:
                raise
            name = type(error).__name__
            allowed = {'AssertionError', 'TimeoutError', 'ConnectionClosedError',
                       'ConnectionClosedOK', 'ConnectionResetError', 'OSError', 'RuntimeError'}
            method = getattr(self.cdp, 'last_method', None)
            commands = {'Target.getTargets', 'Target.createTarget', 'Target.closeTarget',
                        'Target.attachToTarget', 'Target.detachFromTarget', 'Runtime.evaluate',
                        'Runtime.enable', 'Page.enable', 'SystemInfo.getProcessInfo'}
            code = None
            if isinstance(error, AssertionError) and error.args and isinstance(error.args[0], tuple):
                detail = error.args[0]
                if len(detail) == 2 and isinstance(detail[1], dict) and type(detail[1].get('code')) is int:
                    code = detail[1]['code']
            self.command_failure = dict(exception=name if name in allowed else 'other',
                                        method=method if method in commands else 'other', code=code)
            if name in {'ConnectionClosedError', 'ConnectionClosedOK'}:
                cause = error.__cause__
                cause_name = type(cause).__name__ if cause is not None else None
                self.command_failure['transport'] = dict(
                    received_code=getattr(getattr(error, 'rcvd', None), 'code', None),
                    sent_code=getattr(getattr(error, 'sent', None), 'code', None),
                    cause=cause_name if cause_name in {'EOFError', 'OSError', 'TimeoutError',
                        'ConnectionResetError', 'ConnectionAbortedError'} else 'other' if cause else None)
            self.report('startup_command_failed')
            raise AssertionError('Chrome startup failed at ' + self.stage) from None
    return checked


def path_kind(path):
    if str(path).startswith('\\\\'):
        return 'UNC'
    if os.name == 'nt':
        import ctypes
        if ctypes.windll.kernel32.GetDriveTypeW(str(path.drive) + '\\') == 4:
            return 'mapped'
    return 'local'


def target_matches(target, extension, resource, kind):
    return (target.get('type') == kind and
            target.get('url') == f'chrome-extension://{extension}/{resource}')


def url_class(url, extension):
    if url == f'chrome-extension://{extension}/{OPTIONS}':
        return 'chrome-extension://EXPECTED_ID/options/...'
    if url == f'chrome-extension://{extension}/{WORKER}':
        return 'chrome-extension://EXPECTED_ID/service-worker'
    if url.startswith('chrome-extension://'):
        return 'other-extension-resource'
    if url.startswith('chrome-error:'):
        return 'chrome-error'
    if url.rstrip('/') == 'chrome://extensions':
        return 'chrome://extensions'
    return 'about:blank' if url == 'about:blank' else 'other'


def options_expression(extension):
    expected = json.dumps(f'chrome-extension://{extension}/{OPTIONS}')
    identifier = json.dumps(extension)
    return ('({url:location.href===' + expected + ',runtime:globalThis.chrome?.runtime?.id===' + identifier +
            ',body:!!document.body,complete:document.readyState==="complete",'
            'form:!!document.getElementById("grant") && !!document.getElementById("form"),'
            'initialized:!!document.getElementById("max-chars")?.value,'
            'storage:!!globalThis.chrome?.storage?.local?.get,'
            'error:location.href.startsWith("chrome-error:")})')


def options_ready(state):
    return isinstance(state, dict) and all(state.get(key) is True for key in
        ('url', 'runtime', 'body', 'complete', 'form', 'initialized', 'storage')) and not state.get('error')


class ChromeStartup:
    def __init__(self, cdp, process, profile, repo, *, clock=time.monotonic, pause=time.sleep, started=None, persistent=True):
        self.cdp, self.process, self.profile, self.repo = cdp, process, profile, repo
        self.clock, self.pause = clock, pause
        self.started = clock() if started is None else started
        self.persistent = persistent
        self.timings = {}
        self.extension = ''
        self.js_errors = 0
        self.stage = 'browser_identity'
        self.options_state = {}
        self.failure_reported = False
        self.command_failure = None
        self.sessions = set()
        self.cdp.event_handlers.append(self.event)
        self.build = json.loads((repo / 'extension/dist/build-info.json').read_text(encoding='utf-8'))

    def event(self, event):
        if event.get('sessionId') in self.sessions and event.get('method') == 'Runtime.exceptionThrown':
            self.js_errors += 1

    def wait(self, stage, predicate, timeout=15):
        self.stage = stage
        started = self.clock()
        while self.clock() - started < timeout:
            if self.process.poll() is not None:
                self.fail('owned_browser_exited')
            result = predicate()
            if result:
                self.timings[stage] = round(self.clock() - started, 4)
                return result
            self.pause(0.05)
        self.fail('stage_timeout')

    def targets(self):
        return self.cdp.call('Target.getTargets')['targetInfos']

    @diagnosed
    def bind_process(self):
        processes = self.cdp.call('SystemInfo.getProcessInfo')['processInfo']
        if not any(p['type'] == 'browser' and p['id'] == self.process.pid for p in processes):
            self.fail('endpoint_not_owned')
        self.timings['endpoint'] = round(self.clock() - self.started, 4)

    @diagnosed
    def clean_restored_pages(self):
        self.stage = 'restored_page_cleanup'
        # Only this newly spawned, identity-checked, exclusively locked TEST
        # browser. Keep one new blank target so closing restored tabs cannot exit.
        guard = self.cdp.call('Target.createTarget', url='about:blank', newWindow=True)['targetId']
        self.wait('cleanup_guard_target', lambda: any(
            t['targetId'] == guard and t.get('type') == 'page' for t in self.targets()))
        session = self.cdp.attach(guard)
        try:
            self.wait('cleanup_guard_ready', lambda: self.cdp.evaluate(
                'document.readyState === "complete" && location.href === "about:blank"', session))
        finally:
            self.cdp.call('Target.detachFromTarget', sessionId=session)
        # Restoration may create targets while the guard loads. Inventory after
        # its real readiness, and never include the guard in the close set.
        self.stage = 'restored_page_cleanup'
        old = [t['targetId'] for t in self.targets()
               if t.get('type') == 'page' and t['targetId'] != guard]
        for target in old:
            self.cdp.call('Target.closeTarget', targetId=target)
        return set(old)

    @diagnosed
    def open_options(self, extension, *, phase='options'):
        self.extension = extension
        self.stage = phase + '_target_inventory'
        before = {t['targetId'] for t in self.targets()}
        self.stage = phase + '_create_target'
        target = self.cdp.call('Target.createTarget',
            url=f'chrome-extension://{extension}/{OPTIONS}')['targetId']
        if target in before:
            self.fail('stale_target_id')
        self.wait(phase + '_target', lambda: any(t['targetId'] == target and
            target_matches(t, extension, OPTIONS, 'page') for t in self.targets()))
        session = self.cdp.attach(target)
        self.sessions.add(session)
        def ready():
            self.options_state = self.cdp.evaluate(options_expression(extension), session)
            if self.options_state.get('error'):
                self.fail('options_navigation_error')
            return options_ready(self.options_state)
        self.wait(phase + '_context', ready)
        # Read one deliberately unused TEST key, never actual stored settings.
        verified = self.cdp.evaluate('(async()=>{await chrome.storage.local.get("__harness_readiness__");'
            'return chrome.runtime.id===' + json.dumps(extension) + '})()', session)
        if verified is not True:
            self.fail('storage_context_mismatch')
        return target, session

    @diagnosed
    def reload(self, extension):
        self.extension = extension
        target, session = self.open_options(extension, phase='reload_options')
        old_workers = {t['targetId'] for t in self.targets() if
            target_matches(t, extension, WORKER, 'service_worker')}
        self.cdp.evaluate('setTimeout(()=>chrome.runtime.reload(),0);true', session)
        self.wait('reload_unload', lambda: target not in {t['targetId'] for t in self.targets()})
        # Old Options disappearing means UNLOAD, not completion. Do not navigate
        # another extension page until the replacement worker can execute code.
        worker = self.wait('reload_worker', lambda: next((t['targetId'] for t in self.targets()
            if t['targetId'] not in old_workers and target_matches(t, extension, WORKER, 'service_worker')), None))
        worker_session = self.cdp.call('Target.attachToTarget', targetId=worker, flatten=True)['sessionId']
        self.sessions.add(worker_session)
        self.cdp.call('Runtime.enable', session=worker_session)
        self.wait('worker_runtime', lambda: self.cdp.evaluate(
            'globalThis.chrome?.runtime?.id===' + json.dumps(extension), worker_session))
        # Verify THIS loaded generation, not just dist on disk. Extension-local
        # fetch only; never an external request or confidential content.
        self.wait('worker_build', lambda: self.cdp.evaluate(
            '(async()=>{const r=await fetch(chrome.runtime.getURL("build-info.json"),{cache:"no-store"});'
            'const b=await r.json();return Object.entries(' + json.dumps(self.build) +
            ').every(([k,v])=>b[k]===v)})()', worker_session))
        self.wait('worker_guard_clean', lambda: self.cdp.evaluate(
            'chrome.declarativeNetRequest.getSessionRules().then(r=>r.length===0)', worker_session))
        self.cdp.call('Target.detachFromTarget', sessionId=worker_session)

    def report(self, failure=None):
        self.failure_reported = self.failure_reported or failure is not None
        try:
            targets = self.targets() if self.process.poll() is None else []
            version = self.cdp.call('Browser.getVersion').get('product', '') if targets else ''
        except Exception:
            targets, version = [], ''
        result = dict(stage=self.stage, failure=failure, command_failure=self.command_failure,
            elapsed=round(self.clock()-self.started, 4),
            timings=self.timings, process_alive=self.process.poll() is None, process_exit=self.process.poll(),
            browser=version if re.fullmatch(r'Chrome/[0-9.]+', version) else 'unknown',
            target_count=len(targets), target_types=dict(Counter(t.get('type') if t.get('type') in
                {'page','service_worker','background_page','iframe','other'} else 'other' for t in targets)),
            target_classes=dict(Counter(url_class(t.get('url',''), self.extension) for t in targets)),
            expected_worker=any(target_matches(t,self.extension,WORKER,'service_worker') for t in targets),
            options={k: self.options_state.get(k) is True for k in
                ('url','runtime','body','complete','form','initialized','storage','error')},
            js_error_count=self.js_errors, profile_lock_held=self.persistent,
            extension_identity_hash=hashlib.sha256(self.extension.encode()).hexdigest()[:12],
            profile_kind='mapped-network' if str(self.profile.resolve()).startswith('\\\\') else 'local-test',
            load_path_kind=path_kind(self.repo),
            build_commit=self.build.get('git_commit') if re.fullmatch(r'[a-f0-9]{7,40}',str(self.build.get('git_commit'))) else None,
            protocol_version=self.build.get('protocol_version'), protocol_revision=self.build.get('protocol_revision'))
        print('CHROME_STARTUP ' + json.dumps(result, separators=(',', ':')), flush=True)
        return result

    def fail(self, category):
        self.report(category)
        raise AssertionError('Chrome startup: ' + category + ' at ' + self.stage)
