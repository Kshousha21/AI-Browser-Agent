"""Test-only Chrome launch settings and a fixed, isolated persistent profile."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
from urllib.parse import urlsplit
from .chrome_process import wait_owned_tree
import time

REPO = Path(__file__).absolute().parents[2]
ARTIFACTS = REPO / '.test-artifacts'
PROFILE = ARTIFACTS / 'phase4b-chrome'


def permission_query(url):
    """Check only the synthetic fixture's origin, independent of a fixed site."""
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
        raise ValueError('Expected an HTTP(S) fixture URL')
    host = parsed.hostname
    if ':' in host:
        host = '[' + host + ']'
    # Chrome host permission patterns do not include ports or page paths.
    origin = parsed.scheme + '://' + host + '/*'
    return 'chrome.permissions.contains(' + json.dumps({'origins': [origin]}) + ')'


@contextmanager
def phase4b_profile():
    """Refuse arbitrary/unowned profiles and concurrent use; never edit Chrome data."""
    import msvcrt
    assert PROFILE.resolve().is_relative_to(REPO.resolve()), 'Test profile must stay in this workspace'
    ignored = subprocess.run(['git', 'check-ignore', '-q', str(PROFILE / 'ownership-check')], cwd=REPO)
    assert ignored.returncode == 0, 'The isolated profile must be gitignored'
    ARTIFACTS.mkdir(exist_ok=True)
    with (ARTIFACTS / 'phase4b.lock').open('a+b') as lock:
        # Do not read the locked byte: Windows rejects that before LK_NBLCK
        # can produce our actionable concurrent-use diagnostic.
        lock.seek(0, 2)
        if not lock.tell():
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('The isolated Phase 4B profile is already in use; finish its bootstrap/run first.') from exc
        try:
            marker = ARTIFACTS / 'phase4b-profile-owner.txt'
            expected = 'Phase 4B isolated synthetic acceptance profile\n'
            if not marker.exists():
                assert not PROFILE.exists(), 'Refusing an existing profile without test ownership'
                marker.write_text(expected, encoding='utf-8')
            assert marker.read_text(encoding='utf-8') == expected, 'Unrecognized test profile ownership'
            PROFILE.mkdir(exist_ok=True)
            yield PROFILE
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def chrome_arguments(chrome, profile, debug, *, headed=False):
    # Chromium's rendering-only test switch keeps covered headed windows
    # rendering. Otherwise native panel hide/close animations can stall while
    # the real DOM incorrectly remains visible to the acceptance harness.
    return [str(chrome), *(['--disable-backgrounding-occluded-windows'] if headed else ['--headless=new']),
        '--enable-unsafe-extension-debugging', '--disable-background-networking',
        '--disable-component-update', '--disable-default-apps', '--disable-sync',
        '--no-first-run', '--no-default-browser-check', '--no-pings',
        '--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1',
        '--proxy-server=direct://', '--proxy-bypass-list=*',
        '--remote-debugging-address=127.0.0.1', f'--remote-debugging-port={debug}',
        f'--user-data-dir={profile}', 'about:blank']


def wait_for_owned_chrome(debug, process):
    """A missing endpoint is a failed prerequisite, never a skipped test."""
    from .chrome_dom_support import _json_request
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError('Owned Chrome exited before its endpoint became ready')
        try:
            return _json_request(debug, '/json/version')
        except OSError:
            time.sleep(0.05)
    raise AssertionError('Owned Chrome endpoint readiness timed out')


def close_test_chrome(cdp, process):
    if cdp is not None and process.poll() is None:
        try:
            cdp.call('Browser.close')
        except Exception:
            pass  # Closing the owned browser may close CDP before its reply.
    if cdp is not None:
        cdp.socket.close()
    wait_owned_tree(process)


def installed_test_extension(cdp):
    """Find only our built extension through Chrome's extension-list API."""
    expected = (REPO / 'extension/dist').resolve()
    matches = [extension for extension in cdp.call('Extensions.getExtensions')['extensions']
               if Path(extension['path']).resolve() == expected]
    assert len(matches) <= 1, 'Multiple copies of the test extension were reported'
    if not matches:
        return None
    assert matches[0]['enabled'], 'Enable the test extension through the normal Chrome UI'
    return matches[0]['id']
