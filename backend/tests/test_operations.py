"""Operational failures must be local, diagnosable, and free of secret values."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pytest
import yaml
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from browser.build_info import PROTOCOL_REVISION, PROTOCOL_MISMATCH, compatible_hello
from browser.config import Settings
from browser.config_check import config_report, scope_checks
from browser.providers import MockProvider
from browser.server import create_app
from .test_product_scope import node_check
from .test_ws_auth import connect, authenticate, send, EXTENSION_ORIGIN

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from operational import dangerous_tracked, port_status, repository_root
import doctor
from test_runner import selection, Summary

CANARY = "synthetic-private-value-keep-out-of-output"


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"provider": {"active": "mock"}, "providers": {"mock": {"type": "mock", "model": "mock-1"}}, "auth": {"token": CANARY}}))
    return path


def test_config_check_reuses_active_only_loader_without_network(config_file):
    raw = yaml.safe_load(config_file.read_text())
    raw['providers']['unused'] = {'type': 'openai', 'api_key_env': 'UNSET_LIVE_KEY'}
    config_file.write_text(yaml.safe_dump(raw))
    result = config_report(config_file)
    assert result['valid'] and result['active_provider'] == 'mock'
    assert result['backend_token'] == 'PRESENT'
    assert CANARY not in json.dumps(result)


def test_config_check_missing_active_setting_fails(config_file):
    raw = yaml.safe_load(config_file.read_text())
    raw['provider']['active'] = 'openai'
    raw['providers']['openai'] = {'type':'openai', 'model':'local-model', 'api_key_env':'UNSET_LIVE_KEY'}
    config_file.write_text(yaml.safe_dump(raw))
    result = config_report(config_file)
    assert not result['valid'] and 'active-provider' in result['message']
    assert CANARY not in json.dumps(result)


@pytest.mark.parametrize('problem', ['yaml', 'host', 'budget', 'port', 'tls'])
def test_invalid_configuration_never_echoes_values(config_file, problem):
    raw = yaml.safe_load(config_file.read_text())
    if problem == 'yaml':
        config_file.write_text('secret: [' + CANARY)
    else:
        if problem == 'host': raw['server'] = {'host':CANARY}
        if problem == 'port': raw['server'] = {'port':CANARY}
        if problem == 'budget': raw['agent'] = {'token_budget':CANARY}
        if problem == 'tls':
            raw['providers']['mock']['tls'] = {'client_key_env':'ABSENT_KEY_PATH'}
        config_file.write_text(yaml.safe_dump(raw))
    result = config_report(config_file)
    assert not result['valid']
    assert CANARY not in json.dumps(result)


def test_structure_check_needs_no_workstation_credentials(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({'provider':{'active':'anthropic'}, 'providers':{'anthropic':{'type':'anthropic','model':'configured-model','api_key_env':'UNSET_KEY'}}}))
    assert config_report(path, structure=True)['valid']
    assert not config_report(path)['valid']
    assert not (tmp_path / '.env').exists()


def test_config_cli_reports_token_presence_only(config_file):
    result = subprocess.run([sys.executable, '-B', '-m', 'browser.config_check', '--config', str(config_file)], cwd=ROOT/'backend', capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert 'PRESENT' in result.stdout and CANARY not in result.stdout + result.stderr
    config_file.write_text('provider: {active: mock}\nproviders: {mock: {type: mock, model: mock-1}}\n')
    assert config_report(config_file)['backend_token'] == 'MISSING'


@pytest.mark.parametrize('payload', [None, {}, {'protocol_version':1}, {'protocol_version':0,'protocol_revision':PROTOCOL_REVISION}, {'protocol_version':1,'protocol_revision':1}, {'protocol_version':1,'protocol_revision':'2'}, {'protocol_version':True,'protocol_revision':2}])
def test_authenticated_protocol_mismatch_closes_before_run(settings, payload):
    provider = MockProvider()
    with TestClient(create_app(settings, provider)) as client:
        with connect(client, origin=EXTENSION_ORIGIN) as ws:
            authenticate(ws)
            send(ws, 'hello', payload)
            error = json.loads(ws.receive_text())
            assert error['payload']['code'] == 'protocol_mismatch'
            assert error['payload']['message'] == PROTOCOL_MISMATCH
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_text()
            assert exc.value.code == 1002
    assert provider.calls == []


def test_protocol_match_and_old_backend_extension_check():
    correct = {'protocol_version':1,'protocol_revision':PROTOCOL_REVISION}
    assert compatible_hello(correct)
    actual = node_check("const p=load('../shared/protocol.ts'); console.log(JSON.stringify(data.map(x=>p.compatibleHello(x))));", [correct, {'protocol_version':1}, {'protocol_version':1,'protocol_revision':0}, None])
    assert actual == [True, False, False, False]


def test_health_is_safe_and_does_not_claim_live_readiness():
    settings = Settings(auth={'token':CANARY}, provider={'name':'mock','model':CANARY,'headers':{'X-Private':CANARY}})
    app = create_app(settings, MockProvider(model=CANARY))
    with TestClient(app) as client:
        body = client.get('/health').json()
    assert set(body) == {'status','service','backend_version','protocol_version','protocol_revision','provider','active_provider','provider_ready','product_scope'}
    assert body['provider_ready'] and 'read_only_default' not in body
    assert body['active_provider'] == 'mock' and body['protocol_revision'] == PROTOCOL_REVISION
    assert CANARY not in json.dumps(body) and 'model' not in body


def test_scope_diagnostics_use_existing_typescript_and_python_interpretations():
    assert scope_checks()
    result = subprocess.run([shutil.which('node'), str(ROOT/'tools/build-metadata.mjs'), '--scope'], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0 and json.loads(result.stdout)['scope_parity']


def test_incomplete_structure_rejected_and_snapshot_accepted(tmp_path):
    nested = tmp_path / 'snapshot'
    for item in ['backend', 'extension', 'shared']:
        (nested / item).mkdir(parents=True)
    with pytest.raises(ValueError): repository_root(nested)
    (nested / 'shared/build-contract.json').write_text('{}')
    assert repository_root(nested) == nested.absolute()
    assert not (nested / '.git').exists()


def test_tracked_secret_detection_uses_names_without_reading_contents():
    unsafe = ['backend/.env','.env.lab','.agent_token','private/client.key','private/client.pfx','private/client.pem','recon/page.html','backend/audit/actions.jsonl']
    safe = ['backend/.env.example','README.md','backend/tests/test_operations.py']
    assert dangerous_tracked(unsafe + safe) == unsafe


@pytest.mark.parametrize('change', ['source', 'output', 'protocol'])
def test_build_metadata_fallback_and_staleness_without_git(tmp_path, change):
    for path in ['extension/src','extension/public','extension/dist/content','shared','tools']:
        (tmp_path/path).mkdir(parents=True)
    for path in ['extension/package.json','extension/package-lock.json','extension/tsconfig.json','extension/vite.config.ts','extension/vite.content.config.ts']:
        (tmp_path/path).write_text('{}')
    (tmp_path/'extension/src/local.ts').write_text('original')
    (tmp_path/'extension/dist/content/extract.js').write_text('built')
    (tmp_path/'extension/dist/manifest.json').write_text('{}')
    shutil.copy2(ROOT/'tools/build-metadata.mjs',tmp_path/'tools/build-metadata.mjs')
    shutil.copy2(ROOT/'shared/build-contract.json',tmp_path/'shared/build-contract.json')
    env = {**os.environ,'PATH':''}
    node = shutil.which('node')
    args = [node,str(tmp_path/'tools/build-metadata.mjs')]
    built = subprocess.run([*args,'--write'],env=env,capture_output=True,text=True,timeout=15)
    assert built.returncode == 0, built.stderr
    assert json.loads((tmp_path/'extension/dist/build-info.json').read_text())['git_commit'] is None
    assert subprocess.run([*args,'--check'],env=env,capture_output=True).returncode == 0
    if change == 'source': (tmp_path/'extension/src/local.ts').write_text('changed')
    elif change == 'output': (tmp_path/'extension/dist/content/extract.js').write_text('changed')
    else: (tmp_path/'shared/build-contract.json').write_text('{"application_version":"0.1.0","protocol_version":1,"protocol_revision":99}')
    checked = subprocess.run([*args,'--check'],env=env,capture_output=True,text=True)
    assert checked.returncode == 1
    if change == 'protocol':
        diagnostic = json.loads(checked.stdout)
        assert not diagnostic['compatible'] and diagnostic['protocol_revision'] == PROTOCOL_REVISION


def powershell(script, env):
    executable = shutil.which('powershell.exe')
    if not executable:
        pytest.skip('Windows PowerShell is required for script acceptance')
    return subprocess.run([executable,'-NoProfile','-ExecutionPolicy','RemoteSigned','-File',str(ROOT/'scripts'/script)], env=env, capture_output=True, text=True, timeout=45)


def test_doctor_script_read_only_presence_checks(config_file):
    before = config_file.read_bytes()
    result = powershell('doctor.ps1', {**os.environ,'BROWSER_CONFIG':str(config_file)})
    from check_requirements import check
    ready = check((ROOT / "backend/requirements.txt").read_text())["ready"]
    assert result.returncode == (0 if ready else 2), result.stdout + result.stderr
    assert ('LOCAL READY' if ready else 'NOT READY') in result.stdout and 'PRESENT' in result.stdout
    assert 'LIVE PROVIDER NOT CONFIGURED' in result.stdout
    assert CANARY not in result.stdout + result.stderr
    assert config_file.read_bytes() == before


def test_startup_script_refuses_non_loopback_configuration(config_file):
    raw = yaml.safe_load(config_file.read_text()); raw['server']={'host':'0.0.0.0'}
    config_file.write_text(yaml.safe_dump(raw))
    result = powershell('start-backend.ps1', {**os.environ,'BROWSER_CONFIG':str(config_file)})
    assert result.returncode != 0 and 'Configuration is not ready' in ' '.join(result.stderr.split())
    assert CANARY not in result.stdout + result.stderr
    assert port_status('0.0.0.0', 8765) == 'non_loopback'


def test_startup_occupied_port_does_not_kill_listener(config_file):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0)); listener.listen()
        port = listener.getsockname()[1]
        raw=yaml.safe_load(config_file.read_text());raw['server']={'host':'127.0.0.1','port':port}
        config_file.write_text(yaml.safe_dump(raw))
        assert port_status('127.0.0.1',port) == 'occupied_or_unavailable'
        result=powershell('start-backend.ps1',{**os.environ,'BROWSER_CONFIG':str(config_file)})
        assert result.returncode != 0 and 'occupied or unavailable' in ' '.join(result.stderr.split())
        assert CANARY not in result.stdout + result.stderr
        with socket.create_connection(('127.0.0.1',port),timeout=1):
            connection,_=listener.accept();connection.close()
    assert port_status('127.0.0.1',port) == 'free'


def test_test_runner_reports_skips_and_separates_browser_selection():
    from types import SimpleNamespace
    summary = Summary()
    for when, passed, failed, skipped in [('call',True,False,False),('call',False,True,False),('setup',False,False,True),('setup',False,True,False)]:
        summary.pytest_runtest_logreport(SimpleNamespace(when=when,passed=passed,failed=failed,skipped=skipped))
    assert summary.counts == dict(passed=1,failed=1,skipped=1,errors=1)
    assert len(selection('Browser')) == 2 and len(selection('Full')) == 1
    assert len([x for x in selection('Fast') if x.startswith('--ignore=')]) == 2
    assert {x.removeprefix('--ignore=') for x in selection('Fast') if x.startswith('--ignore=')} == set(selection('Browser'))



def test_requirement_versions_report_missing_and_incompatible_without_values(monkeypatch):
    import check_requirements
    from importlib.metadata import PackageNotFoundError
    def installed(name):
        if name == 'missing-package': raise PackageNotFoundError(name)
        return '1.0.0'
    monkeypatch.setattr(check_requirements, 'version', installed)
    assert check_requirements.check('package>=2.0\nmissing-package>=1.0\n') == dict(ready=False, missing=1, incompatible=1)
    assert check_requirements.check('package>=1.0\n')['ready']


def test_doctor_snapshot_skips_git_and_never_queries_parent(tmp_path, monkeypatch):
    for name in ('backend', 'extension', 'shared'):
        (tmp_path / name).mkdir()
    (tmp_path / 'shared/build-contract.json').write_text('{}')
    commands = []
    def command(args, **kwargs):
        commands.append(args)
        return None
    monkeypatch.setattr(doctor, 'command', command)
    monkeypatch.setattr(doctor.shutil, 'which', lambda name: None)
    result = doctor.diagnose(tmp_path)
    assert result['checks'][0]['status'] == 'OK'
    assert next(row for row in result['checks'] if row['name'] == 'Git publication checks')['status'] == 'WARN'
    assert not any(args[0] == 'git' for args in commands)
