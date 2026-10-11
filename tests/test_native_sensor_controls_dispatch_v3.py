"""Operational recovery must neither rerun trials nor accept changed evidence."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/bench' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

module = load('native_sensor_controls_dispatch_v3')
transport = load('native_sensor_controls_transport')

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return module.digest(path)

@pytest.fixture
def plan(tmp_path, monkeypatch):
    root = tmp_path / 'share'
    campaign, control = root / 'native-sensor-controls-v2', root / 'dispatch'
    campaign.mkdir(parents=True)
    control.mkdir()
    monkeypatch.setattr(module, 'SHARED_ROOT', root)
    monkeypatch.setattr(module, 'CAMPAIGN', campaign)
    monkeypatch.setattr(module, '__file__', str(control / 'native_sensor_controls_dispatch_v3.py'))
    monkeypatch.setattr(module, 'validate_recovery', lambda p: None)
    names = (*module.HELPERS, 'native_sensor_controls_dispatch_v3.py', 'worker.sbatch')
    sources = {}
    for name in names:
        (control / name).write_text(name)
        sources[name] = module.digest(control / name)
    jobs, expected = [], []
    for seed in (730000, 730001):
        for condition in range(10):
            name = f'{seed}-{condition}'
            jobs.append(dict(name=name, kind='run', args=['--seed', str(seed), '--condition', str(condition)]))
            expected.append(dict(name=name, seed=seed, expected_fields=dict(phase='development', expected_declared_cells=10)))
    write(campaign / 'protocol.json', dict(jobs=jobs))
    (campaign / 'frozen-campaign.tar.gz').write_bytes(b'frozen')
    intake = dict(protocol_sha256=module.digest(campaign/'protocol.json'), archive_sha256=module.digest(campaign/'frozen-campaign.tar.gz'))
    write(campaign / 'intake.json', intake)
    value = dict(schema='bhl-native-sensor-controls-dispatch-v3', campaign_dir=str(campaign),
        max_concurrency=1, reserved_bytes_per_job=16*module.MIB, minimum_free_bytes=8*module.GIB,
        poll_seconds=30, source_sha256=sources, jobs=expected,
        transport=dict(raw_cap_bytes=8*module.GIB, shared_receipt_cap_bytes=16*module.MIB, part_bytes=module.GIB),
        smoke=dict(expected_fields=dict(phase='smoke', episodes=14, execution_gate=True)), **intake)
    path = control / 'plan.json'
    write(path, value)
    return path, value


def test_complete_plan_keeps_original_twenty_cells_and_compact_reserve(plan):
    path, value = plan
    assert module.validate_plan(path, module.digest(path)) == value

@pytest.mark.parametrize('mutation', ['scope', 'cohort', 'seed', 'smoke', 'reserve', 'worker', 'raw_cap'])
def test_changed_execution_contract_fails_closed(plan, mutation):
    path, value = plan
    if mutation == 'scope': value['campaign_dir'] = '/tmp/wrong'
    elif mutation == 'cohort': value['jobs'].pop()
    elif mutation == 'seed': value['jobs'][0]['seed'] = 730001
    elif mutation == 'smoke': value['smoke']['expected_fields']['execution_gate'] = False
    elif mutation == 'reserve': value['minimum_free_bytes'] = 0
    elif mutation == 'worker': (path.parent / 'worker.sbatch').write_text('changed')
    elif mutation == 'raw_cap': value['transport']['raw_cap_bytes'] *= 2
    write(path, value)
    with pytest.raises(ValueError): module.validate_plan(path, module.digest(path))

@pytest.fixture
def recovery(tmp_path, monkeypatch):
    root = tmp_path / 'share'; campaign = root / 'campaign'; work = root / 'recovery'
    monkeypatch.setattr(module, 'SHARED_ROOT', root)
    monkeypatch.setattr(module, 'CAMPAIGN', campaign)
    name = 'native-controls-v2-s730000-c04-20261011'; job_id = '21776269'
    original = campaign / f'{name}-{job_id}'
    write(original/'launch.json', {'archive_sha256': 'a'*64})
    completion = dict(status='INCOMPLETE', exit_status=0, output_bytes=2*module.GIB+200, archive_sha256='a'*64,
        error=f'output exceeds predeclared archive cap ({2*module.GIB+200} bytes)', files={})
    completion_hash = write(original/'completion.json', completion)
    result = dict(status='PASS', phase='development', expected_declared_cells=10, episodes=10)
    result_hash = write(work/'result.json', result)
    inventory_hash = write(work/'inventory.json', {'a': {'bytes':2*module.GIB+200,'sha256':'c'*64}})
    receipt = dict(schema='bhl-native-controls-packaging-recovery-v1', status='COMPLETE', scientific_rerun=False,
        original_job_id=job_id, source_archive_sha256='a'*64, protocol_sha256='b'*64,
        original_completion_sha256=completion_hash, result_sha256=result_hash, inventory_sha256=inventory_hash,
        raw_output_bytes=2*module.GIB+200, original_runtime_exit_status=0, parts=[dict(files=['a'])])
    receipt_hash = write(work/'recovery.json', receipt)
    spec = dict(job_name=name, job_id=job_id, receipt=str(work/'recovery.json'), receipt_sha256=receipt_hash,
        result=str(work/'result.json'), result_sha256=result_hash,
        inventory=str(work/'inventory.json'), inventory_sha256=inventory_hash)
    return dict(recovery=spec, archive_sha256='a'*64, protocol_sha256='b'*64), original


def test_only_packaging_failure_recovers_without_overwriting_original(recovery):
    plan, original = recovery
    old = (original/'completion.json').read_bytes()
    row = module.recovered_row(plan, dict(name=plan['recovery']['job_name']), 'FAILED', '1:0')
    assert row['packaging_recovered'] is True
    assert (original/'completion.json').read_bytes() == old
    assert row['original_scheduler_state'] == 'FAILED'

@pytest.mark.parametrize('change', ['science_exit', 'new_hash', 'rerun', 'duplicate_volume', 'wrong_job'])
def test_recovery_cannot_hide_scientific_failure_or_change_evidence(recovery, change):
    plan, original = recovery
    spec = plan['recovery']; receipt_path = Path(spec['receipt']); receipt = json.loads(receipt_path.read_text())
    if change == 'science_exit':
        completion = json.loads((original/'completion.json').read_text()); completion['exit_status'] = 1
        receipt['original_completion_sha256'] = write(original/'completion.json', completion)
    elif change == 'new_hash':
        Path(spec['result']).write_text('{}')
    elif change == 'rerun': receipt['scientific_rerun'] = True
    elif change == 'duplicate_volume': receipt['parts'].append(dict(files=['a']))
    elif change == 'wrong_job': spec['job_id'] = '999'
    spec['receipt_sha256'] = write(receipt_path, receipt)
    with pytest.raises(ValueError): module.validate_recovery(plan)


def test_unrelated_failed_job_cannot_be_retried_or_recovered(recovery):
    plan, _ = recovery
    with pytest.raises(ValueError, match='no recovery'):
        module.recovered_row(plan, dict(name='different'), 'FAILED', '1:0')


def test_transport_publication_must_match_completed_output(tmp_path):
    expected = dict(sha256='a'*64, bytes=123)
    write(tmp_path/'completion.json', dict(files={'outputs.tar.gz':expected}))
    plan = dict(archive_sha256='b'*64,protocol_sha256='c'*64)
    receipt = dict(schema='bhl-native-transport-result-v1',status='PASS',runtime_exit_status=0,runner_returncode=0,
        source_archive_sha256='b'*64,protocol_sha256='c'*64,original_max_output_mb=2048,operational_max_output_mb=8192,
        publication=dict(schema='bhl-native-raw-publication-v1',**expected))
    write(tmp_path/'remote-result.json',receipt)
    assert module.validate_transport_result(tmp_path,plan,dict(status='PASS')) == receipt['publication']
    receipt['publication']['bytes'] += 1
    write(tmp_path/'remote-result.json',receipt)
    with pytest.raises(ValueError,match='publication differs'):
        module.validate_transport_result(tmp_path,plan,dict(status='PASS'))


def test_recovered_volumes_merge_disjoint_files_and_reject_duplicate(tmp_path):
    inventory, parts = {}, []
    for index, payload in enumerate((b'left',b'right')):
        name=f'{index}/data.json'; archive=tmp_path/f'part{index}.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            info=tarfile.TarInfo(name); info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
        inventory[name]=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        parts.append(dict(asset_name=archive.name,local_verified_mirror=str(archive),bytes=archive.stat().st_size,
            sha256=module.digest(archive),files=[name]))
    inventory_path=tmp_path/'inventory.json';inventory_hash=write(inventory_path,inventory)
    publication=dict(schema='bhl-native-recovered-tar-volumes-v1',parts=parts,inventory=str(inventory_path),
        inventory_sha256=inventory_hash,raw_output_bytes=9)
    target=tmp_path/'output'
    module.materialize(dict(publication=publication),tmp_path/'work',target,transport,SimpleNamespace())
    assert (target/'0/data.json').read_bytes()==b'left'
    assert (target/'1/data.json').read_bytes()==b'right'
    publication['parts']=[parts[0],parts[0]]
    with pytest.raises(ValueError,match='duplicated or changed'):
        module.materialize(dict(publication=publication),tmp_path/'work2',tmp_path/'output2',transport,SimpleNamespace())
