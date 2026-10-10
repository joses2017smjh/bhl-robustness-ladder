"""Durable orchestration rejects stale source, partial cohorts and failed runs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/bench/methods_dispatch.py'
spec = importlib.util.spec_from_file_location('methods_dispatch_under_test', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def put(path, data):
    path.write_text(json.dumps(data))
    return {'bytes': path.stat().st_size, 'sha256': module.digest(path)}


def completed(tmp_path, *, result_status='PASS', completion_status='PASS', episodes=30):
    value = {'status': result_status, 'phase': 'development', 'episodes': episodes}
    result = put(tmp_path/'campaign_result.json', value)
    put(tmp_path/'launch.json', {'archive_sha256': 'frozen'})
    completion = {'archive_sha256': 'frozen', 'finished_utc': '2026-10-10T20:00:00Z',
                  'exit_status': 0, 'error': None, 'status': completion_status,
                  'files': {'campaign_result.json': result}}
    put(tmp_path/'completion.json', completion)
    return completion


def test_complete_negative_science_is_accepted(tmp_path):
    completed(tmp_path, result_status='NEGATIVE', completion_status='NEGATIVE')
    result, _ = module.checked_result(tmp_path, {'expected_fields': {'episodes': 30, 'phase': 'development'}}, 'frozen')
    assert result['status'] == 'NEGATIVE'


def test_incomplete_scientific_cohort_is_rejected(tmp_path):
    completed(tmp_path, episodes=29)
    with pytest.raises(ValueError, match='cohort mismatch'):
        module.checked_result(tmp_path, {'expected_fields': {'episodes': 30}}, 'frozen')


def test_modified_completed_outcomes_are_rejected(tmp_path):
    completed(tmp_path)
    put(tmp_path/'campaign_result.json', {'status': 'PASS', 'episodes': 30, 'phase': 'development', 'invented': True})
    with pytest.raises(ValueError, match='result changed'):
        module.checked_result(tmp_path, {}, 'frozen')


@pytest.mark.parametrize('field,value', [('exit_status', 1), ('error', 'crash'), ('archive_sha256', 'another')])
def test_runtime_failures_never_count_as_completed_science(tmp_path, field, value):
    value_to_write = completed(tmp_path)
    value_to_write[field] = value
    put(tmp_path/'completion.json', value_to_write)
    with pytest.raises(ValueError):
        module.checked_result(tmp_path, {}, 'frozen')


def test_dispatcher_does_not_implicitly_resubmit_failed_prior_attempt(tmp_path):
    put(tmp_path/'method-job-submission.json', {'status': 'SUBMISSION_FAILED', 'archive_sha256': 'frozen', 'job_id': None})
    with pytest.raises(ValueError, match='no implicit resubmission'):
        module.submitted_id(tmp_path, {'name': 'method-job'}, 'frozen')


def test_existing_scheduler_id_cannot_be_replaced(tmp_path):
    put(tmp_path/'method-job-submission.json', {'status': 'SUBMITTED', 'archive_sha256': 'frozen', 'job_id': '123'})
    with pytest.raises(ValueError, match='scheduler id changed'):
        module.submitted_id(tmp_path, {'name': 'method-job', 'existing_job_id': '124'}, 'frozen')


def test_frozen_dispatcher_source_tamper_is_rejected(tmp_path):
    (tmp_path/'driver.py').write_text('original')
    plan = {'source_sha256': {'driver.py': module.digest(tmp_path/'driver.py')}}
    module.validate_sources(plan, tmp_path)
    (tmp_path/'driver.py').write_text('changed')
    with pytest.raises(ValueError, match='source changed'):
        module.validate_sources(plan, tmp_path)


def test_parent_path_in_dispatcher_source_inventory_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        module.validate_sources({'source_sha256': {'../escape.py': 'whatever'}}, tmp_path)


def test_state_write_is_atomic_and_records_negative_values(tmp_path):
    path = tmp_path/'state.json'
    module.write_atomic(path, {'status': 'SHARD_PUBLISHED', 'goal_gain': -4})
    assert json.loads(path.read_text())['goal_gain'] == -4
    assert not list(tmp_path.glob('*.temporary'))


def test_global_output_promises_prevent_competing_overcommit(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(module, 'AUTHORIZED_ROOT', tmp_path)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _: SimpleNamespace(free=300*1024**2))
    first = tmp_path/'terrain'; first.mkdir()
    second = tmp_path/'stereo'; second.mkdir()
    accepted, _ = module.reserve_output(first, 'terrain-job', 100*1024**2, 128*1024**2)
    assert accepted
    accepted, space = module.reserve_output(second, 'stereo-job', 100*1024**2, 128*1024**2)
    assert not accepted
    assert space['other_outstanding_promises_bytes'] == 100*1024**2
    module.release_output(first, 'terrain-job')
    accepted, _ = module.reserve_output(second, 'stereo-job', 100*1024**2, 128*1024**2)
    assert accepted


def test_existing_active_job_is_adopted_even_when_current_space_is_low(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(module, 'AUTHORIZED_ROOT', tmp_path)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _: SimpleNamespace(free=50*1024**2))
    campaign = tmp_path/'terrain'; campaign.mkdir()
    accepted, _ = module.reserve_output(campaign, 'terrain-job', 100*1024**2, 128*1024**2, job_id='123', existing=True)
    assert accepted
    registry = json.loads((tmp_path/'dispatch-reservations.json').read_text())
    assert len(registry['reservations']) == 1
    assert next(iter(registry['reservations'].values()))['job_id'] == '123'


def test_materialized_output_is_not_double_counted_as_a_promise(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(module, 'AUTHORIZED_ROOT', tmp_path)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _: SimpleNamespace(free=220*1024**2))
    first = tmp_path/'terrain'; first.mkdir()
    second = tmp_path/'stereo'; second.mkdir()
    job = first/'terrain-job-123'; job.mkdir()
    # Sparse file is sufficient: budget counts logical bytes and the mocked
    # statvfs free count already accounts for materialized output.
    with (job/'outputs.tar.gz').open('wb') as stream:
        stream.truncate(90*1024**2)
    module.reserve_output(first, 'terrain-job', 100*1024**2, 128*1024**2, job_id='123', existing=True)
    accepted, space = module.reserve_output(second, 'stereo-job', 80*1024**2, 128*1024**2)
    assert accepted
    assert space['other_outstanding_promises_bytes'] == 10*1024**2


def test_completed_smoke_can_submit_after_controller_dependency_expires(tmp_path, monkeypatch):
    from types import SimpleNamespace
    smoke_dir = tmp_path/'smoke-123'
    smoke_dir.mkdir()
    completion = completed(smoke_dir)
    smoke = {'name': 'smoke', 'existing_job_id': '123'}
    (tmp_path/'frozen-campaign.tar.gz').write_bytes(b'frozen')
    archive_sha = module.digest(tmp_path/'frozen-campaign.tar.gz')
    completion['archive_sha256'] = archive_sha
    put(smoke_dir/'completion.json', completion)
    put(smoke_dir/'launch.json', {'archive_sha256': archive_sha})
    put(tmp_path/'smoke-submission.json', {'status':'SUBMITTED', 'archive_sha256':archive_sha, 'job_id':'123'})
    (tmp_path/'h34_campaign.sbatch').write_text('frozen launcher')
    put(tmp_path/'protocol.json', {'jobs':[{'name':'run','kind':'run','resources':{
        'partition':'share', 'cpus':2,'memory_gb':4,'time_limit':'00:05:00','requeue':False}}]})
    put(tmp_path/'intake.json', {'archive_sha256':archive_sha,
        'launcher_sha256':module.digest(tmp_path/'h34_campaign.sbatch'),
        'protocol_sha256':module.digest(tmp_path/'protocol.json')})
    monkeypatch.setattr(module, 'scheduler_state', lambda _: ('COMPLETED', '0:0'))
    commands = []
    def submit(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout='456\n', stderr='')
    monkeypatch.setattr(module.subprocess, 'run', submit)
    launcher = SimpleNamespace(validate_protocol=lambda p:p, write_json=put)
    result = module.submit_qualified_job(tmp_path, 'run', smoke, archive_sha, launcher)
    assert result['job_id'] == '456'
    assert not any(arg.startswith('--dependency') for arg in commands[0])
    assert result['qualification_gate']['smoke_job_id'] == '123'
    # A modified protocol must fail before another scheduler command occurs.
    (tmp_path/'protocol.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        module.submit_qualified_job(tmp_path, 'run', smoke, archive_sha, launcher)
    assert len(commands) == 1


def test_only_proven_unsubmitted_dependency_rejection_can_be_retried(tmp_path):
    name = 'run'
    path = tmp_path/(name+'-submission.json')
    put(tmp_path/(name+'-submission-start.json'), {'command':['sbatch']})
    put(path, {'status':'SUBMISSION_FAILED','job_id':None,'returncode':1,'stdout':'',
               'stderr':'sbatch: error: Batch job submission failed: Job dependency problem'})
    expected = module.digest(path)
    module.preserve_explicit_unsubmitted_attempt(tmp_path, {'name':name, 'retry_unsubmitted_receipt_sha256':expected})
    assert not path.exists()
    assert (tmp_path/'unsubmitted-attempts'/(name+'-'+expected)/path.name).is_file()
    put(path, {'status':'SUBMITTED','job_id':'123','returncode':0,'stdout':'123','stderr':''})
    with pytest.raises(ValueError, match='limited'):
        module.preserve_explicit_unsubmitted_attempt(tmp_path, {'name':name,
            'retry_unsubmitted_receipt_sha256':module.digest(path)})
