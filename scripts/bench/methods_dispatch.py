#!/usr/bin/env python3
"""Serial durable dispatcher for already frozen, smoke-qualified method jobs.

This utility runs outside the scientific container. It never edits scientific
inputs, reruns a failed job, changes a cohort, or pushes Git branches. Completed
raw archives are published with the separately frozen methods_archive helper.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time

AUTHORIZED_ROOT = Path('/nfs/stak/users/sanchej7/humanoid-methods-20261010')
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL',
            'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_atomic(path, value):
    temporary = path.with_suffix(path.suffix+'.temporary')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_sources(plan, root):
    for name, expected in plan['source_sha256'].items():
        relative = Path(name)
        item = (root/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not item.is_relative_to(root) or digest(item) != expected:
            raise ValueError('frozen dispatcher source changed: '+name)
    if not plan['source_sha256']:
        raise ValueError('dispatcher source inventory required')


def validate_plan(path):
    plan = json.loads(path.read_text())
    if plan.get('schema') != 'bhl-methods-dispatch-v1':
        raise ValueError('wrong dispatcher schema')
    root = path.parent.resolve()
    validate_sources(plan, root)
    if Path(__file__).resolve() != (root/'methods_dispatch.py').resolve():
        raise ValueError('execute only the frozen dispatcher copy')
    campaign = Path(plan['campaign_dir']).resolve()
    if not campaign.is_relative_to(AUTHORIZED_ROOT) or campaign == AUTHORIZED_ROOT:
        raise ValueError('dispatcher is restricted to new methods campaign root')
    intake = json.loads((campaign/'intake.json').read_text())
    if (intake['archive_sha256'] != plan['archive_sha256'] or
            intake['protocol_sha256'] != plan['protocol_sha256'] or
            digest(campaign/'protocol.json') != plan['protocol_sha256'] or
            digest(campaign/'frozen-campaign.tar.gz') != plan['archive_sha256']):
        raise ValueError('frozen scientific source lineage changed')
    protocol = json.loads((campaign/'protocol.json').read_text())
    jobs = {job['name']: job for job in protocol['jobs']}
    names = [job['name'] for job in plan['jobs']]
    if not names or len(names) != len(set(names)) or any(name not in jobs or jobs[name]['kind'] == 'smoke' for name in names):
        raise ValueError('dispatcher jobs must be unique predeclared development jobs')
    if (plan.get('max_concurrency') != 1 or not 1 <= plan.get('poll_seconds', 30) <= 60 or
            not 0 < plan['reserved_output_bytes'] < plan['max_local_campaign_bytes'] <= 1024**3):
        raise ValueError('invalid serial storage/polling limits')
    if jobs.get(plan['smoke']['name'], {}).get('kind') != 'smoke':
        raise ValueError('declared native smoke required')
    for job in [plan['smoke'], *plan['jobs']]:
        if job.get('existing_job_id') and not re.fullmatch(r'\d+', str(job['existing_job_id'])):
            raise ValueError('invalid fixed job id')
        if not isinstance(job.get('expected_fields', {}), dict):
            raise ValueError('expected scientific cohort fields must be explicit')
    return plan, campaign


def load_helper(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def submitted_id(campaign, job, archive_sha256):
    receipt_path = campaign/(job['name']+'-submission.json')
    if not receipt_path.exists():
        if job.get('existing_job_id'):
            raise ValueError('declared existing job has no submission receipt')
        return None
    receipt = json.loads(receipt_path.read_text())
    value = str(receipt.get('job_id', ''))
    if receipt.get('status') != 'SUBMITTED' or receipt.get('archive_sha256') != archive_sha256 or not value.isdecimal():
        raise ValueError('prior submission failed or source identity changed; no implicit resubmission')
    if job.get('existing_job_id') and value != str(job['existing_job_id']):
        raise ValueError('declared existing scheduler id changed')
    return value


def scheduler_state(job_id):
    result = subprocess.run(['sacct', '-X', '-j', job_id, '--noheader', '--parsable2',
                             '--format=JobIDRaw,State,ExitCode'], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Slurm accounting unavailable')
    rows = [line.split('|') for line in result.stdout.splitlines() if line.strip()]
    matches = [row for row in rows if row[0] == job_id]
    if not matches:
        return 'UNKNOWN', None
    state = matches[0][1].split()[0].rstrip('+')
    return state, matches[0][2]


def checked_result(job_dir, job, archive_sha256):
    completion = json.loads((job_dir/'completion.json').read_text())
    launch = json.loads((job_dir/'launch.json').read_text())
    if (completion['archive_sha256'] != archive_sha256 or launch['archive_sha256'] != archive_sha256 or
            not completion.get('finished_utc') or completion.get('exit_status') != 0 or completion.get('error')):
        raise ValueError('scientific runner incomplete or source identity changed')
    if completion.get('status') not in ('PASS', 'NEGATIVE'):
        raise ValueError('runner did not complete a scientific outcome')
    result_path = job_dir/'campaign_result.json'
    expected = completion['files']['campaign_result.json']
    if result_path.stat().st_size != expected['bytes'] or digest(result_path) != expected['sha256']:
        raise ValueError('completed scientific result changed')
    result = json.loads(result_path.read_text())
    if result.get('status') not in ('PASS', 'NEGATIVE'):
        raise ValueError('scientific result is incomplete')
    for field, expected_value in job.get('expected_fields', {}).items():
        if result.get(field) != expected_value:
            raise ValueError('scientific cohort mismatch: '+field)
    return result, completion


def campaign_bytes(campaign):
    return sum(path.stat().st_size for path in campaign.rglob('*') if path.is_file())




def submit_qualified_job(campaign, job_name, smoke, archive_sha256, launcher):
    """Use durable smoke evidence after Slurm forgets completed dependencies.

    Controller MinJobAge is shorter than this serial campaign. Before each
    submission, recheck completed scientific smoke/source evidence AND accounting
    success; the new job needs no dependency on an expired controller record.
    """
    smoke_id = submitted_id(campaign, smoke, archive_sha256)
    smoke_dir = campaign/(smoke['name']+'-'+smoke_id)
    smoke_result, smoke_completion = checked_result(smoke_dir, smoke, archive_sha256)
    state, exit_code = scheduler_state(smoke_id)
    if state != 'COMPLETED' or exit_code != '0:0' or smoke_result['status'] != 'PASS':
        raise ValueError('completed native smoke qualification required before each submission')
    intake = json.loads((campaign/'intake.json').read_text())
    if (intake['archive_sha256'] != archive_sha256 or digest(campaign/'frozen-campaign.tar.gz') != archive_sha256 or
            digest(campaign/'h34_campaign.sbatch') != intake['launcher_sha256'] or
            digest(campaign/'protocol.json') != intake['protocol_sha256']):
        raise ValueError('scientific archive or execution launcher changed')
    protocol = launcher.validate_protocol(json.loads((campaign/'protocol.json').read_text()))
    job = next(row for row in protocol['jobs'] if row['name'] == job_name)
    if job['kind'] == 'smoke' or job['resources'].get('array'):
        raise ValueError('durable serial dispatcher accepts distinct qualified development jobs only')
    resources = job['resources']
    command = ['sbatch', '--parsable', '--ntasks=1', '--account=eecs', '--job-name='+job_name,
        '--partition='+resources['partition'], '--cpus-per-task='+str(resources['cpus']),
        '--mem='+str(resources['memory_gb'])+'G', '--time='+resources['time_limit'],
        '--output='+str(campaign/(job_name+'-%j.out')), '--error='+str(campaign/(job_name+'-%j.out')), '--export=NONE']
    if resources.get('requeue') is False:
        command.append('--no-requeue')
    for name in ('constraint', 'exclude'):
        if resources.get(name):
            command.append('--'+name+'='+resources[name])
    if resources.get('gpus', 0):
        command.append('--gres=gpu:1')
    command.extend([str(campaign/'h34_campaign.sbatch'), str(campaign), job_name, archive_sha256])
    gate = {'policy': 'completed same-source native smoke checked from durable scientific evidence and Slurm accounting; no dependency on expired controller record',
            'smoke_job_id': smoke_id, 'smoke_completion_sha256': digest(smoke_dir/'completion.json'),
            'smoke_result_sha256': smoke_completion['files']['campaign_result.json']['sha256'],
            'scheduler_state': state, 'scheduler_exit_code': exit_code, 'verified_utc': utcnow()}
    launcher.write_json(campaign/(job_name+'-submission-start.json'), {'created_utc': utcnow(), 'command': command, 'qualification_gate': gate})
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'APPTAINERENV_'))}
    environment['TMPDIR'] = '/tmp'
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    match = re.fullmatch(r'(\d+)(?:;[^\n]+)?\n?', result.stdout)
    receipt = {'submitted_utc': utcnow(), 'job_id': match[1] if match and result.returncode == 0 else None,
        'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
        'resources': resources, 'archive_sha256': archive_sha256, 'qualification_gate': gate,
        'status': 'SUBMITTED' if match and result.returncode == 0 else 'SUBMISSION_FAILED'}
    launcher.write_json(campaign/(job_name+'-submission.json'), receipt)
    return receipt


def preserve_explicit_unsubmitted_attempt(campaign, job):
    """One explicit plan may preserve and retry a proven pre-submission rejection."""
    expected = job.get('retry_unsubmitted_receipt_sha256')
    if not expected:
        return
    path = campaign/(job['name']+'-submission.json')
    destination = campaign/'unsubmitted-attempts'/(job['name']+'-'+expected)
    if destination.exists():
        if digest(destination/path.name) != expected:
            raise ValueError('preserved rejected-submission receipt changed')
        return
    if digest(path) != expected:
        raise ValueError('explicit unsubmitted retry receipt mismatch')
    receipt = json.loads(path.read_text())
    if (receipt.get('status') != 'SUBMISSION_FAILED' or receipt.get('job_id') is not None or
            receipt.get('returncode') != 1 or receipt.get('stdout') or 'Job dependency problem' not in receipt.get('stderr', '')):
        raise ValueError('retry is limited to a proved expired-dependency pre-submission rejection')
    destination.mkdir(parents=True, exist_ok=False)
    for name in (job['name']+'-submission-start.json', path.name):
        (campaign/name).rename(destination/name)


def reserve_output(campaign, job_name, reserved_bytes, minimum_free_bytes, *, job_id=None, existing=False):
    """Atomically promise global home space; existing running jobs are adopted.

    This NFS home mount reports the user's25GiB quota through statvfs. Subtract
    outstanding promises, minus already materialized bytes, from actual free
    bytes. The lock serializes competing dispatchers' check-and-reserve step.
    """
    campaign = Path(campaign).resolve()
    if (not campaign.is_relative_to(AUTHORIZED_ROOT) or reserved_bytes <= 0 or minimum_free_bytes < 128*1024**2 or
            not re.fullmatch(r'[A-Za-z0-9_-]+', job_name) or (job_id is not None and not str(job_id).isdecimal())):
        raise ValueError('invalid explicitly scoped home reservation')
    registry_path = AUTHORIZED_ROOT/'dispatch-reservations.json'
    lock_path = AUTHORIZED_ROOT/'dispatch-reservations.lock'
    key = str(campaign)+'/'+job_name
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else {'schema': 'bhl-methods-reservations-v1', 'reservations': {}}
        records = registry['reservations']
        # A completed, verified and retired publication can release a promise
        # even when the original dispatcher stopped just after publication.
        for old_key, row in list(records.items()):
            directory = Path(row['job_dir']) if row.get('job_dir') else None
            if directory and (directory/'publication.json').exists() and (directory/'completion.json').exists():
                published = json.loads((directory/'publication.json').read_text())
                completion = json.loads((directory/'completion.json').read_text())
                raw = completion.get('files', {}).get('outputs.tar.gz', {})
                if (published.get('local_packaging_retired') is True and not (directory/'outputs.tar.gz').exists() and
                    published.get('source_archive_sha256') == completion.get('archive_sha256') and
                    published.get('sha256') == raw.get('sha256') and published.get('bytes') == raw.get('bytes')):
                    del records[old_key]
        promised = 0
        for old_key, row in records.items():
            if old_key == key:
                continue
            directory = Path(row['job_dir']) if row.get('job_dir') else None
            materialized = campaign_bytes(directory) if directory and directory.exists() else 0
            promised += max(0, row['reserved_bytes']-materialized)
        available = shutil.disk_usage(AUTHORIZED_ROOT).free-promised
        directory = campaign/(job_name+'-'+str(job_id)) if job_id else None
        materialized = campaign_bytes(directory) if directory and directory.exists() else 0
        requested = max(0, reserved_bytes-materialized)
        accepted = bool(existing or available >= minimum_free_bytes+requested)
        if accepted:
            records[key] = {'campaign': str(campaign), 'job_name': job_name, 'job_id': str(job_id) if job_id else None,
                            'job_dir': str(directory) if directory else None, 'reserved_bytes': reserved_bytes,
                            'registered_utc': utcnow(), 'adopted_existing_job': bool(existing)}
        registry['updated_utc'] = utcnow()
        write_atomic(registry_path, registry)
        return accepted, {'actual_free_bytes': shutil.disk_usage(AUTHORIZED_ROOT).free,
                          'other_outstanding_promises_bytes': promised, 'requested_unwritten_bytes': requested,
                          'minimum_free_bytes': minimum_free_bytes}


def release_output(campaign, job_name):
    registry_path = AUTHORIZED_ROOT/'dispatch-reservations.json'
    with (AUTHORIZED_ROOT/'dispatch-reservations.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        registry = json.loads(registry_path.read_text())
        registry['reservations'].pop(str(Path(campaign).resolve())+'/'+job_name, None)
        registry['updated_utc'] = utcnow()
        write_atomic(registry_path, registry)


def run(path, expected_plan_sha256):
    path = path.resolve()
    if not expected_plan_sha256 or digest(path) != expected_plan_sha256:
        raise ValueError("frozen dispatcher plan checksum required")
    plan, campaign = validate_plan(path)
    root = path.parent
    launcher = load_helper(root/'h34_campaign.py', 'frozen_methods_launcher')
    archiver = load_helper(root/'methods_archive.py', 'frozen_methods_archiver')
    state_path = root/'state.json'
    existing = json.loads(state_path.read_text()) if state_path.exists() else None
    if existing and existing.get('plan_sha256') != digest(path):
        raise ValueError('dispatcher resume plan differs')
    state = existing or {'schema': 'bhl-methods-dispatch-state-v1', 'plan_sha256': digest(path),
        'started_utc': utcnow(), 'dispatcher_job_id': os.getenv('SLURM_JOB_ID'), 'jobs': [],
        'scientific_archive_sha256': plan['archive_sha256']}
    def save(status, **values):
        state.update(status=status, updated_utc=utcnow(), **values)
        write_atomic(state_path, state)
    def stopped(signum, _):
        save('INTERRUPTED', reason=f'dispatcher signal {signum}; submitted scientific job left intact')
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        smoke_id = submitted_id(campaign, plan['smoke'], plan['archive_sha256'])
        smoke_dir = campaign/(plan['smoke']['name']+'-'+smoke_id)
        smoke_result, _ = checked_result(smoke_dir, plan['smoke'], plan['archive_sha256'])
        smoke_state, smoke_exit = scheduler_state(smoke_id)
        if smoke_state != 'COMPLETED' or smoke_exit != '0:0' or smoke_result['status'] != 'PASS':
            raise ValueError('actual same-source native smoke must already have completed successfully')
        smoke_publication = archiver.publish(smoke_dir, retire=True)
        if not smoke_publication.get('local_packaging_retired') or (smoke_dir/'outputs.tar.gz').exists():
            raise ValueError('smoke archive must be verified remotely before releasing storage')
        release_output(campaign, plan['smoke']['name'])
        dispatcher_deadline = time.monotonic()+23.5*3600
        for index, job in enumerate(plan['jobs']):
            validate_sources(plan, root)
            preserve_explicit_unsubmitted_attempt(campaign, job)
            job_id = submitted_id(campaign, job, plan['archive_sha256'])
            if job_id is None:
                used = campaign_bytes(campaign)
                if used+plan['reserved_output_bytes'] > plan['max_local_campaign_bytes']:
                    raise RuntimeError('campaign home budget lacks reserved output headroom; next job not submitted')
                while True:
                    accepted, space = reserve_output(campaign, job['name'], plan['reserved_output_bytes'],
                                                     plan.get('minimum_free_bytes', 128*1024**2))
                    if accepted:
                        break
                    save('WAITING_FOR_RESERVED_SPACE', current_index=index, current_job=job['name'], storage=space)
                    if time.monotonic() >= dispatcher_deadline:
                        raise RuntimeError('bounded dispatcher reached storage-wait deadline; next job not submitted')
                    time.sleep(plan['poll_seconds'])
                save('SUBMITTING', current_index=index, current_job=job['name'], local_campaign_bytes=used)
                receipt = submit_qualified_job(campaign, job['name'], plan['smoke'], plan['archive_sha256'], launcher)
                if receipt.get('status') != 'SUBMITTED':
                    raise RuntimeError('bounded scientific job submission failed')
                job_id = str(receipt['job_id'])
            reserve_output(campaign, job['name'], plan['reserved_output_bytes'],
                           plan.get('minimum_free_bytes', 128*1024**2), job_id=job_id, existing=True)
            job_dir = campaign/(job['name']+'-'+job_id)
            save('WAITING', current_index=index, current_job=job['name'], current_job_id=job_id)
            transient = 0
            missing_completion = 0
            while True:
                try:
                    scheduler, exit_code = scheduler_state(job_id)
                    transient = 0
                except RuntimeError:
                    transient += 1
                    if transient >= 5:
                        raise
                    time.sleep(plan['poll_seconds'])
                    continue
                if scheduler in TERMINAL:
                    if scheduler != 'COMPLETED' or exit_code != '0:0':
                        raise RuntimeError(f'scientific job {job_id} terminated {scheduler} {exit_code}; no automatic rerun')
                    if (job_dir/'completion.json').exists():
                        break
                    missing_completion += 1
                    if missing_completion >= 6:
                        raise RuntimeError('terminal job lacks a scientific completion receipt')
                time.sleep(plan['poll_seconds'])
            result, completion = checked_result(job_dir, job, plan['archive_sha256'])
            save('PUBLISHING', current_index=index, current_job=job['name'], current_job_id=job_id)
            publication = archiver.publish(job_dir, retire=True)
            expected_raw = completion['files']['outputs.tar.gz']
            if (publication.get('source_archive_sha256') != plan['archive_sha256'] or
                    publication.get('sha256') != expected_raw['sha256'] or publication.get('bytes') != expected_raw['bytes'] or
                    publication.get('local_packaging_retired') is not True or (job_dir/'outputs.tar.gz').exists()):
                raise ValueError('verified raw publication/retirement receipt required before next job')
            release_output(campaign, job['name'])
            record = {'name': job['name'], 'job_id': job_id, 'scientific_status': result.get('scientific_status'),
                'status': result['status'], 'episodes': result.get('episodes'), 'result_sha256': completion['files']['campaign_result.json']['sha256'],
                'publication': str(job_dir/'publication.json'), 'raw_url': publication['url'], 'raw_sha256': publication['sha256'],
                'finished_utc': completion['finished_utc']}
            state['jobs'] = [row for row in state['jobs'] if row['name'] != job['name']]+[record]
            save('SHARD_PUBLISHED', completed_jobs=len(state['jobs']), expected_jobs=len(plan['jobs']))
            print(json.dumps(record), flush=True)
        save('COMPLETE', completed_jobs=len(state['jobs']), expected_jobs=len(plan['jobs']),
             total_episodes=sum(row.get('episodes') or 0 for row in state['jobs']),
             scientific_scope='execution and publication complete; matched-cohort independent collection required before an improvement claim')
        return 0
    except Exception as error:
        save('STOPPED', error_type=type(error).__name__, reason=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--register-existing-campaign', type=Path)
    parser.add_argument('--job-name')
    parser.add_argument('--job-id')
    parser.add_argument('--reservation-bytes', type=int)
    arguments = parser.parse_args()
    if arguments.register_existing_campaign:
        if not arguments.job_name or not arguments.job_id or not arguments.reservation_bytes:
            parser.error('existing reservation requires job name, job id and reservation bytes')
        print(json.dumps(reserve_output(arguments.register_existing_campaign, arguments.job_name,
            arguments.reservation_bytes, 128*1024**2, job_id=arguments.job_id, existing=True)))
    elif arguments.plan:
        raise SystemExit(run(arguments.plan, arguments.plan_sha256))
    else:
        parser.error('--plan or --register-existing-campaign is required')
