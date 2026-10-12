#!/usr/bin/env python3
"""Freeze and execute the 48-cell processing schedule comparison on one GPU.

A single allocation runs a new 12-cell smoke then 24 matched two-episode shards.
Every pair uses the same source, node, GPU, resources and allocation. Scientific
negatives are retained; execution errors stop subsequent shards without retries.
The old campaign contributes only hash-verified actor/runtime/upstream assets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

SHARED_ROOT = Path('/nfs/hpc/share/sanchej7/humanoid-research-20261011')
BASE_ARCHIVE_SHA = '3628b95ced285660e27c0b3b0ca92a097415beaeedfa912da751832acc3b8282'
BASE_PROTOCOL_SHA = '08a71ea1dcaad9460ef6f3990af227323e48492442fd3acb675a334a5808caf0'
ORB_SHA = '91ca4b6e4ea8f318998f400d4f7dfcd6fe8ae88b375399c79d5d633c39d71516'
LIO_SHA = '6fc66e9318be7e8f5d2ee545652928b071ccbe949f1abc21f1247170f0e6b6f4'
PROFILE_ARCHIVE_SHA = 'e21f6ac62d5cd76387b1692276854db6e95767eed28670a4c0ba49804d866c81'
PROFILE_EPISODE_SHA = 'b2528d9ac58e51ea9ee2d5b96f163622bd69107497cb1b557b833cd78d72f690'
CONDITIONS = ('nominal', 'camera_10hz', 'causal_40ms_exposure')
CASES = ('straight', 'occluders')
ARMS = ('stereo', 'lidar_inertial')
SEEDS = (750000, 750001)
GIB = 1024**3
CAP_BYTES = 100 * GIB
PYTHON = '/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python'
RESOURCES = dict(partition='gpu,dgxh,ampere', cpus=2, memory_gb=16, gpus=1,
    time_limit='12:00:00', constraint='a40|rtx8000|l40s|h100|h200|h200-70g',
    exclude='dgxh-1', requeue=False)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def helper(path):
    spec = importlib.util.spec_from_file_location('schedule_frozen_h34', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bounded_directory(path):
    path = Path(path).resolve()
    if not path.is_relative_to(SHARED_ROOT) or path == SHARED_ROOT:
        raise ValueError('campaign must be a dedicated subdirectory of the authorized shared root')
    return path


def quota_receipt(reserve_bytes):
    """Use the explicitly authorized hard quota; preserve the soft-limit receipt."""
    proc = subprocess.run(['lfs', 'quota', '-p', '30762', '/nfs/hpc/share'],
                          capture_output=True, text=True, check=True)
    rows = [line.split() for line in proc.stdout.splitlines() if line.strip().startswith('/nfs/hpc/share')]
    if len(rows) != 1 or len(rows[0]) < 5:
        raise ValueError('unrecognized project quota response')
    row = rows[0]
    used, soft, hard = (int(value.rstrip('*')) * 1024 for value in row[1:4])
    if hard != 2 * 1024**4 or hard - used < reserve_bytes + 8 * GIB:
        raise ValueError('project hard quota cannot cover the bounded campaign plus 8 GiB margin')
    return dict(project=30762, used_bytes=used, soft_limit_bytes=soft,
                hard_limit_bytes=hard, grace=row[4], reserve_bytes=reserve_bytes,
                measured_utc=now(), raw=proc.stdout)


def declared_jobs(campaign):
    common = ['--upstream', '{source}/external/Berkeley-Humanoid-Lite',
        '--deploy', '{campaign}/inputs/actors/dr-default-s0/deploy.yaml',
        '--checkpoint', '{campaign}/inputs/actors/dr-default-s0/policy.onnx',
        '--orb-runtime-archive', '{campaign}/inputs/runtime/orb-runtime.tar.gz',
        '--orb-runtime-sha256', ORB_SHA,
        '--lio-runtime-archive', '{campaign}/inputs/runtime/lio-runtime.tar.gz',
        '--lio-runtime-sha256', LIO_SHA,
        '--protocol', '{campaign}/inputs/scientific-protocol.json', '--output', '{output}']
    def job(name, kind, args):
        return dict(name=name, kind=kind, entrypoint='scripts/bench/processing_schedule_campaign.py',
                    args=common + args, resources=dict(RESOURCES), max_output_mb=2048,
                    env={'MUJOCO_GL': 'egl', 'OMP_NUM_THREADS': '1'})
    result = [dict(name='processing-schedule-profile-20261012', kind='evaluate',
        entrypoint='scripts/bench/processing_schedule_campaign.py',
        args=['--profile-acquisition', '{campaign}/inputs/profile-episode', '--output', '{output}'],
        resources=dict(RESOURCES), max_output_mb=2048,
        env={'MUJOCO_GL': 'egl', 'OMP_NUM_THREADS': '1'}),
        job('processing-schedule-smoke-20261012', 'smoke', ['--phase', 'smoke'])]
    for seed in SEEDS:
        for condition in CONDITIONS:
            for case in CASES:
                for arm in ARMS:
                    name = f'schedule-s{seed}-{condition}-{case}-{arm}'
                    result.append(job(name, 'run', ['--phase', 'development', '--seed', str(seed),
                        '--condition', condition, '--case', case, '--arm', arm,
                        '--qualified-smoke', str(campaign / 'qualified-smoke.json')]))
    return result


def source_overlay(repo, destination):
    """Retain only the verified upstream tree; snapshot the current runtime tree."""
    repo, destination = Path(repo).resolve(), Path(destination)
    for item in list(destination.iterdir()):
        if item.name == 'external':
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    names = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z'], text=True).split('\0')
    prefixes = ('src/', 'scripts/', 'slurm/', 'assets/', 'configs/', 'config/')
    selected = {name for name in names if name.startswith(prefixes) or name in
                {'pyproject.toml', 'uv.lock', '.gitmodules', 'requirements-test.txt'}}
    # Include newly implemented Python files even before commit; intake records dirty state.
    for root in ('src', 'scripts'):
        selected.update(str(path.relative_to(repo)) for path in (repo / root).rglob('*.py')
                        if '__pycache__' not in path.parts)
    for name in sorted(selected):
        source, target = repo / name, destination / name
        if not source.is_file():
            continue
        if source.is_symlink() or not source.resolve().is_relative_to(repo):
            raise ValueError('source file is a link or escapes the repository: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def bootstrap_text():
    return '''#!/bin/bash
#SBATCH --signal=B:TERM@60
set -euo pipefail
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1
PERSIST=$1
EXPECTED_SHA=$2
BOOT=$(mktemp -d /tmp/bhl-schedule-bootstrap-${SLURM_JOB_ID:-local}-XXXXXX)
/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python - "$PERSIST" "$EXPECTED_SHA" "$BOOT" <<'PYBOOT'
import hashlib,pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1]); archive=root/'frozen-campaign.tar.gz'
with archive.open('rb') as stream:
    if hashlib.file_digest(stream,'sha256').hexdigest()!=sys.argv[2]:
        raise SystemExit('Frozen schedule archive changed')
with tarfile.open(archive,'r:gz') as tar:
    for name in ('processing_schedule_dispatch.py','h34_campaign.py'):
        member=tar.getmember('source/scripts/bench/'+name)
        if not member.isfile() or member.size>1024*1024:
            raise SystemExit('Invalid frozen bootstrap member')
        with tar.extractfile(member) as stream:
            (pathlib.Path(sys.argv[3])/name).write_bytes(stream.read())
PYBOOT
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python "$BOOT/processing_schedule_dispatch.py" execute --campaign "$PERSIST" --archive-sha "$EXPECTED_SHA"
'''


def prepare(repo, scientific_protocol, base, campaign, profile_episode):
    repo, base, campaign = Path(repo).resolve(), Path(base).resolve(), bounded_directory(campaign)
    launcher = helper(repo / 'scripts/bench/h34_campaign.py')
    _, original, old_protocol = launcher.load_intake(base)
    if original['archive_sha256'] != BASE_ARCHIVE_SHA or original['protocol_sha256'] != BASE_PROTOCOL_SHA:
        raise ValueError('base must be the untouched native controls v2 snapshot')
    quota = quota_receipt(CAP_BYTES)
    campaign.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='bhl-schedule-freeze-', dir='/tmp') as temporary:
        frozen = Path(temporary)
        launcher.safe_extract(base / 'frozen-campaign.tar.gz', frozen)
        launcher.verify_source(frozen)
        source_overlay(repo, frozen / 'source')
        shutil.copyfile(scientific_protocol, frozen / 'inputs/scientific-protocol.json')
        profile_episode = Path(profile_episode).resolve()
        if not profile_episode.is_relative_to(SHARED_ROOT) or not (profile_episode / 'episode.json').is_file():
            raise ValueError('profile requires the retained, verified original episode under shared storage')
        if any(path.is_symlink() for path in profile_episode.rglob('*')):
            raise ValueError('profile evidence must not contain links')
        if sum(path.stat().st_size for path in profile_episode.rglob('*') if path.is_file()) > 512 * 1024**2:
            raise ValueError('profile evidence exceeds 512 MiB')
        profile_receipt = profile_episode.parent / 'extraction-receipt.json'
        receipt = read(profile_receipt)
        if (digest(profile_episode / 'episode.json') != PROFILE_EPISODE_SHA or
                receipt.get('episode_sha256') != PROFILE_EPISODE_SHA or
                receipt.get('original_archive_sha256') != PROFILE_ARCHIVE_SHA):
            raise ValueError('profile source differs from the selected retained nominal stereo episode')
        shutil.copytree(profile_episode, frozen / 'inputs/profile-episode')
        shutil.copyfile(profile_receipt, frozen / 'inputs/profile-source-receipt.json')
        protocol = dict(schema_version=1, campaign_id=campaign.name, runtime=old_protocol['runtime'],
            scientific_scope='48 new paired closed-loop simulation episodes; same allocation,12-cell fresh smoke; no hardware claims',
            jobs=declared_jobs(campaign))
        launcher.validate_protocol(protocol)
        (frozen / 'protocol.json').unlink()
        (frozen / 'manifest.json').unlink()
        write(frozen / 'protocol.json', protocol)
        files = [path for root in ('source', 'inputs') for path in (frozen / root).rglob('*') if path.is_file()]
        total = sum(path.stat().st_size for path in files)
        if total > 2 * GIB or any(path.stat().st_size > 512 * 1024**2 for path in files):
            raise ValueError('new freeze exceeds source bounds')
        manifest = {str(path.relative_to(frozen)): dict(bytes=path.stat().st_size, sha256=digest(path)) for path in sorted(files)}
        write(frozen / 'manifest.json', manifest)
        for name in ('protocol.json', 'manifest.json'):
            shutil.copyfile(frozen / name, campaign / name)
        archive = campaign / 'frozen-campaign.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for path in [frozen / 'protocol.json', frozen / 'manifest.json', *sorted(files)]:
                tar.add(path, arcname=str(path.relative_to(frozen)), recursive=False)
        launcher.verify_source(frozen)
        shutil.copyfile(repo / 'slurm/repo20261008/h34_campaign.sbatch', campaign / 'h34_campaign.sbatch')
        (campaign / 'processing_schedule.sbatch').write_text(bootstrap_text())
        intake = dict(status='FROZEN_NOT_SUBMITTED', created_utc=now(),
            archive_sha256=digest(archive), protocol_sha256=digest(campaign / 'protocol.json'),
            launcher_sha256=digest(campaign / 'h34_campaign.sbatch'),
            coordinator_launcher_sha256=digest(campaign / 'processing_schedule.sbatch'),
            sif_sha256=digest(protocol['runtime']['sif']), source_files=len(manifest),
            archive_bytes=archive.stat().st_size, uncompressed_input_bytes=total,
            source_commit=launcher.git_output(repo, 'rev-parse', 'HEAD'),
            source_status=launcher.git_output(repo, 'status', '--porcelain'),
            base_archive_sha256=BASE_ARCHIVE_SHA, base_protocol_sha256=BASE_PROTOCOL_SHA,
            upstream_provenance=original['submodules'], scientific_protocol_sha256=digest(scientific_protocol),
            max_campaign_bytes=CAP_BYTES, max_output_bytes_per_shard=2 * GIB,
            new_scored_episodes=48, fresh_smoke_episodes=12, quota_before_freeze=quota,
            profile_episode_sha256=digest(profile_episode / 'episode.json'))
        write(campaign / 'intake.json', intake)
    return intake


def verified_intake(campaign, expected=None):
    campaign = bounded_directory(campaign)
    launcher = helper(Path(__file__).with_name('h34_campaign.py'))
    _, intake, protocol = launcher.load_intake(campaign)
    if expected is not None and expected != intake['archive_sha256']:
        raise ValueError('submitted archive differs from intake')
    if (intake.get('coordinator_launcher_sha256') != digest(campaign / 'processing_schedule.sbatch') or
            intake.get('max_campaign_bytes') != CAP_BYTES or protocol['jobs'] != declared_jobs(campaign)):
        raise ValueError('launch resources, declared pair inventory or coordinator changed')
    return launcher, intake, protocol


def submit(campaign):
    campaign = bounded_directory(campaign)
    _, intake, _ = verified_intake(campaign)
    quota = quota_receipt(CAP_BYTES)
    if (campaign / 'submission-start.json').exists():
        raise ValueError('an attempt was already submitted; no implicit retry')
    r = RESOURCES
    command = ['/apps/slurm/25.11.8/bin/sbatch', '--parsable', '--ntasks=1', '--account=eecs',
        '--job-name=processing-schedule-48', '--partition=' + r['partition'], '--cpus-per-task=2',
        '--mem=16G', '--time=' + r['time_limit'], '--gres=gpu:1', '--constraint=' + r['constraint'],
        '--exclude=' + r['exclude'], '--export=NONE', '--no-requeue',
        '--output=' + str(campaign / 'processing-schedule-%j.out'),
        '--error=' + str(campaign / 'processing-schedule-%j.out'),
        str(campaign / 'processing_schedule.sbatch'), str(campaign), intake['archive_sha256']]
    write(campaign / 'submission-start.json', dict(created_utc=now(), command=command, quota=quota))
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'APPTAINERENV_'))}
    env['TMPDIR'] = '/tmp'
    proc = subprocess.run(command, capture_output=True, text=True, env=env)
    match = re.fullmatch(r'(\d+)(?:;[^\n]+)?\n?', proc.stdout)
    result = dict(status='SUBMITTED' if proc.returncode == 0 and match else 'SUBMISSION_FAILED',
        job_id=match[1] if proc.returncode == 0 and match else None, submitted_utc=now(),
        returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
        archive_sha256=intake['archive_sha256'], resources=r, command=command)
    write(campaign / 'submission.json', result)
    return result


def checked_completion(directory, archive_sha):
    directory = Path(directory)
    launch, completion = read(directory / 'launch.json'), read(directory / 'completion.json')
    if (launch['archive_sha256'] != archive_sha or completion['archive_sha256'] != archive_sha or
            completion['exit_status'] != 0 or completion.get('error') or
            completion['status'] not in ('PASS', 'NEGATIVE')):
        raise ValueError('native execution did not complete with verified source')
    for name, spec in completion['files'].items():
        if Path(name).name != name:
            raise ValueError('completion file escapes its directory')
        path = directory / name
        if path.is_symlink() or path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
            raise ValueError('completed evidence changed')
    result = read(directory / 'campaign_result.json')
    return result, completion, launch


def smoke_gate(result):
    if (result.get('status') != 'PASS' or result.get('phase') != 'smoke' or
            result.get('episodes') != 12 or result.get('execution_gate') is not True):
        raise ValueError('complete fresh 12-episode nominal-readiness smoke required')


def retain_and_retire_scratch(directory, launch, completion):
    """Delete only this attempt's scratch after its full archive verifies."""
    scratch = Path(launch['scratch'])
    if (scratch.is_symlink() or scratch.parent != Path('/tmp') or
            not scratch.name.startswith('bhl-h34-') or not scratch.is_dir()):
        raise ValueError('unexpected scratch directory; preservation required')
    archive = Path(directory) / 'outputs.tar.gz'
    expected = completion['files']['outputs.tar.gz']
    if digest(archive) != expected['sha256']:
        raise ValueError('raw output archive changed')
    files = {str(path.relative_to(scratch / 'output')): path
             for path in (scratch / 'output').rglob('*') if path.is_file()}
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        if len(members) != len(files) or {m.name for m in members} != set(files):
            raise ValueError('archived raw evidence inventory differs')
        for member in members:
            if not member.isfile():
                raise ValueError('non-file raw output member')
            with tar.extractfile(member) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest(files[member.name]):
                    raise ValueError('archived raw evidence differs')
    write(Path(directory) / 'scratch-retirement.json', dict(schema='bhl-verified-scratch-retirement-v1',
        original_scratch=str(scratch), raw_files=len(files), raw_bytes=completion['output_bytes'],
        verified_archive_sha256=expected['sha256'], verified_utc=now()))
    shutil.rmtree(scratch)


def execute(campaign, expected_sha):
    campaign = bounded_directory(campaign)
    launcher, intake, protocol = verified_intake(campaign, expected_sha)
    job_id = os.environ.get('SLURM_JOB_ID')
    if not job_id or not os.environ.get('SLURM_JOB_GPUS', os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('new execution requires an allocated Slurm GPU')
    if read(campaign / 'submission.json').get('job_id') != job_id:
        raise ValueError('allocation differs from the unique submission')
    write(campaign / 'execution-start.json', dict(started_utc=now(), job_id=job_id,
        host=os.uname().nodename, archive_sha256=expected_sha,
        gpu_query=subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'],
                                 capture_output=True, text=True).stdout))
    results = []
    attempted = []
    error = None
    try:
        for index, job in enumerate(protocol['jobs']):
            used = sum(path.stat().st_size for path in campaign.rglob('*') if path.is_file())
            if used + 2 * GIB > CAP_BYTES:
                raise ValueError('bounded 100 GiB campaign output reserve exhausted')
            quota_receipt(2 * GIB)
            if index >= 2:
                qualified = read(campaign / 'qualified-smoke.json')
                smoke_gate(qualified)
                smoke_dir = campaign / (protocol['jobs'][1]['name'] + '-' + job_id)
                original_smoke, _, _ = checked_completion(smoke_dir, expected_sha)
                if qualified != original_smoke:
                    raise ValueError('qualified smoke differs from immutable completed smoke')
            attempted.append(job['name'])
            rc = launcher.execute(campaign, job['name'], expected_sha)
            directory = campaign / (job['name'] + '-' + job_id)
            result, completion, launch = checked_completion(directory, expected_sha)
            if rc:
                raise ValueError('native runner failed its execution gate')
            if index == 0:
                if result.get('status') != 'PASS' or result.get('schema') != 'bhl-processing-schedule-v1-profile':
                    raise ValueError('retained identical-input profiling did not pass')
            elif index == 1:
                smoke_gate(result)
                write(campaign / 'qualified-smoke.json', result)
            elif result.get('phase') != 'development' or result.get('episodes') != 2:
                raise ValueError('matched pair did not contain exactly 2 completed episodes')
            results.append(dict(name=job['name'], status=result['status'], episodes=result.get('episodes', 0),
                                result_sha256=digest(directory / 'campaign_result.json'),
                                raw_archive_sha256=digest(directory / 'outputs.tar.gz')))
            retain_and_retire_scratch(directory, launch, completion)
            write(campaign / f'progress-{index:02d}.json', dict(updated_utc=now(), completed=list(results)))
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
    final = dict(schema='bhl-processing-schedule-dispatch-v1', finished_utc=now(),
        status='COMPLETE' if not error else 'INCOMPLETE', execution_error=error,
        archive_sha256=expected_sha, job_id=job_id, host=os.uname().nodename,
        completed=list(results), attempted=attempted,
        unrun=[job['name'] for job in protocol['jobs'] if job['name'] not in attempted],
        scope='execution completeness only; clean goals and schedule comparisons require the scientific collector')
    write(campaign / 'dispatch-result.json', final)
    return 0 if not error else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--repo', required=True, type=Path)
    prep.add_argument('--scientific-protocol', required=True, type=Path)
    prep.add_argument('--profile-episode', required=True, type=Path)
    prep.add_argument('--base-campaign', type=Path, default=SHARED_ROOT / 'native-sensor-controls-v2')
    prep.add_argument('--campaign', required=True, type=Path)
    submit_p = sub.add_parser('submit')
    submit_p.add_argument('--campaign', required=True, type=Path)
    execute_p = sub.add_parser('execute')
    execute_p.add_argument('--campaign', required=True, type=Path)
    execute_p.add_argument('--archive-sha', required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args.repo, args.scientific_protocol, args.base_campaign, args.campaign, args.profile_episode), indent=2))
    elif args.action == 'submit':
        result = submit(args.campaign)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'SUBMITTED' else 1
    else:
        return execute(args.campaign, args.archive_sha)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
