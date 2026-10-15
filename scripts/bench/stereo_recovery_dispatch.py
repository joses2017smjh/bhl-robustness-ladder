#!/usr/bin/env python3
"""Freeze and execute the 48-episode stereo recovery comparison on one RTX 8000 GPU.

One allocation runs a fresh 6-cell smoke, then 24 matched two-episode shards
(permanent_stop vs resume_verified_relocalization; counterbalanced order), then
the strict collector with the predeclared analysis. Every pair uses the same
source, node, GPU and allocation. Scientific negatives are retained; execution
errors stop the remaining shards without retries. The earlier native-controls
campaign contributes only hash-verified actor/LIO/upstream assets; its ORB
runtime input is replaced by the Atlas-telemetry relink (unchanged library).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import processing_schedule_dispatch as PSD

SHARED_ROOT = PSD.SHARED_ROOT
BASE_ARCHIVE_SHA, BASE_PROTOCOL_SHA = PSD.BASE_ARCHIVE_SHA, PSD.BASE_PROTOCOL_SHA
BASE_ORB_SHA = PSD.ORB_SHA
ORB_SHA = 'df8f927939837cde5edc7c81d75b41c94d27be06f44636e5aa362cb48a37c72c'
ORB_LIBRARY_SHA = 'c4f09d2dc42571310f4324b8d5cc460158d6bc34d9fe63affbc009cdabc80ffa'
ATLAS_TELEMETRY = 'bhl-orb-atlas-telemetry-v1'
LIO_SHA = PSD.LIO_SHA
ENTRYPOINT = 'scripts/bench/stereo_recovery_campaign.py'
CONDITIONS = ('nominal', 'camera_10hz', 'causal_40ms_exposure')
CASES = ('straight', 'occluders')
ARMS = ('permanent_stop', 'resume_verified_relocalization')
SEEDS = (760000, 760001, 760002, 760003)
SMOKE_EPISODES, PAIR_EPISODES = 6, 2
DECLARED_CELLS = len(SEEDS) * len(CONDITIONS) * len(CASES) * len(ARMS)
GIB = PSD.GIB
CAP_BYTES = 40 * GIB
MAX_OUTPUT_MB = 2048
SMOKE_NAME = 'stereo-recovery-smoke-20261014'
COLLECTION_NAME = 'stereo-recovery-collection-20261014'
JOB_NAME = 'stereo-recovery-48'
# Same GPU model as processing-schedule v1 (cn-gpu6, Quadro RTX 8000) for comparability.
RESOURCES = dict(partition='gpu', cpus=2, memory_gb=16, gpus=1, time_limit='06:00:00',
                 constraint='rtx8000', requeue=False)
EVIDENCE = (('inference_sensors.npz', 'inference_sensors_sha256'), ('evaluator_truth.npz', 'evaluator_truth_sha256'),
            ('native/stream_receipt.json', 'stream_receipt_sha256'), ('native/native_frames.jsonl', 'native_frames_sha256'),
            ('native/recovery_receipt.json', 'recovery_receipt_sha256'),
            ('sensors/processing_schedule.json', 'processing_schedule_sha256'))
digest, read, write, now = PSD.digest, PSD.read, PSD.write, PSD.now


def declared_jobs(campaign):
    common = ['--upstream', '{source}/external/Berkeley-Humanoid-Lite',
        '--deploy', '{campaign}/inputs/actors/dr-default-s0/deploy.yaml',
        '--checkpoint', '{campaign}/inputs/actors/dr-default-s0/policy.onnx',
        '--orb-runtime-archive', '{campaign}/inputs/runtime/orb-runtime.tar.gz',
        '--orb-runtime-sha256', ORB_SHA,
        '--lio-runtime-archive', '{campaign}/inputs/runtime/lio-runtime.tar.gz',
        '--lio-runtime-sha256', LIO_SHA,
        '--protocol', '{campaign}/inputs/scientific-protocol.json', '--output', '{output}']
    def job(name, kind, args, **extra):
        return dict(name=name, kind=kind, entrypoint=ENTRYPOINT, args=args, resources=dict(RESOURCES),
                    max_output_mb=MAX_OUTPUT_MB, env={'MUJOCO_GL': 'egl', 'OMP_NUM_THREADS': '1'}, **extra)
    jobs = [job(SMOKE_NAME, 'smoke', common + ['--phase', 'smoke'])]
    for seed in SEEDS:
        for condition in CONDITIONS:
            for case in CASES:
                jobs.append(job(f'recovery-s{seed}-{condition}-{case}', 'run', common + [
                    '--phase', 'development', '--seed', str(seed), '--condition', condition, '--case', case,
                    '--qualified-smoke', str(campaign / 'qualified-smoke.json')]))
    jobs.append(job(COLLECTION_NAME, 'evaluate', ['--collect', str(campaign / 'collector-inputs'),
        '--qualified-smoke', str(campaign / 'qualified-smoke.json'), '--output', '{output}/collection.json'],
        result_path='collection.json'))
    return jobs


def bootstrap_text():
    return '''#!/bin/bash
#SBATCH --signal=B:TERM@60
set -euo pipefail
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1
PERSIST=$1
EXPECTED_SHA=$2
BOOT=$(mktemp -d /tmp/bhl-recovery-bootstrap-${SLURM_JOB_ID:-local}-XXXXXX)
/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python - "$PERSIST" "$EXPECTED_SHA" "$BOOT" <<'PYBOOT'
import hashlib,pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1]); archive=root/'frozen-campaign.tar.gz'
with archive.open('rb') as stream:
    if hashlib.file_digest(stream,'sha256').hexdigest()!=sys.argv[2]:
        raise SystemExit('Frozen recovery archive changed')
with tarfile.open(archive,'r:gz') as tar:
    for name in ('stereo_recovery_dispatch.py','processing_schedule_dispatch.py','h34_campaign.py'):
        member=tar.getmember('source/scripts/bench/'+name)
        if not member.isfile() or member.size>1024*1024:
            raise SystemExit('Invalid frozen bootstrap member')
        with tar.extractfile(member) as stream:
            (pathlib.Path(sys.argv[3])/name).write_bytes(stream.read())
PYBOOT
exec /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python "$BOOT/stereo_recovery_dispatch.py" execute --campaign "$PERSIST" --archive-sha "$EXPECTED_SHA"
'''


def verified_orb_runtime(archive, build_result):
    """The Atlas-telemetry relink, with its library byte-identical to v1."""
    archive, receipt = Path(archive).resolve(), read(build_result)
    provenance = receipt.get('provenance', {})
    if (digest(archive) != ORB_SHA or receipt.get('status') != 'PASS' or receipt.get('runtime_archive_sha256') != ORB_SHA
            or provenance.get('native_library_unchanged_sha256') != ORB_LIBRARY_SHA
            or provenance.get('atlas_telemetry') != ATLAS_TELEMETRY
            or provenance.get('original_runtime_archive_sha256') != BASE_ORB_SHA
            or provenance.get('native_algorithm_sources_modified') is not False):
        raise ValueError('verified Atlas-telemetry ORB relink of the unchanged native library required')
    return archive, receipt


def prepare(repo, scientific_protocol, base, campaign, orb_runtime, orb_build_result):
    repo, base, campaign = Path(repo).resolve(), Path(base).resolve(), PSD.bounded_directory(campaign)
    orb_archive, orb_receipt = verified_orb_runtime(orb_runtime, orb_build_result)
    launcher = PSD.helper(repo / 'scripts/bench/h34_campaign.py')
    _, original, old_protocol = launcher.load_intake(base)
    if original['archive_sha256'] != BASE_ARCHIVE_SHA or original['protocol_sha256'] != BASE_PROTOCOL_SHA:
        raise ValueError('base must be the untouched native controls v2 snapshot')
    quota = PSD.quota_receipt(CAP_BYTES)
    campaign.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='bhl-recovery-freeze-', dir='/tmp') as temporary:
        frozen = Path(temporary)
        launcher.safe_extract(base / 'frozen-campaign.tar.gz', frozen)
        launcher.verify_source(frozen)
        runtime_input = frozen / 'inputs/runtime/orb-runtime.tar.gz'
        if digest(runtime_input) != BASE_ORB_SHA:
            raise ValueError('base ORB runtime input differs from the v1 runtime being replaced')
        PSD.source_overlay(repo, frozen / 'source')
        shutil.copyfile(scientific_protocol, frozen / 'inputs/scientific-protocol.json')
        shutil.copyfile(orb_archive, runtime_input)
        if digest(runtime_input) != ORB_SHA:
            raise ValueError('copied ORB runtime differs')
        protocol = dict(schema_version=1, campaign_id=campaign.name, runtime=old_protocol['runtime'],
            scientific_scope='48 new paired closed-loop native stereo recovery-policy episodes; same allocation, 6-cell fresh smoke; no hardware claims',
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
        (campaign / 'stereo_recovery.sbatch').write_text(bootstrap_text())
        intake = dict(status='FROZEN_NOT_SUBMITTED', created_utc=now(),
            archive_sha256=digest(archive), protocol_sha256=digest(campaign / 'protocol.json'),
            launcher_sha256=digest(campaign / 'h34_campaign.sbatch'),
            coordinator_launcher_sha256=digest(campaign / 'stereo_recovery.sbatch'),
            sif_sha256=digest(protocol['runtime']['sif']), source_files=len(manifest),
            archive_bytes=archive.stat().st_size, uncompressed_input_bytes=total,
            source_commit=launcher.git_output(repo, 'rev-parse', 'HEAD'),
            source_status=launcher.git_output(repo, 'status', '--porcelain'),
            base_archive_sha256=BASE_ARCHIVE_SHA, base_protocol_sha256=BASE_PROTOCOL_SHA,
            upstream_provenance=original['submodules'], scientific_protocol_sha256=digest(scientific_protocol),
            orb_runtime_archive_sha256=ORB_SHA, replaced_orb_runtime_archive_sha256=BASE_ORB_SHA,
            orb_runtime_build_receipt_sha256=digest(orb_build_result), orb_native_library_sha256=ORB_LIBRARY_SHA,
            orb_wrapper_sha256=orb_receipt['provenance']['wrapper_sha256'],
            max_campaign_bytes=CAP_BYTES, max_output_bytes_per_shard=MAX_OUTPUT_MB * 1024**2,
            new_scored_episodes=DECLARED_CELLS, fresh_smoke_episodes=SMOKE_EPISODES, quota_before_freeze=quota,
            resources=RESOURCES)
        write(campaign / 'intake.json', intake)
    return intake


def verified_intake(campaign, expected=None):
    campaign = PSD.bounded_directory(campaign)
    launcher = PSD.helper(Path(__file__).with_name('h34_campaign.py'))
    _, intake, protocol = launcher.load_intake(campaign)
    if expected is not None and expected != intake['archive_sha256']:
        raise ValueError('submitted archive differs from intake')
    if (intake.get('coordinator_launcher_sha256') != digest(campaign / 'stereo_recovery.sbatch') or
            intake.get('max_campaign_bytes') != CAP_BYTES or protocol['jobs'] != declared_jobs(campaign)
            or intake.get('orb_runtime_archive_sha256') != ORB_SHA):
        raise ValueError('launch resources, declared pair inventory, runtime or coordinator changed')
    return launcher, intake, protocol


def submit_command(campaign, archive_sha):
    r = RESOURCES
    return ['/apps/slurm/25.11.8/bin/sbatch', '--parsable', '--ntasks=1', '--account=eecs',
        f'--job-name={JOB_NAME}', '--partition=' + r['partition'], f"--cpus-per-task={r['cpus']}",
        f"--mem={r['memory_gb']}G", '--time=' + r['time_limit'], '--gres=gpu:1', '--constraint=' + r['constraint'],
        '--export=NONE', '--no-requeue',
        '--output=' + str(campaign / 'stereo-recovery-%j.out'), '--error=' + str(campaign / 'stereo-recovery-%j.out'),
        str(campaign / 'stereo_recovery.sbatch'), str(campaign), archive_sha]


def submit(campaign):
    campaign = PSD.bounded_directory(campaign)
    _, intake, _ = verified_intake(campaign)
    quota = PSD.quota_receipt(CAP_BYTES)
    if (campaign / 'submission-start.json').exists():
        raise ValueError('an attempt was already submitted; no implicit retry')
    command = submit_command(campaign, intake['archive_sha256'])
    write(campaign / 'submission-start.json', dict(created_utc=now(), command=command, quota=quota))
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'APPTAINERENV_'))}
    env['TMPDIR'] = '/tmp'
    proc = subprocess.run(command, capture_output=True, text=True, env=env)
    match = re.fullmatch(r'(\d+)(?:;[^\n]+)?\n?', proc.stdout)
    result = dict(status='SUBMITTED' if proc.returncode == 0 and match else 'SUBMISSION_FAILED',
        job_id=match[1] if proc.returncode == 0 and match else None, submitted_utc=now(),
        returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
        archive_sha256=intake['archive_sha256'], resources=RESOURCES, command=command)
    write(campaign / 'submission.json', result)
    return result


def smoke_gate(result):
    if (result.get('status') != 'PASS' or result.get('phase') != 'smoke' or
            result.get('episodes') != SMOKE_EPISODES or result.get('execution_gate') is not True):
        raise ValueError('complete fresh 6-episode smoke required')


def retain_collector_inputs(output, destination, result):
    """Keep hash-linked numerical evidence and native frame logs; PNGs stay in raw archives."""
    output, destination = Path(output).resolve(), Path(destination)
    files = {'campaign_result.json': output / 'campaign_result.json'}
    for relative in result['episode_files']:
        episode = (output / relative).resolve()
        if not episode.is_relative_to(output) or episode.is_symlink():
            raise ValueError('collector input escapes raw episode output')
        if digest(episode) != result['episode_sha256'][relative]:
            raise ValueError('collector episode differs from scientific result')
        row = read(episode)
        files[relative] = episode
        for name, field in EVIDENCE:
            source = episode.parent / name
            if source.is_symlink() or not source.resolve().is_relative_to(output) or digest(source) != row[field]:
                raise ValueError('collector raw evidence differs from episode lineage')
            files[str(source.relative_to(output))] = source
    if sum(path.stat().st_size for path in files.values()) > 256 * 1024**2:
        raise ValueError('compact numerical evidence exceeds 256 MiB per pair')
    destination.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for name, source in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inventory[name] = dict(bytes=target.stat().st_size, sha256=digest(target))
        if inventory[name]['sha256'] != digest(source):
            raise ValueError('collector evidence changed while copying')
    return inventory


def execute(campaign, expected_sha):
    campaign = PSD.bounded_directory(campaign)
    launcher, intake, protocol = verified_intake(campaign, expected_sha)
    job_id = os.environ.get('SLURM_JOB_ID')
    if not job_id or not os.environ.get('SLURM_JOB_GPUS', os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('new execution requires an allocated Slurm GPU')
    if read(campaign / 'submission.json').get('job_id') != job_id:
        raise ValueError('allocation differs from the unique submission')
    write(campaign / 'execution-start.json', dict(started_utc=now(), job_id=job_id,
        host=os.uname().nodename, archive_sha256=expected_sha,
        allocated_gpu_mask=os.environ.get('CUDA_VISIBLE_DEVICES'), slurm_job_gpus=os.environ.get('SLURM_JOB_GPUS'),
        slurm_cpus_per_task=os.environ.get('SLURM_CPUS_PER_TASK'),
        gpu_query=subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'],
                                 capture_output=True, text=True).stdout))
    results, attempted, error = [], [], None
    try:
        for index, job in enumerate(protocol['jobs']):
            used = sum(path.stat().st_size for path in campaign.rglob('*') if path.is_file())
            if used + 2 * GIB > CAP_BYTES:
                raise ValueError('bounded 40 GiB campaign output reserve exhausted')
            PSD.quota_receipt(2 * GIB)
            if index >= 1:
                qualified = read(campaign / 'qualified-smoke.json')
                smoke_gate(qualified)
                original_smoke, _, _ = PSD.checked_completion(campaign / (SMOKE_NAME + '-' + job_id), expected_sha)
                if qualified != original_smoke:
                    raise ValueError('qualified smoke differs from immutable completed smoke')
            attempted.append(job['name'])
            rc = launcher.execute(campaign, job['name'], expected_sha)
            directory = campaign / (job['name'] + '-' + job_id)
            result, completion, launch = PSD.checked_completion(directory, expected_sha)
            if rc:
                raise ValueError('native runner failed its execution gate')
            if index == 0:
                smoke_gate(result)
                write(campaign / 'qualified-smoke.json', result)
            elif job['kind'] == 'run':
                if (result.get('phase') != 'development' or result.get('expected_declared_cells') != PAIR_EPISODES or
                        result.get('episodes', -1) + len(result.get('unrun_cells', [])) != PAIR_EPISODES):
                    raise ValueError('matched pair requires two completed or explicitly gated cells')
                inventory = retain_collector_inputs(Path(launch['scratch']) / 'output',
                    campaign / 'collector-inputs' / job['name'], result)
                write(directory / 'collector-inputs.json', inventory)
            elif (result.get('schema') != 'bhl-stereo-recovery-v1-collection' or result.get('declared_cells') != DECLARED_CELLS or
                  result.get('measured_cells', -1) + result.get('unrun_after_nominal_gate', -1) != DECLARED_CELLS):
                raise ValueError('final strict collector did not account for all 48 declared cells')
            results.append(dict(name=job['name'], status=result['status'], episodes=result.get('episodes', 0),
                                result_sha256=digest(directory / 'campaign_result.json'),
                                raw_archive_sha256=digest(directory / 'outputs.tar.gz')))
            PSD.retain_and_retire_scratch(directory, launch, completion)
            write(campaign / f'progress-{index:02d}.json', dict(updated_utc=now(), completed=list(results)))
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
    final = dict(schema='bhl-stereo-recovery-dispatch-v1', finished_utc=now(),
        status='COMPLETE' if not error else 'INCOMPLETE', execution_error=error,
        archive_sha256=expected_sha, job_id=job_id, host=os.uname().nodename,
        completed=list(results), attempted=attempted,
        unrun=[job['name'] for job in protocol['jobs'] if job['name'] not in attempted],
        scientific_collection=str(campaign / (COLLECTION_NAME + '-' + job_id) / 'campaign_result.json') if not error else None,
        scope='execution completeness; paired outcomes, explicit unrun gates and the predeclared analysis are in the strict collector')
    write(campaign / 'dispatch-result.json', final)
    return 0 if not error else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--repo', required=True, type=Path)
    prep.add_argument('--scientific-protocol', required=True, type=Path)
    prep.add_argument('--orb-runtime', required=True, type=Path)
    prep.add_argument('--orb-build-result', required=True, type=Path)
    prep.add_argument('--base-campaign', type=Path, default=SHARED_ROOT / 'native-sensor-controls-v2')
    prep.add_argument('--campaign', required=True, type=Path)
    submit_p = sub.add_parser('submit')
    submit_p.add_argument('--campaign', required=True, type=Path)
    execute_p = sub.add_parser('execute')
    execute_p.add_argument('--campaign', required=True, type=Path)
    execute_p.add_argument('--archive-sha', required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args.repo, args.scientific_protocol, args.base_campaign, args.campaign,
                                 args.orb_runtime, args.orb_build_result), indent=2))
    elif args.action == 'submit':
        result = submit(args.campaign)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'SUBMITTED' else 1
    else:
        return execute(args.campaign, args.archive_sha)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
