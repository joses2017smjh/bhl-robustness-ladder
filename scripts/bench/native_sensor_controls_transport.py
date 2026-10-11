#!/usr/bin/env python3
"""Bounded node-local packaging for the frozen October 11 native controls.

The scientific archive, protocol, arguments, environment and smoke are unchanged.
Only the output archive budget is overridden by an explicitly pinned operational
plan. Shared storage receives compact receipts; raw bytes are independently
download-verified on release storage before a worker can complete successfully.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import time

GIB = 1024**3
SHARED_ROOT = Path('/nfs/hpc/share/sanchej7/humanoid-research-20261011')
CAMPAIGN = SHARED_ROOT / 'native-sensor-controls-v2'
TAG = 'research-campaign-evidence-20261011'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load(path):
    spec = importlib.util.spec_from_file_location('native_transport_' + Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked(path, spec):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
        raise ValueError('file identity differs: ' + path.name)


def extract_output(archive, destination, raw_cap_bytes=8*GIB):
    """Separate output budget from the unchanged 2 GiB frozen-source guard."""
    if not 0 < raw_cap_bytes <= 8*GIB:
        raise ValueError('output extraction cap exceeds operational bound')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, 'r:gz') as tar:
        members, seen, total = tar.getmembers(), set(), 0
        for member in members:
            name = PurePosixPath(member.name)
            if (not member.name or name.is_absolute() or '..' in name.parts or
                    str(name) != member.name or member.name in seen or
                    not member.isfile() or member.size > 512*1024**2):
                raise ValueError('unsafe, duplicate or oversized output member')
            seen.add(member.name)
            total += member.size
        if total > raw_cap_bytes:
            raise ValueError('raw output exceeds extraction budget')
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
    return total


def publish_file(path, name, remote):
    # Retry transport errors only. Each attempt rechecks release identity, so an
    # upload whose acknowledgement was lost is adopted rather than overwritten.
    for attempt in range(3):
        try:
            return _publish_file_once(path, name, remote)
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def _publish_file_once(path, name, remote):
    """Upload immutable bytes and verify a separate download; never clobber."""
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError('unsafe asset name')
    path = Path(path)
    spec = dict(bytes=path.stat().st_size, sha256=digest(path))
    release = read_release(remote)
    assets = [a for a in release['assets'] if a['name'] == name]
    with tempfile.TemporaryDirectory(prefix='bhl-native-upload-', dir='/tmp') as temporary:
        work = Path(temporary)
        if not assets:
            staged = work / name
            shutil.copyfile(path, staged)
            checked(staged, spec)
            remote.gh('release', 'upload', TAG, staged, '--repo', remote.REPOSITORY)
            staged.unlink()
            assets = [a for a in read_release(remote)['assets'] if a['name'] == name]
        if (len(assets) != 1 or assets[0]['size'] != spec['bytes'] or
                (assets[0].get('digest') and assets[0]['digest'] != 'sha256:' + spec['sha256'])):
            raise ValueError('release asset differs; replacement refused')
        remote.gh('release', 'download', TAG, '--repo', remote.REPOSITORY,
                  '--pattern', name, '--dir', work)
        checked(work / name, spec)
    return dict(asset_name=name, url=assets[0]['browser_download_url'], **spec)


def read_release(remote):
    return json.loads(remote.gh('api', f'repos/{remote.REPOSITORY}/releases/tags/{TAG}'))


def download_file(spec, work, remote):
    """Publish a local filename only after a complete checked network transfer."""
    name = spec['asset_name']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError('unsafe asset name')
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    final = work / name
    if final.exists():
        checked(final, spec)
        return final
    for attempt in range(3):
        with tempfile.TemporaryDirectory(prefix='download-', dir=work) as temporary:
            try:
                remote.gh('release', 'download', TAG, '--repo', remote.REPOSITORY,
                          '--pattern', name, '--dir', temporary)
                received = Path(temporary) / name
                checked(received, spec)
                received.replace(final)
                return final
            except RuntimeError:
                if attempt == 2:
                    raise
        time.sleep(2**attempt)


def publish(archive, prefix, remote, part_bytes=GIB):
    """Record raw concatenation parts with hashes of each part and whole tar."""
    if not 0 < part_bytes <= GIB:
        raise ValueError('invalid publication part bound')
    archive = Path(archive)
    whole = dict(bytes=archive.stat().st_size, sha256=digest(archive))
    parts = []
    with tempfile.TemporaryDirectory(prefix='bhl-native-parts-', dir='/tmp') as temporary:
        with archive.open('rb') as source:
            index = 0
            while source.tell() < whole['bytes']:
                path = Path(temporary) / f'part-{index:03d}'
                remaining = part_bytes
                with path.open('xb') as output:
                    while remaining:
                        block = source.read(min(1024**2, remaining))
                        if not block:
                            break
                        output.write(block)
                        remaining -= len(block)
                name = f'{prefix}-{whole["sha256"][:12]}.tar.gz.part{index:03d}'
                parts.append(publish_file(path, name, remote))
                path.unlink()
                index += 1
    checked(archive, whole)
    return dict(schema='bhl-native-raw-publication-v1', **whole, parts=parts,
                reconstruction='concatenate parts in listed order; verify whole SHA256 before extraction',
                verified_utc=datetime.now(timezone.utc).isoformat(),
                verification='each part independently downloaded and SHA256 verified; whole archive SHA256 retained')


def download(publication, work, remote):
    """Accept new multipart receipts and existing independently verified assets."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if publication.get('schema') != 'bhl-native-raw-publication-v1':
        name = publication['asset_name']
        if Path(name).name != name:
            raise ValueError('unsafe asset name')
        archive = work / name
        mirror = Path(publication.get('local_verified_mirror', '/nonexistent'))
        if not archive.exists():
            if mirror.is_file():
                checked(mirror, publication)
                shutil.copyfile(mirror, archive)
            else:
                download_file(publication, work, remote)
        checked(archive, publication)
        return archive
    archive = work / (publication['sha256'] + '.tar.gz')
    if archive.exists():
        checked(archive, publication)
        return archive
    if not publication.get('parts') or sum(p['bytes'] for p in publication['parts']) != publication['bytes']:
        raise ValueError('incomplete raw publication parts')
    with tempfile.TemporaryDirectory(prefix='assemble-', dir=work) as temporary:
        assembled = Path(temporary) / 'complete.tar.gz'
        with assembled.open('xb') as output:
            for spec in publication['parts']:
                if not 0 < spec['bytes'] <= GIB:
                    raise ValueError('unsafe publication part')
                part = download_file(spec, work, remote)
                with part.open('rb') as source:
                    shutil.copyfileobj(source, output)
                part.unlink()
        checked(assembled, publication)
        assembled.replace(archive)
    return archive


def packaging_job(original, cap_bytes):
    if cap_bytes != 8*GIB:
        raise ValueError('only explicitly bounded 8 GiB operational packaging allowed')
    changed = copy.deepcopy(original)
    changed['max_output_mb'] = cap_bytes // 1024**2
    return changed


def run(plan_path, plan_sha, job_name):
    plan_path = Path(plan_path).resolve()
    if not plan_path.is_relative_to(SHARED_ROOT) or digest(plan_path) != plan_sha:
        raise ValueError('operational plan identity/scope differs')
    plan = read(plan_path)
    if Path(plan['campaign_dir']).resolve() != CAMPAIGN:
        raise ValueError('campaign scope differs')
    for name, sha in plan['source_sha256'].items():
        if Path(name).name != name or digest(plan_path.parent / name) != sha:
            raise ValueError('operational helper changed')
    if Path(__file__).resolve() != plan_path.parent / Path(__file__).name:
        raise ValueError('worker must execute the pinned operational transport')
    caps = plan['transport']
    if caps != dict(raw_cap_bytes=8*GIB, shared_receipt_cap_bytes=16*1024**2, part_bytes=GIB):
        raise ValueError('transport bounds changed')
    original_launcher = load(plan_path.parent / 'h34_campaign.py')
    remote = load(plan_path.parent / 'methods_remote_campaign.py')
    remote.TAG = TAG
    dispatcher = load(plan_path.parent / 'methods_dispatch.py')
    _, intake, protocol = original_launcher.load_intake(CAMPAIGN)
    if intake['archive_sha256'] != plan['archive_sha256'] or intake['protocol_sha256'] != plan['protocol_sha256']:
        raise ValueError('frozen scientific identity differs')
    declared = [j for j in plan['jobs'] if j['name'] == job_name]
    if len(declared) != 1:
        raise ValueError('job not in the complete declared cohort')
    scientific = original_launcher.get_job(protocol, job_name)
    if scientific['kind'] != 'run':
        raise ValueError('only declared scored jobs use this operational transport')
    smoke_id = dispatcher.submitted_id(CAMPAIGN, plan['smoke'], plan['archive_sha256'])
    smoke_dir = CAMPAIGN / (plan['smoke']['name'] + '-' + smoke_id)
    smoke, _ = dispatcher.checked_result(smoke_dir, plan['smoke'], plan['archive_sha256'])
    if dispatcher.scheduler_state(smoke_id) != ('COMPLETED', '0:0') or smoke['status'] != 'PASS':
        raise ValueError('completed same-source smoke required')
    if shutil.disk_usage('/tmp').free < 30*GIB:
        raise ValueError('node-local scratch lacks bounded packaging/publication headroom')
    scheduler_id = os.environ['SLURM_JOB_ID']
    target = CAMPAIGN / f'{job_name}-{scheduler_id}'
    target.mkdir(exist_ok=False)
    local = Path(tempfile.mkdtemp(prefix=f'bhl-native-transport-{scheduler_id}-', dir='/tmp'))
    persistent = local / 'campaign'
    persistent.mkdir()
    for name in ('intake.json', 'protocol.json', 'h34_campaign.sbatch', 'frozen-campaign.tar.gz'):
        shutil.copyfile(CAMPAIGN / name, persistent / name)
    # Import the runner from the frozen archive itself, never current checkout.
    with tarfile.open(persistent / 'frozen-campaign.tar.gz', 'r:gz') as tar:
        member = tar.getmember('source/scripts/bench/h34_campaign.py')
        if not member.isfile() or member.size > 1024**2:
            raise ValueError('invalid frozen runner')
        runner_path = local / 'frozen_runner.py'
        runner_path.write_bytes(tar.extractfile(member).read())
    runner = load(runner_path)
    original_get_job = runner.get_job
    runner.get_job = lambda p, n: packaging_job(original_get_job(p, n), caps['raw_cap_bytes'])
    write(target / 'transport-start.json', dict(
        schema='bhl-native-transport-start-v1', plan_sha256=plan_sha, job_name=job_name,
        source_archive_sha256=plan['archive_sha256'], protocol_sha256=plan['protocol_sha256'],
        original_max_output_mb=scientific.get('max_output_mb', 512),
        operational_max_output_mb=caps['raw_cap_bytes']//1024**2,
        node=os.uname().nodename, local_workspace=str(local),
        scope='packaging budget only; all scientific inputs, arguments, environment and score gates unchanged'))
    rc = runner.execute(persistent, job_name, plan['archive_sha256'])
    job_dir = persistent / f'{job_name}-{scheduler_id}'
    metadata = [p for p in job_dir.iterdir() if p.is_file() and p.name != 'outputs.tar.gz']
    if sum(p.stat().st_size for p in metadata) > caps['shared_receipt_cap_bytes']:
        raise ValueError('compact receipt budget exceeded; node-local evidence retained')
    for path in metadata:
        shutil.copyfile(path, target / path.name)
    completion = read(job_dir / 'completion.json')
    publication = None
    raw = job_dir / 'outputs.tar.gz'
    if raw.exists():
        checked(raw, completion['files']['outputs.tar.gz'])
        publication = publish(raw, f'native-controls-v3-{job_name}-{scheduler_id}', remote, caps['part_bytes'])
    record = dict(schema='bhl-native-transport-result-v1', status=completion['status'],
                  runner_returncode=rc, runtime_exit_status=completion['exit_status'],
                  source_archive_sha256=plan['archive_sha256'], protocol_sha256=plan['protocol_sha256'],
                  original_max_output_mb=scientific.get('max_output_mb', 512),
                  operational_max_output_mb=caps['raw_cap_bytes']//1024**2,
                  publication=publication, plan_sha256=plan_sha,
                  completed_utc=datetime.now(timezone.utc).isoformat())
    write(target / 'remote-result.json', record)
    if rc or completion.get('error') or completion['status'] not in ('PASS', 'NEGATIVE') or not publication:
        raise ValueError('scientific execution or verified publication incomplete; evidence retained')
    # Retain node-local work for inspection; no shared raw output was created.
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run'])
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.plan_sha256, args.job), indent=2))
