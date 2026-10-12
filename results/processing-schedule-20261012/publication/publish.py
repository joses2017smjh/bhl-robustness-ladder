#!/usr/bin/env python3
"""Publish verified completed schedule artifacts; never execute scientific code."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time
from datetime import datetime, timezone
import urllib.request

ROOT = Path('/nfs/hpc/share/sanchej7/humanoid-research-20261011')
CAMPAIGN = ROOT / 'processing-schedule-v1-20261012'
OPS = ROOT / 'processing-schedule-publication-20261012'
GH = ROOT / 'terrain-confirmation-dispatch-v6/controller-tools/gh'
GH_SHA = '141507c337e8b202ad398550c3b73d72f5af92e86f71665214538a81efd4c409'
SOURCE_SHA = '46eb5d49361cb64f4f18dc8290a7bcb002c21b01aa035e3fe8c8e049949a0dda'
PROTOCOL_SHA = '3a83b9652e9cf4a222107a9b864dbd9a8b2f365863cd55a58f04f6b0010a5a63'
REPO = 'joses2017smjh/bhl-robustness-ladder'
TAG = 'research-campaign-evidence-20261011'
RELEASE_ID = 409576667
SCIENTIFIC_JOB = '21801949'
GIB = 1024**3
POLL_SECONDS = 30
MAX_SECONDS = 14 * 3600
RETRY_DELAYS = (10, 30)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def journal(name, value):
    line = json.dumps(value, sort_keys=True, allow_nan=False) + '\n'
    with (OPS / name).open('a') as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def gh(*arguments, timeout=900):
    command = [str(GH), *map(str, arguments)]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError('gh operation failed: ' + proc.stderr[-2000:])
    return proc.stdout


def api(endpoint):
    return json.loads(gh('api', endpoint, timeout=120))


def retry(label, operation):
    for attempt in range(3):
        try:
            return operation()
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            journal('operational-attempts.jsonl', dict(operation=label, attempt=attempt + 1,
                    utc=now(), error=f'{type(error).__name__}: {error}'))
            if attempt == 2:
                raise
            time.sleep(RETRY_DELAYS[attempt])


def release_assets():
    assets = []
    for page in range(1, 21):
        rows = api(f'repos/{REPO}/releases/{RELEASE_ID}/assets?per_page=100&page={page}')
        if not isinstance(rows, list):
            raise ValueError('release assets response is not a list')
        assets.extend(rows)
        if len(rows) < 100:
            return assets
    raise ValueError('release inventory exceeded the bounded2000-asset scan')


def verify_remote(asset, sha, size):
    if asset.get('size') != size or asset.get('state') != 'uploaded':
        raise ValueError('remote asset is incomplete or differs in size')
    expected_url = f'https://github.com/{REPO}/releases/download/{TAG}/'
    url = asset.get('browser_download_url', '')
    if not url.startswith(expected_url):
        raise ValueError('remote asset URL leaves the declared repository/release')
    remote_digest = asset.get('digest')
    if remote_digest:
        if remote_digest != 'sha256:' + sha:
            raise ValueError('existing remote asset digest differs; overwrite refused')
        return 'GitHub release API sha256 digest and byte count'
    # Old release assets may lack an API digest. Hash a bounded streamed download.
    total, hasher, started = 0, hashlib.sha256(), time.monotonic()
    request = urllib.request.Request(url, headers={'User-Agent': 'BHL-schedule-evidence-verifier'})
    with urllib.request.urlopen(request, timeout=60) as response:
        for block in iter(lambda: response.read(8 * 1024**2), b''):
            total += len(block)
            if total > size or time.monotonic() - started > 900:
                raise ValueError('remote verification exceeded declared byte/time bounds')
            hasher.update(block)
    if total != size or hasher.hexdigest() != sha:
        raise ValueError('fresh remote asset download differs from the original')
    return 'fresh streamed download sha256 and byte count'


def publish(path, name, sha, size, *, stage, original_result_sha=None):
    path = Path(path)
    if not re.fullmatch(r'[A-Za-z0-9._-]+', name) or size >= 2 * GIB:
        raise ValueError('unsafe asset name or GitHub2GiB asset limit reached')
    if path.is_symlink() or path.stat().st_size != size or digest(path) != sha:
        raise ValueError('publication input differs from its original hash')
    def once():
        matches = [row for row in release_assets() if row.get('name') == name]
        if len(matches) > 1:
            raise ValueError('duplicate remote asset names')
        if not matches:
            link = OPS / 'staging' / name
            if link.exists():
                if not os.path.samefile(link, path):
                    raise ValueError('staging path is not the intended original hardlink')
            else:
                os.link(path, link)
            try:
                # No --clobber. Concurrent existing-name creation is rechecked on retry.
                gh('release', 'upload', TAG, str(link), '--repo', REPO)
            finally:
                link.unlink(missing_ok=True)
            matches = [row for row in release_assets() if row.get('name') == name]
        if len(matches) != 1:
            raise ValueError('uploaded asset absent from release inventory')
        asset = matches[0]
        verification = verify_remote(asset, sha, size)
        if digest(path) != sha:
            raise ValueError('original artifact changed during publication')
        return dict(schema='bhl-processing-schedule-publication-v1', stage=stage,
            source_archive_sha256=SOURCE_SHA, scientific_job_id=SCIENTIFIC_JOB,
            local_original=str(path), original_result_sha256=original_result_sha,
            asset_name=name, asset_id=asset['id'], url=asset['browser_download_url'],
            sha256=sha, bytes=size, verified_utc=now(), verification=verification)
    receipt = retry('publish:' + name, once)
    journal('publication-manifest.jsonl', receipt)
    return receipt


def checked_stage(job):
    directory = CAMPAIGN / (job['name'] + '-' + SCIENTIFIC_JOB)
    completion_path = directory / 'completion.json'
    if not completion_path.exists():
        return None
    completion, launch = read(completion_path), read(directory / 'launch.json')
    if (completion.get('archive_sha256') != SOURCE_SHA or launch.get('archive_sha256') != SOURCE_SHA or
            launch.get('job') != job or str(launch.get('job_id')) != SCIENTIFIC_JOB):
        raise ValueError('completed stage source or declared job identity differs')
    if completion.get('status') not in ('PASS', 'NEGATIVE') or completion.get('exit_status') != 0 or completion.get('error'):
        return None  # Explicit incomplete evidence is retained locally, never called complete.
    for name, spec in completion['files'].items():
        path = directory / name
        if Path(name).name != name or path.is_symlink() or path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
            raise ValueError('completed stage file changed: ' + name)
    result = read(directory / 'campaign_result.json')
    if result.get('status') != completion['status']:
        raise ValueError('stage result differs from completion status')
    original = directory / 'outputs.tar.gz'
    expected = completion['files']['outputs.tar.gz']
    return directory, original, expected, completion['files']['campaign_result.json']['sha256']


def compact_bundle(jobs, publications, dispatch):
    if dispatch.get('status') != 'COMPLETE' or len(dispatch.get('completed', [])) != len(jobs):
        raise ValueError('only complete27-stage execution can produce the final compact bundle')
    if dispatch.get('archive_sha256') != SOURCE_SHA or dispatch.get('unrun'):
        raise ValueError('final dispatch source differs or stages remain unrun')
    files = {}
    for name in ('intake.json', 'manifest.json', 'protocol.json', 'submission.json',
                 'execution-start.json', 'qualified-smoke.json', 'dispatch-result.json'):
        files[name] = CAMPAIGN / name
    for path in CAMPAIGN.glob('progress-*.json'):
        files[path.name] = path
    for job in jobs:
        directory = CAMPAIGN / (job['name'] + '-' + SCIENTIFIC_JOB)
        checked_stage(job)
        for name in ('campaign_result.json', 'completion.json', 'launch.json', 'collector-inputs.json', 'scratch-retirement.json'):
            path = directory / name
            if path.exists():
                files[str(path.relative_to(CAMPAIGN))] = path
    for path in (CAMPAIGN / 'collector-inputs').rglob('*'):
        if path.is_file():
            files[str(path.relative_to(CAMPAIGN))] = path
    if any(path.is_symlink() or not path.resolve().is_relative_to(CAMPAIGN) for path in files.values()):
        raise ValueError('compact evidence escapes the declared campaign')
    raw_bytes = sum(path.stat().st_size for path in files.values())
    if raw_bytes > 512 * 1024**2:
        raise ValueError('final compact evidence exceeds512MiB')
    campaign_bytes = sum(p.stat().st_size for p in CAMPAIGN.rglob('*') if p.is_file())
    ops_bytes = sum(p.stat().st_size for p in OPS.rglob('*') if p.is_file())
    if campaign_bytes + ops_bytes + raw_bytes > 100 * GIB:
        raise ValueError('publication would exceed100GiB total campaign+operations reserve')
    manifest = {name: dict(bytes=path.stat().st_size, sha256=digest(path)) for name, path in sorted(files.items())}
    manifest_path = OPS / 'compact-file-manifest.json'
    write(manifest_path, dict(schema='bhl-processing-schedule-compact-files-v1', files=manifest,
        source_archive_sha256=SOURCE_SHA, raw_input_bytes=raw_bytes, created_utc=now()))
    archive = OPS / 'final-compact-evidence.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(manifest_path, arcname='compact-file-manifest.json', recursive=False)
        for name, path in sorted(files.items()):
            tar.add(path, arcname=name, recursive=False)
    # Hash every archived member before publication, not just the outside tar.
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        expected = {**manifest, 'compact-file-manifest.json': dict(bytes=manifest_path.stat().st_size, sha256=digest(manifest_path))}
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError('compact archive inventory differs')
        for member in members:
            if not member.isfile() or member.size != expected[member.name]['bytes']:
                raise ValueError('compact archive member differs')
            with tar.extractfile(member) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected[member.name]['sha256']:
                    raise ValueError('compact archive member bytes differ')
    sha = digest(archive)
    return publish(archive, 'processing-schedule-20261012-final-compact-' + sha[:12] + '.tar.gz',
                   sha, archive.stat().st_size, stage='final-compact')


def main():
    if Path(__file__).resolve() != OPS / 'publish.py' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('publisher must run from its durable declared source in a Slurm allocation')
    if digest(GH) != GH_SHA:
        raise ValueError('pinned GitHub CLI changed')
    if digest(CAMPAIGN / 'frozen-campaign.tar.gz') != SOURCE_SHA or digest(CAMPAIGN / 'protocol.json') != PROTOCOL_SHA:
        raise ValueError('frozen scientific source changed before publication')
    release = retry('release-metadata', lambda: api(f'repos/{REPO}/releases/tags/{TAG}'))
    if release.get('id') != RELEASE_ID or release.get('tag_name') != TAG or release.get('prerelease') is not True:
        raise ValueError('existing target prerelease identity changed')
    intake, protocol = read(CAMPAIGN / 'intake.json'), read(CAMPAIGN / 'protocol.json')
    if intake.get('archive_sha256') != SOURCE_SHA or intake.get('protocol_sha256') != PROTOCOL_SHA:
        raise ValueError('intake differs from pinned campaign')
    jobs = protocol['jobs']
    if len(jobs) != 27:
        raise ValueError('unexpected scientific stage inventory')
    (OPS / 'staging').mkdir(exist_ok=True)
    write(OPS / 'execution-start.json', dict(schema='bhl-processing-schedule-publisher-v1',
        started_utc=now(), publisher_job_id=os.environ['SLURM_JOB_ID'], scientific_job_id=SCIENTIFIC_JOB,
        source_archive_sha256=SOURCE_SHA, publisher_source_sha256=digest(__file__),
        gh_sha256=GH_SHA, repo=REPO, release_id=RELEASE_ID, tag=TAG,
        poll_seconds=POLL_SECONDS, max_seconds=MAX_SECONDS, retry_delays_s=RETRY_DELAYS))
    publications = [publish(CAMPAIGN / 'frozen-campaign.tar.gz',
        'processing-schedule-20261012-source-' + SOURCE_SHA[:12] + '.tar.gz', SOURCE_SHA,
        intake['archive_bytes'], stage='frozen-source')]
    published = set()
    started = time.monotonic()
    while time.monotonic() - started < MAX_SECONDS:
        for job in jobs:
            if job['name'] in published:
                continue
            checked = checked_stage(job)
            if checked is None:
                continue
            _, path, spec, result_sha = checked
            stage = job['name'].removeprefix('processing-schedule-').removesuffix('-20261012')
            name = 'processing-schedule-20261012-' + stage + '-' + spec['sha256'][:12] + '.tar.gz'
            publications.append(publish(path, name, spec['sha256'], spec['bytes'],
                                        stage=job['name'], original_result_sha=result_sha))
            published.add(job['name'])
        dispatch_path = CAMPAIGN / 'dispatch-result.json'
        if dispatch_path.exists():
            dispatch = read(dispatch_path)
            if dispatch.get('status') != 'COMPLETE':
                raise ValueError('scientific dispatcher stopped incomplete; complete stage publications retained')
            if len(published) != len(jobs):
                raise ValueError('complete dispatcher has an unpublished/unverified stage')
            publications.append(compact_bundle(jobs, publications, dispatch))
            final = dict(schema='bhl-processing-schedule-publication-index-v1', status='COMPLETE',
                completed_utc=now(), scientific_job_id=SCIENTIFIC_JOB, publisher_job_id=os.environ['SLURM_JOB_ID'],
                source_archive_sha256=SOURCE_SHA, artifacts=publications,
                scope='simulation evidence; includes original frozen source,27 original stage archives and strict collector numerical evidence; no physical robot or EuRoC raw data')
            index = OPS / 'publication-index.json'
            write(index, final)
            sha = digest(index)
            receipt = publish(index, 'processing-schedule-20261012-publication-index-' + sha[:12] + '.json',
                              sha, index.stat().st_size, stage='publication-index')
            write(OPS / 'completion.json', dict(status='COMPLETE', finished_utc=now(),
                artifacts=len(publications), index=receipt, scientific_dispatch_sha256=digest(dispatch_path)))
            print(json.dumps(dict(status='COMPLETE', artifacts=len(publications), index_url=receipt['url'])), flush=True)
            return
        proc = subprocess.run(['/apps/slurm/25.11.8/bin/sacct', '-X', '-j', SCIENTIFIC_JOB,
            '--noheader', '--parsable2', '--format=JobIDRaw,State'], capture_output=True, text=True, timeout=30)
        rows = [line.split('|') for line in proc.stdout.splitlines() if line.startswith(SCIENTIFIC_JOB + '|')]
        if rows and rows[0][1].split()[0] in ('FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY'):
            raise ValueError('scientific allocation ended without a complete dispatch receipt: ' + rows[0][1])
        time.sleep(POLL_SECONDS)
    raise TimeoutError('bounded publisher observation time expired')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        path = OPS / 'failure.json'
        if not path.exists():
            write(path, dict(status='INCOMPLETE', finished_utc=now(), error=f'{type(error).__name__}: {error}',
                source_archive_sha256=SOURCE_SHA, scientific_job_id=SCIENTIFIC_JOB,
                publisher_job_id=os.environ.get('SLURM_JOB_ID'), scientific_rerun=False))
        raise
