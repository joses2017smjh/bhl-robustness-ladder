#!/usr/bin/env python3
"""Independent final local/archive/GitHub audit; no publication or simulation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime, timezone

ROOT = Path('/nfs/hpc/share/sanchej7/humanoid-research-20261011')
CAMPAIGN = ROOT / 'processing-schedule-v1-20261012'
OPS = ROOT / 'processing-schedule-publication-20261012'
GH = ROOT / 'terrain-confirmation-dispatch-v6/controller-tools/gh'
REPO = 'joses2017smjh/bhl-robustness-ladder'
RELEASE_ID = 409576667
SOURCE_SHA = '46eb5d49361cb64f4f18dc8290a7bcb002c21b01aa035e3fe8c8e049949a0dda'
PROTOCOL_SHA = '3a83b9652e9cf4a222107a9b864dbd9a8b2f365863cd55a58f04f6b0010a5a63'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    require(os.environ.get('SLURM_JOB_ID'), 'allocated verification required')
    require(digest(GH) == '141507c337e8b202ad398550c3b73d72f5af92e86f71665214538a81efd4c409', 'GitHub CLI pin changed')
    require(digest(CAMPAIGN / 'frozen-campaign.tar.gz') == SOURCE_SHA, 'source archive changed')
    require(digest(CAMPAIGN / 'protocol.json') == PROTOCOL_SHA, 'protocol changed')
    jobs = read(CAMPAIGN / 'protocol.json')['jobs']
    require(len(jobs) == 27, 'unexpected stage inventory')
    dispatch = read(CAMPAIGN / 'dispatch-result.json')
    completion = read(OPS / 'completion.json')
    require(dispatch['status'] == completion['status'] == 'COMPLETE', 'scientific/publication completion missing')
    require(dispatch['archive_sha256'] == SOURCE_SHA and not dispatch['unrun'], 'scientific source/coverage differs')
    require(len(dispatch['completed']) == 27, 'incomplete stage inventory')
    lines = [json.loads(line) for line in (OPS / 'publication-manifest.jsonl').read_text().splitlines()]
    expected_stages = {'frozen-source', 'final-compact', 'publication-index', *(job['name'] for job in jobs)}
    require(len(lines) == len(expected_stages) == 30, 'publication manifest should contain exactly30 distinct assets')
    require({row['stage'] for row in lines} == expected_stages, 'publication stage inventory differs')
    require(len({row['asset_name'] for row in lines}) == 30, 'duplicate asset name')
    require(len({row['asset_id'] for row in lines}) == 30, 'duplicate asset ID')
    by_stage = {row['stage']: row for row in lines}
    total_bytes = 0
    for job in jobs:
        directory = CAMPAIGN / (job['name'] + '-21801949')
        original = directory / 'outputs.tar.gz'
        final = read(directory / 'completion.json')
        launch = read(directory / 'launch.json')
        require(final['status'] in ('PASS', 'NEGATIVE') and final['exit_status'] == 0 and not final['error'], 'incomplete stage')
        require(final['archive_sha256'] == launch['archive_sha256'] == SOURCE_SHA and launch['job'] == job, 'stage source/identity differs')
        for name, expected in final['files'].items():
            require(Path(name).name == name, 'unsafe completion member')
            path = directory / name
            require(not path.is_symlink() and path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'], 'changed completed evidence: ' + name)
        published = by_stage[job['name']]
        require(published['sha256'] == final['files']['outputs.tar.gz']['sha256'], 'published stage archive digest differs')
        require(published['bytes'] == final['files']['outputs.tar.gz']['bytes'], 'published stage size differs')
        require(published['original_result_sha256'] == final['files']['campaign_result.json']['sha256'], 'published result digest differs')
        require(Path(published['local_original']) == original, 'publication original path differs')
        total_bytes += original.stat().st_size
    index_path = OPS / 'publication-index.json'
    index = read(index_path)
    require(index['status'] == 'COMPLETE' and index['source_archive_sha256'] == SOURCE_SHA, 'index source/status differs')
    require(index['artifacts'] == lines[:-1], 'index artifacts differ from append-only receipts')
    require(completion['index'] == lines[-1], 'completion index receipt differs')
    require(digest(index_path) == lines[-1]['sha256'], 'index bytes changed')
    compact = OPS / 'final-compact-evidence.tar.gz'
    require(digest(compact) == by_stage['final-compact']['sha256'], 'compact archive changed')
    inventory = read(OPS / 'compact-file-manifest.json')['files']
    with tarfile.open(compact, 'r:gz') as archive:
        members = archive.getmembers()
        require(len(members) == len(inventory) + 1 and {m.name for m in members} == set(inventory) | {'compact-file-manifest.json'}, 'compact inventory differs')
        for member in members:
            require(member.isfile(), 'compact archive contains a nonfile')
            expected = (dict(bytes=(OPS / 'compact-file-manifest.json').stat().st_size, sha256=digest(OPS / 'compact-file-manifest.json'))
                        if member.name == 'compact-file-manifest.json' else inventory[member.name])
            require(member.size == expected['bytes'], 'compact member size differs')
            with archive.extractfile(member) as stream:
                require(hashlib.file_digest(stream, 'sha256').hexdigest() == expected['sha256'], 'compact member digest differs')
    assets = []
    for page in range(1, 21):
        proc = subprocess.run([str(GH), 'api', f'repos/{REPO}/releases/{RELEASE_ID}/assets?per_page=100&page={page}'],
                              check=True, capture_output=True, text=True, timeout=120)
        rows = json.loads(proc.stdout)
        assets.extend(rows)
        if len(rows) < 100:
            break
    else:
        raise ValueError('release inventory exceeds bounded pages')
    by_id = {row['id']: row for row in assets}
    for row in lines:
        require(row['source_archive_sha256'] == SOURCE_SHA and row['scientific_job_id'] == '21801949', 'publication lineage differs')
        asset = by_id[row['asset_id']]
        require(asset['name'] == row['asset_name'] and asset['size'] == row['bytes'] and asset['state'] == 'uploaded', 'remote asset identity/size differs')
        require(asset.get('digest') == 'sha256:' + row['sha256'], 'fresh GitHub digest differs or unavailable')
        require(asset['browser_download_url'] == row['url'], 'remote artifact URL differs')
    final = dict(schema='bhl-processing-schedule-independent-publication-audit-v1', status='PASS',
        verified_utc=datetime.now(timezone.utc).isoformat(), job_id=os.environ['SLURM_JOB_ID'],
        host=os.uname().nodename, source_archive_sha256=SOURCE_SHA,
        original_stage_archives=27, remote_assets_verified=30, original_archive_bytes=total_bytes,
        compact_members_verified=len(inventory)+1, scientific_dispatch_sha256=digest(CAMPAIGN/'dispatch-result.json'),
        publication_manifest_sha256=digest(OPS/'publication-manifest.jsonl'),
        publication_index_sha256=digest(index_path), publication_index_url=lines[-1]['url'],
        compact_url=by_stage['final-compact']['url'], independent_verifier_sha256=digest(Path(__file__)),
        scope='independent byte identity/completeness audit; scientific results remain those of the frozen collector')
    with (OPS / 'independent-audit.json').open('x') as stream:
        json.dump(final, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(final, indent=2), flush=True)


if __name__ == '__main__':
    main()
