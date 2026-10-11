#!/usr/bin/env python3
"""Verify and publish one frozen fusion campaign and retire only its freshly verified remote duplicate."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tarfile

sys.path.insert(0,str(Path(__file__).parent))
import methods_remote_campaign as REMOTE

ROOT=Path('/nfs/hpc/share/sanchej7/humanoid-research-20261011')
TAG='research-campaign-evidence-20261011'


def publish(persistent, job, archive_sha):
    persistent=Path(persistent).resolve()
    if not persistent.is_relative_to(ROOT):raise ValueError('publication requires new shared campaign root')
    intake=json.loads((persistent/'intake.json').read_text())
    if intake['archive_sha256']!=archive_sha or REMOTE.digest(persistent/'frozen-campaign.tar.gz')!=archive_sha:raise ValueError('source archive differs')
    submission=json.loads((persistent/(job+'-submission.json')).read_text())
    if submission['status']!='SUBMITTED' or submission['archive_sha256']!=archive_sha:raise ValueError('submission pin differs')
    folder=persistent/(job+'-'+submission['job_id'])
    completion=json.loads((folder/'completion.json').read_text());launch=json.loads((folder/'launch.json').read_text())
    if completion['archive_sha256']!=archive_sha or launch['archive_sha256']!=archive_sha or launch['job_id']!=submission['job_id']:raise ValueError('completed source/job differs')
    for name,spec in completion['files'].items():
        if '/' in name or name in ('.','..'):raise ValueError('unsafe completion artifact')
        REMOTE.checked_file(folder/name,spec)
    spec=completion['files']['outputs.tar.gz'];name='fusion-occlusion-20261011-'+submission['job_id']+'.tar.gz'
    REMOTE.TAG=TAG
    publication=REMOTE.publish_asset(folder/'outputs.tar.gz',name,spec)
    publication.update(source_archive_sha256=archive_sha,job_id=submission['job_id'],
        execution_status=completion['status'],publication_scope='Raw evidence only; no inferred gain or scientific pass')
    result=json.loads((folder/'campaign_result.json').read_text())
    compact=folder/'collected';compact.mkdir(exist_ok=False)
    with tarfile.open(folder/'outputs.tar.gz','r:gz') as archive:
        for phase in result.get('completed_phases',[]):
            name=phase['collection']
            if name not in ('smoke-collection.json','dev-collection.json','test-collection.json'):raise ValueError('invalid collection name')
            member=archive.getmember(name)
            if not member.isfile() or member.size>32*1024**2:raise ValueError('invalid compact collection member')
            payload=archive.extractfile(member).read()
            target=compact/name;target.write_bytes(payload)
            if REMOTE.digest(target)!=phase['sha256']:raise ValueError('retained collection hash differs')
    publication.update(local_packaging_retired=False,retirement_authorized_after_fresh_download=True,
        compact_collections={p.name:REMOTE.digest(p) for p in compact.iterdir() if p.is_file()})
    REMOTE.write(folder/'publication.json',publication)
    # Only the freshly uploaded and independently downloaded identical duplicate
    # is retired. Source archive, compact science and all provenance remain.
    (folder/'outputs.tar.gz').unlink()
    publication['local_packaging_retired']=True
    temporary=folder/'publication.json.tmp';temporary.write_text(json.dumps(publication,indent=2)+'\n');os.replace(temporary,folder/'publication.json')
    import methods_dispatch as dispatch
    dispatch.AUTHORIZED_ROOT=ROOT
    dispatch.release_output(persistent,job)
    return publication


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--persistent',type=Path,required=True);parser.add_argument('--job',required=True);parser.add_argument('--archive-sha',required=True)
    args=vars(parser.parse_args());print(json.dumps(publish(**args),indent=2))
