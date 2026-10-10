#!/usr/bin/env python3
"""Bounded CPU continuation after the frozen fresh-sensor controller completes.

Fetches only verified published recordings. Methods, conditions and source are
fixed at submission. No parameter selection, implicit rerun or Git push occurs.
Scientific subprocesses execute inside the pinned Ubuntu container, without
GitHub credentials. Raw shards are published and fresh-download verified.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import time

REPOSITORY='joses2017smjh/bhl-robustness-ladder'
TAG='methods-campaign-evidence-20261010'
AUTHORIZED=Path('/nfs/stak/users/sanchej7/humanoid-methods-20261010')
NATIVE_FAULTS=['nominal','imu_clock_plus20ms','imu_clock_plus80ms','lidar_extrinsic_yaw2deg_x2cm',
               'gyro_z_bias_after5s','accel_x_bias_after5s','lidar_keep_half','lidar_keep_quarter']
FUSION_FAULTS=['nominal','lidar_extrinsic_yaw2deg_x2cm','lidar_keep_half','lidar_keep_quarter']
GROUPS=[f'{family}-g{seed}' for family in ('textured_boxes','thin_posts','ramp_step') for seed in (610000,610001)]


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temporary.replace(path)


def gh(*args):
    result=subprocess.run(['gh',*map(str,args)],capture_output=True,env={**os.environ,'TMPDIR':'/tmp'},timeout=900)
    if result.returncode: raise RuntimeError(f'GitHub operation failed: {args[:2]}, exit {result.returncode}')
    return result.stdout


def download(asset,destination):
    if Path(asset['asset_name']).name!=asset['asset_name']: raise ValueError('safe asset name required')
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    gh('release','download',TAG,'--repo',REPOSITORY,'--pattern',asset['asset_name'],'--dir',destination)
    path=destination/asset['asset_name']
    if path.stat().st_size!=asset['bytes'] or digest(path)!=asset['sha256']: raise ValueError('download differs from bound evidence')
    return path


def extract_regular(archive,destination,*,prefix=None):
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    total=0; names=set()
    with tarfile.open(archive,'r:gz') as stream:
        for member in stream:
            name=PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not member.isfile() or member.name in names:
                raise ValueError('regular unique bounded archive members required')
            names.add(member.name)
            if prefix and not member.name.startswith(prefix): continue
            total+=member.size
            if total>12*1024**3 or member.size>1024**3: raise ValueError('extracted recording bound exceeded')
            path=destination/str(name); path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists(): raise ValueError('refuse archive overwrite')
            with stream.extractfile(member) as src,path.open('xb') as dst: shutil.copyfileobj(src,dst)
            path.chmod(0o600)
    return total


def publish(directory,asset_name,receipts):
    directory=Path(directory); receipts=Path(receipts)
    with tempfile.TemporaryDirectory(prefix='bhl-fault-publish-',dir='/tmp') as temporary:
        temporary=Path(temporary); archive=temporary/asset_name
        with tarfile.open(archive,'w:gz',compresslevel=6,dereference=True) as stream:
            for path in sorted(directory.rglob('*')):
                if path.is_symlink(): raise ValueError('published output may not contain links')
                if path.is_file(): stream.add(path,arcname=str(path.relative_to(directory)),recursive=False)
        checksum=digest(archive); size=archive.stat().st_size
        if size>1536*1024**2: raise ValueError('single shard publication exceeds 1.5 GiB')
        release=json.loads(gh('api',f'repos/{REPOSITORY}/releases/tags/{TAG}'))
        existing=[a for a in release['assets'] if a['name']==asset_name]
        if existing: raise ValueError('evidence asset already exists; no implicit rerun/overwrite')
        gh('release','upload',TAG,archive,'--repo',REPOSITORY)
        release=json.loads(gh('api',f'repos/{REPOSITORY}/releases/tags/{TAG}'))
        assets=[a for a in release['assets'] if a['name']==asset_name]
        if len(assets)!=1 or assets[0]['size']!=size: raise ValueError('published asset identity differs')
        if assets[0].get('digest') and assets[0]['digest']!='sha256:'+checksum: raise ValueError('server SHA256 differs')
        record={'asset_name':asset_name,'sha256':checksum,'bytes':size,'url':assets[0]['browser_download_url'],
                'verification':'local archive + server identity + independent fresh download SHA256'}
        download(record,temporary/'verified')
        write(receipts/(asset_name+'.json'),record)
        return record


def source_cell(plan,spec):
    root=Path(plan['sensor_campaign'])
    if digest(root/'protocol.json')!=plan['sensor_protocol_sha256']: raise ValueError('source sensor protocol changed')
    intake=json.loads((root/'intake.json').read_text())
    if intake['archive_sha256']!=plan['sensor_archive_sha256']: raise ValueError('source sensor archive changed')
    submission=json.loads((root/(spec['job_name']+'-submission.json')).read_text())
    if submission.get('status')!='SUBMITTED' or submission['archive_sha256']!=plan['sensor_archive_sha256']:
        raise ValueError('required fresh input was not submitted from frozen source')
    job=str(submission['job_id'])
    if not job.isdecimal(): raise ValueError('actual scheduler identity required')
    directory=root/(spec['job_name']+'-'+job)
    completion=json.loads((directory/'completion.json').read_text())
    if (completion.get('status')!='PASS' or completion.get('exit_status')!=0
            or completion.get('error') or completion['archive_sha256']!=plan['sensor_archive_sha256']):
        raise ValueError('required fresh recording is incomplete')
    if digest(directory/'campaign_result.json')!=completion['files']['campaign_result.json']['sha256']:
        raise ValueError('source recording receipt differs')
    result=json.loads((directory/'campaign_result.json').read_text())
    if (result['phase']!='run' or result['cell']['geometry_group']!=spec['group']
            or result['cell']['condition']!='nominal' or result['capture']['frames']!=150):
        raise ValueError('only predeclared full nominal recording can enter follow-up')
    publication=json.loads((directory/'publication.json').read_text())
    if (publication.get('schema')!='bhl-methods-archive-v1'
            or publication['sha256']!=completion['files']['outputs.tar.gz']['sha256']
            or publication['source_archive_sha256']!=plan['sensor_archive_sha256']):
        raise ValueError('source recording publication lineage differs')
    return publication,result['capture']['manifest_sha256']


def validate_plan(plan_path,expected):
    if digest(plan_path)!=expected: raise ValueError('deferred plan hash differs')
    plan=json.loads(Path(plan_path).read_text()); root=Path(plan['durable_dir']).resolve()
    if (plan.get('schema')!='bhl-sensor-followup-deferred-v1' or not root.is_relative_to(AUTHORIZED)
            or plan['native_faults']!=NATIVE_FAULTS or plan['fusion_faults']!=FUSION_FAULTS
            or [s['group'] for s in plan['inputs']]!=GROUPS or plan['maximum_seconds']>22*3600
            or plan['maximum_seconds']<3600 or plan['dispatcher_sha256']!=digest(__file__)):
        raise ValueError('invalid frozen bounded continuation plan')
    if digest(plan['container'])!=plan['container_sha256']: raise ValueError('scientific container changed')
    return plan,root


def main(plan_path,expected):
    plan, durable=validate_plan(plan_path,expected)
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('allocated CPU job required')
    durable.mkdir(parents=True,exist_ok=True)
    state_path=durable/'state.json'
    if state_path.exists(): raise ValueError('one execution only; failed jobs are never silently rerun')
    state={'schema':'bhl-sensor-followup-state-v1','status':'RUNNING','job_id':os.environ['SLURM_JOB_ID'],
        'plan_sha256':expected,'started_utc':datetime.now(timezone.utc).isoformat(),
        'native_completed':0,'fusion_completed':0,'scientific_scope':'development replay; no physical/closed-loop inference'}
    write(state_path,state)
    started=time.monotonic()
    try:
        work=Path(tempfile.mkdtemp(prefix='bhl-sensor-followup-',dir='/tmp'))
        state['node_local_work']=str(work);write(state_path,state)
        if shutil.disk_usage(work).free<20*1024**3: raise RuntimeError('20 GiB free node-local scratch required')
        archive=download(plan['source'],work/'downloads/source')
        extract_regular(archive,work/'source')
        inventory=json.loads((work/'source/source-manifest.json').read_text())
        for name,checksum in inventory['files_sha256'].items():
            if digest(work/'source'/name)!=checksum: raise ValueError('frozen scientific source differs')
        runtime_archive=download(plan['runtimes'],work/'downloads/runtimes')
        extract_regular(runtime_archive,work/'runtimes')
        for folder,binary in [('lio','fastlio_headless'),('livo','fastlivo_headless')]:
            (work/'runtimes'/folder/binary).chmod(0o700)
        manifests=[]
        for spec in plan['inputs']:
            publication,manifest_hash=source_cell(plan,spec)
            raw=download(publication,work/'downloads'/spec['group'])
            target=work/'captures'/spec['group']; extract_regular(raw,target,prefix='replay/')
            manifest=target/'replay/manifest.json'
            if digest(manifest)!=manifest_hash: raise ValueError('published original manifest differs')
            manifests.append(str(manifest)); raw.unlink()
            write(durable/'inputs'/(spec['group']+'.json'),{'publication':publication,'manifest_sha256':manifest_hash})
        smoke_archive=download(plan['smoke_input'],work/'downloads/smoke')
        extract_regular(smoke_archive,work/'smoke_capture',prefix='replay/')
        if digest(work/'smoke_capture/replay/manifest.json')!=plan['smoke_manifest_sha256']:
            raise ValueError('predeclared smoke capture differs')
        private=work/'private_home'; private.mkdir()
        python=plan['python']; source=work/'source'
        env={k:v for k,v in os.environ.items() if not k.startswith(('APPTAINERENV_','SINGULARITYENV_'))}
        def execute(script,args,label):
            if time.monotonic()-started>plan['maximum_seconds']: raise TimeoutError('bounded continuation time exhausted')
            command=['apptainer','exec','--containall','--cleanenv','--home',str(private),
                '--bind',f'/nfs/hpc/share/sanchej7:/nfs/hpc/share/sanchej7:ro',
                '--bind',f'{work}:{work}','--bind',f'{source}:{source}:ro',
                '--env','OMP_NUM_THREADS=1','--pwd',str(source),plan['container'],python,
                str(source/'scripts/bench'/script),*map(str,args)]
            logfile=durable/'logs'/(label+'.log'); logfile.parent.mkdir(exist_ok=True)
            with logfile.open('x') as stream:
                process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=1200)
            if process.returncode: raise RuntimeError(f'{label} failed with exit {process.returncode}; no automatic rerun')
        protocol=work/'full-protocol.json'; smoke_protocol=work/'smoke-protocol.json'
        runtime_args=['--lio-runtime',work/'runtimes/lio','--livo-runtime',work/'runtimes/livo']
        execute('sensor_fault_campaign.py',['freeze','--manifests',*manifests,*runtime_args,'--output',protocol],'freeze-full')
        execute('sensor_fault_campaign.py',['freeze','--manifests',work/'smoke_capture/replay/manifest.json',*runtime_args,
            '--smoke','--output',smoke_protocol],'freeze-smoke')
        shutil.copyfile(protocol,durable/'full-protocol.json'); shutil.copyfile(smoke_protocol,durable/'smoke-protocol.json')
        # Every declared fault and fusion condition executes in the fresh
        # same-source smoke; no scientific improvement gate is selected.
        for kind,faults,script in [('native',NATIVE_FAULTS,'sensor_fault_campaign.py'),('fusion',FUSION_FAULTS,'stereo_consistency_campaign.py')]:
            for fault in faults:
                label=f'smoke-{kind}-{fault}'; output=work/'smoke'/label
                execute(script,['run','--protocol',smoke_protocol,'--group',GROUPS[0],'--fault',fault,
                    '--output',output,'--work',work/'derived'/label],label)
                result=json.loads((output/'campaign_result.json').read_text())
                if result.get('status')!='PASS' or result.get('frames')!=58: raise ValueError('same-source smoke incomplete')
                if kind=='native':
                    visual=next(r for r in result['rows'] if r['method']=='fast_livo2')
                    if visual['native'].get('frames_with_visual_measurements',0)<1:
                        raise ValueError('smoke did not exercise actual native visual measurements')
                shutil.rmtree(work/'derived'/label)
            smoke_directories=[work/'smoke'/f'smoke-{kind}-{fault}' for fault in faults]
            execute(script,['collect','--protocol',smoke_protocol,'--directories',*smoke_directories,
                '--output',work/'smoke'/(kind+'-collection.json')],f'smoke-{kind}-collect')
            shutil.copyfile(work/'smoke'/(kind+'-collection.json'),durable/('smoke-'+kind+'-collection.json'))
        publish(work/'smoke',f"sensor-followup-{state['job_id']}-all-smokes.tar.gz",durable/'publications')
        write(durable/'smoke-qualification.json',{'status':'PASS','source_sha256':plan['source']['sha256'],
            'native_faults':NATIVE_FAULTS,'fusion_faults':FUSION_FAULTS,'closed_loop_episodes':0})
        state['phase']='full';write(state_path,state)
        for kind,faults,script in [('native',NATIVE_FAULTS,'sensor_fault_campaign.py'),('fusion',FUSION_FAULTS,'stereo_consistency_campaign.py')]:
            directories=[]
            for group in GROUPS:
                for fault in faults:
                    label=f'{kind}-{group}-{fault}'; output=work/'outputs'/label
                    execute(script,['run','--protocol',protocol,'--group',group,'--fault',fault,
                        '--output',output,'--work',work/'derived'/label],label)
                    result=json.loads((output/'campaign_result.json').read_text())
                    if result.get('status')!='PASS' or result.get('frames')!=148: raise ValueError('full predeclared shard incomplete')
                    publication=publish(output,f"sensor-followup-{state['job_id']}-{label}.tar.gz",durable/'publications')
                    compact={k:v for k,v in result.items() if k not in ('rows','files_sha256')}
                    compact.update(raw_result_sha256=digest(output/'campaign_result.json'),raw_publication=publication)
                    write(durable/(label+'.json'),compact)
                    shutil.rmtree(work/'derived'/label)
                    directories.append(output);state[kind+'_completed']+=1;write(state_path,state)
            execute(script,['collect','--protocol',protocol,'--directories',*directories,
                '--output',work/(kind+'-collection.json')],kind+'-collect')
            shutil.copyfile(work/(kind+'-collection.json'),durable/(kind+'-collection.json'))
        state.update(status='PASS',phase='complete',finished_utc=datetime.now(timezone.utc).isoformat(),
                     scientific_status='COMPLETE_PAIRED_DEVELOPMENT_REPLAY_NOT_AUTOMATIC_IMPROVEMENT')
    except Exception as error:
        state.update(status='INCOMPLETE',error=f'{type(error).__name__}: {error}',finished_utc=datetime.now(timezone.utc).isoformat())
        if 'output' in locals() and output.exists():
            try:
                state['failed_output_publication']=publish(output,f"sensor-followup-{state['job_id']}-failed-{label}.tar.gz",durable/'publications')
            except Exception as publication_error:
                state['failure_publication_error']=str(publication_error)
        state['node_local_work_retained_on_failure']=state.get('node_local_work')
        write(state_path,state)
        raise
    write(state_path,state)
    shutil.rmtree(work)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args(); print(json.dumps(main(args.plan,args.plan_sha256),indent=2))
