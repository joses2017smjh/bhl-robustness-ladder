#!/usr/bin/env python3
"""Publish bounded terminal metadata for an already-frozen sensor continuation.

No simulation, estimator, collector or training runs here. A complete verdict
requires terminal scheduler success, matching source/input lineage, complete
strict collector matrices, and existing checksum-matched raw release assets.
Negative scientific cells remain negative; missing evidence stays incomplete.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

AUTHORIZED=Path('/nfs/stak/users/sanchej7/humanoid-methods-20261010')
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
METHODS=['fast_lio2','fast_livo2','fast_livo2_no_visual']
ARMS=['sgbm','sgbm_local_cost','lidar_sparse','simple_projected_fusion','cost_consistency_fusion']
METADATA_LIMIT=32*1024**2


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def read(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>8*1024**2:raise ValueError('regular bounded metadata file required')
    return json.loads(path.read_text())


def scheduler(job):
    if not str(job).isdecimal():raise ValueError('numeric parent scheduler job required')
    result=subprocess.run(['sacct','-X','-j',str(job),'--noheader','--parsable2','--format=JobIDRaw,State,ExitCode'],capture_output=True,text=True,timeout=60)
    if result.returncode:raise RuntimeError('scheduler accounting unavailable')
    rows=[line.split('|') for line in result.stdout.splitlines() if line.strip()]
    rows=[row for row in rows if row[0]==str(job)]
    if len(rows)!=1:raise ValueError('one exact parent accounting record required')
    return {'job_id':str(job),'state':rows[0][1].split()[0].rstrip('+'),'exit_code':rows[0][2]}


def validate_bootstrap(path,expected):
    path=Path(path).resolve()
    if digest(path)!=expected:raise ValueError('frozen observer plan hash differs')
    plan=read(path);root=path.parent
    if (plan.get('schema')!='bhl-sensor-followup-observer-plan-v1' or not root.is_relative_to(AUTHORIZED)
            or Path(plan['durable_dir']).resolve()!=root or not Path(plan['parent_dir']).resolve().is_relative_to(AUTHORIZED)
            or not str(plan['parent_job_id']).isdecimal() or plan['metadata_limit_bytes']!=METADATA_LIMIT
            or set(plan['source_sha256'])!={'sensor_followup_observer.py','sensor_followup_deferred.py'}
            or plan['source_sha256'].get('sensor_followup_observer.py')!=digest(__file__)):
        raise ValueError('invalid frozen observer scope')
    for name,checksum in plan['source_sha256'].items():
        if Path(name).name!=name or digest(root/name)!=checksum:raise ValueError('observer helper source changed')
    spec=importlib.util.spec_from_file_location('sensor_followup_observer_publisher',root/'sensor_followup_deferred.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return plan,root,helper


def check_collections(parent,plan):
    """Verify complete identities and lineage; never silently discard negatives."""
    parent=Path(parent);issues=[];facts={};groups=plan['groups'];native_faults=plan['native_faults'];fusion_faults=plan['fusion_faults']
    try:
        source=read(parent/'plan.json')
        if digest(parent/'plan.json')!=plan['parent_plan_sha256']:raise ValueError('parent frozen plan changed')
        if source['source']['sha256']!=plan['science_source_archive_sha256']:raise ValueError('scientific source archive differs')
        protocol=read(parent/'full-protocol.json')
        if (protocol.get('schema')!='bhl-sensor-fault-protocol-v1' or protocol.get('phase')!='run'
                or protocol.get('source_sha256')!=plan['science_protocol_source_sha256']
                or protocol.get('faults')!=plan['fault_specifications'] or protocol.get('methods')!=METHODS
                or protocol.get('gate')!=plan['gate']):raise ValueError('strict full scientific protocol differs')
        for name,expected_runtime in plan['runtime_identity'].items():
            actual={k:v for k,v in protocol['runtimes'][name].items() if k!='path'}
            if actual!=expected_runtime:raise ValueError('native runtime identity differs')
        entries=protocol['inputs']
        if len(entries)!=len(groups) or {e['group'] for e in entries}!=set(groups):raise ValueError('full input groups differ')
        for entry in entries:
            receipt=read(parent/'inputs'/(entry['group']+'.json'))
            if (entry.get('source_frames')!=150 or entry.get('sequence')!=entry['group']+'-nominal'
                    or entry['manifest_sha256']!=receipt['manifest_sha256']
                    or receipt['publication']['source_archive_sha256']!=plan['sensor_source_archive_sha256']):
                raise ValueError('original nominal input lineage differs')
        facts['protocol_sha256']=digest(parent/'full-protocol.json')
    except Exception as error:issues.append('source/input protocol: '+str(error))
    try:
        native=read(parent/'native-collection.json')
        expected={(g,f,m) for g in groups for f in native_faults for m in METHODS}
        rows=native['paired_group_rows'];identities=[(r['group'],r['fault'],r['method']) for r in rows]
        if (native.get('schema')!='bhl-sensor-fault-collection-v1' or native.get('status')!='PASS' or native.get('phase')!='run'
                or native.get('protocol_sha256')!=facts.get('protocol_sha256')
                or native.get('expected_cells')!=48 or native.get('verified_cells')!=48
                or native.get('missing')!=[] or native.get('problems')!=[]
                or len(identities)!=len(expected) or set(identities)!=expected):raise ValueError('native comparison is incomplete or duplicated')
        if any(r['nominal_gate'] not in ('PASS','NEGATIVE') or r['fault_gate'] not in ('PASS','NEGATIVE')
               or not math.isfinite(r['tracking_fraction_delta'])
               or (r['conditional_ate_rmse_delta_m'] is not None and not math.isfinite(r['conditional_ate_rmse_delta_m'])) for r in rows):
            raise ValueError('invalid native comparison values')
        facts['native']={'verified_shards':48,'estimator_cells':144,
            'qualified_cells':sum(r['fault_gate']=='PASS' for r in rows),'negative_cells':sum(r['fault_gate']=='NEGATIVE' for r in rows),
            'conditional_ate_unavailable_cells':sum(r['conditional_ate_rmse_delta_m'] is None for r in rows),
            'collector_sha256':digest(parent/'native-collection.json')}
    except Exception as error:issues.append('native collector: '+str(error))
    try:
        fusion=read(parent/'fusion-collection.json');expected={(g,f,a) for g in groups for f in fusion_faults for a in ARMS}
        rows=fusion['group_metrics'];identities=[(r['group'],r['fault'],r['arm']) for r in rows]
        if (fusion.get('schema')!='bhl-stereo-consistency-collection-v1' or fusion.get('status')!='PASS' or fusion.get('phase')!='run'
                or fusion.get('expected_cells')!=24 or fusion.get('verified_cells')!=24
                or fusion.get('missing')!=[] or fusion.get('problems')!=[]
                or len(identities)!=len(expected) or set(identities)!=expected):raise ValueError('fusion comparison is incomplete or duplicated')
        for row in rows:
            for key in ('mae_m','coverage','near_obstacle_pixel_recall'):
                v=row[key]
                if v is not None and (not isinstance(v,(int,float)) or not math.isfinite(v) or v<0):raise ValueError('invalid fusion metric')
                if key!='mae_m' and v is not None and v>1:raise ValueError('invalid fusion fraction')
            for key in ('unknown_obstacle_pixels','false_free_obstacle_pixels'):
                if type(row[key]) is not int or row[key]<0:raise ValueError('invalid unresolved/false-free accounting')
        facts['fusion']={'verified_shards':24,'arm_cells':120,'collector_sha256':digest(parent/'fusion-collection.json')}
    except Exception as error:issues.append('fusion collector: '+str(error))
    return facts,issues


def expected_assets(plan):
    prefix='sensor-followup-'+str(plan['parent_job_id'])+'-'
    return {prefix+'all-smokes.tar.gz',
        *(prefix+'native-'+g+'-'+f+'.tar.gz' for g in plan['groups'] for f in plan['native_faults']),
        *(prefix+'fusion-'+g+'-'+f+'.tar.gz' for g in plan['groups'] for f in plan['fusion_faults'])}


def check_publications(parent,plan,assets):
    issues=[];verified=[];expected=expected_assets(plan)
    by_name={}
    for asset in assets:by_name.setdefault(asset['name'],[]).append(asset)
    for name in sorted(expected):
        try:
            receipt=read(Path(parent)/'publications'/(name+'.json'))
            if receipt['asset_name']!=name or len(by_name.get(name,[]))!=1:raise ValueError('missing or ambiguous asset identity')
            asset=by_name[name][0]
            if asset['size']!=receipt['bytes'] or asset.get('digest')!='sha256:'+receipt['sha256']:
                raise ValueError('live published asset size/server checksum differs')
            verified.append(name)
        except Exception as error:issues.append(name+': '+str(error))
    return {'expected':len(expected),'verified':len(verified),'verified_asset_names':verified},issues


def stage_metadata(parent,destination,*,limit=METADATA_LIMIT):
    """Copy only bounded public run metadata; never package raw scientific data."""
    parent,destination=Path(parent),Path(destination);destination.mkdir(parents=True,exist_ok=True)
    total=0;inventory={};skipped=[]
    fixed={'state.json','plan.json','plan-sha256.json','submission.json','full-protocol.json','smoke-protocol.json',
        'smoke-qualification.json','native-collection.json','fusion-collection.json','smoke-native-collection.json','smoke-fusion-collection.json'}
    paths=[parent/name for name in sorted(fixed)]
    paths += sorted((parent/'publications').glob('*.json'))+sorted((parent/'inputs').glob('*.json'))+sorted((parent/'logs').glob('*.log'))
    paths += sorted(parent.glob('native-*.json'))+sorted(parent.glob('fusion-*.json'))
    seen=set()
    for path in paths:
        if path in seen:continue
        seen.add(path)
        if not path.exists():continue
        relative=path.relative_to(parent)
        if (not path.resolve().is_relative_to(parent.resolve()) or path.is_symlink() or not path.is_file()
                or path.stat().st_size>8*1024**2 or total+path.stat().st_size>limit):
            skipped.append(str(relative));continue
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        inventory[str(relative)]={'bytes':target.stat().st_size,'sha256':digest(target)};total+=target.stat().st_size
    return {'files':inventory,'bytes':total,'skipped':skipped}


def main(plan_path,expected):
    plan,root,helper=validate_bootstrap(plan_path,expected)
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('allocated observer job required')
    if (root/'observer-result.json').exists():raise ValueError('observer executes once; no implicit publication retry')
    parent=Path(plan['parent_dir']);issues=[];facts={};accounting={}
    try:
        accounting=scheduler(plan['parent_job_id'])
        if accounting['state'] not in TERMINAL:raise ValueError('parent job has not terminated')
        if accounting['state']!='COMPLETED' or accounting['exit_code']!='0:0':issues.append('parent scheduler did not complete successfully')
    except Exception as error:issues.append('scheduler accounting: '+str(error))
    try:
        state=read(parent/'state.json')
        if (state.get('status')!='PASS' or str(state.get('job_id'))!=str(plan['parent_job_id'])
                or state.get('plan_sha256')!=plan['parent_plan_sha256']
                or state.get('native_completed')!=48 or state.get('fusion_completed')!=24):issues.append('parent terminal state is incomplete')
    except Exception as error:issues.append('parent state: '+str(error));state=None
    facts,collection_issues=check_collections(parent,plan);issues.extend(collection_issues)
    publications={}
    try:
        release=json.loads(helper.gh('api',f'repos/{helper.REPOSITORY}/releases/tags/{helper.TAG}'))
        publications,publication_issues=check_publications(parent,plan,release['assets']);issues.extend(publication_issues)
    except Exception as error:issues.append('raw publication verification: '+str(error))
    with tempfile.TemporaryDirectory(prefix='bhl-sensor-terminal-',dir='/tmp') as temporary:
        temporary=Path(temporary);bundle=temporary/'summary';bundle.mkdir()
        inventory=stage_metadata(parent,bundle/'parent-metadata')
        if inventory['skipped']:issues.append('bounded metadata inventory omitted files: '+', '.join(inventory['skipped']))
        summary={'schema':'bhl-sensor-followup-terminal-summary-v1',
            'status':'COMPLETE' if not issues else 'INCOMPLETE','parent_job_id':str(plan['parent_job_id']),
            'observer_job_id':os.environ['SLURM_JOB_ID'],'observer_plan_sha256':expected,
            'parent_plan_sha256':plan['parent_plan_sha256'],'observed_utc':datetime.now(timezone.utc).isoformat(),
            'scheduler':accounting,'parent_state':state,'collections':facts,'raw_publications':publications,'issues':issues,
            'scientific_scope':'Development simulated sensor replay and pixel-depth fusion. No hardware, closed-loop, calibrated-confidence or overall improvement claim.',
            'negative_result_rule':'Native negative gates and unavailable conditional ATE remain explicit; a complete comparison is not an improvement verdict.',
            'verification_scope':'Frozen source/input lineage, strict collector identities, terminal state, exact raw receipt set and live server hashes; observer does not rerun scientific inference or collectors.'}
        write(bundle/'terminal-summary.json',summary);write(bundle/'metadata-manifest.json',inventory)
        shutil.copyfile(plan_path,bundle/'observer-plan.json')
        asset_name=f"sensor-followup-{plan['parent_job_id']}-terminal-observer-{os.environ['SLURM_JOB_ID']}.tar.gz"
        receipt=helper.publish(bundle,asset_name,root/'publications')
        result={'schema':'bhl-sensor-followup-observer-result-v1','publication_status':'PUBLISHED_AND_FRESH_DOWNLOAD_VERIFIED',
            'scientific_status':summary['status'],'summary':summary,'publication':receipt,'metadata_bytes':inventory['bytes']}
        write(root/'observer-result.json',result)
        shutil.copyfile(bundle/'terminal-summary.json',root/'terminal-summary.json')
        shutil.copyfile(bundle/'metadata-manifest.json',root/'metadata-manifest.json')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args();main(args.plan,args.plan_sha256)
