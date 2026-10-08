"""Freeze and submit 14 bounded diagnostic cells in three throttled CPU arrays."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from bhl_robust.mission.overnight import study_matrix

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--submit',action='store_true')
    a=p.parse_args(); out=a.campaign.resolve()
    if not out.is_relative_to(ROOT): p.error('campaign must remain inside repository')
    if (out/'submissions.jsonl').exists(): p.error('already submitted; refusing duplicate arrays')
    matrix=study_matrix(); head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    rows=[]
    for i,cfg in enumerate(matrix):
        rows.append({'index':i,**cfg,'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                     'output':str(out/cfg['name']),'dependency':cfg.get('requires'),'slurm_id':None,'status':'PLANNED',
                     'expected_artifacts':(['learning.jsonl','episodes.jsonl','success-trajectory.json','result.json'] if cfg['mode']=='train'
                                           else ['rewards.json','rewards.md','observations.json','layouts.json','layouts.md','result.json'] if cfg['mode']=='audits'
                                           else ['doors.json','transport.json','component-door_0.json','component-door_1.json','component-transport.json','result.json'])})
    print('| Index | Study | Sensor | Stage | Seed | Budget | Dependency |')
    print('|---|---|---|---|---:|---|---|')
    for r in rows:
        budget=str(r['updates']*r['horizon'])+' decisions' if r['mode']=='train' else '64 reward episodes + sensor/geometry probes' if r['mode']=='audits' else '32 full-route + 24 localized interaction episodes'
        print(f"| {r['index']} | {r['name']}: {r['study']} | {r.get('arm','mixed')} | {r.get('stage','mixed')} | {r['seed']} | {budget} | {r.get('requires','none')} |")
    out.mkdir(parents=True,exist_ok=True)
    plan={'git_commit':head,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
          'tasks':14,'max_cpu_hours':56,'gpus':0,'cpus_per_task':2,'memory_gb_per_task':12,
          'walltime_per_task':'02:00:00','rows':rows}
    (out/'matrix.json').write_text(json.dumps(plan,indent=2)+'\n')
    if not a.submit: return
    qualification=json.loads((out/'qualification.json').read_text())
    if not qualification.get('passed'): p.error('qualification must pass before submission')
    for name,expected in qualification['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            p.error('source changed since qualification: '+name)
    snapshot=out/'source'
    if snapshot.exists(): p.error('snapshot already exists; inspect partial submission rather than overwriting')
    files=list((ROOT/'src/bhl_robust/mission').glob('*.py'))+[ROOT/name for name in (
        'src/bhl_robust/__init__.py','src/bhl_robust/sensor_io.py','src/bhl_robust/eval/__init__.py',
        'src/bhl_robust/eval/multi_robot.py','src/bhl_robust/eval/mjcf_assets.py','src/bhl_robust/eval/livery.py',
        'src/bhl_robust/eval/team_sensors.py','scripts/mission7_overnight.py','scripts/mission7_diagnostic.py',
        'slurm/mission7_overnight.sbatch')]
    hashes={}
    for f in files:
        rel=f.relative_to(ROOT); target=snapshot/rel; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(f,target); hashes[str(rel)]=hashlib.sha256(target.read_bytes()).hexdigest()
    (snapshot/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith('SLURM_') and k not in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','TMPDIR')}
    with (ROOT/'SLURM_JOBS.md').open('a') as ledger:
        ledger.write('\n## Mission7 bounded overnight diagnostics — 2026-09-20\n\n'
                     'Fourteen additional CPU tasks; original pilot and diagnostics preserved. '
                     'No full campaign or held-out policy evaluations submitted.\n\n'
                     '| Array | Studies | State at submission | Dependency |\n|---|---|---|---|\n')
    training_job=None
    for label,array,indices in (('train','0-8%3',range(9)),('audits','12-13%2',range(12,14)),('sensors','9-11%3',range(9,12))):
        dependency=f'afterok:{training_job}_8' if label=='sensors' else None
        cmd=['sbatch','--parsable','--job-name=m7-night-'+label,'--account=eecs','--partition=share',
             '--cpus-per-task=2','--mem=12G','--time=02:00:00','--array='+array,'--chdir='+str(ROOT),
             '--output='+str(out/('%x-%A_%a.out')),'--error='+str(out/('%x-%A_%a.out'))]
        if dependency: cmd+=['--dependency='+dependency,'--kill-on-invalid-dep=yes']
        cmd += [str(snapshot/'slurm/mission7_overnight.sbatch'),str(ROOT),str(snapshot),str(out)]
        raw=subprocess.check_output(cmd,env=clean,text=True).strip(); job=raw.split(';')[0]
        if not job.isdigit(): raise RuntimeError('unexpected scheduler receipt '+raw)
        if label=='train': training_job=job
        receipt={'job_id':job,'array':array,'dependency':dependency,'command':cmd,'git_commit':head,
                 'source_sha256':hashes,'submitted_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
        with (out/'submissions.jsonl').open('a') as f:
            f.write(json.dumps(receipt)+'\n'); f.flush(); os.fsync(f.fileno())
        for i in indices: rows[i].update(slurm_id=f'{job}_{i}',status='SUBMITTED',slurm_dependency=dependency)
        (out/'matrix.json').write_text(json.dumps(plan,indent=2)+'\n')
        with (ROOT/'SLURM_JOBS.md').open('a') as ledger:
            ledger.write(f"| `{job}_[{array}]` | {', '.join(rows[i]['name'] for i in indices)} | **SUBMITTED** | {dependency or 'none'} |\n")
        print('SUBMITTED '+job+' '+array,flush=True)
    with (ROOT/'SLURM_JOBS.md').open('a') as ledger:
        ledger.write('\nEach task: 2 CPUs, 12 GB, zero GPUs, 2 h cap; total maximum 56 CPU-hours. '
                     'Sensor tasks also require C4 JSON learning evidence; otherwise they write '
                     '`SKIPPED_NOT_LEARNABLE` without training. C4 is reused as the matched Both arm, '
                     'and A1 doubles as the current-PPO sweep baseline. '
                     f'Receipts, config hashes, source hashes, commit and output paths: `{out.relative_to(ROOT)}/matrix.json` '
                     f'and `{out.relative_to(ROOT)}/submissions.jsonl`. '
                     'Scheduler completion certifies diagnostic execution only, never learned success.\n')


if __name__=='__main__': main()
