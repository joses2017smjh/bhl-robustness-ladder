"""Submit one qualified diagnostic mode with a unique immutable source snapshot."""
import argparse,datetime,hashlib,json,os,shutil,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('gait','gait_extra','gait_endurance','c4','approach','approach_recovery','rewards','fullroute','fullroute_recovery','fall_replay','train','train_exploration'))
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--submit',action='store_true')
    p.add_argument('--dependency');p.add_argument('--node');a=p.parse_args();out=a.campaign.resolve()
    if not out.is_relative_to(ROOT):p.error('campaign outside repo')
    prior=[json.loads(l) for l in (out/'submissions.jsonl').read_text().splitlines()] if (out/'submissions.jsonl').exists() else []
    if any(r['mode']==a.mode for r in prior):p.error('mode already submitted; preserve results and choose a new campaign for a rerun')
    if not a.submit:print('Planned',a.mode,'dependency',a.dependency);return
    qualification=json.loads((out/'qualification.json').read_text())
    if not qualification.get('passed') or a.mode not in qualification['qualified_modes']:p.error('mode not qualified')
    for name,expected in qualification['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:p.error('source changed after qualification: '+name)
    snapshot=out/('source-'+a.mode);snapshot.mkdir()
    files=list((ROOT/'src/bhl_robust/mission').glob('*.py'))+list((ROOT/'scripts').glob('mission7*.py'))
    files += [ROOT/name for name in ('src/bhl_robust/__init__.py','src/bhl_robust/sensor_io.py',
        'src/bhl_robust/eval/__init__.py','src/bhl_robust/eval/multi_robot.py','src/bhl_robust/eval/mjcf_assets.py',
        'src/bhl_robust/eval/livery.py','src/bhl_robust/eval/team_sensors.py','slurm/mission7_debug.sbatch')]
    hashes={}
    for source in files:
        rel=source.relative_to(ROOT);dest=snapshot/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest);hashes[str(rel)]=hashlib.sha256(dest.read_bytes()).hexdigest()
    (snapshot/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    cmd=['sbatch','--parsable','--job-name=m7-debug-'+a.mode,'--account=eecs','--partition=share',
        '--cpus-per-task=2','--mem=12G','--time=02:00:00','--chdir='+str(ROOT),
        '--output='+str(out/'%x-%A_%a.out'),'--error='+str(out/'%x-%A_%a.out')]
    array='0-2%3' if a.mode=='gait' else None
    if a.mode in ('train','train_exploration'):
        matrix_name='training-matrix-exploration.json' if a.mode=='train_exploration' else 'training-matrix.json'
        count=len(json.loads((out/matrix_name).read_text())['rows'])
        array=f'0-{count-1}%{min(count,4)}'
    if array:cmd+=['--array='+array]
    if a.node:cmd+=['--nodelist='+a.node]
    if a.dependency:cmd+=['--dependency='+a.dependency,'--kill-on-invalid-dep=yes']
    cmd += [str(snapshot/'slurm/mission7_debug.sbatch'),str(ROOT),str(snapshot),str(out),a.mode]
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_') and k not in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','TMPDIR')}
    job=subprocess.check_output(cmd,env=env,text=True).strip().split(';')[0]
    assert job.isdigit()
    row=dict(mode=a.mode,job_id=job,array=array,dependency=a.dependency,requested_node=a.node,command=cmd,source_sha256=hashes,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='SUBMITTED',
        resources=dict(cpus=2,memory_gb=12,gpus=0,walltime_hours=2))
    for name in ('gait-summary.json','training-matrix.json','training-matrix-exploration.json','feasibility-gate.json','preserved-c4/sha256.json'):
        if (out/name).exists():row.setdefault('input_sha256',{})[name]=hashlib.sha256((out/name).read_bytes()).hexdigest()
    with (out/'submissions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    with (ROOT/'SLURM_JOBS.md').open('a') as f:
        f.write(f"\nMission7 Approach diagnosis (2026-09-21): **SUBMITTED** `{job}`"+(f" array `{array}`" if array else '')+f" — {a.mode}; dependency {a.dependency or 'none'}; 2 CPUs /12 GB /0 GPUs /2 h per task. Receipts/source hashes: `{out.relative_to(ROOT)}/submissions.jsonl`.\n")
    print(json.dumps(row,indent=2))

if __name__=='__main__':main()
