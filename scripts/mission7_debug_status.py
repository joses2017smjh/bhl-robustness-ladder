"""Refresh Mission7 diagnostic states from receipts, accounting and result files."""
import argparse,datetime,json,re,subprocess
from pathlib import Path


def refresh(campaign,accounting=False,docs=False):
    receipts=[json.loads(line) for line in (campaign/'submissions.jsonl').read_text().splitlines()]
    superseded={r['supersedes_job'] for r in receipts if 'supersedes_job' in r}
    if accounting:
        text=subprocess.check_output(['sacct','-n','-X','-j',','.join(r['job_id'] for r in receipts),
            '--format=JobID,State,ExitCode,Elapsed,AllocCPUS,CPUTimeRAW,NodeList','-P'],text=True)
        (campaign/'slurm-accounting.psv').write_text(text)
    states={}
    for line in (campaign/'slurm-accounting.psv').read_text().splitlines():
        v=line.split('|')
        states[v[0]]=dict(state=v[1],exit_code=v[2],elapsed=v[3],allocated_cpus=int(v[4]),allocated_cpu_seconds=int(v[5]),node=v[6] if len(v)>6 else '')
    rows=[]
    for receipt in receipts:
        mode=receipt['mode'];job=receipt['job_id']
        ids=[f'{job}_{i}' for i in range(3 if mode=='gait' else 2 if mode=='train_exploration' else 4)] if receipt['array'] else [job]
        for i,identifier in enumerate(ids):
            folder=f'P{i+5}' if mode=='train_exploration' else f'P{i+1}' if mode=='train' else f'gait-{i}' if mode=='gait' else mode
            folder=receipt.get('output',folder).rstrip('/')
            path=campaign/folder/'result.json';result=json.loads(path.read_text()) if path.exists() else {}
            rows.append(dict(job_id=identifier,mode=mode,dependency=receipt['dependency'],output=folder+'/',
                result_complete=result.get('complete',False),scientific_status='INVALID_REPLAY_SUPERSEDED' if identifier in superseded else result.get('status','IN_PROGRESS'),
                **states.get(identifier,dict(state='UNKNOWN'))))
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    status=dict(checked_utc=now,jobs=rows,submissions=len(receipts),tasks=len(rows),
        allocated_cpu_hours=sum(r.get('allocated_cpu_seconds',0) for r in rows)/3600,sensor_release=False)
    status['complete']=all(r['result_complete'] or r['job_id'] in superseded for r in rows)
    (campaign/'campaign-status.json').write_text(json.dumps(status,indent=2)+'\n')
    lines=['## Submitted diagnostic jobs','',f'Snapshot: {now}. All tasks: 2 CPUs, 12 GB, zero GPUs, two-hour cap.',
        f"{len(receipts)} submissions /{len(rows)} tasks; {status['allocated_cpu_hours']:.2f} allocated CPU-hours so far.",
        '', '| Job | Study | Scheduler state | Scheduler dependency | Output under campaign |',
        '|---|---|---|---|---|']
    for r in rows:lines.append(f"| {r['job_id']} | {r['mode']} | {r['state']} | {r['dependency'] or 'none'} | `{r['output']}` |")
    lines+=['','PPO has completed logical prerequisites in `feasibility-gate.json` and a runtime guard; no expired scheduler dependency was attached. No sensor jobs were submitted.']
    (campaign/'job-status.md').write_text('\n'.join(lines)+'\n')
    outcomes=['## Current controlled-study outcomes','',f'Read from result files at {now}. Incomplete cells remain pending.','',
        '| Cell | Completed update | Initial success /16 | Latest success /16 | Median final distance | Training successes | Stable within seed |',
        '|---|---:|---:|---:|---:|---:|---|']
    for name in ('P1','P2','P3','P4','P5','P6'):
        path=campaign/name/'result.json';result=json.loads(path.read_text()) if path.exists() else {}
        path=campaign/name/'learning.jsonl';logs=[]
        if path.exists():
            for line in path.read_text().splitlines():
                try:logs.append(json.loads(line))
                except json.JSONDecodeError:break
        vals=[]
        for path in sorted((campaign/name).glob('validation-*.json')):
            data=json.loads(path.read_text())
            if data['complete']:vals.append(data['summary'])
        row=[name,str(logs[-1]['update']) if logs else 'pending',str(vals[0]['successes']) if vals else 'pending',
            str(vals[-1]['successes']) if vals else 'pending',f"{vals[-1]['final_distance']['median']:.3f} m" if vals else 'pending',
            str(logs[-1]['training_successes']) if logs else 'pending',str(result['stable_within_seed']) if result.get('complete') else 'pending']
        outcomes.append('| '+' | '.join(row)+' |')
    outcomes+=['','| Route controller | Stage | Episodes available | Success | Falls | Complete |','|---|---|---:|---:|---:|---|']
    for folder,variants in (('fullroute',('legacy','measured')),('fullroute_recovery',('recovery',))):
        for variant in variants:
            for stage in ('doors','transport'):
                path=campaign/folder/f'{variant}-{stage}.json'
                if not path.exists():continue
                data=json.loads(path.read_text());es=data['episodes']
                outcomes.append(f"| {variant} | {stage} | {len(es)} | {sum(e['success'] for e in es)} | {sum(e['fall'] for e in es)} | {data['complete']} |")
    (campaign/'study-outcomes.md').write_text('\n'.join(outcomes)+'\n')
    if docs:
        doc=Path(__file__).resolve().parents[1]/'docs/MISSION7_APPROACH_DEBUG.md'
        text=doc.read_text();start='<!-- mission7-debug-status:start -->';end='<!-- mission7-debug-status:end -->'
        assert text.count(start)==text.count(end)==1
        text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda _:start+'\n'+'\n'.join(lines+['']+outcomes)+'\n'+end,text,flags=re.S)
        doc.write_text(text)
    print(json.dumps(dict(checked_utc=now,tasks=len(rows),states={r['job_id']:r['state'] for r in rows})))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--accounting',action='store_true');parser.add_argument('--docs',action='store_true')
    args=parser.parse_args();refresh(args.campaign,args.accounting,args.docs)
