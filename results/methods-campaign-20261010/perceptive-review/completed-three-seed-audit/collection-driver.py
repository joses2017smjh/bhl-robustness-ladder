from pathlib import Path
import sys,json,shutil,hashlib,subprocess,tarfile
repo=Path('/tmp/bhl-methods-20261010');sys.path.insert(0,str(repo/'scripts/bench'));sys.path.insert(0,str(repo/'src'))
import methods_finalize as FINAL
import perceptive_methods_collect as COLLECT
SAFE=FINAL.SAFE
campaign=Path('/nfs/stak/users/sanchej7/humanoid-methods-20261010/perceptive-v4')
work=Path('/tmp/perceptive-final-audit-20261010');work.mkdir(exist_ok=False)
view=work/'source';view.mkdir()
for name in ('intake.json','manifest.json','protocol.json','h34_campaign.sbatch'):
    shutil.copyfile(campaign/name,view/name)
intake=SAFE.read_json(view/'intake.json')
try:
    shutil.copyfile(campaign/'frozen-campaign.tar.gz',view/'frozen-campaign.tar.gz')
    SAFE.verify_frozen_archive(view,intake)
except FileNotFoundError:
    view=FINAL.verified_campaign_view(campaign,intake,work)
mirror=work/'raw';mirror.mkdir()
jobs=[campaign/f'perceptive-run-s{i}-v4-20261010-{jid}' for i,jid in enumerate(('21757635','21757710','21757754'))]
raw_paths=[FINAL.obtain_raw(job,intake['archive_sha256'],mirror) for job in jobs]
result=COLLECT.collect(view,jobs,[mirror])
output=repo/'results/methods-campaign-20261010/perceptive-review/completed-three-seed-audit'
output.mkdir(exist_ok=False)
(output/'collection.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
for job,raw in zip(jobs,raw_paths):
    dest=output/job.name;dest.mkdir()
    for name in ('completion.json','campaign_result.json','launch.json','publication.json'):
        shutil.copyfile(job/name,dest/name)
    with tarfile.open(raw,'r:gz') as archive:
        for name in ('teacher-development.json','teacher-training.json'):
            member=archive.getmember(name)
            with archive.extractfile(member) as stream:
                data=stream.read()
            (dest/name).write_bytes(data)
accounting=subprocess.check_output(['sacct','-X','-j','21757622,21757635,21757710,21757754','--format=JobIDRaw,JobName,State,ExitCode,Elapsed,AllocCPUS','-P','-n'],text=True)
(output/'slurm-accounting.psv').write_text(accounting)
state=Path('/nfs/stak/users/sanchej7/humanoid-methods-20261010/perceptive-dispatch-v1/state.json')
shutil.copyfile(state,output/'controller-state.json')
receipt={'schema':'bhl-perceptive-three-seed-audit-v1','source_archive_sha256':intake['archive_sha256'],'protocol_sha256':intake['protocol_sha256'],'collector_sha256':SAFE.digest(repo/'scripts/bench/perceptive_methods_collect.py'),'scope':'Three independent development seeds; original measurements and qualification retained; no tuning, reruns or hardware validation.','files':{str(p.relative_to(output)):{'sha256':SAFE.digest(p),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()}}
(output/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
print('AUDIT_OUTPUT',output,flush=True)
raise SystemExit(1 if result['status']=='INCOMPLETE' else 0)
