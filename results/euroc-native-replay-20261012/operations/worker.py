import os,sys,json,pathlib,hashlib,subprocess,tempfile,shutil,tarfile,datetime,socket
root=pathlib.Path(__file__).resolve().parent
stage=sys.argv[1];assert stage in ['smoke','full']
plan=json.loads((root/'launch-plan.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
for name,expected in plan['frozen_files'].items():
 if sha(root/name)!=expected:raise ValueError('Frozen launch input mismatch: '+name)
job=os.environ['SLURM_JOB_ID'];work=pathlib.Path(tempfile.mkdtemp(prefix=f'bhl-euroc-{stage}-{job}-',dir='/tmp'))
for name in ['source','private-home','private-tmp','private-cache']:(work/name).mkdir()
with tarfile.open(root/'source.tar') as t:t.extractall(work/'source',filter='data')
protocol=work/'source/results/euroc-external-validation-20261011/protocol.json'
py='/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python'
base=['/bin/apptainer','exec','--containall','--cleanenv','--home',str(work/'private-home'),'--bind','/nfs/hpc/share/sanchej7/Humanoid_Lite:/nfs/hpc/share/sanchej7/Humanoid_Lite:ro','--bind',f'{work}:{work}','--bind',f'{work}/source:{work}/source:ro','--pwd',str(work/'source'),'--env','PYTHONNOUSERSITE=1','--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--env',f'TMPDIR={work}/private-tmp','--env',f'XDG_CACHE_HOME={work}/private-cache']
sif='/nfs/hpc/share/sanchej7/Humanoid_Lite/container/bhl.sif'
script=work/'source/scripts/bench/euroc_native_validation.py'
if stage=='smoke':
 command=base+['--bind',f'{root}/dataset:{root}/dataset:ro',sif,py,str(script),'prepare','--dataset',str(root/'dataset'),'--protocol',str(protocol),'--output',str(work/'selection.json')]
 subprocess.run(command,check=True,timeout=1200)
else:shutil.copyfile(root/'smoke/selection.json',work/'selection.json')
selection=json.loads((work/'selection.json').read_text());dataset=work/'selected';dataset.mkdir()
for name,expected in selection['files'].items():
 source=root/'dataset'/name;target=dataset/name
 if sha(source)!=expected['sha256'] or source.stat().st_size!=expected['bytes']:raise ValueError('Selected input mismatch')
 target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
if stage=='full':shutil.copytree(root/'smoke/native',work/'same-source-smoke')
command=base+['--bind',f'{work}/selected:{work}/selected:ro','--bind',f'{root}/orb-stereo-inertial-runtime-v1-20261010.tar.gz:{root}/orb-stereo-inertial-runtime-v1-20261010.tar.gz:ro',sif,py,str(script),'run','--stage',stage,'--dataset',str(dataset),'--protocol',str(protocol),'--selection',str(work/'selection.json'),'--runtime-archive',str(root/'orb-stereo-inertial-runtime-v1-20261010.tar.gz'),'--output',str(work/'native')]
if stage=='full':command+=['--smoke-receipt',str(work/'same-source-smoke/result.json')]
out=root/stage;out.mkdir(exist_ok=False)
launch={'stage':stage,'scored':stage=='full','job_id':job,'host':socket.gethostname(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'local_work':str(work),'command':command,'launch_plan_sha256':sha(root/'launch-plan.json'),'worker_sha256':sha(pathlib.Path(__file__)),'source_archive_sha256':sha(root/'source.tar'),'selection_sha256':sha(work/'selection.json'),'input_manifest_verified':len(selection['files']),'estimated_dataset_no_ground_truth_bound':True,'resources':{'cpus':int(os.environ['SLURM_CPUS_PER_TASK']),'memory':'8GiB','gpu':False}}
(out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
shutil.copyfile(work/'selection.json',out/'selection.json')
try:
 subprocess.run(command,check=True,timeout=7200)
finally:
 if (work/'native').exists():
  for path in sorted((work/'native').rglob('*')):
   rel=path.relative_to(work/'native')
   if not path.is_file() or any(p.startswith('private-') for p in rel.parts) or path.name.startswith('current-cam'):continue
   target=out/'native'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
(out/'manifest.json').write_text(json.dumps({str(p.relative_to(out)): {'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()},indent=2)+'\n')
print('EUROC_STAGE_COMPLETED',stage,job,flush=True)
