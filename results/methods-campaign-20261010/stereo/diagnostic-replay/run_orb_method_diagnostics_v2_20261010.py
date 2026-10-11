import hashlib, importlib.util, json, os
from pathlib import Path
import shutil, subprocess, sys, time

ARCHIVE=Path('/nfs/stak/users/sanchej7/humanoid-methods-20261010/stereo-v2/frozen-campaign.tar.gz')
ARCHIVE_SHA='9d8ed3780f31cb1a78c1b9ac6006aa57b663011120381e499b08f85167c09cf7'
CAPTURE=Path('/nfs/stak/users/sanchej7/humanoid-native-20261009/capture-v2/native-capture-capture-20261009-v2-21740516/outputs.tar.gz')
CAPTURE_SHA='ec15004ac7be4b58d33f78d6a4b5924d1d4a72b00c9332473d5617818ba08ca1'
ORB_SHA='8ef2f4a0813ff97f457a97b9971e41555241b83f66c68b10c34a4a3986a28c3f'
DEST=Path('/tmp/orb-method-diagnostics-20261010-v2')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')

started=time.monotonic()
assert sha(ARCHIVE)==ARCHIVE_SHA
assert sha(CAPTURE)==CAPTURE_SHA
DEST.mkdir(exist_ok=False)
import tarfile
with tarfile.open(ARCHIVE) as f:
 for m in f.getmembers():
  n=Path(m.name)
  if n.is_absolute() or '..' in n.parts or not (m.isfile() or m.isdir()):raise ValueError('unsafe archive')
 f.extractall(DEST/'frozen')
frozen=DEST/'frozen'
manifest=json.loads((frozen/'manifest.json').read_text())
for name,row in manifest.items():
 if sha(frozen/name)!=row['sha256']:raise ValueError('frozen source/input hash differs: '+name)
inputs=DEST/'inputs';inputs.mkdir()
shutil.copy2(CAPTURE,inputs/'capture.tar.gz')
os.link(frozen/'inputs/runtime/orb-runtime.tar.gz',inputs/'orb-runtime.tar.gz')
protocol={'methods':['orb'],'capture_archive':'capture.tar.gz','orb_runtime_archive':'orb-runtime.tar.gz',
 'orb_runtime_archive_sha256':ORB_SHA,'input_files':{'capture.tar.gz':{'sha256':CAPTURE_SHA},'orb-runtime.tar.gz':{'sha256':ORB_SHA}},
 'replay_gate':{'initialization_exclusion_s':5.,'maximum_ate_rmse_m':.1,'maximum_native_compute_p95_s':.1,'maximum_rotation_p95_deg':5.,'minimum_tracking_fraction':.9},
 'scope':'Read-only actual native feature diagnostics on previously consumed original450pairs; historical replay gates unchanged; no new generalization claim'}
write(DEST/'protocol.json',protocol)
receipts=[]
for phase in ('smoke','replay'):
 output=DEST/phase;work=DEST/(phase+'-work')
 env=dict(os.environ,H34_PROTOCOL=str(DEST/'protocol.json'),H34_OUTPUT_DIR=str(output),H34_WORK_DIR=str(work),OMP_NUM_THREADS='1',TMPDIR='/tmp',TMP='/tmp',TEMP='/tmp')
 cmd=[sys.executable,str(frozen/'source/scripts/bench/native_slam_campaign.py'),'--protocol',str(DEST/'protocol.json'),'--phase',phase]
 print('RUN',json.dumps(cmd),flush=True)
 result=subprocess.run(cmd,env=env,check=True)
 scientific=json.loads((output/'campaign_result.json').read_text())
 if phase=='smoke' and scientific['status']!='PASS':raise ValueError('fresh diagnostic smoke failed')
 receipts.append({'phase':phase,'exit_code':result.returncode,'result_sha256':sha(output/'campaign_result.json'),'status':scientific['status']})
paths=sorted((DEST/'replay').glob('*/orb/native_frames.jsonl'))
if len(paths)!=3:raise ValueError('allthree originalscenes required')
subprocess.run([sys.executable,str(frozen/'source/scripts/bench/stereo_diagnostics.py'),'--frames',*[str(p) for p in paths],'--output',str(DEST/'diagnostics.json')],check=True)
write(DEST/'execution.json',{'status':'PASS','scientific_status':'READ_ONLY_DIAGNOSTICS_ON_CONSUMED_DATA','source_archive_sha256':ARCHIVE_SHA,'verified_frozen_members':len(manifest),'capture_archive_sha256':CAPTURE_SHA,'orb_runtime_archive_sha256':ORB_SHA,'phases':receipts,'slurm_job_id':os.getenv('SLURM_JOB_ID'),'slurm_step_id':os.getenv('SLURM_STEP_ID'),'host':os.uname().nodename,'elapsed_wall_s':time.monotonic()-started,'diagnostics_sha256':sha(DEST/'diagnostics.json')})
print('DIAGNOSTICS_COMPLETE',DEST,flush=True)
