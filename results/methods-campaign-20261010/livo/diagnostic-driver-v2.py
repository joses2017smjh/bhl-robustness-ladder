import argparse,hashlib,importlib.util,json,os,sys,tarfile
from pathlib import Path
SOURCE=Path('/tmp/bhl-livo-replay-source-v2-20261010')
sys.path.insert(0,str(SOURCE/'src'))
def module(name,relative):
 spec=importlib.util.spec_from_file_location(name,SOURCE/relative);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
BENCH=module('livo_diagnostic_bench','scripts/bench/native_livo_methods_campaign.py')
REPLAY=module('livo_diagnostic_replay','scripts/native/livo_replay.py')
SCORE=BENCH.SCORE
WORK=Path('/tmp/bhl-livo-diagnostic-work-20261010')
OUT=Path('/tmp/bhl-livo-diagnostic-results-20261010')
def checksum(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def run(phase):
 intake=json.loads((SOURCE/'intake.json').read_text())
 for relative,expected in intake['files'].items():
  if checksum(SOURCE/relative)!=expected:raise ValueError('Frozen replay source changed: '+relative)
 WORK.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
 capture=Path('/tmp/bhl-livo-diagnostic-input-20261010.tar.gz')
 if checksum(capture)!=intake['capture_sha256']:raise ValueError('Original capture archive changed')
 if not (WORK/'capture').exists():SCORE.LAUNCH.safe_extract(capture,WORK/'capture')
 baseline=Path('/nfs/stak/users/sanchej7/humanoid-native-20261009/artifacts/lio-runtime-inference.tar.gz')
 if checksum(baseline)!=intake['lio_runtime_sha256']:raise ValueError('Original baseline runtime changed')
 if not (WORK/'lio').exists():SCORE.LAUNCH.safe_extract(baseline,WORK/'lio')
 (WORK/'lio/runtime/fastlio_headless').chmod(0o700)
 manifest_path=WORK/'capture/replay/manifest.json';manifest=json.loads(manifest_path.read_text())
 if phase=='smoke':
  chosen=manifest['sequences'][0].copy();chosen['frames']=chosen['frames'][:35]
  selected=dict(manifest,sequences=[chosen],diagnostic_subset={'original_manifest_sha256':checksum(manifest_path),'first_original_frames':35,'scope':'SMOKE_ONLY'})
  subset=manifest_path.with_name('livo-smoke-manifest.json');subset.write_text(json.dumps(selected,indent=2)+'\n')
  results=[]
  for visual in (True,False):
   result=REPLAY.replay(subset,chosen['id'],'/tmp/bhl-livo-runtime-20261010',OUT/('smoke-visual' if visual else 'smoke-no-visual'),visual_enabled=visual)
   if result['frames']!=35 or result['tracked_frames']<1:raise ValueError('Native smoke has no tracked measurement outputs')
   if visual and result['frames_with_visual_measurements']<1:raise ValueError('Native smoke did not exercise actual visual measurements')
   results.append(result)
  record={'status':'PASS','scientific_status':'SMOKE_ONLY','phase':phase,'runs':results,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_step_id':os.environ.get('SLURM_STEP_ID')}
 else:
  smoke=json.loads((OUT/'smoke.json').read_text())
  if smoke['status']!='PASS':raise ValueError('Fresh same-source native smoke required')
  result=BENCH.run(manifest_path,WORK/'lio/runtime','/tmp/bhl-livo-runtime-20261010',SOURCE/'diagnostic-protocol.json',OUT/'diagnostic')
  record={'status':result['status'],'scientific_status':'REUSED_DEVELOPMENT_CAPTURE_DIAGNOSTIC_NOT_FRESH_CONFIRMATION','phase':phase,'cells':result['method_cells'],'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_step_id':os.environ.get('SLURM_STEP_ID')}
 (OUT/(phase+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','diagnostic']);a=p.parse_args();run(a.phase)
