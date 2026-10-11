import hashlib, importlib.util, json, os
from pathlib import Path
import time
ROOT=Path('/tmp/bhl-stereo-inertial-v1-20261010')
manifest=json.loads((ROOT/'source-manifest.json').read_text())
for name,row in manifest.items():
 path=ROOT/'source'/name
 if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('frozen source changed: '+name)
plan=json.loads((ROOT/'plan.json').read_text())
smoke=json.loads((ROOT/'smoke/campaign_result.json').read_text())
if smoke['status']!='PASS' or any(arm['frames']!=35 for arm in smoke['arms'].values()):raise ValueError('both-mode native smoke required')
module_path=ROOT/'source/scripts/native/orb_inertial_replay.py'
spec=importlib.util.spec_from_file_location('inertial_measured_replay',module_path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
started=time.monotonic();results=[]
for sequence in plan['sequences']:
 for mode in plan['arms']:
  rows,result=module.replay('/tmp/orb-method-diagnostics-20261010-v2/replay-work/capture/replay/manifest.json',sequence,ROOT/'native/runtime',ROOT/'replay'/sequence/mode,mode=mode)
  results.append({k:result.get(k) for k in ['sequence','sensor_mode','status','frames','tracked_frames','observed_map_changes','imu_initialized_frames','tracked_and_imu_initialized_frames','first_imu_initialized_timestamp_s','native_compute_p95_ms','manifest_sha256','binary_sha256','frames_sha256','settings_sha256']})
  print(json.dumps(results[-1]),flush=True)
execution={'schema':'bhl-stereo-inertial-diagnostic-execution-v1','status':'PASS','scientific_status':'PREVIOUSLY_CONSUMED_SIMULATION_DATA; NOT_HELD_OUT','source_manifest_sha256':hashlib.sha256((ROOT/'source-manifest.json').read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest(),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'slurm_job_id':os.getenv('SLURM_JOB_ID'),'slurm_step_id':os.getenv('SLURM_STEP_ID'),'host':os.uname().nodename,'results':results,'elapsed_seconds':time.monotonic()-started,'ground_truth_inputs':[],'closed_loop_episodes':0}
(ROOT/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
print(json.dumps(execution),flush=True)
