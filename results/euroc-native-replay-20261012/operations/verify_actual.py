import pathlib,json,hashlib,datetime,math,collections
import numpy as np
root=pathlib.Path(__file__).resolve().parent
stage='full';run=root/stage/'native';result=json.loads((run/'result.json').read_text());selection=json.loads((root/stage/'selection.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert result['status']=='COMPLETED' and result['stage']=='full' and result['scored'] is True
assert result['paired_frames_per_arm']==selection['paired_frames']==1200
assert result['selection_sha256']==sha(root/stage/'selection.json')
assert result['smoke_receipt_sha256']==sha(root/'smoke/native/result.json')
smoke=json.loads((root/'smoke/native/result.json').read_text())
for key in ['source_identity','protocol_sha256','selection_sha256','runtime_sha256']:assert result[key]==smoke[key]
stamps=selection['selected_original_timestamps_ns'];origin=selection['clock_origin_ns']
assert stamps==sorted(set(stamps)) and stamps[0]==origin and stamps[-1]-origin<60_000_000_000
states={-1:'SYSTEM_NOT_READY',0:'NO_IMAGES_YET',1:'NOT_INITIALIZED',2:'OK',3:'RECENTLY_LOST',4:'LOST',5:'OK_KLT'}
checks=[]
for a in result['arms']:
 mode=a['mode'];f=run/mode/'frames.jsonl';assert sha(f)==a['frames_sha256'];assert sha(run/mode/'settings.yaml')==a['settings_sha256']
 rows=[json.loads(x) for x in f.read_text().splitlines()]
 assert len(rows)==1200==a['frames'];assert [r['original_timestamp_ns'] for r in rows]==stamps
 tracked=[];accepted=[];imu_last=None;imu_count=0
 for row in rows:
  native=row['native'];t=(row['original_timestamp_ns']-origin)*1e-9
  assert abs(native['timestamp_s']-t)<1e-12
  assert type(native['tracked']) is bool
  tracked.append(native['tracked']);accepted.append(native['tracked'] and (mode=='stereo' or native['inertial']['map_imu_initialized']))
  if native['tracked']:assert native['T_W_C'] is not None and np.isfinite(native['T_W_C']).all()
  for v in [native['compute_seconds'],*row['timing'].values()]:assert math.isfinite(v) and v>=0
  timing=row['timing'];assert math.isclose(timing['pipeline_wall_s'],timing['rectification_image_io_s']+timing['native_client_wall_s'],rel_tol=1e-12,abs_tol=1e-12)
  if mode=='stereo_inertial':
   imu=native['inertial'];n=imu['imu_samples'];imu_count+=n
   assert type(n) is int and n>=0
   if n:
    assert imu['first_imu_timestamp_s']<=imu['last_imu_timestamp_s']<=t+1e-12
    if imu_last is not None:assert imu['first_imu_timestamp_s']>imu_last
    imu_last=imu['last_imu_timestamp_s']
 assert sum(tracked)==a['tracked_frames'];assert sum(accepted)==a['accepted_initialized_frames']
 assert sum(x and not y for x,y in zip(tracked,tracked[1:]))==a['tracking_loss_transitions_after_OK']
 counts=collections.Counter(states[r['native']['tracking_state']] for r in rows)
 assert {k:v for k,v in a['tracking_states'].items() if v}==dict(counts)
 assert next((r['native']['timestamp_s'] for r,ok in zip(rows,accepted) if ok),None)==a['initialization_first_sensor_time_s']
 for name,values in [('native_compute',[r['native']['compute_seconds'] for r in rows]),('pipeline_wall',[r['timing']['pipeline_wall_s'] for r in rows]),('native_client_wall',[r['timing']['native_client_wall_s'] for r in rows]),('rectification_image_io_wall',[r['timing']['rectification_image_io_s'] for r in rows])]:
  expected={'count':len(values),'mean_ms':np.mean(values)*1000,'p50_ms':np.quantile(values,.5)*1000,'p95_ms':np.quantile(values,.95)*1000,'max_ms':max(values)*1000}
  for k,v in expected.items():assert math.isclose(a[name][k],v,rel_tol=1e-10)
 if mode=='stereo_inertial':assert imu_count==a['original_imu_rows_acknowledged']==selection['imu_original_rows']
 epoch_count=0;boundaries=[];last_map=None;last_init=False;last_state=None;gap=True;ever=False
 for index,row in enumerate(rows):
  n=row['native'];m=n['map_id'];init=n.get('inertial',{}).get('map_imu_initialized',False);reasons=[]
  if last_map is not None and m is None:reasons.append('native_map_disappeared')
  if last_map is not None and m is not None and m!=last_map:reasons.append('native_map_id_changed')
  if last_init and not init:reasons.append('native_imu_initialization_lost')
  if ever and not gap and n['tracking_state']==1 and last_state!=1:reasons.append('native_returned_to_not_initialized')
  if reasons:boundaries.append({'frame_index':index,'timestamp_s':n['timestamp_s'],'previous_map_id':last_map,'map_id':m,'reasons':reasons});gap=True
  if m is not None:
   if gap:epoch_count+=1
   gap=False;ever=True
  elif ever:gap=True
  last_map,last_init,last_state=m,init,n['tracking_state']
 assert epoch_count==a['coordinate_epochs'] and boundaries==a['observed_epoch_boundaries']
 checks.append({'coordinate_epochs':epoch_count,'epoch_boundaries_recomputed':True,'mode':mode,'rows':len(rows),'tracked_frames':sum(tracked),'initialized_tracked_frames':sum(accepted),'native_states':dict(counts),'latency_recomputed':True,'exact_original_timestamps':True,'IMU_causality_checked':mode=='stereo_inertial','source_trace_sha256':sha(f)})
for stage in ['smoke','full']:
 for name,entry in json.loads((root/stage/'manifest.json').read_text()).items():
  p=root/stage/name;assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
r={'schema':'bhl-euroc-independent-trace-check-v1','status':'PASS','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(pathlib.Path(__file__)),'result_sha256':sha(run/'result.json'),'exact_frozen_smoke_full_identity':True,'compact_manifests_verified':True,'arms':checks,'ATE':'UNMEASURED; full pose accuracy was not independently established','hardware_validation':False}
(root/'independent-verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
