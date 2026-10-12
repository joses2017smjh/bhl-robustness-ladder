#!/usr/bin/env python3
"""Independent read-only recomputation, with new output artifacts only.
No imports from the scientific implementation or collector.
"""
from pathlib import Path
import collections, hashlib, itertools, json, math, tarfile
from datetime import datetime, timezone
import numpy as np

BASE = Path('/nfs/hpc/share/sanchej7/humanoid-research-20261011/processing-schedule-v1-20261012')
OUT = Path(__file__).resolve().parent
COLLECTION = BASE/'processing-schedule-collection-20261012-21801949/campaign_result.json'
KEYS = ('case', 'seed', 'condition', 'arm', 'pipeline')

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-8), (a,b)

def arrayclose(a,b):
    assert np.allclose(a,b,rtol=1e-7,atol=1e-7), float(np.max(np.abs(np.asarray(a)-np.asarray(b))))

intake = json.loads((BASE/'intake.json').read_text())
with tarfile.open(BASE/'frozen-campaign.tar.gz') as t:
    protocol_bytes = t.extractfile('inputs/scientific-protocol.json').read()
assert hashlib.sha256(protocol_bytes).hexdigest() == intake['scientific_protocol_sha256']
protocol = json.loads(protocol_bytes)
expected = set(itertools.product(('straight','occluders'), (750000,750001),
    ('nominal','camera_10hz','causal_40ms_exposure'), ('stereo','lidar_inertial'),
    ('legacy_cumulative','bounded_latest')))
assert expected == {tuple(c[k] for k in KEYS) for c in protocol['development']['cells']}
assert protocol['development']['seconds'] == 40 and len(expected)==48
smoke = json.loads((BASE/'qualified-smoke.json').read_text())
assert smoke['status']=='PASS' and smoke['episodes']==12 and all(smoke['arm_development_ready'].values())
rows=[]; seen=set(); lineage=[]; verified_files=0; source_results=[]
for result_path in sorted((BASE/'collector-inputs').rglob('campaign_result.json')):
    result = json.loads(result_path.read_text())
    assert result['status']=='PASS' and result['phase']=='development'
    assert not result['problems'] and not result['unrun_cells']
    assert result['source_sha256']==smoke['source_sha256'] and result['input_sha256']==smoke['input_sha256']
    assert result['arm_development_ready']==smoke['arm_development_ready']
    assert result['episodes']==len(result['episode_files'])==len(result['completed_cells'])==2
    source_results.append({'path':str(result_path),'sha256':sha(result_path)})
    for cell, relative in zip(result['completed_cells'],result['episode_files']):
        key=tuple(cell[k] for k in KEYS)
        assert key in expected and key not in seen
        seen.add(key)
        ep=result_path.parent/relative
        assert ep.resolve().is_relative_to(result_path.parent.resolve())
        assert sha(ep)==result['episode_sha256'][relative]; verified_files+=1
        r=json.loads(ep.read_text())
        assert tuple(r[k] for k in ('case','seed','condition','method','pipeline'))==key
        assert not r['ground_truth_inputs_to_estimator_or_planner']
        for name, field in [('inference_sensors.npz','inference_sensors_sha256'),('evaluator_truth.npz','evaluator_truth_sha256'),('native/stream_receipt.json','stream_receipt_sha256'),('sensors/processing_schedule.json','processing_schedule_sha256')]:
            assert sha(ep.parent/name)==r[field], (str(ep),name)
            verified_files+=1
        tr=r['trace']; assert len(tr)==1000
        decision=np.array([x['decision_timestamp_s'] for x in tr]); timestamp=np.array([x['timestamp_s'] for x in tr])
        arrayclose(decision,np.arange(1000)*.04);arrayclose(timestamp,decision+.04)
        close(r['elapsed_simulation_s'],40);assert r['full_horizon']
        with np.load(ep.parent/'evaluator_truth.npz',allow_pickle=False) as truth:
            arrayclose(truth['timestamp_s'],timestamp)
            body=truth['T_W_I']@np.linalg.inv(np.asarray(r['T_body_imu']))
        arrayclose(body[:,:2,3],[x['evaluator_xy'] for x in tr])
        arrayclose(body[:,2,3],[x['evaluator_height_m'] for x in tr])
        dist=np.linalg.norm(body[:,:2,3]-np.asarray(r['route_external'][-1]),axis=1)
        tilt=np.arccos(np.clip(body[:,2,2],-1,1))
        arrayclose(dist,[x['evaluator_goal_distance_m'] for x in tr]);arrayclose(tilt,[x['evaluator_tilt_rad'] for x in tr])
        reached=bool(np.any(dist<=.3)); contact=any(x['evaluator_contact'] for x in tr)
        fell=bool(np.any(tilt>=.78)|np.any(body[:,2,3]<r['evaluator_initial_height_m']-.25))
        finite=bool(np.isfinite(body).all() and np.isfinite(np.asarray([x['command'] for x in tr])).all())
        assert finite and not r['nonfinite']
        success=reached and not contact and not fell
        assert all(bool(r[k])==val for k,val in [('goal_reached_evaluator',reached),('collision',contact),('fell',fell),('success',success)])
        nav=tr[250:];assert len(nav)==750
        assert all(x['decision_timestamp_s']>=10.-1e-9 for x in nav)
        reasons=collections.Counter(x['reason'] for x in nav)
        stale=reasons['native_stale_stop']
        assert r['navigation_decisions']==750 and r['stale_navigation_decisions']==stale
        close(r['stale_navigation_fraction'],stale/750)
        nz=[x for x in nav if max(map(abs,x['command']))>1e-10]
        fresh=[x for x in nav if x['native_tracked'] and x['native_timestamp_s'] is not None and -1e-8 <= x['decision_timestamp_s']-x['native_timestamp_s'] <=.35]
        ev=json.loads((ep.parent/'sensors/processing_schedule.json').read_text())
        assert ev==r['acquisition_schedule'] and ev['pipeline']==r['pipeline']
        q=ev['queue']; assert q['produced']==q['selected']+q['superseded']+q['pending']
        assert 0<=q['pending']<=1 and q['maximum_waiting_events']==1
        assert q['selected']==r['native_frames']==len(ev['requests'])==len(r['schedule_timing'])
        for name,s in q['streams'].items():
            assert s['produced_unique']==s['selected_unique']+s['superseded_before_selection']+s['pending_unselected']
            assert 0<=s['pending_unselected']<=1
            assert s['dispatches']==s['selected_unique']+s['reused_dispatches']==q['selected']
            assert ev[name+'_captures_total']==s['produced_unique']+len(ev[name+'_never_joined'])
        rec=ev['recorder']; assert rec['joined'] and rec.get('pending',0)==0 and not rec.get('error')
        if r['pipeline']=='bounded_latest':
            assert rec['maximum_pending']<=rec['capacity']==8
            assert rec['tasks']==ev['camera_captures_total']
        for request,timing in zip(ev['requests'],r['schedule_timing']):
            cap=request['camera_capture_s' if r['method']=='stereo' else 'lidar_capture_s']
            close(timing['capture_s'],cap);assert cap<=request['request_s']+1e-8
            close(timing['preprocessing_s'],request['preprocessing_service_s'])
            service=request['preprocessing_service_s']+timing['native_client_s']
            close(timing['service_s'],service)
            ticks=max(1,math.ceil(service/.04));assert ticks==timing['delivery_ticks']
            close(timing['delivery_s'],request['request_s']+ticks*.04)
            close(timing['capture_age_at_delivery_s'],timing['delivery_s']-cap)
            qt=request['queue'];assert qt['enqueue_s']<=request['request_s']
            close(qt['queue_residence_s'],request['request_s']-qt['enqueue_s'])
        deliveries=r['response_deliveries']
        assert len(deliveries)+r['responses_still_pending_at_horizon']==r['native_frames']
        for delivered,timing in zip(deliveries,r['schedule_timing']):
            close(delivered['arrival_s'],timing['delivery_s']);close(delivered['wall_seconds'],timing['service_s'])
        close(r['full_pipeline_p95_ms'],float(np.percentile([d['wall_seconds'] for d in deliveries],95))*1000)
        close(r['client_wall_p95_ms'],float(np.percentile([d['native_client_s'] for d in r['schedule_timing']],95))*1000)
        rows.append(dict(case=r['case'],seed=r['seed'],condition=r['condition'],method=r['method'],pipeline=r['pipeline'],
            clean_goal=success,fall=fell,contact=contact,full_horizon=True,policy_decisions=1000,navigation_decisions=750,
            stale_decisions=stale,stale_fraction=stale/750,nonzero_command_decisions=len(nz),fresh_accepted_pose_decisions=len(fresh),
            reasons=dict(reasons),nonzero_command_by_reason=dict(collections.Counter(x['reason'] for x in nz)),
            logical_coordinate_invalidation_events=r['native_coordinate_invalidations'],observed_map_id_changes=r['native_map_changes'],
            final_goal_distance_m=float(dist[-1]),minimum_goal_distance_m=float(dist.min()),
            client_p95_ms=r['client_wall_p95_ms'],delivered_modeled_service_p95_ms=r['full_pipeline_p95_ms'],
            pending_responses=r['responses_still_pending_at_horizon'],evidence_writer_backpressure_wall_s=rec.get('backpressure_wall_s',0),
            episode_path=str(ep),episode_sha256=sha(ep)))
assert seen==expected and len(rows)==48 and len(source_results)==24

def aggregate(rs):
    out={k:sum(r[k] for r in rs) for k in ['clean_goal','fall','contact','policy_decisions','navigation_decisions','stale_decisions','nonzero_command_decisions','fresh_accepted_pose_decisions','logical_coordinate_invalidation_events','observed_map_id_changes','pending_responses','evidence_writer_backpressure_wall_s']}
    out['episodes']=len(rs);out['full_horizon_episodes']=sum(r['full_horizon'] for r in rs)
    out['stale_fraction']=out['stale_decisions']/out['navigation_decisions']
    out['episodes_with_coordinate_invalidation']=sum(r['logical_coordinate_invalidation_events']>0 for r in rs)
    out['episodes_with_observed_map_id_change']=sum(r['observed_map_id_changes']>0 for r in rs)
    out['episodes_with_invalidation_but_no_observed_map_id_change']=sum(r['logical_coordinate_invalidation_events']>0 and r['observed_map_id_changes']==0 for r in rs)
    for k in ['reasons','nonzero_command_by_reason']:
        total=collections.Counter()
        for r in rs:total.update(r[k])
        out[k]=dict(total)
    for k in ['final_goal_distance_m','minimum_goal_distance_m','client_p95_ms','delivered_modeled_service_p95_ms']:
        v=[r[k] for r in rs];out[k]={'min':min(v),'median':float(np.median(v)),'max':max(v)}
    return out

groups={f'{p}/{m}':aggregate([r for r in rows if r['pipeline']==p and r['method']==m]) for p in ['legacy_cumulative','bounded_latest'] for m in ['stereo','lidar_inertial']}
by_seed={f'{seed}/{p}/{m}':aggregate([r for r in rows if r['seed']==seed and r['pipeline']==p and r['method']==m]) for seed in [750000,750001] for p in ['legacy_cumulative','bounded_latest'] for m in ['stereo','lidar_inertial']}
collection=json.loads(COLLECTION.read_text())
assert collection['status']=='PASS' and collection['declared_cells']==collection['measured_cells']==48
assert not collection['missing_cells'] and collection['unrun_after_nominal_gate']==0
assert collection['source_results']==source_results
for key,g in groups.items():
    ref=collection['by_pipeline_arm'][key]
    assert ref['episodes']==g['episodes'] and ref['clean_goals']==g['clean_goal'] and ref['falls']==g['fall'] and ref['contacts']==g['contact']
    close(ref['mean_stale_fraction'],g['stale_fraction'])
pairs=[]
for case,seed,condition,method in itertools.product(['straight','occluders'],[750000,750001],['nominal','camera_10hz','causal_40ms_exposure'],['stereo','lidar_inertial']):
    rs=[r for r in rows if (r['case'],r['seed'],r['condition'],r['method'])==(case,seed,condition,method)]
    legacy=next(r for r in rs if r['pipeline']=='legacy_cumulative'); candidate=next(r for r in rs if r['pipeline']=='bounded_latest')
    pairs.append({'case':case,'seed':seed,'condition':condition,'method':method,'stale_fraction_difference':candidate['stale_fraction']-legacy['stale_fraction'],'clean_goal_difference':int(candidate['clean_goal'])-int(legacy['clean_goal'])})
assert collection['paired']['complete_pairs']==collection['paired']['stale_comparable_pairs']==len(pairs)==24
assert not collection['paired']['incomplete_pairs']
close(collection['paired']['mean_paired_stale_fraction_difference'],float(np.mean([p['stale_fraction_difference'] for p in pairs])))
assert collection['paired']['clean_goal_count_difference']==sum(p['clean_goal_difference'] for p in pairs)
result=dict(status='PASS',audited_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),campaign=str(BASE),
    collection_sha256=sha(COLLECTION),scientific_protocol_sha256=intake['scientific_protocol_sha256'],source_commit=intake['source_commit'],
    verified_episode_and_lineage_files=verified_files,verified_policy_decisions=48000,verified_navigation_decisions=36000,
    checks=['Exact independent Cartesian cohort and frozen protocol match','Same-source/input fresh smoke and development lineage','Episode plus four retained receipts SHA256 per episode','Evaluator truth transform independently reconstructs body position, height, tilt and goal distance','Full 40-second trace with correct pre-step and post-step timestamps','Trace-derived goal, contact, fall and stale outcomes','Queue and stream conservation, bounded joined writer','Capture/service timing, delayed response conservation, and actual delivered service p95','Strict final collection aggregates and paired outcomes agree'],
    definitions={'navigation':'750 pre-step decisions from 10.00 through 39.96 seconds per episode; initialization excluded','fresh_accepted_pose':'native_tracked wrapper acceptance is true, timestamp exists, and pre-step pose age is in [-1e-8,0.35] seconds; includes poses held across decisions','nonzero_command':'maximum absolute command component > 1e-10, including motion holds and turning; not proof of physical progress','stale_fraction':'native_stale_stop decisions divided by all navigation decisions','contact':'reported simulator contact flag independently aggregated from the retained trace; geometry was not rerun','coordinate_invalidation':'logical wrapper continuity invalidation; separate from observed native map-ID change','latency_hold':'may hold a nonzero previous command; not necessarily a stopped robot','modeled_service_p95':'within-episode p95 of delivered native response service; excludes any response pending at horizon','client_p95':'within-episode p95 of actual client.track elapsed time including client I/O'},
    limitations=['Two seed clusters; descriptive development results, no independent confirmation or population interval','Stereo always legacy-first, lidar always candidate-first; execution order confounded with pipeline within each method','Candidate service separation is modeled; simulator generation and evidence archival measured separately, not concurrent robot or end-to-end hardware latency','Simulated ideal 16-ring 3D lidar differs from physical planar RPLIDAR C1','Ground-truth body trajectory used only by independent outcome audit; not estimator or controller'],
    overall=aggregate(rows),by_pipeline_method=groups,by_seed_pipeline_method=by_seed,paired=pairs,episodes=rows)
(OUT/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'PASS','verified_files':verified_files,'groups':groups,'per_seed':{k:{f:v[f] for f in ['episodes','clean_goal','stale_decisions','stale_fraction','nonzero_command_decisions','fresh_accepted_pose_decisions','logical_coordinate_invalidation_events','observed_map_id_changes']} for k,v in by_seed.items()},'audit_json':str(OUT/'audit.json')},indent=2))
