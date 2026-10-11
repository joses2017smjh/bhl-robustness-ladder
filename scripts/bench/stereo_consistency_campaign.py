#!/usr/bin/env python3
"""Frozen paired local-cost / projected-LiDAR fusion experiment on fresh capture.

Uses the sensor-fault protocol's bound source, inputs and declared geometry
cohort. Original binary SGBM support remains a separate baseline.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/'src'))
import sensor_fault_campaign as FAULT
from bhl_robust.research.stereo_consistency import predict, SETTINGS
from bhl_robust.stereo_depth import RectifiedCalibration
from bhl_robust.research.stereo_benchmark import load_rgb, depth_metrics

FAULTS = ['nominal', 'lidar_extrinsic_yaw2deg_x2cm', 'lidar_keep_half', 'lidar_keep_quarter']
ARMS = ['sgbm', 'sgbm_local_cost', 'lidar_sparse', 'simple_projected_fusion', 'cost_consistency_fusion']


def run(protocol, group, fault, output, work):
    import cv2
    cv2.setNumThreads(1)
    plan = FAULT.read_plan(protocol)
    if fault not in FAULTS: raise ValueError('declared geometric/sparsity fusion condition required')
    entry = next(e for e in plan['inputs'] if e['group'] == group)
    output, work = Path(output), Path(work); output.mkdir(parents=True, exist_ok=False)
    derived = FAULT.derive_inputs(entry['manifest'], entry['sequence'], fault, work)
    FAULT.write(output/'fault_manifest.json', derived)
    original, root = FAULT.ADAPTER.read_manifest(entry['manifest'])
    frames = FAULT.ADAPTER.choose_sequence(original, entry['sequence'])['frames'][1:-1]
    calibration = derived['calibration']; rectified = RectifiedCalibration.from_dict(calibration)
    rows, latency = [], []
    for index, frame in enumerate(frames):
        start = time.perf_counter()
        left, right = load_rgb(work/frame['left']), load_rgb(work/frame['right'])
        with np.load(work/frame['lidar'], allow_pickle=False) as cloud: xyz = cloud['points_xyz_m'].copy()
        predictions, confidence = predict(left, right, xyz, rectified, calibration)
        latency.append(1000*(time.perf_counter()-start))
        # Every inference arm completes before opening independent labels.
        truth_path = root/frame['truth']
        if FAULT.sha256(truth_path) != original['file_sha256'][frame['truth']]: raise ValueError('evaluator truth hash changed')
        with np.load(truth_path, allow_pickle=False) as truth:
            gt, obstacle = truth['depth_z_m'], truth['obstacle_mask']
            metrics = {}
            common = np.logical_and.reduce([mask for _, mask in predictions.values()])
            for name, (depth, valid) in predictions.items():
                eligible = obstacle & np.isfinite(gt) & (gt >= .1) & (gt <= 3.)
                false_free = eligible & valid & (depth > gt+.15)
                metrics[name] = {'native_support': depth_metrics(depth, valid, gt, obstacle_mask=obstacle),
                    'common_support': depth_metrics(depth, valid, gt, common_mask=common, obstacle_mask=obstacle),
                    'false_free_obstacle_pixels': int(false_free.sum()),
                    'false_free_obstacle_fraction': float(false_free.sum()/eligible.sum()) if eligible.any() else None,
                    'unknown_obstacle_pixels': int((eligible & ~valid).sum())}
        # Retain actual depth/support/confidence predictions for independent audit.
        arrays = {name+'_depth_m': depth for name,(depth,_) in predictions.items()}
        arrays.update({name+'_valid': valid for name,(_,valid) in predictions.items()})
        arrays.update(confidence)
        prediction_path = output/f'predictions-{index:04d}.npz'; np.savez_compressed(prediction_path, **arrays)
        rows.append({'index': index, 'timestamp_s': frame['timestamp_s'], 'metrics': metrics,
                     'conflicting_pixels': int(confidence['conflict'].sum()),
                     'prediction_file': prediction_path.name, 'prediction_sha256': FAULT.sha256(prediction_path)})
    result = {'schema':'bhl-stereo-consistency-cell-v1', 'status':'PASS', 'phase':plan['phase'], 'group':group, 'fault':fault,
        'protocol_sha256':FAULT.sha256(protocol), 'original_manifest_sha256':entry['manifest_sha256'],
        'arms':ARMS, 'frames':len(rows), 'rows':rows, 'settings':SETTINGS,
        'latency':{'samples_ms':latency, 'p95_ms':float(np.quantile(latency,.95)),
            'scope':'actual per-frame PNG/NPZ load, two-direction SGBM, photometric costs, projection and all fusion arms; excludes capture and evaluation/output serialization'},
        'opencv_version':cv2.__version__, 'ground_truth_inputs':[],
        'limitations':['local SAD margin is not the aggregated SGM cost volume or calibrated confidence',
            'sparse nearest-pixel LiDAR projection; no depth completion or three-view semidensification',
            'original per-point times retained but projected scan is not motion deskewed in this map arm',
            'pixel recall is not object detection; no closed-loop or physical outcome'],
        'scientific_status':'SMOKE_ONLY' if plan['phase']=='smoke' else 'PAIRED_DEVELOPMENT_CELL'}
    FAULT.write(output/'campaign_result.json',result)
    return result


def collect(protocol, directories, output):
    plan = FAULT.read_plan(protocol)
    expected = {(e['group'], f) for e in plan['inputs'] for f in FAULTS}
    seen, problems = {}, []
    for directory in map(Path,directories):
        try:
            record=json.loads((directory/'campaign_result.json').read_text()); key=(record['group'],record['fault'])
            if key not in expected or key in seen: raise ValueError('unknown or duplicate paired cell')
            entry=next(e for e in plan['inputs'] if e['group']==key[0])
            if (record.get('status')!='PASS' or record.get('phase')!=plan['phase'] or record.get('arms')!=ARMS
                or record.get('protocol_sha256')!=FAULT.sha256(protocol)
                or record.get('original_manifest_sha256')!=entry['manifest_sha256']
                or record.get('settings')!=SETTINGS or record.get('ground_truth_inputs')!=[]):
                raise ValueError('wrong source/input/protocol/method scope')
            if record['frames'] != entry['source_frames']-2 or len(record['rows']) != record['frames']:
                raise ValueError('incomplete original frame coverage')
            original,root=FAULT.ADAPTER.read_manifest(entry['manifest'])
            frames=FAULT.ADAPTER.choose_sequence(original,entry['sequence'])['frames'][1:-1]
            for index,(row,frame) in enumerate(zip(record['rows'],frames)):
                if row['index']!=index or row['timestamp_s']!=frame['timestamp_s']: raise ValueError('frame identity differs')
                filename=f'predictions-{index:04d}.npz'
                if row['prediction_file']!=filename or FAULT.sha256(directory/filename)!=row['prediction_sha256']:
                    raise ValueError('raw prediction hash differs')
                truth_path=root/frame['truth']
                if FAULT.sha256(truth_path)!=original['file_sha256'][frame['truth']]: raise ValueError('evaluator hash differs')
                with np.load(directory/filename,allow_pickle=False) as pred, np.load(truth_path,allow_pickle=False) as truth:
                    common=np.logical_and.reduce([pred[name+'_valid'] for name in ARMS])
                    for name in ARMS:
                        depth,valid=pred[name+'_depth_m'],pred[name+'_valid']
                        actual=depth_metrics(depth,valid,truth['depth_z_m'],obstacle_mask=truth['obstacle_mask'])
                        if actual!=row['metrics'][name]['native_support']: raise ValueError('independent depth metric differs')
                        actual=depth_metrics(depth,valid,truth['depth_z_m'],common_mask=common,obstacle_mask=truth['obstacle_mask'])
                        if actual!=row['metrics'][name]['common_support']: raise ValueError('independent common support metric differs')
                        eligible=truth['obstacle_mask'] & np.isfinite(truth['depth_z_m']) & (truth['depth_z_m']>=.1) & (truth['depth_z_m']<=3.)
                        count=int((eligible & valid & (depth>truth['depth_z_m']+.15)).sum())
                        if count!=row['metrics'][name]['false_free_obstacle_pixels'] or int((eligible & ~valid).sum())!=row['metrics'][name]['unknown_obstacle_pixels']:
                            raise ValueError('independent hazard accounting differs')
            seen[key]=record
        except Exception as error: problems.append({'directory':str(directory),'error':f'{type(error).__name__}: {error}'})
    missing=sorted(expected-seen.keys())
    # Macro average within each geometry cell; never count frames as independent trials.
    groups=[]
    for (group,fault),record in sorted(seen.items()):
        for name in ARMS:
            metrics=[row['metrics'][name]['native_support'] for row in record['rows']]
            groups.append({'group':group,'fault':fault,'arm':name,
                **{key:float(np.mean([m[key] for m in metrics if m[key] is not None])) if any(m[key] is not None for m in metrics) else None
                   for key in ('mae_m','coverage','near_obstacle_pixel_recall')},
                'unknown_obstacle_pixels':sum(row['metrics'][name]['unknown_obstacle_pixels'] for row in record['rows']),
                'false_free_obstacle_pixels':sum(row['metrics'][name]['false_free_obstacle_pixels'] for row in record['rows'])})
    result={'schema':'bhl-stereo-consistency-collection-v1','status':'PASS' if not missing and not problems else 'INCOMPLETE',
        'phase':plan['phase'],'expected_cells':len(expected),'verified_cells':len(seen),'missing':missing,'problems':problems,
        'group_metrics':groups,'scope':'fresh development geometry; no generalization or calibrated-confidence claim'}
    FAULT.write(output,result); return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    runp=sub.add_parser('run'); runp.add_argument('--protocol',type=Path,required=True); runp.add_argument('--group',required=True)
    runp.add_argument('--fault',choices=FAULTS,required=True); runp.add_argument('--output',type=Path,required=True); runp.add_argument('--work',type=Path,required=True)
    cp=sub.add_parser('collect'); cp.add_argument('--protocol',type=Path,required=True); cp.add_argument('--directories',nargs='*',type=Path,default=[]); cp.add_argument('--output',type=Path,required=True)
    args=vars(p.parse_args()); result=globals()[args.pop('command')](**args)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2)); raise SystemExit(0 if result['status']=='PASS' else 1)
