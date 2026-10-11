#!/usr/bin/env python3
"""Bounded, source-pinned geometric occlusion and matched fusion controls."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(Path(__file__).parent))
import sensor_capture as BASE
from bhl_robust.research.fusion_occlusion import ARMS, SETTINGS, predict, evaluate_predictions, visibility_mask
from bhl_robust.research.stereo_benchmark import (FastFoundationStereoMethod, FFS_MODEL_SHA256,
    FFS_MODEL_BYTES, sha256_file, load_rgb)
from bhl_robust.stereo_depth import RectifiedCalibration

DESIGN = ROOT/'protocols/fusion_occlusion_20261011.json'
DURABLE = Path('/nfs/hpc/share/sanchej7/humanoid-research-20261011')
SOURCE_FILES = ('scripts/bench/fusion_occlusion_campaign.py', 'scripts/bench/sensor_capture.py',
    'src/bhl_robust/research/fusion_occlusion.py', 'src/bhl_robust/research/stereo_consistency.py',
    'src/bhl_robust/research/stereo_benchmark.py', 'src/bhl_robust/research/pose_metrics.py',
    'src/bhl_robust/stereo_depth.py', 'protocols/fusion_occlusion_20261011.json',
    'scripts/bench/fusion_occlusion_serial.py', 'scripts/bench/sensor_campaign.py')


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')


def cells(design):
    return [dict(id=f'{phase}-{family}-s{seed}-{condition}', phase=phase, family=family,
                 seed=seed, group=f'{family}-s{seed}', condition=condition, frames=design['frames'][phase])
            for phase in ('smoke', 'dev', 'test') for family in design['families']
            for seed in design['geometry_seeds'][phase] for condition in design['conditions']]


def freeze(output, model):
    output, model = Path(output), Path(model)
    if output.exists(): raise FileExistsError('protocol must be new')
    if model.stat().st_size != FFS_MODEL_BYTES or sha256_file(model) != FFS_MODEL_SHA256:
        raise ValueError('official frozen model mismatch')
    design = json.loads(DESIGN.read_text())
    plan = {'schema': 'bhl-fusion-occlusion-protocol-v1', 'design': design, 'cells': cells(design),
        'arms': list(ARMS), 'settings': SETTINGS, 'source_sha256': {f: sha256_file(ROOT/f) for f in SOURCE_FILES},
        'model': {'path': str(model.resolve()), 'bytes': FFS_MODEL_BYTES, 'sha256': FFS_MODEL_SHA256},
        'inference_inputs': ['left_rgb', 'right_rgb', 'raw_lidar_xyz', 'fixed_calibration'],
        'evaluator_inputs': ['left_depth', 'right_depth', 'lidar_view_depth', 'left_geom_segmentation'],
        'positive_result_gate': 'No predeclared gain guarantee; report full accuracy, coverage and false-free tradeoffs.'}
    write(output, plan); return plan


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    if (plan.get('schema') != 'bhl-fusion-occlusion-protocol-v1' or plan['design'] != json.loads(DESIGN.read_text())
        or plan['cells'] != cells(plan['design']) or plan['arms'] != list(ARMS) or plan['settings'] != SETTINGS):
        raise ValueError('predeclared geometry/arms/settings changed')
    if plan['source_sha256'] != {f: sha256_file(ROOT/f) for f in SOURCE_FILES}: raise ValueError('source pin changed')
    return plan


def scene_xml(family, seed, condition, texture, design):
    root = ET.fromstring(BASE.scene_xml('textured_boxes', texture))
    world = root.find('worldbody')
    for i, geom in enumerate(list(world.findall('geom'))):
        if i >= 4: world.remove(geom)
        else: geom.set('name', 'ground' if i == 0 else f'wall_{i}')
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-.08, .08, 2)
    layouts = {'panel_background': [(2.55, .02, .65, .15, .70, .65)],
               'staggered_posts': [(2.25, -.24, .65, .07, .07, .65), (2.55, .02, .65, .07, .07, .65),
                                   (2.8, .30, .65, .10, .10, .65)]}
    for index, (x,y,z,sx,sy,sz) in enumerate(layouts[family]):
        ET.SubElement(world, 'geom', name=f'target_{index}', type='box',
            pos=f'{x+jitter[0]} {y+jitter[1]} {z}', size=f'{sx} {sy} {sz}', material='pattern')
    width = design['occluder_half_width_m'][condition]
    if width:
        ET.SubElement(world, 'geom', name='foreground_occluder', type='box',
            pos=f'{1.15+jitter[0]} {jitter[1]} .65', size=f'.025 {width} .60', material='pattern')
    rig = world.find("body[@name='sensor_rig']")
    ET.SubElement(rig, 'camera', name='lidar_evaluator', pos='0 0 0', xyaxes='0 -1 0 0 0 1', fovy='60')
    return ET.tostring(root, encoding='unicode')


def static_scan(model, data, position, stamp):
    points = []
    for azimuth in np.linspace(math.radians(-80), math.radians(80), SETTINGS['columns']):
        for elevation in np.linspace(math.radians(-40), math.radians(20), SETTINGS['rings']):
            direction = np.array([math.cos(elevation)*math.cos(azimuth), math.cos(elevation)*math.sin(azimuth), math.sin(elevation)])
            distance = BASE.ray(model, data, position, direction)
            if .1 <= distance <= 12.: points.append(BASE.R_B_C.T@direction*distance)
    return np.asarray(points, np.float32).reshape(-1,3)


def check_storage(output, design):
    output = Path(output).resolve()
    allocated_scratch = (os.getenv('SLURM_JOB_ID') or os.getenv('H34_PROTOCOL')) and output.is_relative_to(Path('/tmp'))
    if not output.is_relative_to(DURABLE) and not allocated_scratch: raise ValueError('outputs require declared share or allocated /tmp')
    if shutil.disk_usage(DURABLE.parent).free < design['minimum_share_free_bytes'] + design['maximum_cell_bytes']:
        raise RuntimeError('preserve >=8GiB share free-space reserve')
    if output.exists() and sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > design['maximum_cell_bytes']:
        raise RuntimeError('per-cell output bound exceeded')


def verify_smoke(protocol, receipt):
    if receipt is None: raise ValueError('actual same-source GPU smoke collection required before scored cells')
    value = json.loads(Path(receipt).read_text())
    if (value.get('status') != 'PASS' or value.get('phase') != 'smoke' or value.get('protocol_sha256') != sha256_file(protocol)
        or not value.get('readiness_gate') or value.get('verified_cells') != 6): raise ValueError('invalid or negative smoke gate')


def run(protocol, cell, output, smoke_receipt=None):
    plan = read_plan(protocol); design = plan['design']
    selected = [c for c in plan['cells'] if c['id'] == cell]
    if len(selected) != 1: raise ValueError('undeclared cell')
    cell = selected[0]
    if cell['phase'] != 'smoke': verify_smoke(protocol, smoke_receipt)
    allocated = os.getenv('SLURM_JOB_ID') or (os.getenv('H34_PROTOCOL') and os.getenv('H34_OUTPUT_DIR'))
    if not allocated or os.getenv('MUJOCO_GL') != 'egl': raise RuntimeError('actual allocated EGL GPU smoke required')
    import cv2
    import torch
    import mujoco
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    cv2.setNumThreads(1)
    output = Path(output); check_storage(output, design); output.mkdir(parents=True, exist_ok=False)
    texture = output/'texture.png'
    pixels = np.random.default_rng(cell['seed']).integers(30,235,(256,256,3),dtype=np.uint8)
    if not cv2.imwrite(str(texture), pixels): raise RuntimeError('texture write failed')
    xml = scene_xml(cell['family'],cell['seed'],cell['condition'],texture,design)
    (output/'scene.xml').write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    calibration = BASE.calibration(design['image_width'],design['image_height'])
    rectified = RectifiedCalibration.from_dict(calibration)
    start = time.perf_counter_ns()
    foundation = FastFoundationStereoMethod(rectified,os.getenv('FUSION_MODEL_PATH',plan['model']['path']))
    initialization_ms = (time.perf_counter_ns()-start)/1e6
    rows=[]; renderer=mujoco.Renderer(model,height=design['image_height'],width=design['image_width'])
    try:
        for index in range(cell['frames']):
            check_storage(output, design)
            fraction=index/max(cell['frames']-1,1)
            position=np.array([np.interp(fraction,[0,1],design['viewpoint_x_m']),
                               np.interp(fraction,[0,1],design['viewpoint_y_m']),BASE.RIG_HEIGHT])
            data.mocap_pos[0]=position;data.mocap_quat[0]=[1,0,0,0];data.time=float(index);mujoco.mj_forward(model,data)
            start=time.perf_counter_ns(); rendered={}
            for name in ('left','right'):
                renderer.update_scene(data,camera=name);rendered[name]=renderer.render().copy()
                if not cv2.imwrite(str(output/f'{index:04d}-{name}.png'),cv2.cvtColor(rendered[name],cv2.COLOR_RGB2BGR)):raise RuntimeError('RGB write failed')
            points=static_scan(model,data,position,float(index))
            np.savez_compressed(output/f'{index:04d}-lidar.npz',points_xyz_m=points,timestamp_s=np.array(float(index)))
            capture_ms=(time.perf_counter_ns()-start)/1e6
            start=time.perf_counter_ns();left=load_rgb(output/f'{index:04d}-left.png');right=load_rgb(output/f'{index:04d}-right.png');decode_ms=(time.perf_counter_ns()-start)/1e6
            start=time.perf_counter_ns()
            with np.load(output/f'{index:04d}-lidar.npz',allow_pickle=False) as raw: points=raw['points_xyz_m'].copy()
            lidar_load_ms=(time.perf_counter_ns()-start)/1e6
            if index==0: predict(left,right,points,rectified,calibration,foundation)  # one same-input warm-up, excluded
            predictions,latency,diagnostics=predict(left,right,points,rectified,calibration,foundation)
            for arm in ARMS:
                latency[arm]+=(0. if arm=='lidar_sparse' else decode_ms)+(lidar_load_ms if arm=='lidar_sparse' or '_' in arm else 0.)
            # Independent truth renders occur only after every prediction arm.
            depthviews={};renderer.enable_depth_rendering()
            for camera in ('left','right','lidar_evaluator'):
                renderer.update_scene(data,camera=camera);depthviews[camera]=renderer.render().copy().astype(np.float32)
            renderer.disable_depth_rendering();renderer.enable_segmentation_rendering();renderer.update_scene(data,camera='left');labels=renderer.render().copy();renderer.disable_segmentation_rendering()
            obstacle=(labels[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&(labels[...,0]>0)
            right_occ,right_inside=visibility_mask(depthviews['left'],depthviews['right'],calibration['fx_px'],calibration['baseline_m'])
            lidar_occ,lidar_inside=visibility_mask(depthviews['left'],depthviews['lidar_evaluator'],calibration['fx_px'],calibration['baseline_m']/2)
            metrics=evaluate_predictions(predictions,depthviews['left'],obstacle,right_occ,lidar_occ,right_inside&lidar_inside)
            arrays={name+'_depth_m':depth for name,(depth,_) in predictions.items()};arrays.update({name+'_valid':valid for name,(_,valid) in predictions.items()});arrays.update(diagnostics)
            arrays.update(truth_depth_m=depthviews['left'],truth_right_m=depthviews['right'],truth_lidar_view_m=depthviews['lidar_evaluator'],obstacle=obstacle,right_occluded=right_occ,lidar_occluded=lidar_occ,visibility_support=right_inside&lidar_inside)
            artifact=output/f'{index:04d}-predictions-evaluation.npz';np.savez_compressed(artifact,**arrays)
            rows.append({'frame_index':index,'viewpoint_m':position.tolist(),'metrics':metrics,'latency_ms':latency,'capture_serialize_ms':capture_ms,'artifact':artifact.name,'artifact_sha256':sha256_file(artifact)})
            print(json.dumps({'cell':cell['id'],'frame':index}),flush=True)
    finally:renderer.close()
    result={'schema':'bhl-fusion-occlusion-cell-v1','status':'PASS','cell':cell,'protocol_sha256':sha256_file(protocol),'calibration':calibration,'arms':list(ARMS),'rows':rows,
        'foundation':foundation.provenance(),'foundation_session_initialization_ms':initialization_ms,
        'runtime':{'host':platform.node(),'mujoco':mujoco.__version__,'opencv':cv2.__version__,'cuda_device':torch.cuda.get_device_name(0),'job_id':os.getenv('SLURM_JOB_ID'),'renderer':'egl'},
        'latency_scope':'measured shared pipeline components + required PNG/NPZ decode, summed perarm; one excluded warmup; no capture, evaluator or serialization; model edgepad320x256 included',
        'ground_truth_inputs':[],'scientific_status':'SMOKE_ONLY' if cell['phase']=='smoke' else 'PREDECLARED_SIMULATION_EVALUATION'}
    result['file_sha256']={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}
    write(output/'campaign_result.json',result);check_storage(output,design);return result


def pooled_scores(rows):
    total=lambda key:sum(row[key] for row in rows)
    predicted,eligible,near=total('predicted_pixels'),total('eligible_pixels'),total('near_obstacle_pixels')
    return {'predicted_pixels':predicted,'eligible_pixels':eligible,'near_obstacle_pixels':near,
            'mae_m':total('absolute_error_sum_m')/predicted if predicted else None,
            'coverage':predicted/eligible if eligible else None,
            'false_free_obstacle_pixels':total('false_free_obstacle_pixels'),
            'missed_obstacle_pixels':total('missed_obstacle_pixels'),
            'unknown_obstacle_pixels':total('unknown_obstacle_pixels'),
            'obstacle_pixel_recall':total('detected_obstacle_pixels')/near if near else None}


def collect(protocol, phase, directories, output):
    plan=read_plan(protocol);expected={c['id']:c for c in plan['cells'] if c['phase']==phase};seen={};problems=[]
    for directory in map(Path,directories):
        try:
            record=json.loads((directory/'campaign_result.json').read_text());identity=record['cell']['id']
            if identity not in expected or identity in seen or record['cell']!=expected[identity]:raise ValueError('unknown/duplicate/mismatched geometry cell')
            if record.get('status')!='PASS' or record['protocol_sha256']!=sha256_file(protocol) or record['arms']!=list(ARMS) or record['ground_truth_inputs']!=[]:raise ValueError('wrong source/protocol/arm scope')
            if len(record['rows'])!=expected[identity]['frames']:raise ValueError('missing frame')
            if record['foundation']['model_sha256']!=FFS_MODEL_SHA256 or record['foundation']['provider_execution_evidence']['status']!='VERIFIED_CUDA_KERNEL_EXECUTION':raise ValueError('CUDA/checkpoint not proven')
            if record['calibration']!=BASE.calibration(plan['design']['image_width'],plan['design']['image_height']):raise ValueError('fixed calibration changed')
            expected_files={'scene.xml','texture.png'}
            for index in range(expected[identity]['frames']):
                expected_files.update({f'{index:04d}-left.png',f'{index:04d}-right.png',f'{index:04d}-lidar.npz',f'{index:04d}-predictions-evaluation.npz'})
            actual={p.name for p in directory.iterdir() if p.is_file()}-{'campaign_result.json'}
            if actual!=expected_files:raise ValueError('declared raw/evaluator artifact set incomplete')
            if actual!=set(record['file_sha256']):raise ValueError('artifact inventory differs')
            for name,digest in record['file_sha256'].items():
                if '/' in name or directory.joinpath(name).is_symlink() or sha256_file(directory/name)!=digest:raise ValueError('artifact changed')
            for index,row in enumerate(record['rows']):
                if row['frame_index']!=index or row['artifact']!=f'{index:04d}-predictions-evaluation.npz' or sha256_file(directory/row['artifact'])!=row['artifact_sha256']:raise ValueError('frame identity/hash differs')
                with np.load(directory/row['artifact'],allow_pickle=False) as values:
                    predictions={arm:(values[arm+'_depth_m'],values[arm+'_valid']) for arm in ARMS}
                    shape=(plan['design']['image_height'],plan['design']['image_width'])
                    if any(depth.shape!=shape or valid.shape!=shape for depth,valid in predictions.values()):raise ValueError('prediction dimensions changed')
                    right,ri=visibility_mask(values['truth_depth_m'],values['truth_right_m'],record['calibration']['fx_px'],record['calibration']['baseline_m'])
                    lidar,li=visibility_mask(values['truth_depth_m'],values['truth_lidar_view_m'],record['calibration']['fx_px'],record['calibration']['baseline_m']/2)
                    if not all(np.array_equal(a,b) for a,b in ((right,values['right_occluded']),(lidar,values['lidar_occluded']),(ri&li,values['visibility_support']))):raise ValueError('visibility labels differ')
                    independently=evaluate_predictions(predictions,values['truth_depth_m'],values['obstacle'],right,lidar,ri&li)
                    if independently!=row['metrics']:raise ValueError('independent metric recomputation differs')
                if set(row['latency_ms'])!=set(ARMS) or any(not np.isfinite(v) or v<=0 for v in row['latency_ms'].values()):raise ValueError('invalid measured timings')
            seen[identity]=record
        except Exception as error:problems.append({'directory':str(directory),'error':f'{type(error).__name__}: {error}'})
    missing=sorted(set(expected)-set(seen));readiness=not missing and not problems
    if phase=='smoke':
        readiness &= all(sum(r['metrics']['visibility_pixels']['either_occluded'] for r in rec['rows'])>=plan['design']['smoke_gate']['minimum_occluded_pixels_per_nonclear_cell'] for rec in seen.values() if rec['cell']['condition']!='clear')
    groups=[];pairs=[];strata=[]
    for identity,record in sorted(seen.items()):
        for arm in ARMS:
            rows=[r['metrics']['arms'][arm]['native'] for r in record['rows']]
            sumkey=lambda k:sum(r[k] for r in rows)
            predicted,eligible,near=sumkey('predicted_pixels'),sumkey('eligible_pixels'),sumkey('near_obstacle_pixels')
            groups.append({'cell':identity,'group':record['cell']['group'],'condition':record['cell']['condition'],'arm':arm,'frames':len(rows),
                'predicted_pixels':predicted,'eligible_pixels':eligible,'near_obstacle_pixels':near,
                'mae_m':sumkey('absolute_error_sum_m')/predicted if predicted else None,'coverage':predicted/eligible if eligible else None,
                'false_free_obstacle_pixels':sumkey('false_free_obstacle_pixels'),'missed_obstacle_pixels':sumkey('missed_obstacle_pixels'),
                'unknown_obstacle_pixels':sumkey('unknown_obstacle_pixels'),'obstacle_pixel_recall':sumkey('detected_obstacle_pixels')/near if near else None,
                'p95_pipeline_ms':float(np.quantile([r['latency_ms'][arm] for r in record['rows']],.95)),
                'all_arm_common':pooled_scores([r['metrics']['arms'][arm]['all_arm_common'] for r in record['rows']])})
            for stratum in record['rows'][0]['metrics']['arms'][arm]['visibility_strata']:
                strata.append({'cell':identity,'group':record['cell']['group'],'condition':record['cell']['condition'],
                    'arm':arm,'stratum':stratum,**pooled_scores([r['metrics']['arms'][arm]['visibility_strata'][stratum] for r in record['rows']])})
        for comparison,sides in record['rows'][0]['metrics']['pairwise_common'].items():
            pairs.append({'cell':identity,'group':record['cell']['group'],'condition':record['cell']['condition'],
                'comparison':comparison,'arms':{arm:pooled_scores([r['metrics']['pairwise_common'][comparison][arm] for r in record['rows']]) for arm in sides}})
    total_bytes=sum(p.stat().st_size for d in map(Path,directories) for p in d.rglob('*') if p.is_file())
    if total_bytes>plan['design']['maximum_campaign_bytes']:problems.append({'error':'aggregate payload exceeds2GiB'});readiness=False
    result={'schema':'bhl-fusion-occlusion-collection-v1','status':'PASS' if not missing and not problems else 'INCOMPLETE','phase':phase,'protocol_sha256':sha256_file(protocol),
        'expected_cells':len(expected),'verified_cells':len(seen),'missing':missing,'problems':problems,'readiness_gate':bool(readiness),
        'group_metrics':groups,'pairwise_common_metrics':pairs,'visibility_group_metrics':strata,'payload_bytes':total_bytes,'geometry_groups':len({r['cell']['group'] for r in seen.values()}),'independent_seed_clusters':len({r['cell']['seed'] for r in seen.values()}),
        'inference_scope':'static simulation viewpoints, geometry-group comparisons; no frame-level statistical significance or hardware claim',
        'pairwise_common_support':'independent pairwise masks; pergeometry pooled errors and exact denominators, also retained perframe',
        'data_quality_note':'gated rejection may lower false-free while reducing coverage/recall; always report all three together'}
    write(output,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('freeze');p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--protocol',type=Path,required=True);p.add_argument('--cell',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke-receipt',type=Path)
    p=sub.add_parser('collect');p.add_argument('--protocol',type=Path,required=True);p.add_argument('--phase',choices=['smoke','dev','test'],required=True);p.add_argument('--directories',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=vars(parser.parse_args());result=globals()[args.pop('command')](**args);print(json.dumps({k:v for k,v in result.items() if k not in ('rows','cells','source_sha256','group_metrics','file_sha256','pairwise_common_metrics','visibility_group_metrics')},indent=2));raise SystemExit(0 if result.get('status','PASS')=='PASS' else 1)
