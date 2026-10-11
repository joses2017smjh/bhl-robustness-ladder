#!/usr/bin/env python3
"""Freeze, run and strictly collect fresh paired native sensor-fault replay.

A run is one geometry group x one fault x three genuine native estimators.
All fault arms share interior camera frames; no source recording is edited.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from bhl_robust.research.sensor_faults import FAULTS, METHODS, GROUPS, GATE, perturb_calibration, perturb_cloud, perturb_imu
from bhl_robust.research.native_lio import sha256


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


NATIVE = module('sensor_fault_native_methods', 'scripts/bench/native_livo_methods_campaign.py')
ADAPTER, SCORE = NATIVE.SCORE.ADAPTER, NATIVE.SCORE
SOURCE_FILES = ['scripts/bench/sensor_fault_campaign.py', 'src/bhl_robust/research/sensor_faults.py',
    'scripts/bench/native_livo_methods_campaign.py', 'scripts/bench/native_slam_campaign.py',
    'scripts/native/lio_replay.py', 'scripts/native/livo_replay.py', 'scripts/bench/pose_research.py',
    'src/bhl_robust/research/native_lio.py', 'src/bhl_robust/research/native_livo.py',
    'src/bhl_robust/research/pose_metrics.py', 'scripts/bench/stereo_consistency_campaign.py',
    'src/bhl_robust/research/stereo_consistency.py', 'src/bhl_robust/research/stereo_benchmark.py',
    'src/bhl_robust/stereo_depth.py', 'scripts/bench/h34_campaign.py',
    'src/bhl_robust/research/native_orb.py']


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def runtime_receipt(path, binary):
    path = Path(path).resolve()
    receipt = json.loads((path / 'runtime.json').read_text())
    if receipt.get('status') != 'PASS' or sha256(path / binary) != receipt.get('binary_sha256'):
        raise ValueError('verified actual native runtime required')
    return {'path': str(path), 'receipt_sha256': sha256(path / 'runtime.json'),
            'binary': binary, 'binary_sha256': sha256(path / binary)}


def freeze(manifests, lio_runtime, livo_runtime, output, *, smoke=False):
    output = Path(output)
    if output.exists(): raise ValueError('new protocol path required')
    inputs = []
    for path in manifests:
        manifest, root = ADAPTER.read_manifest(path)
        if len(manifest['sequences']) != 1: raise ValueError('one fresh group per capture required')
        sequence = manifest['sequences'][0]; cell = manifest.get('cell', {})
        if (cell.get('condition') != 'nominal' or cell.get('geometry_group') not in GROUPS
                or sequence['id'] != cell.get('id') or sequence['split'] != 'dev'):
            raise ValueError('fresh predeclared nominal development geometry required')
        count = len(sequence['frames'])
        if (smoke and not 35 <= count <= 150) or (not smoke and count != 150):
            raise ValueError('full groups require 150 original frames; smoke requires at least 35')
        inputs.append({'group': cell['geometry_group'], 'manifest': str(Path(path).resolve()),
                       'manifest_sha256': sha256(path), 'sequence': sequence['id'], 'source_frames': count})
    groups = [x['group'] for x in inputs]
    if len(groups) != len(set(groups)) or (smoke and len(groups) != 1) or (not smoke and set(groups) != set(GROUPS)):
        raise ValueError('exactly six declared groups, or one explicit smoke group required')
    plan = {'schema': 'bhl-sensor-fault-protocol-v1', 'phase': 'smoke' if smoke else 'run',
        'inputs': inputs, 'methods': METHODS, 'faults': FAULTS, 'gate': GATE,
        'source_sha256': {name: sha256(ROOT / name) for name in SOURCE_FILES},
        'runtimes': {'lio': runtime_receipt(lio_runtime, 'fastlio_headless'),
                     'livo': runtime_receipt(livo_runtime, 'fastlivo_headless')},
        'frame_selection': 'same original interior frames [1:-1] for every arm; all original IMU samples retained',
        'causality': 'positive reported IMU offset delays availability; original acquisition time <= reported time; no future samples or truth inputs',
        'unit': 'paired development geometry group; no frame-level significance',
        'scientific_scope': 'sensor replay, no closed-loop goals, collisions, recovery or calibrated confidence claim'}
    write(output, plan)
    write(str(output) + '.sha256.json', {'protocol_sha256': sha256(output)})
    return plan


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    if (plan.get('schema') != 'bhl-sensor-fault-protocol-v1' or plan.get('methods') != METHODS
            or plan.get('faults') != FAULTS or plan.get('gate') != GATE):
        raise ValueError('unchanged predeclared fault protocol required')
    if plan['source_sha256'] != {name: sha256(ROOT / name) for name in SOURCE_FILES}:
        raise ValueError('frozen fault source changed')
    for name, value in plan['runtimes'].items():
        if value != runtime_receipt(value['path'], value['binary']):
            raise ValueError('frozen native runtime changed')
    for entry in plan['inputs']:
        if sha256(entry['manifest']) != entry['manifest_sha256']:
            raise ValueError('frozen raw capture manifest changed')
    return plan


def derive_inputs(manifest_path, sequence_id, fault, destination):
    """Construct a new inference-only dataset; never open evaluator files."""
    manifest, source = ADAPTER.read_manifest(manifest_path)
    seq = ADAPTER.choose_sequence(manifest, sequence_id)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    value = copy.deepcopy(manifest)
    value['calibration'] = perturb_calibration(manifest['calibration'], fault)
    value['file_sha256'] = {}; value.pop('payload_bytes', None)
    derived = copy.deepcopy(seq); derived['frames'] = copy.deepcopy(seq['frames'][1:-1])
    if len(derived['frames']) < 2: raise ValueError('at least four source frames required')
    # Merge original per-frame IMU packets before trimming camera boundaries.
    # This preserves initialization samples and identical edge coverage across
    # delayed-clock arms without extrapolation or invented measurements.
    samples = {}
    for frame in seq['frames']:
        data = ADAPTER._arrays(ADAPTER.inference_path(source, frame['imu']),
                              ('timestamp_s', 'gyro_rad_s', 'specific_force_m_s2'))
        for stamp, gyro, accel in zip(data['timestamp_s'], data['gyro_rad_s'], data['specific_force_m_s2']):
            sample = np.r_[gyro, accel]
            if float(stamp) in samples and not np.array_equal(samples[float(stamp)], sample):
                raise ValueError('conflicting original IMU samples')
            samples[float(stamp)] = sample
    stamps = np.array(sorted(samples)); values = np.stack([samples[float(t)] for t in stamps])
    imu = perturb_imu(stamps, values[:, :3], values[:, 3:], fault)
    rel_imu = 'inference/fault-imu.npz'; (destination / 'inference').mkdir()
    np.savez_compressed(destination / rel_imu, **imu)
    value['file_sha256'][rel_imu] = sha256(destination / rel_imu)
    original_hashes = ADAPTER._input_receipt(source, seq)
    for frame in derived['frames']:
        frame.pop('truth', None)
        for key in ('left', 'right'):
            src = ADAPTER.inference_path(source, frame[key]); dst = destination / frame[key]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try: os.link(src, dst)
                except OSError: shutil.copyfile(src, dst)
            value['file_sha256'][frame[key]] = sha256(dst)
        cloud = ADAPTER._arrays(ADAPTER.inference_path(source, frame['lidar']),
                    ('points_xyz_m', 'point_time_s', 'ring_index'), ('intensity',))
        dst = destination / frame['lidar']; dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, **perturb_cloud(cloud, fault))
        value['file_sha256'][frame['lidar']] = sha256(dst)
        frame['imu'] = rel_imu
    value['sequences'] = [derived]
    value['sensor_fault'] = {'name': fault, 'specification': FAULTS[fault],
        'original_manifest_sha256': sha256(manifest_path), 'original_inputs_sha256': original_hashes,
        'ground_truth_inputs': [], 'frames': len(derived['frames'])}
    write(destination / 'manifest.json', value)
    ADAPTER.fastlio_packets(destination / 'manifest.json', sequence_id)
    return value


def run(protocol, group, fault, output, work):
    plan = read_plan(protocol)
    selected = [entry for entry in plan['inputs'] if entry['group'] == group]
    if len(selected) != 1 or fault not in plan['faults']: raise ValueError('undeclared fault cell')
    entry = selected[0]; output, work = Path(output), Path(work)
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'bhl-sensor-fault-result-v1', 'status': 'INCOMPLETE', 'group': group,
        'fault': fault, 'phase': plan['phase'], 'protocol_sha256': sha256(protocol),
        'original_manifest_sha256': entry['manifest_sha256'], 'rows': [], 'ground_truth_inputs': [],
        'closed_loop_episodes': 0}
    write(output / 'campaign_result.json', result)
    try:
        derived = derive_inputs(entry['manifest'], entry['sequence'], fault, work)
        write(output / 'fault_manifest.json', derived)
        # All native inference finishes before opening evaluator truth.
        for method in METHODS:
            destination = output / method
            if method == 'fast_lio2':
                measured = NATIVE.LIO.replay(work / 'manifest.json', entry['sequence'], plan['runtimes']['lio']['path'], destination)
            else:
                measured = NATIVE.LIVO.replay(work / 'manifest.json', entry['sequence'], plan['runtimes']['livo']['path'], destination,
                                             visual_enabled=method == 'fast_livo2')
            result['rows'].append({'method': method, 'native': measured})
        original, root = ADAPTER.read_manifest(entry['manifest'])
        sequence = copy.deepcopy(ADAPTER.choose_sequence(original, entry['sequence']))
        sequence['frames'] = sequence['frames'][1:-1]
        for row in result['rows']:
            destination = output / row['method']
            native = [json.loads(line) for line in (destination / 'native_frames.jsonl').read_text().splitlines()]
            row['evaluation'] = SCORE.evaluate(native, method='lio', replay_root=root, manifest=original,
                                                sequence=sequence, gate=plan['gate'])
        result.update(status='PASS', scientific_status='SMOKE_ONLY' if plan['phase'] == 'smoke' else 'PAIRED_DEVELOPMENT_CELL',
                      frames=len(sequence['frames']), no_visual_or_cloud_truth_feedback=True)
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc()
    result['files_sha256'] = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob('*'))
                              if path.is_file() and path != output / 'campaign_result.json'}
    write(output / 'campaign_result.json', result)
    return result


def collect(protocol, directories, output):
    plan = read_plan(protocol)
    expected = {(entry['group'], fault) for entry in plan['inputs'] for fault in plan['faults']}
    observed, problems = {}, []
    for directory in directories:
        directory = Path(directory)
        try:
            value = json.loads((directory / 'campaign_result.json').read_text())
            key = (value['group'], value['fault'])
            if key not in expected or key in observed: raise ValueError('unexpected or duplicate cell')
            if value.get('status') != 'PASS' or value.get('protocol_sha256') != sha256(protocol):
                raise ValueError('incomplete cell or wrong frozen protocol')
            if value.get('phase') != plan['phase']: raise ValueError('smoke/full mismatch')
            entry = next(e for e in plan['inputs'] if e['group'] == key[0])
            if value.get('original_manifest_sha256') != entry['manifest_sha256']:
                raise ValueError('wrong original paired recording')
            inventory = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p != directory / 'campaign_result.json'}
            if set(value['files_sha256']) != inventory: raise ValueError('complete raw inventory required')
            for relative, checksum in value['files_sha256'].items():
                p = Path(relative)
                if p.is_absolute() or '..' in p.parts or (directory / p).is_symlink() or sha256(directory / p) != checksum:
                    raise ValueError('retained raw output hash mismatch')
            if [r['method'] for r in value['rows']] != METHODS: raise ValueError('all native arms required')
            original, root = ADAPTER.read_manifest(entry['manifest'])
            sequence = copy.deepcopy(ADAPTER.choose_sequence(original, entry['sequence'])); sequence['frames'] = sequence['frames'][1:-1]
            for row in value['rows']:
                dest = directory / row['method']
                transport = json.loads((dest / 'transport.json').read_text())
                checker = NATIVE.LIO.validate_native_output if row['method'] == 'fast_lio2' else NATIVE.LIVO.validate_native_output
                raw = checker(dest / 'native_frames.jsonl', transport)
                if len(raw) != entry['source_frames'] - 2: raise ValueError('native frame count differs')
                if sha256(dest / 'input.bin') != transport['transport_sha256']: raise ValueError('native input changed')
                receipt = json.loads((dest / 'estimator_receipt.json').read_text())
                runtime = plan['runtimes']['lio' if row['method'] == 'fast_lio2' else 'livo']
                if receipt['runtime_sha256'] != runtime['binary_sha256'] or receipt.get('ground_truth_inputs') != []:
                    raise ValueError('native runtime/causality provenance differs')
                score = SCORE.evaluate(raw, method='lio', replay_root=root, manifest=original, sequence=sequence, gate=plan['gate'])
                if score != row['evaluation']: raise ValueError('independent native score differs')
            observed[key] = value
        except Exception as error:
            problems.append({'directory': str(directory), 'error': f'{type(error).__name__}: {error}'})
    missing = sorted(expected - observed.keys())
    paired = []
    for group in sorted(e['group'] for e in plan['inputs']):
        baseline = observed.get((group, 'nominal'))
        if baseline is None: continue
        for fault in plan['faults']:
            value = observed.get((group, fault))
            if value is None: continue
            for before, after in zip(baseline['rows'], value['rows']):
                bm, am = before['evaluation']['segments'][0]['metrics'], after['evaluation']['segments'][0]['metrics']
                paired.append({'group': group, 'fault': fault, 'method': after['method'],
                    'tracking_fraction_delta': after['evaluation']['post_initialization_tracked_fraction'] - before['evaluation']['post_initialization_tracked_fraction'],
                    'conditional_ate_rmse_delta_m': am['ate_translation_m']['rmse'] - bm['ate_translation_m']['rmse'] if bm and am else None,
                    'nominal_gate': before['evaluation']['replay_readiness_gate'], 'fault_gate': after['evaluation']['replay_readiness_gate']})
    result = {'schema': 'bhl-sensor-fault-collection-v1', 'status': 'PASS' if not missing and not problems else 'INCOMPLETE',
        'phase': plan['phase'], 'protocol_sha256': sha256(protocol), 'expected_cells': len(expected),
        'verified_cells': len(observed), 'missing': missing, 'problems': problems, 'paired_group_rows': paired,
        'scope': plan['scientific_scope'], 'claim_rule': 'Conditional ATE always accompanied by tracking; do not treat untracked or unscorable as zero error. Six development groups, no inferential population claim.'}
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('freeze'); p.add_argument('--manifests', nargs='+', type=Path, required=True)
    p.add_argument('--lio-runtime', type=Path, required=True); p.add_argument('--livo-runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--smoke', action='store_true')
    p = commands.add_parser('run'); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--group', required=True); p.add_argument('--fault', choices=tuple(FAULTS), required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--work', type=Path, required=True)
    p = commands.add_parser('collect'); p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--directories', nargs='*', type=Path, default=[]); p.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args()); command = args.pop('command')
    result = globals()[command](**args)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'source_sha256', 'files_sha256')}, indent=2))
    return 0 if result.get('status', 'PASS') == 'PASS' else 1


if __name__ == '__main__': raise SystemExit(main())
