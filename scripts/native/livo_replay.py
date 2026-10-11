#!/usr/bin/env python3
"""Replay calibrated original left-camera/LiDAR/IMU with native FAST-LIVO2."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from bhl_robust.research.native_livo import UPSTREAM_COMMIT, sha256, validate_native_output, write_transport
from bhl_robust.research.pose_metrics import transform


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def replay(manifest_path, sequence, runtime_dir, output, *, visual_enabled=True, surface_voxel_m=.15, blind_m=.1):
    runtime_dir, output = Path(runtime_dir).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('native replay output must be new')
    runtime = json.loads((runtime_dir / 'runtime.json').read_text())
    binary = runtime_dir / 'fastlivo_headless'
    if (runtime.get('schema') != 'bhl-native-livo-runtime-v1' or runtime.get('status') != 'PASS'
            or runtime.get('upstream_commit') != UPSTREAM_COMMIT or sha256(binary) != runtime.get('binary_sha256')):
        raise ValueError('checksum-verified pinned genuine FAST-LIVO2 runtime required')
    for name, checksum in runtime['libraries_sha256'].items():
        if Path(name).name != name or sha256(runtime_dir / 'lib' / name) != checksum:
            raise ValueError('native runtime library checksum differs')
    adapter_path = Path(__file__).parents[1] / 'bench/pose_research.py'
    spec = importlib.util.spec_from_file_location('bhl_pose_adapter_livo', adapter_path)
    adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    manifest, seq, t_i_l, packets, max_gap = adapter.fastlio_packets(manifest_path, sequence)
    calibration = manifest['calibration']
    t_b_c = transform(calibration['T_B_C']); t_b_l = transform(calibration['T_B_L'])
    t_c_l = np.linalg.inv(t_b_c) @ t_b_l
    if 'T_C_L' in calibration and not np.allclose(transform(calibration['T_C_L']), t_c_l, atol=1e-9, rtol=0):
        raise ValueError('camera and lidar calibration transforms disagree')
    root = Path(manifest_path).resolve().parent
    images = [{'timestamp_s': frame['timestamp_s'], 'png_bytes': adapter.inference_path(root, frame['left']).read_bytes()}
              for frame in seq['frames']]
    output.mkdir(parents=True)
    started = time.monotonic()
    transport = write_transport(packets, images, t_i_l, t_c_l, calibration, output / 'input.bin',
                                surface_voxel_m=surface_voxel_m, blind_m=blind_m, visual_enabled=visual_enabled)
    write_json(output / 'transport.json', transport)
    env = dict(os.environ, LD_LIBRARY_PATH=str(runtime_dir / 'lib'), OMP_NUM_THREADS='1')
    command = [str(binary), str(output / 'input.bin'), str(output / 'native_frames.jsonl')]
    native_started = time.monotonic()
    with (output / 'native.log').open('x') as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    native_wall = time.monotonic() - native_started
    if process.returncode:
        write_json(output / 'campaign_result.json', {'status': 'INCOMPLETE', 'method': 'fast_livo2_camera_lidar_imu',
                   'native_returncode': process.returncode, 'reason': 'native process failed; see retained log',
                   'estimated_pose_outputs': None, 'closed_loop_episodes': 0})
        raise RuntimeError('native FAST-LIVO2 process failed')
    rows = validate_native_output(output / 'native_frames.jsonl', transport)
    trajectory = {'schema': 'bhl-pose-trajectory-v1', 'sequence': sequence, 'sensor_frame': 'imu_body',
                  'clock_domain': manifest['clock_domain'], 'origin': seq['origin'], 'split': seq['split'],
                  'end_time_s': rows[-1]['timestamp_s'], 'frames': [
                      {'timestamp_s': row['timestamp_s'], 'tracked': row['tracked'],
                       'T_W_S': row['T_W_I'] if row['tracked'] else np.eye(4).tolist(),
                       'native_state': row['state'], 'pose_is_storage_placeholder': not row['tracked']} for row in rows],
                  'untracked_pose_semantics': 'Identity is a storage placeholder, never scored or supplied to control; authoritative native frames use null.'}
    write_json(output / 'trajectory.json', trajectory)
    method = 'fast_livo2_camera_lidar_imu' if visual_enabled else 'fast_livo2_lidar_imu_ablation'
    receipt = {'schema': 'bhl-pose-estimator-run-v1', 'method': method, 'upstream_commit': UPSTREAM_COMMIT,
               'runtime_sha256': runtime['binary_sha256'], 'runtime_receipt_sha256': sha256(runtime_dir / 'runtime.json'),
               'input_manifest_sha256': sha256(manifest_path), 'sequence': sequence,
               'native_command': command, 'transport': transport, 'sensor_frame': 'imu_body', 'ground_truth_inputs': [],
               'calibration': calibration, 'config': transport['config'],
               'input_files_sha256': adapter._input_receipt(root, seq, ('left', 'lidar', 'imu')),
               'trajectory_sha256': sha256(output / 'trajectory.json'),
               'native_frames_sha256': sha256(output / 'native_frames.jsonl'),
               'right_camera_consumed': False, 'max_imu_gap_s': max_gap,
               'timing_scope': 'Native per-camera IMU deskew, voxel LIO, map update and direct photometric VIO. PNG decoding separately retained. Full native process wall includes startup and IO; capture is excluded.',
               'native_process_wall_seconds': native_wall,
               'tracking_semantics': 'At least one native lidar residual or visual patch and finite state; this is not calibrated confidence.'}
    import hashlib
    for key in ('calibration', 'config'):
        receipt[key + '_sha256'] = hashlib.sha256(json.dumps(receipt[key], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    write_json(output / 'estimator_receipt.json', receipt)
    result = {'schema': 'bhl-native-livo-result-v1', 'status': 'PASS',
              'scientific_status': 'NATIVE_REPLAY_MEASURED_NOT_COMPARISON', 'method': method,
              'sequence': sequence, 'origin': seq['origin'], 'split': seq['split'], 'frames': len(rows),
              'tracked_frames': sum(row['tracked'] for row in rows),
              'estimated_pose_outputs': sum(row['tracked'] for row in rows),
              'visual_update_attempted_frames': sum(row['visual_update_attempted'] for row in rows),
              'frames_with_visual_measurements': sum(row['visual_points'] > 0 for row in rows),
              'states': {state: sum(row['state'] == state for row in rows) for state in sorted({r['state'] for r in rows})},
              'native_compute_p95_ms': float(np.quantile([r['compute_seconds'] * 1000 for r in rows], .95)),
              'native_decode_and_compute_p95_ms': float(np.quantile([(r['image_decode_seconds'] + r['compute_seconds']) * 1000 for r in rows], .95)),
              'native_process_wall_seconds': native_wall, 'elapsed_seconds': time.monotonic() - started,
              'ground_truth_inputs': [], 'closed_loop_episodes': 0,
              'trajectory_sha256': receipt['trajectory_sha256'], 'estimator_receipt_sha256': sha256(output / 'estimator_receipt.json')}
    write_json(output / 'campaign_result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True); parser.add_argument('--sequence', required=True)
    parser.add_argument('--runtime', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--disable-visual', action='store_true')
    parser.add_argument('--surface-voxel-m', type=float, default=.15); parser.add_argument('--blind-m', type=float, default=.1)
    args = parser.parse_args()
    print(json.dumps(replay(args.manifest, args.sequence, args.runtime, args.output,
                           visual_enabled=not args.disable_visual, surface_voxel_m=args.surface_voxel_m, blind_m=args.blind_m)))


if __name__ == '__main__': main()
