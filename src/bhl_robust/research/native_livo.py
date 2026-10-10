"""Causal recorded-sensor transport for the actual pinned FAST-LIVO2 runtime.

No trajectory inference is implemented in Python. Sensor timestamps, calibration,
and original image bytes are sent to the unchanged C++ estimator units.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import struct

import numpy as np

from .native_lio import IMU, POINT, sha256
from .pose_metrics import transform

UPSTREAM_COMMIT = '0d2c0346107b75b59934975adec9a6eeeb913c64'
HEADER = struct.Struct('<II6dII')
FRAME_HEADER = struct.Struct('<ddIII')


def write_transport(packets, images, t_imu_lidar, t_camera_lidar, calibration, destination,
                    *, surface_voxel_m=.15, blind_m=.1, visual_enabled=True):
    """Partition raw points at camera times using the upstream strict-before rule.

    Each native sequential LIO/VIO update sees points captured before that image
    and IMU captured through it. Late returns are retained for the next image;
    tail points after the last image are counted explicitly as unconsumed.
    """
    t_i_l, t_c_l = transform(t_imu_lidar), transform(t_camera_lidar)
    width, height = calibration['image_width'], calibration['image_height']
    intrinsics = [calibration[k] for k in ('fx_px', 'fy_px', 'cx_left_px', 'cy_px')]
    if (type(width) is not int or type(height) is not int or not 128 <= width <= 4096 or not 128 <= height <= 4096
            or not np.isfinite(intrinsics).all() or min(intrinsics[:2]) <= 0):
        raise ValueError('valid bounded original camera calibration required')
    if (not math.isfinite(surface_voxel_m) or surface_voxel_m <= 0
            or not math.isfinite(blind_m) or blind_m < 0 or type(visual_enabled) is not bool):
        raise ValueError('invalid native LIVO parameters')
    points, imus, starts = [], [], []
    for packet in packets:
        if packet.get('kind') == 'pointcloud2':
            start = float(packet['capture_time_s']); rows = np.asarray(packet['points'], dtype=float)
            if (not math.isfinite(start) or rows.ndim != 2 or rows.shape[1] != 6 or not 1 < len(rows) <= 100000
                    or not np.isfinite(rows).all() or np.any(rows[:, 4] < 0) or rows[-1, 4] <= 0
                    or np.any(np.diff(rows[:, 4]) < 0) or np.any(rows[:, 5] != np.floor(rows[:, 5]))
                    or np.any(rows[:, 5] < 0) or np.any(rows[:, 5] > 127)):
                raise ValueError('finite original XYZ, intensity, sorted point offsets and integer rings required')
            starts.append((start, start + rows[-1, 4]))
            points.extend((start + p[4], [*p[:4], 0., p[5]]) for p in rows)
        elif packet.get('kind') == 'imu':
            values = np.asarray([packet['capture_time_s'], *packet['gyro_rad_s'], *packet['specific_force_m_s2']], dtype=float)
            if values.shape != (7,) or not np.isfinite(values).all(): raise ValueError('finite original SI IMU required')
            imus.append(values)
        else: raise ValueError('unsupported native packet kind')
    if not points or not imus or not images: raise ValueError('camera, timed LiDAR and IMU are all required')
    if any(b[0] <= a[1] for a, b in zip(starts, starts[1:])):
        raise ValueError('nonoverlapping increasing original scan times required')
    times = np.asarray([float(image['timestamp_s']) for image in images])
    imu_times = np.asarray([row[0] for row in imus])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or np.any(np.diff(imu_times) <= 0):
        raise ValueError('camera and IMU clocks must increase strictly')
    if imu_times[0] > min(times[0], points[0][0]) or imu_times[-1] < times[-1]:
        raise ValueError('original high-rate IMU must span all camera updates')
    if np.max(np.diff(imu_times), initial=0) > .02 + 1e-8:
        raise ValueError('original IMU gaps may not exceed 20 ms')
    if times[0] < points[0][0] - .2 or times[-1] > points[-1][0] + .2 + 1e-8:
        raise ValueError('camera clock does not overlap recorded LiDAR coverage')
    prepared, point_index, imu_index = [], 0, 0
    started_cloud = False
    prior = min(float(times[0]), float(points[0][0]))
    image_receipts = []
    for image, timestamp in zip(images, times):
        payload = image['png_bytes']
        if (not isinstance(payload, bytes) or len(payload) > 32 * 1024**2 or len(payload) < 24
                or payload[:8] != b'\x89PNG\r\n\x1a\n' or payload[12:16] != b'IHDR'
                or struct.unpack('>II', payload[16:24]) != (width, height)):
            raise ValueError('original PNG must match camera calibration')
        frame_points = []
        while point_index < len(points) and points[point_index][0] < timestamp:
            point_time, point = points[point_index]
            if point_time < prior - 1e-9: raise ValueError('point would be replayed into an earlier interval')
            point = point.copy(); point[4] = max(0., point_time - prior)
            frame_points.append(point); point_index += 1
        next_imu = int(np.searchsorted(imu_times, timestamp, side='right'))
        # The first image can precede every return. Native initialization needs
        # those initial IMUs with the first usable cloud, so do not discard them.
        started_cloud = started_cloud or len(frame_points) >= 2
        samples = imus[imu_index:next_imu] if started_cloud else []
        if len(frame_points) > 100000 or len(samples) > 10000: raise ValueError('native packet exceeds bounded buffers')
        if started_cloud:
            imu_index = next_imu
        prepared.append((prior, float(timestamp), frame_points, samples, payload))
        prior = float(timestamp)
        import hashlib
        image_receipts.append({'timestamp_s': float(timestamp), 'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)})
    path = Path(destination)
    with path.open('xb') as stream:
        stream.write(b'BHLLIVO1')
        for matrix in (t_i_l, t_c_l):
            stream.write(struct.pack('<12d', *matrix[:3, :3].ravel(), *matrix[:3, 3]))
        stream.write(HEADER.pack(width, height, *intrinsics, surface_voxel_m, blind_m, len(prepared), int(visual_enabled)))
        for beginning, timestamp, cloud, samples, png in prepared:
            stream.write(FRAME_HEADER.pack(beginning, timestamp, len(cloud), len(samples), len(png)))
            for point in cloud: stream.write(POINT.pack(*point[:5], int(point[5])))
            for sample in samples: stream.write(IMU.pack(*sample))
            stream.write(png)
    return {'schema': 'bhl-native-livo-transport-v1', 'expected_timestamps_s': times.tolist(), 'image_count': len(images),
            'raw_point_count': len(points), 'consumed_point_count': point_index,
            'unconsumed_lidar_tail_points': len(points) - point_index, 'imu_count': imu_index,
            'T_imu_lidar': t_i_l.tolist(), 'T_camera_lidar': t_c_l.tolist(), 'images': image_receipts,
            'config': {'surface_voxel_m': surface_voxel_m, 'blind_m': blind_m, 'visual_enabled': visual_enabled,
                       'voxel_size_m': .5, 'max_layer': 2, 'max_iterations': 5,
                       'depth_error_m': .02, 'beam_error_deg': .05, 'plane_eigenvalue_threshold': .0025,
                       'imu_init_samples': 30, 'gyro_covariance': .3, 'acceleration_covariance': .5,
                       'patch_size_px': 8, 'pyramid_levels': 4, 'exposure_estimation': True},
            'ground_truth_inputs': [], 'right_camera_consumed': False,
            'transport_sha256': sha256(path), 'transport_bytes': path.stat().st_size}


def validate_native_output(path, transport):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    expected = transport['expected_timestamps_s']
    if len(rows) != len(expected): raise ValueError('native estimator omitted original camera frames')
    for index, (row, timestamp) in enumerate(zip(rows, expected)):
        if row.get('frame_index') != index or not math.isclose(row.get('timestamp_s', math.nan), timestamp, rel_tol=0, abs_tol=1e-9):
            raise ValueError('native frame index or timestamp changed')
        if type(row.get('tracked')) is not bool or not isinstance(row.get('state'), str) or row.get('map_reset_id') != 0:
            raise ValueError('explicit native tracking/reset state required')
        for key in ('compute_seconds', 'image_decode_seconds'):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid native timing')
        for key in ('effective_points', 'visual_points'):
            if type(row.get(key)) is not int or row[key] < 0: raise ValueError('invalid native measurement counts')
        if type(row.get('visual_update_attempted')) is not bool: raise ValueError('explicit visual participation required')
        if not row['visual_update_attempted'] and row['visual_points'] != 0:
            raise ValueError('visual measurements reported without native VIO')
        if not transport['config']['visual_enabled'] and row['visual_update_attempted']:
            raise ValueError('visual ablation unexpectedly consumed images')
        if row['tracked']:
            transform(row.get('T_W_I'))
            if row['effective_points'] + row['visual_points'] < 1 or row['state'] != 'TRACKED':
                raise ValueError('tracked state lacks native measurement support')
        elif row.get('T_W_I') is not None:
            raise ValueError('untracked output cannot fabricate a pose')
    return rows
