"""Declared sensor-only perturbations for paired native-estimator replay.

Positive IMU clock errors delay availability; they never make future acquisition
samples available early. No transform below accepts an evaluator pose.
"""
from __future__ import annotations
import copy
import math
import numpy as np

FAULTS = {
    'nominal': {},
    'imu_clock_plus20ms': {'imu_clock_offset_s': .02},
    'imu_clock_plus80ms': {'imu_clock_offset_s': .08},
    'lidar_extrinsic_yaw2deg_x2cm': {'lidar_yaw_error_deg': 2., 'lidar_x_error_m': .02},
    'gyro_z_bias_after5s': {'gyro_bias_rad_s': [0., 0., .05], 'bias_start_s': 5.},
    'accel_x_bias_after5s': {'accel_bias_m_s2': [.3, 0., 0.], 'bias_start_s': 5.},
    'lidar_keep_half': {'keep_every': 2},
    'lidar_keep_quarter': {'keep_every': 4},
}
METHODS = ['fast_lio2', 'fast_livo2', 'fast_livo2_no_visual']
GROUPS = [f'{family}-g{seed}' for family in ('textured_boxes', 'thin_posts', 'ramp_step')
          for seed in (610000, 610001)]
GATE = {'initialization_exclusion_s': 5., 'maximum_ate_rmse_m': .1,
        'maximum_native_compute_p95_s': .1, 'maximum_rotation_p95_deg': 5.,
        'minimum_tracking_fraction': .9}


def fault_spec(name):
    if name not in FAULTS:
        raise ValueError('unknown predeclared sensor fault')
    return copy.deepcopy(FAULTS[name])


def perturb_imu(timestamp_s, gyro_rad_s, specific_force_m_s2, fault):
    spec = fault_spec(fault)
    times = np.asarray(timestamp_s, dtype=float)
    gyro, accel = np.array(gyro_rad_s, dtype=float), np.array(specific_force_m_s2, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or gyro.shape != (len(times), 3) or accel.shape != gyro.shape
            or not all(np.isfinite(x).all() for x in (times, gyro, accel)) or np.any(np.diff(times) <= 0)):
        raise ValueError('finite increasing original SI-unit IMU required')
    active = times >= spec.get('bias_start_s', math.inf)
    gyro[active] += np.asarray(spec.get('gyro_bias_rad_s', [0., 0., 0.]))
    accel[active] += np.asarray(spec.get('accel_bias_m_s2', [0., 0., 0.]))
    return {'timestamp_s': times + spec.get('imu_clock_offset_s', 0.),
            'gyro_rad_s': gyro, 'specific_force_m_s2': accel}


def perturb_cloud(arrays, fault):
    spec = fault_spec(fault)
    names = {'points_xyz_m', 'point_time_s', 'ring_index', 'intensity'}
    if not {'points_xyz_m', 'point_time_s', 'ring_index'} <= set(arrays):
        raise ValueError('original timed point arrays required')
    n = len(arrays['point_time_s'])
    if n < 8:
        raise ValueError('at least eight original returns required')
    # Stable original index subsampling. Preserve both temporal endpoints so
    # every arm has exactly the same scan beginning/tail and scoring timestamps.
    keep = np.arange(n) % spec.get('keep_every', 1) == 0
    keep[[0, -1]] = True
    return {k: np.array(v)[keep] for k, v in arrays.items() if k in names}


def perturb_calibration(calibration, fault):
    from .pose_metrics import transform
    spec = fault_spec(fault)
    value = copy.deepcopy(calibration)
    correction = np.eye(4)
    yaw = math.radians(spec.get('lidar_yaw_error_deg', 0.))
    correction[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    correction[0, 3] = spec.get('lidar_x_error_m', 0.)
    value['T_B_L'] = (correction @ transform(value['T_B_L'])).tolist()
    value['T_C_L'] = (np.linalg.inv(transform(value['T_B_C'])) @ transform(value['T_B_L'])).tolist()
    # T_B_I and all sensor measurements remain unchanged by calibration error.
    return value
