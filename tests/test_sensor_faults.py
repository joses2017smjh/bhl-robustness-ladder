"""Causality and paired acquisition invariants, including native binary parsing."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import numpy as np
import pytest

from bhl_robust.research.sensor_faults import FAULTS, perturb_imu, perturb_cloud, perturb_calibration
from bhl_robust.research.native_lio import IMU, POINT, write_transport, sha256

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sensor_fault_tests_campaign', ROOT/'scripts/bench/sensor_fault_campaign.py')
CAMPAIGN = importlib.util.module_from_spec(spec); spec.loader.exec_module(CAMPAIGN)


def original(tmp_path):
    root = tmp_path/'original'; (root/'inference').mkdir(parents=True)
    png = b'\x89PNG\r\n\x1a\n'+struct.pack('>I', 13)+b'IHDR'+struct.pack('>II', 640, 480)
    c = dict(schema='bhl-rectified-stereo-v1', rectified=True, image_width=640, image_height=480,
        fx_px=400., fy_px=400., cx_left_px=320., cx_right_px=320., cy_px=240., baseline_m=.12,
        T_B_C=np.eye(4).tolist(), T_B_L=np.eye(4).tolist(), T_B_I=np.eye(4).tolist())
    seq = dict(id='textured_boxes-g610000-nominal', scene_id='textured_boxes-g610000', origin='simulation', split='dev', frames=[])
    for index in range(8):
        for camera in ('left', 'right'): (root/f'inference/{camera}{index}.png').write_bytes(png)
        times = index*.2 + np.linspace(0., .05, 16)
        np.savez_compressed(root/f'inference/lidar{index}.npz', points_xyz_m=np.ones((16, 3)),
                            point_time_s=times, ring_index=np.arange(16))
        imu = index*.2 + np.arange(40)*.005
        np.savez_compressed(root/f'inference/imu{index}.npz', timestamp_s=imu,
            gyro_rad_s=np.zeros((40, 3)), specific_force_m_s2=np.tile([0., 0., 9.81], (40, 1)))
        seq['frames'].append(dict(timestamp_s=index*.2+.05, left=f'inference/left{index}.png',
            right=f'inference/right{index}.png', lidar=f'inference/lidar{index}.npz', imu=f'inference/imu{index}.npz',
            truth='evaluator/must-never-open.npz'))
    value = dict(schema='bhl-sensor-replay-v1', calibration=c, lidar_dimension=3,
        clock_domain='simulation_time', frame_rate_hz=5., sequences=[seq],
        cell=dict(condition='nominal', geometry_group='textured_boxes-g610000', id=seq['id']),
        file_sha256={str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()})
    (root/'manifest.json').write_text(json.dumps(value))
    return root/'manifest.json', value


@pytest.mark.parametrize('fault', list(FAULTS))
def test_derived_inputs_exclude_truth_and_preserve_exact_paired_frame_times(tmp_path, fault):
    path, source = original(tmp_path); source_hash = sha256(path)
    value = CAMPAIGN.derive_inputs(path, source['sequences'][0]['id'], fault, tmp_path/fault)
    frames = value['sequences'][0]['frames']
    assert [f['timestamp_s'] for f in frames] == [f['timestamp_s'] for f in source['sequences'][0]['frames'][1:-1]]
    assert all('truth' not in f for f in frames)
    assert all(p.startswith('inference/') for p in value['file_sha256'])
    assert sha256(path) == source_hash
    for frame in frames:
        for key in ('left', 'right'):
            assert value['file_sha256'][frame[key]] == source['file_sha256'][frame[key]]
    CAMPAIGN.ADAPTER.fastlio_packets(tmp_path/fault/'manifest.json', source['sequences'][0]['id'])


def test_positive_clock_offset_cannot_expose_future_real_samples(tmp_path):
    path, source = original(tmp_path)
    CAMPAIGN.derive_inputs(path, source['sequences'][0]['id'], 'imu_clock_plus80ms', tmp_path/'derived')
    _, _, extrinsic, packets, _ = CAMPAIGN.ADAPTER.fastlio_packets(tmp_path/'derived/manifest.json', source['sequences'][0]['id'])
    receipt = write_transport(packets, extrinsic, tmp_path/'native.bin')
    data = (tmp_path/'native.bin').read_bytes(); cursor = 8 + 12*8 + struct.calcsize('<3dII')
    for tail in receipt['expected_timestamps_s']:
        beginning, end, npoints, nimus = struct.unpack_from('<ddII', data, cursor); cursor += struct.calcsize('<ddII')
        assert end == tail
        cursor += npoints*POINT.size
        for _ in range(nimus):
            sample = IMU.unpack_from(data, cursor); cursor += IMU.size
            real_acquisition = sample[0] - .08
            assert real_acquisition <= sample[0] <= end
    assert cursor == len(data)


def test_bias_is_injected_after_stationary_initialization_only():
    times = np.array([0., 4.9, 5., 5.1]); gyro = np.zeros((4, 3)); accel = np.ones((4, 3))
    out = perturb_imu(times, gyro, accel, 'gyro_z_bias_after5s')
    assert np.array_equal(out['gyro_rad_s'][:2], gyro[:2])
    assert np.all(out['gyro_rad_s'][2:, 2] == .05)
    assert np.array_equal(out['specific_force_m_s2'], accel)
    assert np.array_equal(gyro, np.zeros((4, 3)))
    out = perturb_imu(times, gyro, accel, 'accel_x_bias_after5s')
    assert np.all(out['specific_force_m_s2'][2:, 0] == 1.3)


def test_dropout_retains_original_offsets_and_common_temporal_endpoints():
    arrays = dict(points_xyz_m=np.arange(48).reshape(16,3), point_time_s=np.linspace(0, .05, 16), ring_index=np.arange(16))
    out = perturb_cloud(arrays, 'lidar_keep_quarter')
    assert np.array_equal(out['ring_index'], [0,4,8,12,15])
    assert out['point_time_s'][0] == arrays['point_time_s'][0]
    assert out['point_time_s'][-1] == arrays['point_time_s'][-1]
    assert np.array_equal(out['points_xyz_m'], arrays['points_xyz_m'][out['ring_index']])


def test_calibration_error_changes_declared_lidar_transform_only(tmp_path):
    _, source = original(tmp_path); before = copy.deepcopy(source['calibration'])
    out = perturb_calibration(before, 'lidar_extrinsic_yaw2deg_x2cm')
    assert out['T_B_I'] == before['T_B_I'] and out['T_B_C'] == before['T_B_C']
    assert before == source['calibration']
    assert out['T_B_L'][0][3] == .02
    assert np.allclose(np.array(out['T_C_L']), np.linalg.inv(out['T_B_C']) @ out['T_B_L'])


def test_fault_names_cannot_inject_unregistered_tuning():
    with pytest.raises(ValueError, match='predeclared'):
        perturb_imu([0., .1], np.zeros((2,3)), np.zeros((2,3)), 'imu_clock_minus80ms')


def test_full_protocol_refuses_old_or_partial_cohort(tmp_path):
    path, _ = original(tmp_path)
    with pytest.raises(ValueError, match='150'):
        CAMPAIGN.freeze([path], '/absent', '/absent', tmp_path/'protocol.json')


def test_collector_reports_all_missing_without_inventing_results(tmp_path, monkeypatch):
    plan = dict(inputs=[dict(group='textured_boxes-g610000')], faults=FAULTS, phase='run', scientific_scope='replay')
    p = tmp_path/'plan.json'; p.write_text('{}')
    monkeypatch.setattr(CAMPAIGN, 'read_plan', lambda path: plan)
    result = CAMPAIGN.collect(p, [], tmp_path/'collection.json')
    assert result['status'] == 'INCOMPLETE'
    assert result['expected_cells'] == 8 and result['verified_cells'] == 0
    assert result['paired_group_rows'] == []


def test_collector_rejects_smoke_as_full(tmp_path, monkeypatch):
    plan = dict(inputs=[dict(group='textured_boxes-g610000')], faults=FAULTS, phase='run', scientific_scope='replay')
    p = tmp_path/'plan.json'; p.write_text('{}'); run = tmp_path/'run'; run.mkdir()
    (run/'campaign_result.json').write_text(json.dumps(dict(group='textured_boxes-g610000', fault='nominal',
        status='PASS', protocol_sha256=sha256(p), phase='smoke')))
    monkeypatch.setattr(CAMPAIGN, 'read_plan', lambda path: plan)
    result = CAMPAIGN.collect(p, [run], tmp_path/'collection.json')
    assert result['status'] == 'INCOMPLETE' and result['verified_cells'] == 0
    assert 'smoke/full' in result['problems'][0]['error']
