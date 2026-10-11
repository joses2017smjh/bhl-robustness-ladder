"""Sensor-causality and failure-accounting checks for genuine FAST-LIVO2 IO."""
import copy
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from bhl_robust.research.native_livo import FRAME_HEADER, HEADER, IMU, POINT, validate_native_output, write_transport


def inputs():
    calibration = {'image_width': 640, 'image_height': 480, 'fx_px': 400., 'fy_px': 400., 'cx_left_px': 320., 'cy_px': 240.}
    png = b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', 640, 480)
    images = [{'timestamp_s': t, 'png_bytes': png} for t in (0., .1, .2)]
    packets = [{'kind': 'imu', 'capture_time_s': float(t), 'gyro_rad_s': [0., 0., .2], 'specific_force_m_s2': [0., 0., 9.81]}
               for t in np.arange(0., .251, .005)]
    for t in (0., .1, .2):
        packets.append({'kind': 'pointcloud2', 'capture_time_s': t,
                        'points': [[1., 0., 0., 0., 0., 0], [1., .1, 0., 0., .04, 1], [1., .2, 0., 0., .09, 2]]})
    return packets, images, calibration


def write(tmp_path, **kwargs):
    packets, images, calibration = inputs()
    return write_transport(packets, images, np.eye(4), np.eye(4), calibration, tmp_path / 'input.bin', **kwargs)


def test_sensor_time_partition_retains_initial_imu_and_excludes_future(tmp_path):
    receipt = write(tmp_path)
    raw = (tmp_path / 'input.bin').read_bytes()
    assert raw[:8] == b'BHLLIVO1'
    cursor = 8 + 2 * 12 * 8 + HEADER.size
    consumed_imus, consumed_points = [], []
    for timestamp in (0., .1, .2):
        beginning, end, npoints, nimus, nimage = FRAME_HEADER.unpack_from(raw, cursor); cursor += FRAME_HEADER.size
        assert end == timestamp
        for _ in range(npoints):
            point = POINT.unpack_from(raw, cursor); cursor += POINT.size
            capture = beginning + point[4]
            assert capture < timestamp + 1e-8
            consumed_points.append(capture)
        for _ in range(nimus):
            sample = IMU.unpack_from(raw, cursor); cursor += IMU.size
            assert sample[0] <= timestamp
            consumed_imus.append(sample[0])
            assert sample[-1] == 9.81
        cursor += nimage
    assert cursor == len(raw)
    assert consumed_imus[0] == 0.
    assert len(set(consumed_imus)) == len(consumed_imus)
    assert len(consumed_points) == 6
    assert receipt['consumed_point_count'] == 6
    assert receipt['unconsumed_lidar_tail_points'] == 3
    assert receipt['ground_truth_inputs'] == []
    assert receipt['right_camera_consumed'] is False


def test_invalid_extrinsics_rejected(tmp_path):
    packets, images, calibration = inputs()
    matrix = np.eye(4); matrix[0, 0] = 2
    with pytest.raises(ValueError, match='SO'):
        write_transport(packets, images, matrix, np.eye(4), calibration, tmp_path / 'bad.bin')


def test_image_bytes_calibration_mismatch_rejected(tmp_path):
    packets, images, calibration = inputs(); calibration['image_width'] = 320
    with pytest.raises(ValueError, match='PNG'):
        write_transport(packets, images, np.eye(4), np.eye(4), calibration, tmp_path / 'bad.bin')


def test_incomplete_imu_horizon_rejected(tmp_path):
    packets, images, calibration = inputs()
    packets = [p for p in packets if p['kind'] != 'imu' or p['capture_time_s'] < .15]
    with pytest.raises(ValueError, match='span'):
        write_transport(packets, images, np.eye(4), np.eye(4), calibration, tmp_path / 'bad.bin')


def test_overlapping_lidar_scans_rejected(tmp_path):
    packets, images, calibration = inputs()
    [p for p in packets if p['kind'] == 'pointcloud2'][1]['capture_time_s'] = .05
    with pytest.raises(ValueError, match='nonoverlapping'):
        write_transport(packets, images, np.eye(4), np.eye(4), calibration, tmp_path / 'bad.bin')


def test_transport_never_overwrites_existing_evidence(tmp_path):
    (tmp_path / 'input.bin').write_bytes(b'original')
    with pytest.raises(FileExistsError): write(tmp_path)
    assert (tmp_path / 'input.bin').read_bytes() == b'original'


def rows(receipt):
    return [{'frame_index': i, 'timestamp_s': t, 'tracked': False, 'state': 'IMU_INITIALIZING',
             'map_reset_id': 0, 'compute_seconds': .01, 'image_decode_seconds': .002,
             'effective_points': 0, 'visual_points': 0, 'visual_update_attempted': False, 'T_W_I': None}
            for i, t in enumerate(receipt['expected_timestamps_s'])]


def output(tmp_path, data):
    path = tmp_path / 'frames.jsonl'; path.write_text('\n'.join(json.dumps(row) for row in data) + '\n'); return path


def test_untracked_frames_are_explicit_and_cannot_invent_pose(tmp_path):
    receipt = write(tmp_path); data = rows(receipt)
    assert len(validate_native_output(output(tmp_path, data), receipt)) == 3
    data[0]['T_W_I'] = np.eye(4).tolist()
    with pytest.raises(ValueError, match='fabricate'):
        validate_native_output(output(tmp_path, data), receipt)


def test_vio_ablation_cannot_silently_use_camera_updates(tmp_path):
    receipt = write(tmp_path, visual_enabled=False); data = rows(receipt)
    data[1].update(visual_update_attempted=True, visual_points=4)
    with pytest.raises(ValueError, match='ablation'):
        validate_native_output(output(tmp_path, data), receipt)


def test_tracked_requires_finite_pose_and_native_measurement_support(tmp_path):
    receipt = write(tmp_path); data = rows(receipt)
    data[1].update(tracked=True, state='TRACKED', T_W_I=np.eye(4).tolist())
    with pytest.raises(ValueError, match='support'):
        validate_native_output(output(tmp_path, data), receipt)
    data[1].update(visual_points=4, visual_update_attempted=True)
    assert validate_native_output(output(tmp_path, data), receipt)[1]['tracked'] is True
    data[1]['compute_seconds'] = float('nan')
    with pytest.raises(ValueError, match='timing'):
        validate_native_output(output(tmp_path, data), receipt)


def test_missing_native_camera_frame_rejected(tmp_path):
    receipt = write(tmp_path)
    with pytest.raises(ValueError, match='omitted'):
        validate_native_output(output(tmp_path, rows(receipt)[:-1]), receipt)
