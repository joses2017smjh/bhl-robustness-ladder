"""Original timing, causal stops and IMU continuity across sparse-scan loss."""
import json
import numpy as np
import pytest
from bhl_robust.research.terrain_confirmation_acquisition import scan_time_problem, validated_sensor_class
from bhl_robust.research.terrain_methods import path_command


def scan(times, frame=None):
    return dict(points_xyz_m=np.ones((len(times), 3)), point_time_s=np.array(times),
                ring_index=np.arange(len(times)), frame_timestamp_s=max(times) if frame is None else frame,
                origin="simulation")


class BaseSensors:
    def __init__(self):
        self.ready, self.recorded = [], []
        self.imu_cursor = 0
        self.imu_times = np.arange(.05, 1.01, .05)

    def newest(self):
        packet = self.ready[-1]
        self.ready.clear()
        end = int(np.searchsorted(self.imu_times, packet["frame_timestamp_s"], side="right"))
        selected = self.imu_times[self.imu_cursor:end]
        self.imu_cursor = end
        return packet, dict(timestamp_s=selected), 0

    def save(self, path):
        np.savez(path, imu=self.imu_times)


def test_zero_span_original_returns_are_retained_without_native_or_imu_consumption(tmp_path):
    sensors = validated_sensor_class(BaseSensors)()
    bad = scan([.2, .2, .2])
    sensors.ready.append(bad)
    assert sensors.newest() is None
    assert sensors.imu_cursor == 0
    assert np.array_equal(bad["point_time_s"], [.2, .2, .2])
    sensors.ready.append(scan([.35, .4]))
    packet, imu, _ = sensors.newest()
    assert packet["frame_timestamp_s"] == .4
    assert np.allclose(imu["timestamp_s"], np.arange(.05, .401, .05))
    sensors.save(tmp_path / "sensors.npz")
    receipt = json.loads((tmp_path / "rejected-scans.json").read_text())
    assert receipt["rejected_count"] == 1
    assert receipt["rejected_scans"][0]["packet"]["point_time_s"] == [.2, .2, .2]
    assert receipt["rejected_scans"][0]["reason"] == "zero_original_time_span"
    assert receipt["synthetic_points_or_timestamps"] is False


def test_valid_scan_retains_original_tail_and_point_order():
    sensors = validated_sensor_class(BaseSensors)()
    value = scan([.2, .15, .18])
    sensors.ready.append(value)
    assert sensors.newest()[0] is value
    assert np.array_equal(value["point_time_s"], [.2, .15, .18])
    assert not sensors.rejected_scans


def test_repeated_scan_never_rewinds_imu_cursor():
    sensors = validated_sensor_class(BaseSensors)()
    sensors.ready.append(scan([.15, .2]))
    sensors.newest()
    cursor = sensors.imu_cursor
    sensors.ready.append(scan([.15, .2]))
    assert sensors.newest() is None
    assert sensors.imu_cursor == cursor
    sensors.ready.append(scan([.35, .4]))
    _, imu, _ = sensors.newest()
    assert np.all(imu["timestamp_s"] > .2)


@pytest.mark.parametrize("value,previous,reason", [
    (scan([.2, .2]), None, "zero_original_time_span"),
    (scan([.2, .3]), .21, "overlapping_or_repeated_original_scan"),
    (scan([.2, .3], frame=.4), None, "original_tail_mismatch")])
def test_original_native_time_contract_preserved(value, previous, reason):
    assert scan_time_problem(value, previous) == reason


def test_measurement_loss_stops_every_arm_at_original_pose_age_limit():
    native = dict(tracked=True, map_reset_id=0, timestamp_s=1., T_W_I=np.eye(4).tolist())
    for arm in ("baseline", "heading", "regulated"):
        command, diagnostic, _ = path_command(native, np.eye(4), np.eye(4), np.empty((0, 3)), 1.36, arm)
        assert not command.any()
        assert diagnostic["reason"] == "native_stale_stop"
