"""Physical consistency of the generator's time/IMU conventions."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("slam_capture_test", Path(__file__).parents[1] / "scripts/bench/slam_capture_campaign.py")
CAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAP)


def test_stationary_initialization_has_gravity_and_no_angular_rate():
    for t in [0., 1., 4.999, 5.]:
        pose, gyro, force = CAP.motion(t)
        np.testing.assert_allclose(pose[:3, 3], [0., 0., .75], atol=1e-12)
        np.testing.assert_allclose(gyro, 0., atol=1e-12)
        np.testing.assert_allclose(force, [0., 0., 9.81], atol=1e-12)


@pytest.mark.parametrize("t", [5.2, 7.8, 14.1, 26.6])
def test_imu_acceleration_matches_finite_difference_motion(t):
    h = 1e-3
    a, b, c = CAP.motion(t-h), CAP.motion(t), CAP.motion(t+h)
    acceleration = (c[0][:3, 3] - 2*b[0][:3, 3] + a[0][:3, 3]) / h**2
    world_force = b[0][:3, :3] @ b[2]
    np.testing.assert_allclose(world_force + [0., 0., -9.81], acceleration, atol=2e-7)
    yaw = lambda pose: np.arctan2(pose[1, 0], pose[0, 0])
    assert b[1][2] == pytest.approx((yaw(c[0])-yaw(a[0]))/(2*h), abs=2e-7)


def test_packet_grid_continuity_and_closed_loop_path():
    first, second = CAP.imu_packet(.1, .1), CAP.imu_packet(.2, .1)
    times = np.r_[first["timestamp_s"], second["timestamp_s"]]
    np.testing.assert_allclose(np.diff(times), .005, atol=1e-12)
    np.testing.assert_allclose(CAP.rig_pose(30.), CAP.rig_pose(0.), atol=1e-12)


@pytest.mark.parametrize("stamp", [-1., float("nan"), float("inf")])
def test_bad_capture_time_rejected(stamp):
    with pytest.raises(ValueError): CAP.motion(stamp)
