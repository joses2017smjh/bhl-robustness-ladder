"""Do not allow pre-initialization or reset camera poses into navigation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/bench/stereo_inertial_navigation_campaign.py"
    spec = importlib.util.spec_from_file_location("test_inertial_nav", path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def raw(*, initialized, tracked=True):
    return {"timestamp_s": 5., "tracked": tracked, "tracking_state": 2 if tracked else 1,
            "map_id": 0, "compute_seconds": .02, "T_W_C": np.eye(4).tolist() if tracked else None,
            "inertial": {"map_imu_initialized": initialized, "imu_samples": 40}}


@pytest.mark.parametrize("tracked", [True, False])
def test_preinertial_visual_pose_cannot_command_navigation(tracked):
    result = module().inertial_navigation_packet(raw(initialized=False, tracked=tracked), np.eye(4), map_changed=False)
    assert result["tracked"] is False
    assert result["T_W_I"] is None
    assert result["state"] == "IMU_INITIALIZING_STOP"


def test_actual_initialized_native_camera_pose_converts_using_mounting_calibration():
    mod = module()
    result = mod.inertial_navigation_packet(raw(initialized=True), np.eye(4), map_changed=False)
    assert result["tracked"] is True
    np.testing.assert_allclose(result["T_W_I"], mod.STEREO.native_camera_to_imu(np.eye(4), np.eye(4)))


def test_map_frame_change_stops_even_after_inertial_initialization():
    result = module().inertial_navigation_packet(raw(initialized=True), np.eye(4), map_changed=True)
    assert result["tracked"] is False
    assert result["T_W_I"] is None
    assert result["state"] == "MAP_CHANGED_STOP"
    assert result["map_reset_id"] == 1
