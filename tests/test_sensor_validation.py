import numpy as np
import pytest

from bhl_robust.research.sensor_validation import (
    hardware_evidence, match_rectified_pair, rectification_residuals,
)


def test_vertical_shift_is_not_hidden_by_matching_filter():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(129)
    left = cv2.GaussianBlur(rng.integers(0, 256, (200, 320), dtype=np.uint8), (3, 3), .6)
    right = cv2.warpAffine(left, np.float32([[1, 0, -8], [0, 1, 5]]), (320, 200))
    lp, rp, _ = match_rectified_pair(left, right)
    row = rectification_residuals(lp, rp)
    assert row["matches"] > 20
    assert row["status"] == "EXCEEDS_DIAGNOSTIC_LIMIT"
    assert row["vertical_median_px"] == pytest.approx(5, abs=.15)
    assert row["signed_vertical_median_px"] == pytest.approx(-5, abs=.15)


def test_positive_control_and_textureless_negative():
    points = np.column_stack((np.arange(25) + 10, np.arange(25)))
    assert rectification_residuals(points, points - [4, 0])["status"] == "WITHIN_DIAGNOSTIC_LIMIT"
    assert rectification_residuals([], [])["status"] == "INSUFFICIENT_MATCHES"
    assert rectification_residuals(points[:2], points[:2])["status"] == "INSUFFICIENT_MATCHES"


@pytest.mark.parametrize("left,right", [([[1, float("nan")]], [[1, 2]]), ([[1, 2]], [])])
def test_invalid_points_fail(left, right):
    with pytest.raises(ValueError):
        rectification_residuals(left, right)


def test_simulated_inputs_cannot_become_hardware_validation():
    rig = {k: "recorded" for k in ("robot_or_rig", "left_camera", "right_camera", "lidar", "imu",
                                   "capture_date_utc", "clock_sync_evidence", "calibration_evidence")}
    manifest = {"physical_rig": rig, "sequences": [{"id": "a", "origin": "simulation",
                "frames": [{"left_timestamp_s": t, "right_timestamp_s": t + .0001,
                            "timestamp_s": t + .00005} for t in (0., .1)]}]}
    result = hardware_evidence(manifest, lidar_packets_valid=True,
                               epipolar_by_sequence={"a": {"status": "WITHIN_DIAGNOSTIC_LIMIT"}})
    assert result["status"] == "INCOMPLETE_PHYSICAL_EVIDENCE"
    assert "physical_origin" in result["missing_requirements"]
    manifest["sequences"][0]["origin"] = "physical"
    result = hardware_evidence(manifest, lidar_packets_valid=True,
                               epipolar_by_sequence={"a": {"status": "WITHIN_DIAGNOSTIC_LIMIT"}},
                               provenance_files_verified=True)
    assert result["status"] == "READY_FOR_NATIVE_REPLAY"
    assert result["hardware_validation_complete"] is False


def test_missing_or_unmeasured_sequence_cannot_pass_intake():
    result = hardware_evidence({"sequences": [{"id": "a", "origin": "physical"}]},
                               lidar_packets_valid=False, epipolar_by_sequence={})
    assert "all_sequences_rectification_measured" in result["missing_requirements"]
    assert "timed_3d_lidar_and_imu_valid" in result["missing_requirements"]


@pytest.mark.parametrize("times", [(0., 0.), (0., .1)])
def test_equal_stereo_clocks_do_not_prove_canonical_synchronization(times):
    frames = [{"left_timestamp_s": t, "right_timestamp_s": t, "timestamp_s": 10. + i}
              for i, t in enumerate(times)]
    result = hardware_evidence({"sequences": [{"id": "a", "origin": "physical", "frames": frames}]},
                               lidar_packets_valid=True,
                               epipolar_by_sequence={"a": {"status": "WITHIN_DIAGNOSTIC_LIMIT"}})
    assert result["checks"]["stereo_recorded_timestamps_within_1ms"]
    assert not result["checks"]["camera_clocks_match_canonical_monotonic_time"]
