"""Adversarial sensor/frame checks for hybrid map recovery (synthetic unit data)."""
from dataclasses import replace
import numpy as np
import pytest
from bhl_robust.research.stereo_recovery import RecoveryConfig, VerifiedMapRecovery, mean_transform, pose_residual
from bhl_robust.research.native_orb import validate_response


def pose(x=0., yaw=0.):
    value = np.eye(4)
    value[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    value[0, 3] = x
    return value


def samples(t, map_id=0, offset=0.):
    reference = {"timestamp_s": t, "tracked": True, "map_reset_id": 0,
                 "effective_points": 100, "T_W_I": pose(.1*t).tolist()}
    stereo = {"timestamp_s": t, "tracked": True, "map_id": map_id,
              "T_W_I": pose(.1*t+offset).tolist()}
    return stereo, reference


def test_initial_map_pose_is_stereo_not_reference_fallback():
    recovery = VerifiedMapRecovery()
    stereo, reference = samples(1., offset=2.)
    result = recovery.update(stereo, reference)
    assert result["tracked"]
    np.testing.assert_allclose(result["T_W_I"], stereo["T_W_I"])
    assert not np.allclose(result["T_W_I"], reference["T_W_I"])


def test_map_change_stops_until_disjoint_later_validation_window():
    recovery = VerifiedMapRecovery()
    recovery.update(*samples(1.))
    outcomes = [recovery.update(*samples(1.2+.2*i, map_id=7, offset=5.)) for i in range(6)]
    assert all(not row["tracked"] and row["T_W_I"] is None for row in outcomes[:-1])
    assert outcomes[-1]["state"] == "VERIFIED_MAP_RECOVERY_RESUME"
    assert outcomes[-1]["map_reset_id"] == 0
    np.testing.assert_allclose(outcomes[-1]["T_W_I"], pose(.22), atol=1e-12)
    evidence = outcomes[-1]["recovery_evidence"]
    assert max(evidence["calibration_timestamps_s"]) < min(evidence["validation_timestamps_s"])
    assert recovery.recoveries == 1


def test_validation_failure_does_not_retune_candidate_on_validation_data():
    recovery = VerifiedMapRecovery()
    recovery.update(*samples(1.))
    for i in range(3):
        recovery.update(*samples(1.2+.2*i, 9, 3.))
    result = recovery.update(*samples(1.8, 9, 3.4))
    assert result["state"] == "RECOVERY_VALIDATION_REJECTED_STOP"
    assert not result["tracked"]
    assert recovery.recoveries == 0
    assert recovery.candidate is None


def test_inconsistent_calibration_rejected():
    recovery = VerifiedMapRecovery()
    recovery.update(*samples(1.))
    for i, offset in enumerate((2., 2.5, 2.)):
        result = recovery.update(*samples(1.2+.2*i, 1, offset))
    assert result["state"] == "RECOVERY_INCONSISTENT_CALIBRATION_STOP"


def test_distinct_observation_spacing_is_required():
    recovery = VerifiedMapRecovery()
    recovery.update(*samples(1.))
    for i in range(8):
        result = recovery.update(*samples(1.01+i*.01, 3, 4.))
    assert not result["tracked"]
    assert len(recovery.calibration) == 1


@pytest.mark.parametrize("failure", ["future", "stale", "untracked", "few_matches"])
def test_unhealthy_reference_never_returns_stereo_as_verified(failure):
    recovery = VerifiedMapRecovery()
    recovery.update(*samples(1.))
    stereo, reference = samples(1.2, 3, 4.)
    if failure == "future": reference["timestamp_s"] += .001
    if failure == "stale": reference["timestamp_s"] -= .02
    if failure == "untracked": reference["tracked"] = False
    if failure == "few_matches": reference["effective_points"] = 1
    result = recovery.update(stereo, reference)
    assert result["state"] == "REFERENCE_UNHEALTHY_STOP"
    assert result["T_W_I"] is None


def test_reference_reset_permanently_latches_stop():
    recovery = VerifiedMapRecovery()
    recovery.update(*samples(1.))
    stereo, reference = samples(1.2)
    reference["map_reset_id"] = 1
    assert not recovery.update(stereo, reference)["tracked"]
    assert recovery.update(*samples(1.4))["state"] == "REFERENCE_RESET_STOP"


def test_same_id_pose_jump_requires_reverification():
    recovery = VerifiedMapRecovery()
    recovery.update(*samples(1.))
    result = recovery.update(*samples(1.2, offset=4.))
    assert result["state"] == "CONTINUITY_WATCHDOG_STOP"
    assert result["T_W_I"] is None


def test_repeated_or_invalid_times_and_nonrigid_poses_rejected():
    recovery = VerifiedMapRecovery()
    recovery.update(*samples(1.))
    with pytest.raises(ValueError): recovery.update(*samples(1.))
    stereo, reference = samples(1.2)
    stereo["T_W_I"][0][0] = 2.
    with pytest.raises(ValueError): recovery.update(stereo, reference)


def test_lost_stereo_has_no_lio_pose_fallback():
    recovery = VerifiedMapRecovery()
    recovery.update(*samples(1.))
    stereo, reference = samples(1.2)
    stereo["tracked"], stereo["T_W_I"] = False, None
    result = recovery.update(stereo, reference)
    assert result["state"] == "STEREO_UNTRACKED_STOP"
    assert result["T_W_I"] is None


def test_rotation_averaging_keeps_rigid_unit_scale_and_rejects_large_error():
    average = mean_transform([pose(yaw=-.01), pose(yaw=.01), pose()])
    np.testing.assert_allclose(average, np.eye(4), atol=1e-12)
    translation, rotation = pose_residual(pose(), pose(1., .3))
    assert translation == pytest.approx(1.)
    assert rotation == pytest.approx(.3)


@pytest.mark.parametrize("key,value", [("calibration_samples", 1), ("validation_samples", 2),
    ("minimum_sample_spacing_s", 0.), ("maximum_pair_skew_s", float("nan")),
    ("minimum_effective_lidar_points", True)])
def test_invalid_recovery_configuration_rejected(key, value):
    with pytest.raises(ValueError): replace(RecoveryConfig(), **{key: value})


def native_row(features=400, tracked=False):
    return {"schema": "bhl-orb-native-frame-v1", "timestamp_s": 1., "tracking_state": 2 if tracked else 1,
            "tracked": tracked, "map_id": 0, "T_W_C": np.eye(4).tolist() if tracked else None,
            "compute_seconds": .02, "diagnostics": {"schema": "bhl-orb-frame-diagnostics-v1",
            "features_total": features, "features_left": features, "features_right": features,
            "positive_stereo_depth_matches": 100, "initialization_feature_threshold_exclusive": 500,
            "initialization_reason": "NOT_IN_INITIALIZATION" if tracked else "FEATURE_COUNT_NOT_ABOVE_500",
            "read_only_native_frame": True}}


def test_actual_diagnostic_gate_includes_exactly_500_features():
    assert validate_response(native_row(500), 1.)["diagnostics"]["features_total"] == 500
    with pytest.raises(ValueError): validate_response(native_row(501), 1.)


@pytest.mark.parametrize("key,value", [("features_total", -1), ("features_left", True),
    ("positive_stereo_depth_matches", 401), ("initialization_feature_threshold_exclusive", 100),
    ("read_only_native_frame", False), ("initialization_reason", "invented")])
def test_invalid_or_altered_native_diagnostics_rejected(key, value):
    row = native_row()
    row["diagnostics"][key] = value
    with pytest.raises(ValueError): validate_response(row, 1.)
