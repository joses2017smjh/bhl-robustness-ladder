"""Causality, held asynchronous packets, and scientific gate boundaries."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bhl_robust.research.navigation_sensor_controls import (
    ARMS, AvailableImu, condition_spec, exposure_image, measured_brake_comparison, tracking_events,
)


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/bench/native_sensor_controls_campaign.py"
    spec = importlib.util.spec_from_file_location("test_native_sensor_controls_campaign", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_positive_clock_error_delays_availability_without_future_samples():
    stream = AvailableImu(clock_offset_s=.08)
    stream.append(1., [1, 2, 3], [0, 0, 9.81])
    stream.append(1.005, [4, 5, 6], [0, 0, 9.81])
    assert stream.consume(1.079)["timestamp_s"].size == 0
    a = stream.consume(1.081)
    np.testing.assert_allclose(a["timestamp_s"], [1.08])
    np.testing.assert_allclose(a["original_timestamp_s"], [1.])
    np.testing.assert_allclose(stream.consume(1.085)["timestamp_s"], [1.085])
    assert stream.consume(2.)["timestamp_s"].size == 0


def test_imu_bias_does_not_mutate_original_measurements():
    original = np.zeros(3)
    stream = AvailableImu(gyro_bias=(0, 0, .05), bias_start_s=5.)
    stream.append(4.995, original, [0, 0, 9.81])
    stream.append(5., original, [0, 0, 9.81])
    np.testing.assert_array_equal(original, np.zeros(3))
    np.testing.assert_allclose(stream.consume(6.)["gyro_rad_s"][:, 2], [0, .05])


@pytest.mark.parametrize("offset", [-.1, float("nan"), float("inf")])
def test_future_imu_or_nonfinite_offset_rejected(offset):
    with pytest.raises(ValueError):
        AvailableImu(clock_offset_s=offset)


def test_exposure_integrates_acquired_past_renders_with_known_gain():
    black = np.zeros((2, 3, 3), dtype=np.uint8)
    white = np.full_like(black, 200)
    result = exposure_image([black, white], [.96, 1.], acquisition_s=1., exposure_s=.04, gain=.5)
    assert np.all(result == 50)
    assert np.all(white == 200)


@pytest.mark.parametrize("stamps", [[1., 1.02], [.94, 1.], [1., .98], [1., 1.]])
def test_exposure_rejects_future_outside_window_or_nonincreasing_samples(stamps):
    sample = np.zeros((1, 1, 3), np.uint8)
    with pytest.raises(ValueError):
        exposure_image([sample, sample], stamps, acquisition_s=1., exposure_s=.04)


def test_blackout_is_acquired_sensor_transform_without_future_data():
    sample = np.full((2, 3, 3), 255, np.uint8)
    result = exposure_image([sample], [1.], acquisition_s=1., blackout=True)
    assert not result.any()
    assert sample.all()


def test_unknown_space_not_counted_as_false_clear():
    obstacle = np.tile([.3, 0, .3], (24, 1))
    empty = np.empty((0, 3))
    result = measured_brake_comparison(obstacle, empty)
    assert result["measured_obstacle_present"]
    assert result["unknown_stop"]
    assert not result["false_clear"]


def test_false_clear_uses_matched_original_surface_and_declared_brake_box():
    original = np.tile([.3, 0, .3], (24, 1))
    faulted = np.tile([1., 0, .3], (24, 1))
    result = measured_brake_comparison(original, faulted)
    assert result["false_clear"]
    assert not result["unknown_stop"]
    assert not measured_brake_comparison(faulted, faulted)["false_clear"]


def test_tracking_duration_keeps_unrecovered_loss_censored():
    records = [{"timestamp_s": float(t), "tracked": ok, "native_ORB_map_id": m}
               for t, ok, m in [(1, False, None), (2, True, 0), (3, False, 0), (5, True, 1), (6, False, 1)]]
    result = tracking_events(records, 10.)
    assert result["tracking_loss_events"] == 2
    assert result["tracking_recoveries"] == 1
    assert result["completed_recovery_durations_s"] == [2.]
    assert result["right_censored_loss_duration_s"] == 4.
    assert result["native_map_changes"] == 1


def test_protocol_is_bounded_and_smoke_development_seeds_disjoint():
    mod = module()
    p = mod.declared_protocol()
    assert len(p["smoke"]["cells"]) == 14
    assert len(p["development"]["cells"]) == 200
    assert {c["seed"] for c in p["smoke"]["cells"]}.isdisjoint(c["seed"] for c in p["development"]["cells"])
    assert set(c["arm"] for c in p["development"]["cells"]) == set(ARMS)
    assert len({tuple(sorted(c.items())) for c in p["development"]["cells"]}) == 200
    assert p["conditions"]["causal_40ms_exposure"]["exposure_samples"] == 3


def test_negative_inertial_initialization_blocks_only_that_arm():
    mod = module()
    rows = [{"method": arm, "condition": "nominal", "native_tracked_frames": 10,
             "full_horizon": True, "fell": False, "collision": False, "nonfinite": False,
             "native_imu_initialized_frames": 0} for arm in ARMS]
    rows += [{"method": "verified_lio_recovery", "condition": "visual_blackout", "native_tracked_frames": 0}]
    ready = mod.nominal_arm_readiness(rows)
    assert not ready["stereo_inertial"]
    assert all(ready[a] for a in ARMS if a != "stereo_inertial")


def test_smoke_cannot_be_reused_after_source_change_or_missing_cell():
    mod = module()
    smoke = {"schema": mod.SCHEMA, "phase": "smoke", "status": "PASS", "execution_gate": True,
             "source_sha256": {"a": "b"}, "input_sha256": {"c": "d"},
             "completed_cells": mod.declared_protocol()["smoke"]["cells"],
             "arm_development_ready": {arm: arm != "stereo_inertial" for arm in ARMS}}
    assert not mod.validate_smoke(smoke, sources={"a": "b"}, inputs={"c": "d"})["stereo_inertial"]
    with pytest.raises(ValueError):
        mod.validate_smoke(smoke, sources={"a": "new"}, inputs={"c": "d"})
    smoke["completed_cells"] = smoke["completed_cells"][:-1]
    with pytest.raises(ValueError):
        mod.validate_smoke(smoke, sources={"a": "b"}, inputs={"c": "d"})


def test_declared_initialization_excitation_bounded_and_time_limited():
    mod = module()
    for t in (0, 2.99, 9, 10, 40):
        assert np.array_equal(mod.initialization_command(t), np.zeros(3))
    samples = np.asarray([mod.initialization_command(t) for t in np.linspace(3, 9, 100)])
    assert np.max(np.abs(samples[:, 0])) <= .1
    assert np.max(np.abs(samples[:, 2])) <= .3
    assert np.all(samples[:, 1] == 0)


def test_estimator_clocks_do_not_resend_identical_lidar_scan():
    mod = module()
    obj = mod.MatchedNativeClient({"arm": "lidar_inertial", "condition": "nominal"}, np.eye(4), Path("unused"))
    calls = []
    def track(scan, imu):
        calls.append((scan["frame_timestamp_s"], imu["timestamp_s"].copy()))
        return {"timestamp_s": scan["frame_timestamp_s"], "tracked": True, "T_W_I": np.eye(4).tolist(),
                "state": "TRACKING", "map_reset_id": 0, "compute_seconds": .001}
    obj.lio = SimpleNamespace(track=track)
    def packet(stamp):
        return {"frame_timestamp_s": 1., "event_timestamp_s": stamp, "stereo": {"timestamp_s": stamp},
                "additional_sensor_compute_seconds": .01}
    imu = {"timestamp_s": np.array([.9, 1.]), "gyro_rad_s": np.zeros((2, 3)), "specific_force_m_s2": np.zeros((2, 3))}
    obj.track(packet(1.), imu)
    empty = {"timestamp_s": np.empty(0), "gyro_rad_s": np.empty((0, 3)), "specific_force_m_s2": np.empty((0, 3))}
    second = obj.track(packet(1.1), empty)
    assert len(calls) == 1
    assert second["timestamp_s"] == 1.
    assert second["new_native_lio_frame"] is False


def test_condition_lookup_does_not_mutate_frozen_specification():
    spec = condition_spec("nominal")
    spec["camera_hz"] = 100
    assert condition_spec("nominal")["camera_hz"] == 5


@pytest.mark.parametrize("rate", [2.5, 5., 10.])
@pytest.mark.parametrize("inertial", [True, False])
def test_acquisition_fps_changes_no_native_feature_threshold(tmp_path, rate, inertial):
    mod = module()
    path = tmp_path / "stereo.yaml"
    mod.write_controlled_settings(path, camera_hz=rate, t_imu_camera=np.eye(4) if inertial else None)
    text = path.read_text()
    assert f"Camera.fps: {rate:g}\n" in text
    assert "ORBextractor.nFeatures: 1200\n" in text
    assert "ORBextractor.iniThFAST: 20\n" in text
    assert "ORBextractor.minThFAST: 7\n" in text


def test_collector_rejects_missing_trials_instead_of_fabricating_zeros(tmp_path):
    mod = module()
    smoke = {"schema": mod.SCHEMA, "phase": "smoke", "status": "PASS", "execution_gate": True,
             "source_sha256": {"a": "b"}, "input_sha256": {"c": "d"},
             "completed_cells": mod.declared_protocol()["smoke"]["cells"],
             "arm_development_ready": {arm: True for arm in ARMS}}
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(json.dumps(smoke))
    with pytest.raises(ValueError, match="incomplete development collection"):
        mod.collect_campaign(tmp_path, smoke_path, tmp_path / "collected.json")
    assert not (tmp_path / "collected.json").exists()


def test_collector_retains_explicit_all_negative_arm_gates_as_unrun(tmp_path):
    mod = module()
    smoke = {"schema": mod.SCHEMA, "phase": "smoke", "status": "PASS", "execution_gate": True,
             "source_sha256": {"a": "b"}, "input_sha256": {"c": "d"},
             "completed_cells": mod.declared_protocol()["smoke"]["cells"],
             "arm_development_ready": {arm: False for arm in ARMS}}
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text(json.dumps(smoke))
    cells = mod.declared_protocol()["development"]["cells"]
    (tmp_path / "campaign_result.json").write_text(json.dumps(dict(smoke, phase="development", episodes=0,
        completed_cells=[], episode_files=[], problems=[], unrun_cells=[dict(c, reason="UNRUN_AFTER_NEGATIVE_NOMINAL_ARM_SMOKE") for c in cells])))
    result = mod.collect_campaign(tmp_path, smoke_path, tmp_path / "collected.json")
    assert result["measured_cells"] == 0
    assert result["unrun_after_nominal_gate"] == 200
    assert result["by_arm_condition"]["stereo/nominal"]["mean_episode_full_pipeline_p95_ms"] is None
