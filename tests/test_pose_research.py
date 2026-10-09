"""Adversarial geometry/time/provenance checks for sensor-route replay."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import zlib

import numpy as np
import pytest

from bhl_robust.research.pose_metrics import (
    EstimatedPoseGate, PosePacket, Trajectory, align_metric_poses,
    associate_timestamps, estimated_navigation_observation, score_trajectories,
    transform,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bench/pose_research.py"
spec = importlib.util.spec_from_file_location("pose_research_test_driver", SCRIPT)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def pose(x=0., y=0., z=0., yaw=0.):
    t = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    t[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    t[:3, 3] = [x, y, z]
    return t


def trajectory(positions, tracked=None):
    return Trajectory(np.arange(len(positions), dtype=float), np.stack([pose(*p) for p in positions]), tracked)


def png(path, width=8, height=6):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress((b"\0" + bytes([42, 16, 12]) * width) * height)) + chunk(b"IEND", b""))


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "inference").mkdir(parents=True)
    (root / "evaluator").mkdir()
    frames = []
    for i in range(3):
        stamp = i * .1
        for k in ("left", "right"): png(root / "inference" / f"{k}{i}.png")
        np.savez(root / "inference" / f"scan{i}.npz", points_xyz_m=[[.1, 0, 1], [.2, 0, 1.2]],
                 point_time_s=[stamp, stamp + .05], ring_index=[0, 1], intensity=[1., 1.])
        ts = stamp + np.arange(21) * .005
        np.savez(root / "inference" / f"imu{i}.npz", timestamp_s=ts, gyro_rad_s=np.zeros((len(ts), 3)),
                 specific_force_m_s2=np.tile([0., 0., 9.81], (len(ts), 1)))
        frames.append({"frame_id": str(i), "timestamp_s": stamp, "left": f"inference/left{i}.png",
                       "right": f"inference/right{i}.png", "lidar": f"inference/scan{i}.npz",
                       "imu": f"inference/imu{i}.npz", "truth": f"evaluator/truth{i}.npz"})
    manifest = {"schema": "bhl-sensor-replay-v1", "clock_domain": "simulation_time", "lidar_dimension": 3,
                "calibration": {"schema": "bhl-rectified-stereo-v1", "image_width": 8, "image_height": 6,
                                "fx_px": 20., "fy_px": 20., "cx_left_px": 4., "cx_right_px": 4., "cy_px": 3.,
                                "baseline_m": .1, "rectified": True, "T_B_L": np.eye(4).tolist()},
                "sequences": [{"id": "seq0", "scene_id": "room0", "split": "dev", "origin": "simulation", "frames": frames}]}
    manifest["file_sha256"] = {str(p.relative_to(root)): driver.sha256(p) for p in (root / "inference").iterdir()}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, manifest


def save_manifest(root, manifest):
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root / "manifest.json"


@pytest.mark.parametrize("bad", [np.eye(3), np.full((4, 4), np.nan), np.diag([2., 1., 1., 1.]), np.diag([-1., 1., 1., 1.])])
def test_invalid_pose_rejected(bad):
    with pytest.raises(ValueError): transform(bad)


@pytest.mark.parametrize("times", [[0, 0], [1, 0], [0, float("nan")]])
def test_invalid_time_order_rejected(times):
    with pytest.raises(ValueError): Trajectory(times, [np.eye(4), np.eye(4)])


def test_associations_are_unique_and_nearest():
    ei, gi = associate_timestamps([.001, .004, .051, .099], [0., .05, .1], max_difference_s=.01)
    assert ei.tolist() == [0, 2, 3]
    assert gi.tolist() == [0, 1, 2]


def test_stereo_rigid_alignment_preserves_metric_scale():
    g = trajectory([(0, 0), (1, 0), (1, 1), (0, 1)])
    e = Trajectory(g.timestamp_s, pose(3, 4, yaw=.4)[None] @ g.T_W_S)
    result = score_trajectories(e, g, alignment="se3")
    assert result["scale"] == 1
    assert result["ate_translation_m"]["rmse"] < 1e-12
    assert result["rpe_translation_m"]["rmse"] < 1e-12
    scaled = g.T_W_S.copy()
    scaled[:, :3, 3] *= 2
    bad = score_trajectories(Trajectory(g.timestamp_s, scaled), g, alignment="se3")
    assert bad["ate_translation_m"]["rmse"] > .5
    assert bad["endpoint_drift_m_per_m"] > .3


def test_lio_alignment_cannot_hide_roll_error():
    g = trajectory([(0, 0), (1, 0), (1, 1), (0, 1)])
    roll = np.eye(4)
    a = .25
    roll[:3, :3] = [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    e = Trajectory(g.timestamp_s, roll[None] @ g.T_W_S)
    result = score_trajectories(e, g, alignment="gravity_yaw_translation")
    assert result["ate_rotation_deg"]["mean"] > 10
    assert result["ate_translation_m"]["rmse"] > .1


def test_tracking_failures_use_recorded_intervals():
    g = trajectory([(i, 0) for i in range(5)])
    e = Trajectory(g.timestamp_s, g.T_W_S, [True, False, False, True, False])
    result = score_trajectories(e, g, alignment="se3", end_time_s=5.)
    assert result["tracking_failure_segments"] == 2
    assert result["tracking_lost_time_s"] == 3
    assert result["rpe_interval_s"] == [3]


def test_no_motion_drift_ratio_is_unknown():
    g = trajectory([(0, 0), (0, 0)])
    result = score_trajectories(g, g, alignment="gravity_yaw_translation")
    assert result["endpoint_drift_m_per_m"] is None


def test_no_associations_does_not_score():
    g = trajectory([(0, 0), (1, 0)])
    e = Trajectory([5., 6.], g.T_W_S)
    with pytest.raises(ValueError, match="associated"):
        score_trajectories(e, g, alignment="se3")


@pytest.mark.parametrize("path", ["evaluator/truth.npz", "inference/../evaluator/truth.npz", "/tmp/a.npz"])
def test_inference_namespace_escape_rejected(tmp_path, path):
    with pytest.raises(ValueError): driver.inference_path(tmp_path, path)


def test_symlink_truth_leak_rejected(dataset):
    root, _ = dataset
    (root / "evaluator" / "private.png").write_bytes(b"secret")
    (root / "inference" / "leak.png").symlink_to(root / "evaluator" / "private.png")
    with pytest.raises(ValueError, match="symbolic"):
        driver.inference_path(root, "inference/leak.png")


def test_orb_export_preserves_capture_ns_and_has_no_truth(dataset, tmp_path):
    root, manifest = dataset
    receipt = driver.export_orb(root / "manifest.json", "seq0", tmp_path / "orb")
    assert receipt["scientific_status"] == "BLOCKED_RUNTIME"
    assert receipt["ground_truth_inputs"] == []
    assert (tmp_path / "orb/timestamps.txt").read_text().splitlines() == ["0", "100000000", "200000000"]
    assert not (tmp_path / "orb/evaluator").exists()
    assert "data: [1,0,0,0.1" in (tmp_path / "orb/stereo.yaml").read_text()
    assert (tmp_path / "orb/mav0/cam0/data/100000000.png").read_bytes() == (root / "inference/left1.png").read_bytes()
    cv2 = pytest.importorskip("cv2")
    settings = cv2.FileStorage(str(tmp_path / "orb/stereo.yaml"), cv2.FILE_STORAGE_READ)
    assert settings.getNode("Camera.type").string() == "Rectified"
    assert settings.getNode("Camera.fps").isInt()
    assert settings.getNode("Stereo.b").isReal()
    assert settings.getNode("Stereo.b").real() == .1
    settings.release()


def test_orb_different_principal_points_are_not_silently_ignored(dataset, tmp_path):
    root, manifest = dataset
    manifest["calibration"]["cx_right_px"] = 5
    with pytest.raises(ValueError, match="matching cx"):
        driver.export_orb(save_manifest(root, manifest), "seq0", tmp_path / "orb")


def test_image_resolution_mismatch_rejected(dataset):
    root, manifest = dataset
    manifest["calibration"]["image_width"] = 9
    with pytest.raises(ValueError, match="resolution"):
        driver.read_manifest(save_manifest(root, manifest))


def test_fastlio_export_has_original_point_offsets_and_si_imu(dataset, tmp_path):
    root, _ = dataset
    receipt = driver.export_fastlio(root / "manifest.json", "seq0", tmp_path / "lio")
    packets = [json.loads(p) for p in (tmp_path / "lio/packets.jsonl").read_text().splitlines()]
    clouds = [p for p in packets if p["kind"] == "pointcloud2"]
    assert receipt["scan_packets"] == 3
    assert clouds[0]["points"][1][4] == .05
    assert clouds[0]["points"][1][5] == 1
    config = json.loads((tmp_path / "lio/fastlio.yaml").read_text())
    assert config["preprocess"]["timestamp_unit"] == 0
    assert config["mapping"]["extrinsic_est_en"] is False
    assert packets[0]["specific_force_m_s2"] == [0, 0, 9.81]


@pytest.mark.parametrize("origin,accepted", [("simulation", True), ("physical", False), ("external", False)])
def test_missing_intensity_only_explicit_simulation_may_package_placeholder(dataset, tmp_path, origin, accepted):
    root, manifest = dataset
    manifest["sequences"][0]["origin"] = origin
    for f in manifest["sequences"][0]["frames"]:
        path = root / f["lidar"]
        with np.load(path) as original:
            data = {k: np.array(original[k]) for k in original.files if k != "intensity"}
        np.savez(path, **data)
        manifest["file_sha256"][f["lidar"]] = driver.sha256(path)
    manifest_path = save_manifest(root, manifest)
    if accepted:
        receipt = driver.export_fastlio(manifest_path, "seq0", tmp_path / "lio")
        assert receipt["intensity_kind"] == "unmeasured_zero_placeholder_for_ROS_field"
        packets = [json.loads(p) for p in (tmp_path / "lio/packets.jsonl").read_text().splitlines()]
        for p in packets:
            if p["kind"] == "pointcloud2":
                assert p["intensity_kind"] == receipt["intensity_kind"]
                assert all(row[3] == 0 for row in p["points"])
        # Packaging did not manufacture an intensity field in original scans.
        with np.load(root / "inference/scan0.npz") as original:
            assert "intensity" not in original.files
    else:
        with pytest.raises(ValueError, match="intensity is missing"):
            driver.export_fastlio(manifest_path, "seq0", tmp_path / "lio")


@pytest.mark.parametrize("mutation", ["two_dimensional", "no_point_times", "zero_point_offsets", "truth_array", "imu_gap"])
def test_lio_inadequate_inputs_are_blocked(dataset, tmp_path, mutation):
    root, manifest = dataset
    if mutation == "two_dimensional": manifest["lidar_dimension"] = 2
    else:
        path = root / "inference" / ("imu0.npz" if mutation == "imu_gap" else "scan0.npz")
        with np.load(path) as original: data = {k: np.array(original[k]) for k in original.files}
        if mutation == "no_point_times": del data["point_time_s"]
        elif mutation == "zero_point_offsets": data["point_time_s"][:] = 0
        elif mutation == "truth_array": data["T_W_L"] = np.eye(4)
        elif mutation == "imu_gap":
            data = {k: v[[0, -1]] for k, v in data.items()}
        np.savez(path, **data)
        manifest["file_sha256"][str(path.relative_to(root))] = driver.sha256(path)
    with pytest.raises(ValueError):
        driver.export_fastlio(save_manifest(root, manifest), "seq0", tmp_path / "lio")


def packet(**changes):
    args = dict(T_W_B=pose(7, 8, yaw=.3), capture_time_s=1., arrival_time_s=1.02,
                confidence=.8, tracking=True, clock_domain="sim", source_method="orb_slam3_stereo")
    args.update(changes)
    return PosePacket(**args)


@pytest.mark.parametrize("changes,now,reason", [
    ({"tracking": False}, 1.1, "tracking_lost"),
    ({"confidence": .2}, 1.1, "low_confidence"),
    ({"clock_domain": "hardware"}, 1.1, "clock_domain_mismatch"),
    ({"source_method": "fast_lio2_lidar_imu"}, 1.1, "method_mismatch"),
    ({}, 1.01, "packet_not_arrived"), ({}, 1.3, "stale_capture")])
def test_pose_gate_fails_to_safe_command(changes, now, reason):
    gate = EstimatedPoseGate(clock_domain="sim", source_method="orb_slam3_stereo")
    result = gate.accept(packet(**changes), now)
    assert result["reason"] == reason
    assert result["safe_command_vx_vy_wz"] == [0, 0, 0]


def test_packet_age_uses_capture_not_fast_arrival():
    gate = EstimatedPoseGate(clock_domain="sim", source_method="orb_slam3_stereo")
    assert not gate.accept(packet(arrival_time_s=1.4), 1.4)["accepted"]


def test_pose_gate_rejects_old_packet_after_newer_acceptance():
    gate = EstimatedPoseGate(clock_domain="sim", source_method="orb_slam3_stereo")
    assert gate.accept(packet(capture_time_s=1.02), 1.1)["accepted"]
    assert gate.accept(packet(), 1.1)["reason"] == "out_of_order_capture"


def test_new_tracking_loss_invalidates_older_confident_pose():
    gate = EstimatedPoseGate(clock_domain="sim", source_method="orb_slam3_stereo")
    assert gate.accept(packet(capture_time_s=1.02, tracking=False), 1.1)["reason"] == "tracking_lost"
    assert gate.accept(packet(), 1.1)["reason"] == "out_of_order_capture"


def test_navigation_builder_receives_only_estimated_pose():
    seen = {}
    def builder(keys, ranges, emap, x, y, yaw, goal, prev, **kw):
        seen.update(x=x, y=y, yaw=yaw)
        return {"fixture": np.array([x, y, yaw])}
    gate = EstimatedPoseGate(clock_domain="sim", source_method="orb_slam3_stereo")
    result = estimated_navigation_observation(gate=gate, packet=packet(), now_s=1.1,
        ranges_m=np.full(108, 3.), scan_capture_time_s=1., goal_xy=[10, 10], previous_action=[0, 0],
        observation_keys=["goal"], occupancy_map=object(), builder=builder)
    assert result["gate"]["accepted"]
    assert seen["x"] == 7 and seen["y"] == 8 and seen["yaw"] == pytest.approx(.3)


def test_unknown_scan_cannot_become_free_space():
    gate = EstimatedPoseGate(clock_domain="sim", source_method="orb_slam3_stereo")
    result = estimated_navigation_observation(gate=gate, packet=packet(), now_s=1.1,
        ranges_m=np.full(108, np.nan), scan_capture_time_s=1., goal_xy=[10, 10], previous_action=[0, 0],
        observation_keys=["goal"], occupancy_map=object(), builder=lambda *a, **k: pytest.fail("must not infer policy"))
    assert result["observation"] is None
    assert result["gate"]["safe_command_vx_vy_wz"] == [0, 0, 0]


def test_intake_pilot_does_not_claim_slam_or_closedloop(dataset, tmp_path):
    root, _ = dataset
    result = driver.intake(root / "manifest.json", tmp_path / "intake")
    assert result["status"] == "PASS"
    assert result["scientific_status"] == "BLOCKED_RUNTIME"
    assert result["estimated_pose_outputs"] == result["closed_loop_episodes"] == 0
    assert result["sequences"][0]["slam_protocol_suitability"] == "PILOT_ONLY_SHORT_SEQUENCE"


def test_standalone_export_rejects_missing_checksum(dataset, tmp_path):
    root, manifest = dataset
    del manifest["file_sha256"]["inference/scan0.npz"]
    with pytest.raises(ValueError, match="checksum"):
        driver.export_orb(save_manifest(root, manifest), "seq0", tmp_path / "orb")


def test_standalone_export_rejects_tampered_imu(dataset, tmp_path):
    root, _ = dataset
    with (root / "inference/imu0.npz").open("ab") as f: f.write(b"tampered")
    with pytest.raises(ValueError, match="hash differs"):
        driver.export_fastlio(root / "manifest.json", "seq0", tmp_path / "lio")


def test_fastlio_extrinsic_is_in_imu_frame(dataset, tmp_path):
    root, manifest = dataset
    manifest["calibration"]["T_B_I"] = pose(.2, 0, yaw=.1).tolist()
    receipt = driver.export_fastlio(save_manifest(root, manifest), "seq0", tmp_path / "lio")
    expected = np.linalg.inv(pose(.2, 0, yaw=.1))
    assert np.allclose(receipt["T_imu_lidar"], expected)


def test_actual_capture_packet_helpers_are_replay_compatible(dataset, tmp_path, monkeypatch):
    capture_spec = importlib.util.spec_from_file_location("pose_actual_capture_contract", SCRIPT.parent / "sensor_capture.py")
    capture = importlib.util.module_from_spec(capture_spec)
    capture_spec.loader.exec_module(capture)
    # Geometry stub only: this is a packet-format test, never rendered data.
    monkeypatch.setattr(capture, "ray", lambda *a, **k: 2.)
    root, manifest = dataset
    for i, f in enumerate(manifest["sequences"][0]["frames"]):
        stamp = i / 15.
        f["timestamp_s"] = stamp
        np.savez(root / f["lidar"], **capture.lidar_scan(None, None, stamp))
        np.savez(root / f["imu"], **capture.imu_packet(stamp, 1 / 15.))
        for key in ("lidar", "imu"):
            manifest["file_sha256"][f[key]] = driver.sha256(root / f[key])
    receipt = driver.export_fastlio(save_manifest(root, manifest), "seq0", tmp_path / "lio")
    assert receipt["scan_packets"] == 3
    assert receipt["max_imu_gap_s"] <= .005 + 1e-8
    assert receipt["intensity_kind"] == "unmeasured_zero_placeholder_for_ROS_field"


def test_evaluation_rejects_unpinned_oracle_substitute(tmp_path):
    g = trajectory([(0, 0), (1, 0), (1, 1)])
    data = {"schema": "bhl-pose-trajectory-v1", "clock_domain": "sim", "sequence": "a", "sensor_frame": "camera",
            "frames": [{"timestamp_s": float(s), "T_W_S": t.tolist(), "tracked": True} for s, t in zip(g.timestamp_s, g.T_W_S)]}
    e, t, r = [tmp_path / name for name in ("estimate.json", "truth.json", "receipt.json")]
    e.write_text(json.dumps(data)); t.write_text(json.dumps(data))
    r.write_text(json.dumps({"schema": "bhl-pose-estimator-run-v1", "method": "ground_truth_pose"}))
    with pytest.raises(ValueError, match="native estimator"):
        driver.evaluate(e, t, r, tmp_path / "score.json")
