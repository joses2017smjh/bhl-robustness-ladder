import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/bench/euroc_native_evaluate.py"
SPEC = importlib.util.spec_from_file_location("euroc_evaluate", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
ORIGINAL_PATH = sys.path.copy()
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path[:] = ORIGINAL_PATH

ORIGIN = 1403715273262142976
EURO_ROOT = Path("/nfs/hpc/share/sanchej7/humanoid-research-20261011/euroc-actual-20261012")


def rot(axis, degrees):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.radians(degrees) * np.asarray(axis, dtype=float)).as_matrix()


def pose(r=np.eye(3), t=(0, 0, 0)):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, t
    return m


def wxyz(matrix):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return [w, x, y, z]


def body_trajectory(n, period_ns=50_000_000, start_ns=ORIGIN):
    """Smooth IMU poses: 1 m circle at 0.3 rad/s with small roll/pitch wobble."""
    ns = start_ns + period_ns * np.arange(n, dtype=np.int64)
    t = (ns - ORIGIN) * 1e-9
    poses = []
    for s in t:
        r = rot([0, 0, 1], np.degrees(0.3 * s)) @ rot([1, 0, 0], 4 * np.sin(0.7 * s)) @ rot([0, 1, 0], 3 * np.cos(0.4 * s))
        poses.append(pose(r, [np.cos(0.3 * s), np.sin(0.3 * s), 1.0 + 0.2 * np.sin(0.5 * s)]))
    return ns, np.stack(poses)


def write_pose_csv(path, ns, poses, header="#timestamp [ns],p_x,p_y,p_z,q_w,q_x,q_y,q_z\n"):
    with Path(path).open("w") as stream:
        stream.write(header)
        for stamp, p in zip(ns, poses):
            q = wxyz(p)
            stream.write(f"{int(stamp)},{p[0, 3]:.9f},{p[1, 3]:.9f},{p[2, 3]:.9f},"
                         f"{q[0]:.12f},{q[1]:.12f},{q[2]:.12f},{q[3]:.12f}\n")


def sensor_yaml(path, matrix):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = ", ".join(f"{v:.17g}" for v in np.asarray(matrix).reshape(-1))
    Path(path).write_text(f"T_BS:\n  cols: 4\n  rows: 4\n  data: [{data}]\n")


def test_git_blob_sha1_matches_git_hash_object(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_bytes(b"hello\n")
    assert MODULE.git_blob_sha1(path) == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_reference_pin_refuses_mismatch_and_keeps_integer_clock(tmp_path):
    ns, poses = body_trajectory(3)
    path = tmp_path / "gt.csv"
    write_pose_csv(path, ns, poses)
    with pytest.raises(ValueError, match="refusing"):
        MODULE.load_reference(path, "0" * 40, "original")
    reference = MODULE.load_reference(path, MODULE.git_blob_sha1(path), "original")
    assert reference["timestamps_ns"].dtype == np.int64
    assert reference["timestamps_ns"][0] == ORIGIN
    np.testing.assert_allclose(reference["T_W_I"], poses, atol=1e-8)


def test_quaternion_order_and_norm_tolerance():
    q = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    np.testing.assert_allclose(MODULE.quaternion_matrix(q) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    six_decimal = np.array([0.060514, -0.828459, -0.058956, -0.553641])
    matrix = MODULE.quaternion_matrix(six_decimal)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    with pytest.raises(ValueError, match="unit norm"):
        MODULE.quaternion_matrix([1.0002, 0, 0, 0])


def test_integer_association_allows_only_float64_ulp_or_declared_window():
    camera = ORIGIN + 50_000_000 * np.arange(10, dtype=np.int64)
    recomputed = camera.copy()
    recomputed[[2, 5]] -= 256
    ei, gi, offset = MODULE.associate_ns(camera, recomputed, 256)
    assert ei.tolist() == list(range(10)) and sorted(set(offset.tolist())) == [-256, 0]
    exact, _, _ = MODULE.associate_ns(camera, recomputed, 0)
    assert len(exact) == 8
    original = ORIGIN + 1_040_000_000 + 5_000_000 * np.arange(200, dtype=np.int64)
    ei, gi, offset = MODULE.associate_ns(camera + 1_000, original, MODULE.ORIGINAL_TOLERANCE_NS)
    assert ei.tolist() == []
    later = ORIGIN + 50_000_000 * np.arange(21, 41, dtype=np.int64) + 1_000
    ei, gi, offset = MODULE.associate_ns(later, original, MODULE.ORIGINAL_TOLERANCE_NS)
    assert len(ei) == 20 and np.all(offset == -1_000)
    report = MODULE.association_report(camera, {"timestamps_ns": recomputed}, 256, ORIGIN)["report"]
    assert report["exact_ns_matches"] == 8 and report["nonzero_offset_matches"] == 2
    assert report["pose_metrics_associate_timestamps_identical"]


def test_estimate_converted_to_imu_frame_and_rectified_extrinsic_rechecked(tmp_path):
    t_b_c = pose(rot([0.3, -0.2, 0.9] / np.linalg.norm([0.3, -0.2, 0.9]), 92), [-0.02, -0.06, 0.01])
    r_rect = rot([1, 2, 3] / np.linalg.norm([1, 2, 3]), 0.6)
    sensor_yaml(tmp_path / "mav0/cam0/sensor.yaml", t_b_c)
    sensor_yaml(tmp_path / "mav0/imu0/sensor.yaml", np.eye(4))
    expected = t_b_c @ pose(r_rect.T)
    rectification = {"R_left_rectification": r_rect.tolist(), "T_imu_rectified_left_camera": expected.tolist()}
    t_i_c, difference = MODULE.independent_t_i_c(tmp_path, rectification)
    assert difference < 1e-12
    _, body = body_trajectory(5)
    camera = body @ t_i_c
    np.testing.assert_allclose(MODULE.calibrated_body_pose(camera[3], t_i_c), body[3], atol=1e-12)
    rectification["T_imu_rectified_left_camera"] = (expected @ pose(rot([0, 0, 1], 0.01))).tolist()
    with pytest.raises(ValueError, match="differs"):
        MODULE.independent_t_i_c(tmp_path, rectification)


def frame(index, tracked=True, map_id=0, initialized=None):
    return {"index": index, "tracked": tracked, "map_id": map_id if tracked else None,
            "imu_initialized": initialized}


def test_epochs_split_on_map_change_initialization_and_tracking_gap():
    frames = ([frame(0, False, initialized=False)] + [frame(i, initialized=False) for i in (1, 2)]
              + [frame(i, initialized=True) for i in (3, 4)] + [frame(5, False, initialized=True)]
              + [frame(6, initialized=True), frame(7, map_id=1, initialized=True)])
    epochs = MODULE.epochs(frames)
    assert [e["frames"] for e in epochs] == [[1, 2], [3, 4], [6], [7]]
    assert [e["start_reason"] for e in epochs] == ["first_tracked_frame", "imu_initialization_changed",
                                                   "after_untracked_frames", "map_id_changed"]


def straight_line(steps_m, steps_deg):
    poses, current = [pose()], pose()
    for d, theta in zip(steps_m, steps_deg):
        current = current @ pose(rot([0, 0, 1], theta), [d, 0, 0])
        poses.append(current)
    return np.stack(poses)


def test_rotation_clause_mirrors_translation_clause_but_literal_rule_fires_on_fast_turns():
    steps_m = np.full(60, 0.016)
    steps_deg = np.full(60, 0.7)
    steps_deg[[10, 11, 12]] = 2.2          # genuine fast rotation burst
    steps_m[30] = 0.3                      # translation jump
    steps_deg[45] = 6.0                    # rotation-only jump
    thresholds, primary, literal, _, _ = MODULE.discontinuities(straight_line(steps_m, steps_deg))
    assert thresholds["translation_threshold_m"] == pytest.approx(0.08)
    assert thresholds["rotation_threshold_deg"] == pytest.approx(3.5)
    assert primary == [31, 46]
    assert literal == [11, 12, 13, 31, 46]


def test_translation_floor_applies_when_median_motion_is_slow():
    slow = np.full(30, 0.002)
    slow[20] = 0.06
    _, primary, _, _, _ = MODULE.discontinuities(straight_line(slow, np.zeros(30)))
    assert primary == [21]
    fast = np.full(30, 0.02)
    fast[20] = 0.06
    thresholds, primary, _, _, _ = MODULE.discontinuities(straight_line(fast, np.zeros(30)))
    assert thresholds["translation_threshold_m"] == pytest.approx(0.1) and primary == []


def test_segment_arm_labels_pre_initialization_and_records_jump():
    _, body = body_trajectory(80)
    frames = [frame(i, tracked=i >= 10, initialized=i >= 30) for i in range(80)]
    imu = {}
    for i in range(10, 80):
        world = pose(rot([0, 0, 1], 20)) if i < 30 else pose(rot([0, 0, 1], 50), [1, 2, 0])
        if i >= 55:
            world = world @ pose(t=[0.4, 0, 0])
        imu[i] = world @ body[i]
    segments, epochs, literal = MODULE.segment_arm({"mode": "stereo_inertial", "frames": frames}, imu)
    assert [(s["label"], s["frames"][0], s["frames"][-1]) for s in segments] == [
        ("pre_imu_initialization", 10, 29), ("primary", 30, 54), ("primary", 55, 79)]
    assert segments[2]["boundary"]["before_frame"] == 55
    assert segments[2]["boundary"]["estimate_translation_increment_m"] > 0.3
    assert [e["start_reason"] for e in epochs] == ["first_tracked_frame", "imu_initialization_changed"]
    assert MODULE.alignments_for("stereo_inertial", "pre_imu_initialization") == ["se3"]
    assert MODULE.alignments_for("stereo_inertial", "primary") == ["se3", "gravity_yaw_translation"]


def test_rpe_pairs_use_time_not_index_and_respect_gaps():
    jitter = np.array([0, 128, -128] * 10)[:30]
    ns = ORIGIN + 50_000_000 * np.arange(30, dtype=np.int64) + jitter
    pairs = MODULE.rpe_pairs(ns, 1.0)
    assert pairs.tolist() == [[i, i + 20] for i in range(10)]
    gapped = np.delete(ns, [21, 22])
    pairs = MODULE.rpe_pairs(gapped, 1.0)
    starts = pairs[:, 0].tolist()
    assert 0 in starts and 1 not in starts and 2 not in starts
    assert all(abs((gapped[j] - gapped[i]) - 1_000_000_000) <= MODULE.RPE_PAIR_TOLERANCE_NS for i, j in pairs)


def test_rpe_and_drift_recover_known_scale_error_and_exclude_short_paths():
    ns = ORIGIN + 50_000_000 * np.arange(201, dtype=np.int64)
    t = (ns - ORIGIN) * 1e-9
    reference = np.stack([pose(t=[0.5 * s, 0, 0]) for s in t])
    estimated = np.stack([pose(t=[1.02 * 0.5 * s, 0, 0]) for s in t])
    cumulative = MODULE.cumulative_path(reference)
    rpe = MODULE.rpe_interval(estimated, reference, ns, cumulative, 5.0)
    assert rpe["pairs"] == 101
    assert rpe["translation_m"]["mean"] == pytest.approx(0.05, abs=1e-9)
    assert rpe["drift_percent"]["mean"] == pytest.approx(2.0, abs=1e-6)
    still = np.stack([pose(t=[1e-4 * s, 0, 0]) for s in t])
    rpe = MODULE.rpe_interval(still, still, ns, MODULE.cumulative_path(still), 1.0)
    assert rpe["drift_pairs"] == 0 and rpe["drift_percent"] is None and rpe["drift_excluded_short_path_pairs"] == 181


def test_alignment_modes_scale_one_tilt_evidence_and_minimum_segment():
    ns, reference = body_trajectory(200)
    yaw_world = pose(rot([0, 0, 1], 37), [3, -1, 0.5])
    estimated = np.linalg.inv(yaw_world) @ reference
    for alignment in ("se3", "gravity_yaw_translation"):
        value, _ = MODULE.score(estimated, reference, ns, alignment, ORIGIN)
        assert value["status"] == "MEASURED" and value["scale"] == 1.0
        assert value["ate_m"]["max"] < 1e-9 and value["rotation_error_deg"]["max"] < 1e-6
        assert value["internal_score_trajectories_max_abs_difference"]["ate_translation_m"] < 1e-9
        assert value["internal_score_trajectories_max_abs_difference"]["rotation_deg"] < 1e-5
        assert value["alignment_yaw_deg"] == pytest.approx(37, abs=1e-6)
    tilted = np.linalg.inv(pose(rot([1, 0, 0], 3))) @ estimated
    se3, _ = MODULE.score(tilted, reference, ns, "se3", ORIGIN)
    four, _ = MODULE.score(tilted, reference, ns, "gravity_yaw_translation", ORIGIN)
    assert se3["ate_m"]["max"] < 1e-9 and se3["alignment_tilt_deg"] == pytest.approx(3, abs=1e-6)
    assert four["ate_m"]["rmse"] > 1e-3 and four["rotation_error_deg"]["mean"] > 2.9
    short, frames = MODULE.score(estimated[:19], reference[:19], ns[:19], "se3", ORIGIN)
    assert short["status"] == "UNSCORED_TOO_SHORT" and frames is None


def test_constant_body_frame_reference_rotation_survives_rigid_alignment():
    ns, reference = body_trajectory(200)
    offset = pose(rot([0.2, 0.9, 0.1] / np.linalg.norm([0.2, 0.9, 0.1]), 5.5))
    value, _ = MODULE.score(reference @ offset, reference, ns, "se3", ORIGIN)
    assert value["rotation_error_deg"]["mean"] == pytest.approx(5.5, abs=1e-6)
    assert value["mean_body_frame_rotation_residual_deg"] == pytest.approx(5.5, abs=1e-6)
    assert value["ate_m"]["max"] < 1e-9


def test_vectorized_rigid_fit_matches_pose_metrics_se3():
    ns, reference = body_trajectory(150)
    world = pose(rot([0.2, -0.5, 0.84] / np.linalg.norm([0.2, -0.5, 0.84]), 33), [0.4, -1.2, 0.3])
    noisy = world @ reference
    noisy[:, :3, 3] += np.random.default_rng(3).normal(scale=0.01, size=(150, 3))
    _, expected, mode = MODULE.align_metric_poses(noisy, reference, alignment="se3")
    assert mode == "se3"
    np.testing.assert_allclose(MODULE.rigid_fit(noisy[:, :3, 3], reference[:, :3, 3]), expected, atol=1e-12)


def test_vicon_convention_check_discriminates(tmp_path):
    t_bs = pose(rot([0.6, -0.1, 0.8] / np.linalg.norm([0.6, -0.1, 0.8]), 109), [0.069, -0.028, -0.124])
    vicon_ns, body_at_vicon = body_trajectory(700, period_ns=10_000_000, start_ns=ORIGIN - 3_000_000)
    sensor_yaml(tmp_path / "mav0/vicon0/sensor.yaml", t_bs)
    write_pose_csv(tmp_path / "mav0/vicon0/data.csv", vicon_ns, body_at_vicon @ t_bs)
    ns, body = body_trajectory(120)
    references = {"recomputed": {"timestamps_ns": ns, "T_W_I": body}}
    value = MODULE.vicon_validation(tmp_path, references, (int(ns[0]), int(ns[-1])))
    candidates = value["references"]["recomputed"]["candidates"]
    assert value["references"]["recomputed"]["interpolated_samples"] == 120
    declared = candidates["declared_T_W_V_inv_T_BS"]
    assert declared["residual_rotation_deg"]["max"] < 1e-3 and declared["residual_translation_m"]["max"] < 1e-4
    assert candidates["alternative_T_W_V_T_BS"]["mean_residual_rotation_deg"] > 100
    assert candidates["alternative_T_W_V_marker_as_body"]["residual_translation_m"]["mean"] > 0.1


def test_tum_export_is_xyzw_and_round_trips(tmp_path):
    from scipy.spatial.transform import Rotation
    ns, poses = body_trajectory(4)
    path = tmp_path / "estimate.tum"
    MODULE.write_tum(path, ns, poses)
    rows = [line.split() for line in path.read_text().splitlines()]
    assert rows[1][0] == "1403715273.312142976"
    values = np.array(rows, dtype=float)
    np.testing.assert_allclose(values[:, 1:4], poses[:, :3, 3], atol=1e-9)
    np.testing.assert_allclose(Rotation.from_quat(values[:, 4:8]).as_matrix(), poses[:, :3, :3], atol=1e-10)


def test_protocol_freeze_is_immutable_and_checked(tmp_path):
    MODULE.freeze_protocol(tmp_path)
    assert MODULE.require_frozen_protocol(tmp_path)["sha256"] == hashlib.sha256(
        (tmp_path / "protocol.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        MODULE.freeze_protocol(tmp_path)
    frozen = json.loads((tmp_path / "protocol.json").read_text())
    frozen["protocol"]["segmentation"]["rotation_floor_deg"] = 3.0
    (tmp_path / "protocol.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="differs"):
        MODULE.require_frozen_protocol(tmp_path)


def synthetic_campaign(tmp_path):
    """1200-frame replay with a known rigid world offset and one injected stereo-IMU jump."""
    dataset, replay, out = tmp_path / "dataset", tmp_path / "replay", tmp_path / "out"
    ns, body = body_trajectory(MODULE.FRAMES)
    t_b_c = pose(rot([0.01, -0.01, 1] / np.linalg.norm([0.01, -0.01, 1]), 90.9), [-0.0216, -0.0647, 0.0098])
    r_rect = rot([1, 1, 0] / np.sqrt(2), 0.5)
    t_i_c = t_b_c @ pose(r_rect.T)
    sensor_yaml(dataset / "mav0/cam0/sensor.yaml", t_b_c)
    sensor_yaml(dataset / "mav0/imu0/sensor.yaml", np.eye(4))
    t_bs = pose(rot([0, 1, 0], 100), [0.07, -0.03, -0.12])
    sensor_yaml(dataset / "mav0/vicon0/sensor.yaml", t_bs)
    vicon_ns, vicon_body = body_trajectory(6100, period_ns=10_000_000, start_ns=ORIGIN - 1_000_000)
    write_pose_csv(dataset / "mav0/vicon0/data.csv", vicon_ns, vicon_body @ t_bs)
    original_ns, original_body = body_trajectory(11800, period_ns=5_000_000, start_ns=ORIGIN + 1_040_000_000)
    (dataset / "mav0/state_groundtruth_estimate0").mkdir(parents=True)
    write_pose_csv(dataset / "mav0/state_groundtruth_estimate0/data.csv", original_ns, original_body)
    recomputed_ns = ns.copy()
    recomputed_ns[1::5] -= 256
    recomputed = tmp_path / "reference" / "V1_01_easy.csv"
    recomputed.parent.mkdir()
    write_pose_csv(recomputed, recomputed_ns, body, header="#time(ns),px,py,pz,qw,qx,qy,qz\n")
    stereo_world = pose(rot([0.3, 0.4, 0.86] / np.linalg.norm([0.3, 0.4, 0.86]), 70), [0.2, -0.4, 1.1])
    inertial_world = pose(rot([0, 0, 1], -25), [0.5, 0.1, -0.2])
    replay.mkdir()
    arms = []
    for mode in ("stereo", "stereo_inertial"):
        (replay / mode).mkdir()
        lines = []
        for i, stamp in enumerate(ns):
            tracked = mode == "stereo" or i >= 100
            initialized = mode == "stereo_inertial" and i >= 160
            if mode == "stereo":
                world = np.linalg.inv(stereo_world)
            else:
                world = np.linalg.inv(inertial_world) if initialized else np.linalg.inv(stereo_world)
                if i >= 400:
                    world = pose(t=[0.6, 0, 0]) @ world
            native = {"timestamp_s": (int(stamp) - ORIGIN) * 1e-9, "tracking_state": 2 if tracked else 1,
                      "tracked": tracked, "map_id": 0 if tracked else None,
                      "T_W_C": (world @ body[i] @ t_i_c).tolist() if tracked else None}
            if mode == "stereo_inertial":
                native["inertial"] = {"map_imu_initialized": initialized}
            lines.append(json.dumps({"original_timestamp_ns": int(stamp), "native": native}))
        (replay / mode / "frames.jsonl").write_text("\n".join(lines) + "\n")
        tracked = MODULE.FRAMES if mode == "stereo" else MODULE.FRAMES - 100
        arms.append({"mode": mode, "frames": MODULE.FRAMES, "tracked_frames": tracked,
                     "accepted_initialized_frames": MODULE.FRAMES if mode == "stereo" else MODULE.FRAMES - 160,
                     "initialization_first_sensor_time_s": 0 if mode == "stereo" else 8.0,
                     "tracking_states": {}, "observed_map_ids": [0],
                     "frames_sha256": MODULE.sha256(replay / mode / "frames.jsonl")})
    (replay / "result.json").write_text(json.dumps({
        "schema": "bhl-external-euroc-result-v1", "status": "COMPLETED", "stage": "full",
        "paired_frames_per_arm": MODULE.FRAMES, "clock_origin_ns": ORIGIN, "arms": arms}))
    (replay / "rectification.json").write_text(json.dumps({
        "R_left_rectification": r_rect.tolist(), "T_imu_rectified_left_camera": t_i_c.tolist()}))
    return dataset, replay, recomputed, out


def test_end_to_end_synthetic_campaign_scores_segments_and_refuses_reruns(tmp_path):
    dataset, replay, recomputed, out = synthetic_campaign(tmp_path)
    with pytest.raises(ValueError, match="freeze-protocol"):
        MODULE.evaluate(dataset, replay, recomputed, out)
    MODULE.freeze_protocol(out)
    with pytest.raises(ValueError, match="refusing"):
        MODULE.evaluate(dataset, replay, recomputed, out)
    assert not (out / "results.json").exists()
    blobs = dict(original_blob=MODULE.git_blob_sha1(dataset / "mav0/state_groundtruth_estimate0/data.csv"),
                 recomputed_blob=MODULE.git_blob_sha1(recomputed))
    results = MODULE.evaluate(dataset, replay, recomputed, out, **blobs)
    assert results["association"]["recomputed"]["associated_frames"] == 1200
    assert results["association"]["recomputed"]["nonzero_offset_matches"] == 240
    assert results["association"]["original"]["frames_without_reference"] == 21
    stereo, inertial = results["arms"]["stereo"], results["arms"]["stereo_inertial"]
    assert len(stereo["segments"]) == 1
    assert stereo["segments"][0]["scores"]["recomputed"]["se3"]["ate_m"]["max"] < 1e-6
    assert [(s["label"], *s["frame_range"]) for s in inertial["segments"]] == [
        ("pre_imu_initialization", 100, 159), ("primary", 160, 399), ("primary", 400, 1199)]
    for segment in inertial["segments"][1:]:
        for alignment in ("se3", "gravity_yaw_translation"):
            assert segment["scores"]["original"][alignment]["ate_m"]["max"] < 1e-6
    assert inertial["stitched"]["recomputed"]["se3"]["ate_m"]["rmse"] > 0.1
    assert inertial["coverage"]["imu_initialized_tracked_frames"] == 1040
    assert results["validation"]["internal_score_trajectories"]["agree"]
    smooth = results["arms"]["stereo_inertial"]["one_frame_step_error_post_run"]["recomputed"]
    assert smooth["pairs"] == 239 + 799 and smooth["translation_m"]["max"] < 1e-6 and smooth["steps_over_2cm"] == 0
    assert results["validation"]["front_end_stereo_depth_matches"]["whole_run"] is None
    vicon = results["validation"]["vicon"]["references"]["recomputed"]["candidates"]
    assert vicon["declared_T_W_V_inv_T_BS"]["residual_translation_m"]["max"] < 1e-4
    assert results["evo_crosscheck"]["status"] == "NOT_RUN"
    assert "Stereo vs stereo-IMU on identical timestamps" in (out / "REPORT.md").read_text()
    with pytest.raises(ValueError, match="immutable"):
        MODULE.evaluate(dataset, replay, recomputed, out, **blobs)


@pytest.mark.skipif(not EURO_ROOT.exists(), reason="EuRoC replay inputs not present on this host")
def test_on_disk_original_reference_matches_pinned_blob():
    path = EURO_ROOT / "dataset/mav0/state_groundtruth_estimate0/data.csv"
    assert MODULE.git_blob_sha1(path) == MODULE.ORIGINAL_BLOB
