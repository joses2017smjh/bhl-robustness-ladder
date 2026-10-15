#!/usr/bin/env python3
"""Separate pose-accuracy evaluator for the EuRoC V1_01_easy native replay.

Offline evaluation only: it is never imported by the estimator runner and no
reference pose reaches the estimator. Transforms T_A_B map coordinates in B
into A; timestamps are integer nanoseconds on the original EuRoC clock.

The full predeclared protocol is the PROTOCOL constant below. It was frozen to
<output>/protocol.json (``freeze-protocol``) before any score was computed, and
``evaluate`` refuses to run unless that file still equals PROTOCOL. Summary:

* Two references, each pinned by git blob SHA-1 (refuse on mismatch): the
  on-disk original state_groundtruth_estimate0/data.csv (200 Hz) and the
  OpenVINS vicon2gt recomputation ov_data/euroc_mav/V1_01_easy.csv (20 Hz on
  camera timestamps, GPL-3.0 repository, read in place and never copied here).
* Estimated IMU pose T_W_I = T_W_Crect @ inv(T_I_Crect).
* Integer-nanosecond association: recomputed reference exact or within 256 ns
  (one float64 ulp at 1.4e18 ns), original reference nearest within 2.5 ms.
* Segments: split at map-id / IMU-initialization changes and at causal pose
  discontinuities, d_k > max(0.05 m, 5 x median d) or
  theta_k > max(2 deg, 5 x median theta). The literal task rule used an
  absolute 2 deg rotation clause; ground-truth motion alone exceeds 2 deg per
  frame 20 times in this window, so the rotation clause mirrors the
  translation clause (see PROTOCOL["segmentation"]).
* Per segment: SE3 alignment (both arms) and 4-DoF gravity-yaw alignment
  (IMU-initialized stereo-IMU), scale fixed at 1; ATE, rotation error,
  RPE and drift at 1, 5 and 10 s; the stitched whole is secondary.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bhl_robust.research.native_orb import sha256  # noqa: E402
from bhl_robust.research.pose_metrics import (  # noqa: E402
    Trajectory, align_metric_poses, associate_timestamps, calibrated_body_pose,
    score_trajectories, transform)

FRAMES = 1200
ORIGINAL_BLOB = "98c4a0211931b84368793fad489175bb7b7f1505"
RECOMPUTED_BLOB = "8063da27df1039f3f70f52e5e771672b27c0026a"
OPENVINS_INTRODUCED_COMMIT = "cadf793463d728bfc5b015840a96c071f088066e"
OPENVINS_DOWNLOAD_COMMIT = "69488123ed9362dd44b6f28e7f4680abbff1442b"
RECOMPUTED_URL = ("https://raw.githubusercontent.com/rpng/open_vins/"
                  f"{OPENVINS_DOWNLOAD_COMMIT}/ov_data/euroc_mav/V1_01_easy.csv")
RECOMPUTED_TOLERANCE_NS = 256
ORIGINAL_TOLERANCE_NS = 2_500_000
RPE_INTERVALS_S = (1.0, 5.0, 10.0)
RPE_PAIR_TOLERANCE_NS = 25_000_000
MIN_SEGMENT_FRAMES = 20
DRIFT_MIN_PATH_M = 0.10
TRANSLATION_FLOOR_M = 0.05
ROTATION_FLOOR_DEG = 2.0
MEDIAN_MULTIPLIER = 5.0
QUATERNION_NORM_TOLERANCE = 1e-4
T_I_C_TOLERANCE = 1e-9
VICON_MAX_BRACKET_NS = 20_000_000
INTERNAL_TOLERANCE = 1e-9
EVO_TRANSLATION_TOLERANCE_M = 1e-6
EVO_ROTATION_TOLERANCE_DEG = 1e-4
PROTOCOL_SCHEMA = "bhl-euroc-accuracy-protocol-v1"

PROTOCOL = {
    "version": 1,
    "question": ("How close are the causal native ORB-SLAM3 stereo and stereo-inertial poses, "
                 "expressed in the IMU frame, to the recorded EuRoC V1_01_easy references over "
                 "the first 60 s (1200 stereo pairs)?"),
    "inputs": {
        "replay": ("euroc-actual-20261012/full/native: result.json, <mode>/frames.jsonl "
                   "(sha256 must equal result.json), rectification.json"),
        "dataset": "euroc-actual-20261012/dataset/mav0: cam0, imu0 and vicon0 sensor.yaml; vicon0/data.csv",
    },
    "references": {
        "original": {
            "file": "mav0/state_groundtruth_estimate0/data.csv",
            "git_blob_sha1": ORIGINAL_BLOB,
            "identity": "byte-identical to OpenVINS ov_data/euroc_mav/V1_01_easy_original.csv",
            "rate": "200 Hz IMU-body poses starting 1.04 s after the first camera frame",
            "association": ("unique nearest sample within 2.5 ms (half the 200 Hz period) of the "
                            "integer camera timestamp; no interpolation; achieved maximum reported"),
            "known_issue": "OpenVINS documents its orientation as inaccurate",
        },
        "recomputed": {
            "file": "OpenVINS ov_data/euroc_mav/V1_01_easy.csv (vicon2gt recomputation)",
            "git_blob_sha1": RECOMPUTED_BLOB,
            "source_url": RECOMPUTED_URL,
            "introduced_in_commit": OPENVINS_INTRODUCED_COMMIT,
            "license": "GPL-3.0 repository; read in place for evaluation, not copied into this repository",
            "rate": "20 Hz on camera timestamps",
            "association": ("integer nanoseconds: exact equality, otherwise the unique nearest sample "
                            "within 256 ns (one float64 ulp at 1.4e18 ns; 240 of the 1200 window rows "
                            "were written through double precision and sit 256 ns early); exact and "
                            "256 ns counts reported"),
        },
        "pinning": "refuse to run when either file's git blob SHA-1 differs from the pinned value",
    },
    "quaternions": "w, x, y, z Hamilton order; unit norm within 1e-4, then renormalized",
    "estimate_frame": ("T_W_I = T_W_Crect @ inv(T_I_Crect); T_I_Crect from rectification.json must equal "
                       "inv(T_BS_imu0) @ T_BS_cam0 @ [R_left_rectification^T] recomputed from sensor.yaml "
                       "within 1e-9"),
    "accepted_frames": {
        "stereo": "native tracked frames",
        "stereo_inertial": ("native tracked frames with map_imu_initialized (primary); tracked frames "
                            "before IMU initialization form a separate non-primary epoch scored with SE3 only"),
    },
    "segmentation": {
        "epochs": ("split at a native map_id change or disappearance, at any change of the "
                   "IMU-initialization flag, and at untracked frames"),
        "literal_rule_as_tasked": ("split at map-id/initialization changes AND at causal pose "
                                   "discontinuities (per-frame translation increment > max(0.05 m, 5x the "
                                   "median increment) or rotation increment > 2 deg)"),
        "amendment_evidence_non_score": (
            "Measured before any score: over the 60 s window, ground-truth motion at the associated "
            "camera frames reaches 2.25 deg per 50 ms frame (original 2.248, recomputed 2.246) with 20 "
            "frames above 2 deg in both references, while ground-truth translation increments stay "
            "below 0.039 m. On the estimates the literal absolute 2 deg clause fires 25 times (stereo) "
            "and 23 times (stereo-IMU), all but two of them at those genuine fast rotations."),
        "primary_rule": (
            "Within each epoch, over consecutive accepted frames k-1 -> k of the estimated IMU-frame "
            "trajectory in the estimator's own world frame, with d_k = |p_k - p_(k-1)| and "
            "theta_k = angle(R_(k-1)^T R_k): split before frame k when d_k > max(0.05 m, 5 x median(d) "
            "over the epoch) or theta_k > max(2 deg, 5 x median(theta) over the epoch)."),
        "translation_floor_m": TRANSLATION_FLOOR_M,
        "rotation_floor_deg": ROTATION_FLOOR_DEG,
        "median_multiplier": MEDIAN_MULTIPLIER,
        "expected_boundaries_from_increment_probe": {
            "stereo": [],
            "stereo_inertial": [
                {"before_frame": 161, "time_s": 8.05, "translation_m": 0.1241, "rotation_deg": 2.311},
                {"before_frame": 412, "time_s": 20.60, "translation_m": 0.8255, "rotation_deg": 5.583},
            ],
        },
        "blind_spot": ("a rotation-only jump between 2 deg and 5 x the median rotation increment "
                       "(about 3.3-3.8 deg here) would not split; both observed jumps carry translation "
                       "far above the translation threshold"),
        "boundary_evidence": "each boundary reports the reference increment over the same frame pair",
        "literal_rule_sensitivity": ("boundary count only, with how many coincide with reference rotation "
                                     "increments above 2 deg; never scored"),
        "min_segment_frames": MIN_SEGMENT_FRAMES,
        "min_segment_note": ("segments with fewer than 20 reference-associated frames (1 s) or degenerate "
                             "positions are listed as unscored; the 2-frame 7.95-8.00 s stereo-IMU fragment "
                             "is expected to be unscored"),
    },
    "alignment": {
        "scale": 1.0,
        "stereo": ["se3"],
        "stereo_inertial": ["se3", "gravity_yaw_translation"],
        "stereo_inertial_pre_initialization": ["se3"],
        "method": ("pose_metrics.align_metric_poses on the reference-associated positions of each "
                   "segment independently: Kabsch/Umeyama without scale (se3) or yaw about gravity plus "
                   "translation (gravity_yaw_translation, 4-DoF)"),
        "gravity_evidence": "SE3 alignment tilt (angle between aligned estimator z and reference z) reported",
    },
    "metrics": {
        "ate": "|p_ref - p_aligned| in m: RMSE, mean, median, max",
        "rotation_error": "angle(R_ref^T R_aligned) in deg: mean, RMSE, median, max",
        "rpe": ("intervals 1, 5 and 10 s: pairs (i, j) of reference-associated frames in the same segment, "
                "j the frame nearest to t_i + interval within 25 ms; E = inv(inv(G_i) G_j) inv(E_i) E_j; "
                "translation |E_t| in m and rotation angle in deg: RMSE, mean, median, max; alignment-invariant"),
        "rpe_intervals_s": list(RPE_INTERVALS_S),
        "rpe_pair_tolerance_s": RPE_PAIR_TOLERANCE_NS * 1e-9,
        "drift_per_interval": ("|E_t| / reference path length between i and j (sum of consecutive associated "
                               "reference position increments), in percent; pairs with path < 0.10 m are "
                               "excluded and counted"),
        "drift_min_path_m": DRIFT_MIN_PATH_M,
        "segment_endpoint_drift": "the same ratio for the first and last frame of a segment with path >= 0.10 m",
        "coverage": ("tracked frames / 1200; frames with each reference; IMU-initialized frames (stereo-IMU); "
                     "frames in scored primary segments"),
    },
    "stitched_whole": ("secondary: one alignment over all accepted primary frames of an arm (stereo-IMU: all "
                       "IMU-initialized frames); RPE pairs may cross segment boundaries"),
    "comparison": ("stereo vs stereo-IMU on identical timestamps: the stereo arm is re-scored (SE3) on exactly "
                   "the reference-associated frames of each scored primary stereo-IMU segment and of the "
                   "stereo-IMU stitched whole"),
    "validation": {
        "vicon": ("T_W_I(vicon) = T_W_V @ inv(T_BS_vicon0), compared at each reference's timestamps inside the "
                  "60 s window after linear/SLERP interpolation between vicon samples at most 20 ms apart; "
                  "alternatives T_W_V @ T_BS and T_W_V reported to show the check discriminates; residual "
                  "inv(T_ref) @ T_vicon summarized as rotation and translation statistics"),
        "reference_vs_reference": "inv(T_recomputed) @ T_original at matched timestamps (constant body-frame offset check)",
        "conversion_alternatives": ("diagnostic only, never used to choose: stereo SE3 ATE and mean body-frame "
                                    "rotation residual for T_W_C @ inv(T_I_C) (declared), T_W_C @ T_I_C and T_W_C"),
        "internal": "SE3 and 4-DoF ATE recomputed with pose_metrics.score_trajectories must agree within 1e-9",
    },
    "evo_crosscheck": {
        "tool": "evo 1.34 command line (depth-env)",
        "arm": "stereo, longest scored segment",
        "commands": ("evo_ape euroc <reference csv> <estimate TUM> -a --t_max_diff 0.0025 with -r trans_part "
                     "and -r angle_deg; evo_rpe euroc ... --delta 20 --delta_unit f --all_pairs with -r "
                     "trans_part and -r angle_deg (20 frames = 1 s)"),
        "agreement": "equal pose and pair counts; |difference| <= 1e-6 m and <= 1e-4 deg on RMSE, mean, median, max",
    },
}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def git_blob_sha1(path) -> str:
    """The object id git assigns to the file's bytes (``git hash-object``)."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def freeze_protocol(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "protocol.json"
    value = {"schema": PROTOCOL_SCHEMA, "frozen_utc": utc_now(),
             "evaluator": "scripts/bench/euroc_native_evaluate.py",
             "evaluator_sha256_at_freeze": sha256(Path(__file__)),
             "scores_computed_before_freeze": False, "protocol": PROTOCOL}
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return path


def require_frozen_protocol(output_dir: Path) -> dict:
    path = output_dir / "protocol.json"
    if not path.is_file():
        raise ValueError("Run freeze-protocol before evaluate: no frozen protocol.json")
    frozen = json.loads(path.read_text())
    if frozen.get("schema") != PROTOCOL_SCHEMA or frozen.get("protocol") != json.loads(json.dumps(PROTOCOL)):
        raise ValueError("Frozen protocol.json differs from this evaluator's PROTOCOL; refusing to score")
    return {"path": str(path), "sha256": sha256(path), "frozen_utc": frozen["frozen_utc"],
            "evaluator_sha256_at_freeze": frozen["evaluator_sha256_at_freeze"]}


# ---------------------------------------------------------------- geometry

def rotation_angle_deg(rotations):
    """Rotation angle in degrees of one (3, 3) or many (n, 3, 3) matrices.

    atan2 of the skew and trace parts stays accurate near 0 deg, where the
    arccos form loses about half of the significant digits.
    """
    r = np.asarray(rotations, dtype=float)
    single = r.ndim == 2
    r = r.reshape(-1, 3, 3)
    skew = np.stack([r[:, 2, 1] - r[:, 1, 2], r[:, 0, 2] - r[:, 2, 0], r[:, 1, 0] - r[:, 0, 1]], axis=1)
    angles = np.degrees(np.arctan2(0.5 * np.linalg.norm(skew, axis=1),
                                   0.5 * (np.trace(r, axis1=1, axis2=2) - 1.0)))
    return float(angles[0]) if single else angles


def summary(values):
    v = np.asarray(values, dtype=float)
    if not len(v):
        return None
    return {"n": int(len(v)), "rmse": float(np.sqrt(np.mean(v * v))), "mean": float(v.mean()),
            "median": float(np.median(v)), "max": float(v.max())}


def quaternion_matrix(wxyz):
    """Hamilton w, x, y, z quaternion to a rotation matrix (renormalized)."""
    q = np.asarray(wxyz, dtype=float)
    norm = float(np.linalg.norm(q))
    if q.shape != (4,) or not math.isfinite(norm) or abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        raise ValueError("Reference quaternion must have unit norm within 1e-4")
    w, x, y, z = q / norm
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_from_row(position, wxyz):
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_matrix(wxyz)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return transform(matrix)


def read_pose_csv(path):
    """EuRoC/OpenVINS CSV: integer ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z, ... ('#' comments)."""
    stamps, poses = [], []
    with Path(path).open() as stream:
        for row in csv.reader(stream):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 8:
                raise ValueError("Pose CSV rows need an integer timestamp, a position and a w,x,y,z quaternion")
            stamps.append(int(row[0]))
            poses.append(pose_from_row([float(v) for v in row[1:4]], [float(v) for v in row[4:8]]))
    ns = np.asarray(stamps, dtype=np.int64)
    if len(ns) < 2 or np.any(np.diff(ns) <= 0):
        raise ValueError("Pose CSV timestamps must be strictly increasing integer nanoseconds")
    return ns, np.stack(poses)


def checked_t_bs(sensor_yaml) -> np.ndarray:
    import yaml
    value = yaml.safe_load(Path(sensor_yaml).read_text())["T_BS"]
    if value.get("rows") != 4 or value.get("cols") != 4 or len(value.get("data", [])) != 16:
        raise ValueError("EuRoC T_BS must be a declared 4x4 matrix")
    return transform(np.asarray(value["data"], dtype=float).reshape(4, 4))


def independent_t_i_c(dataset: Path, rectification: dict):
    """Recompute T_imu_rectifiedleft from sensor.yaml and compare to the replay receipt."""
    unrectify = np.eye(4)
    unrectify[:3, :3] = np.asarray(rectification["R_left_rectification"], dtype=float).T
    expected = (np.linalg.inv(checked_t_bs(dataset / "mav0/imu0/sensor.yaml"))
                @ checked_t_bs(dataset / "mav0/cam0/sensor.yaml") @ unrectify)
    stored = transform(rectification["T_imu_rectified_left_camera"])
    difference = float(np.abs(expected - stored).max())
    if difference > T_I_C_TOLERANCE:
        raise ValueError("rectification.json T_imu_rectified_left_camera differs from sensor.yaml recomputation")
    return stored, difference


# ------------------------------------------------------------- association

def associate_ns(estimated_ns, reference_ns, tolerance_ns: int):
    """Unique nearest reference sample for each estimate stamp, on integer ns.

    Returns estimate indices, reference indices and signed reference-minus-
    estimate differences in ns. Ties go to the earlier reference sample.
    """
    e = np.asarray(estimated_ns, dtype=np.int64)
    g = np.asarray(reference_ns, dtype=np.int64)
    for stamps in (e, g):
        if stamps.ndim != 1 or not len(stamps) or np.any(np.diff(stamps) <= 0):
            raise ValueError("Timestamps must be strictly increasing integer nanoseconds")
    if tolerance_ns < 0:
        raise ValueError("Association tolerance must be nonnegative")
    position = np.searchsorted(g, e)
    lower = np.clip(position - 1, 0, len(g) - 1)
    upper = np.clip(position, 0, len(g) - 1)
    pick = np.where(np.abs(g[upper] - e) < np.abs(g[lower] - e), upper, lower)
    difference = g[pick] - e
    keep = np.abs(difference) <= tolerance_ns
    ei, gi = np.flatnonzero(keep), pick[keep]
    if len(np.unique(gi)) != len(gi):
        raise ValueError("A reference sample would be associated twice")
    return ei, gi, difference[keep]


def association_report(frame_ns, reference, tolerance_ns, origin_ns):
    ei, gi, difference = associate_ns(frame_ns, reference["timestamps_ns"], tolerance_ns)
    # Same pairs through the existing float-seconds associator (relative clock).
    fe = (np.asarray(frame_ns, dtype=np.int64) - origin_ns) * 1e-9
    fg = (reference["timestamps_ns"] - origin_ns) * 1e-9
    e2, g2 = associate_timestamps(fe, fg, max_difference_s=tolerance_ns * 1e-9 + 1e-12)
    missing = sorted(set(range(len(frame_ns))) - set(ei.tolist()))
    return {"pairs": dict(zip(ei.tolist(), gi.tolist())), "report": {
        "tolerance_ns": tolerance_ns, "associated_frames": int(len(ei)),
        "exact_ns_matches": int(np.sum(difference == 0)),
        "nonzero_offset_matches": int(np.sum(difference != 0)),
        "offset_ns_histogram": {str(k): int(v) for k, v in zip(*np.unique(difference, return_counts=True))},
        "max_abs_offset_ns": int(np.abs(difference).max()) if len(difference) else None,
        "frames_without_reference": len(missing),
        "frames_without_reference_index_range": [missing[0], missing[-1]] if missing else None,
        "pose_metrics_associate_timestamps_identical": bool(np.array_equal(ei, e2) and np.array_equal(gi, g2)),
    }}


# ------------------------------------------------------------------ inputs

def load_reference(path, expected_blob: str, label: str) -> dict:
    path = Path(path)
    blob = git_blob_sha1(path)
    if blob != expected_blob:
        raise ValueError(f"{label} reference {path} has git blob {blob}, pinned {expected_blob}; refusing to evaluate")
    ns, poses = read_pose_csv(path)
    return {"label": label, "path": str(path), "git_blob_sha1": blob, "sha256": sha256(path),
            "bytes": path.stat().st_size, "rows": int(len(ns)), "timestamps_ns": ns, "T_W_I": poses,
            "median_period_s": float(np.median(np.diff(ns))) * 1e-9,
            "first_timestamp_ns": int(ns[0]), "last_timestamp_ns": int(ns[-1])}


def load_arm(replay: Path, mode: str, result: dict) -> dict:
    path = replay / mode / "frames.jsonl"
    native = next(a for a in result["arms"] if a["mode"] == mode)
    if sha256(path) != native["frames_sha256"]:
        raise ValueError("Native frame trace differs from completed replay checksum")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != result["paired_frames_per_arm"] or len(rows) != native["frames"] or len(rows) != FRAMES:
        raise ValueError("Native frame count differs from the completed 1200-pair replay")
    origin = int(result["clock_origin_ns"])
    frames = []
    for index, row in enumerate(rows):
        frame = row["native"]
        ns = int(row["original_timestamp_ns"])
        if abs(float(frame["timestamp_s"]) - (ns - origin) * 1e-9) > 1e-9:
            raise ValueError("Native relative timestamp disagrees with the original integer clock")
        tracked = frame["tracked"] is True
        if tracked and frame.get("T_W_C") is None:
            raise ValueError("Tracked native frame without a pose")
        frames.append({"index": index, "ns": ns, "tracked": tracked, "state": frame["tracking_state"],
                       "map_id": frame["map_id"],
                       "stereo_depth_matches": frame.get("diagnostics", {}).get("positive_stereo_depth_matches"),
                       "imu_initialized": (bool(frame["inertial"]["map_imu_initialized"])
                                           if mode == "stereo_inertial" else None),
                       "T_W_C": transform(frame["T_W_C"]) if tracked else None})
    ns = np.array([f["ns"] for f in frames], dtype=np.int64)
    if np.any(np.diff(ns) <= 0):
        raise ValueError("Native frame timestamps must increase")
    tracked = sum(f["tracked"] for f in frames)
    if tracked != native["tracked_frames"]:
        raise ValueError("Tracked-frame count differs from replay summary")
    accepted = sum(f["tracked"] and (mode == "stereo" or f["imu_initialized"]) for f in frames)
    if accepted != native["accepted_initialized_frames"]:
        raise ValueError("Accepted initialized frame count differs from replay summary")
    return {"mode": mode, "frames": frames, "ns": ns, "frames_sha256": native["frames_sha256"],
            "native_summary": {k: native[k] for k in ("tracked_frames", "accepted_initialized_frames",
                                                      "initialization_first_sensor_time_s",
                                                      "tracking_states", "observed_map_ids")}}


# ------------------------------------------------------------ segmentation

def epochs(frames):
    """Maximal runs of consecutive tracked frames with one map id and IMU flag."""
    result, previous = [], None
    for frame in frames:
        if not frame["tracked"]:
            previous = None
            continue
        key = (frame["map_id"], frame["imu_initialized"])
        if previous is None or key != previous[0] or frame["index"] != previous[1] + 1:
            if not result:
                reason = "first_tracked_frame"
            elif previous is None:
                reason = "after_untracked_frames"
            elif key[0] != previous[0][0]:
                reason = "map_id_changed"
            else:
                reason = "imu_initialization_changed"
            result.append({"epoch": len(result), "map_id": key[0], "imu_initialized": key[1],
                           "start_reason": reason, "frames": []})
        result[-1]["frames"].append(frame["index"])
        previous = (key, frame["index"])
    return result


def increments(poses):
    poses = np.asarray(poses, dtype=float)
    d = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    relative = np.einsum("nji,njk->nik", poses[:-1, :3, :3], poses[1:, :3, :3])
    return d, rotation_angle_deg(relative)


def step_error(estimated_pair, reference_pair):
    """One-step relative pose error inv(inv(G_a) G_b) inv(E_a) E_b: translation m, rotation deg."""
    estimated = np.linalg.inv(estimated_pair[0]) @ estimated_pair[1]
    reference = np.linalg.inv(reference_pair[0]) @ reference_pair[1]
    error = np.linalg.inv(reference) @ estimated
    return float(np.linalg.norm(error[:3, 3])), rotation_angle_deg(error[:3, :3])


def discontinuities(poses):
    """Primary-rule and literal-rule boundaries (positions k: split before k)."""
    d, theta = increments(poses)
    if not len(d):
        return {"translation_threshold_m": None, "rotation_threshold_deg": None,
                "median_translation_increment_m": None, "median_rotation_increment_deg": None}, [], [], d, theta
    median_d, median_theta = float(np.median(d)), float(np.median(theta))
    tau_t = max(TRANSLATION_FLOOR_M, MEDIAN_MULTIPLIER * median_d)
    tau_r = max(ROTATION_FLOOR_DEG, MEDIAN_MULTIPLIER * median_theta)
    primary = [k for k in range(1, len(d) + 1) if d[k - 1] > tau_t or theta[k - 1] > tau_r]
    literal = [k for k in range(1, len(d) + 1) if d[k - 1] > tau_t or theta[k - 1] > ROTATION_FLOOR_DEG]
    return {"translation_threshold_m": tau_t, "rotation_threshold_deg": tau_r,
            "literal_rotation_threshold_deg": ROTATION_FLOOR_DEG,
            "median_translation_increment_m": median_d, "median_rotation_increment_deg": median_theta}, \
        primary, literal, d, theta


def segment_arm(arm: dict, imu: dict):
    """Epochs split by the primary discontinuity rule; literal-rule hits recorded."""
    segments, epoch_records, literal = [], [], []
    for epoch in epochs(arm["frames"]):
        index = epoch["frames"]
        thresholds, primary, literal_cuts, d, theta = discontinuities(np.stack([imu[i] for i in index]))
        label = "primary" if arm["mode"] == "stereo" or epoch["imu_initialized"] else "pre_imu_initialization"
        epoch_records.append({"epoch": epoch["epoch"], "label": label, "map_id": epoch["map_id"],
                              "imu_initialized": epoch["imu_initialized"], "start_reason": epoch["start_reason"],
                              "frames": len(index), "first_frame": index[0], "last_frame": index[-1],
                              **thresholds, "primary_boundaries_before_frames": [index[k] for k in primary],
                              "literal_rule_boundaries": len(literal_cuts)})
        cuts = [0, *primary, len(index)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            record = {"segment": len(segments), "epoch": epoch["epoch"], "label": label,
                      "frames": index[a:b], "start_reason": epoch["start_reason"] if a == 0 else "pose_discontinuity"}
            if a > 0:
                record["boundary"] = {"after_frame": index[a - 1], "before_frame": index[a],
                                      "estimate_translation_increment_m": float(d[a - 1]),
                                      "estimate_rotation_increment_deg": float(theta[a - 1]),
                                      "translation_threshold_m": thresholds["translation_threshold_m"],
                                      "rotation_threshold_deg": thresholds["rotation_threshold_deg"]}
            segments.append(record)
        literal.extend({"after_frame": index[k - 1], "before_frame": index[k],
                        "estimate_translation_increment_m": float(d[k - 1]),
                        "estimate_rotation_increment_deg": float(theta[k - 1])} for k in literal_cuts)
    return segments, epoch_records, literal


def step_error_profile(segments, imu, references, association, frame_ns, origin_ns, matches=None) -> dict:
    """Post-run diagnostic: one-frame relative pose error (RPE over 50 ms) inside scored primary segments.

    Not predeclared; added after the first run showed a stretch of large
    stereo frame-to-frame errors below the discontinuity thresholds.
    """
    value = {}
    for name, reference in references.items():
        pairs = association[name]["pairs"]
        rows = []
        for segment in segments:
            if segment["label"] != "primary" or segment["scores"][name]["se3"]["status"] != "MEASURED":
                continue
            for a, b in zip(segment["frames"][:-1], segment["frames"][1:]):
                if a in pairs and b in pairs:
                    error = step_error((imu[a], imu[b]), reference["T_W_I"][[pairs[a], pairs[b]]])
                    rows.append((error[0], error[1], b, (int(frame_ns[b]) - origin_ns) * 1e-9))
        if not rows:
            value[name] = None
            continue
        translation = np.array([r[0] for r in rows])
        rotation = np.array([r[1] for r in rows])
        large = sorted((r for r in rows if r[0] > 0.02), key=lambda r: r[3])
        clusters = []
        for row in large:
            if clusters and row[3] - clusters[-1]["last_time_s"] < 1.0:
                clusters[-1].update(last_time_s=row[3], last_frame=row[2], steps=clusters[-1]["steps"] + 1,
                                    max_translation_m=max(clusters[-1]["max_translation_m"], row[0]))
            else:
                clusters.append({"first_time_s": row[3], "last_time_s": row[3], "first_frame": row[2] - 1,
                                 "last_frame": row[2], "steps": 1, "max_translation_m": row[0]})
        for cluster in clusters:
            counts = [matches[i] for i in range(cluster["first_frame"], cluster["last_frame"] + 1)
                      if matches and matches.get(i) is not None]
            cluster["front_end_stereo_depth_matches"] = ({"min": min(counts), "max": max(counts)} if counts else None)
        value[name] = {
            "pairs": len(rows),
            "translation_m": {"median": float(np.median(translation)), "p99": float(np.quantile(translation, .99)),
                              "max": float(translation.max())},
            "rotation_deg": {"median": float(np.median(rotation)), "p99": float(np.quantile(rotation, .99)),
                             "max": float(rotation.max())},
            "steps_over_2cm": len(large), "clusters_over_2cm_within_1s": clusters,
            "largest": [{"before_frame": r[2], "time_s": r[3], "translation_m": r[0], "rotation_deg": r[1]}
                        for r in sorted(rows, key=lambda r: -r[0])[:5]]}
    return value


# ----------------------------------------------------------------- scoring

def cumulative_path(reference_poses):
    steps = np.linalg.norm(np.diff(reference_poses[:, :3, 3], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def relative_errors(estimated, reference, i, j):
    dg = np.linalg.inv(reference[i]) @ reference[j]
    de = np.linalg.inv(estimated[i]) @ estimated[j]
    error = np.linalg.inv(dg) @ de
    return np.linalg.norm(error[..., :3, 3], axis=-1), rotation_angle_deg(error[..., :3, :3])


def rpe_pairs(ns, delta_s: float):
    """(i, j) with j > i the frame nearest t_i + delta, within the pairing tolerance."""
    ns = np.asarray(ns, dtype=np.int64)
    target = ns + int(round(delta_s * 1e9))
    position = np.searchsorted(ns, target)
    pairs = []
    for i in range(len(ns)):
        best = None
        for j in (position[i] - 1, position[i]):
            if i < j < len(ns):
                gap = abs(int(ns[j]) - int(target[i]))
                if gap <= RPE_PAIR_TOLERANCE_NS and (best is None or gap < best[0]):
                    best = (gap, j)
        if best is not None:
            pairs.append((i, best[1]))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def rpe_interval(estimated, reference, ns, cumulative, delta_s: float) -> dict:
    pairs = rpe_pairs(ns, delta_s)
    if not len(pairs):
        return {"status": "NO_PAIRS", "interval_s": delta_s, "pairs": 0}
    i, j = pairs[:, 0], pairs[:, 1]
    translation, rotation = relative_errors(estimated, reference, i, j)
    length = cumulative[j] - cumulative[i]
    keep = length >= DRIFT_MIN_PATH_M
    actual = (np.asarray(ns)[j] - np.asarray(ns)[i]) * 1e-9
    return {"status": "MEASURED", "interval_s": delta_s, "pairs": int(len(pairs)),
            "actual_interval_s": {"min": float(actual.min()), "max": float(actual.max())},
            "translation_m": summary(translation), "rotation_deg": summary(rotation),
            "drift_percent": summary(100.0 * translation[keep] / length[keep]),
            "drift_pairs": int(keep.sum()), "drift_excluded_short_path_pairs": int((~keep).sum()),
            "reference_path_m": summary(length)}


def internal_difference(estimated, reference, ns, alignment, ate, rotation) -> dict:
    """Max |difference| against pose_metrics.score_trajectories on the same pairs.

    The protocol's 1e-9 agreement applies to ATE translation. Rotation is
    reported too; pose_metrics uses arccos, which near 0 deg resolves only
    about 1e-6 deg, so that difference is informational.
    """
    times = (np.asarray(ns, dtype=np.int64) - int(ns[0])) * 1e-9
    other = score_trajectories(Trajectory(times, estimated), Trajectory(times, reference),
                               alignment=alignment, max_difference_s=1e-6)
    if other["associated_poses"] != len(ns):
        return {"ate_translation_m": math.inf, "rotation_deg": math.inf}
    return {"ate_translation_m": max(abs(other["ate_translation_m"][k] - ate[k]) for k in ("rmse", "mean", "max")),
            "rotation_deg": max(abs(other["ate_rotation_deg"][k] - rotation[k]) for k in ("rmse", "mean", "max"))}


def score(estimated, reference, ns, alignment: str, origin_ns: int, *, internal=True):
    """One rigid alignment over the supplied associated poses; ATE, rotation, RPE, drift."""
    from scipy.spatial.transform import Rotation
    estimated = np.asarray(estimated, dtype=float)
    reference = np.asarray(reference, dtype=float)
    ns = np.asarray(ns, dtype=np.int64)
    base = {"alignment": alignment, "associated_frames": int(len(ns))}
    if len(ns):
        base.update(first_time_s=(int(ns[0]) - origin_ns) * 1e-9, last_time_s=(int(ns[-1]) - origin_ns) * 1e-9)
    if len(ns) < MIN_SEGMENT_FRAMES:
        return {**base, "status": "UNSCORED_TOO_SHORT"}, None
    aligned, a, mode = align_metric_poses(estimated, reference, alignment=alignment)
    if mode != alignment:
        return {**base, "status": "UNSCORED_DEGENERATE_GEOMETRY", "alignment_fallback": mode}, None
    ate = np.linalg.norm(aligned[:, :3, 3] - reference[:, :3, 3], axis=1)
    residual = np.einsum("nji,njk->nik", reference[:, :3, :3], aligned[:, :3, :3])
    rotation = rotation_angle_deg(residual)
    mean_residual = Rotation.from_matrix(residual).mean()
    cumulative = cumulative_path(reference)
    translation_end, rotation_end = relative_errors(aligned, reference, 0, len(ns) - 1)
    path = float(cumulative[-1])
    value = {**base, "status": "MEASURED", "scale": 1.0,
             "T_referenceworld_estimatorworld": a.tolist(),
             "alignment_tilt_deg": float(np.degrees(np.arccos(np.clip(a[2, 2], -1.0, 1.0)))),
             "alignment_yaw_deg": float(np.degrees(math.atan2(a[1, 0], a[0, 0]))),
             "ate_m": summary(ate), "rotation_error_deg": summary(rotation),
             "mean_body_frame_rotation_residual_deg": float(np.degrees(mean_residual.magnitude())),
             "reference_path_length_m": path,
             "rpe": {f"{d:g}s": rpe_interval(aligned, reference, ns, cumulative, d) for d in RPE_INTERVALS_S},
             "endpoint_drift": {"translation_m": float(translation_end), "rotation_deg": float(rotation_end),
                                "reference_path_m": path,
                                "drift_percent": (100.0 * float(translation_end) / path
                                                  if path >= DRIFT_MIN_PATH_M else None)}}
    if internal:
        value["internal_score_trajectories_max_abs_difference"] = internal_difference(
            estimated, reference, ns, alignment, value["ate_m"], value["rotation_error_deg"])
    return value, {"ate_m": ate, "rotation_deg": rotation}


def gather(indices, pairs, imu, reference, frame_ns):
    """Associated subset of frame indices with estimate poses and reference poses."""
    kept = [i for i in indices if i in pairs]
    if not kept:
        return kept, np.empty((0, 4, 4)), np.empty((0, 4, 4)), np.empty(0, dtype=np.int64)
    estimated = np.stack([imu[i] for i in kept])
    truth = reference["T_W_I"][[pairs[i] for i in kept]]
    return kept, estimated, truth, np.asarray(frame_ns, dtype=np.int64)[kept]


def alignments_for(mode, label):
    if mode == "stereo":
        return PROTOCOL["alignment"]["stereo"]
    if label == "pre_imu_initialization":
        return PROTOCOL["alignment"]["stereo_inertial_pre_initialization"]
    return PROTOCOL["alignment"]["stereo_inertial"]


# -------------------------------------------------------------- validation

def interpolate_poses(source_ns, source_poses, query_ns, max_bracket_ns):
    """Linear position / SLERP rotation between bracketing samples <= max_bracket_ns apart."""
    from scipy.spatial.transform import Rotation
    source_ns = np.asarray(source_ns, dtype=np.int64)
    query_ns = np.asarray(query_ns, dtype=np.int64)
    upper = np.searchsorted(source_ns, query_ns, side="left")
    exact = (upper < len(source_ns)) & (source_ns[np.clip(upper, 0, len(source_ns) - 1)] == query_ns)
    lower = np.where(exact, upper, upper - 1)
    inside = (lower >= 0) & (upper < len(source_ns))
    lower, upper = np.clip(lower, 0, len(source_ns) - 1), np.clip(upper, 0, len(source_ns) - 1)
    gap = source_ns[upper] - source_ns[lower]
    valid = inside & (exact | (gap <= max_bracket_ns))
    alpha = np.where(exact | (gap == 0), 0.0, (query_ns - source_ns[lower]) / np.where(gap == 0, 1, gap))
    a, b = source_poses[lower], source_poses[upper]
    relative = Rotation.from_matrix(np.einsum("nji,njk->nik", a[:, :3, :3], b[:, :3, :3])).as_rotvec()
    poses = np.repeat(np.eye(4)[None], len(query_ns), axis=0)
    poses[:, :3, :3] = (Rotation.from_matrix(a[:, :3, :3])
                        * Rotation.from_rotvec(alpha[:, None] * relative)).as_matrix()
    poses[:, :3, 3] = (1 - alpha[:, None]) * a[:, :3, 3] + alpha[:, None] * b[:, :3, 3]
    return valid, poses


def residual_summary(reference, candidate) -> dict:
    """Body-frame residual inv(T_reference) @ T_candidate and world position difference."""
    from scipy.spatial.transform import Rotation
    residual = np.linalg.inv(reference) @ candidate
    rotations = Rotation.from_matrix(residual[:, :3, :3])
    mean = rotations.mean()
    spread = rotation_angle_deg(np.einsum("ij,njk->nik", mean.inv().as_matrix(), residual[:, :3, :3]))
    body = residual[:, :3, 3]
    world = candidate[:, :3, 3] - reference[:, :3, 3]
    return {"samples": int(len(residual)),
            "residual_rotation_deg": summary(rotation_angle_deg(residual[:, :3, :3])),
            "mean_residual_rotation_deg": float(np.degrees(mean.magnitude())),
            "mean_residual_rotvec_deg": np.degrees(mean.as_rotvec()).tolist(),
            "rotation_spread_about_mean_deg": summary(spread),
            "residual_translation_m": summary(np.linalg.norm(body, axis=1)),
            "mean_residual_translation_body_m": body.mean(axis=0).tolist(),
            "residual_translation_body_std_m": body.std(axis=0).tolist(),
            "world_position_difference_m": summary(np.linalg.norm(world, axis=1)),
            "mean_world_position_difference_m": world.mean(axis=0).tolist()}


def rigid_fit(source_positions, target_positions) -> np.ndarray:
    """Least-squares rotation and translation, no scale, mapping source onto target.

    The same Kabsch estimator as pose_metrics.align_metric_poses(alignment="se3"),
    vectorized for the vicon and reference diagnostics (about 12k poses per call);
    every score still uses align_metric_poses.
    """
    x = np.asarray(source_positions, dtype=float)
    y = np.asarray(target_positions, dtype=float)
    u, _, vt = np.linalg.svd((x - x.mean(axis=0)).T @ (y - y.mean(axis=0)))
    r = vt.T @ np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)]) @ u.T
    a = np.eye(4)
    a[:3, :3], a[:3, 3] = r, y.mean(axis=0) - r @ x.mean(axis=0)
    return a


def aligned_residual(reference, candidate) -> dict:
    """Residual after one SE3 world alignment of candidate onto reference.

    Separates a constant world-frame offset (absorbed by the alignment) from a
    body-frame offset: a wrong lever arm leaves position error that rotates
    with the body, a wrong body rotation leaves a constant rotation residual.
    """
    from scipy.spatial.transform import Rotation
    a = rigid_fit(candidate[:, :3, 3], reference[:, :3, 3])
    aligned = a[None] @ candidate
    residual = np.einsum("nji,njk->nik", reference[:, :3, :3], aligned[:, :3, :3])
    mean = Rotation.from_matrix(residual).mean()
    spread = rotation_angle_deg(np.einsum("ij,njk->nik", mean.inv().as_matrix(), residual))
    return {"world_alignment_rotation_deg": rotation_angle_deg(a[:3, :3]),
            "world_alignment_translation_m": a[:3, 3].tolist(),
            "position_error_m": summary(np.linalg.norm(aligned[:, :3, 3] - reference[:, :3, 3], axis=1)),
            "mean_body_rotation_residual_deg": float(np.degrees(mean.magnitude())),
            "body_rotation_spread_about_mean_deg": summary(spread)}


def vicon_validation(dataset: Path, references: dict, window_ns) -> dict:
    """Independent IMU-frame check: vicon marker pose composed with its T_BS."""
    sensor = dataset / "mav0/vicon0/sensor.yaml"
    data = dataset / "mav0/vicon0/data.csv"
    t_bs = checked_t_bs(sensor)
    vicon_ns, vicon_poses = read_pose_csv(data)
    inverse = np.linalg.inv(t_bs)
    candidates = {"declared_T_W_V_inv_T_BS": lambda t: t @ inverse,
                  "alternative_T_W_V_T_BS": lambda t: t @ t_bs,
                  "alternative_T_W_V_marker_as_body": lambda t: t}
    value = {"vicon_data_sha256": sha256(data), "vicon_sensor_yaml_sha256": sha256(sensor),
             "vicon_rows": int(len(vicon_ns)),
             "vicon_median_period_s": float(np.median(np.diff(vicon_ns))) * 1e-9,
             "T_BS_vicon0": t_bs.tolist(),
             "T_BS_rotation_angle_deg": rotation_angle_deg(t_bs[:3, :3]),
             "T_BS_rotation_note": ("the vicon0 T_BS rotation is within 0.2 deg of a half-turn (R ~ R^T), so "
                                    "T_BS and inv(T_BS) differ in rotation by only "
                                    f"{rotation_angle_deg(t_bs[:3, :3] @ t_bs[:3, :3]):.2f} deg; only the lever "
                                    "arm (translation) separates those two candidates"),
             "convention": "T_BS maps vicon-marker coordinates into body (IMU) coordinates: T_W_I = T_W_V @ inv(T_BS)",
             "interpolation": "vicon pose linearly / SLERP interpolated at each reference timestamp when the "
                              "bracketing vicon samples are at most 20 ms apart",
             "post_run_additions": ("aligned (one SE3 world alignment per candidate) residuals and the vicon "
                                    "time-offset scan were added after the first evaluation run, because the "
                                    "unaligned residual cannot separate world- from body-frame offsets; they do "
                                    "not feed any score"),
             "window_ns": [int(window_ns[0]), int(window_ns[1])], "references": {}}
    for name, reference in references.items():
        stamps = reference["timestamps_ns"]
        select = (stamps >= window_ns[0]) & (stamps <= window_ns[1])
        valid, interpolated = interpolate_poses(vicon_ns, vicon_poses, stamps[select], VICON_MAX_BRACKET_NS)
        truth = reference["T_W_I"][select][valid]
        # The scan uses every stride-th window sample (<= ~1500) to bound its cost.
        stride = max(1, int(math.ceil(select.sum() / 1500)))
        scan_stamps, scan_truth = stamps[select][::stride], reference["T_W_I"][select][::stride]
        scan = []
        for shift_ms in range(-30, 31):
            ok, shifted = interpolate_poses(vicon_ns, vicon_poses, scan_stamps + shift_ms * 1_000_000,
                                            VICON_MAX_BRACKET_NS)
            check = aligned_residual(scan_truth[ok], shifted[ok] @ inverse)
            scan.append((check["position_error_m"]["rmse"], shift_ms, check))
        best = min(scan, key=lambda row: row[0])
        value["references"][name] = {
            "reference_samples_in_window": int(select.sum()), "interpolated_samples": int(valid.sum()),
            "candidates": {key: residual_summary(truth, convert(interpolated[valid]))
                           for key, convert in candidates.items()},
            "candidates_after_se3_world_alignment": {key: aligned_residual(truth, convert(interpolated[valid]))
                                                     for key, convert in candidates.items()},
            "time_offset_scan_declared": {
                "shift_definition": "vicon sampled at reference time + shift; shifts -30..+30 ms in 1 ms steps",
                "sample_stride": stride, "samples": int(len(scan_stamps)),
                "best_shift_ms": best[1], "best": best[2], "zero_shift": scan[30][2]}}
    return value


def reference_comparison(original: dict, recomputed: dict, window_ns) -> dict:
    from scipy.spatial.transform import Rotation
    ei, gi, difference = associate_ns(recomputed["timestamps_ns"], original["timestamps_ns"], ORIGINAL_TOLERANCE_NS)
    stamps = recomputed["timestamps_ns"][ei]
    inside = (stamps >= window_ns[0]) & (stamps <= window_ns[1])
    a, b = recomputed["T_W_I"][ei], original["T_W_I"][gi]
    residual = np.linalg.inv(a) @ b
    offset = np.eye(4)
    offset[:3, :3] = Rotation.from_matrix(residual[:, :3, :3]).mean().as_matrix()
    offset[:3, 3] = residual[:, :3, 3].mean(axis=0)
    return {"residual": "inv(T_recomputed) @ T_original (body frame of the recomputed reference)",
            "matched_samples": int(len(ei)), "max_abs_association_offset_ns": int(np.abs(difference).max()),
            "full_overlap": residual_summary(a, b),
            "evaluation_window": residual_summary(a[inside], b[inside]),
            "inverse_direction_window": residual_summary(b[inside], a[inside]),
            "inverse_direction_note": "inv(T_original) @ T_recomputed, body frame of the original reference",
            "original_after_se3_world_alignment_window": aligned_residual(a[inside], b[inside]),
            "mean_offset_T_recomputed_original": offset.tolist()}


def reference_offset_reproduction(estimated, recomputed, original, ns, offset, origin_ns) -> dict:
    """Post-run diagnostic: does the constant inter-reference offset explain the score gap?

    Scores the same estimate against the recomputed reference, the original
    reference and the recomputed reference right-multiplied by the mean
    constant offset inv(T_recomputed) @ T_original.
    """
    rows = {}
    for name, truth in (("recomputed", recomputed), ("original", original),
                        ("recomputed_times_constant_offset", recomputed @ offset)):
        value, _ = score(estimated, truth, ns, "se3", origin_ns, internal=False)
        rows[name] = {"ate_rmse_m": value["ate_m"]["rmse"], "rotation_error_mean_deg": value["rotation_error_deg"]["mean"],
                      **{f"rpe_{k}_translation_rmse_m": value["rpe"][k]["translation_m"]["rmse"] for k in ("1s", "5s", "10s")},
                      **{f"drift_{k}_mean_percent": value["rpe"][k]["drift_percent"]["mean"] for k in ("1s", "5s", "10s")}}
    return {"note": ("post-run diagnostic, added after the first evaluation run; stereo arm, SE3, frames that "
                     "have both references"), "frames": int(len(ns)),
            "offset_rotation_deg": rotation_angle_deg(offset[:3, :3]),
            "offset_translation_m": offset[:3, 3].tolist(),
            "relative_translation_error_from_offset_rotation_percent":
                200.0 * math.sin(math.radians(rotation_angle_deg(offset[:3, :3])) / 2),
            "scores": rows}


# ------------------------------------------------------------------- evo

def write_tum(path: Path, ns, poses):
    """TUM rows: seconds tx ty tz qx qy qz qw (absolute EuRoC clock)."""
    from scipy.spatial.transform import Rotation
    poses = np.asarray(poses, dtype=float)
    quaternions = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    with Path(path).open("x") as stream:
        for stamp, position, q in zip(np.asarray(ns, dtype=np.int64), poses[:, :3, 3], quaternions):
            stamp = int(stamp)
            stream.write(f"{stamp // 10**9}.{stamp % 10**9:09d} {position[0]:.9f} {position[1]:.9f} "
                         f"{position[2]:.9f} {q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f}\n")


def run_evo(command, results_zip: Path) -> dict:
    process = subprocess.run([*map(str, command), "--save_results", str(results_zip), "--no_warnings"],
                             capture_output=True, text=True, timeout=600)
    if process.returncode != 0:
        raise RuntimeError(f"evo failed ({process.returncode}): {process.stderr[-2000:]}")
    with zipfile.ZipFile(results_zip) as archive:
        stats = json.loads(archive.read("stats.json"))
        errors = np.load(io.BytesIO(archive.read("error_array.npy")))
    return {"command": " ".join(map(str, command)), "stats": {k: float(v) for k, v in stats.items()},
            "count": int(len(errors)), "results_zip": str(results_zip)}


def compare_stats(evo: dict, own: dict, count: int, tolerance: float) -> dict:
    differences = {k: abs(evo["stats"][k] - own[k]) for k in ("rmse", "mean", "median", "max")}
    return {"evo": {k: evo["stats"][k] for k in differences}, "own": {k: own[k] for k in differences},
            "abs_difference": differences, "evo_count": evo["count"], "own_count": count,
            "agree": bool(evo["count"] == count and max(differences.values()) <= tolerance)}


def evo_crosscheck(evo_ape: Path, evo_rpe: Path, directory: Path, ns, estimated, references: dict,
                   own_scores: dict, label: str) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    estimate = directory / f"{label}_imu_estimate.tum"
    write_tum(estimate, ns, estimated)
    value = {"estimate_tum": str(estimate), "estimate_tum_sha256": sha256(estimate), "references": {}}
    for name, reference in references.items():
        own = own_scores[name]
        common = ["euroc", reference["path"], estimate, "--t_max_diff", "0.0025"]
        ape_t = run_evo([evo_ape, *common, "-a", "-r", "trans_part"], directory / f"ape_{name}_trans.zip")
        ape_r = run_evo([evo_ape, *common, "-a", "-r", "angle_deg"], directory / f"ape_{name}_angle.zip")
        rpe_args = ["--delta", "20", "--delta_unit", "f", "--all_pairs"]
        rpe_t = run_evo([evo_rpe, *common, *rpe_args, "-r", "trans_part"], directory / f"rpe_{name}_trans.zip")
        rpe_r = run_evo([evo_rpe, *common, *rpe_args, "-r", "angle_deg"], directory / f"rpe_{name}_angle.zip")
        rpe = own["rpe"]["1s"]
        checks = {"ape_translation_m": compare_stats(ape_t, own["ate_m"], own["associated_frames"],
                                                     EVO_TRANSLATION_TOLERANCE_M),
                  "ape_rotation_deg": compare_stats(ape_r, own["rotation_error_deg"], own["associated_frames"],
                                                    EVO_ROTATION_TOLERANCE_DEG),
                  "rpe_1s_translation_m": compare_stats(rpe_t, rpe["translation_m"], rpe["pairs"],
                                                        EVO_TRANSLATION_TOLERANCE_M),
                  "rpe_1s_rotation_deg": compare_stats(rpe_r, rpe["rotation_deg"], rpe["pairs"],
                                                       EVO_ROTATION_TOLERANCE_DEG)}
        value["references"][name] = {"commands": [r["command"] for r in (ape_t, ape_r, rpe_t, rpe_r)],
                                     "checks": checks, "agree": all(c["agree"] for c in checks.values())}
    value["agree"] = all(r["agree"] for r in value["references"].values())
    return value


# ------------------------------------------------------------------ driver

def evaluate(dataset: Path, replay: Path, recomputed_path: Path, output_dir: Path, *,
             evo_ape: Path | None = None, evo_rpe: Path | None = None,
             original_blob: str = ORIGINAL_BLOB, recomputed_blob: str = RECOMPUTED_BLOB) -> dict:
    protocol = require_frozen_protocol(output_dir)
    results_path = output_dir / "results.json"
    if results_path.exists() or (output_dir / "evo").exists() or (output_dir / "per_frame_errors.csv").exists():
        raise ValueError("Evaluator outputs already exist; results are immutable")
    references = {"original": load_reference(dataset / "mav0/state_groundtruth_estimate0/data.csv",
                                             original_blob, "original"),
                  "recomputed": load_reference(recomputed_path, recomputed_blob, "recomputed")}
    tolerances = {"original": ORIGINAL_TOLERANCE_NS, "recomputed": RECOMPUTED_TOLERANCE_NS}
    result = json.loads((replay / "result.json").read_text())
    if (result.get("schema") != "bhl-external-euroc-result-v1" or result.get("status") != "COMPLETED"
            or result.get("stage") != "full" or result.get("paired_frames_per_arm") != FRAMES):
        raise ValueError("A completed full 1200-pair EuRoC native replay is required")
    origin = int(result["clock_origin_ns"])
    rectification = json.loads((replay / "rectification.json").read_text())
    t_i_c, t_i_c_difference = independent_t_i_c(dataset, rectification)
    arms = {mode: load_arm(replay, mode, result) for mode in ("stereo", "stereo_inertial")}
    if not np.array_equal(arms["stereo"]["ns"], arms["stereo_inertial"]["ns"]):
        raise ValueError("Both arms must share the same original camera timestamps")
    frame_ns = arms["stereo"]["ns"]
    window = (int(frame_ns[0]), int(frame_ns[-1]))
    association = {name: association_report(frame_ns, ref, tolerances[name], origin)
                   for name, ref in references.items()}
    per_frame_rows = []
    output_arms = {}
    imu_by_arm, segments_by_arm = {}, {}
    for mode, arm in arms.items():
        imu = {f["index"]: calibrated_body_pose(f["T_W_C"], t_i_c) for f in arm["frames"] if f["tracked"]}
        imu_by_arm[mode] = imu
        segments, epoch_records, literal = segment_arm(arm, imu)
        segments_by_arm[mode] = segments
        for segment in segments:
            segment["first_time_s"] = (int(frame_ns[segment["frames"][0]]) - origin) * 1e-9
            segment["last_time_s"] = (int(frame_ns[segment["frames"][-1]]) - origin) * 1e-9
            if "boundary" in segment:
                boundary = segment["boundary"]
                boundary["time_s"] = segment["first_time_s"]
                boundary["reference_increments"] = {}
                for name, ref in references.items():
                    pairs = association[name]["pairs"]
                    a, b = boundary["after_frame"], boundary["before_frame"]
                    if a in pairs and b in pairs:
                        truth = ref["T_W_I"][[pairs[a], pairs[b]]]
                        d, theta = increments(truth)
                        error_t, error_r = step_error((imu[a], imu[b]), truth)
                        boundary["reference_increments"][name] = {"translation_m": float(d[0]),
                                                                  "rotation_deg": float(theta[0]),
                                                                  "step_error_translation_m": error_t,
                                                                  "step_error_rotation_deg": error_r}
            segment["scores"] = {}
            for name, ref in references.items():
                kept, estimated, truth, ns = gather(segment["frames"], association[name]["pairs"], imu, ref, frame_ns)
                segment["scores"][name] = {}
                for alignment in alignments_for(mode, segment["label"]):
                    value, per_frame = score(estimated, truth, ns, alignment, origin)
                    segment["scores"][name][alignment] = value
                    if per_frame is not None:
                        per_frame_rows += [(mode, name, f"segment{segment['segment']}", alignment, i, int(t), e, r)
                                           for i, t, e, r in zip(kept, ns, per_frame["ate_m"], per_frame["rotation_deg"])]
        primary_cuts = {s["boundary"]["before_frame"] for s in segments if "boundary" in s}
        literal_summary = {"boundaries": len(literal), "coinciding_with_reference_rotation_over_2deg": {},
                           "coinciding_with_primary_boundaries": sum(hit["before_frame"] in primary_cuts
                                                                     for hit in literal),
                           "reference_rotation_at_non_primary_hits_deg": {},
                           "step_error_at_non_primary_hits": {}, "hits": literal}
        for name, ref in references.items():
            pairs = association[name]["pairs"]
            count, at_hits, errors = 0, [], []
            for hit in literal:
                a, b = hit["after_frame"], hit["before_frame"]
                if a in pairs and b in pairs:
                    truth = ref["T_W_I"][[pairs[a], pairs[b]]]
                    _, theta = increments(truth)
                    error = step_error((imu[a], imu[b]), truth)
                    hit.setdefault("reference_rotation_increment_deg", {})[name] = float(theta[0])
                    hit.setdefault("step_error_translation_m", {})[name] = error[0]
                    hit.setdefault("step_error_rotation_deg", {})[name] = error[1]
                    count += int(theta[0] > ROTATION_FLOOR_DEG)
                    if b not in primary_cuts:
                        at_hits.append(float(theta[0]))
                        errors.append(error)
            literal_summary["coinciding_with_reference_rotation_over_2deg"][name] = count
            literal_summary["reference_rotation_at_non_primary_hits_deg"][name] = (
                {"n": len(at_hits), "min": min(at_hits), "median": float(np.median(at_hits)), "max": max(at_hits)}
                if at_hits else None)
            worst = max((hit for hit in literal if hit["before_frame"] not in primary_cuts
                         and name in hit.get("step_error_translation_m", {})),
                        key=lambda hit: hit["step_error_translation_m"][name], default=None)
            literal_summary["step_error_at_non_primary_hits"][name] = (
                {"translation_m_max": max(e[0] for e in errors), "rotation_deg_max": max(e[1] for e in errors),
                 "largest_before_frame": worst["before_frame"],
                 "largest_time_s": (int(frame_ns[worst["before_frame"]]) - origin) * 1e-9}
                if errors else None)
        accepted = [f["index"] for f in arm["frames"] if f["tracked"] and (mode == "stereo" or f["imu_initialized"])]
        stitched = {}
        for name, ref in references.items():
            kept, estimated, truth, ns = gather(accepted, association[name]["pairs"], imu, ref, frame_ns)
            stitched[name] = {}
            for alignment in alignments_for(mode, "primary"):
                value, per_frame = score(estimated, truth, ns, alignment, origin)
                stitched[name][alignment] = value
                if per_frame is not None:
                    per_frame_rows += [(mode, name, "stitched", alignment, i, int(t), e, r)
                                       for i, t, e, r in zip(kept, ns, per_frame["ate_m"], per_frame["rotation_deg"])]
        primary_scored = {name: sorted({i for s in segments if s["label"] == "primary"
                                        for i in s["frames"]
                                        if s["scores"][name][alignments_for(mode, "primary")[0]]["status"] == "MEASURED"
                                        and i in association[name]["pairs"]}) for name in references}
        tracked = sum(f["tracked"] for f in arm["frames"])
        coverage = {"frames": FRAMES, "tracked_frames": tracked, "tracked_fraction": tracked / FRAMES,
                    "accepted_primary_frames": len(accepted), "accepted_primary_fraction": len(accepted) / FRAMES,
                    "accepted_primary_first_time_s": (int(frame_ns[accepted[0]]) - origin) * 1e-9 if accepted else None,
                    "accepted_with_reference": {n: sum(i in association[n]["pairs"] for i in accepted) for n in references},
                    "tracked_with_reference": {n: sum(f["tracked"] and f["index"] in association[n]["pairs"]
                                                      for f in arm["frames"]) for n in references},
                    "frames_in_scored_primary_segments": {n: len(v) for n, v in primary_scored.items()}}
        if mode == "stereo_inertial":
            coverage["imu_initialized_tracked_frames"] = len(accepted)
            coverage["tracked_before_imu_initialization"] = tracked - len(accepted)
        profile = step_error_profile(segments, imu, references, association, frame_ns, origin,
                                     {f["index"]: f["stereo_depth_matches"] for f in arm["frames"]})
        output_arms[mode] = {"coverage": coverage, "epochs": epoch_records, "segments": segments,
                             "one_frame_step_error_post_run": profile,
                             "literal_rule_sensitivity": literal_summary, "stitched": stitched,
                             "frames_sha256": arm["frames_sha256"], "native_summary": arm["native_summary"]}
    # Stereo re-scored on exactly the stereo-IMU timestamps.
    comparison = []
    stereo_poses = imu_by_arm["stereo"]
    for name, ref in references.items():
        pairs = association[name]["pairs"]
        windows = [(f"segment{s['segment']}", s["frames"], s["scores"][name])
                   for s in segments_by_arm["stereo_inertial"] if s["label"] == "primary"
                   and s["scores"][name]["se3"]["status"] == "MEASURED"]
        windows.append(("stitched", [i for s in segments_by_arm["stereo_inertial"] if s["label"] == "primary"
                                     for i in s["frames"]], output_arms["stereo_inertial"]["stitched"][name]))
        for label, frames, inertial in windows:
            kept, estimated, truth, ns = gather(frames, pairs, stereo_poses, ref, frame_ns)
            stereo_score, _ = score(estimated, truth, ns, "se3", origin)
            comparison.append({"window": label, "reference": name, "frames": len(kept),
                               "stereo_se3": stereo_score, "stereo_inertial_se3": inertial["se3"],
                               "stereo_inertial_gravity_yaw_translation": inertial["gravity_yaw_translation"]})
    # Diagnostics that never select anything.
    longest = max((s for s in segments_by_arm["stereo"] if s["scores"]["recomputed"]["se3"]["status"] == "MEASURED"),
                  key=lambda s: len(s["frames"]))
    alternatives = {}
    inverse = np.linalg.inv(t_i_c)
    for name, ref in references.items():
        pairs = association[name]["pairs"]
        kept = [i for i in longest["frames"] if i in pairs]
        truth = ref["T_W_I"][[pairs[i] for i in kept]]
        alternatives[name] = {}
        for key, convert in {"declared_T_W_C_inv_T_I_C": lambda t: t @ inverse,
                             "alternative_T_W_C_T_I_C": lambda t: t @ t_i_c,
                             "alternative_T_W_C_camera_as_body": lambda t: t}.items():
            estimated = np.stack([convert(arms["stereo"]["frames"][i]["T_W_C"]) for i in kept])
            value, _ = score(estimated, truth, frame_ns[kept], "se3", origin, internal=False)
            alternatives[name][key] = {"ate_rmse_m": value["ate_m"]["rmse"],
                                       "rotation_error_mean_deg": value["rotation_error_deg"]["mean"],
                                       "mean_body_frame_rotation_residual_deg": value["mean_body_frame_rotation_residual_deg"]}
    internal_values = [v["internal_score_trajectories_max_abs_difference"]
                       for arm in output_arms.values()
                       for block in [*(s["scores"] for s in arm["segments"]), arm["stitched"]]
                       for per_ref in block.values() for v in per_ref.values()
                       if v["status"] == "MEASURED"]
    internal_translation = max(v["ate_translation_m"] for v in internal_values)
    stereo_matches = [f["stereo_depth_matches"] for f in arms["stereo"]["frames"]]
    inertial_matches = [f["stereo_depth_matches"] for f in arms["stereo_inertial"]["frames"]]
    known = [m for m in stereo_matches if m is not None]
    front_end = {"field": "native diagnostics.positive_stereo_depth_matches per frame",
                 "identical_across_arms_all_frames": stereo_matches == inertial_matches,
                 "whole_run": ({"median": float(np.median(known)), "p5": float(np.percentile(known, 5)),
                                "min": min(known), "max": max(known)} if known else None)}
    validation = {
        "front_end_stereo_depth_matches": front_end,
        "T_imu_rectified_left_camera": {"max_abs_difference_vs_sensor_yaml": t_i_c_difference,
                                        "tolerance": T_I_C_TOLERANCE, "matrix": t_i_c.tolist()},
        "vicon": vicon_validation(dataset, references, window),
        "reference_vs_reference": reference_comparison(references["original"], references["recomputed"], window),
        "reference_offset_reproduction": None,
        "conversion_alternatives_stereo_se3": {"segment": longest["segment"], "references": alternatives,
                                               "note": "diagnostic only; the declared conversion was fixed before scoring"},
        "internal_score_trajectories": {"comparisons": len(internal_values),
                                        "ate_translation_max_abs_difference_m": internal_translation,
                                        "rotation_max_abs_difference_deg": max(v["rotation_deg"] for v in internal_values),
                                        "tolerance_m": INTERNAL_TOLERANCE,
                                        "agree": internal_translation <= INTERNAL_TOLERANCE},
    }
    both = [i for i in longest["frames"]
            if i in association["original"]["pairs"] and i in association["recomputed"]["pairs"]]
    offset = np.asarray(validation["reference_vs_reference"]["mean_offset_T_recomputed_original"])
    validation["reference_offset_reproduction"] = reference_offset_reproduction(
        np.stack([imu_by_arm["stereo"][i] for i in both]),
        references["recomputed"]["T_W_I"][[association["recomputed"]["pairs"][i] for i in both]],
        references["original"]["T_W_I"][[association["original"]["pairs"][i] for i in both]],
        frame_ns[both], offset, origin)
    evo = {"status": "NOT_RUN", "reason": "no evo executables supplied"}
    if evo_ape and evo_rpe:
        kept = [i for i in longest["frames"] if i in imu_by_arm["stereo"]]
        evo = evo_crosscheck(evo_ape, evo_rpe, output_dir / "evo", frame_ns[kept],
                             np.stack([imu_by_arm["stereo"][i] for i in kept]), references,
                             {n: longest["scores"][n]["se3"] for n in references}, "stereo")
        evo.update(status="RUN", arm="stereo", segment=longest["segment"])
    with (output_dir / "per_frame_errors.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "reference", "scope", "alignment", "frame_index", "timestamp_ns",
                         "ate_m", "rotation_error_deg"])
        writer.writerows((a, r, s, al, i, t, f"{e:.9f}", f"{d:.9f}") for a, r, s, al, i, t, e, d in per_frame_rows)
    for arm in output_arms.values():
        for segment in arm["segments"]:
            frames = segment.pop("frames")
            if frames != list(range(frames[0], frames[-1] + 1)):
                raise ValueError("Segments must be contiguous runs of frame indices")
            segment["frame_range"], segment["frame_count"] = [frames[0], frames[-1]], len(frames)
    value = {
        "schema": "bhl-euroc-accuracy-results-v1", "status": "COMPLETED", "evaluated_utc": utc_now(),
        "protocol": protocol, "evaluator_sha256": sha256(Path(__file__)),
        "pose_metrics_sha256": sha256(ROOT / "src/bhl_robust/research/pose_metrics.py"),
        "sequence": "EuRoC MAV V1_01_easy", "window": {
            "frames": FRAMES, "first_timestamp_ns": window[0], "last_timestamp_ns": window[1],
            "clock_origin_ns": origin, "last_relative_s": (window[1] - origin) * 1e-9},
        "inputs": {"replay": str(replay), "replay_result_sha256": sha256(replay / "result.json"),
                   "rectification_sha256": sha256(replay / "rectification.json"), "dataset": str(dataset),
                   "references": {n: {k: v for k, v in r.items() if k not in ("timestamps_ns", "T_W_I")}
                                  | {"pinned_git_blob_sha1": {"original": original_blob,
                                                              "recomputed": recomputed_blob}[n]}
                                  for n, r in references.items()},
                   "recomputed_source": {"url": RECOMPUTED_URL, "introduced_in_commit": OPENVINS_INTRODUCED_COMMIT,
                                         "license": "GPL-3.0 (rpng/open_vins repository); read in place, not copied"}},
        "estimate_frame": PROTOCOL["estimate_frame"],
        "association": {n: a["report"] for n, a in association.items()},
        "headline": headline(output_arms, comparison),
        "arms": output_arms, "comparison_identical_timestamps": comparison,
        "validation": validation, "evo_crosscheck": evo,
        "per_frame_errors_csv": str(output_dir / "per_frame_errors.csv"),
        "truth_inputs_to_estimator": False, "own_hardware_validation": False,
    }
    with results_path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    (output_dir / "REPORT.md").write_text(render_report(value))
    return value


def _brief(value):
    if value is None or value.get("status") != "MEASURED":
        return {"status": None if value is None else value.get("status")}
    rpe = {k: value["rpe"][k] for k in ("1s", "5s", "10s")}
    return {"frames": value["associated_frames"], "time_s": [value["first_time_s"], value["last_time_s"]],
            "ate_rmse_m": value["ate_m"]["rmse"], "ate_mean_m": value["ate_m"]["mean"],
            "ate_median_m": value["ate_m"]["median"], "ate_max_m": value["ate_m"]["max"],
            "rotation_mean_deg": value["rotation_error_deg"]["mean"], "rotation_rmse_deg": value["rotation_error_deg"]["rmse"],
            "alignment_tilt_deg": value["alignment_tilt_deg"],
            **{f"rpe_{k}_translation_rmse_m": (r["translation_m"]["rmse"] if r["status"] == "MEASURED" else None)
               for k, r in rpe.items()},
            **{f"rpe_{k}_rotation_rmse_deg": (r["rotation_deg"]["rmse"] if r["status"] == "MEASURED" else None)
               for k, r in rpe.items()},
            **{f"drift_{k}_mean_percent": (r["drift_percent"]["mean"] if r["status"] == "MEASURED" and r["drift_percent"]
                                           else None) for k, r in rpe.items()}}


def headline(arms: dict, comparison: list) -> dict:
    """Compact copy of the primary numbers (every value also appears in full elsewhere)."""
    value = {}
    for mode, arm in arms.items():
        value[mode] = {"coverage": arm["coverage"], "segments": {}, "stitched_secondary": {}}
        for segment in arm["segments"]:
            key = f"segment{segment['segment']}_{segment['label']}"
            value[mode]["segments"][key] = {ref: {al: _brief(v) for al, v in per.items()}
                                            for ref, per in segment["scores"].items()}
        value[mode]["stitched_secondary"] = {ref: {al: _brief(v) for al, v in per.items()}
                                             for ref, per in arm["stitched"].items()}
    value["identical_timestamps"] = [{"window": row["window"], "reference": row["reference"], "frames": row["frames"],
                                      "stereo_se3": _brief(row["stereo_se3"]),
                                      "stereo_inertial_se3": _brief(row["stereo_inertial_se3"]),
                                      "stereo_inertial_gravity_yaw_translation":
                                          _brief(row["stereo_inertial_gravity_yaw_translation"])}
                                     for row in comparison]
    return value


# ------------------------------------------------------------------ report

def _m(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _rpe_cell(rpe):
    if rpe["status"] != "MEASURED":
        return "no pairs"
    drift = rpe["drift_percent"]["mean"] if rpe["drift_percent"] else None
    return f"{_m(rpe['translation_m']['rmse'])} / {_m(rpe['rotation_deg']['rmse'], 2)} / {_m(drift, 2)}"


def _headline(score_value):
    """Frames, ATE RMSE/mean/median/max, rotation mean/RMSE, then RPE t / r / drift at 1, 5, 10 s."""
    if score_value is None or score_value.get("status") != "MEASURED":
        return None
    ate, rot = score_value["ate_m"], score_value["rotation_error_deg"]
    return ([str(score_value["associated_frames"]), _m(ate["rmse"]), _m(ate["mean"]), _m(ate["median"]),
             _m(ate["max"]), _m(rot["mean"], 2), _m(rot["rmse"], 2)]
            + [_rpe_cell(score_value["rpe"][k]) for k in ("1s", "5s", "10s")])


ACCURACY_HEADER = ("| arm | scope | alignment | frames | ATE RMSE | ATE mean | ATE median | ATE max | rot mean | rot RMSE "
                   "| 1 s: RPE t / r / drift | 5 s: RPE t / r / drift | 10 s: RPE t / r / drift |\n"
                   "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
ALIGNMENT_LABEL = {"se3": "SE3", "gravity_yaw_translation": "4-DoF"}
ARM_LABEL = {"stereo": "stereo", "stereo_inertial": "stereo-IMU"}


def _primary_segments(arm):
    return [s for s in arm["segments"] if s["label"] == "primary"
            and all(v["status"] == "MEASURED" for per in s["scores"].values() for v in per.values())]


def _derived(results):
    """Ranges quoted in the prose, computed so the text cannot drift from the numbers."""
    validation = results["validation"]
    alternatives = validation["conversion_alternatives_stereo_se3"]["references"]
    wrong = [v["mean_body_frame_rotation_residual_deg"] for block in alternatives.values()
             for key, v in block.items() if key != "declared_T_W_C_inv_T_I_C"]
    vicon = validation["vicon"]["references"]
    declared = [b["candidates"]["declared_T_W_V_inv_T_BS"] for b in vicon.values()]
    declared_aligned = [b["candidates_after_se3_world_alignment"]["declared_T_W_V_inv_T_BS"] for b in vicon.values()]
    marker = [b["candidates"]["alternative_T_W_V_marker_as_body"] for b in vicon.values()]
    marker_aligned = [b["candidates_after_se3_world_alignment"]["alternative_T_W_V_marker_as_body"] for b in vicon.values()]
    windows = [r for r in results["comparison_identical_timestamps"] if r["window"] != "stitched"]
    gaps = [abs(r["stereo_se3"]["ate_m"]["rmse"] - r[k]["ate_m"]["rmse"]) for r in windows
            for k in ("stereo_inertial_se3", "stereo_inertial_gravity_yaw_translation")]
    stereo = _primary_segments(results["arms"]["stereo"])[0]["scores"]["recomputed"]["se3"]["rpe"]
    flips = []
    for label in sorted({r["window"] for r in windows}):
        signs = {r["reference"]: np.sign(r["stereo_se3"]["ate_m"]["rmse"] - r["stereo_inertial_se3"]["ate_m"]["rmse"])
                 for r in windows if r["window"] == label}
        if len(set(signs.values())) > 1:
            flips.append(label)
    ordering_text = (f"the ordering of the two arms flips between references on {', '.join(flips)}, so these "
                     "millimetre gaps are not a ranking" if flips else
                     "the ordering of the two arms is the same under both references")
    longest = max((r for r in windows if r["reference"] == "recomputed"), key=lambda r: r["frames"])
    a, b = longest["stereo_se3"], longest["stereo_inertial_se3"]
    rotation_text = (f"On the longest common window ({longest['window']}, recomputed) the rotation RPE RMSE is "
                     f"{a['rpe']['1s']['rotation_deg']['rmse']:.2f} (stereo) vs {b['rpe']['1s']['rotation_deg']['rmse']:.2f} deg "
                     f"(stereo-IMU) at 1 s and {a['rpe']['10s']['rotation_deg']['rmse']:.2f} vs "
                     f"{b['rpe']['10s']['rotation_deg']['rmse']:.2f} deg at 10 s, while the absolute rotation error mean is "
                     f"{a['rotation_error_deg']['mean']:.2f} vs {b['rotation_error_deg']['mean']:.2f} deg.")
    inertial = [s["scores"] for s in _primary_segments(results["arms"]["stereo_inertial"])]
    smooth = {m: results["arms"][m]["one_frame_step_error_post_run"]["recomputed"] for m in ("stereo", "stereo_inertial")}
    stretches = [c for c in smooth["stereo"]["clusters_over_2cm_within_1s"] if c["steps"] >= 5]
    smoothness_text = (f"Frame to frame (post-run diagnostic, jumps excluded), the one-frame step error median / p99 is "
                       f"{1000 * smooth['stereo']['translation_m']['median']:.1f} / {1000 * smooth['stereo']['translation_m']['p99']:.1f} mm "
                       f"for stereo vs {1000 * smooth['stereo_inertial']['translation_m']['median']:.1f} / "
                       f"{1000 * smooth['stereo_inertial']['translation_m']['p99']:.1f} mm for stereo-IMU"
                       + (", and stereo has " + ", ".join(f"a {c['first_time_s']:.1f}-{c['last_time_s']:.1f} s stretch of "
                                                          f"{c['steps']} steps above 2 cm (max {100 * c['max_translation_m']:.1f} cm)"
                                                          for c in stretches)
                          + " that stays below the discontinuity thresholds." if stretches else "."))
    return {"ordering_text": ordering_text, "rotation_text": rotation_text, "ordering_flips": bool(flips),
            "smoothness_text": smoothness_text,
            "inertial_body_recomputed": [x["recomputed"]["se3"]["mean_body_frame_rotation_residual_deg"] for x in inertial],
            "inertial_body_original": [x["original"]["se3"]["mean_body_frame_rotation_residual_deg"] for x in inertial],
            "wrong_min": min(wrong), "wrong_max": max(wrong),
            "vicon_rot_min": min(v["mean_body_rotation_residual_deg"] for v in declared_aligned),
            "vicon_rot_max": max(v["mean_body_rotation_residual_deg"] for v in declared_aligned),
            "vicon_pos_min": min(v["position_error_m"]["rmse"] for v in declared_aligned),
            "vicon_pos_max": max(v["position_error_m"]["rmse"] for v in declared_aligned),
            "vicon_world_rot_max": max(v["world_alignment_rotation_deg"] for v in declared_aligned),
            "vicon_unaligned_rot_min": min(v["mean_residual_rotation_deg"] for v in declared),
            "marker_rot_min": min(v["mean_residual_rotation_deg"] for v in marker),
            "marker_pos_min": min(v["position_error_m"]["rmse"] for v in marker_aligned),
            "arm_gap_max_mm": 1000 * max(gaps) if gaps else None,
            "stereo_drift_1s": stereo["1s"]["drift_percent"]["mean"],
            "stereo_drift_10s": stereo["10s"]["drift_percent"]["mean"]}


def _glance(results):
    arms, validation = results["arms"], results["validation"]
    stereo, inertial = arms["stereo"], arms["stereo_inertial"]
    d = _derived(results)
    lines = []
    s0 = _primary_segments(stereo)
    for segment in s0:
        rec, ori = segment["scores"]["recomputed"]["se3"], segment["scores"]["original"]["se3"]
        lines.append(
            f"- **Stereo** tracked {stereo['coverage']['tracked_frames']}/1200 frames as one continuous segment "
            f"({segment['first_time_s']:.2f}-{segment['last_time_s']:.2f} s, no map change or discontinuity). SE3, scale 1: "
            f"ATE RMSE {_m(rec['ate_m']['rmse'])} m (max {_m(rec['ate_m']['max'])} m), rotation error mean "
            f"{_m(rec['rotation_error_deg']['mean'], 2)} deg against the recomputed reference; "
            f"{_m(ori['ate_m']['rmse'])} m and {_m(ori['rotation_error_deg']['mean'], 2)} deg against the original. "
            f"Drift over 10 s intervals {_m(rec['rpe']['10s']['drift_percent']['mean'], 2)} % (recomputed) / "
            f"{_m(ori['rpe']['10s']['drift_percent']['mean'], 2)} % (original) of distance travelled.")
    cov = inertial["coverage"]
    jumps = [s["boundary"] for s in inertial["segments"] if "boundary" in s]
    jump_text = "; ".join(f"{b['time_s']:.2f} s: {b['estimate_translation_increment_m']:.3f} m / "
                          f"{b['estimate_rotation_increment_deg']:.1f} deg in one frame where the reference moved "
                          f"{b['reference_increments']['recomputed']['translation_m']:.3f} m / "
                          f"{b['reference_increments']['recomputed']['rotation_deg']:.1f} deg" for b in jumps)
    lines.append(
        f"- **Stereo-IMU** tracked {cov['tracked_frames']}/1200 frames; {cov['imu_initialized_tracked_frames']}/1200 are "
        f"IMU-initialized (from {cov['accepted_primary_first_time_s']:.2f} s). Its causal trajectory jumps with the map id "
        f"unchanged ({jump_text}), so the primary numbers are per segment:")
    for segment in _primary_segments(inertial):
        rec, ori = segment["scores"]["recomputed"], segment["scores"]["original"]
        lines.append(
            f"  - {segment['first_time_s']:.2f}-{segment['last_time_s']:.2f} s ({segment['frame_count']} frames): ATE RMSE "
            f"{_m(rec['se3']['ate_m']['rmse'])} m SE3 / {_m(rec['gravity_yaw_translation']['ate_m']['rmse'])} m 4-DoF, "
            f"rotation mean {_m(rec['se3']['rotation_error_deg']['mean'], 2)} / "
            f"{_m(rec['gravity_yaw_translation']['rotation_error_deg']['mean'], 2)} deg (recomputed); "
            f"{_m(ori['se3']['ate_m']['rmse'])} / {_m(ori['gravity_yaw_translation']['ate_m']['rmse'])} m and "
            f"{_m(ori['se3']['rotation_error_deg']['mean'], 2)} / {_m(ori['gravity_yaw_translation']['rotation_error_deg']['mean'], 2)} deg "
            f"(original). SE3 alignment tilt {rec['se3']['alignment_tilt_deg']:.2f} deg (recomputed), "
            f"{ori['se3']['alignment_tilt_deg']:.2f} deg (original): gravity-consistent.")
    st = inertial["stitched"]["recomputed"]
    lines.append(f"  - Stitched whole (secondary, one alignment across the jumps): ATE RMSE {_m(st['se3']['ate_m']['rmse'])} m SE3 / "
                 f"{_m(st['gravity_yaw_translation']['ate_m']['rmse'])} m 4-DoF against the recomputed reference.")
    rows = [r for r in results["comparison_identical_timestamps"] if r["reference"] == "recomputed"]
    parts = [f"{r['window']} ({r['frames']} frames): stereo {_m(r['stereo_se3']['ate_m']['rmse'])} m vs stereo-IMU "
             f"{_m(r['stereo_inertial_se3']['ate_m']['rmse'])} m SE3 / "
             f"{_m(r['stereo_inertial_gravity_yaw_translation']['ate_m']['rmse'])} m 4-DoF" for r in rows]
    lines.append("- **Same timestamps** (recomputed reference, ATE RMSE): " + "; ".join(parts) + ". Between the jumps the "
                 f"two arms differ by at most {d['arm_gap_max_mm']:.1f} mm in ATE RMSE (either reference, either "
                 f"alignment), and {d['ordering_text']}; across the whole initialized window the stereo-IMU jumps dominate. "
                 f"{d['rotation_text']} {d['smoothness_text']}")
    rr = validation["reference_vs_reference"]["evaluation_window"]
    rep = validation["reference_offset_reproduction"]
    lines.append(
        f"- **References disagree by a constant body-frame offset**: inv(T_recomputed) T_original has mean rotation "
        f"{rr['mean_residual_rotation_deg']:.2f} deg and translation "
        f"{np.linalg.norm(rr['mean_residual_translation_body_m']):.4f} m (std below 2 mm). A rigid world alignment cannot "
        f"remove it, so it sets a ~{rr['mean_residual_rotation_deg']:.1f} deg floor on rotation error against the original "
        f"and rotates every relative displacement, adding about {rep['relative_translation_error_from_offset_rotation_percent']:.1f} % "
        f"of distance travelled to RPE. Applying the offset to the recomputed reference reproduces the original-reference "
        f"scores (stereo ATE RMSE {_m(rep['scores']['recomputed_times_constant_offset']['ate_rmse_m'])} vs "
        f"{_m(rep['scores']['original']['ate_rmse_m'])} m, 1 s drift "
        f"{_m(rep['scores']['recomputed_times_constant_offset']['drift_1s_mean_percent'], 2)} vs "
        f"{_m(rep['scores']['original']['drift_1s_mean_percent'], 2)} %).")
    alternatives = validation["conversion_alternatives_stereo_se3"]["references"]
    lines.append(
        f"- **Frame checks**: the declared IMU conversion leaves a mean body-frame rotation residual of "
        f"{alternatives['recomputed']['declared_T_W_C_inv_T_I_C']['mean_body_frame_rotation_residual_deg']:.2f} deg against "
        f"the recomputed reference ({alternatives['original']['declared_T_W_C_inv_T_I_C']['mean_body_frame_rotation_residual_deg']:.2f} deg "
        f"against the original); wrong conversions leave {d['wrong_min']:.0f}-{d['wrong_max']:.0f} deg. Vicon marker poses "
        f"composed with inv(T_BS) land within {d['vicon_rot_min']:.1f}-{d['vicon_rot_max']:.1f} deg and "
        f"{100 * d['vicon_pos_min']:.1f}-{100 * d['vicon_pos_max']:.1f} cm of both references after a world alignment, while "
        f"treating the marker as the body leaves at least {d['marker_rot_min']:.0f} deg, so both references are IMU-frame "
        f"poses. evo agrees with these numbers to at most "
        f"{max(max(c['abs_difference'].values()) for b in results['evo_crosscheck'].get('references', {}).values() for c in b['checks'].values()):.0e} "
        f"(m or deg)." if results["evo_crosscheck"].get("status") == "RUN" else "")
    return lines


def render_report(results: dict) -> str:
    arms, validation = results["arms"], results["validation"]
    lines = [
        "# EuRoC V1_01_easy pose accuracy: native ORB-SLAM3 stereo vs stereo-IMU (first 60 s)",
        "",
        f"Evaluated {results['evaluated_utc']} by `scripts/bench/euroc_native_evaluate.py` "
        f"(sha256 `{results['evaluator_sha256'][:16]}`). The protocol was frozen at {results['protocol']['frozen_utc']}, "
        f"before any score was computed (`protocol.json` sha256 `{results['protocol']['sha256'][:16]}`). "
        "Full numbers are in `results.json` and per-frame errors in `per_frame_errors.csv`.",
        "",
        "Units: positions and ATE in metres (m), rotations in degrees (deg), drift in percent of reference path "
        "length. Estimates are the causal per-frame native ORB-SLAM3 poses converted to the IMU frame, "
        "T_W_I = T_W_Crect inv(T_I_Crect), and associated to reference samples without interpolation. Scale is "
        "fixed at 1.",
        "",
        "## At a glance",
        "",
        *[line for line in _glance(results) if line],
        "",
        "## References",
        "",
        "| reference | file | git blob (pinned, verified) | rate | association to camera stamps | frames with reference |",
        "|---|---|---|---|---|---|",
    ]
    for name, description in (("original", "EuRoC `state_groundtruth_estimate0/data.csv` (on disk)"),
                              ("recomputed", "OpenVINS `ov_data/euroc_mav/V1_01_easy.csv` (vicon2gt; GPL-3.0 repository)")):
        ref, assoc = results["inputs"]["references"][name], results["association"][name]
        offsets = ", ".join(f"{count} at {offset} ns" for offset, count in assoc["offset_ns_histogram"].items())
        tolerance = assoc["tolerance_ns"]
        tolerance_text = f"{tolerance / 1e6:g} ms" if tolerance >= 1_000_000 else f"{tolerance} ns"
        lines.append(f"| {name} | {description} | `{ref['git_blob_sha1']}` | {1 / ref['median_period_s']:.0f} Hz | "
                     f"nearest within {tolerance_text}; achieved max {assoc['max_abs_offset_ns']} ns "
                     f"({offsets}) | {assoc['associated_frames']}/1200 |")
    lines += ["",
              "The recomputed file was written through double precision: 240 of its 1200 window stamps sit 256 ns "
              "(one float64 ulp at 1.4e18 ns) before the camera stamp. 'Exact nanosecond' association therefore uses "
              "a declared 256 ns tolerance; 960 stamps match exactly. The original reference starts 1.04 s after the "
              "first camera frame, so frames 0-20 (0.00-1.00 s) have no original reference. The recomputed file was "
              f"downloaded from `{results['inputs']['recomputed_source']['url']}` (introduced in OpenVINS commit "
              f"`{results['inputs']['recomputed_source']['introduced_in_commit'][:12]}` together with the "
              "`V1_01_easy_original.csv` copy of the EuRoC file and the documentation note on its orientation).",
              "",
              "## Predeclared segmentation rule",
              "",
              f"- Rule as tasked: {PROTOCOL['segmentation']['literal_rule_as_tasked']}.",
              f"- {PROTOCOL['segmentation']['amendment_evidence_non_score']}",
              f"- Primary rule (frozen): {PROTOCOL['segmentation']['primary_rule']}",
              f"- Blind spot: {PROTOCOL['segmentation']['blind_spot']}.",
              f"- Segments need at least {MIN_SEGMENT_FRAMES} reference-associated frames to be scored. RPE pairs use "
              "the frame nearest t + 1, 5 and 10 s (within 25 ms) in the same segment. Drift is RPE translation divided "
              f"by the reference path over the pair; pairs with a path under {DRIFT_MIN_PATH_M} m are excluded.",
              "- Alignment per segment: SE3 for stereo; SE3 and 4-DoF (yaw about gravity + translation) for the "
              "IMU-initialized stereo-IMU segments. The stitched whole (one alignment over all accepted frames) is "
              "secondary.",
              "",
              "| arm | epoch | frames | translation threshold | rotation threshold | primary boundaries | literal-rule boundaries |",
              "|---|---|---:|---:|---:|---|---:|"]
    for mode, arm in arms.items():
        for epoch in arm["epochs"]:
            lines.append(f"| {ARM_LABEL[mode]} | {epoch['epoch']} ({epoch['label']}, frames {epoch['first_frame']}-"
                         f"{epoch['last_frame']}) | {epoch['frames']} | {_m(epoch['translation_threshold_m'])} m | "
                         f"{_m(epoch['rotation_threshold_deg'], 2)} deg | before frames "
                         f"{epoch['primary_boundaries_before_frames'] or 'none'} | {epoch['literal_rule_boundaries']} |")
    lines.append("")
    lines.append("Exact per-frame-pair check of the frozen evidence statement (which counted literal-rule hits by "
                 "fast-turn episode):")
    lines.append("")
    for mode, arm in arms.items():
        literal = arm["literal_rule_sensitivity"]
        at_hits = literal["reference_rotation_at_non_primary_hits_deg"]["recomputed"]
        errors = literal["step_error_at_non_primary_hits"]["recomputed"]
        lines.append(f"- {ARM_LABEL[mode]}: {literal['boundaries']} literal-rule boundaries, "
                     f"{literal['coinciding_with_primary_boundaries']} of them primary boundaries. Over the same frame pairs "
                     f"the reference itself rotates more than 2 deg at {literal['coinciding_with_reference_rotation_over_2deg']['original']} "
                     f"(original) / {literal['coinciding_with_reference_rotation_over_2deg']['recomputed']} (recomputed). "
                     + (f"At the {at_hits['n']} non-primary hits the recomputed reference rotates "
                        f"{at_hits['min']:.2f}-{at_hits['max']:.2f} deg per frame (median {at_hits['median']:.2f}), so they "
                        "sit in genuine fast turns; the estimate step differs from the reference step by at most "
                        f"{errors['translation_m_max']:.4f} m / {errors['rotation_deg_max']:.2f} deg (frame "
                        f"{errors['largest_before_frame']}, {errors['largest_time_s']:.2f} s)"
                        + ("; that hit lies in the stretch of large stereo frame-to-frame errors described under "
                           "'Frame-to-frame smoothness'." if errors["translation_m_max"] > 0.02 else
                           ", i.e. the turns were tracked step for step.")
                        if at_hits else "There are no non-primary hits."))
    lines += ["", "Boundaries that fired (estimate step vs reference step over the same frame pair; step error = "
              "one-frame relative pose error):", "",
              "| arm | before frame | time | estimate step | original reference step | recomputed reference step | "
              "step error vs recomputed |",
              "|---|---:|---:|---|---|---|---|"]
    for mode, arm in arms.items():
        for segment in arm["segments"]:
            boundary = segment.get("boundary")
            if not boundary:
                continue
            refs = boundary["reference_increments"]
            cell = lambda n: (f"{refs[n]['translation_m']:.4f} m / {refs[n]['rotation_deg']:.2f} deg"
                              if n in refs else "n/a")
            error = refs.get("recomputed")
            lines.append(f"| {ARM_LABEL[mode]} | {boundary['before_frame']} | {boundary['time_s']:.2f} s | "
                         f"{boundary['estimate_translation_increment_m']:.4f} m / "
                         f"{boundary['estimate_rotation_increment_deg']:.2f} deg | {cell('original')} | "
                         f"{cell('recomputed')} | "
                         + (f"{error['step_error_translation_m']:.4f} m / {error['step_error_rotation_deg']:.2f} deg |"
                            if error else "n/a |"))
    lines += ["", "## Coverage", "",
              "| arm | tracked / 1200 | accepted (IMU-initialized for stereo-IMU) | first accepted | "
              "accepted with original / recomputed reference | in scored primary segments (original / recomputed) |",
              "|---|---:|---:|---:|---|---|"]
    for mode, arm in arms.items():
        c = arm["coverage"]
        lines.append(f"| {ARM_LABEL[mode]} | {c['tracked_frames']} ({100 * c['tracked_fraction']:.1f}%) | "
                     f"{c['accepted_primary_frames']} ({100 * c['accepted_primary_fraction']:.1f}%) | "
                     f"{c['accepted_primary_first_time_s']:.2f} s | {c['accepted_with_reference']['original']} / "
                     f"{c['accepted_with_reference']['recomputed']} | "
                     f"{c['frames_in_scored_primary_segments']['original']} / "
                     f"{c['frames_in_scored_primary_segments']['recomputed']} |")
    lines += ["", "| arm | segment | label | frames | time | starts at | scored |", "|---|---:|---|---:|---|---|---|"]
    for mode, arm in arms.items():
        for s in arm["segments"]:
            statuses = sorted({v["status"] for per in s["scores"].values() for v in per.values()})
            lines.append(f"| {ARM_LABEL[mode]} | {s['segment']} | {s['label']} | {s['frame_count']} | "
                         f"{s['first_time_s']:.2f}-{s['last_time_s']:.2f} s | {s['start_reason']} | "
                         f"{', '.join(statuses)} |")
    lines += ["", "The pre-initialization stereo-IMU segment is scored for completeness only (SE3, not gravity-aligned) "
              "and is not part of the stereo-IMU numbers."]
    for name in ("recomputed", "original"):
        lines += ["", f"## Accuracy per segment vs the {name} reference", "",
                  "RPE cells: translation RMSE (m) / rotation RMSE (deg) / mean drift (% of distance travelled).", "",
                  ACCURACY_HEADER]
        for mode, arm in arms.items():
            for s in arm["segments"]:
                for alignment, value in s["scores"][name].items():
                    cells = _headline(value)
                    if cells:
                        lines.append(f"| {ARM_LABEL[mode]} | segment {s['segment']} ({s['label']}, "
                                     f"{s['first_time_s']:.2f}-{s['last_time_s']:.2f} s) | "
                                     f"{ALIGNMENT_LABEL[alignment]} | " + " | ".join(cells) + " |")
            single = len(arm["segments"]) == 1
            for alignment, value in arm["stitched"][name].items():
                cells = _headline(value)
                if cells:
                    scope = "stitched whole (same frames as segment 0)" if single else "stitched whole (secondary)"
                    lines.append(f"| {ARM_LABEL[mode]} | {scope} | {ALIGNMENT_LABEL[alignment]} | " + " | ".join(cells) + " |")
        tilts = [f"segment {s['segment']} {s['scores'][name]['se3']['alignment_tilt_deg']:.2f} deg"
                 for s in _primary_segments(arms["stereo_inertial"])]
        lines += ["", f"Stereo-IMU SE3 alignment tilt (angle between the aligned estimator z axis and the reference z "
                  f"axis): {', '.join(tilts)}. The small tilt is why 4-DoF and SE3 ATE nearly coincide."]
    lines += ["", "## Stereo vs stereo-IMU on identical timestamps", "",
              "The stereo arm is re-scored with its own SE3 alignment on exactly the frames of each scored stereo-IMU "
              "segment, so both arms see the same time window and the same reference samples.", "",
              "| window | reference | arm, alignment | frames | ATE RMSE | ATE max | rot mean | rot RMSE | 1 s: RPE t / r / drift "
              "| 5 s: RPE t / r / drift | 10 s: RPE t / r / drift |",
              "|---|---|---|---:|---:|---:|---:|---:|---|---|---|"]
    for row in sorted(results["comparison_identical_timestamps"], key=lambda r: (r["reference"] != "recomputed",)):
        for label, key in (("stereo, SE3", "stereo_se3"), ("stereo-IMU, SE3", "stereo_inertial_se3"),
                           ("stereo-IMU, 4-DoF", "stereo_inertial_gravity_yaw_translation")):
            v = row[key]
            window = f"{row['window']} ({v['first_time_s']:.2f}-{v['last_time_s']:.2f} s)"
            lines.append(f"| {window} | {row['reference']} | {label} | {v['associated_frames']} | {_m(v['ate_m']['rmse'])} | "
                         f"{_m(v['ate_m']['max'])} | {_m(v['rotation_error_deg']['mean'], 2)} | "
                         f"{_m(v['rotation_error_deg']['rmse'], 2)} | "
                         + " | ".join(_rpe_cell(v["rpe"][k]) for k in ("1s", "5s", "10s")) + " |")
    lines += ["", "## Frame-to-frame smoothness (post-run diagnostic)", "",
              "One-frame relative pose error (RPE over 50 ms) between consecutive frames inside the scored primary "
              "segments; segment boundaries, i.e. the two stereo-IMU jumps, are excluded. This was added after the "
              "first run and is not part of the predeclared metrics.", "",
              "| arm | reference | pairs | translation median / p99 / max (m) | rotation median / p99 / max (deg) | "
              "steps > 2 cm (clusters) |",
              "|---|---|---:|---|---|---|"]
    for mode, arm in arms.items():
        for name in ("recomputed", "original"):
            v = arm["one_frame_step_error_post_run"][name]
            if not v:
                continue
            clusters = "; ".join(f"{c['first_time_s']:.2f}-{c['last_time_s']:.2f} s: {c['steps']} "
                                 f"step{'s' if c['steps'] != 1 else ''}, max {c['max_translation_m']:.4f} m"
                                 for c in v["clusters_over_2cm_within_1s"])
            lines.append(f"| {ARM_LABEL[mode]} | {name} | {v['pairs']} | {v['translation_m']['median']:.4f} / "
                         f"{v['translation_m']['p99']:.4f} / {v['translation_m']['max']:.4f} | "
                         f"{v['rotation_deg']['median']:.3f} / {v['rotation_deg']['p99']:.3f} / "
                         f"{v['rotation_deg']['max']:.3f} | {v['steps_over_2cm']}" + (f" ({clusters})" if clusters else "") + " |")
    front_end = validation["front_end_stereo_depth_matches"]
    stretches = [c for c in arms["stereo"]["one_frame_step_error_post_run"]["recomputed"]["clusters_over_2cm_within_1s"]
                 if c["steps"] >= 5 and c.get("front_end_stereo_depth_matches")]
    lines += ["", "The stereo arm's large steps stay below the frozen discontinuity thresholds, so they remain inside "
              "its single segment and count toward its ATE and RPE."
              + "".join(f" During {c['first_time_s']:.2f}-{c['last_time_s']:.2f} s the native front end reported "
                        f"{c['front_end_stereo_depth_matches']['min']}-{c['front_end_stereo_depth_matches']['max']} "
                        f"stereo depth matches per frame (whole run: median {front_end['whole_run']['median']:.0f}, "
                        f"5th percentile {front_end['whole_run']['p5']:.0f})." for c in stretches)
              + ("" if front_end["whole_run"] is None else
                 " The per-frame match counts are identical in both arms over all 1200 frames, so the input to the two "
                 "estimators was the same and the stereo-only estimator produced the large steps."
                 if front_end["identical_across_arms_all_frames"] else
                 " The per-frame match counts differ between arms.")]
    rr = validation["reference_vs_reference"]
    rep = validation["reference_offset_reproduction"]
    window, inverse, aligned = rr["evaluation_window"], rr["inverse_direction_window"], rr["original_after_se3_world_alignment_window"]
    lines += ["", "## Why the two references give different numbers", "",
              f"- inv(T_recomputed) T_original over the window ({window['samples']} samples): mean rotation "
              f"{window['mean_residual_rotation_deg']:.2f} deg (spread about the mean: median "
              f"{window['rotation_spread_about_mean_deg']['median']:.2f}, max {window['rotation_spread_about_mean_deg']['max']:.2f} deg), "
              f"body-frame translation ({', '.join(f'{v:.4f}' for v in window['mean_residual_translation_body_m'])}) m with std "
              f"({', '.join(f'{v:.4f}' for v in window['residual_translation_body_std_m'])}) m. The inverse direction, "
              f"inv(T_original) T_recomputed, has translation ({', '.join(f'{v:.4f}' for v in inverse['mean_residual_translation_body_m'])}) m "
              f"in the original body frame. Over the full overlap ({rr['full_overlap']['samples']} samples) the mean rotation is "
              f"{rr['full_overlap']['mean_residual_rotation_deg']:.2f} deg.",
              f"- An SE3 world alignment of the original onto the recomputed reference rotates the world by only "
              f"{aligned['world_alignment_rotation_deg']:.2f} deg and leaves a {aligned['mean_body_rotation_residual_deg']:.2f} deg "
              "body-frame rotation: the references share a world frame and disagree about the body (IMU) orientation.",
              f"- Post-run diagnostic (stereo, SE3, the {rep['frames']} frames with both references): scoring against the recomputed "
              "reference right-multiplied by the constant mean offset reproduces the original-reference scores.", "",
              "| scored against | ATE RMSE (m) | rot mean (deg) | RPE 1 s t (m) | RPE 5 s t (m) | RPE 10 s t (m) | drift 1 s (%) | drift 5 s (%) | drift 10 s (%) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label in (("recomputed", "recomputed"), ("original", "original"),
                       ("recomputed_times_constant_offset", "recomputed x constant offset")):
        v = rep["scores"][key]
        lines.append(f"| {label} | {_m(v['ate_rmse_m'])} | {_m(v['rotation_error_mean_deg'], 2)} | "
                     f"{_m(v['rpe_1s_translation_rmse_m'])} | {_m(v['rpe_5s_translation_rmse_m'])} | "
                     f"{_m(v['rpe_10s_translation_rmse_m'])} | {_m(v['drift_1s_mean_percent'], 2)} | "
                     f"{_m(v['drift_5s_mean_percent'], 2)} | {_m(v['drift_10s_mean_percent'], 2)} |")
    lines += ["", f"A constant {rep['offset_rotation_deg']:.2f} deg body-frame rotation turns every relative displacement by "
              f"that angle, which alone adds 2 sin(theta/2) = {rep['relative_translation_error_from_offset_rotation_percent']:.1f} % of "
              "the distance travelled to RPE translation. That is why drift against the original reference is several "
              f"times larger at every interval, while the recomputed-reference stereo drift falls from "
              f"{_derived(results)['stereo_drift_1s']:.1f} % at 1 s to {_derived(results)['stereo_drift_10s']:.1f} % at 10 s. "
              "OpenVINS documents the original orientation as inaccurate. Independently of vicon, the "
              "camera-derived IMU orientation (ORB poses composed with the dataset camera-IMU calibration) leaves a "
              "mean body-frame rotation residual of "
              f"{validation['conversion_alternatives_stereo_se3']['references']['recomputed']['declared_T_W_C_inv_T_I_C']['mean_body_frame_rotation_residual_deg']:.2f} deg "
              f"(stereo) and {', '.join(f'{v:.2f}' for v in _derived(results)['inertial_body_recomputed'])} deg (stereo-IMU "
              "segments) against the recomputed reference, versus "
              f"{validation['conversion_alternatives_stereo_se3']['references']['original']['declared_T_W_C_inv_T_I_C']['mean_body_frame_rotation_residual_deg']:.2f} "
              f"and {', '.join(f'{v:.2f}' for v in _derived(results)['inertial_body_original'])} deg against the original. "
              "Together with the OpenVINS note, that supports the recomputed reference for orientation, provided the "
              "dataset camera-IMU calibration is right. Both references are reported throughout."]
    vicon = validation["vicon"]
    lines += ["", "## Validation", "",
              f"- T_I_Crect recomputed from `cam0`/`imu0` sensor.yaml and R_left_rectification: max |difference| "
              f"{validation['T_imu_rectified_left_camera']['max_abs_difference_vs_sensor_yaml']:.1e}.",
              f"- Association: identical pairs from the integer-ns associator and `pose_metrics.associate_timestamps` "
              f"for both references: {all(a['pose_metrics_associate_timestamps_identical'] for a in results['association'].values())}.",
              f"- Internal: ATE recomputed with `pose_metrics.score_trajectories` on "
              f"{validation['internal_score_trajectories']['comparisons']} scored sets, max |difference| "
              f"{validation['internal_score_trajectories']['ate_translation_max_abs_difference_m']:.1e} m (rotation "
              f"{validation['internal_score_trajectories']['rotation_max_abs_difference_deg']:.1e} deg; pose_metrics uses arccos).",
              "", "Conversion alternatives (diagnostic only, stereo SE3; the declared conversion was fixed before scoring):", "",
              "| reference | conversion | ATE RMSE | rotation error mean | mean body-frame rotation residual |",
              "|---|---|---:|---:|---:|"]
    for name, block in validation["conversion_alternatives_stereo_se3"]["references"].items():
        for key, value in block.items():
            lines.append(f"| {name} | `{key}` | {value['ate_rmse_m']:.4f} m | {value['rotation_error_mean_deg']:.2f} deg | "
                         f"{value['mean_body_frame_rotation_residual_deg']:.2f} deg |")
    lines += ["", f"Vicon frame check: {vicon['vicon_rows']} vicon0 rows ({1 / vicon['vicon_median_period_s']:.0f} Hz), "
              "interpolated at the reference stamps in the 60 s window. Candidate T_W_I built from the marker pose "
              f"T_W_V. {vicon['T_BS_rotation_note'][0].upper() + vicon['T_BS_rotation_note'][1:]}.", "",
              "| reference | candidate | samples | unaligned: mean rotation residual / translation residual | "
              "after SE3 world alignment: world rotation / position error RMSE / mean body rotation residual |",
              "|---|---|---:|---|---|"]
    for name, block in vicon["references"].items():
        for key, value in block["candidates"].items():
            aligned = block["candidates_after_se3_world_alignment"][key]
            lines.append(f"| {name} | `{key}` | {value['samples']} | {value['mean_residual_rotation_deg']:.2f} deg / "
                         f"{value['residual_translation_m']['mean']:.4f} m | {aligned['world_alignment_rotation_deg']:.2f} deg / "
                         f"{aligned['position_error_m']['rmse']:.4f} m / {aligned['mean_body_rotation_residual_deg']:.2f} deg |")
    scans = "; ".join(f"{name}: best {block['time_offset_scan_declared']['best_shift_ms']:+d} ms, position error RMSE "
                      f"{block['time_offset_scan_declared']['best']['position_error_m']['rmse']:.4f} m vs "
                      f"{block['time_offset_scan_declared']['zero_shift']['position_error_m']['rmse']:.4f} m at 0 ms"
                      for name, block in vicon["references"].items())
    d = _derived(results)
    lines += ["", f"Time-offset scan of the declared candidate (vicon at reference time + shift, -30..+30 ms): {scans}. "
              "Clock offset does not explain the residual. Composing the marker pose with inv(T_BS) puts the IMU within "
              f"{d['vicon_rot_min']:.1f}-{d['vicon_rot_max']:.1f} deg and {100 * d['vicon_pos_min']:.1f}-"
              f"{100 * d['vicon_pos_max']:.1f} cm of both references in a shared world frame (world rotation at most "
              f"{d['vicon_world_rot_max']:.2f} deg). The marker-as-body hypothesis is off by at least "
              f"{d['marker_rot_min']:.0f} deg and {100 * d['marker_pos_min']:.0f} cm, and the non-inverted T_BS leaves a "
              "larger lever-arm error. Both references are therefore IMU-frame poses. The published vicon0 T_BS itself "
              f"differs from each reference's body frame by at least {d['vicon_unaligned_rot_min']:.1f} deg, so vicon cannot "
              "arbitrate between the two references' orientations. The aligned vicon residuals and the time scan "
              "were added after the first run and feed no score."]
    evo = results["evo_crosscheck"]
    lines += ["", "### evo cross-check", ""]
    if evo.get("status") == "RUN":
        lines += [f"evo 1.34 read both reference CSVs itself (`evo_ape euroc ... -a`: SE3 Umeyama, no scale; "
                  f"`evo_rpe euroc ... --delta 20 --delta_unit f --all_pairs`) for the stereo arm, segment {evo['segment']}, "
                  f"from a TUM export of the IMU-frame estimate. All checks agree within 1e-6 m / 1e-4 deg: {evo['agree']}.", "",
                  "| reference | metric | evo RMSE | own RMSE | max abs difference (RMSE, mean, median, max) | count evo / own |",
                  "|---|---|---:|---:|---:|---|"]
        for name, block in evo["references"].items():
            for metric, check in block["checks"].items():
                lines.append(f"| {name} | {metric} | {check['evo']['rmse']:.6f} | {check['own']['rmse']:.6f} | "
                             f"{max(check['abs_difference'].values()):.1e} | {check['evo_count']} / {check['own_count']} |")
    else:
        lines.append(f"Not run: {evo.get('reason')}.")
    lines += ["", "## Limits", "",
              "- One sequence (V1_01_easy, the easiest Vicon-room flight) and only its first 60 s (1200 stereo pairs, "
              "about 19 m of reference path). There was one run per arm, so there is no run-to-run spread. The "
              f"between-jump ATE differences between arms are at most {_derived(results)['arm_gap_max_mm']:.1f} mm"
              + (" and change sign with the reference, so they do not rank the arms."
                 if _derived(results)["ordering_flips"] else ", untested against run-to-run spread."),
              "- Causal per-frame outputs as returned at frame time, not a final bundle-adjusted or loop-closed "
              "trajectory. The stereo-IMU arm initializes its IMU at 7.95 s and then jumps twice without a map change. "
              "Per-segment alignment removes each jump's rigid offset, so the per-segment numbers describe accuracy "
              "between jumps; the stitched whole shows what the jumps cost a consumer of the raw output.",
              "- Ground-truth orientation: OpenVINS documents the original EuRoC V1_01_easy orientation as inaccurate. "
              "The two references differ by a constant body-frame rotation and offset that explains the whole "
              "score difference between them. The recomputed reference is itself an optimization output (vicon2gt), "
              "not an independent measurement, and the published vicon0 extrinsic matches neither reference to "
              f"better than {_derived(results)['vicon_unaligned_rot_min']:.1f} deg.",
              "- License: the recomputed reference comes from the GPL-3.0 rpng/open_vins repository. It was downloaded "
              "to `reference/` with a receipt and read in place; it is not copied into the project repository.",
              "- No interpolation: original-reference association uses samples that coincide with camera stamps "
              "(max 256 ns offset). Path lengths use the 20 Hz associated samples, which slightly shortens the path "
              "used for drift.",
              "- Recorded external data, not this project's robot or hardware.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-protocol", help="write <output-dir>/protocol.json (never overwrites)")
    freeze.add_argument("--output-dir", type=Path, required=True)
    run = sub.add_parser("evaluate", help="score both arms against both pinned references")
    for name in ("dataset", "replay", "recomputed-reference", "output-dir"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--evo-ape", type=Path)
    run.add_argument("--evo-rpe", type=Path)
    report = sub.add_parser("report", help="re-render REPORT.md from results.json")
    report.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze-protocol":
        print(freeze_protocol(args.output_dir))
    elif args.command == "evaluate":
        evaluate(args.dataset, args.replay, args.recomputed_reference, args.output_dir,
                 evo_ape=args.evo_ape, evo_rpe=args.evo_rpe)
        print(args.output_dir / "results.json")
    else:
        results = json.loads((args.output_dir / "results.json").read_text())
        (args.output_dir / "REPORT.md").write_text(render_report(results))
        print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
