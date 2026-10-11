#!/usr/bin/env python3
"""Separate metric-scale evaluator for publisher-calibrated TUM VI IMU truth.

Never imported by the estimator runner. The official calibrated export labels
its reference dso/gt_imu.csv, and its numeric rows must match mocap0/data.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bhl_robust.research.native_orb import sha256
from bhl_robust.research.pose_metrics import Trajectory, score_trajectories, transform


def truth_pose(position, quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or abs(norm - 1) > 1e-6:
        raise ValueError("Publisher ground-truth quaternion must have unit norm")
    w, x, y, z = q / norm
    matrix = np.eye(4)
    matrix[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    matrix[:3, 3] = position
    return transform(matrix)


def csv_rows(path):
    with Path(path).open() as stream:
        return [row for row in csv.reader(stream) if row and not row[0].startswith("#")]


def load_truth(dataset, origin):
    rows = csv_rows(dataset / "dso/gt_imu.csv")
    mocap = csv_rows(dataset / "mav0/mocap0/data.csv")
    if rows != mocap:
        raise ValueError("Independent calibrated IMU-labelled truth export must match mocap numeric rows")
    timestamps = np.array([(int(row[0]) - origin) * 1e-9 for row in rows])
    poses = np.stack([truth_pose(list(map(float, row[1:4])), list(map(float, row[4:8]))) for row in rows])
    return Trajectory(timestamps, poses)


def map_epochs(rows):
    """Do not equate a reused map ID with continuous native coordinates.

    ORB's active-map reset can temporarily remove its map and later reuse ID0.
    The initialization flag can also regress without an ID change. Either
    observation splits epochs, even when the next map ID is identical.
    """
    epochs, events = [], []
    previous_map = None
    initialized = False
    previous_tracking_state = None
    new_epoch = True
    ever_had_map = False
    for index, frame in enumerate(rows):
        current_map = frame["map_id"]
        current_initialized = frame.get("inertial", {}).get("map_imu_initialized", False)
        reasons = []
        if previous_map is not None and current_map is None:
            reasons.append("native_map_disappeared")
        if previous_map is not None and current_map is not None and previous_map != current_map:
            reasons.append("native_map_id_changed")
        if initialized and not current_initialized:
            reasons.append("native_imu_initialization_lost")
        if ever_had_map and not new_epoch and frame.get("tracking_state") == 1 and previous_tracking_state != 1:
            reasons.append("native_returned_to_not_initialized")
        if reasons:
            events.append({"frame_index": index, "timestamp_s": frame["timestamp_s"],
                           "previous_map_id": previous_map, "map_id": current_map, "reasons": reasons})
            new_epoch = True
        if current_map is not None:
            if new_epoch:
                epochs.append({"epoch": len(epochs), "map_id": current_map,
                               "frames": [], "original_frame_indices": []})
                new_epoch = False
            epochs[-1]["frames"].append(frame)
            epochs[-1]["original_frame_indices"].append(index)
            ever_had_map = True
        elif ever_had_map:
            new_epoch = True
        previous_map, initialized = current_map, current_initialized
        previous_tracking_state = frame.get("tracking_state")
    return epochs, events


def accepted_frames(epoch, mode):
    return [f for f in epoch["frames"] if f["tracked"] and
            (mode == "stereo" or f["inertial"]["map_imu_initialized"])]


def score_frames(selected, truth, t_i_c):
    if len(selected) < 2:
        return {"status": "INSUFFICIENT_MATCHES", "frames": len(selected)}
    times = np.array([f["timestamp_s"] for f in selected])
    poses = np.stack([transform(f["T_W_C"]) @ np.linalg.inv(t_i_c) for f in selected])
    score = score_trajectories(Trajectory(times, poses), truth,
                              alignment="se3", max_difference_s=.005)
    intervals = score.pop("rpe_interval_s")
    score["rpe_interval_seconds_summary"] = {"min": min(intervals), "max": max(intervals),
                                               "mean": float(np.mean(intervals))}
    score.pop("associated_estimate_indices")
    score.pop("associated_truth_indices")
    for key in ("tracking_failure_segments", "tracking_lost_frames", "tracking_lost_time_s",
                "tracking_observation_end_s"):
        score.pop(key)
    score["tracking_scope"] = "Pose metrics use accepted initialized frames only; full native coverage and failures are reported separately."
    return {"status": "MEASURED", "first_sensor_time_s": float(times[0]),
            "last_sensor_time_s": float(times[-1]), **score}


def evaluate(dataset, replay, output):
    if output.exists():
        raise ValueError("Evaluator output must be new")
    result = json.loads((replay / "result.json").read_text())
    calibration = json.loads((replay / "rectification.json").read_text())
    t_i_c = transform(calibration["T_imu_rectified_left_camera"])
    truth = load_truth(dataset, result["clock_origin_ns"])
    arms, epochs_by_mode = [], {}
    for mode in ("stereo", "stereo_inertial"):
        frames_path = replay / mode / "frames.jsonl"
        native_summary = next(a for a in result["arms"] if a["mode"] == mode)
        if sha256(frames_path) != native_summary["frames_sha256"]:
            raise ValueError("Native frame trace differs from completed replay checksum")
        rows = [json.loads(line)["native"] for line in frames_path.read_text().splitlines()]
        if len(rows) != result["paired_frames_per_arm"] or len(rows) != native_summary["frames"]:
            raise ValueError("Full native frame count differs from completed replay")
        epochs, events = map_epochs(rows)
        epochs_by_mode[mode] = epochs
        accepted = [f for epoch in epochs for f in accepted_frames(epoch, mode)]
        maps = sorted(set(f["map_id"] for f in accepted))
        scores = []
        for epoch in epochs:
            scores.append({"map_id": epoch["map_id"], "epoch": epoch["epoch"],
                           "original_frame_indices": epoch["original_frame_indices"],
                           "epoch_first_sensor_time_s": epoch["frames"][0]["timestamp_s"],
                           "epoch_last_sensor_time_s": epoch["frames"][-1]["timestamp_s"],
                           **score_frames(accepted_frames(epoch, mode), truth, t_i_c)})
        arms.append({"mode": mode, "input_frames": len(rows), "initialized_tracked_frames": len(accepted),
                     "maps_with_accepted_poses": maps, "map_scores": scores,
                     "observed_epoch_boundaries": events, "map_epochs": len(epochs),
                     "full_native_tracked_frames": native_summary["tracked_frames"],
                     "full_native_untracked_frames": len(rows) - native_summary["tracked_frames"],
                     "full_native_tracking_states": native_summary["tracking_states"],
                     "full_native_tracking_loss_transitions": native_summary["tracking_loss_transitions_after_OK"],
                     "missing_initialized_pose_frames": len(rows) - len(accepted),
                     "frames_sha256": sha256(frames_path),
                     "continuous_single_map_trajectory": len(epochs) == 1 and not events,
                     "coverage_warning": "ATE uses associated initialized tracked poses; report tracking coverage separately. Each uninterrupted native map epoch is rigidly aligned independently; no scale correction or reset stitching."})
    shared = []
    for left in epochs_by_mode["stereo"]:
        a = {f["timestamp_s"]: f for f in accepted_frames(left, "stereo")}
        for right in epochs_by_mode["stereo_inertial"]:
            b = {f["timestamp_s"]: f for f in accepted_frames(right, "stereo_inertial")}
            common = sorted(a.keys() & b.keys())
            if len(common) >= 2:
                shared.append({"stereo_epoch": left["epoch"], "inertial_epoch": right["epoch"],
                               "shared_initialized_tracked_timestamps": len(common),
                               "stereo": score_frames([a[t] for t in common], truth, t_i_c),
                               "stereo_inertial": score_frames([b[t] for t in common], truth, t_i_c)})
    value = {"schema": "bhl-tumvi-separate-pose-evaluation-v2", "status": "COMPLETED",
             "source_sha256": sha256(Path(__file__)), "replay_result_sha256": sha256(replay / "result.json"),
             "reference": "Official calibrated room1 dso/gt_imu.csv; all numeric rows independently match mav0/mocap0/data.csv",
             "reference_sha256": sha256(dataset / "dso/gt_imu.csv"),
             "reference_frame_primary_source": "https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset",
             "reference_frame_evidence": "Publisher states calibrated sequences contain consistent clocks and ground-truth poses in the IMU frame; archive labels this file gt_imu.csv.",
             "estimated_pose_conversion": "T_nativeworld_rectifiedcamera @ inverse(T_imu_rectifiedcamera)",
             "association": "Unique nearest original ground-truth sample within 5 ms, no pose interpolation",
             "alignment": "Per-continuous-map-epoch SE3, scale fixed 1; offline evaluation only",
             "evaluator_correction": "v1 grouped by map ID, which can be reused after active-map reset; v2 splits explicit map disappearance and initialization loss. v1 is superseded, preserved and not a valid cross-reset ATE.",
             "shared_timestamp_comparisons": shared,
             "shared_comparison_scope": "Descriptive follow-up after discovering reset; paired identical native frame times within each overlapping map epoch, independent SE3 alignments, not a new native trial or a full-sequence continuity claim.",
             "truth_inputs_to_estimator": False, "own_hardware_validation": False, "arms": arms}
    output.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "replay", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.dataset, args.replay, args.output)


if __name__ == "__main__":
    main()
