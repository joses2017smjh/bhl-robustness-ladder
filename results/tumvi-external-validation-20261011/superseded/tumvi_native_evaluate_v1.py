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


def evaluate(dataset, replay, output):
    if output.exists():
        raise ValueError("Evaluator output must be new")
    result = json.loads((replay / "result.json").read_text())
    calibration = json.loads((replay / "rectification.json").read_text())
    t_i_c = transform(calibration["T_imu_rectified_left_camera"])
    truth = load_truth(dataset, result["clock_origin_ns"])
    arms = []
    for mode in ("stereo", "stereo_inertial"):
        frames_path = replay / mode / "frames.jsonl"
        rows = [json.loads(line)["native"] for line in frames_path.read_text().splitlines()]
        accepted = [f for f in rows if f["tracked"] and (mode == "stereo" or f["inertial"]["map_imu_initialized"])]
        maps = sorted(set(f["map_id"] for f in accepted))
        scores = []
        for map_id in maps:
            selected = [f for f in accepted if f["map_id"] == map_id]
            if len(selected) < 2:
                scores.append({"map_id": map_id, "status": "INSUFFICIENT_MATCHES", "frames": len(selected)})
                continue
            times = np.array([f["timestamp_s"] for f in selected])
            poses = np.stack([transform(f["T_W_C"]) @ np.linalg.inv(t_i_c) for f in selected])
            score = score_trajectories(Trajectory(times, poses), truth,
                                      alignment="se3", max_difference_s=.005)
            # Compact metrics retain explicit counts and timing bounds without
            # publishing original dataset poses or a redundant full index list.
            intervals = score.pop("rpe_interval_s")
            score["rpe_interval_seconds_summary"] = {"min": min(intervals), "max": max(intervals),
                                                       "mean": float(np.mean(intervals))}
            score.pop("associated_estimate_indices")
            score.pop("associated_truth_indices")
            scores.append({"map_id": map_id, "status": "MEASURED", **score})
        arms.append({"mode": mode, "input_frames": len(rows), "initialized_tracked_frames": len(accepted),
                     "maps_with_accepted_poses": maps, "map_scores": scores,
                     "frames_sha256": sha256(frames_path),
                     "continuous_single_map_trajectory": len(maps) == 1,
                     "coverage_warning": "ATE uses associated initialized tracked poses; report tracking coverage separately. Each native map is rigidly aligned independently; no scale correction or cross-map stitching."})
    value = {"schema": "bhl-tumvi-separate-pose-evaluation-v1", "status": "COMPLETED",
             "source_sha256": sha256(Path(__file__)), "replay_result_sha256": sha256(replay / "result.json"),
             "reference": "Official calibrated room1 dso/gt_imu.csv; all numeric rows independently match mav0/mocap0/data.csv",
             "reference_sha256": sha256(dataset / "dso/gt_imu.csv"),
             "reference_frame_primary_source": "https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset",
             "reference_frame_evidence": "Publisher states calibrated sequences contain consistent clocks and ground-truth poses in the IMU frame; archive labels this file gt_imu.csv.",
             "estimated_pose_conversion": "T_nativeworld_rectifiedcamera @ inverse(T_imu_rectifiedcamera)",
             "association": "Unique nearest original ground-truth sample within5ms, no pose interpolation",
             "alignment": "Per-native-map SE3, scale fixed1; offline evaluation only",
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
