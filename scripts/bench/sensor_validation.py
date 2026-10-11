#!/usr/bin/env python3
"""Measure stereo rectification and audit physical replay inputs without control."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pose_research as replay
from bhl_robust.research.sensor_validation import (
    hardware_evidence, match_rectified_pair, rectification_residuals,
)


def evaluate(manifest_path, frames_per_sequence=10, physical=False):
    import cv2
    cv2.setNumThreads(1)
    manifest, root = replay.read_manifest(manifest_path)
    if frames_per_sequence < 2:
        raise ValueError("at least two diagnostic frames per sequence required")
    sequences = []
    for seq in manifest["sequences"]:
        selected = np.unique(np.linspace(0, len(seq["frames"]) - 1,
                                         min(frames_per_sequence, len(seq["frames"])), dtype=int))
        rows, all_left, all_right = [], [], []
        for index in selected:
            frame = seq["frames"][int(index)]
            images = [cv2.imread(str(replay.inference_path(root, frame[k])), cv2.IMREAD_GRAYSCALE)
                      for k in ("left", "right")]
            left, right, counts = match_rectified_pair(*images)
            row = {"frame_index": int(index), "timestamp_s": frame["timestamp_s"],
                   **counts, **rectification_residuals(left, right)}
            rows.append(row)
            all_left.append(left)
            all_right.append(right)
        measured = rectification_residuals(np.concatenate(all_left), np.concatenate(all_right))
        # A textured frame cannot hide a textureless/poorly calibrated frame.
        counts = dict(collections.Counter(row["status"] for row in rows))
        measured["pooled_match_status"] = measured["status"]
        if any(row["status"] != "WITHIN_DIAGNOSTIC_LIMIT" for row in rows):
            measured["status"] = "INCOMPLETE_OR_EXCEEDS_FRAME_LIMIT"
        sequences.append({"id": seq["id"], "origin": seq["origin"], "scene_id": seq["scene_id"],
                          "split": seq["split"], **measured, "frame_status_counts": counts, "frames": rows})
    result = {"schema": "bhl-stereo-rectification-diagnostic-v1", "manifest_sha256": replay.sha256(manifest_path),
              "input_files_sha256": manifest["file_sha256"], "sequences": sequences,
              "settings": {"matching": "mutual SIFT descriptor ratio; no epipolar prefilter",
                           "ratio": .75, "max_features": 2000, "minimum_matches_per_frame": 20,
                           "maximum_vertical_p95_px": 2.0, "sampling": "deterministic equal frame-index spacing"},
              "scope": "Read-only rectification diagnosis; engineering limits, no calibration fit or ORB telemetry substitution"}
    if physical:
        errors = {}
        for seq in manifest["sequences"]:
            try:
                replay.fastlio_packets(manifest_path, seq["id"])
            except (ValueError, KeyError, OSError) as exc:
                errors[seq["id"]] = str(exc)
        provenance_errors = []
        for key in ("clock_sync_evidence", "calibration_evidence"):
            try:
                relative = manifest.get("physical_rig", {}).get(key)
                replay.verify_input_hash(manifest, relative, replay.inference_path(root, relative))
            except (ValueError, KeyError, OSError) as exc:
                provenance_errors.append(key + ": " + str(exc))
        result["physical_intake"] = hardware_evidence(
            manifest, lidar_packets_valid=not errors,
            epipolar_by_sequence={s["id"]: s for s in sequences},
            provenance_files_verified=not provenance_errors)
        result["physical_intake"]["sensor_packet_errors"] = errors
        result["physical_intake"]["provenance_errors"] = provenance_errors
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-sequence", type=int, default=10)
    parser.add_argument("--physical", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.manifest, args.frames_per_sequence, args.physical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sequences": len(result["sequences"]),
                      "statuses": {s["id"]: s["status"] for s in result["sequences"]}}))


if __name__ == "__main__":
    main()
