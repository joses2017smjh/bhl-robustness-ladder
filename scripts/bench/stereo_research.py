#!/usr/bin/env python3
"""Run SGBM and the pinned official pretrained C-Fast-FoundationStereo."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from bhl_robust.research.stereo_benchmark import (
    FastFoundationStereoMethod, SGBMMethod, checked_path, depth_metrics,
    load_replay, load_rgb, paired_sequence_bootstrap, timed_replay,
)


def aggregate(rows):
    """Pool sums/counts, preserving invalid coverage and empty-common cases."""
    gt = sum(row["gt_pixels"] for row in rows)
    native = sum(row["valid_gt_pixels"] for row in rows)
    scored = sum(row["scored_pixels"] for row in rows)
    near = sum(row["near_obstacle_pixels"] for row in rows)
    recalled = sum(row["recalled_near_obstacle_pixels"] for row in rows)
    result = {"gt_pixels": gt, "valid_gt_pixels": native, "scored_pixels": scored,
              "coverage": native/gt if gt else None,
              "near_obstacle_pixel_recall": recalled/near if near else None,
              "near_obstacle_pixels": near, "recalled_near_obstacle_pixels": recalled}
    for key in ("mae_m", "abs_rel"):
        result[key] = sum(row[key]*row["scored_pixels"] for row in rows if row[key] is not None)/scored if scored else None
    result["rmse_m"] = np.sqrt(sum(row["rmse_m"]**2*row["scored_pixels"] for row in rows
                                 if row["rmse_m"] is not None)/scored).item() if scored else None
    return result


def run(dataset, output, *, model=None, provider="CUDAExecutionProvider", warmup=30, timed=200, protocol=None):
    output, dataset = Path(output), Path(dataset)
    output.mkdir(parents=True, exist_ok=True)
    manifest, calibration = load_replay(dataset)
    if warmup < 0 or timed < 1 or warmup > 1000 or timed > 10000:
        raise ValueError("Timing budgets are bounded")
    methods = [SGBMMethod(calibration)]
    initialization = {}
    blocked = []
    if model:
        started = time.perf_counter()
        try:
            methods.append(FastFoundationStereoMethod(calibration, model, provider=provider))
            initialization[methods[-1].name] = (time.perf_counter()-started)*1000
        except Exception as exc:
            blocked.append({"method": "c_fast_foundationstereo", "status": "INCOMPLETE",
                            "reason": str(exc), "substitution": False})
    else:
        blocked.append({"method": "c_fast_foundationstereo", "status": "INCOMPLETE",
                        "reason": "Pinned pretrained checkpoint not supplied", "substitution": False})
    prediction_records, frame_metrics = [], []
    first_pass = {method.name: [] for method in methods}
    pairs = []
    for sequence in manifest["sequences"]:
        for frame in sequence["frames"]:
            left_path = checked_path(dataset, frame["left"], namespace="inference")
            right_path = checked_path(dataset, frame["right"], namespace="inference")
            pairs.append((left_path, right_path))
            predictions = {}
            for method in methods:
                started = time.perf_counter()
                # The inference API receives only these RGB files/calibration.
                depth, valid = method(load_rgb(left_path), load_rgb(right_path))
                first_pass[method.name].append((time.perf_counter()-started)*1000)
                predictions[method.name] = (depth, valid)
                directory = output/"predictions"/method.name/sequence["id"]
                directory.mkdir(parents=True, exist_ok=True)
                paths = {}
                for key, value in (("depth", depth), ("valid", valid), ("confidence", valid.astype(np.float32))):
                    path = directory/(frame["frame_id"] + "_" + key + ".npy")
                    np.save(path, value, allow_pickle=False)
                    paths[key] = path.relative_to(output).as_posix()
                prediction_records.append({"sequence_id": sequence["id"], "frame_id": frame["frame_id"],
                    "method": method.name, **paths, "confidence_kind": "binary_validity_support_not_calibrated_probability"})
            # Load independent truth AFTER all methods have returned predictions.
            with np.load(checked_path(dataset, frame["truth"], namespace="evaluator"), allow_pickle=False) as truth_file:
                truth = truth_file["depth_z_m"]
                obstacle_mask = truth_file["obstacle_mask"] if "obstacle_mask" in truth_file else None
                if (obstacle_mask is not None and str(truth_file.get("obstacle_truth_source", ""))
                        != "independent_scene_geom_segmentation"):
                    raise ValueError("Obstacle labels lack independent scene provenance")
            common = np.logical_and.reduce([v[1] for v in predictions.values()])
            for name, (depth, valid) in predictions.items():
                frame_metrics.append({"sequence_id": sequence["id"], "scene_id": sequence["scene_id"],
                    "split": sequence["split"], "frame_id": frame["frame_id"], "method": name,
                    "native": depth_metrics(depth, valid, truth, obstacle_mask=obstacle_mask),
                    "common": depth_metrics(depth, valid, truth, common_mask=common, obstacle_mask=obstacle_mask),
                    "comparison_methods": list(predictions)})
    provenance = [method.provenance() for method in methods]
    timing = {}
    for method in methods:
        timing[method.name] = timed_replay(method, pairs, warmup=warmup, timed=timed)
        timing[method.name]["session_initialization_ms"] = initialization.get(method.name)
        timing[method.name]["first_replay_call_ms"] = first_pass[method.name][0]
        timing[method.name]["host"] = platform.node()
        timing[method.name]["provider"] = provider if method.name != "sgbm" else "CPU"
    by_sequence = []
    for sequence in manifest["sequences"]:
        for method in methods:
            rows = [r for r in frame_metrics if r["sequence_id"] == sequence["id"] and r["method"] == method.name]
            by_sequence.append({"sequence_id": sequence["id"], "scene_id": sequence["scene_id"],
                "split": sequence["split"], "method": method.name,
                "native": aggregate([r["native"] for r in rows]), "common": aggregate([r["common"] for r in rows])})
    by_split = []
    for split in ("dev", "validation", "test"):
        for method in methods:
            rows = [r for r in frame_metrics if r["split"] == split and r["method"] == method.name]
            if rows:
                by_split.append({"split": split, "method": method.name,
                    "native": aggregate([r["native"] for r in rows]), "common": aggregate([r["common"] for r in rows])})
    differences = []
    if len(methods) == 2:
        for sequence in manifest["sequences"]:
            if sequence["split"] != "test":
                continue
            entries = {r["method"]: r for r in by_sequence if r["sequence_id"] == sequence["id"]}
            a, b = entries["sgbm"]["common"]["rmse_m"], entries["c_fast_foundationstereo"]["common"]["rmse_m"]
            if a is not None and b is not None:
                differences.append(b-a)
    bootstrap = paired_sequence_bootstrap(differences)
    predictions = {"schema_version": 1, "schema": "bhl-stereo-predictions-v1",
                   "dataset_manifest_sha256": hashlib.sha256((dataset/"manifest.json").read_bytes()).hexdigest(),
                   "predictions": prediction_records, "methods": provenance}
    (output/"predictions.json").write_text(json.dumps(predictions, indent=2, sort_keys=True)+"\n")
    report = {"schema_version": 1, "route": "stereo_depth", "origin": manifest.get("origin"),
        "status": "INCOMPLETE" if blocked else "PASS",
        "scientific_status": "SIMULATION_PILOT_ONLY_NO_CONFIRMED_SUPERIORITY",
        "physical_accuracy_measured": False, "methods": provenance, "blocked_methods": blocked,
        "frame_count": len(pairs), "scene_group_count": len(manifest["sequences"]),
        "timing": timing, "by_sequence": by_sequence, "by_split": by_split,
        "heldout_common_RMSE_difference_FFS_minus_SGBM": bootstrap,
        "metric_scope": "pixel_depth_error_coverage_independently_labeled_near_obstacle_pixel_recall",
        "comparison_caveat": "Native masks differ; report native coverage with common-mask error",
        "end_to_end_navigation_latency": "UNMEASURED_capture_rectification_fusion_controller_excluded",
        "dataset_manifest_sha256": predictions["dataset_manifest_sha256"]}
    if protocol:
        report["protocol_sha256"] = hashlib.sha256(Path(protocol).read_bytes()).hexdigest()
    (output/"frame_metrics.json").write_text(json.dumps(frame_metrics, indent=2)+"\n")
    (output/"campaign_result.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--provider", choices=("CUDAExecutionProvider", "CPUExecutionProvider"), default="CUDAExecutionProvider")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--timed", type=int, default=200)
    parser.add_argument("--protocol")
    args = parser.parse_args()
    try:
        report = run(args.dataset, args.output, model=args.model, provider=args.provider,
                     warmup=args.warmup, timed=args.timed, protocol=args.protocol)
    except Exception as exc:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        report = {"status": "INCOMPLETE", "route": "stereo_depth", "error": str(exc)}
        (Path(args.output)/"campaign_result.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("status", "route")}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
