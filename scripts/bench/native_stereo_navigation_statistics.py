#!/usr/bin/env python3
"""Audit actual native states and measured stereo pipeline latency."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(job_root, output):
    job_root, output = Path(job_root), Path(output)
    archive_path = job_root/"outputs.tar.gz"
    raw_sha = sha256(archive_path)
    completion = json.loads((job_root/"completion.json").read_text())
    if raw_sha != completion["files"]["outputs.tar.gz"]["sha256"]:
        raise ValueError("completed raw archive checksum differs")
    rows, native, images, campaign = {}, {}, {}, None
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            parent = str(Path(member.name).parent)
            if member.name == "campaign_result.json":
                campaign = json.load(archive.extractfile(member))
            elif member.name.endswith("/episode.json"):
                rows[parent] = json.load(archive.extractfile(member))
            elif member.name.endswith("/native/native_frames.jsonl"):
                native[str(Path(parent).parent)] = [json.loads(line) for line in archive.extractfile(member)]
            elif member.name.endswith("/stereo/image_manifest.json"):
                images[str(Path(parent).parent)] = json.load(archive.extractfile(member))
    results, all_cpp, all_capture, charged = [], [], [], []
    for prefix, row in rows.items():
        frames, manifest = native[prefix], images[prefix]
        post = [frame for frame in frames if frame["timestamp_s"] >= 5.]
        captures = [frame["capture_wall_seconds"] for frame in manifest["images"]]
        cpp = [frame["compute_seconds"] for frame in frames]
        all_cpp.extend(cpp)
        all_capture.extend(captures)
        by_stamp = {frame["timestamp_s"]: frame for frame in frames}
        deliveries = row["response_deliveries"]
        if len(deliveries)+row["responses_still_pending_at_horizon"] != len(frames):
            raise ValueError("delivered/pending native request count differs")
        for delivery in deliveries:
            frame = by_stamp[delivery["capture_s"]]
            capture = frame["additional_sensor_compute_seconds"]
            combined = delivery["wall_seconds"]
            if combined < capture or not np.isfinite(combined):
                raise ValueError("recorded combined request wall time is invalid")
            charged.append({"case": row["case"], "group": row["seed"],
                "capture_s": delivery["capture_s"], "request_s": delivery["request_s"],
                "arrival_s": delivery["arrival_s"], "capture_cost_s": capture,
                "client_wall_s": combined-capture, "combined_charged_wall_s": combined,
                "quantized_physics_delivery_delay_s": delivery["arrival_s"]-delivery["request_s"]})
        results.append({
            "case": row["case"], "group": row["seed"], "success": row["success"],
            "full_horizon_s": row["elapsed_simulation_s"], "frames_submitted_to_native": len(frames),
            "post5_native_frames": len(post),
            "post5_raw_native_OK_frames": sum(frame["native_ORB_tracking_state"] == 2 for frame in post),
            "post5_controller_accepted_frames": sum(frame["tracked"] for frame in post),
            "post5_controller_accepted_fraction": sum(frame["tracked"] for frame in post)/len(post) if post else None,
            "raw_native_OK_frames": sum(frame["native_ORB_tracking_state"] == 2 for frame in frames),
            "controller_accepted_tracked_frames": sum(frame["tracked"] for frame in frames),
            "native_state_histogram": dict(Counter(str(frame["native_ORB_tracking_state"]) for frame in frames)),
            "map_changed": any(frame["map_reset_id"] != 0 for frame in frames),
            "native_map_ids": sorted({frame["native_ORB_map_id"] for frame in frames if frame["native_ORB_map_id"] is not None}),
            "original_stereo_pairs": len(manifest["images"]), "dropped_scans": row["dropped_scans"],
            "capture_total_s": sum(captures), "capture_p95_ms": float(np.percentile(captures, 95))*1000,
            "native_cpp_compute_p95_ms": float(np.percentile(cpp, 95))*1000,
            "client_wall_p95_ms": row["client_wall_p95_ms"],
            "combined_charged_delivered_request_p95_ms": float(np.percentile([d["wall_seconds"] for d in deliveries], 95))*1000,
            "delivered_requests": len(deliveries), "pending_requests_excluded_from_combined_percentile": row["responses_still_pending_at_horizon"],
            "capture_cost_charged_s": row["additional_sensor_compute_total_s"],
            "capture_cost_unconsumed_at_horizon_s": manifest["unbilled_unconsumed_capture_seconds_at_horizon"],
            "final_goal_distance_m": row["trace"][-1]["evaluator_goal_distance_m"],
            "command_reason_steps": dict(Counter(tick["reason"] for tick in row["trace"]))})
    results.sort(key=lambda row: (row["case"], row["group"]))
    if campaign != json.loads((job_root/"campaign_result.json").read_text()) or len(results) != campaign["episodes"]:
        raise ValueError("whole-campaign original result/episode inventory differs")
    charged.sort(key=lambda row: (row["case"], row["group"], row["capture_s"]))
    canonical = json.dumps(charged, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    vector_path = output.with_name(output.stem+"-combined-latency-vector.json")
    vector = {"schema": "bhl-stereo-combined-charged-latency-v1", "raw_archive_sha256": raw_sha,
        "derivation": "original episode.response_deliveries[].wall_seconds is measured client.track wall + accumulated capture/render/PNG/hash wall for that same request; no summation of independent percentiles. Pending horizon requests have no retained exact combined wall and are excluded. Physics delivery delay is separately quantized.",
        "rows_canonical_sha256": hashlib.sha256(canonical).hexdigest(), "rows": charged}
    vector_path.write_text(json.dumps(vector, indent=2, allow_nan=False)+"\n")
    post_count = sum(row["post5_native_frames"] for row in results)
    totals = {"episodes": len(results), "goals": sum(row["success"] for row in results),
        "native_frames": sum(row["frames_submitted_to_native"] for row in results),
        "post5_native_frames": post_count,
        "post5_raw_native_OK_frames": sum(row["post5_raw_native_OK_frames"] for row in results),
        "post5_controller_accepted_frames": sum(row["post5_controller_accepted_frames"] for row in results),
        "post5_controller_accepted_fraction": sum(row["post5_controller_accepted_frames"] for row in results)/post_count,
        "raw_native_OK_frames": sum(row["raw_native_OK_frames"] for row in results),
        "controller_accepted_tracked_frames": sum(row["controller_accepted_tracked_frames"] for row in results),
        "map_changed_episodes": sum(row["map_changed"] for row in results),
        "original_stereo_pairs": sum(row["original_stereo_pairs"] for row in results),
        "capture_total_s": sum(all_capture), "capture_pair_p95_ms": float(np.percentile(all_capture, 95))*1000,
        "native_cpp_frame_p95_ms": float(np.percentile(all_cpp, 95))*1000,
        "episode_client_wall_p95_range_ms": [min(row["client_wall_p95_ms"] for row in results), max(row["client_wall_p95_ms"] for row in results)],
        "combined_charged_delivered_request_p95_ms": float(np.percentile([row["combined_charged_wall_s"] for row in charged], 95))*1000,
        "quantized_physics_delivery_delay_p95_ms": float(np.percentile([row["quantized_physics_delivery_delay_s"] for row in charged], 95))*1000,
        "combined_charged_delivered_requests": len(charged),
        "pending_requests_excluded_from_combined_percentile": sum(row["pending_requests_excluded_from_combined_percentile"] for row in results)}
    summary = {"schema": "bhl-stereo-navigation-final-statistics-v1", "scientific_status": campaign["status"], "phase": campaign["phase"],
        "scope": "DEVELOPMENT_ONLY_NO_CONFIRMATION. Actual simulation images/genuine unchanged native ORB; prescribed routes/known start/one frozen gait plus independent LiDAR brake; no hardware claim",
        "raw_archive_sha256": raw_sha, "generator_source_sha256": sha256(__file__),
        "combined_latency_vector": {"path": vector_path.name, "sha256": sha256(vector_path), "rows_canonical_sha256": vector["rows_canonical_sha256"]},
        "episodes": results, "totals": totals}
    output.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.job_root, args.output)["totals"], indent=2))
