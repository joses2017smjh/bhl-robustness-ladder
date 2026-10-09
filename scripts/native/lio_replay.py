#!/usr/bin/env python3
"""Execute real, pinned native FAST-LIO2 from calibrated inference-only data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from bhl_robust.research.native_lio import UPSTREAM_COMMIT, sha256, validate_native_output, write_transport


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def replay(manifest_path, sequence, runtime_dir, output, **config):
    runtime_dir, output = Path(runtime_dir).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError("native replay output directory must be new")
    runtime = json.loads((runtime_dir / "runtime.json").read_text())
    binary = runtime_dir / "fastlio_headless"
    if (runtime.get("status") != "PASS" or runtime.get("upstream_commit") != UPSTREAM_COMMIT
            or sha256(binary) != runtime.get("binary_sha256")):
        raise ValueError("runtime must be the checksum-verified pinned native FAST-LIO2 build")
    # Reuse the previously audited adapter's strict inference-namespace,
    # calibration, original timestamps, raw-array and checksum checks.
    adapter_path = Path(__file__).parents[1] / "bench/pose_research.py"
    spec = importlib.util.spec_from_file_location("bhl_pose_research_for_lio", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    manifest, seq, extrinsic, packets, max_gap = adapter.fastlio_packets(manifest_path, sequence)
    output.mkdir(parents=True)
    started = time.monotonic()
    transport = write_transport(packets, extrinsic, output / "input.bin", **config)
    write_json(output / "transport.json", transport)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(runtime_dir / "lib")
    env["OMP_NUM_THREADS"] = "1"
    with (output / "native.log").open("w") as log:
        command = [str(binary), str(output / "input.bin"), str(output / "native_frames.jsonl")]
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    if completed.returncode:
        write_json(output / "campaign_result.json", {"status": "INCOMPLETE", "native_returncode": completed.returncode,
                   "estimated_pose_outputs": 0, "reason": "Native FAST-LIO2 process failed; see native.log"})
        raise RuntimeError("native FAST-LIO2 process failed")
    rows = validate_native_output(output / "native_frames.jsonl", transport)
    input_root = Path(manifest_path).resolve().parent
    calibration_hash = hashlib.sha256(json.dumps(manifest["calibration"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    config_hash = hashlib.sha256(json.dumps(transport["config"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    frames = [{"timestamp_s": row["timestamp_s"], "tracked": row["tracked"],
               "T_W_S": row["T_W_I"] if row["tracked"] else np.eye(4).tolist(),
               "native_state": row["state"], "pose_is_storage_placeholder": not row["tracked"]} for row in rows]
    trajectory = {"schema": "bhl-pose-trajectory-v1", "sequence": sequence, "sensor_frame": "imu_body",
                  "clock_domain": manifest["clock_domain"], "origin": seq["origin"], "split": seq["split"],
                  "end_time_s": rows[-1]["timestamp_s"], "frames": frames,
                  "untracked_pose_semantics": "Identity is only a schema storage placeholder; tracked=false is never scored or supplied to navigation; authoritative native_frames.jsonl uses null."}
    write_json(output / "trajectory.json", trajectory)
    receipt = {"schema": "bhl-pose-estimator-run-v1", "method": "fast_lio2_lidar_imu",
               "upstream_commit": UPSTREAM_COMMIT, "runtime_sha256": runtime["binary_sha256"],
               "runtime_receipt_sha256": sha256(runtime_dir / "runtime.json"),
               "input_manifest_sha256": sha256(manifest_path), "calibration_sha256": calibration_hash,
               "config_sha256": config_hash, "trajectory_sha256": sha256(output / "trajectory.json"),
               "native_frames_sha256": sha256(output / "native_frames.jsonl"),
               "input_files_sha256": adapter._input_receipt(input_root, seq, ("lidar", "imu")),
               "ground_truth_inputs": [], "sensor_frame": "imu_body", "sequence": sequence,
               "native_command": command, "transport": transport,
               "point_intensity_kinds": sorted({p["intensity_kind"] for p in packets if p["kind"] == "pointcloud2"}),
               "timing_scope": "Native C++ per-scan IMU deskew, voxel filtering, map matching/EKF/update; excludes capture, input validation/export and process startup", "max_imu_gap_s": max_gap}
    write_json(output / "estimator_receipt.json", receipt)
    result = {"schema": "bhl-native-lio-result-v1", "status": "PASS", "scientific_status": "NATIVE_REPLAY_MEASURED_NOT_COMPARISON",
              "method": "fast_lio2_lidar_imu", "sequence": sequence, "origin": seq["origin"], "split": seq["split"],
              "frames": len(rows), "tracked_frames": sum(row["tracked"] for row in rows),
              "estimated_pose_outputs": sum(row["tracked"] for row in rows),
              "states": {state: sum(row["state"] == state for row in rows) for state in sorted({r["state"] for r in rows})},
              "native_compute_p95_ms": float(np.quantile([row["compute_seconds"] * 1000 for row in rows], .95)),
              "elapsed_seconds": time.monotonic() - started, "ground_truth_inputs": [], "closed_loop_episodes": 0,
              "trajectory_sha256": receipt["trajectory_sha256"], "estimator_receipt_sha256": sha256(output / "estimator_receipt.json")}
    write_json(output / "campaign_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--surface-voxel-m", type=float, default=.15)
    parser.add_argument("--map-voxel-m", type=float, default=.15)
    parser.add_argument("--blind-m", type=float, default=.1)
    args = parser.parse_args()
    result = replay(args.manifest, args.sequence, args.runtime, args.output,
                    surface_voxel_m=args.surface_voxel_m, map_voxel_m=args.map_voxel_m, blind_m=args.blind_m)
    print(json.dumps(result, allow_nan=False))
