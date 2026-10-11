#!/usr/bin/env python3
"""Paired native ORB stereo / stereo–IMU replay with original recorded sensors.

This is an estimator development replay, not closed-loop navigation or a hardware
benchmark. Every native frame is retained, including visual/IMU startup failures.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from bhl_robust.research.native_orb import NativeOrbClient, UPSTREAM_COMMIT, sha256, tracking_summary, write_rectified_stereo_settings
from bhl_robust.research.native_orb_inertial import ImuSettings, partition_imu, write_stereo_inertial_settings
from bhl_robust.research.pose_metrics import transform


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def adapter_module():
    path = Path(__file__).parents[1] / "bench/pose_research.py"
    spec = importlib.util.spec_from_file_location("orb_inertial_pose_adapter", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def read_original_imu(adapter, root, sequence):
    """Read only inference IMU arrays; exact repeated packet boundaries deduplicate."""
    samples, duplicate_boundaries = {}, 0
    for frame in sequence["frames"]:
        payload = adapter._arrays(adapter.inference_path(root, frame.get("imu")),
                                  ("timestamp_s", "gyro_rad_s", "specific_force_m_s2"))
        ts, gyro, accel = (payload[key] for key in ("timestamp_s", "gyro_rad_s", "specific_force_m_s2"))
        if (ts.ndim != 1 or not len(ts) or gyro.shape != (len(ts), 3) or accel.shape != gyro.shape
                or not all(np.isfinite(a).all() for a in (ts, gyro, accel)) or np.any(np.diff(ts) <= 0)):
            raise ValueError("Original finite increasing SI-unit IMU packets required")
        for stamp, g, a in zip(ts, gyro, accel):
            row = [float(stamp), *map(float, g), *map(float, a)]
            if row[0] in samples:
                if samples[row[0]] != row: raise ValueError("Conflicting original IMU boundary samples")
                duplicate_boundaries += 1
            samples[row[0]] = row
    return [samples[t] for t in sorted(samples)], duplicate_boundaries


def replay(manifest_path, sequence_id, runtime, output, *, mode, frame_limit=None, imu=ImuSettings()):
    if mode not in ("stereo", "stereo_inertial"): raise ValueError("Unsupported native mode")
    runtime, output = Path(runtime).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError("Each native replay must use a new output directory")
    receipt = json.loads((runtime / "runtime.json").read_text())
    if receipt.get("status") != "PASS" or receipt.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("Verified pinned native ORB runtime required")
    for name, digest in receipt.get("files_sha256", {}).items():
        member = runtime / name
        if not member.resolve().is_relative_to(runtime) or sha256(member) != digest:
            raise ValueError("Native runtime member changed: " + name)
    if any(name not in receipt.get("files_sha256", {}) for name in ("bin/orb_native", "ORBvoc.txt", "lib/libORB_SLAM3.so")):
        raise ValueError("Required native binary, vocabulary and library provenance missing")
    adapter = adapter_module()
    manifest, root = adapter.read_manifest(manifest_path)
    sequence = adapter.choose_sequence(manifest, sequence_id)
    calibration = manifest["calibration"]
    if not math.isclose(calibration["cx_left_px"], calibration["cx_right_px"], abs_tol=1e-7, rel_tol=0):
        raise ValueError("Native rectified camera principal points must match")
    frames = sequence["frames"]
    if frame_limit is not None:
        if type(frame_limit) is not int or not 2 <= frame_limit <= len(frames):
            raise ValueError("Explicit frame limit must retain at least two original frames")
        frames = frames[:frame_limit]
    timestamps = [frame["timestamp_s"] for frame in frames]
    nominal_fps = int(round(1 / float(np.median(np.diff(timestamps)))))
    settings = dict(fx=calibration["fx_px"], fy=calibration["fy_px"], cx=calibration["cx_left_px"],
                    cy=calibration["cy_px"], width=calibration["image_width"], height=calibration["image_height"],
                    fps=nominal_fps, baseline_m=calibration["baseline_m"])
    output.mkdir(parents=True)
    inertial_input = None
    if mode == "stereo_inertial":
        # Both transforms are measured mounting calibration, never evaluator poses.
        t_i_c = np.linalg.inv(transform(calibration["T_B_I"])) @ transform(calibration["T_B_C"])
        config = write_stereo_inertial_settings(output / "settings.yaml", t_imu_camera=t_i_c, imu=imu, **settings)
        original_imu, boundaries = read_original_imu(adapter, root, sequence)
        batches, inertial_input = partition_imu(timestamps, original_imu)
        inertial_input.update(duplicate_packet_boundaries_deduplicated=boundaries,
                              units="timestamp s; angular velocity rad/s; specific force m/s^2",
                              interpolation="none", imu_rate_hz=manifest.get("imu_rate_hz"))
        if manifest.get("imu_rate_hz") != imu.frequency_hz:
            raise ValueError("Frozen native IMU frequency must match original sensor declaration")
    else:
        write_rectified_stereo_settings(output / "settings.yaml", **settings)
        batches, config = [None] * len(frames), {"stereo_only": True}
    env = dict(os.environ, LD_LIBRARY_PATH=str(runtime / "lib"), OMP_NUM_THREADS="1")
    started, rows = time.monotonic(), []
    with NativeOrbClient(runtime / "bin/orb_native", runtime / "ORBvoc.txt", output / "settings.yaml",
                         output / "process", sensor_mode=mode, environment=env) as client:
        wall_epoch, sensor_epoch = time.monotonic(), timestamps[0]
        with (output / "native_frames.jsonl").open("x") as stream:
            for frame, batch in zip(frames, batches):
                deadline = wall_epoch + frame["timestamp_s"] - sensor_epoch
                remaining = deadline-time.monotonic()
                if remaining > 0: time.sleep(remaining)
                outer = time.monotonic()
                kwargs = {} if batch is None else {"imu_samples": batch}
                row = client.track(frame["timestamp_s"], adapter.inference_path(root, frame["left"]),
                                   adapter.inference_path(root, frame["right"]), **kwargs)
                row.update(client_wall_seconds=time.monotonic()-outer,
                           replay_start_lateness_seconds=max(0., outer-deadline))
                stream.write(json.dumps(row, allow_nan=False)+"\n"); stream.flush(); rows.append(row)
    result = tracking_summary(rows)
    initialized = [row for row in rows if row.get("inertial", {}).get("map_imu_initialized")]
    result.update(schema="bhl-native-orb-inertial-replay-v1", status="PASS", sensor_mode=mode,
                  scientific_status="DEVELOPMENT_REPLAY_NO_GENERALIZATION_CLAIM",
                  sequence=sequence_id, origin=sequence["origin"], split=sequence["split"],
                  imu_initialized_frames=len(initialized),
                  first_imu_initialized_timestamp_s=initialized[0]["timestamp_s"] if initialized else None,
                  imu_initialization_observed=bool(initialized),
                  tracked_and_imu_initialized_frames=sum(row["tracked"] for row in initialized),
                  inertial_input=inertial_input, settings=config,
                  runtime_receipt_sha256=sha256(runtime / "runtime.json"),
                  binary_sha256=sha256(runtime / "bin/orb_native"),
                  settings_sha256=sha256(output / "settings.yaml"),
                  manifest_sha256=sha256(manifest_path),
                  input_files_sha256=adapter._input_receipt(root, sequence, ("left", "right", "imu") if mode == "stereo_inertial" else ("left", "right")),
                  frames_sha256=sha256(output / "native_frames.jsonl"),
                  elapsed_seconds=time.monotonic()-started, ground_truth_inputs=[], closed_loop_episodes=0,
                  native_compute_p95_ms=1000*float(np.quantile([r["compute_seconds"] for r in rows], .95)),
                  qualification="Execution PASS does not imply native visual or inertial initialization succeeded")
    # Visual tracking alone is insufficient evidence of successful inertial fusion.
    result["inertial_scoring_ready"] = bool(mode == "stereo_inertial" and initialized and result["metric_scoring_ready"])
    write_json(output / "estimator_receipt.json", result)
    write_json(output / "campaign_result.json", result)
    return rows, result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("manifest", "sequence", "runtime", "output"): parser.add_argument("--"+name, required=True)
    parser.add_argument("--mode", choices=("stereo", "stereo_inertial", "paired"), default="paired")
    parser.add_argument("--frame-limit", type=int)
    args = parser.parse_args()
    if args.mode == "paired":
        destination = Path(args.output); destination.mkdir(parents=True, exist_ok=False)
        results = {}
        for mode in ("stereo", "stereo_inertial"):
            _, results[mode] = replay(args.manifest, args.sequence, args.runtime, destination / mode,
                                      mode=mode, frame_limit=args.frame_limit)
        result = {"schema": "bhl-orb-stereo-inertial-pair-v1", "status": "PASS", "arms": results,
                  "ground_truth_inputs": [], "closed_loop_episodes": 0,
                  "scientific_status": "PAIRED_REPLAY_EXECUTED; INITIALIZATION_AND_DRIFT_REQUIRE_SEPARATE_REVIEW"}
        write_json(destination / "campaign_result.json", result)
    else:
        _, result = replay(args.manifest, args.sequence, args.runtime, args.output,
                           mode=args.mode, frame_limit=args.frame_limit)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__": main()
