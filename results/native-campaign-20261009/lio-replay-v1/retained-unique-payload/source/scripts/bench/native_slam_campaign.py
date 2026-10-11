#!/usr/bin/env python3
"""Run actual native stereo/LIO and independently score every recorded state.

This is a simulation replay experiment, not closed-loop navigation. Estimator
inputs contain no evaluator poses. Failed tracking is a measured negative.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bhl_robust.research.native_orb import NativeOrbClient, sha256, tracking_summary
from bhl_robust.research.pose_metrics import Trajectory, score_trajectories, transform


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADAPTER = load_module("native_slam_adapter", ROOT / "scripts/bench/pose_research.py")
LAUNCH = load_module("native_slam_launcher", ROOT / "scripts/bench/h34_campaign.py")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def interpolate_truth(stamps, poses, target):
    """Evaluator-only SE(3) interpolation, without extrapolation or scale fit."""
    stamps = np.asarray(stamps, float)
    j = int(np.searchsorted(stamps, target))
    if j < len(stamps) and abs(stamps[j] - target) < 1e-9:
        return transform(poses[j])
    if j == 0 or j == len(stamps):
        return None
    a, b = transform(poses[j-1]), transform(poses[j])
    fraction = float((target - stamps[j-1]) / (stamps[j] - stamps[j-1]))
    relative = a[:3, :3].T @ b[:3, :3]
    theta = math.acos(float(np.clip((np.trace(relative)-1)/2, -1, 1)))
    if theta > math.pi - 1e-5:
        raise ValueError("reference rotation too large for local interpolation")
    if theta < 1e-8:
        rotation = np.eye(3)
    else:
        skew = (relative - relative.T) / (2*math.sin(theta))
        rotation = np.eye(3) + math.sin(fraction*theta)*skew + (1-math.cos(fraction*theta))*(skew @ skew)
    out = np.eye(4)
    out[:3, :3] = a[:3, :3] @ rotation
    out[:3, 3] = (1-fraction)*a[:3, 3] + fraction*b[:3, 3]
    return transform(out)


def segments(rows, *, method):
    """Never align across native map changes; lost frames stay in their segment."""
    result, current, prior_map = [], [], None
    for row in rows:
        map_id = row.get("map_id") if method == "orb" else row.get("map_reset_id")
        if map_id is not None and prior_map is not None and map_id != prior_map:
            result.append(current)
            current = []
        current.append(row)
        if map_id is not None:
            prior_map = map_id
    if current:
        result.append(current)
    return result


def evaluate(rows, *, method, replay_root, manifest, sequence, gate):
    """Only this evaluator opens truth, after all native inference has finished."""
    key = "T_W_C" if method == "orb" else "T_W_I"
    calibration = manifest["calibration"]
    extrinsic = transform(calibration["T_B_C" if method == "orb" else "T_B_I"])
    truth_times, truth_poses = [], []
    truth_hashes = {}
    for frame in sequence["frames"]:
        p = replay_root / frame["truth"]
        expected = manifest["file_sha256"][frame["truth"]]
        if sha256(p) != expected:
            raise ValueError("independent evaluator file hash mismatch")
        with np.load(p, allow_pickle=False) as truth:
            truth_poses.append(transform(truth[key]) @ np.linalg.inv(extrinsic))
        truth_times.append(frame["timestamp_s"])
        truth_hashes[frame["truth"]] = expected
    scored = []
    for part in segments(rows, method=method):
        est_times, est_poses, tracked = [], [], []
        reference_times, reference_poses = [], []
        for row in part:
            stamp = row["timestamp_s"]
            est_times.append(stamp)
            est_poses.append(transform(row[key]) @ np.linalg.inv(extrinsic) if row["tracked"] else np.eye(4))
            tracked.append(row["tracked"])
            truth = interpolate_truth(truth_times, truth_poses, stamp)
            if truth is not None:
                reference_times.append(stamp)
                reference_poses.append(truth)
        record = {"frames": len(part), "tracked_frames": sum(tracked), "metrics": None}
        try:
            if len(reference_times) < 2:
                raise ValueError("fewer than two covered reference samples")
            record["metrics"] = score_trajectories(
                Trajectory(np.array(est_times), np.array(est_poses), np.array(tracked, bool)),
                Trajectory(np.array(reference_times), np.array(reference_poses)),
                alignment="se3" if method == "orb" else "gravity_yaw_translation",
                max_difference_s=1e-8, end_time_s=est_times[-1]+1/manifest["frame_rate_hz"])
        except ValueError as error:
            record["unscorable_reason"] = str(error)
        scored.append(record)
    post_init = [r for r in rows if r["timestamp_s"] >= gate["initialization_exclusion_s"]]
    tracked_fraction = sum(r["tracked"] for r in post_init) / len(post_init) if post_init else 0.
    p95 = float(np.quantile([r["compute_seconds"] for r in rows], .95))
    metrics = scored[0]["metrics"] if len(scored) == 1 else None
    checks = {
        "post_initialization_tracking": tracked_fraction >= gate["minimum_tracking_fraction"],
        "one_continuous_native_map": len(scored) == 1,
        "ate_translation": bool(metrics and metrics["ate_translation_m"]["rmse"] <= gate["maximum_ate_rmse_m"]),
        "ate_rotation": bool(metrics and metrics["ate_rotation_deg"]["p95"] <= gate["maximum_rotation_p95_deg"]),
        "native_compute_p95": p95 <= gate["maximum_native_compute_p95_s"],
    }
    return {"segments": scored, "native_map_changes": len(scored)-1,
            "post_initialization_frames": len(post_init), "post_initialization_tracked_fraction": tracked_fraction,
            "native_compute_p95_ms": 1000*p95, "checks": checks,
            "replay_readiness_gate": "PASS" if all(checks.values()) else "NEGATIVE",
            "reference_interpolation": "evaluator-only bracketed linear translation and SO3 rotation; final uncovered LiDAR tail excluded",
            "truth_files_sha256": truth_hashes,
            "gate_scope": "limited simulated replay readiness, not navigation or hardware performance"}


def orb_replay(manifest_path, sequence, runtime, output, *, frame_limit=None):
    manifest, replay_root = ADAPTER.read_manifest(manifest_path)
    seq = ADAPTER.choose_sequence(manifest, sequence)
    output.mkdir(parents=True, exist_ok=False)
    ADAPTER.export_orb(manifest_path, sequence, output / "adapter")
    settings = output / "adapter/stereo.yaml"
    # Export is immutable audit evidence; duplicate original PNGs are transient.
    import shutil
    shutil.rmtree(output / "adapter/mav0")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(runtime / "lib")
    env["OMP_NUM_THREADS"] = "1"
    started = time.monotonic()
    rows = []
    with NativeOrbClient(runtime / "bin/orb_native", runtime / "ORBvoc.txt", settings,
                         output / "process", environment=env) as client:
        with (output / "native_frames.jsonl").open("x") as stream:
            for frame in seq["frames"][:frame_limit]:
                outer = time.monotonic()
                row = client.track(frame["timestamp_s"], ADAPTER.inference_path(replay_root, frame["left"]),
                                   ADAPTER.inference_path(replay_root, frame["right"]))
                row["client_wall_seconds"] = time.monotonic()-outer
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                stream.flush()
    result = tracking_summary(rows)
    result.update(method="orb_slam3_stereo", elapsed_seconds=time.monotonic()-started,
                  runtime_receipt_sha256=sha256(runtime / "runtime.json"),
                  vocabulary_sha256=sha256(runtime / "ORBvoc.txt"), settings_sha256=sha256(settings),
                  binary_sha256=sha256(runtime / "bin/orb_native"),
                  manifest_sha256=sha256(manifest_path), ground_truth_inputs=[], closed_loop_episodes=0,
                  input_files_sha256=ADAPTER._input_receipt(replay_root, seq, ("left", "right")),
                  client_wall_p95_ms=1000*float(np.quantile([r["client_wall_seconds"] for r in rows], .95)))
    write_json(output / "estimator_receipt.json", result)
    return rows, result


def run(protocol, *, phase, work, output, inputs):
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    methods = protocol["methods"]
    if not methods or len(set(methods)) != len(methods) or any(m not in {"orb", "lio"} for m in methods):
        raise ValueError("explicit unique native replay methods required")
    artifacts = [("capture", "capture_archive")] + [(method, method+"_runtime_archive") for method in methods]
    for name, member in artifacts:
        target = work / name
        target.mkdir()
        archive = inputs / protocol[member]
        expected = protocol["input_files"][protocol[member]]["sha256"]
        if sha256(archive) != expected:
            raise ValueError("frozen replay/runtime artifact hash mismatch")
        LAUNCH.safe_extract(archive, target)
        if name in {"orb", "lio"}:
            binary = target / "runtime" / ("bin/orb_native" if name == "orb" else "fastlio_headless")
            # The strict extractor deliberately creates mode0644 files. Only
            # this checksum-bound private runtime executable regains execution.
            binary.chmod(0o700)
    manifest_path = work / "capture/replay/manifest.json"
    manifest, replay_root = ADAPTER.read_manifest(manifest_path)
    chosen = manifest["sequences"][:1] if phase == "smoke" else manifest["sequences"]
    runs = []
    LIO = load_module("native_lio_replayer", ROOT / "scripts/native/lio_replay.py")
    for seq in chosen:
        for method in methods:
            target = output / seq["id"] / method
            target.parent.mkdir(parents=True, exist_ok=True)
            if method == "orb":
                rows, receipt = orb_replay(manifest_path, seq["id"], work / "orb/runtime", target,
                                           frame_limit=40 if phase == "smoke" else None)
            else:
                # Native LIO consumes original timed scans and high-rate IMU.
                receipt = LIO.replay(manifest_path, seq["id"], work / "lio/runtime", target)
                rows = [json.loads(line) for line in (target / "native_frames.jsonl").read_text().splitlines()]
            measurement = evaluate(rows, method=method, replay_root=replay_root, manifest=manifest,
                                   sequence=seq, gate=protocol["replay_gate"])
            write_json(target / "metrics.json", measurement)
            # Binary transport is reconstructable from original hashed sensors;
            # preserving it again would double the per-run artifact budget.
            (target / "input.bin").unlink(missing_ok=True)
            runs.append({"sequence": seq["id"], "split": seq["split"], "method": method,
                         "frames": len(rows), "tracked_frames": sum(r["tracked"] for r in rows),
                         "measurement": measurement, "receipt": str((target / "estimator_receipt.json").relative_to(output))})
            print(json.dumps({"sequence": seq["id"], "method": method,
                              "frames": len(rows), "readiness": measurement["replay_readiness_gate"]}), flush=True)
    # Smoke tests executable/transport operation, never counts as a scientific
    # success or a confirmation replicate. Lost states are legitimate outputs.
    status = "PASS" if phase == "smoke" or all(r["measurement"]["replay_readiness_gate"] == "PASS" for r in runs) else "NEGATIVE"
    return {"schema": "bhl-native-slam-campaign-v1", "status": status, "phase": phase,
            "scientific_status": "SMOKE_ONLY" if phase == "smoke" else "SIMULATED_NATIVE_REPLAY_MEASURED",
            "runs": runs, "closed_loop_navigation_episodes": 0,
            "sampling_unit": "three preassigned scene groups; one heldout scene, no population superiority claim",
            "input_artifacts": {key: protocol["input_files"][protocol[key]]["sha256"] for _, key in artifacts}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=["smoke", "replay"], required=True)
    args = parser.parse_args()
    output, work = Path(os.environ["H34_OUTPUT_DIR"]), Path(os.environ["H34_WORK_DIR"])
    inputs = Path(os.environ["H34_PROTOCOL"]).parent / "inputs"
    try:
        result = run(json.loads(args.protocol.read_text()), phase=args.phase, output=output, work=work, inputs=inputs)
    except Exception as error:
        import traceback
        traceback.print_exc()
        result = {"status": "INCOMPLETE", "phase": args.phase, "error": str(error)}
    write_json(output / "campaign_result.json", result)
    return 0 if result["status"] != "INCOMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
