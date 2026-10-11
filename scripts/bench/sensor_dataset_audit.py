#!/usr/bin/env python3
"""Audit retained sensor snapshots without reconstructing missing measurements.

Supports the repository's inspection episode/trace JSON and maze-video summary
JSON. Observed ages and cadence describe logged snapshots, not all policy ticks
or the nominal sensor publication rate. No simulator, image matcher, estimator,
training, or hardware device is run. Missing evidence fails replay readiness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


SCHEMA = "bhl-sensor-dataset-audit-v1"
FRESHNESS_S = .15  # Existing SensorTiming default; application choice, not a measurement.
DEFAULT_INPUTS = (
    "results/inspection-airlock-20260923/inspection-dropout035-s3-14.json",
    "results/inspection-airlock-20260923/inspection-wrongbranch-s3-14.json",
    "results/inspection_maze_deadend_probe.json",
    "results/inspection_maze_probe.json",
    "results/weekend-20260919/inspection-maze-failure-video.json",
    "results/weekend-20260919/inspection-maze-gate.json",
    "results/weekend-20260919/inspection-maze-video.json",
    "results/maze-explore-20260924/hard-6x6/seed1_sensors.json",
    "results/maze-humanoid-20260928/humanoid-hard-6x6/seed12_sensors.json",
)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vector(value, shape):
    """Strict shape/value validation without importing a numerical runtime."""
    if not shape:
        return number(value)
    return (isinstance(value, list) and len(value) == shape[0]
            and all(vector(child, shape[1:]) for child in value))


def flatten(value):
    for item in value:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def distribution(values):
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def audit_record(record, *, source="<memory>", source_sha256=None):
    if not isinstance(record, dict):
        raise ValueError("Expected an inspection or maze JSON object")
    metadata = record.get("sensor_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("sensor_metadata must be an object")
    if isinstance(record.get("episodes"), list):
        episodes, artifact_kind = record["episodes"], "inspection_episode_traces"
    elif isinstance(record.get("trace"), list) and "sensor_stats" in record:
        episodes, artifact_kind = [record], "maze_episode_summary"
    else:
        raise ValueError("Unsupported artifact: expected episodes[] or trace[] with sensor_stats")
    issues, detail = [], []
    count = {name: 0 for name in ("trace_rows", "sensor_snapshots", "missing_extero_snapshots",
        "stale_extero_snapshots", "fresh_extero_snapshots", "raw_lidar_snapshots", "lidar_sector_snapshots",
        "paired_ray_depth_snapshots", "imu_snapshots", "distinct_logged_extero_stamps")}
    extero_ages, imu_ages, gaps = [], [], []
    mode_counts = {}

    def problem(code, episode, row):
        issues.append({"code": code, "episode_index": episode, "trace_index": row})

    lidar_metadata, stereo_metadata = metadata.get("lidar", {}), metadata.get("stereo", {})
    sectors = lidar_metadata.get("sectors", 36) if isinstance(lidar_metadata, dict) else 36
    depth_side = stereo_metadata.get("side", 8) if isinstance(stereo_metadata, dict) else 8
    if not isinstance(sectors, int) or isinstance(sectors, bool) or sectors <= 0:
        raise ValueError("Invalid declared lidar sector count")
    if not isinstance(depth_side, int) or isinstance(depth_side, bool) or depth_side <= 0:
        raise ValueError("Invalid declared paired depth size")
    for ei, episode in enumerate(episodes):
        if not isinstance(episode, dict) or not isinstance(episode.get("trace"), list):
            raise ValueError(f"Episode {ei} has no trace array")
        mode = str(episode.get("sensor_mode", "undeclared"))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        previous_time, previous_capture, stamps = None, None, set()
        episode_rows = 0
        for ri, row in enumerate(episode["trace"]):
            count["trace_rows"] += 1
            if not isinstance(row, dict):
                problem("invalid_trace_row", ei, ri)
                continue
            now = row.get("time_s", row.get("t"))
            if not number(now):
                problem("missing_or_nonfinite_consumption_timestamp", ei, ri)
                continue
            if previous_time is not None:
                if now <= previous_time:
                    problem("nonincreasing_trace_timestamp", ei, ri)
                else:
                    gaps.append(now - previous_time)
            previous_time = now
            packet = row.get("sensors")
            if packet is None:
                continue
            if not isinstance(packet, dict):
                problem("invalid_sensor_snapshot", ei, ri)
                continue
            count["sensor_snapshots"] += 1
            episode_rows += 1
            stamp, fresh = packet.get("extero_stamp_s"), packet.get("extero_fresh")
            if not isinstance(fresh, bool):
                problem("missing_boolean_freshness_flag", ei, ri)
            if stamp is None:
                count["missing_extero_snapshots"] += 1
                if fresh is True:
                    problem("missing_packet_marked_fresh", ei, ri)
            elif not number(stamp):
                problem("nonfinite_extero_timestamp", ei, ri)
            else:
                age = now - stamp
                extero_ages.append(age)
                stamps.add(stamp)
                if stamp > now + 1.e-9:
                    problem("future_extero_timestamp", ei, ri)
                if previous_capture is not None and stamp < previous_capture - 1.e-9:
                    problem("regressing_extero_timestamp", ei, ri)
                previous_capture = stamp
                expected = 0 <= age <= FRESHNESS_S
                if isinstance(fresh, bool) and fresh != expected:
                    problem("freshness_flag_disagrees_with_logged_age", ei, ri)
                count["fresh_extero_snapshots" if expected else "stale_extero_snapshots"] += 1
                # An absent packet's zero-filled policy features are not raw measurements.
                for key, shape, maximum, name in (
                    ("lidar_sector_m", (sectors,), 12., "lidar_sector_snapshots"),
                    ("paired_idealized_depth_m", (2, depth_side, depth_side), 6., "paired_ray_depth_snapshots"),
                ):
                    values = packet.get(key)
                    if not vector(values, shape):
                        problem(f"invalid_{key}_shape_or_values", ei, ri)
                    else:
                        count[name] += 1
                        if any(not 0 < value <= maximum + 1.e-6 for value in flatten(values)):
                            problem(f"nonpositive_or_out_of_range_{key}", ei, ri)
                if "lidar_raw_m" in packet:
                    rays = lidar_metadata.get("rays") if isinstance(lidar_metadata, dict) else None
                    if not isinstance(rays, int) or not vector(packet["lidar_raw_m"], (rays,)):
                        problem("invalid_raw_lidar_shape_or_values", ei, ri)
                    else:
                        count["raw_lidar_snapshots"] += 1
            imu = packet.get("imu_gravity_gyro_specific_force_valid")
            if not vector(imu, (10,)) or imu[-1] not in (0, 1):
                problem("invalid_imu_feature_vector", ei, ri)
            else:
                count["imu_snapshots"] += 1
            imu_stamp = packet.get("imu_stamp_s")
            if not number(imu_stamp):
                problem("missing_or_nonfinite_imu_timestamp", ei, ri)
            else:
                imu_ages.append(now - imu_stamp)
                if imu_stamp > now + 1.e-9:
                    problem("future_imu_timestamp", ei, ri)
        count["distinct_logged_extero_stamps"] += len(stamps)
        detail.append({"episode_index": ei, "seed": episode.get("seed"), "sensor_mode": mode,
                       "trace_rows": len(episode["trace"]), "sensor_snapshots": episode_rows})

    ray_depth = count["paired_ray_depth_snapshots"] > 0
    declared_stereo = stereo_metadata.get("kind") if isinstance(stereo_metadata, dict) else None
    if ray_depth and declared_stereo != "paired_idealized_ray_depth_not_rgb_matching":
        issues.append({"code": "ray_depth_source_not_explicitly_declared", "episode_index": None, "trace_index": None})
    localization = metadata.get("localization")
    oracle = ("oracle" in str(localization).lower() or "oracle" in str(record.get("control", "")).lower())
    has_snapshots = count["sensor_snapshots"] > 0
    missing = ["raw lidar ranges and angles; sector minima cannot reconstruct a scan",
        "per-ray acquisition timestamps for scan deskew",
        "original left/right RGB images and per-eye capture timestamps",
        "measured stereo intrinsics, distortion and rectification with artifact hashes",
        "camera/lidar/IMU extrinsics and calibration uncertainty",
        "declared clock domain, capture-to-receive delay and synchronization method",
        "full sensor stream rather than sparse control trace snapshots",
        "explicit hit/invalid masks; maximum-range clipping is not an observed obstacle",
        "reference trajectory independent of the deployed estimator"]
    if not has_snapshots:
        missing.insert(0, "framewise sensor measurements (this file only retains summary/pose traces)")
    # This intentionally recognizes existing evidence only, rather than allowing a
    # caller to assert readiness through arbitrary new flags in legacy metadata.
    return {"input": source, "input_sha256": source_sha256, "artifact_kind": artifact_kind,
        "episodes": len(episodes), "episodes_by_sensor_mode": mode_counts, "observed": count,
        "logged_snapshot_spacing_s": distribution(gaps),
        "logged_extero_age_at_consumption_s": distribution(extero_ages),
        "logged_imu_age_at_consumption_s": distribution(imu_ages),
        "declared_sensor_metadata": metadata,
        "source_classification": {"depth": "simulated_paired_ray_depth" if ray_depth else "no_retained_depth_frames",
            "lidar": "sector_minima_only" if count["lidar_sector_snapshots"] and not count["raw_lidar_snapshots"]
                       else "raw_scan_present" if count["raw_lidar_snapshots"] else "no_retained_lidar_frames",
            "oracle_localization_declared": oracle, "localization": localization},
        "data_quality": {"verdict": "PASS" if not issues else "FAIL", "issues": issues,
                         "scope": "integrity and logged timestamp checks only; not sensor accuracy or independence"},
        "replay_readiness": {"verdict": "NOT_READY", "missing_evidence": missing,
            "usable_now": "audit sparse simulated policy/brake inputs" if has_snapshots else "episode outcome and controller trace analysis",
            "unsupported_claims": ["physical stereo depth accuracy", "stereo correspondence benchmark",
                "lidar scan matching or scan deskew benchmark", "sensor-to-host latency measurement",
                "estimated-pose navigation without oracle inputs"]},
        "episode_details": detail}


def audit_paths(paths, *, repo_root):
    root = Path(repo_root).resolve()
    records = []
    for filename in paths:
        path = Path(filename)
        if not path.is_absolute():
            path = root / path
        payload = path.read_bytes()
        try:
            name = str(path.resolve().relative_to(root))
        except ValueError:
            name = str(path.resolve())
        records.append(audit_record(json.loads(payload), source=name,
                                   source_sha256=hashlib.sha256(payload).hexdigest()))
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "repo_commit": commit,
        "scope": "read-only audit of existing published JSON; no new sensor capture, simulation or training",
        "freshness_rule_s": FRESHNESS_S, "freshness_rule_source": "src/bhl_robust/sensor_io.py SensorTiming default",
        "statistical_warning": "Files include reused probes/video episodes. Do not pool them as independent episodes. Cadence is sparse logging cadence, not measured publication Hz.",
        "summary": {"artifacts": len(records),
            "artifacts_with_sensor_snapshots": sum(r["observed"]["sensor_snapshots"] > 0 for r in records),
            "integrity_pass": sum(r["data_quality"]["verdict"] == "PASS" for r in records),
            "replay_ready": sum(r["replay_readiness"]["verdict"] == "READY" for r in records)},
        "artifacts": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="Existing inspection/maze JSON, relative to repo root")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True, help="New report JSON; never overwritten")
    args = parser.parse_args(argv)
    report = audit_paths(args.inputs or DEFAULT_INPUTS, repo_root=args.repo_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.out), **report["summary"]}))
    # Missing research prerequisites are informational; corrupted retained data
    # must fail automation, while preserving the written diagnostic report.
    return 0 if report["summary"]["integrity_pass"] == report["summary"]["artifacts"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
