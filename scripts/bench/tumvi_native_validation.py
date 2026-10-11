#!/usr/bin/env python3
"""Bounded native stereo/IMU replay of an external TUM VI recording.

The input is the official calibrated EuRoC-format room1 export. This is an
offline recording experiment, never a robot deployment or own-hardware test.
Ground-truth poses never enter native inference. The Python provenance check
hashes selected dataset files, including reference CSV bytes, without parsing
their poses; only the separate evaluator parses reference poses.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bhl_robust.research.native_orb import NativeOrbClient, sha256, tracking_summary
from bhl_robust.research.native_orb_inertial import (
    ImuSettings, partition_imu, write_stereo_inertial_settings,
)
from bhl_robust.research.pose_metrics import transform

RUNTIME_SHA256 = "91ca4b6e4ea8f318998f400d4f7dfcd6fe8ae88b375399c79d5d633c39d71516"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def selected_timestamps(left, right):
    """Integer-clock selection, unchanged by native outcomes."""
    common = sorted(set(int(t) for t in left) & set(int(t) for t in right))
    if not common:
        raise ValueError("No exact original camera pairs")
    origin = common[0]
    selected = [t for t in common if t - origin < 60_000_000_000][:1200]
    return origin, selected


def relative_seconds(stamps, origin):
    return np.array([(int(t) - int(origin)) * 1e-9 for t in stamps], dtype=float)


def camera_matrix(camera):
    fx, fy, cx, cy = camera["intrinsics"]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)


def rectification(calibration):
    import cv2
    c0, c1 = calibration["cam0"], calibration["cam1"]
    for camera in (c0, c1):
        if camera["camera_model"] != "pinhole" or camera["distortion_model"] != "equidistant":
            raise ValueError("Official pinhole-equidistant model is required")
        if camera["resolution"] != [512, 512]:
            raise ValueError("Declared 512x512 calibration is required")
    t_c0_i, t_c1_i = transform(c0["T_cam_imu"]), transform(c1["T_cam_imu"])
    t_c1_c0 = t_c1_i @ np.linalg.inv(t_c0_i)
    if not np.allclose(t_c1_c0, transform(c1["T_cn_cnm1"]), atol=1e-8):
        raise ValueError("Independent stereo and IMU extrinsics disagree")
    k0, k1 = camera_matrix(c0), camera_matrix(c1)
    d0, d1 = np.array(c0["distortion_coeffs"]), np.array(c1["distortion_coeffs"])
    r0, r1, p0, p1, q = cv2.fisheye.stereoRectify(
        k0, d0, k1, d1, (512, 512), t_c1_c0[:3, :3], t_c1_c0[:3, 3],
        flags=cv2.CALIB_ZERO_DISPARITY, balance=0.0, fov_scale=1.0,
    )
    baseline = -float(p1[0, 3] / p1[0, 0])
    if baseline <= 0 or not np.allclose(p0[:3, :3], p1[:3, :3], atol=1e-8):
        raise ValueError("Native horizontal rectified stereo intrinsics are invalid")
    t_c0_c0rect = np.eye(4)
    t_c0_c0rect[:3, :3] = r0.T
    t_i_c0rect = transform(np.linalg.inv(t_c0_i) @ t_c0_c0rect)
    maps = [cv2.fisheye.initUndistortRectifyMap(k, d, r, p, (512, 512), cv2.CV_32FC1)
            for k, d, r, p in ((k0, d0, r0, p0), (k1, d1, r1, p1))]
    stereo = dict(fx=float(p0[0, 0]), fy=float(p0[1, 1]), cx=float(p0[0, 2]),
                  cy=float(p0[1, 2]), width=512, height=512, fps=20, baseline_m=baseline)
    receipt = {"T_imu_rectified_left_camera": t_i_c0rect.tolist(),
               "T_right_original_left_camera": t_c1_c0.tolist(),
               "R_left_rectification": r0.tolist(), "R_right_rectification": r1.tolist(),
               "P_left": p0.tolist(), "P_right": p1.tolist(), "Q": q.tolist(),
               "native_stereo_settings": stereo,
               "rectifier": "cv2.fisheye.stereoRectify; zero disparity, balance=0, fov_scale=1",
               "extrinsics_check": "T_cam1_imu @ inverse(T_cam0_imu) equals independent T_cn_cnm1"}
    return maps, stereo, t_i_c0rect, receipt


def fixed_uint16_to_uint8(image):
    if image is None or image.dtype != np.uint16 or image.shape != (512, 512):
        raise ValueError("Original TUM VI 512x512 uint16 monochrome image required")
    return (image >> 8).astype(np.uint8)


def read_imu(path, origin, end):
    rows, original_ns = [], []
    with Path(path).open() as stream:
        for fields in csv.reader(stream):
            if not fields or fields[0].startswith("#"):
                continue
            stamp = int(fields[0])
            if origin - 20_000_000 <= stamp <= end:
                original_ns.append(stamp)
                rows.append([(stamp - origin) * 1e-9, *map(float, fields[1:7])])
    if not rows:
        raise ValueError("No genuine original causal IMU rows")
    return np.asarray(rows), original_ns


def latency_stats(values):
    a = np.asarray(values, dtype=float)
    return {"count": len(a), "mean_ms": float(a.mean() * 1000),
            "p50_ms": float(np.quantile(a, .5) * 1000),
            "p95_ms": float(np.quantile(a, .95) * 1000),
            "max_ms": float(a.max() * 1000)} if len(a) else None


def summarize(frames, timings, mode):
    result = tracking_summary(frames)
    tracked = np.array([f["tracked"] for f in frames], dtype=bool)
    accepted = np.array([f["tracked"] and (mode == "stereo" or
                         f.get("inertial", {}).get("map_imu_initialized", False)) for f in frames])
    transitions = int(np.sum(tracked[:-1] & ~tracked[1:])) if len(tracked) > 1 else 0
    result.update({"mode": mode, "tracked_fraction": float(tracked.mean()) if len(tracked) else None,
                   "accepted_initialized_frames": int(accepted.sum()),
                   "initialization_first_sensor_time_s": next((f["timestamp_s"] for f, ok in zip(frames, accepted) if ok), None),
                   "tracking_loss_transitions_after_OK": transitions,
                   "native_compute": latency_stats([f["compute_seconds"] for f in frames]),
                   "pipeline_wall": latency_stats([t["pipeline_wall_s"] for t in timings]),
                   "rectification_image_io_wall": latency_stats([t["rectification_image_io_s"] for t in timings]),
                   "native_client_wall": latency_stats([t["native_client_wall_s"] for t in timings]),
                   "pipeline_scope": "original PNG read + fixed bit-depth conversion + rectification + PNG write + native client transport, IMU TSV I/O and native TrackStereo",
                   "latency_exclusions": "one-time startup, calibration map construction, evaluator and diagnostic JSON writes; offline unpaced recording replay",
                   "ate": {"status": "UNMEASURED", "reason": "Separate evaluator required; the replay runner hashes reference-file bytes for provenance but never parses their poses or supplies them to native inference."}})
    if mode == "stereo_inertial":
        result["native_map_imu_initialized_frames"] = sum(bool(f["inertial"]["map_imu_initialized"]) for f in frames)
        result["original_imu_rows_acknowledged"] = sum(f["inertial"]["imu_samples"] for f in frames)
    return result


def run(args):
    import cv2
    import yaml
    from native_stereo_navigation_campaign import unpack_runtime
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    if protocol["runtime_sha256"] != RUNTIME_SHA256 or protocol["dataset"] != "TUM VI calibrated room1 512x512 16-bit, EuRoC-format export":
        raise ValueError("Exact predeclared dataset and runtime required")
    source = json.loads(args.selection.read_text())
    stamps = source["selected_original_timestamps_ns"]
    origin, selected = selected_timestamps(stamps, stamps)
    if selected != stamps or origin != source["clock_origin_ns"]:
        raise ValueError("Dataset selection differs from declared integer-clock prefix")
    for name, expected in source["files"].items():
        path = args.dataset / name
        if not path.resolve().is_relative_to(args.dataset.resolve()) or sha256(path) != expected["sha256"]:
            raise ValueError("Original selected input hash mismatch")
    calibration = yaml.safe_load(args.calibration.read_text())
    maps, stereo, t_i_c, rect_receipt = rectification(calibration)
    write_json(args.output / "rectification.json", rect_receipt)
    samples, imu_ns = read_imu(args.dataset / "mav0/imu0/data.csv", origin, stamps[-1])
    times = relative_seconds(stamps, origin)
    batches, imu_receipt = partition_imu(times, samples)
    imu_receipt.update({"first_original_ns": imu_ns[0], "last_original_ns": imu_ns[-1],
                        "clock_origin_ns": origin, "no_additional_clock_offset": True,
                        "publisher_corrected_measurements": True})
    write_json(args.output / "imu-causality.json", imu_receipt)
    runtime = unpack_runtime(args.runtime_archive, RUNTIME_SHA256, args.output / "private-runtime")
    environment = {**os.environ, "LD_LIBRARY_PATH": str(runtime / "lib"),
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    summaries = []
    for mode in ("stereo", "stereo_inertial"):
        work = args.output / mode
        work.mkdir()
        settings = work / "settings.yaml"
        if mode == "stereo_inertial":
            noise = {key: protocol["imu_noise"][key]
                     for key in ("noise_gyro", "noise_acc", "gyro_walk", "acc_walk")}
            write_stereo_inertial_settings(settings, t_imu_camera=t_i_c,
                                           imu=ImuSettings(**noise), **stereo)
        else:
            from bhl_robust.research.native_orb import write_rectified_stereo_settings
            write_rectified_stereo_settings(settings, **stereo)
        frames, timings = [], []
        ranges = {"original_uint16_min": 65535, "original_uint16_max": 0,
                  "rectified_uint8_min": 255, "rectified_uint8_max": 0,
                  "conversion": "uint16 >> 8; fixed scale 1/256 with truncation"}
        began = time.monotonic()
        with NativeOrbClient(runtime / "bin/orb_native", runtime / "ORBvoc.txt", settings,
                             work / "private-native", environment=environment, sensor_mode=mode) as client:
            startup = time.monotonic() - began
            with (work / "frames.jsonl").open("x") as stream:
                for index, (stamp, timestamp) in enumerate(zip(stamps, times)):
                    began = time.monotonic()
                    outputs = []
                    for camera in (0, 1):
                        raw = cv2.imread(str(args.dataset / f"mav0/cam{camera}/data/{stamp}.png"), cv2.IMREAD_UNCHANGED)
                        converted = fixed_uint16_to_uint8(raw)
                        ranges["original_uint16_min"] = min(ranges["original_uint16_min"], int(raw.min()))
                        ranges["original_uint16_max"] = max(ranges["original_uint16_max"], int(raw.max()))
                        rectified = cv2.remap(converted, *maps[camera], cv2.INTER_LINEAR)
                        ranges["rectified_uint8_min"] = min(ranges["rectified_uint8_min"], int(rectified.min()))
                        ranges["rectified_uint8_max"] = max(ranges["rectified_uint8_max"], int(rectified.max()))
                        destination = work / f"current-cam{camera}.png"
                        if not cv2.imwrite(str(destination), rectified):
                            raise ValueError("Rectified image write failed")
                        outputs.append(destination)
                    image_end = time.monotonic()
                    frame = client.track(float(timestamp), *outputs,
                        **({"imu_samples": batches[index]} if mode == "stereo_inertial" else {}))
                    end = time.monotonic()
                    timing = {"pipeline_wall_s": end - began,
                              "rectification_image_io_s": image_end - began,
                              "native_client_wall_s": end - image_end}
                    frames.append(frame)
                    timings.append(timing)
                    stream.write(json.dumps({"original_timestamp_ns": stamp, "native": frame, "timing": timing}) + "\n")
                    stream.flush()
                    if (index + 1) % 100 == 0:
                        print(json.dumps({"mode": mode, "frames": index + 1}), flush=True)
            summary = summarize(frames, timings, mode)
            summary.update({"startup_wall_seconds": startup, "ready": client.ready,
                            "pixel_ranges": ranges, "frames_sha256": sha256(work / "frames.jsonl"),
                            "settings_sha256": sha256(settings)})
            write_json(work / "result.json", summary)
            summaries.append(summary)
    result = {"schema": "bhl-external-tumvi-result-v1", "status": "COMPLETED",
              "scope": protocol["scope"], "protocol_sha256": sha256(args.protocol),
              "selection_sha256": sha256(args.selection), "calibration_sha256": sha256(args.calibration),
              "source_sha256": sha256(Path(__file__)), "clock_origin_ns": origin,
              "paired_frames_per_arm": len(stamps), "window_last_sensor_time_s": float(times[-1]),
              "runtime_sha256": RUNTIME_SHA256, "arms": summaries,
              "simulation": False, "external_recorded_data": True, "own_hardware_validation": False,
              "lidar_validation": False, "physical_navigation_validation": False,
              "opencv_version": cv2.__version__, "original_data_loaded_by_estimator": "calibrated camera PNGs and genuine IMU only"}
    write_json(args.output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "selection", "calibration", "protocol", "runtime-archive", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "failure.json", {"status": "FAILED", "exception": type(error).__name__,
                   "message": str(error), "partial_native_outputs_retained": True})
        raise


if __name__ == "__main__":
    main()
