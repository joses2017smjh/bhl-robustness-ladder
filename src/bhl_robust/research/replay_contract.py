"""Read-only integrity audit for calibrated replay inputs and evaluator files.

This observer may inspect evaluator metadata. Sensor inference functions must
only consume paths under inference/, not the observer's evaluation data.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np

from bhl_robust.stereo_depth import RectifiedCalibration


def member(root, name, namespace):
    if not isinstance(name, str) or "\\" in name:
        raise ValueError("invalid replay member")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != namespace:
        raise ValueError(f"replay member escapes {namespace} namespace")
    root = Path(root).resolve()
    target = root / name
    current = root
    for component in path.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("replay members cannot follow symbolic links")
    if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root):
        raise ValueError("missing, linked or escaping replay member")
    return target


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def transform(value):
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not math.isclose(float(np.linalg.det(matrix[:3, :3])), 1., abs_tol=1e-6)):
        raise ValueError("replay transform must be finite rigid SE(3)")
    return matrix


def audit_replay(root):
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "bhl-sensor-replay-v1":
        raise ValueError("unsupported replay manifest")
    if manifest.get("origin") not in {"simulation", "physical", "derived"}:
        raise ValueError("replay origin must be declared")
    calibration = manifest["calibration"]
    parsed = RectifiedCalibration.from_dict(calibration)
    if not finite_number(calibration.get("fy_px")) or calibration["fy_px"] <= 0 or not finite_number(calibration.get("cy_px")):
        raise ValueError("vertical intrinsics are required")
    transform(calibration["T_C_L"])
    if manifest.get("lidar_dimension") != 3:
        raise ValueError("3D replay pilot requires a 3D lidar declaration")
    if not manifest.get("clock_domain"):
        raise ValueError("clock domain is required")
    hashes = manifest["file_sha256"]
    if not isinstance(hashes, dict):
        raise ValueError("file checksums are required")
    for name, expected in hashes.items():
        namespace = PurePosixPath(name).parts[0] if isinstance(name, str) and PurePosixPath(name).parts else ""
        if namespace not in {"inference", "evaluator", "capture_assets"}:
            raise ValueError("unexpected recorded file namespace")
        path = member(root, name, namespace)
        if (not isinstance(expected, str) or len(expected) != 64
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError(f"replay file hash mismatch: {name}")
    sequence_ids, scene_splits, used_files = set(), {}, set()
    counts = {"sequences": 0, "frames": 0, "rgb_images": 0, "lidar_points": 0, "imu_samples": 0}
    splits, rates = {}, []
    shape = (parsed.image_height, parsed.image_width)
    for sequence in manifest["sequences"]:
        if sequence["id"] in sequence_ids:
            raise ValueError("duplicate sequence identity")
        sequence_ids.add(sequence["id"])
        scene, split = sequence["scene_id"], sequence["split"]
        if split not in {"dev", "development", "validation", "test"}:
            raise ValueError("invalid scene split")
        if scene in scene_splits and scene_splits[scene] != split:
            raise ValueError("scene leaks across splits")
        scene_splits[scene] = split
        if sequence.get("origin") != manifest["origin"] or not sequence["frames"]:
            raise ValueError("empty or inconsistently labeled sequence")
        counts["sequences"] += 1
        splits[split] = splits.get(split, 0) + 1
        previous_stamp, previous_imu, frame_ids = None, None, set()
        for frame in sequence["frames"]:
            stamp = frame["timestamp_s"]
            if not finite_number(stamp) or (previous_stamp is not None and stamp <= previous_stamp):
                raise ValueError("nonfinite or nonmonotonic frame timestamps")
            if frame["frame_id"] in frame_ids:
                raise ValueError("duplicate frame identity")
            frame_ids.add(frame["frame_id"])
            if previous_stamp is not None:
                rates.append(1. / (stamp - previous_stamp))
            previous_stamp = stamp
            for camera in ("left", "right"):
                camera_stamp = frame[camera + "_timestamp_s"]
                if not finite_number(camera_stamp) or abs(camera_stamp - stamp) > .035:
                    raise ValueError("invalid stereo capture timestamp")
            if abs(frame["left_timestamp_s"] - frame["right_timestamp_s"]) > .035:
                raise ValueError("stereo pair exposure difference exceeds 35 ms")
            paths = {}
            for key in ("left", "right", "lidar", "imu", "truth"):
                namespace = "evaluator" if key == "truth" else "inference"
                name = frame[key]
                if name in used_files:
                    raise ValueError("reused replay file identity")
                used_files.add(name)
                paths[key] = member(root, name, namespace)
                if hashlib.sha256(paths[key].read_bytes()).hexdigest() != hashes.get(name):
                    raise ValueError(f"replay file hash mismatch: {name}")
            if paths["left"] == paths["right"]:
                raise ValueError("stereo images must be separate captures")
            import cv2
            for camera in ("left", "right"):
                image = cv2.imread(str(paths[camera]), cv2.IMREAD_UNCHANGED)
                if image is None or image.dtype != np.uint8 or image.shape not in {shape, (*shape, 3)}:
                    raise ValueError("original stereo image dimensions/type do not match calibration")
            with np.load(paths["lidar"], allow_pickle=False) as scan:
                points = scan["points_xyz_m"]
                times = scan["point_time_s"]
                if (points.ndim != 2 or points.shape[1] != 3 or times.shape != (len(points),)
                        or not np.isfinite(points).all() or not np.isfinite(times).all()
                        or len(points) == 0 or np.any(np.diff(times) < 0)
                        or np.any(np.abs(times - stamp) > 1.0)):
                    raise ValueError("invalid raw timed 3D scan")
                counts["lidar_points"] += len(points)
            with np.load(paths["imu"], allow_pickle=False) as imu:
                times = imu["timestamp_s"]
                n = len(times)
                if (times.shape != (n,) or n == 0 or not np.isfinite(times).all()
                        or np.any(np.diff(times) <= 0) or (previous_imu is not None and times[0] <= previous_imu)
                        or imu["gyro_rad_s"].shape != (n, 3) or imu["specific_force_m_s2"].shape != (n, 3)
                        or not np.isfinite(imu["gyro_rad_s"]).all() or not np.isfinite(imu["specific_force_m_s2"]).all()):
                    raise ValueError("invalid SI-unit IMU stream")
                previous_imu = float(times[-1])
                counts["imu_samples"] += n
            with np.load(paths["truth"], allow_pickle=False) as truth:
                depth = truth["depth_z_m"]
                if depth.shape != shape or not np.isfinite(depth).any() or np.any(depth[np.isfinite(depth)] <= 0):
                    raise ValueError("invalid independent evaluator optical depth")
                transform(truth["T_W_C"])
                transform(truth["T_W_L"])
            counts["frames"] += 1
            counts["rgb_images"] += 2
    if not counts["frames"] or not {"validation", "test"}.issubset(splits) or not ({"dev", "development"} & splits.keys()):
        raise ValueError("pilot requires whole-scene development, validation and test splits")
    return {"schema": "bhl-sensor-replay-audit-v1", "status": "PASS", "origin": manifest["origin"],
            "scope": "file integrity, timing and geometry declarations; not sensor accuracy or estimator success",
            "counts": counts, "scene_groups": len(scene_splits), "splits": splits,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "verified_files": len(used_files), "observed_frame_rate_hz_median": float(np.median(rates)) if rates else None}
