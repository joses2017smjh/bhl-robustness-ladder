#!/usr/bin/env python3
"""Longer original stereo/timed-LiDAR/IMU replay for native estimators.

The rig is kinematic. Its analytic motion is used only for sensor generation
and evaluator truth, never as an estimated trajectory. Earlier pilot code and
its frozen measurements remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("native_capture_helpers", Path(__file__).with_name("sensor_capture.py"))
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


def motion(stamp):
    """Five stationary seconds then a smooth loop; return pose, gyro, force."""
    if not math.isfinite(stamp) or stamp < 0:
        raise ValueError("nonnegative finite simulation timestamp required")
    u = max(0., stamp - 5.)
    w, k = 2 * math.pi / 25., 2 * math.pi / 5.
    angle = w * (u - math.sin(k * u) / k)
    speed = w * (1 - math.cos(k * u))
    accel = w * k * math.sin(k * u)
    yaw = .15 * math.sin(angle)
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    position = [.45 * (1 - math.cos(angle)), .25 * math.sin(angle),
                .75 + .025 * (1 - math.cos(2 * angle))]
    acceleration = np.array([
        .45 * (math.cos(angle) * speed**2 + math.sin(angle) * accel),
        .25 * (-math.sin(angle) * speed**2 + math.cos(angle) * accel),
        .05 * (2 * math.cos(2 * angle) * speed**2 + math.sin(2 * angle) * accel)])
    gyro = np.array([0., 0., .15 * math.cos(angle) * speed])
    force = rotation.T @ (acceleration - np.array([0., 0., -9.81]))
    return BASE.transform(rotation, position), gyro, force


def rig_pose(stamp):
    return motion(stamp)[0]


def imu_packet(stamp, interval):
    times = np.arange(math.ceil(stamp * 200 - 1e-9),
                      math.ceil((stamp + interval) * 200 - 1e-9)) / 200.
    samples = [motion(float(t)) for t in times]
    return {"timestamp_s": times, "gyro_rad_s": np.stack([s[1] for s in samples]),
            "specific_force_m_s2": np.stack([s[2] for s in samples])}


def capture(output, *, seconds, rate=5., protocol=None):
    if not (0 < seconds <= 60 and rate == 5.):
        raise ValueError("predeclared capture bound: at most60s at5Hz")
    if not ((os.environ.get("SLURM_JOB_ID") or os.environ.get("H34_PROTOCOL"))
            and os.environ.get("MUJOCO_GL") == "egl"):
        raise RuntimeError("capture requires allocated Slurm GPU EGL context")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no allocated rendering GPU")
    import mujoco
    import cv2
    from bhl_robust.research.replay_contract import audit_replay
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "capture_assets"
    assets.mkdir()
    texture = assets / "fixed_texture.png"
    rng = np.random.default_rng(20261009)
    cv2.imwrite(str(texture), rng.integers(30, 235, (256, 256, 3), dtype=np.uint8))
    # The original helper projects each timed ray at its own generation pose.
    BASE.rig_pose = rig_pose
    calib = BASE.calibration(640, 480)
    manifest = {"schema": "bhl-sensor-replay-v1", "origin": "simulation",
        "data_kind": "mujoco_rendered_rgb_timed_ray_lidar", "lidar_dimension": 3,
        "clock_domain": "simulation_time_monotonic_seconds", "calibration": calib,
        "frame_rate_hz": rate, "imu_rate_hz": 200., "initial_stationary_s": 5.,
        "scope": "native_estimator_kinematic_replay_not_humanoid_navigation",
        "split_policy": "three_fixed_scene_groups_preassigned_before_rendering",
        "limitations": ["ideal_pinhole", "ideal_analytic_IMU", "ray_lidar_no_measured_intensity",
                        "50ms_uncompensated_raw_scans", "one_heldout_scene"],
        "runtime": {"host": platform.node(), "mujoco": mujoco.__version__,
                    "gpu": torch.cuda.get_device_name(0), "renderer": "egl"},
        "sequences": [], "file_sha256": {}}
    if protocol:
        manifest["protocol_sha256"] = hashlib.sha256(Path(protocol).read_bytes()).hexdigest()
    frames = int(round(seconds * rate))
    for scene, split in BASE.SCENES:
        xml = BASE.scene_xml(scene, texture).replace('offwidth="320" offheight="240"', 'offwidth="640" offheight="480"')
        (assets / (scene + ".xml")).write_text(xml)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        seq = {"id": scene + "_native", "scene_id": scene, "split": split,
               "origin": "simulation", "frames": []}
        renderer = mujoco.Renderer(model, height=480, width=640)
        try:
            for i in range(frames):
                stamp = i / rate
                pose = rig_pose(stamp)
                yaw = math.atan2(pose[1, 0], pose[0, 0])
                data.mocap_pos[0] = pose[:3, 3]
                data.mocap_quat[0] = [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
                data.time = stamp
                mujoco.mj_forward(model, data)
                prefix = f"{scene}/{i:05d}"
                frame = {"frame_id": f"{i:05d}", "timestamp_s": stamp,
                         "left_timestamp_s": stamp, "right_timestamp_s": stamp,
                         "left": f"inference/{prefix}_left.png", "right": f"inference/{prefix}_right.png",
                         "lidar": f"inference/{prefix}_lidar.npz", "imu": f"inference/{prefix}_imu.npz",
                         "truth": f"evaluator/{prefix}_truth.npz"}
                for key in ["left", "right", "lidar", "imu", "truth"]:
                    (output / frame[key]).parent.mkdir(parents=True, exist_ok=True)
                for camera in ["left", "right"]:
                    renderer.disable_depth_rendering()
                    renderer.update_scene(data, camera=camera)
                    image = renderer.render().copy()
                    if not cv2.imwrite(str(output / frame[camera]), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError("PNG serialization failed")
                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera="left")
                depth = renderer.render().copy().astype(np.float32)
                depth[(depth < .1) | (depth > 12) | ~np.isfinite(depth)] = np.nan
                renderer.disable_depth_rendering()
                np.savez_compressed(output / frame["lidar"], **BASE.lidar_scan(model, data, stamp, columns=128))
                np.savez_compressed(output / frame["imu"], **imu_packet(stamp, 1. / rate))
                np.savez_compressed(output / frame["truth"], depth_z_m=depth,
                    depth_truth_source=np.array("independent_camera_geometry"),
                    T_W_C=pose @ np.asarray(calib["T_B_C"]),
                    T_W_L=pose @ np.asarray(calib["T_B_L"]), T_W_I=pose @ np.asarray(calib["T_B_I"]))
                for key in ["left", "right", "lidar", "imu", "truth"]:
                    manifest["file_sha256"][frame[key]] = hashlib.sha256((output / frame[key]).read_bytes()).hexdigest()
                seq["frames"].append(frame)
        finally:
            renderer.close()
        manifest["sequences"].append(seq)
    for path in assets.iterdir():
        manifest["file_sha256"][path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if size > 512 * 1024**2:
        raise RuntimeError("replay payload exceeds512MiB cap")
    manifest["payload_bytes"] = size
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    audit = audit_replay(output)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return {"status": "PASS", "route": "native_slam_capture", "seconds_per_scene": seconds,
            "counts": audit["counts"], "manifest_sha256": audit["manifest_sha256"],
            "scientific_status": "CAPTURE_ONLY_NO_ESTIMATED_TRAJECTORIES"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--phase", choices=["smoke", "capture"], required=True)
    p.add_argument("--output", type=Path, default=Path(os.environ.get("H34_OUTPUT_DIR", "output")))
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=True)
    try:
        result = capture(a.output / "replay", seconds=1. if a.phase == "smoke" else 30., protocol=a.protocol)
    except Exception as exc:
        result = {"status": "INCOMPLETE", "error": str(exc), "route": "native_slam_capture"}
    (a.output / "campaign_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
