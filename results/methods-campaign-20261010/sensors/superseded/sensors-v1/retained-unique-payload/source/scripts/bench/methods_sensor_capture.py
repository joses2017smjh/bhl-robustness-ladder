#!/usr/bin/env python3
"""Fresh, grouped sensor scenes with camera exposure at the LiDAR scan tail."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
spec = importlib.util.spec_from_file_location("methods_analytic_capture", Path(__file__).with_name("slam_capture_campaign.py"))
NATIVE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NATIVE)
BASE = NATIVE.BASE


def scene_variant(family, seed, texture):
    root = ET.fromstring(BASE.scene_xml(family, texture))
    root.find("visual/global").set("offwidth", "640")
    root.find("visual/global").set("offheight", "480")
    rng = np.random.default_rng(seed)
    for index, geom in enumerate(root.findall("worldbody/geom")):
        name = "ground" if index == 0 else f"wall_{index}" if index < 4 else f"surface_{index}"
        geom.set("name", name)
        if index >= 4:
            xyz = np.fromstring(geom.get("pos"), sep=" ")
            xyz[:2] += rng.uniform(-.12, .12, 2)
            geom.set("pos", " ".join(map(str, xyz)))
    return ET.tostring(root, encoding="unicode")


def independent_ground(model, data, pose, settings):
    """Evaluator-only downward geometry queries; no labels from input returns."""
    import mujoco
    x, y = settings.axes()
    height = np.full((len(y), len(x)), np.nan, np.float32)
    ground = np.zeros_like(height, bool)
    for iy, yy in enumerate(y):
        for ix, xx in enumerate(x):
            point = pose[:3, 3]+pose[:3, :3]@np.array([xx, yy, 0.])
            point[2] = 4.
            geom_id = np.array([-1], np.int32)
            distance = mujoco.mj_ray(model, data, point, np.array([0., 0., -1.]), None, 1, -1, geom_id)
            if distance < 0:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id[0])) or ""
            if name == "ground" or name.startswith("surface_"):
                height[iy, ix] = 4.-distance-pose[2, 3]+.75
                ground[iy, ix] = True
    step = np.zeros_like(height)
    support = np.zeros_like(ground)
    for axis in (0, 1):
        sl0, sl1 = [slice(None)]*2, [slice(None)]*2
        sl0[axis], sl1[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(sl0), tuple(sl1)
        good = ground[a] & ground[b]
        delta = np.where(good, np.abs(height[a]-height[b]), 0.)
        step[a], step[b] = np.maximum(step[a], delta), np.maximum(step[b], delta)
        support[a] |= good
        support[b] |= good
    hazard = (step >= settings.hazard_step_m) | (np.degrees(np.arctan(step/settings.resolution_m)) >= settings.hazard_slope_deg)
    return dict(ground_height_m=height, ground_mask=ground, ground_hazard=hazard,
                ground_hazard_support=support, ground_x_m=x, ground_y_m=y)


def capture(output, cell, *, frames, protocol):
    import torch
    if not os.getenv("SLURM_JOB_ID") or os.getenv("MUJOCO_GL") != "egl" or not torch.cuda.is_available():
        raise RuntimeError("allocated EGL GPU required for actual sensor rendering")
    if not 2 <= frames <= 150 or cell["condition"] not in ("nominal", "low_texture", "lidar_half"):
        raise ValueError("capture exceeds predeclared bounds")
    import cv2
    import mujoco
    from bhl_robust.research.terrain_methods import MapSettings
    from bhl_robust.research.stereo_benchmark import load_replay
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assets = output/"capture_assets"
    assets.mkdir()
    texture = assets/"texture.png"
    rng = np.random.default_rng(cell["geometry_seed"])
    pixels = rng.integers(30, 235, (256, 256, 3), dtype=np.uint8)
    if cell["condition"] == "low_texture":
        pixels[:] = 128
    if not cv2.imwrite(str(texture), pixels):
        raise RuntimeError("texture serialization failed")
    xml = scene_variant(cell["family"], cell["geometry_seed"], texture)
    (assets/"scene.xml").write_text(xml)
    model, calibration = mujoco.MjModel.from_xml_string(xml), BASE.calibration(640, 480)
    data = mujoco.MjData(model)
    BASE.rig_pose = NATIVE.rig_pose
    manifest = dict(schema="bhl-sensor-replay-v1", origin="simulation", lidar_dimension=3,
        data_kind="mujoco_rendered_rgb_timed_ray_lidar", clock_domain="simulation_time_monotonic_seconds",
        calibration=calibration, frame_rate_hz=5., imu_rate_hz=200., initial_stationary_s=5.,
        scope="fresh development variant within previously used scene families; kinematic replay, no robot navigation",
        grouping="geometry seed assigned before corruption; all conditions stay in the same development group",
        camera_scan_timing="image at scan tail; each ray retains actual acquisition time",
        protocol_sha256=hashlib.sha256(Path(protocol).read_bytes()).hexdigest(),
        cell=cell, sequences=[], file_sha256={},
        limitations=["ideal analytic IMU", "ideal pinhole", "ray LiDAR without measured intensity", "uncorrected scan motion in instantaneous terrain map"])
    sequence = dict(id=cell["id"], scene_id=cell["geometry_group"], split="dev", origin="simulation", frames=[])
    renderer = mujoco.Renderer(model, height=480, width=640)
    dropout_rng = np.random.default_rng(cell["geometry_seed"]+90000)
    try:
        for i in range(frames):
            scan_start, stamp = i/5., i/5.+.05
            pose = NATIVE.rig_pose(stamp)
            yaw = math.atan2(pose[1, 0], pose[0, 0])
            data.mocap_pos[0], data.mocap_quat[0] = pose[:3, 3], [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
            data.time = stamp
            mujoco.mj_forward(model, data)
            frame = dict(frame_id=f"{i:05d}", timestamp_s=stamp, left_timestamp_s=stamp, right_timestamp_s=stamp)
            for key, suffix in (("left", "png"), ("right", "png"), ("lidar", "npz"), ("imu", "npz"), ("truth", "npz")):
                namespace = "evaluator" if key == "truth" else "inference"
                frame[key] = f"{namespace}/{i:05d}_{key}.{suffix}"
                (output/frame[key]).parent.mkdir(exist_ok=True)
            for camera in ("left", "right"):
                renderer.update_scene(data, camera=camera)
                if not cv2.imwrite(str(output/frame[camera]), cv2.cvtColor(renderer.render().copy(), cv2.COLOR_RGB2BGR)):
                    raise RuntimeError("image serialization failed")
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera="left")
            depth = renderer.render().copy().astype(np.float32)
            renderer.disable_depth_rendering()
            depth[(depth < .1) | (depth > 12) | ~np.isfinite(depth)] = np.nan
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera="left")
            labels = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            obstacle = (labels[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (labels[..., 0] > 0)
            scan = BASE.lidar_scan(model, data, scan_start, columns=128)
            if cell["condition"] == "lidar_half":
                keep = dropout_rng.random(len(scan["points_xyz_m"])) >= .5
                for key in ("points_xyz_m", "point_time_s", "ring_index"):
                    scan[key] = scan[key][keep]
            if scan["point_time_s"].max() > stamp+1e-9:
                raise ValueError("future LiDAR point at camera exposure")
            np.savez_compressed(output/frame["lidar"], **scan)
            np.savez_compressed(output/frame["imu"], **NATIVE.imu_packet(scan_start, .2))
            np.savez_compressed(output/frame["truth"], depth_z_m=depth, obstacle_mask=obstacle,
                depth_truth_source=np.array("independent_camera_geometry"),
                T_W_C=pose@np.asarray(calibration["T_B_C"]), T_W_L=pose@np.asarray(calibration["T_B_L"]),
                T_W_I=pose@np.asarray(calibration["T_B_I"]), **independent_ground(model, data, pose, MapSettings()))
            sequence["frames"].append(frame)
            print(json.dumps({"capture_frame": i, "cell": cell["id"]}), flush=True)
    finally:
        renderer.close()
    manifest["sequences"] = [sequence]
    for path in sorted(output.rglob("*")):
        if path.is_file():
            manifest["file_sha256"][path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["payload_bytes"] = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if manifest["payload_bytes"] > 512*1024**2:
        raise ValueError("per-cell capture exceeds512MiB")
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    load_replay(output)
    return manifest
