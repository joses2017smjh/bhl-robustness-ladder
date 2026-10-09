#!/usr/bin/env python3
"""Small rendered RGB + timed 3D ray-lidar sensor replay pilot.

This is a known kinematic rig in simulation. It does not execute humanoid gait,
SLAM, navigation, learned control, or physical-camera data acquisition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET

import numpy as np

R_B_C = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
RIG_HEIGHT = .75
BASELINE = .12
SCENES = (("textured_boxes", "dev"), ("thin_posts", "validation"), ("ramp_step", "test"))


def transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, translation
    return result


def rig_pose(stamp):
    yaw = .025 * math.sin(.6 * stamp)
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    position = np.array([.10 * stamp, .03 * math.sin(.6 * stamp), RIG_HEIGHT])
    return transform(rotation, position)


def calibration(width=320, height=240):
    f = .5 * height / math.tan(math.radians(60)/2)
    t_b_c = transform(R_B_C, [0., BASELINE/2, 0.])
    t_b_l = transform(R_B_C, [0., 0., 0.])
    t_c_l = np.linalg.inv(t_b_c) @ t_b_l
    t_m_l = transform(R_B_C, [0., 0., RIG_HEIGHT])
    return {"schema": "bhl-rectified-stereo-v1", "image_width": width, "image_height": height,
        "fx_px": f, "fy_px": f, "cx_left_px": (width-1)/2, "cx_right_px": (width-1)/2,
        "cy_px": (height-1)/2, "baseline_m": BASELINE, "rectified": True,
        "T_C_L": t_c_l.tolist(), "T_B_C": t_b_c.tolist(), "T_B_L": t_b_l.tolist(),
        "T_B_I": np.eye(4).tolist(), "T_M_L": t_m_l.tolist(),
        "transform_convention": "T_A_B_maps_B_coordinates_into_A",
        "camera_frame": "C_optical_x_right_y_down_z_forward",
        "lidar_frame": "L_optical_aligned_rig_center",
        "body_frame": "B_x_forward_y_left_z_up",
        "map_frame": "M_local_z_up",
        "map_frame_description": "instantaneous_rig_local_x_forward_y_left_z_up_ground_origin",
        "distortion": "none_simulated_pinhole", "stereo_exposure_offset_s": 0.0}


def scene_xml(scene_id, texture):
    geoms = {
        "textured_boxes": '<geom type="box" pos="2.2 .3 .55" size=".24 .32 .55" material="pattern"/>\n'
                         '<geom type="box" pos="3.4 -.65 .4" size=".35 .3 .4" material="pattern"/>',
        "thin_posts": '<geom type="cylinder" pos="1.9 .15 .55" size=".035 .55" material="pattern"/>\n'
                      '<geom type="cylinder" pos="2.7 -.35 .65" size=".055 .65" material="pattern"/>\n'
                      '<geom type="box" pos="4.5 .75 .45" size=".2 .3 .45" material="pattern"/>',
        "ramp_step": '<geom type="box" pos="2.3 .15 .15" size=".35 .5 .15" material="pattern"/>\n'
                     '<geom type="box" pos="3.9 -.6 .24" size=".8 .5 .1" euler="0 -.14 0" material="pattern"/>\n'
                     '<geom type="box" pos="5 .5 .45" size=".3 .25 .45" material="pattern"/>'
    }
    if scene_id not in geoms:
        raise ValueError("Unknown fixed scene")
    # Build path as XML attribute to avoid quoting malformed XML on unusual paths.
    root = ET.fromstring(f'''<mujoco model="bhl_sensor_{scene_id}">
      <compiler angle="radian"/>
      <option timestep=".002" gravity="0 0 -9.81"/>
      <visual><global offwidth="320" offheight="240"/><map znear=".01" zfar="20"/></visual>
      <asset><texture name="noise" type="2d"/><material name="pattern" texture="noise" texrepeat="4 4" texuniform="true"/></asset>
      <worldbody><light pos="0 0 5" dir="0 0 -1" diffuse=".8 .8 .8"/>
        <geom type="plane" size="15 10 .1" material="pattern"/>
        <geom type="box" pos="6 0 1.5" size=".05 4 1.5" material="pattern"/>
        <geom type="box" pos="3 2 1.5" size="3 .05 1.5" material="pattern"/>
        <geom type="box" pos="3 -2 1.5" size="3 .05 1.5" material="pattern"/>
        {geoms[scene_id]}
        <body name="sensor_rig" mocap="true" pos="0 0 .75">
          <camera name="left" pos="0 .06 0" xyaxes="0 -1 0 0 0 1" fovy="60"/>
          <camera name="right" pos="0 -.06 0" xyaxes="0 -1 0 0 0 1" fovy="60"/>
        </body>
      </worldbody></mujoco>''')
    root.find("asset/texture").set("file", str(Path(texture).resolve()))
    return ET.tostring(root, encoding="unicode")


def ray(model, data, position, direction):
    import mujoco
    geomid = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, np.asarray(position, dtype=np.float64),
        np.asarray(direction, dtype=np.float64), None, 1, -1, geomid)
    return float(distance)


def lidar_scan(model, data, stamp, *, rings=16, columns=64):
    elevations = np.linspace(math.radians(-40), math.radians(20), rings)
    azimuths = np.linspace(math.radians(-80), math.radians(80), columns)
    points, times, ring_ids = [], [], []
    for column, azimuth in enumerate(azimuths):
        point_stamp = stamp + .05 * column / max(columns-1, 1)
        pose = rig_pose(point_stamp)
        for ring, elevation in enumerate(elevations):
            # Ray is constructed in B, then represented in L optical coordinates.
            direction_b = np.array([math.cos(elevation)*math.cos(azimuth),
                                    math.cos(elevation)*math.sin(azimuth), math.sin(elevation)])
            direction_w = pose[:3, :3] @ direction_b
            distance = ray(model, data, pose[:3, 3], direction_w)
            if .1 <= distance <= 12:
                points.append(R_B_C.T @ direction_b * distance)
                times.append(point_stamp)
                ring_ids.append(ring)
    return {"points_xyz_m": np.asarray(points, dtype=np.float32).reshape(-1, 3),
            "point_time_s": np.asarray(times, dtype=np.float64),
            "ring_index": np.asarray(ring_ids, dtype=np.int16),
            "frame_timestamp_s": np.array(stamp),
            "scan_duration_s": np.array(.05), "deskewed": np.array(False)}


def imu_packet(stamp, interval):
    first = math.ceil(stamp * 200 - 1.e-9)
    last = math.ceil((stamp + interval) * 200 - 1.e-9)
    times = np.arange(first, last, dtype=np.float64) / 200.
    gyro, force = [], []
    for t in times:
        pose = rig_pose(float(t))
        acc_w = np.array([0., -.03*.6**2 * math.sin(.6*t), 0.])
        gyro.append([0., 0., .025*.6 * math.cos(.6*t)])
        force.append(pose[:3, :3].T @ (acc_w - np.array([0., 0., -9.81])))
    return {"timestamp_s": times.astype(np.float64), "gyro_rad_s": np.asarray(gyro, dtype=np.float64),
            "specific_force_m_s2": np.asarray(force, dtype=np.float64)}


def terrain_truth(model, data, stamp):
    """Independent dense vertical rays; NEVER derive labels from input lidar."""
    axis_x = np.linspace(.25, 5.0, 39)
    axis_y = np.linspace(-1.5, 1.5, 25)
    heights = np.full((len(axis_y), len(axis_x)), np.nan, dtype=np.float32)
    pose = rig_pose(stamp)
    for iy, y in enumerate(axis_y):
        for ix, x in enumerate(axis_x):
            p = pose[:3, 3] + pose[:3, :3] @ np.array([x, y, 0.])
            p[2] = 4.
            distance = ray(model, data, p, [0., 0., -1.])
            if distance >= 0:
                heights[iy, ix] = 4. - distance
    return {"terrain_x_m": axis_x.astype(np.float32), "terrain_y_m": axis_y.astype(np.float32),
            "terrain_height_m": heights, "terrain_grid_frame": np.array("M_local"),
            "terrain_truth_source": np.array("independent_dense_vertical_ray_queries")}


def capture(output, *, frames_per_scene=24, frame_rate=15., protocol=None):
    if not 1 <= frames_per_scene <= 60 or not 1 <= frame_rate <= 60:
        raise ValueError("Capture bounds: 1..60 frames/scene, 1..60 Hz")
    in_slurm = bool(os.environ.get("SLURM_JOB_ID"))
    in_frozen_launcher = bool(os.environ.get("H34_PROTOCOL") and os.environ.get("H34_OUTPUT_DIR"))
    if not (in_slurm or in_frozen_launcher) or os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("Rendering requires a Slurm GPU job and MUJOCO_GL=egl")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Capture execution context has no available CUDA GPU")
    import mujoco
    import cv2
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Capture output already contains a manifest")
    assets = output / "capture_assets"
    assets.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261008)
    noise = rng.integers(30, 235, size=(256, 256, 3), dtype=np.uint8)
    texture = assets / "fixed_texture.png"
    cv2.imwrite(str(texture), noise)
    manifest = {"schema": "bhl-sensor-replay-v1", "calibration": calibration(),
        "origin": "simulation", "data_kind": "mujoco_rendered_rgb_timed_ray_lidar",
        "lidar_dimension": 3, "clock_domain": "simulation_time_monotonic_seconds",
        "imu_rate_hz": 200., "frame_rate_hz": frame_rate,
        "split_policy": "fixed_scene_groups_declared_before_any_rendering",
        "obstacle_label_definition": "visible_non_plane_scene_geometry_pixels_excludes_ground_plane_not_semantic_traversability",
        "scope": "kinematic_sensor_pilot_not_humanoid_or_closed_loop_navigation",
        "runtime": {"mujoco": mujoco.__version__, "host": platform.node(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "renderer": "egl",
                    "execution_context": "slurm" if in_slurm else "frozen_h34_container",
                    "cuda_device": torch.cuda.get_device_name(0)},
        "limitations": ["ideal_pinhole_cameras", "deterministic_texture", "ideal_analytic_IMU",
                        "ray_lidar_no_intensity_or_return_physics", "uncompensated_50ms_scans",
                        "short_sequences_for_readiness_not_SLAM_accuracy_claims"],
        "sequences": [], "file_sha256": {}}
    if protocol:
        manifest["protocol_sha256"] = hashlib.sha256(Path(protocol).read_bytes()).hexdigest()
    interval = 1./frame_rate
    for scene_id, split in SCENES:
        xml = scene_xml(scene_id, texture)
        (assets / f"{scene_id}.xml").write_text(xml)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        sequence = {"id": scene_id + "_pilot", "scene_id": scene_id, "split": split,
                    "origin": "simulation", "frames": []}
        renderer = mujoco.Renderer(model, height=240, width=320)
        try:
            for frame_index in range(frames_per_scene):
                stamp = frame_index * interval
                pose = rig_pose(stamp)
                yaw = math.atan2(pose[1, 0], pose[0, 0])
                data.mocap_pos[0] = pose[:3, 3]
                data.mocap_quat[0] = [math.cos(yaw/2), 0., 0., math.sin(yaw/2)]
                data.time = stamp
                mujoco.mj_forward(model, data)
                prefix = f"{scene_id}/{frame_index:04d}"
                frame = {"frame_id": f"{frame_index:04d}", "timestamp_s": stamp,
                    "left_timestamp_s": stamp, "right_timestamp_s": stamp,
                    "left": f"inference/{prefix}_left.png", "right": f"inference/{prefix}_right.png",
                    "lidar": f"inference/{prefix}_lidar.npz", "imu": f"inference/{prefix}_imu.npz",
                    "truth": f"evaluator/{prefix}_truth.npz"}
                for key in ("left", "right", "lidar", "imu", "truth"):
                    (output/frame[key]).parent.mkdir(parents=True, exist_ok=True)
                for name in ("left", "right"):
                    renderer.disable_depth_rendering()
                    renderer.update_scene(data, camera=name)
                    image = renderer.render().copy()
                    if not cv2.imwrite(str(output/frame[name]), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError("Failed PNG serialization")
                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera="left")
                depth = renderer.render().copy().astype(np.float32)
                depth[(depth < .1) | (depth > 12) | ~np.isfinite(depth)] = np.nan
                renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera="left")
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                geom_ids = segmentation[..., 0]
                visible_geom = ((segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                                & (geom_ids >= 0) & (geom_ids < model.ngeom))
                clipped_ids = np.clip(geom_ids, 0, model.ngeom-1)
                obstacle_mask = (visible_geom & np.isfinite(depth)
                    & (model.geom_type[clipped_ids] != int(mujoco.mjtGeom.mjGEOM_PLANE)))
                np.savez_compressed(output/frame["lidar"], **lidar_scan(model, data, stamp))
                np.savez_compressed(output/frame["imu"], **imu_packet(stamp, interval))
                np.savez_compressed(output/frame["truth"], depth_z_m=depth,
                    depth_truth_source=np.array("independent_camera_geometry"),
                    depth_truth_engine=np.array("mujoco_renderer_metric_z"),
                    obstacle_mask=obstacle_mask.astype(bool),
                    obstacle_truth_source=np.array("independent_scene_geom_segmentation"),
                    T_W_C=pose @ np.asarray(manifest["calibration"]["T_B_C"]),
                    T_W_L=pose @ np.asarray(manifest["calibration"]["T_B_L"]),
                    **terrain_truth(model, data, stamp))
                for key in ("left", "right", "lidar", "imu", "truth"):
                    manifest["file_sha256"][frame[key]] = hashlib.sha256((output/frame[key]).read_bytes()).hexdigest()
                sequence["frames"].append(frame)
        finally:
            renderer.close()
        manifest["sequences"].append(sequence)
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if size > 128 * 1024 * 1024:
        raise RuntimeError("Sensor pilot exceeded 128MiB budget")
    manifest["payload_bytes"] = size
    for path in assets.iterdir():
        manifest["file_sha256"][path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    result = {"status": "PASS", "schema_version": 1, "route": "sensor_capture",
        "scientific_status": "SIMULATION_PILOT_CAPTURED", "physical_data": False,
        "frame_count": frames_per_scene*len(SCENES), "scene_groups": len(SCENES),
        "payload_bytes": size, "manifest_sha256": hashlib.sha256((output/"manifest.json").read_bytes()).hexdigest()}
    (output/"campaign_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames-per-scene", type=int, default=24)
    parser.add_argument("--frame-rate", type=float, default=15.)
    parser.add_argument("--protocol")
    args = parser.parse_args()
    try:
        result = capture(args.output, frames_per_scene=args.frames_per_scene, frame_rate=args.frame_rate, protocol=args.protocol)
    except Exception as exc:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        result = {"status": "INCOMPLETE", "route": "sensor_capture", "error": str(exc)}
        (Path(args.output)/"campaign_result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
        return 1
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
