#!/usr/bin/env python3
"""Prepare real stereo/LIO replay inputs and score independently supplied poses.

No estimator output is manufactured by this script. Intake/export jobs can
complete while the scientific comparison remains blocked on native runtimes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import time

import numpy as np

from bhl_robust.research.pose_metrics import METHODS, Trajectory, score_trajectories, transform


UPSTREAM = {
    "orb_slam3_stereo": {"repository": "https://github.com/UZ-SLAMLab/ORB_SLAM3",
                         "commit": "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"},
    "fast_lio2_lidar_imu": {"repository": "https://github.com/hku-mars/FAST_LIO",
                           "commit": "7cc4175de6f8ba2edf34bab02a42195b141027e9"},
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def inference_path(root: Path, relative: str) -> Path:
    """Inference readers cannot follow links or paths into evaluator data."""
    if not isinstance(relative, str):
        raise ValueError("input path must be relative text")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] != "inference":
        raise ValueError("input must be inside the inference namespace")
    p = root
    for component in rel.parts:
        p = p / component
        if p.is_symlink():
            raise ValueError("inference paths may not follow symbolic links")
    if not p.is_file():
        raise ValueError(f"missing inference input: {relative}")
    return p


def _numeric(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"invalid {name}")
    return float(value)


def verify_input_hash(manifest, relative, path):
    expected = manifest.get("file_sha256", {}).get(relative)
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("original inference file checksum is required")
    if expected != sha256(path):
        raise ValueError("original inference hash differs from dataset receipt")


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "bhl-sensor-replay-v1":
        raise ValueError("expected bhl-sensor-replay-v1 manifest")
    calibration = manifest.get("calibration", {})
    if calibration.get("schema") != "bhl-rectified-stereo-v1" or calibration.get("rectified") is not True:
        raise ValueError("calibrated, pre-rectified stereo is required")
    for k in ("image_width", "image_height"):
        if type(calibration.get(k)) is not int or calibration[k] < 1:
            raise ValueError(f"invalid {k}")
    for k in ("fx_px", "fy_px", "baseline_m"):
        _numeric(calibration.get(k), k, positive=True)
    for k in ("cx_left_px", "cx_right_px", "cy_px"):
        _numeric(calibration.get(k), k)
    if not manifest.get("clock_domain"):
        raise ValueError("explicit shared clock_domain required")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("one or more recorded sequences required")
    ids, scene_splits = set(), {}
    for sequence in sequences:
        sid = sequence.get("id")
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", sid) or sid in ids:
            raise ValueError("sequence IDs must be unique safe names")
        ids.add(sid)
        if sequence.get("split") not in ("dev", "validation", "test"):
            raise ValueError("each sequence requires a declared split")
        scene = sequence.get("scene_id")
        if not isinstance(scene, str) or not scene:
            raise ValueError("declared scene grouping is required")
        if scene in scene_splits and scene_splits[scene] != sequence["split"]:
            raise ValueError("scene leaks across dataset splits")
        scene_splits[scene] = sequence["split"]
        if sequence.get("origin") not in ("simulation", "physical", "external"):
            raise ValueError("each sequence requires declared data origin")
        frames = sequence.get("frames")
        if not isinstance(frames, list) or len(frames) < 2:
            raise ValueError("at least two original image frames per sequence required")
        stamps = [_numeric(f.get("timestamp_s"), "timestamp_s") for f in frames]
        if np.any(np.diff(stamps) <= 0):
            raise ValueError("capture times must be strictly increasing")
        for f in frames:
            for key in ("left", "right"):
                image_path = inference_path(path.parent, f.get(key))
                with image_path.open("rb") as image:
                    header = image.read(24)
                if (len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n"
                        or header[12:16] != b"IHDR"
                        or struct.unpack(">II", header[16:24]) != (calibration["image_width"], calibration["image_height"])):
                    raise ValueError("original PNG resolution must match calibration")
                verify_input_hash(manifest, f[key], image_path)
            for key in ("lidar", "imu"):
                if key in f:
                    sensor_path = inference_path(path.parent, f[key])
                    verify_input_hash(manifest, f[key], sensor_path)
    return manifest, path.parent


def choose_sequence(manifest, sequence_id):
    for sequence in manifest["sequences"]:
        if sequence["id"] == sequence_id:
            return sequence
    raise ValueError(f"unknown sequence: {sequence_id}")


def _new_directory(path):
    path = Path(path)
    if path.exists():
        raise ValueError("output directory must be new")
    path.mkdir(parents=True)
    return path


def _input_receipt(root, sequence, keys=("left", "right", "lidar", "imu")):
    paths = sorted({f[k] for f in sequence["frames"] for k in keys if k in f})
    return {relative: sha256(inference_path(root, relative)) for relative in paths}


def export_orb(manifest_path, sequence_id, out):
    manifest, root = read_manifest(manifest_path)
    seq = choose_sequence(manifest, sequence_id)
    c = manifest["calibration"]
    # The pinned native rectified-stereo path uses u_left-u_right directly.
    # Avoid silently scoring cameras with an unsupported principal-point shift.
    if not math.isclose(c["cx_left_px"], c["cx_right_px"], abs_tol=1e-7):
        raise ValueError("ORB rectified input requires matching cx; re-rectify before export")
    out = _new_directory(out)
    for camera in ("cam0", "cam1"):
        (out / "mav0" / camera / "data").mkdir(parents=True)
    timestamp_ns = []
    for f in seq["frames"]:
        ns = int(round(f["timestamp_s"] * 1e9))
        if timestamp_ns and ns <= timestamp_ns[-1]:
            raise ValueError("nanosecond image timestamp collision")
        timestamp_ns.append(ns)
        for source_key, cam in (("left", "cam0"), ("right", "cam1")):
            shutil.copyfile(inference_path(root, f[source_key]), out / "mav0" / cam / "data" / f"{ns}.png")
    (out / "timestamps.txt").write_text("\n".join(map(str, timestamp_ns)) + "\n")
    measured_fps = 1 / float(np.median(np.diff([f["timestamp_s"] for f in seq["frames"]])))
    fps = int(round(measured_fps))
    if fps < 1:
        raise ValueError("ORB-SLAM3 requires a positive integer nominal Camera.fps")
    settings = ["%YAML:1.0", 'File.version: "1.0"', 'Camera.type: "Rectified"']
    for i in (1, 2):
        settings += [f"Camera{i}.fx: {float(c['fx_px'])}", f"Camera{i}.fy: {float(c['fy_px'])}",
                     f"Camera{i}.cx: {float(c['cx_left_px'])}", f"Camera{i}.cy: {float(c['cy_px'])}"]
        settings += [f"Camera{i}.{k}: 0.0" for k in ("k1", "k2", "p1", "p2")]
    settings += [f"Camera.width: {c['image_width']}", f"Camera.height: {c['image_height']}",
                 f"Camera.fps: {fps}", "Camera.RGB: 1", "Stereo.ThDepth: 40.0", f"Stereo.b: {float(c['baseline_m'])}",
                 "Stereo.T_c1_c2: !!opencv-matrix", "  rows: 4", "  cols: 4", "  dt: f",
                 f"  data: [1,0,0,{c['baseline_m']},0,1,0,0,0,0,1,0,0,0,0,1]",
                 "ORBextractor.nFeatures: 1200", "ORBextractor.scaleFactor: 1.2",
                 "ORBextractor.nLevels: 8", "ORBextractor.iniThFAST: 20", "ORBextractor.minThFAST: 7",
                 "Viewer.KeyFrameSize: 0.05", "Viewer.KeyFrameLineWidth: 1.0", "Viewer.GraphLineWidth: 0.9",
                 "Viewer.PointSize: 2.0", "Viewer.CameraSize: 0.08", "Viewer.CameraLineWidth: 3.0",
                 "Viewer.ViewpointX: 0.0", "Viewer.ViewpointY: -0.7", "Viewer.ViewpointZ: -1.8",
                 "Viewer.ViewpointF: 500.0", "Viewer.imageViewScale: 1.0"]
    (out / "stereo.yaml").write_text("\n".join(settings) + "\n")
    receipt = {"schema": "bhl-pose-replay-adapter-v1", "method": "orb_slam3_stereo",
               "status": "READY_INPUTS_WAITING_RUNTIME", "scientific_status": "BLOCKED_RUNTIME",
               "upstream": UPSTREAM["orb_slam3_stereo"], "manifest_sha256": sha256(manifest_path),
               "sequence": sequence_id, "origin": seq["origin"], "split": seq["split"],
               "clock_domain": manifest["clock_domain"], "images": len(timestamp_ns),
               "measured_frame_rate_hz": measured_fps, "native_nominal_camera_fps": fps,
               "input_files_sha256": _input_receipt(root, seq, ("left", "right")), "ground_truth_inputs": [],
               "command_argv_template": ["<ORB_SLAM3>/Examples/Stereo/stereo_euroc", "<ORB_SLAM3>/Vocabulary/ORBvoc.txt",
                                          "stereo.yaml", ".", "timestamps.txt", sequence_id],
               "runtime_required": ["pinned compiled ORB-SLAM3", "hashed ORB vocabulary", "OpenCV C++", "Eigen3", "Pangolin display or verified headless frontend"],
               "limitations": ["stock stereo_euroc starts a viewer", "stock EuRoC output omits lost frames; record per-frame tracking states separately",
                               "input preparation does not measure drift or demonstrate closed-loop control"]}
    receipt["export_files_sha256"] = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out / "adapter.json", receipt)
    return receipt


def _arrays(path, required, optional=()):
    with np.load(path, allow_pickle=False) as data:
        if any(k.lower().startswith(("t_w_", "truth", "pose", "ground_truth")) for k in data.files):
            raise ValueError("evaluator/pose arrays are prohibited in inference sensor input")
        if not set(required).issubset(data.files):
            raise ValueError(f"missing raw sensor arrays: {sorted(set(required) - set(data.files))}")
        return {k: np.array(data[k]) for k in (*required, *optional) if k in data.files}


def fastlio_packets(manifest_path, sequence_id):
    manifest, root = read_manifest(manifest_path)
    if manifest.get("lidar_dimension") != 3:
        raise ValueError("FAST-LIO2 route requires timed 3D LiDAR")
    seq = choose_sequence(manifest, sequence_id)
    t_b_l = transform(manifest["calibration"].get("T_B_L"))
    t_b_i = transform(manifest["calibration"].get("T_B_I", np.eye(4)))
    t_i_l = np.linalg.inv(t_b_i) @ t_b_l
    scan_packets, imu_by_time = [], {}
    last_scan_stamp = -math.inf
    for frame in seq["frames"]:
        cloud = _arrays(inference_path(root, frame.get("lidar")), ("points_xyz_m", "point_time_s", "ring_index"), ("intensity",))
        xyz = cloud["points_xyz_m"]
        if "intensity" in cloud:
            intensity_kind = "recorded_input_field_semantics_require_sensor_declaration"
        elif seq["origin"] == "simulation":
            # FAST's compatible ROS structure requires the field, although
            # this ideal ray sensor has no return-intensity physics.
            cloud["intensity"] = np.zeros(len(xyz), dtype=np.float32)
            intensity_kind = "unmeasured_zero_placeholder_for_ROS_field"
        else:
            raise ValueError("physical/external raw intensity is missing; no silent placeholder permitted")
        stamps, ring, intensity = (cloud[k] for k in ("point_time_s", "ring_index", "intensity"))
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or any(a.shape != (len(xyz),) for a in (stamps, ring, intensity)):
            raise ValueError("invalid timed 3D scan shape")
        if not all(np.isfinite(a).all() for a in (xyz, stamps, ring, intensity)) or np.any(np.linalg.norm(xyz, axis=1) <= 0):
            raise ValueError("only finite, positive-range raw scan returns are allowed")
        if np.any(ring < 0) or np.any(ring > 127) or np.any(ring != ring.astype(int)):
            raise ValueError("ring indices must be integers in [0,127]")
        order = np.argsort(stamps, kind="stable")
        stamp = float(stamps.min())
        offsets = stamps[order] - stamp
        if stamp <= last_scan_stamp or offsets.max() <= 0:
            raise ValueError("scans require increasing capture times and original nonzero per-point offsets")
        last_scan_stamp = stamp
        scan_packets.append({"kind": "pointcloud2", "capture_time_s": stamp, "frame_id": "lidar",
                             "intensity_kind": intensity_kind,
                             "fields": ["x", "y", "z", "intensity", "time", "ring"],
                             "field_units": ["m", "m", "m", "sensor_native_or_documented_placeholder", "s_offset_from_header", "index"],
                             "points": [[*map(float, xyz[i]), float(intensity[i]), float(offsets[j]), int(ring[i])] for j, i in enumerate(order)]})
        imu = _arrays(inference_path(root, frame.get("imu")), ("timestamp_s", "gyro_rad_s", "specific_force_m_s2"))
        ts, gyro, accel = (imu[k] for k in ("timestamp_s", "gyro_rad_s", "specific_force_m_s2"))
        if ts.ndim != 1 or not len(ts) or gyro.shape != (len(ts), 3) or accel.shape != gyro.shape or np.any(np.diff(ts) <= 0):
            raise ValueError("IMU requires increasing original timestamps and Nx3 gyro/specific force")
        if not all(np.isfinite(a).all() for a in (ts, gyro, accel)):
            raise ValueError("finite IMU SI-unit samples required")
        for t, g, a in zip(ts, gyro, accel):
            sample = {"kind": "imu", "capture_time_s": float(t), "frame_id": "imu_body",
                      "gyro_rad_s": g.tolist(), "specific_force_m_s2": a.tolist()}
            if float(t) in imu_by_time and sample != imu_by_time[float(t)]:
                raise ValueError("conflicting IMU sample at reused timestamp")
            imu_by_time[float(t)] = sample
    imu_times = sorted(imu_by_time)
    if len(imu_times) < 2 or imu_times[0] > scan_packets[0]["capture_time_s"] or imu_times[-1] < max(p["capture_time_s"] + max(row[4] for row in p["points"]) for p in scan_packets):
        raise ValueError("IMU must span all original scan point times")
    max_imu_gap = float(np.max(np.diff(imu_times)))
    if max_imu_gap > .02 + 1e-8:
        raise ValueError("IMU gaps above 20 ms are unsuitable for the declared high-rate route")
    packets = sorted([*scan_packets, *imu_by_time.values()], key=lambda p: (p["capture_time_s"], 0 if p["kind"] == "imu" else 1))
    return manifest, seq, t_i_l, packets, max_imu_gap


def export_fastlio(manifest_path, sequence_id, out):
    manifest, seq, t_i_l, packets, max_imu_gap = fastlio_packets(manifest_path, sequence_id)
    out = _new_directory(out)
    with (out / "packets.jsonl").open("w") as stream:
        for packet in packets:
            stream.write(json.dumps(packet, allow_nan=False, separators=(",", ":")) + "\n")
    clouds = [p for p in packets if p["kind"] == "pointcloud2"]
    max_ring = max(row[5] for p in clouds for row in p["points"])
    intensity_kinds = sorted({p["intensity_kind"] for p in clouds})
    scan_rate = int(round(1 / float(np.median(np.diff([p["capture_time_s"] for p in clouds])))))
    config = {"common": {"lid_topic": "/bhl/lidar", "imu_topic": "/bhl/imu", "time_sync_en": False, "time_offset_lidar_to_imu": 0.0},
              "preprocess": {"lidar_type": 2, "scan_line": max_ring + 1, "scan_rate": scan_rate, "timestamp_unit": 0, "blind": .1},
              "mapping": {"extrinsic_est_en": False, "extrinsic_T": t_i_l[:3, 3].tolist(), "extrinsic_R": t_i_l[:3, :3].reshape(-1).tolist()},
              "publish": {"path_en": True, "scan_publish_en": False}, "pcd_save": {"pcd_save_en": False}}
    # JSON is valid YAML and preserves typed values without a PyYAML dependency.
    write_json(out / "fastlio.yaml", config)
    receipt = {"schema": "bhl-pose-replay-adapter-v1", "method": "fast_lio2_lidar_imu",
               "status": "READY_INPUTS_WAITING_RUNTIME", "scientific_status": "BLOCKED_RUNTIME",
               "upstream": UPSTREAM["fast_lio2_lidar_imu"], "manifest_sha256": sha256(manifest_path),
               "sequence": sequence_id, "origin": seq["origin"], "split": seq["split"],
               "clock_domain": manifest["clock_domain"], "scan_packets": len(clouds),
               "imu_packets": len(packets) - len(clouds), "max_imu_gap_s": max_imu_gap,
               "input_files_sha256": _input_receipt(Path(manifest_path).resolve().parent, seq, ("lidar", "imu")), "ground_truth_inputs": [],
               "T_imu_lidar": t_i_l.tolist(), "pointcloud_schema": "Velodyne-compatible XYZ,intensity,time(float32 seconds),ring(uint16)",
               "intensity_kind": intensity_kinds[0] if len(intensity_kinds) == 1 else "mixed_explicit_packet_declarations",
               "intensity_kinds": intensity_kinds,
               "runtime_required": ["ROS >= Melodic", "PCL >=1.8", "Eigen >=3.3.4", "livox_ros_driver build dependency", "pinned FAST-LIO2 plus pinned ikd-Tree"],
               "commands": ["roscore", "rosparam load fastlio.yaml", "rosrun fast_lio fastlio_mapping",
                            "python scripts/bench/pose_research.py publish-fastlio --packets packets.jsonl --out publisher-receipt.json"],
               "limitations": ["simulated intensity placeholder has no calibrated reflectivity meaning" if seq["origin"] == "simulation" else "record original intensity semantics",
                               "no synthesized point times or zero-time scan substitution", "native odometry and closed-loop outcomes remain unmeasured"]}
    receipt["export_files_sha256"] = {p.name: sha256(p) for p in out.iterdir() if p.is_file()}
    write_json(out / "adapter.json", receipt)
    return receipt


def runtime_inventory(orb_executable=None):
    executable = Path(orb_executable) if orb_executable else None
    return {"orb_executable_exists": bool(executable and executable.is_file()),
            "orb_executable_sha256": sha256(executable) if executable and executable.is_file() else None,
            "roscore": shutil.which("roscore"), "roslaunch": shutil.which("roslaunch"),
            "catkin_make": shutil.which("catkin_make"),
            "python_rospy": importlib.util.find_spec("rospy") is not None}


def intake(manifest_path, out, *, export_inputs=True, orb_executable=None):
    manifest, _ = read_manifest(manifest_path)
    out = _new_directory(out)
    records = []
    for seq in manifest["sequences"]:
        record = {"sequence": seq["id"], "split": seq["split"], "origin": seq["origin"],
                  "images": len(seq["frames"]), "duration_s": seq["frames"][-1]["timestamp_s"] - seq["frames"][0]["timestamp_s"]}
        for key, exporter in (("stereo_slam", export_orb), ("lidar_imu", export_fastlio)):
            try:
                if export_inputs:
                    receipt = exporter(manifest_path, seq["id"], out / seq["id"] / key)
                    record[key] = {"status": receipt["status"], "adapter": f"{seq['id']}/{key}/adapter.json"}
                else:
                    if key == "lidar_imu": fastlio_packets(manifest_path, seq["id"])
                    record[key] = {"status": "READY_INPUTS_WAITING_RUNTIME"}
            except (ValueError, KeyError, OSError) as error:
                record[key] = {"status": "BLOCKED_CALIBRATED_DATA", "reason": str(error)}
        record["slam_protocol_suitability"] = "PILOT_ONLY_SHORT_SEQUENCE" if record["duration_s"] < 20 else "DURATION_READY_OTHER_ACCEPTANCE_PENDING"
        records.append(record)
    result = {"schema": "bhl-pose-intake-v1", "status": "PASS", "scientific_status": "BLOCKED_RUNTIME",
              "manifest_sha256": sha256(manifest_path), "sequences": records,
              "runtime_inventory": runtime_inventory(orb_executable), "estimated_pose_outputs": 0,
              "closed_loop_episodes": 0, "ground_truth_inputs_to_estimators": [],
              "next_step": "Build isolated pinned native runtimes, record complete per-frame tracking status, then run held-out estimator replay before closed-loop navigation."}
    if not all(r["stereo_slam"]["status"] == "READY_INPUTS_WAITING_RUNTIME" and r["lidar_imu"]["status"] == "READY_INPUTS_WAITING_RUNTIME" for r in records):
        result["status"] = "INCOMPLETE"
        result["scientific_status"] = "BLOCKED_CALIBRATED_DATA"
    write_json(out / "campaign_result.json", result)
    return result


def read_trajectory(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "bhl-pose-trajectory-v1":
        raise ValueError("expected bhl-pose-trajectory-v1")
    frames = data["frames"]
    if not all(type(f.get("tracked")) is bool for f in frames):
        raise ValueError("each output frame must explicitly record tracking state")
    return data, Trajectory([f["timestamp_s"] for f in frames], [f["T_W_S"] for f in frames], [f["tracked"] for f in frames])


def evaluate(estimate_path, truth_path, receipt_path, out, *, max_difference_s=.02):
    estimate_data, estimate = read_trajectory(estimate_path)
    truth_data, truth = read_trajectory(truth_path)
    receipt = json.loads(Path(receipt_path).read_text())
    method = receipt.get("method")
    if receipt.get("schema") != "bhl-pose-estimator-run-v1" or method not in METHODS:
        raise ValueError("a native estimator run receipt is required")
    if receipt.get("upstream_commit") != UPSTREAM[method]["commit"]:
        raise ValueError("estimator code does not match the declared pinned upstream")
    for key in ("runtime_sha256", "input_manifest_sha256", "calibration_sha256", "config_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", receipt.get(key, "")):
            raise ValueError(f"missing exact estimator provenance: {key}")
    if receipt.get("ground_truth_inputs") != [] or any(PurePosixPath(p).parts[0] != "inference" for p in receipt.get("input_files_sha256", {})):
        raise ValueError("estimator must use only declared inference inputs")
    if not receipt.get("input_files_sha256") or receipt.get("trajectory_sha256") != sha256(estimate_path):
        raise ValueError("estimator output and original input hashes must be recorded")
    if estimate_data.get("clock_domain") != truth_data.get("clock_domain") or not estimate_data.get("clock_domain"):
        raise ValueError("estimate/truth clock domains must agree")
    if estimate_data.get("sensor_frame") != truth_data.get("sensor_frame") or not estimate_data.get("sensor_frame"):
        raise ValueError("estimate/truth must represent the same calibrated sensor frame")
    if estimate_data.get("sequence") != truth_data.get("sequence"):
        raise ValueError("estimate/truth sequence mismatch")
    alignment = "se3" if method == "orb_slam3_stereo" else "gravity_yaw_translation"
    metrics = score_trajectories(estimate, truth, alignment=alignment, max_difference_s=max_difference_s,
                                 end_time_s=estimate_data.get("end_time_s"))
    metrics.update({"status": "PASS", "scientific_status": "MEASURED_SINGLE_REPLAY_NOT_COMPARISON",
                    "method": method, "sequence": estimate_data["sequence"], "origin": estimate_data.get("origin"),
                    "estimate_sha256": sha256(estimate_path), "truth_sha256": sha256(truth_path),
                    "native_run_receipt_sha256": sha256(receipt_path), "closed_loop_episodes": 0})
    write_json(out, metrics)
    return metrics


def publish_fastlio(packets_path, out):
    """Optional ROS1 replay publisher; requires a separately built ROS runtime."""
    try:
        import rospy
        from std_msgs.msg import Header
        from sensor_msgs.msg import Imu, PointField
        from sensor_msgs import point_cloud2
    except ImportError as error:
        result = {"status": "INCOMPLETE", "scientific_status": "BLOCKED_RUNTIME", "reason": str(error), "published_packets": 0}
        write_json(out, result)
        return result
    rospy.init_node("bhl_timed_sensor_replay", anonymous=True)
    cloud_pub = rospy.Publisher("/bhl/lidar", point_cloud2.PointCloud2, queue_size=4)
    imu_pub = rospy.Publisher("/bhl/imu", Imu, queue_size=100)
    # Original stamps remain unchanged. Wall-clock pacing does not replace them.
    packets = [json.loads(line) for line in Path(packets_path).read_text().splitlines() if line]
    if not packets or any(p["capture_time_s"] < q["capture_time_s"] for q, p in zip(packets, packets[1:])):
        raise ValueError("ROS replay packets must preserve original monotonic clock order")
    deadline = time.monotonic() + 10
    while (not cloud_pub.get_num_connections() or not imu_pub.get_num_connections()) and time.monotonic() < deadline:
        if rospy.is_shutdown(): raise RuntimeError("ROS shutdown before subscribers connected")
        time.sleep(.01)
    if not cloud_pub.get_num_connections() or not imu_pub.get_num_connections():
        raise RuntimeError("native FAST-LIO subscribers did not connect within 10s")
    fields = [PointField(name, offset, datatype, 1) for name, offset, datatype in
              [("x", 0, PointField.FLOAT32), ("y", 4, PointField.FLOAT32), ("z", 8, PointField.FLOAT32),
               ("intensity", 12, PointField.FLOAT32), ("time", 16, PointField.FLOAT32), ("ring", 20, PointField.UINT16)]]
    first = packets[0]["capture_time_s"]
    start = time.monotonic()
    for packet in packets:
        remaining = start + packet["capture_time_s"] - first - time.monotonic()
        if remaining > 0: time.sleep(remaining)
        header = Header(stamp=rospy.Time.from_sec(packet["capture_time_s"]), frame_id=packet["frame_id"])
        if packet["kind"] == "pointcloud2":
            cloud_pub.publish(point_cloud2.create_cloud(header, fields, packet["points"]))
        elif packet["kind"] == "imu":
            msg = Imu(header=header)
            msg.orientation_covariance[0] = -1
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = packet["gyro_rad_s"]
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = packet["specific_force_m_s2"]
            imu_pub.publish(msg)
        else: raise ValueError("unexpected replay packet kind")
    result = {"status": "PASS", "scientific_status": "PUBLISHED_SENSORS_ODOMETRY_NOT_SCORED",
              "published_packets": len(packets), "packets_sha256": sha256(packets_path), "timestamps": "original_capture_time"}
    write_json(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("intake")
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--manifest"); choice.add_argument("--dataset")
    p.add_argument("--out", "--output", dest="out", required=True)
    p.add_argument("--protocol")
    p.add_argument("--no-export", action="store_true"); p.add_argument("--orb-executable")
    for command in ("export-orb", "export-fastlio"):
        p = sub.add_parser(command)
        p.add_argument("--manifest", required=True); p.add_argument("--sequence", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("evaluate")
    for name in ("estimate", "truth", "receipt", "out"): p.add_argument("--" + name, required=True)
    p.add_argument("--max-difference-s", type=float, default=.02)
    p = sub.add_parser("publish-fastlio")
    p.add_argument("--packets", required=True); p.add_argument("--out", required=True)
    a = parser.parse_args()
    if a.command == "intake":
        manifest_path = a.manifest or str(Path(a.dataset) / "manifest.json")
        result = intake(manifest_path, a.out, export_inputs=not a.no_export, orb_executable=a.orb_executable)
        if a.protocol:
            protocol = json.loads(Path(a.protocol).read_text())
            result["protocol_sha256"] = sha256(a.protocol)
            result["campaign_id"] = protocol.get("campaign_id")
            write_json(Path(a.out) / "campaign_result.json", result)
    elif a.command == "export-orb": result = export_orb(a.manifest, a.sequence, a.out)
    elif a.command == "export-fastlio": result = export_fastlio(a.manifest, a.sequence, a.out)
    elif a.command == "evaluate": result = evaluate(a.estimate, a.truth, a.receipt, a.out, max_difference_s=a.max_difference_s)
    else: result = publish_fastlio(a.packets, a.out)
    print(json.dumps({k: result[k] for k in ("status", "scientific_status") if k in result}, sort_keys=True))
    return 0 if result.get("status") in ("PASS", "READY_INPUTS_WAITING_RUNTIME") else 2


if __name__ == "__main__":
    raise SystemExit(main())
