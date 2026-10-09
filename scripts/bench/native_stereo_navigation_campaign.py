#!/usr/bin/env python3
"""Actual ORB-SLAM3 stereo pose plus raw LiDAR obstacle braking in the gait loop."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(Path(__file__).parent))
from bhl_robust.research.native_orb import NativeOrbClient, TRACKING_STATES, sha256, tracking_summary
from bhl_robust.research.pose_metrics import transform
import native_navigation_campaign as NAV

WIDTH, HEIGHT, BASELINE, VFOV_DEG, CAMERA_HZ = 640, 480, .12, 60., 5
R_BODY_CAMERA_GL = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
R_BODY_CAMERA_OPTICAL = R_BODY_CAMERA_GL @ np.diag([1., -1., -1.])


def body_camera_transform(side=1):
    if side not in (-1, 1):
        raise ValueError("left/right camera side must be +1/-1")
    matrix = np.eye(4)
    matrix[:3, :3] = R_BODY_CAMERA_OPTICAL
    matrix[:3, 3] = [.10, side*BASELINE/2, .38]
    return transform(matrix)


def native_camera_to_imu(t_world_camera, t_imu_lidar):
    """Fixed optical-camera/IMU calibration only; no evaluator pose input."""
    t_body_lidar = np.eye(4)
    t_body_lidar[:3, 3] = [.10, 0., .38]
    t_body_imu = t_body_lidar @ np.linalg.inv(transform(t_imu_lidar))
    return transform(t_world_camera) @ np.linalg.inv(body_camera_transform(1)) @ t_body_imu


def add_stereo_cameras(child):
    import mujoco
    body = child.bodies[1]
    if body.name != "base":
        raise ValueError("attached stereo cameras require the actual base body")
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, R_BODY_CAMERA_GL.reshape(-1))
    for side, name in ((1, "native_left"), (-1, "native_right")):
        body.add_camera(name=name, pos=[.10, side*BASELINE/2, .38], quat=quaternion,
                        fovy=VFOV_DEG)


def build_stereo_model(upstream, cache, n, labels, *, variant="biped", world="native_navigation"):
    """Add fixed cameras while proving the original gait/sensor slots survive."""
    import mujoco
    from bhl_robust.eval.multi_robot import build_multi
    if n != 1 or variant != "biped":
        raise ValueError("the bounded stereo navigation route uses one frozen biped")
    original, slots = build_multi(upstream, cache, n, labels, variant=variant, world=world)
    scene = cache/"mjcf_biped/berkeley_humanoid_lite_biped.xml"
    parent = mujoco.MjSpec.from_file(str(cache/f"multi_world_{world}.xml"))
    child = mujoco.MjSpec.from_file(str(scene))
    add_stereo_cameras(child)
    parent.worldbody.add_frame().attach_body(child.bodies[1], "r0_", "")
    model = parent.compile()
    if (model.nq, model.nv, model.nu, model.nsensor, model.nbody) != (
            original.nq, original.nv, original.nu, original.nsensor, original.nbody):
        raise ValueError("stereo attachment changed physical gait/sensor dimensions")
    for slot in slots:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, slot.prefix+"base") != slot.body_id:
            raise ValueError("base slot changed while attaching cameras")
    for attr in ("actuator_trnid", "sensor_adr", "sensor_dim", "sensor_objid", "jnt_qposadr", "jnt_dofadr", "qpos0"):
        if not np.array_equal(getattr(model, attr), getattr(original, attr)):
            raise ValueError(f"stereo attachment changed frozen control layout: {attr}")
    return model, slots


def textured_world(case, seed, directory):
    """Deterministic recorded visual features; collision geometry is unchanged."""
    from PIL import Image, ImageDraw
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed+910000)
    coarse = rng.integers(25, 225, (32, 32, 3), dtype=np.uint8)
    image = Image.fromarray(coarse).resize((512, 512), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    for _ in range(180):
        x, y = rng.integers(0, 500, 2)
        radius = int(rng.integers(3, 13))
        shade = (245, 245, 245) if rng.random() > .5 else (5, 5, 5)
        draw.rectangle((int(x), int(y), int(x)+radius, int(y)+radius), fill=shade)
    texture = directory/"scene_texture.png"
    image.save(texture)
    tree = ET.fromstring(NAV.world_xml(case, seed))
    asset = tree.find("asset")
    ET.SubElement(asset, "texture", name="stereo_features", type="2d", file=str(texture), width="512", height="512")
    ET.SubElement(asset, "material", name="stereo_features", texture="stereo_features", texuniform="true",
                  texrepeat="2 2", reflectance="0")
    for geom in tree.find("worldbody").findall("geom"):
        geom.set("material", "stereo_features")
        geom.set("rgba", "1 1 1 1")
    xml = ET.tostring(tree, encoding="unicode")
    NAV.write_json(directory/"texture_receipt.json", {"texture_sha256": sha256(texture), "seed": seed+910000,
                   "case": case, "geometry_source": "same native navigation fixtures; visual texture only",
                   "world_xml_sha256": __import__("hashlib").sha256(xml.encode()).hexdigest()})
    return xml


def write_settings(path):
    focal = HEIGHT/2/math.tan(math.radians(VFOV_DEG/2))
    rows = ["%YAML:1.0", 'File.version: "1.0"', 'Camera.type: "Rectified"']
    for index in (1, 2):
        rows += [f"Camera{index}.fx: {focal}", f"Camera{index}.fy: {focal}",
                 f"Camera{index}.cx: {(WIDTH-1)/2}", f"Camera{index}.cy: {(HEIGHT-1)/2}"]
        rows += [f"Camera{index}.{key}: 0.0" for key in ("k1", "k2", "p1", "p2")]
    rows += [f"Camera.width: {WIDTH}", f"Camera.height: {HEIGHT}", f"Camera.fps: {CAMERA_HZ}",
             "Camera.RGB: 1", "Stereo.ThDepth: 40.0", f"Stereo.b: {BASELINE}",
             "Stereo.T_c1_c2: !!opencv-matrix", "  rows: 4", "  cols: 4", "  dt: f",
             f"  data: [1,0,0,{BASELINE},0,1,0,0,0,0,1,0,0,0,0,1]",
             "ORBextractor.nFeatures: 1200", "ORBextractor.scaleFactor: 1.2", "ORBextractor.nLevels: 8",
             "ORBextractor.iniThFAST: 20", "ORBextractor.minThFAST: 7",
             "Viewer.KeyFrameSize: 0.05", "Viewer.KeyFrameLineWidth: 1.0", "Viewer.GraphLineWidth: 0.9",
             "Viewer.PointSize: 2.0", "Viewer.CameraSize: 0.08", "Viewer.CameraLineWidth: 3.0",
             "Viewer.ViewpointX: 0.0", "Viewer.ViewpointY: -0.7", "Viewer.ViewpointZ: -1.8",
             "Viewer.ViewpointF: 500.0", "Viewer.imageViewScale: 1.0"]
    path.write_text("\n".join(rows)+"\n")


class StereoSensors(NAV.CausalSensors):
    """Original fixed-camera RGB at causal scan boundaries, with actual roll."""
    def __init__(self, runner, slot, owners, *, output):
        import mujoco
        super().__init__(runner, slot, owners)
        self.output = output
        output.mkdir(parents=True, exist_ok=False)
        self.renderer = mujoco.Renderer(runner.m, height=HEIGHT, width=WIDTH)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[1] = 0
        self.records = []
        self.unbilled_capture_seconds = 0.
        self.total_capture_seconds = 0.

    def on_substep(self, data):
        before = len(self.recorded)
        super().on_substep(data)
        if len(self.recorded) == before:
            return
        from PIL import Image
        # All cached camera/body fields were computed before mj_step integrated
        # qpos. Rendering these fields uses their original preintegration time.
        stamp = float(data.time)-self.physics_dt
        scan = self.recorded[-1]
        if stamp+1e-8 < scan["frame_timestamp_s"]:
            raise ValueError("stereo capture cannot predate its obstacle scan")
        began = time.monotonic()
        index = len(self.records)
        paths = []
        for side in ("left", "right"):
            self.renderer.update_scene(data, camera=self.slot.prefix+"native_"+side, scene_option=self.option)
            rgb = self.renderer.render().copy()
            path = self.output/f"{index:05d}-{side}.png"
            Image.fromarray(rgb).save(path)
            paths.append(path)
        left_hash, right_hash = sha256(paths[0]), sha256(paths[1])
        capture_seconds = time.monotonic()-began
        self.unbilled_capture_seconds += capture_seconds
        self.total_capture_seconds += capture_seconds
        record = {"timestamp_s": stamp, "left": str(paths[0]), "right": str(paths[1]),
                  "left_sha256": left_hash, "right_sha256": right_hash,
                  "capture_wall_seconds": capture_seconds,
                  "timestamp_scope": "cached MuJoCo camera fields before integration: data.time-physics_dt"}
        self.records.append(record)
        scan["stereo"] = record

    def newest(self):
        packet = super().newest()
        if packet is None:
            return None
        scan, imu, dropped = packet
        scan["additional_sensor_compute_seconds"] = self.unbilled_capture_seconds
        self.unbilled_capture_seconds = 0.
        return scan, imu, dropped

    def save(self, path):
        super().save(path)
        NAV.write_json(self.output/"image_manifest.json", {"schema": "bhl-causal-stereo-navigation-images-v1",
            "width": WIDTH, "height": HEIGHT, "fovy_deg": VFOV_DEG, "baseline_m": BASELINE,
            "T_B_C_left": body_camera_transform(1).tolist(), "T_B_C_right": body_camera_transform(-1).tolist(),
            "images": self.records, "total_capture_wall_seconds": self.total_capture_seconds,
            "unbilled_unconsumed_capture_seconds_at_horizon": self.unbilled_capture_seconds,
            "render_scope": "fixed body-attached cameras preserve roll/pitch/yaw; no truth/depth images sent to ORB"})

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class OrbNavigationClient:
    """Convert genuine camera poses to the shared body/IMU planner convention."""
    def __init__(self, runtime, t_imu_lidar, directory, *, max_frames=1000):
        self.runtime, self.directory = Path(runtime), Path(directory)
        self.t_imu_lidar = transform(t_imu_lidar)
        self.maximum = max_frames
        self.original_map = None
        self.map_changed = False
        self.frames = []

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        settings = self.directory/"stereo.yaml"
        write_settings(settings)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(self.runtime/"lib")
        env["OMP_NUM_THREADS"] = "1"
        self.client = NativeOrbClient(self.runtime/"bin/orb_native", self.runtime/"ORBvoc.txt", settings,
                                      self.directory/"process", environment=env)
        self.client.__enter__()
        return self

    def track(self, scan, imu):
        # IMU and raw lidar are intentionally never forwarded into ORB-SLAM3.
        if len(self.frames) >= self.maximum:
            raise ValueError("bounded native stereo frame count exceeded")
        original = scan["stereo"]
        raw = self.client.track(original["timestamp_s"], original["left"], original["right"])
        map_id = raw["map_id"]
        if raw["tracked"] and map_id is not None:
            if self.original_map is None:
                self.original_map = map_id
            elif map_id != self.original_map:
                self.map_changed = True
        accepted = bool(raw["tracked"] and map_id is not None and not self.map_changed)
        packet = {"timestamp_s": raw["timestamp_s"], "tracked": accepted,
                  "T_W_I": native_camera_to_imu(raw["T_W_C"], self.t_imu_lidar).tolist() if accepted else None,
                  "state": "MAP_CHANGED_STOP" if self.map_changed else
                           "MAP_ID_UNAVAILABLE_STOP" if raw["tracked"] and map_id is None else TRACKING_STATES[raw["tracking_state"]],
                  "map_reset_id": int(self.map_changed), "compute_seconds": raw["compute_seconds"],
                  "native_ORB_T_W_C": raw["T_W_C"], "native_ORB_map_id": map_id,
                  "native_ORB_tracking_state": raw["tracking_state"],
                  "additional_sensor_compute_seconds": float(scan.get("additional_sensor_compute_seconds", 0.)),
                  "original_left_sha256": original["left_sha256"], "original_right_sha256": original["right_sha256"]}
        self.frames.append(packet)
        return packet

    def __exit__(self, kind, value, traceback):
        try:
            return self.client.__exit__(kind, value, traceback)
        finally:
            with (self.directory/"native_frames.jsonl").open("x") as stream:
                for frame in self.frames:
                    stream.write(json.dumps(frame, allow_nan=False)+"\n")
            NAV.write_json(self.directory/"stream_receipt.json", {"schema": "bhl-native-orb-navigation-stream-v1",
                "native_runtime_sha256": sha256(self.runtime/"runtime.json"),
                "binary_sha256": sha256(self.runtime/"bin/orb_native"),
                "vocabulary_sha256": sha256(self.runtime/"ORBvoc.txt"),
                "settings_sha256": sha256(self.directory/"stereo.yaml"),
                "frames": len(self.frames), "tracking": tracking_summary(self.client.responses),
                "first_map_id": self.original_map, "map_changed": self.map_changed,
                "T_I_L": self.t_imu_lidar.tolist(), "T_B_C_left": body_camera_transform(1).tolist(),
                "estimator_inputs": "original stereo PNG paths and original camera timestamps only",
                "ground_truth_inputs": [], "no_lidar_or_imu_to_stereo_estimator": True,
                "planner_sensor_scope": "ORB-SLAM3 stereo pose plus independent raw LiDAR obstacle brake"})


def unpack_runtime(archive, expected, work):
    if sha256(archive) != expected:
        raise ValueError("exact frozen native ORB runtime archive checksum required")
    spec = importlib.util.spec_from_file_location("orb_nav_unpack", ROOT/"scripts/bench/h34_campaign.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    work.mkdir(parents=True, exist_ok=False)
    launcher.safe_extract(archive, work)
    runtime = work/"runtime"
    receipt = json.loads((runtime/"runtime.json").read_text())
    if receipt.get("status") != "PASS" or receipt.get("schema") != "bhl-native-orb-runtime-v1":
        raise ValueError("actual successful pinned native ORB build required")
    for name, checksum in receipt["files_sha256"].items():
        path = runtime/name
        if not path.resolve().is_relative_to(runtime) or sha256(path) != checksum:
            raise ValueError("runtime member hash mismatch")
    (runtime/"bin/orb_native").chmod(0o700)
    return runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "development"), required=True)
    parser.add_argument("--upstream", type=Path, default=Path(os.getenv("UPSTREAM", "external/Berkeley-Humanoid-Lite")))
    parser.add_argument("--deploy", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-archive", type=Path, required=True)
    parser.add_argument("--runtime-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path(os.getenv("H34_OUTPUT_DIR", "output")))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"status": "INCOMPLETE", "route": "native_estimated_pose_navigation",
              "method": "orb_slam3_stereo_pose_plus_lidar_obstacle_brake", "phase": args.phase,
              "ground_truth_inputs_to_estimator_or_planner": [], "episodes": 0, "problems": []}
    began = time.monotonic()
    try:
        work = Path(os.getenv("H34_WORK_DIR", tempfile.mkdtemp(prefix="bhl-orb-nav-")))
        runtime = unpack_runtime(args.runtime_archive, args.runtime_archive_sha256, work/"orb-runtime")
        actor = {"name": "frozen_dr_default_s0", "deploy": str(args.deploy.resolve()), "checkpoint": str(args.checkpoint.resolve())}
        inventory = {str(p.resolve()): sha256(p) for p in (args.deploy, args.checkpoint, args.runtime_archive)}
        cases = ("straight",) if args.phase == "smoke" else NAV.CASES
        seeds = (370000, 370001) if args.phase == "smoke" else (380000, 380001, 380002)
        rows = []
        for case in cases:
            for seed in seeds:
                output = args.output/f"{case}-s{seed}"
                sensors_created = []
                def sensor_factory(runner, slot, owners):
                    sensors = StereoSensors(runner, slot, owners, output=output/"stereo")
                    sensors_created.append(sensors)
                    return sensors
                try:
                    row = NAV.episode(args.upstream.resolve(), actor, runtime, case, seed, output,
                        seconds=8. if args.phase == "smoke" else 40., client_factory=OrbNavigationClient,
                        world_factory=lambda c, s: textured_world(c, s, output/"visual_assets"),
                        build_factory=build_stereo_model, sensor_factory=sensor_factory,
                        method=result["method"], scope="Actual ORB-SLAM3 stereo pose and raw LiDAR obstacle brake; frozen 12-DoF gait; prescribed route/known start; simulated images and sensors, no physical validation")
                finally:
                    for sensor in sensors_created:
                        sensor.close()
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("case", "seed", "success", "fell", "collision", "native_tracked_frames")}), flush=True)
        for name, checksum in inventory.items():
            if sha256(name) != checksum:
                raise ValueError("native stereo navigation input changed")
        ready = all(r["native_tracked_frames"] > 0 and not r["nonfinite"] for r in rows)
        confirm = args.phase == "development" and len(rows) == 9 and all(r["success"] for r in rows)
        result.update(status="PASS" if args.phase == "smoke" and ready or confirm else
                      "INCOMPLETE" if args.phase == "smoke" else "NEGATIVE",
                      scientific_status="SMOKE_ONLY" if args.phase == "smoke" else "DEVELOPMENT_ONLY_NO_CONFIRMATION",
                      episodes=len(rows), goals=sum(r["success"] for r in rows), falls=sum(r["fell"] for r in rows),
                      collisions=sum(r["collision"] for r in rows), native_tracked_frames=sum(r["native_tracked_frames"] for r in rows),
                      gate_to_confirmation=confirm, input_sha256=inventory,
                      episode_files=[f"{r['case']}-s{r['seed']}/episode.json" for r in rows],
                      scope="Original body-attached 640x480 stereo with full roll and measured capture/native delay; stereo pose plus LiDAR obstacle sensing; simulation development only")
        if not ready and args.phase == "smoke":
            result["problems"].append("Native stereo smoke must actually track in both physical episodes; initialization failure is preserved")
    except Exception as error:
        result["problems"].append(f"{type(error).__name__}: {error}")
    result["wall_seconds"] = time.monotonic()-began
    NAV.write_json(args.output/"campaign_result.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] in ("PASS", "NEGATIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
