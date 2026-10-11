#!/usr/bin/env python3
"""Bounded matched native sensing controls with causal camera/IMU faults.

This separate experiment preserves the earlier frozen negative campaigns.
All arms share simulated cameras, timed LiDAR, IMU, gait, route and acquisition
cost. Estimator-alone refers to pose estimation: the raw LiDAR brake is shared.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import native_navigation_campaign as NAV
import native_stereo_navigation_campaign as STEREO
import stereo_methods_campaign as HYBRID
from bhl_robust.research.native_lio import NativeLioClient
from bhl_robust.research.native_orb import NativeOrbClient, TRACKING_STATES, sha256, write_rectified_stereo_settings
from bhl_robust.research.native_orb_inertial import write_stereo_inertial_settings
from bhl_robust.research.navigation_sensor_controls import (
    ARMS, CONDITIONS, AvailableImu, condition_spec, exposure_image,
    measured_brake_comparison, tracking_events,
)
from bhl_robust.research.pose_metrics import transform
from bhl_robust.research.sensor_faults import fault_spec, perturb_cloud
from bhl_robust.research.stereo_recovery import VerifiedMapRecovery

SCHEMA = "bhl-native-sensor-controls-v1"
SMOKE_SEED, DEVELOPMENT_SEEDS = 720000, (730000, 730001)
CASES = ("straight", "occluders")


def initialization_command(now):
    """Fixed simulation-only body excitation, matched across arms and conditions.

    Six seconds of alternating yaw and bounded forward velocity deliberately
    provide actual inertial excitation. No native thresholds are changed. The
    original known-start registration is an approximation after this motion;
    that limitation is recorded and independently evaluated, never truth-fixed.
    """
    if now < 3. or now >= 9.:
        return np.zeros(3)
    phase = (now - 3.) / 1.5
    return np.array([.10 * math.sin(2 * math.pi * phase), 0., .30 * math.sin(math.pi * phase)])


class ControlledSensors(NAV.CausalSensors):
    """Independent camera schedules and actual past-pose exposure renders."""
    def __init__(self, runner, slot, owners, *, output, condition):
        import mujoco
        super().__init__(runner, slot, owners, rate_hz=5.)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.spec = condition_spec(condition)
        self.condition = condition
        self.camera_period = 1 / self.spec["camera_hz"]
        self.camera_end = self.start_s + self.camera_period
        self.exposure_samples = self.spec.get("exposure_samples", 1)
        self.exposure_s = self.spec.get("exposure_s", 0.)
        self.camera_sample = 0
        self.sample_pixels = [[], []]
        self.sample_timestamps = []
        self.images = []
        self.events = []
        self.latest_scan = self.latest_camera = None
        self.renderer = mujoco.Renderer(runner.m, height=STEREO.HEIGHT, width=STEREO.WIDTH)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[1] = 0
        self.fault = fault_spec(self.spec.get("sensor_fault", "nominal"))
        self.imu_fault = AvailableImu(clock_offset_s=self.fault.get("imu_clock_offset_s", 0.),
            gyro_bias=self.fault.get("gyro_bias_rad_s", (0., 0., 0.)),
            bias_start_s=self.fault.get("bias_start_s", math.inf))
        self.imu_original_cursor = 0
        self.unbilled = self.total_sensor_wall = 0.
        self.brake_measurements = []
        self.reported_t_body_lidar = self.t_body_lidar.copy()
        yaw = math.radians(self.fault.get("lidar_yaw_error_deg", 0.))
        correction = np.eye(4)
        correction[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        correction[0, 3] = self.fault.get("lidar_x_error_m", 0.)
        self.reported_t_body_lidar = correction @ self.reported_t_body_lidar

    def on_substep(self, data):
        started = time.monotonic()
        before = len(self.recorded)
        super().on_substep(data)
        stamp = float(data.time) - self.physics_dt
        while self.imu_original_cursor < len(self.imu):
            self.imu_fault.append(*self.imu[self.imu_original_cursor])
            self.imu_original_cursor += 1
        changed = False
        if len(self.recorded) > before:
            original = self.recorded[-1]
            cloud = perturb_cloud(original, self.spec.get("sensor_fault", "nominal"))
            cloud.update(frame_timestamp_s=original["frame_timestamp_s"], origin="simulation")
            truth_body = original["points_xyz_m"] @ self.t_body_lidar[:3, :3].T + self.t_body_lidar[:3, 3]
            reported_body = cloud["points_xyz_m"] @ self.reported_t_body_lidar[:3, :3].T + self.reported_t_body_lidar[:3, 3]
            measurement = measured_brake_comparison(truth_body, reported_body)
            measurement["timestamp_s"] = cloud["frame_timestamp_s"]
            self.brake_measurements.append(measurement)
            cloud["planner_points_body"] = reported_body
            self.latest_scan = cloud
            self.ready.clear()
            changed = True
        sample_at = self.camera_end - self.exposure_s
        if self.exposure_samples > 1:
            sample_at += self.camera_sample * self.exposure_s / (self.exposure_samples - 1)
        if stamp >= sample_at - 1e-9:
            if stamp > sample_at + self.physics_dt + 1e-8:
                raise ValueError("camera acquisition schedule skipped a physical pose")
            for side, name in enumerate(("left", "right")):
                self.renderer.update_scene(data, camera=self.slot.prefix + "native_" + name, scene_option=self.option)
                self.sample_pixels[side].append(self.renderer.render().copy())
            self.sample_timestamps.append(stamp)
            self.camera_sample += 1
            if self.camera_sample == self.exposure_samples:
                self._finish_camera(stamp)
                self.camera_end += self.camera_period
                self.camera_sample = 0
                self.sample_pixels = [[], []]
                self.sample_timestamps = []
                changed = True
        if changed and self.latest_scan is not None and self.latest_camera is not None:
            self.events.append({"scan": self.latest_scan, "stereo": self.latest_camera,
                                "event_timestamp_s": stamp})
        elapsed = time.monotonic() - started
        self.unbilled += elapsed
        self.total_sensor_wall += elapsed

    def _finish_camera(self, stamp):
        from PIL import Image
        blackout = self.spec.get("blackout_start_s", math.inf) <= stamp < self.spec.get("blackout_end_s", -math.inf)
        paths, sample_hashes = [], []
        index = len(self.images)
        for side, name in enumerate(("left", "right")):
            rgb = exposure_image(self.sample_pixels[side], self.sample_timestamps,
                acquisition_s=stamp, exposure_s=self.exposure_s + self.physics_dt,
                gain=self.spec.get("exposure_gain", 1.), blackout=blackout)
            path = self.output / f"{index:05d}-{name}.png"
            Image.fromarray(rgb).save(path)
            paths.append(path)
            sample_hashes.append([hashlib.sha256(x.tobytes()).hexdigest() for x in self.sample_pixels[side]])
            if self.exposure_samples > 1:
                for n, original in enumerate(self.sample_pixels[side]):
                    Image.fromarray(original).save(self.output / f"{index:05d}-{name}-exposure{n}.png")
        record = {"timestamp_s": stamp, "left": str(paths[0]), "right": str(paths[1]),
                  "left_sha256": sha256(paths[0]), "right_sha256": sha256(paths[1]),
                  "exposure_sample_timestamps_s": list(self.sample_timestamps), "exposure_sample_rgb_sha256": sample_hashes,
                  "blackout": blackout, "exposure_gain": self.spec.get("exposure_gain", 1.),
                  "exposure_model": "equal-weight actual past-pose renders; no sensor shot/read noise or hardware shutter model"}
        self.images.append(record)
        self.latest_camera = record

    def newest(self):
        if not self.events:
            return None
        event, dropped = self.events[-1], len(self.events) - 1
        self.events.clear()
        scan = dict(event["scan"])
        scan["stereo"] = event["stereo"]
        scan["event_timestamp_s"] = event["event_timestamp_s"]
        scan["additional_sensor_compute_seconds"] = self.unbilled
        self.unbilled = 0.
        return scan, self.imu_fault.consume(event["event_timestamp_s"]), dropped

    def save(self, path):
        super().save(path)
        rows = self.imu_fault.delivered
        np.savez_compressed(self.output / "delivered_faulted_imu.npz",
            timestamp_s=np.asarray([r[0] for r in rows]), original_timestamp_s=np.asarray([r[3] for r in rows]),
            gyro_rad_s=np.asarray([r[1] for r in rows]), specific_force_m_s2=np.asarray([r[2] for r in rows]))
        NAV.write_json(self.output / "acquisition.json", {"schema": SCHEMA + "-acquisition", "condition": self.condition,
            "spec": self.spec, "images": self.images, "original_lidar_hz": 5., "original_imu_hz": 200.,
            "total_sensor_wall_s": self.total_sensor_wall, "unbilled_at_horizon_s": self.unbilled,
            "pending_causal_imu_samples": len(self.imu_fault.pending), "brake_measurements": self.brake_measurements,
            "evaluator_original_returns_never_forwarded": True,
            "T_B_L_true": self.t_body_lidar.tolist(), "T_B_L_reported": self.reported_t_body_lidar.tolist()})

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def empty_packet(stamp, state):
    return {"timestamp_s": float(stamp), "tracked": False, "T_W_I": None,
            "state": state, "map_reset_id": 0, "compute_seconds": 0.}


def write_controlled_settings(path, *, camera_hz, t_imu_camera=None):
    """Use the same native settings, with only the declared acquisition FPS.

    The legacy repository writer restricts its Python argument to integers;
    native OpenCV Camera.fps is numeric. A fractional-rate condition changes
    exactly that YAML field after the unchanged shared writer constructs it.
    """
    focal = STEREO.HEIGHT / 2 / math.tan(math.radians(STEREO.VFOV_DEG / 2))
    if camera_hz not in (2.5, 5., 10.):
        raise ValueError("predeclared camera rate required")
    writer_fps = int(math.ceil(camera_hz))
    config = dict(fx=focal, fy=focal, cx=(STEREO.WIDTH-1)/2, cy=(STEREO.HEIGHT-1)/2,
        width=STEREO.WIDTH, height=STEREO.HEIGHT, fps=writer_fps, baseline_m=STEREO.BASELINE)
    if t_imu_camera is None:
        write_rectified_stereo_settings(path, **config)
    else:
        write_stereo_inertial_settings(path, t_imu_camera=t_imu_camera, **config)
    old = f"Camera.fps: {writer_fps}\n"
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError("exact shared Camera.fps field expected")
    path.write_text(text.replace(old, f"Camera.fps: {camera_hz:g}\n"))


class MatchedNativeClient:
    """Independent native clocks and sensor buffers; hybrid never returns LIO pose."""
    def __init__(self, runtime, t_imu_lidar, directory, *, max_frames=1000):
        self.runtime, self.directory, self.maximum = runtime, Path(directory), max_frames
        self.arm = runtime["arm"]
        self.t_imu_lidar = transform(t_imu_lidar)
        self.spec = condition_spec(runtime["condition"])
        self.records, self.imu_rows = [], []
        self.camera_cursor = self.lio_cursor = -math.inf
        self.last_camera = self.last_cloud = -math.inf
        self.latest_orb = self.latest_lio = None
        self.original_map = None
        self.map_changed = False
        self.recovery = VerifiedMapRecovery()

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        try:
            if self.arm != "lidar_inertial":
                settings = self.directory / "stereo.yaml"
                if self.arm == "stereo_inertial":
                    t_body_lidar = np.eye(4)
                    t_body_lidar[:3, 3] = [.10, 0., .38]
                    t_body_imu = t_body_lidar @ np.linalg.inv(self.runtime["true_t_imu_lidar"])
                    write_controlled_settings(settings, camera_hz=self.spec["camera_hz"], t_imu_camera=np.linalg.inv(t_body_imu) @ STEREO.body_camera_transform(1))
                else:
                    write_controlled_settings(settings, camera_hz=self.spec["camera_hz"])
                orb = self.runtime["orb"]
                self.orb = self.stack.enter_context(NativeOrbClient(orb / "bin/orb_native", orb / "ORBvoc.txt", settings,
                    self.directory / "orb", environment=dict(os.environ, LD_LIBRARY_PATH=str(orb / "lib"), OMP_NUM_THREADS="1"),
                    sensor_mode="stereo_inertial" if self.arm == "stereo_inertial" else "stereo"))
            if self.arm in ("lidar_inertial", "permanent_stop", "verified_lio_recovery"):
                self.lio = self.stack.enter_context(NativeLioClient(self.runtime["lio"], self.t_imu_lidar,
                    self.directory / "lio", max_frames=self.maximum))
        except BaseException:
            self.stack.close()
            raise
        return self

    def _imu(self, after, until):
        rows = [r for r in self.imu_rows if after < r[0] <= until + 1e-9]
        return {"timestamp_s": np.asarray([r[0] for r in rows]),
                "gyro_rad_s": np.asarray([r[1] for r in rows]).reshape(-1, 3),
                "specific_force_m_s2": np.asarray([r[2] for r in rows]).reshape(-1, 3)}

    def track(self, scan, imu):
        if len(self.records) >= self.maximum:
            raise ValueError("bounded multi-sensor event count exceeded")
        self.imu_rows.extend(zip(imu["timestamp_s"], imu["gyro_rad_s"], imu["specific_force_m_s2"]))
        stamp = scan["stereo"]["timestamp_s"]
        new_lio = hasattr(self, "lio") and scan["frame_timestamp_s"] > self.last_cloud
        if new_lio:
            tail = scan["frame_timestamp_s"]
            self.latest_lio = self.lio.track(scan, self._imu(self.lio_cursor, tail))
            self.lio_cursor = tail
            self.last_cloud = tail
        new_camera = hasattr(self, "orb") and stamp > self.last_camera
        if new_camera:
            frame = scan["stereo"]
            kwargs = {}
            if self.arm == "stereo_inertial":
                batch = self._imu(self.camera_cursor, stamp)
                kwargs["imu_samples"] = np.column_stack((batch["timestamp_s"], batch["gyro_rad_s"], batch["specific_force_m_s2"]))
            self.latest_orb = self.orb.track(stamp, frame["left"], frame["right"], **kwargs)
            self.camera_cursor = stamp
            self.last_camera = stamp
        if self.arm == "lidar_inertial":
            packet = dict(self.latest_lio or empty_packet(scan["event_timestamp_s"], "LIO_INITIALIZING"))
        elif self.latest_orb is None:
            packet = empty_packet(scan["event_timestamp_s"], "CAMERA_INITIALIZING")
        elif not new_camera:
            packet = dict(self.records[-1]) if self.records else empty_packet(stamp, "CAMERA_INITIALIZING")
        else:
            raw = self.latest_orb
            initialized = self.arm != "stereo_inertial" or raw["inertial"]["map_imu_initialized"]
            if raw["tracked"] and initialized and raw["map_id"] is not None:
                if self.original_map is None:
                    self.original_map = raw["map_id"]
                elif raw["map_id"] != self.original_map:
                    self.map_changed = True
            accepted = bool(raw["tracked"] and initialized and raw["map_id"] is not None and not self.map_changed)
            pose = STEREO.native_camera_to_imu(raw["T_W_C"], self.runtime["true_t_imu_lidar"]) if raw["tracked"] else None
            packet = {"timestamp_s": stamp, "tracked": accepted, "T_W_I": pose.tolist() if accepted else None,
                "state": "IMU_INITIALIZING_STOP" if not initialized else "MAP_CHANGED_STOP" if self.map_changed else TRACKING_STATES[raw["tracking_state"]],
                "map_reset_id": int(self.map_changed), "compute_seconds": raw["compute_seconds"],
                "native_ORB_map_id": raw["map_id"], "native_ORB_tracking_state": raw["tracking_state"],
                "native_ORB_diagnostics": raw.get("diagnostics"),
                "native_imu_initialized": bool(initialized) if self.arm == "stereo_inertial" else None}
            if self.arm == "verified_lio_recovery":
                packet.update(self.recovery.update({"timestamp_s": stamp, "tracked": raw["tracked"],
                    "map_id": raw["map_id"], "T_W_I": pose.tolist() if pose is not None else None},
                    self.latest_lio or empty_packet(stamp, "LIO_INITIALIZING")))
        packet["additional_sensor_compute_seconds"] = float(scan["additional_sensor_compute_seconds"])
        packet["event_timestamp_s"] = scan["event_timestamp_s"]
        packet["method_arm"] = self.arm
        packet["new_native_camera_frame"] = bool(new_camera)
        packet["new_native_lio_frame"] = bool(new_lio)
        packet["reference_native_LIO"] = self.latest_lio
        self.records.append(packet)
        return packet

    def __exit__(self, kind, value, traceback):
        try:
            return self.stack.__exit__(kind, value, traceback)
        finally:
            with (self.directory / "native_frames.jsonl").open("x") as f:
                for row in self.records:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
            NAV.write_json(self.directory / "stream_receipt.json", {"schema": SCHEMA + "-native", "arm": self.arm,
                "native_camera_frames": len(self.orb.responses) if hasattr(self, "orb") else 0,
                "native_lio_frames": self.lio.index if hasattr(self, "lio") else 0,
                "recovery": self.recovery.receipt(), "ground_truth_inputs": [], "lio_pose_fallback": False,
                "camera_settings_sha256": sha256(self.directory / "stereo.yaml") if hasattr(self, "orb") else None,
                "native_orb_runtime_sha256": sha256(self.runtime["orb"] / "runtime.json"),
                "native_lio_runtime_sha256": sha256(self.runtime["lio"] / "runtime.json"),
                "initialization_thresholds": "unchanged native settings", "estimator_alone_scope": "shared independent raw LiDAR brake"})


def run_episode(upstream, actor, runtime, case, seed, output, *, condition, arm, seconds):
    import mujoco
    from bhl_robust.eval.multi_robot import _WORLDS,build_multi
    from team_airlock import ContactRunner
    from team_airlock import CpuPolicy
    from omegaconf import OmegaConf
    from bhl_robust.eval.gait_clock import make_controller
    cfg=OmegaConf.load(actor["deploy"]);cfg.policy_checkpoint_path=actor["checkpoint"]
    controller=make_controller(cfg);controller.policy=CpuPolicy(actor["checkpoint"])
    output.mkdir(parents=True,exist_ok=False)
    _WORLDS["native_navigation"]=STEREO.textured_world(case,seed,output/"visual_assets")
    cache=Path(tempfile.mkdtemp(prefix="bhl-native-nav-assets-"))
    model,slots=STEREO.build_stereo_model(upstream,cache,1,[arm],variant="biped",world="native_navigation")
    (output/"world.xml").write_text(_WORLDS["native_navigation"])
    slot=slots[0];runner=ContactRunner(model,slots,[cfg],[controller]);runner.reset(np.random.default_rng(seed))
    owners=np.array([0 if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or "").startswith(slot.prefix) else -1 for g in range(model.ngeom)])
    runner.configure_contacts(owners);sensors=ControlledSensors(runner,slot,owners,output=output/"sensors",condition=condition)
    runtime=dict(runtime, arm=arm, condition=condition, true_t_imu_lidar=sensors.t_imu_lidar.copy())
    policy_dt=float(cfg.policy_dt)
    known_start=np.eye(4);known_start[:3,3]=model.qpos0[slot.qpos_adr:slot.qpos_adr+3]
    registration=None;delivery=NAV.DelayedNativeState();waypoint=0;command=np.zeros(3)
    traces=[];native_records=[];client_latencies=[];additional_sensor_latencies=[];dropped_scans=0;goals=False;fell=collision=nonfinite=False
    latency_steps=0;goal_time=None;truth_imu=[];last_consumed_s=None
    initial_z=float(runner.d.qpos[slot.qpos_adr+2])
    maximum_steps=int(round(seconds/policy_dt))
    with MatchedNativeClient(runtime,np.linalg.inv(sensors.t_body_imu)@sensors.reported_t_body_lidar,output/"native",max_frames=int(seconds*20)+10) as client:
        for step in range(maximum_steps):
            now=float(runner.d.time)
            arrived=delivery.advance(step)
            latest=delivery.available
            if arrived and latest["tracked"] and registration is None:
                registration=known_start@sensors.t_body_imu@np.linalg.inv(transform(latest["T_W_I"]))
            if delivery.pending is not None:
                reason="measured_native_latency_hold";latency_steps+=1
            else:
                packet=None if arrived else sensors.newest()
                if packet is not None:
                    scan,imu,dropped=packet;dropped_scans+=dropped
                    started=time.monotonic();computed=client.track(scan,imu);elapsed=time.monotonic()-started
                    client_latencies.append(elapsed);native_records.append(computed)
                    extra=float(computed.get("additional_sensor_compute_seconds",0.))
                    if not math.isfinite(extra) or extra<0:raise ValueError("additional sensor compute latency must be finite and nonnegative")
                    additional_sensor_latencies.append(extra)
                    computed["client_wall_seconds"] = elapsed
                    computed["full_pipeline_wall_seconds"] = elapsed + extra
                    body_points=scan["planner_points_body"]
                    delivery.stage(computed,body_points,step,elapsed+extra,policy_dt)
                    reason="measured_native_latency_hold";latency_steps+=1
                else:
                    command,waypoint,reason,_=NAV.navigation_command(latest,registration,sensors.t_body_imu,NAV.ROUTES[case],waypoint,now,delivery.raw)
                    last_consumed_s=now if latest is not None else None
            # Loss/reset can affect commands only after the response arrives.
            # The older available pose may independently become stale during
            # computation, which correctly stops the held command.
            if latest is None or not latest["tracked"] or latest.get("map_reset_id")!=0: command=np.zeros(3)
            if latest is not None and now-latest["timestamp_s"]>.35:command=np.zeros(3);reason="native_stale_stop"
            if now < 10.:
                command=initialization_command(now);reason="predeclared_simulation_initialization_excitation"
            target=controller.update(runner.observe(0,command))
            if not np.isfinite(target).all():nonfinite=True;break
            runner.step([target])
            nonfinite=not np.isfinite(runner.d.qpos).all() or not np.isfinite(runner.d.qvel).all()
            if nonfinite:break
            # All simulator truth below is evaluator-only and never fed back.
            position=runner.d.qpos[slot.qpos_adr:slot.qpos_adr+3].copy();tilt=runner.tilt(0)
            collision=collision or runner.hit_wall;fell=tilt>=.78 or position[2]<initial_z-.25
            distance=float(np.linalg.norm(position[:2]-np.asarray(NAV.ROUTES[case][-1])))
            if distance<=.30 and not goals:goals=True;goal_time=float(runner.d.time)
            true_imu=np.eye(4);rotation=np.empty(9)
            mujoco.mju_quat2Mat(rotation,runner.d.qpos[slot.qpos_adr+3:slot.qpos_adr+7])
            true_imu[:3,:3]=rotation.reshape(3,3);true_imu[:3,3]=position
            truth_imu.append((float(runner.d.time),true_imu@sensors.t_body_imu))
            traces.append({"timestamp_s":float(runner.d.time),"command":command.tolist(),"reason":reason,
                "waypoint_index":waypoint,"native_timestamp_s":latest["timestamp_s"] if latest else None,
                "native_available_at_s":delivery.available_at_s,"native_last_consumed_at_s":last_consumed_s,
                "pending_capture_s":delivery.pending["capture_s"] if delivery.pending else None,
                "pending_arrival_s":delivery.pending["arrival_s"] if delivery.pending else None,
                "native_tracked":latest["tracked"] if latest else False,"evaluator_xy":position[:2].tolist(),
                "evaluator_height_m":float(position[2]),
                "evaluator_goal_distance_m":distance,"evaluator_tilt_rad":tilt,"evaluator_contact":bool(runner.hit_wall)})
            if fell or collision:break
    sensors.save(output/"inference_sensors.npz")
    sensors.close()
    np.savez_compressed(output/"evaluator_truth.npz",timestamp_s=np.asarray([x[0] for x in truth_imu]),T_W_I=np.asarray([x[1] for x in truth_imu]))
    actual_seconds=float(runner.d.time)
    result={"status":"PASS","scientific_status":"MEASURED_DEVELOPMENT_EPISODE","method":arm,"condition":condition,
        "case":case,"seed":seed,"success":bool(goals and not(fell or collision or nonfinite) and actual_seconds>=seconds-1e-6),
        "goal_reached_evaluator":goals,"first_goal_time_s":goal_time,"fell":bool(fell),"collision":bool(collision),"nonfinite":bool(nonfinite),
        "elapsed_simulation_s":actual_seconds,"full_horizon":actual_seconds>=seconds-1e-6,"controller_goal_stop":"estimated_pose_only",
        "native_frames":len(native_records),"native_tracked_frames":sum(r["tracked"] for r in native_records),
        "native_states":{state:sum(r["state"]==state for r in native_records) for state in sorted({r["state"] for r in native_records})},
        "client_wall_p95_ms":float(np.quantile(client_latencies,.95)*1000) if client_latencies else None,
        "additional_sensor_compute_total_s":float(sum(additional_sensor_latencies)),
        "compute_latency_policy_steps":latency_steps,"latency_quantization_s":policy_dt,"dropped_scans":dropped_scans,
        "latency_semantics":"ceil((measured client.track wallseconds + declared additional sensor capture/encoding cost)/policy_dt) physics/gait updates holding previous command and old available pose; native response and loss/reset status become visible only at arrival; old-pose staleness independently stops commands",
        "response_deliveries":delivery.history,"responses_still_pending_at_horizon":int(delivery.pending is not None),
        "initial_registration":registration.tolist() if registration is not None else None,"known_start_pose":known_start.tolist(),
        "T_body_imu":sensors.t_body_imu.tolist(),"T_body_lidar":sensors.t_body_lidar.tolist(),"route_external":NAV.ROUTES[case],
        "evaluator_initial_height_m":initial_z,
        "trace":traces,"inference_sensors_sha256":sha256(output/"inference_sensors.npz"),"evaluator_truth_sha256":sha256(output/"evaluator_truth.npz"),
        "stream_receipt_sha256":sha256(output/"native/stream_receipt.json"),"ground_truth_inputs_to_estimator_or_planner":[],
        "scope":"matched simulated sensor-control development; five native estimator/control arms; shared raw LiDAR brake; no hardware or independent confirmation",
        "initialization_excitation":{"stationary_until_s":3.,"excitation_until_s":9.,"navigation_begins_s":10.,"max_vx_command_m_s":.1,"max_yaw_command_rad_s":.3,"native_threshold_changes":False,"known_start_registration_limitation":"first accepted native pose is registered to nominal known station; initialization-motion error is not evaluator-corrected"},
        "full_pipeline_p95_ms":float(np.quantile([r["wall_seconds"] for r in delivery.history],.95)*1000) if delivery.history else None,
        "native_imu_initialized_frames":sum(r.get("native_imu_initialized") is True and r.get("new_native_camera_frame") for r in native_records),
        "measured_brake_false_clear_packets":sum(r["false_clear"] for r in sensors.brake_measurements),
        "measured_brake_unknown_packets":sum(r["unknown_stop"] for r in sensors.brake_measurements),
        "measured_brake_obstacle_packets":sum(r["measured_obstacle_present"] for r in sensors.brake_measurements),
        "false_free_space_scope":"only original simulated surface returns in raw-brake box; complete occupancy and pixel false-free space NOT_MEASURED"}
    update_key = "new_native_lio_frame" if arm == "lidar_inertial" else "new_native_camera_frame"
    measured_native = [r for r in native_records if r.get(update_key)]
    result["unique_estimator_updates"] = len(measured_native)
    result["accepted_unique_estimator_updates"] = sum(r["tracked"] for r in measured_native)
    result["tracking_fraction_unique_estimator_updates"] = sum(r["tracked"] for r in measured_native)/len(measured_native) if measured_native else None
    effective_points = [r["reference_native_LIO"]["effective_points"] for r in native_records
        if r.get("reference_native_LIO") is not None and "effective_points" in r["reference_native_LIO"]]
    result["minimum_effective_lidar_points"] = min(effective_points) if effective_points else None
    result.update(tracking_events(measured_native, actual_seconds))
    NAV.write_json(output/"episode.json",result)
    import shutil
    shutil.rmtree(cache)
    return result


def declared_protocol():
    smoke = [{"case": "straight", "seed": SMOKE_SEED, "condition": "nominal", "arm": arm} for arm in ARMS]
    smoke += [{"case": "straight", "seed": SMOKE_SEED, "condition": condition, "arm": "verified_lio_recovery"}
              for condition in CONDITIONS if condition != "nominal"]
    development = [{"case": case, "seed": seed, "condition": condition, "arm": arm}
                   for seed in DEVELOPMENT_SEEDS for condition in CONDITIONS for case in CASES for arm in ARMS]
    return {"schema": SCHEMA + "-protocol", "conditions": CONDITIONS, "arms": list(ARMS),
        "smoke": {"seconds": 18., "cells": smoke}, "development": {"seconds": 40., "cells": development},
        "initialization": {"stationary_until_s": 3., "excitation_until_s": 9., "navigation_begins_s": 10.,
            "command": "vx=.10*sin(2*pi*(t-3)/1.5), vy=0, yaw=.30*sin(pi*(t-3)/1.5) during [3,9)",
            "same_for_all_arms_conditions": True, "native_initialization_thresholds_changed": False,
            "known_start_assumption": "first accepted pose anchored to nominal station, initialization movement not truth-corrected"},
        "execution_gate": "all 14 same-source/input smoke cells execute with retained finite sensor/latency evidence; degraded outcomes remain scored",
        "arm_development_gate": "nominal smoke has a native accepted pose, full horizon and no fall/contact/nonfinite; inertial arm additionally actually initializes",
        "fault_negative_policy": "do not require tracking/goals under smoke faults; scientific degradation never masquerades as runtime failure",
        "planner_inputs": "native accepted pose, prescribed route, nominal known start, shared measured raw LiDAR brake",
        "false_free_metric": "evaluator-only false-clear raw-brake packets against original same-time ray surfaces; no complete occupancy claim",
        "sensor_noise": "ideal simulated cameras and IMU; three past-render exposure approximation, no hardware noise/shutter claim",
        "scope": "bounded matched development, no hardware or independent confirmation",
        "source_change_rule": "new frozen source requires fresh complete smoke; never reuse outcomes from superseded source"}


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for subtree in (ROOT / "src", ROOT / "scripts/bench", ROOT / "scripts/native")
            for p in sorted(subtree.rglob("*.py")) if p.is_file()}


def nominal_arm_readiness(rows):
    ready = {}
    for arm in ARMS:
        candidates = [r for r in rows if r["method"] == arm and r["condition"] == "nominal"]
        ready[arm] = bool(len(candidates) == 1 and candidates[0]["native_tracked_frames"] > 0
            and candidates[0]["full_horizon"] and not any(candidates[0][k] for k in ("fell", "collision", "nonfinite"))
            and (arm != "stereo_inertial" or candidates[0]["native_imu_initialized_frames"] > 0))
    return ready


def validate_smoke(smoke, *, sources, inputs):
    expected = declared_protocol()["smoke"]["cells"]
    if (smoke.get("schema") != SCHEMA or smoke.get("phase") != "smoke" or smoke.get("status") != "PASS"
            or smoke.get("execution_gate") is not True or smoke.get("source_sha256") != sources
            or smoke.get("input_sha256") != inputs or smoke.get("completed_cells") != expected):
        raise ValueError("complete matching-source/input smoke execution required")
    readiness = smoke.get("arm_development_ready")
    if not isinstance(readiness, dict) or set(readiness) != set(ARMS) or any(type(x) is not bool for x in readiness.values()):
        raise ValueError("explicit five-arm readiness required")
    return readiness


def collect_campaign(root, smoke_path, output):
    """Strict exact-cohort collection; skipped gates remain unrun, never zero."""
    root = Path(root).resolve()
    smoke = json.loads(Path(smoke_path).read_text())
    ready = validate_smoke(smoke, sources=smoke.get("source_sha256"), inputs=smoke.get("input_sha256"))
    expected = {tuple(c[k] for k in ("case", "seed", "condition", "arm")) for c in declared_protocol()["development"]["cells"]}
    observed, unrun, rows, lineage = set(), set(), [], []
    for path in sorted(root.rglob("campaign_result.json")):
        result = json.loads(path.read_text())
        if result.get("schema") != SCHEMA or result.get("phase") != "development":
            continue
        if (result.get("status") != "PASS" or result.get("problems") or result.get("source_sha256") != smoke["source_sha256"]
                or result.get("input_sha256") != smoke["input_sha256"] or result.get("arm_development_ready") != ready):
            raise ValueError("incomplete or mixed-source development result")
        if len(result["completed_cells"]) != len(result["episode_files"]) or result["episodes"] != len(result["completed_cells"]):
            raise ValueError("episode list differs from completed cells")
        for cell, relative in zip(result["completed_cells"], result["episode_files"]):
            key = tuple(cell[k] for k in ("case", "seed", "condition", "arm"))
            if key not in expected or key in observed or key in unrun or not ready[cell["arm"]]:
                raise ValueError("undeclared, duplicate or gated development cell")
            episode_path = (path.parent / relative).resolve()
            if not episode_path.is_relative_to(path.parent.resolve()):
                raise ValueError("episode file leaves its result directory")
            if sha256(episode_path) != result["episode_sha256"].get(relative):
                raise ValueError("retained episode differs from result hash")
            row = json.loads(episode_path.read_text())
            if (row.get("case"), row.get("seed"), row.get("condition"), row.get("method")) != key:
                raise ValueError("episode identities differ from declared cell")
            for name, field in (("inference_sensors.npz", "inference_sensors_sha256"),
                                ("evaluator_truth.npz", "evaluator_truth_sha256"),
                                ("native/stream_receipt.json", "stream_receipt_sha256")):
                if sha256(episode_path.parent / name) != row[field]:
                    raise ValueError("retained raw/evaluator lineage differs")
            trace = row["trace"]
            if not trace or any(not math.isfinite(r["evaluator_goal_distance_m"]) for r in trace):
                raise ValueError("finite retained evaluator trace required")
            reached = any(r["evaluator_goal_distance_m"] <= .30 for r in trace)
            contact = any(r["evaluator_contact"] for r in trace)
            fell = any(r["evaluator_tilt_rad"] >= .78 or r["evaluator_height_m"] < row["evaluator_initial_height_m"] - .25 for r in trace)
            full = row["elapsed_simulation_s"] >= 40. - 1e-6
            success = reached and full and not (contact or fell or row["nonfinite"])
            if any(bool(row[k]) != actual for k, actual in (("goal_reached_evaluator", reached), ("collision", contact),
                    ("fell", fell), ("full_horizon", full), ("success", success))):
                raise ValueError("reported outcome differs from retained evaluator trace")
            rows.append(row)
            observed.add(key)
        for cell in result["unrun_cells"]:
            key = tuple(cell[k] for k in ("case", "seed", "condition", "arm"))
            if (key not in expected or key in observed or key in unrun or ready[cell["arm"]]
                    or cell["reason"] != "UNRUN_AFTER_NEGATIVE_NOMINAL_ARM_SMOKE"):
                raise ValueError("invalid unrun cell or gate")
            unrun.add(key)
        lineage.append({"path": str(path), "sha256": sha256(path)})
    missing = expected - observed - unrun
    if missing:
        raise ValueError(f"incomplete development collection: {len(missing)} undeclared-missing outcomes")
    groups = {}
    for arm in ARMS:
        for condition in CONDITIONS:
            values = [r for r in rows if r["method"] == arm and r["condition"] == condition]
            groups[arm + "/" + condition] = {"episodes": len(values), "clean_goals": sum(r["success"] for r in values),
                "falls": sum(r["fell"] for r in values), "contacts": sum(r["collision"] for r in values),
                "mean_episode_full_pipeline_p95_ms": float(np.mean([r["full_pipeline_p95_ms"] for r in values if r["full_pipeline_p95_ms"] is not None])) if values else None,
                "tracking_loss_events": sum(r["tracking_loss_events"] for r in values),
                "native_map_changes": sum(r["native_map_changes"] for r in values),
                "completed_recovery_durations_s": [duration for r in values for duration in r["completed_recovery_durations_s"]],
                "right_censored_losses": sum(r["right_censored_loss_duration_s"] is not None for r in values),
                "measured_brake_false_clear_packets": sum(r["measured_brake_false_clear_packets"] for r in values),
                "measured_brake_unknown_packets": sum(r["measured_brake_unknown_packets"] for r in values)}
    result = {"schema": SCHEMA + "-collection", "status": "PASS", "declared_cells": 200,
        "measured_cells": len(observed), "unrun_after_nominal_gate": len(unrun), "missing_cells": [],
        "arm_development_ready": ready, "by_arm_condition": groups, "source_results": lineage,
        "qualified_smoke_sha256": sha256(smoke_path), "scientific_status": "COMPLETE_BOUNDED_DEVELOPMENT_NO_CONFIRMATION",
        "scope": "four matched route/seed pairs per condition; camera-rate changes deliberately retain all measured latency and fixed .35s age gate; no population/hardware inference"}
    NAV.write_json(output, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--write-protocol", type=Path)
    p.add_argument("--collect", type=Path)
    p.add_argument("--phase", choices=("smoke", "development"))
    for name in ("upstream", "deploy", "checkpoint", "orb-runtime-archive", "lio-runtime-archive", "protocol", "output", "qualified-smoke"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--orb-runtime-sha256")
    p.add_argument("--lio-runtime-sha256")
    p.add_argument("--case", choices=CASES)
    p.add_argument("--seed", type=int, choices=DEVELOPMENT_SEEDS)
    p.add_argument("--condition", choices=CONDITIONS)
    a = p.parse_args(argv)
    if a.write_protocol:
        NAV.write_json(a.write_protocol, declared_protocol())
        return 0
    if a.collect:
        if a.qualified_smoke is None or a.output is None:
            p.error("collection requires --qualified-smoke and --output")
        collect_campaign(a.collect, a.qualified_smoke, a.output)
        return 0
    for key in ("phase", "upstream", "deploy", "checkpoint", "orb_runtime_archive", "lio_runtime_archive", "protocol", "output", "orb_runtime_sha256", "lio_runtime_sha256"):
        if getattr(a, key) is None:
            p.error("--" + key.replace("_", "-") + " is required")
    shard = any(x is not None for x in (a.case, a.seed, a.condition))
    if shard and (a.phase != "development" or any(x is None for x in (a.seed, a.condition))):
        p.error("development shards require --seed and --condition; --case optionally narrows the paired routes")
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "campaign_result.json").exists():
        raise ValueError("refuse to overwrite a prior campaign result")
    result = {"schema": SCHEMA, "phase": a.phase, "status": "INCOMPLETE", "episodes": 0, "problems": [],
              "completed_cells": [], "unrun_cells": [], "ground_truth_inputs_to_estimator_or_planner": []}
    started = time.monotonic()
    try:
        protocol = json.loads(a.protocol.read_text())
        if protocol != declared_protocol():
            raise ValueError("exact predeclared protocol required")
        sources = source_hashes()
        inputs = {key: sha256(getattr(a, key)) for key in ("deploy", "checkpoint", "orb_runtime_archive", "lio_runtime_archive", "protocol")}
        readiness = {arm: True for arm in ARMS}
        if a.phase == "development":
            if a.qualified_smoke is None:
                raise ValueError("complete same-source/input smoke is required")
            readiness = validate_smoke(json.loads(a.qualified_smoke.read_text()), sources=sources, inputs=inputs)
        work = Path(os.getenv("H34_WORK_DIR", tempfile.mkdtemp(prefix="bhl-native-sensor-controls-")))
        orb = STEREO.unpack_runtime(a.orb_runtime_archive, a.orb_runtime_sha256, work / "sensor-controls-orb")
        lio = HYBRID.unpack_lio(a.lio_runtime_archive, a.lio_runtime_sha256, work / "sensor-controls-lio")
        actor = {"name": "frozen_dr_default_s0", "deploy": str(a.deploy.resolve()), "checkpoint": str(a.checkpoint.resolve())}
        cells = protocol[a.phase]["cells"]
        if shard:
            cells = [c for c in cells if (a.case is None or c["case"] == a.case) and c["seed"] == a.seed and c["condition"] == a.condition]
        rows = []
        for cell in cells:
            if not readiness[cell["arm"]]:
                result["unrun_cells"].append(dict(cell, reason="UNRUN_AFTER_NEGATIVE_NOMINAL_ARM_SMOKE"))
                continue
            name = f"{cell['case']}-s{cell['seed']}-{cell['condition']}-{cell['arm']}"
            row = run_episode(a.upstream.resolve(), actor, {"orb": orb, "lio": lio}, cell["case"], cell["seed"], a.output / name,
                condition=cell["condition"], arm=cell["arm"], seconds=protocol[a.phase]["seconds"])
            if row["nonfinite"] or row["native_frames"] < 1:
                raise ValueError("execution smoke/episode requires finite state and actual native input/output")
            row["episode_file"] = name + "/episode.json"
            rows.append(row)
            result["completed_cells"].append(cell)
            print(json.dumps({k: row[k] for k in ("method", "condition", "case", "seed", "success", "native_tracked_frames", "native_imu_initialized_frames", "fell", "collision")}), flush=True)
        if source_hashes() != sources or any(sha256(getattr(a, k)) != v for k, v in inputs.items()):
            raise ValueError("source or frozen input mutated during the experiment")
        result.update(status="PASS", scientific_status="DEVELOPMENT_ONLY_NO_CONFIRMATION" if a.phase == "development" else "EXECUTION_SMOKE_WITH_EXPLICIT_SCIENTIFIC_ARM_GATES",
            execution_gate=True, source_sha256=sources, input_sha256=inputs, episodes=len(rows),
            episode_files=[r["episode_file"] for r in rows], expected_declared_cells=len(cells),
            episode_sha256={r["episode_file"]: sha256(a.output / r["episode_file"]) for r in rows},
            total_declared_development_cells=200, arm_development_ready=nominal_arm_readiness(rows) if a.phase == "smoke" else readiness,
            arms={arm: {"episodes": sum(r["method"] == arm for r in rows),
                "clean_goals": sum(r["success"] for r in rows if r["method"] == arm),
                "falls": sum(r["fell"] for r in rows if r["method"] == arm),
                "contacts": sum(r["collision"] for r in rows if r["method"] == arm)} for arm in ARMS},
            aggregate_improvement_verdict="UNAVAILABLE_UNTIL_COMPLETE_STRICT_PAIRED_COLLECTION",
            scope="new matched simulated acquisition/fault development; no changes to previous negative cohorts")
    except Exception as error:
        result["problems"].append(type(error).__name__ + ": " + str(error))
    result["wall_seconds"] = time.monotonic() - started
    NAV.write_json(a.output / "campaign_result.json", result)
    print(json.dumps({k: result[k] for k in ("schema", "phase", "status", "episodes", "problems")}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
