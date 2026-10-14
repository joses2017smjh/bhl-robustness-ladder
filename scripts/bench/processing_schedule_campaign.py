#!/usr/bin/env python3
"""Predeclared processing-schedule comparison using actual native estimators.

The candidate models separate simulation acquisition, bounded latest joined
input, measured inference service, and a real bounded evidence writer. This
is a scheduling simulation, not a measured concurrent robot implementation.
The completed native-sensor-controls-v2 source and cohorts remain untouched.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
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
    ARMS, CONDITIONS, AvailableImu, NativeCoordinateEpochTracker, condition_spec, exposure_image,
    measured_brake_comparison, tracking_events,
)
from bhl_robust.research.pose_metrics import transform
from bhl_robust.research.sensor_faults import fault_spec, perturb_cloud
from bhl_robust.research.stereo_recovery import VerifiedMapRecovery

import native_sensor_controls_campaign as BASE
from bhl_robust.research.processing_schedule import LatestJoinedQueue, service_timing, decoded_frame_identity, compare_replay_inputs, replay_transport_queue
from concurrent.futures import ThreadPoolExecutor
from collections import deque

SCHEMA = "bhl-processing-schedule-v1"
PIPELINES = ("legacy_cumulative", "bounded_latest")
ARMS = ("stereo", "lidar_inertial")
CONDITIONS = ("nominal", "camera_10hz", "causal_40ms_exposure")
CASES = ("straight", "occluders")
SMOKE_SEED, DEVELOPMENT_SEEDS = 749000, (750000, 750001)
# Frozen designs. A protocol without a "design" key is the completed v1
# protocol (job 21801949), which must stay byte-for-byte reproducible.
V1_DESIGN = "v1-20261012"
ORDER_DESIGN = "order-confirmation-20261014"
DESIGNS = (V1_DESIGN, ORDER_DESIGN)
ORDER_ARMS = ("lidar_inertial",)
ORDER_SEEDS = (751000, 751001, 751002, 751003, 751004, 751005, 751006, 751007)
WARMUP_SEED, WARMUP_CASE, WARMUP_SECONDS = 749900, "straight", 18.
WARMUP_RETAINED = ("episode.json", "sensors/processing_schedule.json", "native/stream_receipt.json")
initialization_command = BASE.initialization_command
MatchedNativeClient = BASE.MatchedNativeClient


class BoundedRecorder:
    """Immutable sensor buffers, bounded outstanding writes, joined errors.

    Backpressure is measured and reported as instrumentation overhead. It is
    not silently added to modeled native inference service. No worker touches
    the simulator or changes frame timestamps.
    """
    def __init__(self, capacity=8):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("positive integer recorder capacity required")
        self.capacity = capacity
        self.pending = deque()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sensor-evidence")
        self.backpressure_s = 0.
        self.maximum_pending = 0
        self.tasks = 0
        self.closed = False
        self.error = None

    def _join_one(self):
        try:
            self.pending.popleft().result()
        except BaseException as error:
            self.error = error
            raise

    def submit(self, function, *args):
        if self.error is not None:
            raise self.error
        if self.closed:
            raise ValueError("recorder already closed")
        while self.pending and self.pending[0].done():
            self._join_one()
        if len(self.pending) >= self.capacity:
            started = time.monotonic()
            self._join_one()
            self.backpressure_s += time.monotonic() - started
        self.pending.append(self.pool.submit(function, *args))
        self.tasks += 1
        self.maximum_pending = max(self.maximum_pending, len(self.pending))

    def close(self):
        if not self.closed:
            try:
                while self.pending:
                    try:
                        self._join_one()
                    except BaseException:
                        pass  # drain every outstanding writer, retain failure
            finally:
                self.pool.shutdown(wait=True)
                self.closed = True
        if self.error is not None:
            raise self.error

    def receipt(self):
        return dict(capacity=self.capacity, maximum_pending=self.maximum_pending,
                    tasks=self.tasks, backpressure_wall_s=self.backpressure_s,
                    joined=self.closed and self.error is None, pending=len(self.pending),
                    error=str(self.error) if self.error is not None else None)


class ScheduledSensors(BASE.ControlledSensors):
    def __init__(self, *args, pipeline, **kwargs):
        if pipeline not in PIPELINES:
            raise ValueError("unknown pipeline")
        self.pipeline = pipeline
        self.latest_queue = LatestJoinedQueue()
        self.recorder = BoundedRecorder() if pipeline == "bounded_latest" else None
        self.wall_origin = time.monotonic()
        self.stage_costs = dict(simulator_ray_imu_s=0., simulator_camera_render_s=0.,
            camera_synthesis_s=0., selected_png_ipc_s=0., evidence_worker_s=0.)
        self.requests = []
        self.selected_images = {}
        self.last_queue_metadata = None
        self.previous_dropped = 0
        super().__init__(*args, **kwargs)
        self.inference_output = self.output / "selected-inference"
        self.inference_output.mkdir()

    def _archive(self, record, pixels, samples):
        from PIL import Image
        started = time.monotonic()
        paths = (Path(record["left"]), Path(record["right"]))
        for side, path in enumerate(paths):
            Image.fromarray(pixels[side]).save(path)
            record[("left_sha256", "right_sha256")[side]] = sha256(path)
            if self.exposure_samples > 1:
                for n, original in enumerate(samples[side]):
                    Image.fromarray(original).save(path.with_name(path.stem + f"-exposure{n}.png"))
        record["decoded_identity"] = decoded_frame_identity(*pixels)
        record["exposure_sample_rgb_sha256"] = [[hashlib.sha256(x.tobytes()).hexdigest() for x in side] for side in samples]
        record["archive_start_wall_s"] = started - self.wall_origin
        record["archive_complete_wall_s"] = time.monotonic() - self.wall_origin
        self.stage_costs["evidence_worker_s"] += time.monotonic() - started

    def _finish_camera(self, stamp):
        index = len(self.images)
        started = time.monotonic()
        blackout = self.spec.get("blackout_start_s", math.inf) <= stamp < self.spec.get("blackout_end_s", -math.inf)
        pixels = tuple(exposure_image(side, self.sample_timestamps,
            acquisition_s=stamp, exposure_s=self.exposure_s + self.physics_dt,
            gain=self.spec.get("exposure_gain", 1.), blackout=blackout) for side in self.sample_pixels)
        self.stage_costs["camera_synthesis_s"] += time.monotonic() - started
        samples = tuple(tuple(side) for side in self.sample_pixels)
        for array in (*pixels, *(a for side in samples for a in side)):
            array.setflags(write=False)
        record = dict(timestamp_s=stamp, frame_id=index,
            left=str(self.output / f"{index:05d}-left.png"), right=str(self.output / f"{index:05d}-right.png"),
            exposure_sample_timestamps_s=list(self.sample_timestamps), blackout=blackout,
            exposure_gain=self.spec.get("exposure_gain", 1.),
            exposure_model="equal-weight actual past-pose renders; no hardware shutter/noise model",
            capture_wall_s=time.monotonic() - self.wall_origin,
            archive_enqueue_wall_s=time.monotonic() - self.wall_origin)
        if self.recorder:
            self.recorder.submit(self._archive, record, pixels, samples)
        else:
            self._archive(record, pixels, samples)
        self.images.append(record)
        # Pixel buffers belong only to the current camera slot and outstanding
        # bounded writer tasks, not to the unbounded metadata evidence list.
        self.latest_camera = dict(record, _pixels=pixels)

    def on_substep(self, data):
        started = time.monotonic()
        before = len(self.recorded)
        ray_started = time.monotonic()
        NAV.CausalSensors.on_substep(self, data)
        self.stage_costs["simulator_ray_imu_s"] += time.monotonic() - ray_started
        stamp = float(data.time) - self.physics_dt
        while self.imu_original_cursor < len(self.imu):
            self.imu_fault.append(*self.imu[self.imu_original_cursor])
            self.imu_original_cursor += 1
        changed = False
        if len(self.recorded) > before:
            original = self.recorded[-1]
            cloud = perturb_cloud(original, self.spec.get("sensor_fault", "nominal"))
            cloud.update(frame_timestamp_s=original["frame_timestamp_s"], origin="simulation", frame_id=len(self.recorded)-1)
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
            render_started = time.monotonic()
            for side, name in enumerate(("left", "right")):
                self.renderer.update_scene(data, camera=self.slot.prefix + "native_" + name, scene_option=self.option)
                self.sample_pixels[side].append(self.renderer.render().copy())
            self.stage_costs["simulator_camera_render_s"] += time.monotonic() - render_started
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
            event = dict(scan=self.latest_scan, stereo=self.latest_camera, event_timestamp_s=stamp)
            self.latest_queue.offer(event, event_s=stamp,
                camera_id=self.latest_camera["frame_id"], camera_capture_s=self.latest_camera["timestamp_s"],
                lidar_id=self.latest_scan["frame_id"], lidar_capture_s=self.latest_scan["frame_timestamp_s"])
            if self.pipeline == "legacy_cumulative":
                # Same latest-event selection and accumulated callback charge
                # as the old baseline; retain only metadata rather than pixels
                # in the discarded-event count.
                self.events.append(1)
        elapsed = time.monotonic() - started
        self.unbilled += elapsed
        self.total_sensor_wall += elapsed

    def newest(self):
        now = float(self.runner.d.time)
        selected = self.latest_queue.pop(now_s=now)
        if selected is None:
            return None
        event = selected["payload"]
        self.last_queue_metadata = selected["queue"]
        scan, frame = dict(event["scan"]), dict(event["stereo"])
        pixels = frame.pop("_pixels")
        extra = self.unbilled if self.pipeline == "legacy_cumulative" else 0.
        ipc_started = time.monotonic()
        if self.pipeline == "bounded_latest":
            # Both arms retain matched acquisition/IPC work; lidar does not
            # silently benefit from disabling the shared camera stream.
            from PIL import Image
            for side, name in enumerate(("left", "right")):
                target = self.inference_output / f'{frame["frame_id"]:05d}-{name}.png'
                if not target.exists():
                    Image.fromarray(pixels[side]).save(target)
                frame[name] = str(target)
            extra = time.monotonic() - ipc_started
            self.stage_costs["selected_png_ipc_s"] += extra
        self.selected_images[frame["frame_id"]] = (frame["left"], frame["right"])
        self.unbilled = 0.
        scan.update(stereo=frame, event_timestamp_s=event["event_timestamp_s"],
                    additional_sensor_compute_seconds=extra)
        receipt = self.latest_queue.receipt()
        # Exact joined-event count is supplied by queue metadata below; this
        # counter also retains the old baseline's reported aggregate meaning.
        dropped = receipt["superseded"] - self.previous_dropped
        self.previous_dropped = receipt["superseded"]
        self.events.clear()
        self.requests.append(dict(request_s=now, camera_id=frame["frame_id"], lidar_id=scan["frame_id"],
            camera_capture_s=frame["timestamp_s"], lidar_capture_s=scan["frame_timestamp_s"],
            preprocessing_service_s=extra, queue=selected["queue"]))
        return scan, self.imu_fault.consume(event["event_timestamp_s"]), dropped

    def save(self, path):
        if self.recorder:
            self.recorder.close()
        from PIL import Image
        verification = []
        for index, paths in sorted(self.selected_images.items()):
            actual = decoded_frame_identity(*(np.asarray(Image.open(p).convert("RGB")) for p in paths))
            if actual != self.images[index]["decoded_identity"]:
                raise ValueError("selected inference pixels differ from retained acquisition")
            verification.append(index)
        super().save(path)
        self.schedule_receipt = dict(schema=SCHEMA + "-timing", pipeline=self.pipeline,
            stage_wall_seconds=self.stage_costs, total_acquisition_callback_s=self.total_sensor_wall,
            recorder=self.recorder.receipt() if self.recorder else dict(mode="synchronous", joined=True),
            queue=self.latest_queue.receipt(), requests=self.requests,
            decoded_selected_images_verified=verification,
            camera_captures_total=len(self.images), lidar_captures_total=len(self.recorded),
            camera_never_joined=[r["frame_id"] for r in self.images if r["frame_id"] not in self.latest_queue.streams["camera"]["identities"]],
            lidar_never_joined=[i for i in range(len(self.recorded)) if i not in self.latest_queue.streams["lidar"]["identities"]],
            wall_time_scope="monotonic seconds relative to episode sensor construction; distinct from modeled simulation clock",
            modeled_service="legacy: all cumulative acquisition callbacks + native; candidate: selected PNG IPC + native",
            simulator_generation_and_instrumentation_reported_separately=True)
        NAV.write_json(self.output / "processing_schedule.json", self.schedule_receipt)

    def close(self):
        try:
            if self.recorder:
                self.recorder.close()
        finally:
            super().close()


def run_episode(upstream, actor, runtime, case, seed, output, *, condition, arm, seconds, pipeline, execution=None):
    started_utc, started_wall = datetime.now(timezone.utc).isoformat(), time.monotonic()
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
    runner.configure_contacts(owners);sensors=ScheduledSensors(runner,slot,owners,output=output/"sensors",condition=condition,pipeline=pipeline)
    runtime=dict(runtime, arm=arm, condition=condition, true_t_imu_lidar=sensors.t_imu_lidar.copy())
    policy_dt=float(cfg.policy_dt)
    known_start=np.eye(4);known_start[:3,3]=model.qpos0[slot.qpos_adr:slot.qpos_adr+3]
    registration=None;delivery=NAV.DelayedNativeState();waypoint=0;command=np.zeros(3)
    traces=[];native_records=[];client_latencies=[];additional_sensor_latencies=[];dropped_scans=0;goals=False;fell=collision=nonfinite=False
    latency_steps=0;goal_time=None;truth_imu=[];last_consumed_s=None
    initial_z=float(runner.d.qpos[slot.qpos_adr+2])
    maximum_steps=int(round(seconds/policy_dt))
    with ExitStack() as resources:
        resources.callback(sensors.close)
        client = resources.enter_context(MatchedNativeClient(runtime,np.linalg.inv(sensors.t_body_imu)@sensors.reported_t_body_lidar,output/"native",max_frames=int(seconds*20)+10))
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
                    computed["schedule_timing"] = service_timing(request_s=now, capture_s=computed["timestamp_s"], preprocessing_s=extra, native_client_s=elapsed, policy_dt=policy_dt)
                    computed["schedule_timing"]["pipeline"] = pipeline
                    computed["schedule_timing"]["preprocessing_scope"] = "cumulative prior acquisition callbacks" if pipeline == "legacy_cumulative" else "selected PNG IPC only"
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
            traces.append({"decision_timestamp_s":now,"timestamp_s":float(runner.d.time),"command":command.tolist(),"reason":reason,
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
    result={"status":"PASS","scientific_status":"MEASURED_DEVELOPMENT_EPISODE","method":arm,"pipeline":pipeline,"condition":condition,
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
        "scope":"paired schedule-model simulation development; shared raw LiDAR brake; no hardware or independent confirmation",
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
    nav = [t for t in traces if t["decision_timestamp_s"] >= 10.-1e-9]
    ages = [t["decision_timestamp_s"]-t["native_timestamp_s"] for t in nav if t["native_timestamp_s"] is not None]
    result["navigation_decisions"] = len(nav)
    result["stale_navigation_decisions"] = sum(t["reason"] == "native_stale_stop" for t in nav)
    result["stale_navigation_fraction"] = result["stale_navigation_decisions"]/len(nav) if nav else None
    result["navigation_pose_age_p95_ms"] = float(np.quantile(ages, .95)*1000) if ages else None
    result["processing_schedule_sha256"] = sha256(output/"sensors/processing_schedule.json")
    result["schedule_timing"] = [r["schedule_timing"] for r in native_records]
    result["acquisition_schedule"] = sensors.schedule_receipt
    validate_schedule_evidence(result, sensors.schedule_receipt)
    result["latency_semantics"] = "same DelayedNativeState quantization and arrival-step hold; legacy cumulative callbacks+native, candidate selected PNG IPC+native; simulation generation/evidence archival separately measured"
    if execution is not None:
        # Durable per-episode order evidence: declared position in the shard,
        # actual position in this process and wall-clock bounds.
        result["execution"] = dict(execution, started_utc=started_utc,
            finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started_wall)
    NAV.write_json(output/"episode.json",result)
    import shutil
    shutil.rmtree(cache)
    return result

def declared_protocol(design=V1_DESIGN):
    """Exact predeclared protocol for one frozen design (default: completed v1)."""
    if design == V1_DESIGN:
        return _v1_protocol()
    if design == ORDER_DESIGN:
        return _order_protocol()
    raise ValueError(f"unknown processing-schedule design: {design!r}")


def protocol_design(protocol):
    design = protocol.get("design", V1_DESIGN) if isinstance(protocol, dict) else None
    if design not in DESIGNS:
        raise ValueError("unknown processing-schedule design")
    return design


def other_pipeline(pipeline):
    return PIPELINES[1 - PIPELINES.index(pipeline)]


def counterbalanced_development(seeds=ORDER_SEEDS, arms=ORDER_ARMS):
    """Within-arm ABBA/BAAB order with one fresh shard per seed x condition x arm.

    Pair (seed index i, condition index c, route index r) runs
    PIPELINES[(i + c + r) % 2] first. Each shard runs one unscored warm-up per
    pipeline, then the straight pair, then the occluders pair in the opposite
    order. Each pipeline therefore runs first in exactly half of the pairs of
    every arm, condition and route, and in half of every seed's pairs.
    """
    cells, shards, pair_index = [], [], 0
    for i, seed in enumerate(seeds):
        for c, condition in enumerate(CONDITIONS):
            for arm in arms:
                name = f"s{seed}-{condition}-{arm}"
                lead = PIPELINES[(i + c) % 2]
                warmups = [dict(case=WARMUP_CASE, seed=WARMUP_SEED, condition=condition, arm=arm, pipeline=pipeline,
                                role="warmup", shard=name, shard_execution_index=k)
                           for k, pipeline in enumerate((lead, other_pipeline(lead)))]
                index, pairs = len(warmups), []
                for r, case in enumerate(CASES):
                    first = PIPELINES[(i + c + r) % 2]
                    for position, pipeline in enumerate((first, other_pipeline(first)), 1):
                        cells.append(dict(case=case, seed=seed, condition=condition, arm=arm, pipeline=pipeline,
                            role="scored", pair_index=pair_index, position=position, first_pipeline=first,
                            shard=name, shard_execution_index=index))
                        index += 1
                    pairs.append(pair_index)
                    pair_index += 1
                shards.append(dict(name=name, seed=seed, condition=condition, arm=arm, cases=list(CASES),
                                   warmup_cells=warmups, pair_indices=pairs))
    return cells, shards


def counterbalanced_smoke(arms=ORDER_ARMS):
    """One smoke shard: nominal warm-ups, then alternating-order condition pairs."""
    warmups = [dict(case=WARMUP_CASE, seed=WARMUP_SEED, condition="nominal", arm=arm, pipeline=pipeline,
                    role="warmup", shard="smoke", shard_execution_index=2 * a + k)
               for a, arm in enumerate(arms)
               for k, pipeline in enumerate((PIPELINES[a % 2], other_pipeline(PIPELINES[a % 2])))]
    cells, index, pair_index = [], len(warmups), 0
    for c, condition in enumerate(CONDITIONS):
        for a, arm in enumerate(arms):
            first = PIPELINES[(c + a) % 2]
            for position, pipeline in enumerate((first, other_pipeline(first)), 1):
                cells.append(dict(case="straight", seed=SMOKE_SEED, condition=condition, arm=arm, pipeline=pipeline,
                    role="smoke", pair_index=pair_index, position=position, first_pipeline=first,
                    shard="smoke", shard_execution_index=index))
                index += 1
            pair_index += 1
    return warmups, cells


# Exact two-sided McNemar power (alpha .05) for 6 pairs per seed, recomputed
# by tests/test_processing_schedule_campaign.py from scipy.stats.binomtest.
ORDER_POWER_SCENARIOS = {
    "v1_like": dict(bounded_only=.90, legacy_only=0., power={"12": 1.0, "24": 1.0, "48": 1.0}),
    "half_v1_effect": dict(bounded_only=.48, legacy_only=.02, power={"12": .478, "24": .951, "48": 1.0}),
    "third_v1_effect": dict(bounded_only=.32, legacy_only=.02, power={"12": .130, "24": .680, "48": .978}),
    "pure_order_artifact": dict(bounded_only=.40, legacy_only=.40, power={"12": .023, "24": .030, "48": .034}),
}


def _order_protocol():
    cells, shards = counterbalanced_development()
    smoke_warmups, smoke_cells = counterbalanced_smoke()
    v1 = _v1_protocol()
    pairs = len(cells) // 2
    warmups = len(smoke_warmups) + sum(len(s["warmup_cells"]) for s in shards)
    first = {arm: {p: sum(c["position"] == 1 and c["pipeline"] == p for c in cells if c["arm"] == arm) for p in PIPELINES}
             for arm in ORDER_ARMS}
    return dict(schema=SCHEMA + "-counterbalanced-protocol", design=ORDER_DESIGN,
        parent=dict(design=V1_DESIGN, scientific_protocol_sha256="b41d516b7810fa851deea4c160c1efa2eff0d75e39037bb7bb24f265d2e5d34d",
            source_commit="152ccf4d8a616c5725bfb0c44ac838818e028ee3", campaign_job="21801949",
            result="LiDAR-IMU clean goals 0/12 legacy_cumulative vs 11/12 bounded_latest; stereo 0/12 in both",
            order_audit="within_arm_counterbalanced=false: stereo always ran legacy first, LiDAR-IMU always bounded first"),
        pipelines=list(PIPELINES), arms=list(ORDER_ARMS),
        conditions={c: condition_spec(c) for c in CONDITIONS},
        smoke=dict(seconds=18., warmup_cells=smoke_warmups, cells=smoke_cells),
        development=dict(seconds=40., seeds=list(ORDER_SEEDS), cells=cells, shards=shards),
        warmup=dict(seconds=WARMUP_SECONDS, seed=WARMUP_SEED, case=WARMUP_CASE, scored=False,
            per="every shard (smoke and development), one per pipeline, same arm and condition as the shard (smoke: nominal)",
            order="same as the shard's first pair, before any smoke or scored cell",
            raw_images_archived=False, retained_receipts=list(WARMUP_RETAINED),
            reason="each shard re-extracts scratch and starts a fresh Python process and every episode starts fresh native processes; warm-ups bring imports, EGL/GPU, page cache, recorder threads and CPU/GPU clocks to a running state on both code paths before any scored cell",
            exclusion="reserved seed, role=warmup, separate output tree, never in completed cells, gates, groups or pairs; collector rejects any scored row that is a warm-up"),
        counterbalance=dict(rule="pair(seed index i, condition index c, route index r) runs PIPELINES[(i+c+r)%2] first; shard order: warm-ups, straight pair, occluders pair",
            within_shard="ABBA or BAAB", first_pipeline_counts_by_arm=first,
            balanced_within=["arm", "condition", "route", "condition x route (4/4)", "seed (3/3)", "shard position (12/12)"]),
        pairing="same seed/condition/route/arm; both routes of one seed/condition in one fresh shard; every pair in the same GPU allocation and process",
        initialization=v1["initialization"], unchanged=v1["unchanged"], candidate=v1["candidate"], reference=v1["reference"],
        evidence=v1["evidence"], profile=v1["profile"],
        smoke_execution_gate=f"all {len(smoke_cells)} same-source/input smoke cells and {len(smoke_warmups)} warm-ups execute with finite physics, native I/O and verified evidence",
        nominal_readiness_gate="per pipeline/arm: nominal 18 s smoke cell (warm-ups excluded) full horizon, accepted native pose, zero falls/contacts/nonfinite; scientific negative gates retain declared development cells as UNRUN",
        arms_decision="LiDAR-IMU only. The 48-pair budget of both arms x 4 seeds is spent on 8 seed clusters of the one estimator with a goal gain to confirm. Stereo was 0/12 in both schedules (every bounded stereo episode latched a permanent stop after native map identity was lost), so it has no goal gain to order-test, and its wrapper is being changed in a separate follow-up. Cost: the stereo stale-fraction and latency differences keep their v1 order qualification. Both arms x 8 seeds would need 51 launcher jobs, above the 32-job cap.",
        hypotheses=dict(
            H1_primary="In LiDAR-IMU, bounded_latest yields more clean goals than legacy_cumulative in matched pairs when each schedule runs first in exactly half of the pairs and both pipelines are warmed up in every shard.",
            H2_order="Execution position within a pair (first or second) does not change clean goals or latency (secondary period-effect test).",
            H3_latency="bounded_latest lowers per-episode p95 modeled pipeline service and p95 navigation pose age relative to legacy_cumulative (secondary estimation)."),
        primary_outcome=dict(arm="lidar_inertial", pairs=pairs,
            metric="paired clean-goal difference, bounded_latest minus legacy_cumulative, over all declared LiDAR-IMU pairs",
            clean_goal="evaluator-trace goal within 0.30 m, full 40 s horizon, no fall, contact or nonfinite state (unchanged from v1)",
            test="exact two-sided McNemar: scipy.stats.binomtest(bounded_only, bounded_only + legacy_only, 0.5); zero discordant pairs reported as p = 1",
            alpha=.05),
        decision_rule=dict(
            CONFIRMED=f"all {pairs} pairs measured AND McNemar p < 0.05 AND bounded_only > legacy_only AND seed-cluster bootstrap 95% interval of the goal-rate difference has lower bound > 0 AND bounded clean goals exceed legacy clean goals within each execution sequence (legacy-first pairs and bounded-first pairs separately)",
            NOT_CONFIRMED="complete cohort but any CONFIRMED condition fails; the predeclared negative result",
            INCOMPLETE_NO_VERDICT="any execution error, missing, duplicated or gate-UNRUN primary cell; no primary verdict is computed",
            ORDER_EFFECT_DETECTED="flag: first-position test or sequence-by-winner Fisher test p < 0.05 for clean goals, reported regardless of the primary verdict",
            ATTENUATED="flag: CONFIRMED but the bootstrap interval upper bound is below the v1 LiDAR-IMU difference 11/12"),
        negative_result="NOT_CONFIRMED means the v1 0/12 -> 11/12 LiDAR-IMU gain did not survive balanced order and warm-up. An order artifact would show bounded wins concentrated in bounded-first pairs, no bounded advantage in legacy-first pairs and a significant order test. A significant excess of legacy-only pairs is reported as a reversal. Every outcome is reported beside v1; none replaces it.",
        statistics=dict(
            per_pair="difference bounded_latest minus legacy_cumulative for clean goal, stale fraction and each latency metric",
            goal_rates="Wilson 95% interval per pipeline, scipy.stats.binomtest(k, n).proportion_ci(method='wilson'); episodes treated as independent",
            discordant_share="exact Clopper-Pearson 95% interval of bounded_only / discordant",
            goal_difference_interval="seed-cluster bootstrap percentile 95% interval: resample seeds with replacement keeping all pairs of a seed together; statistic is the pooled mean paired difference; exact enumeration of all multisets with multinomial weights (6435 for 8 seeds), inverted-CDF percentiles; Monte Carlo (20000 resamples, numpy seed 20261014) only above 200000 multisets",
            cluster_robustness="seed-level sign test on per-seed mean goal difference (binomtest, zero seeds dropped)",
            latency="per-pair differences of per-episode p95 modeled pipeline service (full_pipeline_p95_ms), p95 navigation pose age (navigation_pose_age_p95_ms) and p95 native client wall time (client_wall_p95_ms): mean and median with seed-cluster bootstrap 95% intervals; a pair missing a value is excluded from that metric only and counted",
            stale="paired stale-stop fraction difference (v1 co-primary, secondary here) with a seed-cluster bootstrap 95% interval",
            seed_variability="per-seed goal rates, goal difference and mean latency differences; SD (ddof=1), minimum and maximum across seeds",
            order_effect="first versus second position within arm: (a) discordant pairs won by the first-run pipeline, binomtest against 0.5; (b) Fisher exact test of sequence (legacy-first, bounded-first) by winner (bounded, legacy) among discordant pairs; (c) goal rate by position for each pipeline with Wilson intervals; (d) latency period effect (mean difference in bounded-first pairs minus legacy-first pairs)/2, two-sided permutation test (exact up to 200000 assignments, else 20000 resamples, numpy seed 20261014)",
            per_condition="goal rates and differences by condition, descriptive",
            multiplicity="one primary test; all other p-values are secondary and unadjusted",
            implementation="scripts/bench/processing_schedule_order_analysis.py on the strict collector campaign_result.json; scipy 1.15.3, numpy 1.26.0"),
        exclusion_rules=[
            "warm-up cells are excluded by construction: reserved seed, role=warmup, raw images not archived, compact receipts in a separate output tree, never in completed cells, readiness gates, groups or pairs",
            "smoke cells are never scored",
            "no scored episode is excluded after execution; falls, contacts, early termination or nonfinite physics count as no clean goal",
            "a pair missing a continuous metric stays in the goal analysis and is excluded from that metric only, with counts reported",
            "cells left UNRUN by a negative nominal smoke gate are retained as UNRUN, never zero; any UNRUN primary cell gives INCOMPLETE_NO_VERDICT"],
        stopping_rules=[
            "one allocation and one attempt: the dispatcher refuses a second submission in the campaign directory",
            "no interim look at outcomes, no optional stopping and no seeds added after outcomes",
            "an execution error (runner exception, evidence or hash mismatch, archive cap, quota or wall time) stops all remaining shards without retry; the campaign is INCOMPLETE and no primary verdict is computed; completed shards stay descriptive",
            "a rerun needs a new campaign directory, a fresh smoke and this unchanged protocol",
            "the analysis runs once, after strict collection, with the frozen analysis script"],
        power=dict(test="exact two-sided McNemar, alpha 0.05", pairs_per_seed=6,
            scenarios=ORDER_POWER_SCENARIOS,
            design_effect="seed clustering with design effect 1.5-2 lowers effective pairs: 24 -> 12-16 (half-effect power 0.48-0.75), 48 -> 24-32 (half-effect power 0.95-0.99)",
            choice="8 new seeds (48 pairs) keep power >= 0.95 for half the v1 effect under design effect 2 and give 8 bootstrap clusters (6435 multisets) instead of 4 (35)"),
        resources=dict(partition="gpu", constraint="rtx8000", cpus=2, memory_gb=16, gpus=1, time_limit="12:00:00",
            reason="v1 ran on cn-gpu6 (Quadro RTX 8000, cascadelake); legacy modeled latency charges measured wall-clock callback cost, so hardware is pinned"),
        budget=dict(scored_episodes=len(cells), pairs=pairs, development_shards=len(shards), warmup_episodes=warmups,
            smoke_episodes=len(smoke_cells), estimated_wall_hours="2.3-3.0", estimated_raw_gib="18-22", campaign_cap_gib=48,
            estimate_basis="v1 job 21801949 LiDAR-IMU runner time per episode 46/60/87 s (nominal/10 Hz/exposure), 18 s warm-ups 25-45 s, per-shard archive and dispatcher verification about 50 s; v1 LiDAR-IMU raw output 2.2 GiB per seed (4 pairs per condition x 3); warm-up images not archived"),
        primary_metrics=["paired clean-goal difference (LiDAR-IMU)"],
        secondary_metrics=["paired navigation stale-stop fraction difference", "p95 modeled pipeline service", "p95 navigation pose age",
            "p95 native client", "order (period) effect", "variability across seeds", "falls/contacts", "warm-up episodes (reported separately)"],
        navigation_denominator=v1["navigation_denominator"],
        scope="eight-seed within-arm counterbalanced LiDAR-IMU schedule-model confirmation in simulation; ideal simulated sensing; no hardware timing, population or stereo claims",
        no_retuning_after_outcomes=True, source_change_requires_fresh_smoke=True)


def _v1_protocol():
    def cells(seeds, cases):
        result, pair_index = [], 0
        for seed in seeds:
            for condition in CONDITIONS:
                for case in cases:
                    for arm in ARMS:
                        order = PIPELINES if pair_index % 2 == 0 else PIPELINES[::-1]
                        for pipeline in order:
                            result.append(dict(case=case, seed=seed, condition=condition, arm=arm, pipeline=pipeline))
                        pair_index += 1
        return result
    return dict(schema=SCHEMA + "-protocol", pipelines=list(PIPELINES), arms=list(ARMS),
        conditions={c: condition_spec(c) for c in CONDITIONS},
        smoke=dict(seconds=18., cells=cells([SMOKE_SEED], ["straight"])),
        development=dict(seconds=40., cells=cells(DEVELOPMENT_SEEDS, CASES)),
        pairing="same seed/condition/route/arm; pipeline order alternates deterministically by declared pair index; each pair in same GPU allocation",
        initialization=BASE.declared_protocol()["initialization"],
        unchanged="frozen actor, routes, native settings/thresholds, .35s capture-age cutoff, known-start registration, raw LiDAR brake, delivery quantization and no submission on arrival step",
        candidate="one pending latest joined camera/LiDAR event (one latest identity per stream), one in-service estimator request; all causal intervening IMU retained; selected PNG IPC plus native measured service, simulator generation and evidence separate",
        reference="current cumulative acquisition callback charging plus native client service; measured again in this new source with timing/identity instrumentation",
        evidence="candidate real single-worker queue capacity8, immutable buffers, backpressure/errors reported and joined before PASS; archive clock wall-time separate from modeled simulation time",
        profile="before native smoke, first40 retained stereo pairs and original cloud/IMU arrays: required PNG transport and evidence recording microbenchmark with exact decoded values/timestamp checks, no new navigation score",
        smoke_execution_gate="all12 same-source/input cells execute with finite physics, native I/O and verified evidence",
        nominal_readiness_gate="per pipeline/arm: nominal18s full horizon, accepted native pose, zero falls/contacts/nonfinite; scientific negative gates retain declared development cells as UNRUN",
        primary_metrics=["paired navigation stale-stop fraction difference", "paired clean-goal difference"],
        secondary_metrics=["capture age p95", "modeled pipeline p95", "queue/supersession", "initialization/tracking", "falls/contacts"],
        navigation_denominator="pre-step decision timestamps >=10s, actual decision count; shortened horizons explicit",
        scope="two-seed bounded schedule-model development; ideal simulated sensing, no concurrent hardware timing or population/confirmation claims",
        no_retuning_after_outcomes=True, source_change_requires_fresh_smoke=True)


def source_hashes():
    return BASE.source_hashes()


def gate_key(pipeline, arm):
    return pipeline + "/" + arm


def nominal_arm_readiness(rows, arms=ARMS):
    readiness = {}
    for pipeline in PIPELINES:
        for arm in arms:
            candidates = [r for r in rows if r["pipeline"] == pipeline and r["method"] == arm and r["condition"] == "nominal"]
            readiness[gate_key(pipeline, arm)] = bool(len(candidates) == 1 and candidates[0]["native_tracked_frames"] > 0
                and candidates[0]["full_horizon"] and not any(candidates[0][k] for k in ("fell", "collision", "nonfinite")))
    return readiness


def validate_smoke(smoke, *, sources, inputs, design=V1_DESIGN):
    protocol = declared_protocol(design)
    cells, warmups = protocol["smoke"]["cells"], protocol["smoke"].get("warmup_cells", [])
    if (smoke.get("schema") != SCHEMA or smoke.get("phase") != "smoke" or smoke.get("status") != "PASS"
            or smoke.get("execution_gate") is not True or smoke.get("source_sha256") != sources
            or smoke.get("input_sha256") != inputs or smoke.get("completed_cells") != cells
            or smoke.get("problems") or smoke.get("unrun_cells") or smoke.get("episodes") != len(cells)
            or smoke.get("design", V1_DESIGN) != design or smoke.get("warmup_cells", []) != warmups
            or len(smoke.get("warmup_episode_files", [])) != len(warmups)):
        raise ValueError("complete matching-source/input smoke of the declared design required")
    readiness = smoke.get("arm_development_ready")
    expected = {gate_key(p, a) for p in PIPELINES for a in protocol["arms"]}
    if not isinstance(readiness, dict) or set(readiness) != expected or any(type(x) is not bool for x in readiness.values()):
        raise ValueError("explicit pipeline/arm readiness required")
    return readiness


PAIR_METRICS = ("full_pipeline_p95_ms", "navigation_pose_age_p95_ms", "client_wall_p95_ms", "fell", "collision")


def paired_summary(rows, orders=None, uncertainty="two seed clusters, descriptive paired outcomes only; no population confidence interval"):
    keys = ("case", "seed", "condition", "method")
    grouped = {}
    for row in rows:
        pair = grouped.setdefault(tuple(row[k] for k in keys), {})
        if row["pipeline"] in pair:
            raise ValueError("duplicate paired pipeline")
        pair[row["pipeline"]] = row
    pairs, incomplete = [], []
    for key, pair in sorted(grouped.items()):
        if set(pair) != set(PIPELINES):
            incomplete.append(dict(zip(keys, key)))
            continue
        legacy, candidate = (pair[p] for p in PIPELINES)
        comparable = legacy["stale_navigation_fraction"] is not None and candidate["stale_navigation_fraction"] is not None
        entry = dict(zip(keys, key), clean_goal_difference=int(candidate["success"])-int(legacy["success"]),
            stale_fraction_difference=candidate["stale_navigation_fraction"]-legacy["stale_navigation_fraction"] if comparable else None,
            stale_comparison_available=comparable,
            legacy_stale_fraction=legacy["stale_navigation_fraction"], candidate_stale_fraction=candidate["stale_navigation_fraction"],
            legacy_clean_goal=legacy["success"], candidate_clean_goal=candidate["success"],
            both_full_horizon=legacy["full_horizon"] and candidate["full_horizon"])
        if orders is not None:
            # Retained per-pair metrics and the verified execution order feed
            # the predeclared order-confirmation analysis.
            for metric in PAIR_METRICS:
                entry["legacy_" + metric], entry["candidate_" + metric] = legacy.get(metric), candidate.get(metric)
            order = orders[key]
            entry.update(execution_order=list(order), first_pipeline=order[0],
                         sequence="legacy_first" if order[0] == PIPELINES[0] else "bounded_first")
        pairs.append(entry)
    stale_pairs = [p for p in pairs if p["stale_comparison_available"]]
    return dict(complete_pairs=len(pairs), stale_comparable_pairs=len(stale_pairs), incomplete_pairs=incomplete, pairs=pairs,
        mean_paired_stale_fraction_difference=float(np.mean([p["stale_fraction_difference"] for p in stale_pairs])) if stale_pairs else None,
        clean_goal_count_difference=sum(p["clean_goal_difference"] for p in pairs) if pairs else None,
        negative_difference_means="candidate fewer stale stops; positive clean-goal difference means candidate more goals",
        uncertainty=uncertainty)


def declared_pair_orders(protocol):
    """Pair key (case, seed, condition, arm) -> declared pipeline execution order."""
    orders = {}
    for cell in protocol["development"]["cells"]:
        orders.setdefault((cell["case"], cell["seed"], cell["condition"], cell["arm"]), []).append(cell["pipeline"])
    if any(sorted(order) != sorted(PIPELINES) for order in orders.values()):
        raise ValueError("every declared pair needs each pipeline exactly once")
    return {key: tuple(order) for key, order in orders.items()}


def order_audit(pairs, arms):
    by_arm = {}
    for arm in arms:
        rows = [p for p in pairs if p["method"] == arm]
        counts = {p: sum(row.get("first_pipeline") == p for row in rows) for p in PIPELINES}
        by_arm[arm] = dict(pairs=len(rows), first_pipeline_counts=counts,
            counterbalanced=bool(rows) and counts[PIPELINES[0]] == counts[PIPELINES[1]] == len(rows) / 2)
    return dict(by_arm=by_arm, within_arm_counterbalanced=bool(by_arm) and all(v["counterbalanced"] for v in by_arm.values()),
                scope="declared order verified against recorded execution order of every scored episode")


def validate_schedule_evidence(row, evidence):
    if evidence != row["acquisition_schedule"] or evidence["pipeline"] != row["pipeline"]:
        raise ValueError("embedded timing evidence differs from retained receipt")
    queue = evidence["queue"]
    def count(value):
        if type(value) is not int or value < 0:
            raise ValueError("nonnegative integer queue counts required")
        return value
    if (count(queue["produced"]) != count(queue["selected"])+count(queue["superseded"])+count(queue["pending"])
            or queue["pending"] > 1 or queue["maximum_waiting_events"] != 1
            or queue["selected"] != row["native_frames"] or queue["selected"] != len(evidence["requests"])):
        raise ValueError("joined queue count conservation failure")
    for name, stream in queue["streams"].items():
        if (count(stream["produced_unique"]) != count(stream["selected_unique"])+count(stream["superseded_before_selection"])+count(stream["pending_unselected"])
                or stream["pending_unselected"] > 1
                or count(stream["dispatches"]) != count(stream["selected_unique"])+count(stream["reused_dispatches"])
                or stream["dispatches"] != queue["selected"]):
            raise ValueError("sensor stream count conservation failure")
        prefix = "camera" if name == "camera" else "lidar"
        if evidence[prefix + "_captures_total"] != stream["produced_unique"] + len(evidence[prefix + "_never_joined"]):
            raise ValueError("pre-join sensor captures missing from ledger")
    recorder = evidence["recorder"]
    if recorder.get("joined") is not True or recorder.get("pending", 0) != 0:
        raise ValueError("unjoined evidence writer")
    if row["pipeline"] == "bounded_latest" and (recorder["maximum_pending"] > recorder["capacity"] or recorder["tasks"] != evidence["camera_captures_total"]):
        raise ValueError("evidence writer bound or count failure")
    timings = row["schedule_timing"]
    if len(timings) != len(evidence["requests"]):
        raise ValueError("missing service timing")
    for timing, request in zip(timings, evidence["requests"]):
        expected = service_timing(request_s=request["request_s"], capture_s=timing["capture_s"],
            preprocessing_s=request["preprocessing_service_s"], native_client_s=timing["native_client_s"], policy_dt=row["latency_quantization_s"])
        if any(timing[k] != v for k, v in expected.items()):
            raise ValueError("service or capture-age timing mismatch")
        capture = request["camera_capture_s"] if row["method"] == "stereo" else request["lidar_capture_s"]
        if not math.isclose(timing["capture_s"], capture, abs_tol=1e-8):
            raise ValueError("estimator capture timestamp was rewritten")
        q = request["queue"]
        if q["enqueue_s"] > request["request_s"] or not math.isclose(q["queue_residence_s"], request["request_s"]-q["enqueue_s"], abs_tol=1e-8):
            raise ValueError("noncausal queue timing")
    if len(row["response_deliveries"]) + row["responses_still_pending_at_horizon"] != row["native_frames"]:
        raise ValueError("native response conservation failure")
    for delivered, timing in zip(row["response_deliveries"], timings):
        if not math.isclose(delivered["arrival_s"], timing["delivery_s"], abs_tol=1e-8) or delivered["wall_seconds"] != timing["service_s"]:
            raise ValueError("reported delivery differs from actual causal state")
    return True


def aggregate_groups(rows, arms=ARMS):
    groups = {}
    for pipeline in PIPELINES:
        for arm in arms:
            values = [r for r in rows if r["method"] == arm and r["pipeline"] == pipeline]
            available = [r["stale_navigation_fraction"] for r in values if r["stale_navigation_fraction"] is not None]
            groups[gate_key(pipeline, arm)] = dict(episodes=len(values), clean_goals=sum(r["success"] for r in values),
                falls=sum(r["fell"] for r in values), contacts=sum(r["collision"] for r in values),
                mean_stale_fraction=float(np.mean(available)) if available else None)
    return groups


def cell_key(cell):
    return tuple(cell[k] for k in ("case", "seed", "condition", "arm", "pipeline"))


def verified_episode_row(directory, relative, expected_sha, key, seconds, *, raw=True):
    """Load one retained episode; re-derive outcome and stale stops from its trace.

    ``raw=False`` is only for unscored warm-ups, whose sensor/evaluator arrays
    are deliberately not archived; their timing receipts are still verified.
    """
    episode_path = (Path(directory) / relative).resolve()
    if not episode_path.is_relative_to(Path(directory).resolve()):
        raise ValueError("episode file leaves its result directory")
    if expected_sha is None or sha256(episode_path) != expected_sha:
        raise ValueError("retained episode differs from result hash")
    row = json.loads(episode_path.read_text())
    if (row.get("case"), row.get("seed"), row.get("condition"), row.get("method"), row.get("pipeline")) != key:
        raise ValueError("episode identities differ from declared cell")
    lineage = (("inference_sensors.npz", "inference_sensors_sha256"), ("evaluator_truth.npz", "evaluator_truth_sha256")) if raw else ()
    for name, field in lineage + (("native/stream_receipt.json", "stream_receipt_sha256"),
                                  ("sensors/processing_schedule.json", "processing_schedule_sha256")):
        if sha256(episode_path.parent / name) != row[field]:
            raise ValueError("retained raw/evaluator lineage differs")
    validate_schedule_evidence(row, json.loads((episode_path.parent / "sensors/processing_schedule.json").read_text()))
    trace = row["trace"]
    if not trace or any(not math.isfinite(r["evaluator_goal_distance_m"]) for r in trace):
        raise ValueError("finite retained evaluator trace required")
    reached = any(r["evaluator_goal_distance_m"] <= .30 for r in trace)
    contact = any(r["evaluator_contact"] for r in trace)
    fell = any(r["evaluator_tilt_rad"] >= .78 or r["evaluator_height_m"] < row["evaluator_initial_height_m"] - .25 for r in trace)
    full = row["elapsed_simulation_s"] >= seconds - 1e-6
    success = reached and full and not (contact or fell or row["nonfinite"])
    if any(bool(row[k]) != actual for k, actual in (("goal_reached_evaluator", reached), ("collision", contact),
            ("fell", fell), ("full_horizon", full), ("success", success))):
        raise ValueError("reported outcome differs from retained evaluator trace")
    nav = [r for r in trace if r["decision_timestamp_s"] >= 10.-1e-9]
    stale = sum(r["reason"] == "native_stale_stop" for r in nav)
    fraction = stale/len(nav) if nav else None
    if row["navigation_decisions"] != len(nav) or row["stale_navigation_decisions"] != stale or row["stale_navigation_fraction"] != fraction:
        raise ValueError("reported stale stops differ from decision trace")
    return row


WARMUP_SUMMARY_FIELDS = ("success", "goal_reached_evaluator", "fell", "collision", "full_horizon", "native_frames",
    "native_tracked_frames", "client_wall_p95_ms", "full_pipeline_p95_ms", "navigation_pose_age_p95_ms",
    "stale_navigation_fraction", "elapsed_simulation_s")


def warmup_summary(row, cell):
    return dict({k: cell[k] for k in ("shard", "shard_execution_index", "case", "seed", "condition", "arm", "pipeline")},
                wall_seconds=(row.get("execution") or {}).get("wall_seconds"), **{k: row.get(k) for k in WARMUP_SUMMARY_FIELDS})


def warmup_report(warmups):
    def mean(values, field):
        present = [v[field] for v in values if v[field] is not None]
        return float(np.mean(present)) if present else None
    by_pipeline = {}
    for pipeline in PIPELINES:
        values = [w for w in warmups if w["pipeline"] == pipeline]
        by_pipeline[pipeline] = dict(episodes=len(values), clean_goals=sum(bool(w["success"]) for w in values),
            falls=sum(bool(w["fell"]) for w in values), contacts=sum(bool(w["collision"]) for w in values),
            mean_client_wall_p95_ms=mean(values, "client_wall_p95_ms"), mean_full_pipeline_p95_ms=mean(values, "full_pipeline_p95_ms"),
            mean_navigation_pose_age_p95_ms=mean(values, "navigation_pose_age_p95_ms"), mean_wall_seconds=mean(values, "wall_seconds"))
    return dict(episodes=len(warmups), scored=False, by_pipeline=by_pipeline, episodes_detail=warmups,
                scope="unscored warm-ups reported separately; never in readiness gates, groups, pairs or the primary analysis")


def verify_counterbalanced_shard(result, protocol, path, seen):
    """Warm-ups ran first and scored cells ran in declared order, once per shard."""
    shards = {s["name"]: s for s in protocol["development"]["shards"]}
    names = {c.get("shard") for c in result["completed_cells"] + result["unrun_cells"]}
    shard = shards.get(next(iter(names))) if len(names) == 1 else None
    if shard is None or shard["name"] in seen:
        raise ValueError("each development result must hold exactly one declared, unduplicated shard")
    seen.add(shard["name"])
    declared = [c for c in protocol["development"]["cells"] if c["shard"] == shard["name"]]
    ran = result["completed_cells"]
    unran = [{k: v for k, v in c.items() if k != "reason"} for c in result["unrun_cells"]]
    if (len(ran) + len(unran) != len(declared) or any(c not in ran and c not in unran for c in declared)
            or [c for c in declared if c in ran] != ran):
        raise ValueError("shard cells differ from the declared execution order")
    if result.get("warmup_cells") != shard["warmup_cells"] or len(result.get("warmup_episode_files", [])) != len(shard["warmup_cells"]):
        raise ValueError("declared shard warm-ups did not run")
    scored_seeds = {c["seed"] for c in protocol["development"]["cells"]}
    summaries = []
    for k, (cell, relative) in enumerate(zip(result["warmup_cells"], result["warmup_episode_files"])):
        if cell["role"] != "warmup" or cell["seed"] in scored_seeds:
            raise ValueError("warm-up cell collides with the scored cohort")
        row = verified_episode_row(path.parent, relative, result.get("warmup_episode_sha256", {}).get(relative), cell_key(cell),
                                   protocol["warmup"]["seconds"], raw=False)
        execution = row.get("execution") or {}
        if (execution.get("role") != "warmup" or execution.get("shard") != shard["name"]
                or execution.get("shard_execution_index") != cell["shard_execution_index"] or execution.get("process_episode_index") != k):
            raise ValueError("warm-up execution record differs from its declared position")
        summaries.append(warmup_summary(row, cell))
    return shard, summaries


def collect_campaign(root, smoke_path, output):
    """Strict exact-cohort collection; skipped gates remain unrun, never zero."""
    root = Path(root).resolve()
    smoke = json.loads(Path(smoke_path).read_text())
    design = protocol_design(smoke)
    protocol = declared_protocol(design)
    arms, seconds, counterbalanced = tuple(protocol["arms"]), protocol["development"]["seconds"], design != V1_DESIGN
    ready = validate_smoke(smoke, sources=smoke.get("source_sha256"), inputs=smoke.get("input_sha256"), design=design)
    declared = {cell_key(c): c for c in protocol["development"]["cells"]}
    expected = set(declared)
    observed, unrun, rows, lineage, warmups, seen_shards, positions = set(), set(), [], [], [], set(), {}
    for path in sorted(root.rglob("campaign_result.json")):
        result = json.loads(path.read_text())
        if result.get("schema") != SCHEMA or result.get("phase") != "development":
            continue
        if (result.get("status") != "PASS" or result.get("problems") or result.get("source_sha256") != smoke["source_sha256"]
                or result.get("input_sha256") != smoke["input_sha256"] or result.get("arm_development_ready") != ready
                or result.get("design", V1_DESIGN) != design):
            raise ValueError("incomplete, mixed-source or mixed-design development result")
        if len(result["completed_cells"]) != len(result["episode_files"]) or result["episodes"] != len(result["completed_cells"]):
            raise ValueError("episode list differs from completed cells")
        shard = None
        if counterbalanced:
            shard, summaries = verify_counterbalanced_shard(result, protocol, path, seen_shards)
            warmups.extend(summaries)
        for position, (cell, relative) in enumerate(zip(result["completed_cells"], result["episode_files"])):
            key = cell_key(cell)
            if key not in expected or key in observed or key in unrun or not ready[gate_key(cell["pipeline"], cell["arm"])]:
                raise ValueError("undeclared, duplicate or gated development cell")
            if counterbalanced and cell != declared[key]:
                raise ValueError("completed cell differs from its declaration")
            row = verified_episode_row(path.parent, relative, result["episode_sha256"].get(relative), key, seconds)
            if counterbalanced:
                execution = row.get("execution") or {}
                if (execution.get("role") != "scored" or execution.get("shard") != shard["name"]
                        or execution.get("shard_execution_index") != cell["shard_execution_index"]
                        or execution.get("process_episode_index") != len(shard["warmup_cells"]) + position):
                    raise ValueError("scored execution record differs from its declared position")
            positions[key] = (str(path), position)
            rows.append(row)
            observed.add(key)
        for cell in result["unrun_cells"]:
            key = cell_key(cell)
            if (key not in expected or key in observed or key in unrun or ready[gate_key(cell["pipeline"], cell["arm"])]
                    or cell["reason"] != "UNRUN_AFTER_NEGATIVE_NOMINAL_ARM_SMOKE"):
                raise ValueError("invalid unrun cell or gate")
            unrun.add(key)
        lineage.append({"path": str(path), "sha256": sha256(path)})
    missing = expected - observed - unrun
    if missing:
        raise ValueError(f"incomplete development collection: {len(missing)} undeclared-missing outcomes")
    if counterbalanced and seen_shards != {s["name"] for s in protocol["development"]["shards"]}:
        raise ValueError("every declared shard must report its warm-ups")
    orders = declared_pair_orders(protocol)
    for pair, order in orders.items():
        places = [positions.get(pair + (pipeline,)) for pipeline in order]
        if None not in places and (places[0][0] != places[1][0] or places[0][1] >= places[1][1]):
            raise ValueError("observed pair execution order differs from the declaration")
    groups = aggregate_groups(rows, arms)
    paired = paired_summary(rows, orders=orders, uncertainty=(
        "eight reset-seed clusters; predeclared inference by processing_schedule_order_analysis.py" if counterbalanced else
        "two seed clusters, descriptive paired outcomes only; no population confidence interval"))
    result = dict(schema=SCHEMA + "-collection", status="PASS", declared_cells=len(expected), measured_cells=len(observed),
        unrun_after_nominal_gate=len(unrun), missing_cells=[], arm_development_ready=ready, by_pipeline_arm=groups,
        paired=paired, source_results=lineage, qualified_smoke_sha256=sha256(smoke_path),
        scientific_status="COMPLETE_COUNTERBALANCED_COLLECTION_ANALYSIS_PENDING" if counterbalanced else "COMPLETE_BOUNDED_DEVELOPMENT_NO_CONFIRMATION",
        scope=protocol["scope"], design=design, development_seconds=seconds,
        order_audit=order_audit(paired["pairs"], arms), warmups=warmup_report(warmups))
    NAV.write_json(output, result)
    return result

def profile_retained_sequence(episode, output):
    """Before smoke: verify/reprocess the same first40 retained stereo pairs.

    This bounds input/transport validation without inventing another native
    trajectory. Both paths receive identical camera, scan and IMU values;
    measured costs here are PNG transport/evidence costs only. Native service
    and goal outcomes are measured by the subsequent real closed-loop cohort.
    """
    from PIL import Image
    episode, output = Path(episode), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("profile output must be empty")
    original = json.loads((episode / "episode.json").read_text())
    acquisition_path = episode / "sensors/acquisition.json"
    acquisition = json.loads(acquisition_path.read_text())
    sensor_path = episode / "inference_sensors.npz"
    if sha256(sensor_path) != original["inference_sensors_sha256"]:
        raise ValueError("retained original sensor data hash mismatch")
    sources = source_hashes()
    arrays = np.load(sensor_path, allow_pickle=False)
    scalar_inputs = {}
    for name in arrays.files:
        first, second = arrays[name].copy(), arrays[name].copy()
        if not np.array_equal(first, second) or not np.isfinite(first).all():
            raise ValueError("finite identical original cloud and IMU values required")
        scalar_inputs[name] = dict(shape=list(first.shape), dtype=str(first.dtype),
            sha256=hashlib.sha256(first.tobytes()).hexdigest())
    if np.any(np.diff(arrays["imu_timestamp_s"]) <= 0) or np.any(np.diff(arrays["scan_tail_s"]) <= 0):
        raise ValueError("ordered original LiDAR and IMU required")
    arrays.close()
    images = acquisition["images"][:40]
    if len(images) != 40:
        raise ValueError("first40 retained image pairs required")
    offered, timings, source_images = {}, {}, {}
    for pipeline in PIPELINES:
        directory = output / pipeline
        directory.mkdir()
        fake = object.__new__(ScheduledSensors)
        fake.output, fake.exposure_samples = directory, 1
        fake.wall_origin = time.monotonic()
        fake.stage_costs = dict(evidence_worker_s=0.)
        recorder = BoundedRecorder(8) if pipeline == "bounded_latest" else None
        offered[pipeline], timings[pipeline] = [], []
        try:
            for index, source in enumerate(images):
                paths = [episode / "sensors" / Path(source[name]).name for name in ("left", "right")]
                for name, path in zip(("left", "right"), paths):
                    digest = sha256(path)
                    if digest != source[name + "_sha256"]:
                        raise ValueError("retained original PNG hash mismatch")
                    source_images[str(path.relative_to(episode))] = digest
                pixels = tuple(np.asarray(Image.open(path).convert("RGB")) for path in paths)
                for a in pixels:
                    a.setflags(write=False)
                record = dict(timestamp_s=source["timestamp_s"],
                    left=str(directory / f"{index:05d}-left.png"), right=str(directory / f"{index:05d}-right.png"))
                samples = ((pixels[0],), (pixels[1],))
                started = time.monotonic()
                if recorder:
                    recorder.submit(fake._archive, record, pixels, samples)
                    inference_started = time.monotonic()
                    for name, array in zip(("left", "right"), pixels):
                        Image.fromarray(array).save(directory / f"{index:05d}-ipc-{name}.png")
                    service = time.monotonic() - inference_started
                    selected_paths = [directory / f"{index:05d}-ipc-{name}.png" for name in ("left", "right")]
                else:
                    fake._archive(record, pixels, samples)
                    service = time.monotonic() - started
                    selected_paths = [Path(record[name]) for name in ("left", "right")]
                decoded = decoded_frame_identity(*(np.asarray(Image.open(path).convert("RGB")) for path in selected_paths))
                if decoded != decoded_frame_identity(*pixels):
                    raise ValueError("PNG transport changed decoded sensor values")
                offered[pipeline].append(dict(timestamp_s=source["timestamp_s"], decoded_identity=decoded))
                timings[pipeline].append(service)
        finally:
            if recorder:
                recorder.close()
        timings[pipeline] = dict(selected_transport_and_charged_evidence_s=timings[pipeline],
            total_wall_s=time.monotonic()-fake.wall_origin, archive_worker_s=fake.stage_costs["evidence_worker_s"],
            recorder=recorder.receipt() if recorder else dict(mode="synchronous", joined=True))
    for pipeline in PIPELINES:
        timings[pipeline]["transport_queue_model"] = replay_transport_queue(offered[pipeline],
            timings[pipeline]["selected_transport_and_charged_evidence_s"], cumulative=pipeline == "legacy_cumulative")
    if source_hashes() != sources:
        raise ValueError("source changed during same-input profile")
    result = dict(schema=SCHEMA + "-profile", status="PASS", input_equivalence=compare_replay_inputs(*(offered[p] for p in PIPELINES)),
        source_sha256=sources, original_episode_sha256=sha256(episode/"episode.json"),
        original_acquisition_sha256=sha256(acquisition_path), original_sensors_sha256=sha256(sensor_path),
        selected_source_image_sha256=source_images, unchanged_lidar_imu_arrays=scalar_inputs, timings=timings,
        frames=40, scope="same offered recorded sensor values; preprocessing/evidence microbenchmark only; no new native replay or navigation score",
        timing_limitations="sequential pipeline order and file-cache effects; no inferential timing speedup claim")
    NAV.write_json(output / "campaign_result.json", result)
    return result


def shard_plan(protocol, phase, *, seed=None, condition=None, case=None, arm=None):
    """Cells for one process in declared execution order: (warm-up cells, cells)."""
    design = protocol_design(protocol)
    section = protocol[phase]
    selected = any(x is not None for x in (seed, condition, case, arm))
    if design == V1_DESIGN:
        cells = section["cells"]
        if selected:
            cells = [c for c in cells if (case is None or c["case"] == case) and c["seed"] == seed
                     and c["condition"] == condition and (arm is None or c["arm"] == arm)]
        if not cells:
            raise ValueError("no declared cells match this shard")
        return [], cells
    if phase == "smoke":
        if selected:
            raise ValueError("the counterbalanced smoke is one declared shard")
        return section["warmup_cells"], section["cells"]
    if case is not None or None in (seed, condition, arm):
        raise ValueError("counterbalanced shards require --seed, --condition and --arm and keep both routes together")
    matches = [s for s in section["shards"] if (s["seed"], s["condition"], s["arm"]) == (seed, condition, arm)]
    if len(matches) != 1:
        raise ValueError("undeclared counterbalanced development shard")
    return matches[0]["warmup_cells"], [c for c in section["cells"] if c["shard"] == matches[0]["name"]]


def retain_warmup(raw, destination):
    """Archive compact numerical warm-up receipts; raw warm-up images stay in scratch."""
    for name in WARMUP_RETAINED:
        source, target = Path(raw) / name, Path(destination) / name
        if target.exists():
            raise ValueError("warm-up receipt already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256(target) != sha256(source):
            raise ValueError("warm-up receipt changed while copying")


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--write-protocol", type=Path)
    p.add_argument("--design", choices=DESIGNS, default=V1_DESIGN, help="protocol written by --write-protocol")
    p.add_argument("--collect", type=Path)
    p.add_argument("--profile-acquisition", type=Path)
    p.add_argument("--phase", choices=("smoke", "development"))
    for name in ("upstream", "deploy", "checkpoint", "orb-runtime-archive", "lio-runtime-archive", "protocol", "output", "qualified-smoke"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--orb-runtime-sha256")
    p.add_argument("--lio-runtime-sha256")
    p.add_argument("--case", choices=CASES)
    p.add_argument("--seed", type=int, help="validated against the loaded protocol's declared cells")
    p.add_argument("--condition", choices=CONDITIONS)
    p.add_argument("--arm", choices=ARMS)
    a = p.parse_args(argv)
    if a.profile_acquisition:
        if a.output is None:
            p.error("profile requires --output")
        profile_retained_sequence(a.profile_acquisition, a.output)
        return 0
    if a.write_protocol:
        NAV.write_json(a.write_protocol, declared_protocol(a.design))
        return 0
    if a.collect:
        if a.qualified_smoke is None or a.output is None:
            p.error("collection requires --qualified-smoke and --output")
        collect_campaign(a.collect, a.qualified_smoke, a.output)
        return 0
    for key in ("phase", "upstream", "deploy", "checkpoint", "orb_runtime_archive", "lio_runtime_archive", "protocol", "output", "orb_runtime_sha256", "lio_runtime_sha256"):
        if getattr(a, key) is None:
            p.error("--" + key.replace("_", "-") + " is required")
    shard = any(x is not None for x in (a.case, a.seed, a.condition, a.arm))
    if shard and (a.phase != "development" or any(x is None for x in (a.seed, a.condition))):
        p.error("development shards require --seed and --condition; --case optionally narrows the paired routes")
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "campaign_result.json").exists():
        raise ValueError("refuse to overwrite a prior campaign result")
    result = {"schema": SCHEMA, "phase": a.phase, "status": "INCOMPLETE", "episodes": 0, "problems": [],
              "completed_cells": [], "unrun_cells": [], "warmup_cells": [], "warmup_episode_files": [],
              "warmup_episode_sha256": {}, "warmup_summaries": [], "ground_truth_inputs_to_estimator_or_planner": []}
    started = time.monotonic()
    try:
        protocol = json.loads(a.protocol.read_text())
        design = protocol_design(protocol)
        if protocol != declared_protocol(design):
            raise ValueError("exact predeclared protocol required")
        result["design"] = design
        arms = tuple(protocol["arms"])
        warmups, cells = shard_plan(protocol, a.phase, seed=a.seed, condition=a.condition, case=a.case, arm=a.arm)
        sources = source_hashes()
        inputs = {key: sha256(getattr(a, key)) for key in ("deploy", "checkpoint", "orb_runtime_archive", "lio_runtime_archive", "protocol")}
        readiness = {gate_key(p, arm): True for p in PIPELINES for arm in arms}
        if a.phase == "development":
            if a.qualified_smoke is None:
                raise ValueError("complete same-source/input smoke is required")
            readiness = validate_smoke(json.loads(a.qualified_smoke.read_text()), sources=sources, inputs=inputs, design=design)
        work = Path(os.getenv("H34_WORK_DIR", tempfile.mkdtemp(prefix="bhl-native-sensor-controls-")))
        orb = STEREO.unpack_runtime(a.orb_runtime_archive, a.orb_runtime_sha256, work / "schedule-orb")
        lio = HYBRID.unpack_lio(a.lio_runtime_archive, a.lio_runtime_sha256, work / "schedule-lio")
        actor = {"name": "frozen_dr_default_s0", "deploy": str(a.deploy.resolve()), "checkpoint": str(a.checkpoint.resolve())}
        runtimes, executed = {"orb": orb, "lio": lio}, 0
        for cell in warmups:
            # Unscored by construction: reserved seed, separate raw tree under
            # work/ (not archived) and compact receipts under output/warmup/.
            name = f"warmup-{cell['pipeline']}-{cell['case']}-s{cell['seed']}-{cell['condition']}-{cell['arm']}"
            raw = work / "warmup-raw" / name
            row = run_episode(a.upstream.resolve(), actor, runtimes, cell["case"], cell["seed"], raw,
                condition=cell["condition"], arm=cell["arm"], seconds=protocol["warmup"]["seconds"], pipeline=cell["pipeline"],
                execution=dict(design=design, role="warmup", shard=cell["shard"],
                               shard_execution_index=cell["shard_execution_index"], process_episode_index=executed))
            executed += 1
            if row["nonfinite"] or row["native_frames"] < 1:
                raise ValueError("warm-up requires finite state and actual native input/output")
            retain_warmup(raw, a.output / "warmup" / name)
            relative = f"warmup/{name}/episode.json"
            result["warmup_cells"].append(cell)
            result["warmup_episode_files"].append(relative)
            result["warmup_episode_sha256"][relative] = sha256(a.output / relative)
            result["warmup_summaries"].append(warmup_summary(row, cell))
            print(json.dumps(dict(role="warmup", **{k: row[k] for k in ("pipeline", "method", "condition", "case", "seed", "native_frames")})), flush=True)
        rows = []
        for cell in cells:
            if not readiness[gate_key(cell["pipeline"], cell["arm"])]:
                result["unrun_cells"].append(dict(cell, reason="UNRUN_AFTER_NEGATIVE_NOMINAL_ARM_SMOKE"))
                continue
            name = f"{cell['pipeline']}-{cell['case']}-s{cell['seed']}-{cell['condition']}-{cell['arm']}"
            row = run_episode(a.upstream.resolve(), actor, runtimes, cell["case"], cell["seed"], a.output / name,
                condition=cell["condition"], arm=cell["arm"], seconds=protocol[a.phase]["seconds"], pipeline=cell["pipeline"],
                execution=dict(design=design, role=cell.get("role", "smoke" if a.phase == "smoke" else "scored"),
                               shard=cell.get("shard"), shard_execution_index=cell.get("shard_execution_index"),
                               process_episode_index=executed))
            executed += 1
            if row["nonfinite"] or (row["native_frames"] < 1 and (a.phase == "smoke" or not (row["fell"] or row["collision"]))):
                raise ValueError("execution smoke/episode requires finite state and actual native input/output")
            row["episode_file"] = name + "/episode.json"
            rows.append(row)
            result["completed_cells"].append(cell)
            print(json.dumps({k: row[k] for k in ("pipeline", "method", "condition", "case", "seed", "success", "native_tracked_frames", "native_imu_initialized_frames", "fell", "collision")}), flush=True)
        if source_hashes() != sources or any(sha256(getattr(a, k)) != v for k, v in inputs.items()):
            raise ValueError("source or frozen input mutated during the experiment")
        result.update(status="PASS", scientific_status="DEVELOPMENT_ONLY_NO_CONFIRMATION" if a.phase == "development" else "EXECUTION_SMOKE_WITH_EXPLICIT_SCIENTIFIC_ARM_GATES",
            execution_gate=True, source_sha256=sources, input_sha256=inputs, episodes=len(rows),
            episode_files=[r["episode_file"] for r in rows], expected_declared_cells=len(cells),
            episode_sha256={r["episode_file"]: sha256(a.output / r["episode_file"]) for r in rows},
            total_declared_development_cells=len(protocol["development"]["cells"]), warmup_episodes=len(result["warmup_cells"]),
            arm_development_ready=nominal_arm_readiness(rows, arms) if a.phase == "smoke" else readiness,
            arms={arm: {"episodes": sum(r["method"] == arm for r in rows),
                "clean_goals": sum(r["success"] for r in rows if r["method"] == arm),
                "falls": sum(r["fell"] for r in rows if r["method"] == arm),
                "contacts": sum(r["collision"] for r in rows if r["method"] == arm)} for arm in arms},
            aggregate_improvement_verdict="UNAVAILABLE_UNTIL_COMPLETE_STRICT_PAIRED_COLLECTION",
            scope=protocol["scope"])
    except Exception as error:
        result["problems"].append(type(error).__name__ + ": " + str(error))
    result["wall_seconds"] = time.monotonic() - started
    NAV.write_json(a.output / "campaign_result.json", result)
    print(json.dumps({k: result[k] for k in ("schema", "phase", "status", "episodes", "problems")}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
