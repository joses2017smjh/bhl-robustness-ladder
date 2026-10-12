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


def run_episode(upstream, actor, runtime, case, seed, output, *, condition, arm, seconds, pipeline):
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
    NAV.write_json(output/"episode.json",result)
    import shutil
    shutil.rmtree(cache)
    return result

def declared_protocol():
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


def nominal_arm_readiness(rows):
    readiness = {}
    for pipeline in PIPELINES:
        for arm in ARMS:
            candidates = [r for r in rows if r["pipeline"] == pipeline and r["method"] == arm and r["condition"] == "nominal"]
            readiness[gate_key(pipeline, arm)] = bool(len(candidates) == 1 and candidates[0]["native_tracked_frames"] > 0
                and candidates[0]["full_horizon"] and not any(candidates[0][k] for k in ("fell", "collision", "nonfinite")))
    return readiness


def validate_smoke(smoke, *, sources, inputs):
    if (smoke.get("schema") != SCHEMA or smoke.get("phase") != "smoke" or smoke.get("status") != "PASS"
            or smoke.get("execution_gate") is not True or smoke.get("source_sha256") != sources
            or smoke.get("input_sha256") != inputs or smoke.get("completed_cells") != declared_protocol()["smoke"]["cells"]
            or smoke.get("problems") or smoke.get("unrun_cells") or smoke.get("episodes") != 12):
        raise ValueError("complete matching-source/input twelve-cell smoke required")
    readiness = smoke.get("arm_development_ready")
    expected = {gate_key(p, a) for p in PIPELINES for a in ARMS}
    if not isinstance(readiness, dict) or set(readiness) != expected or any(type(x) is not bool for x in readiness.values()):
        raise ValueError("explicit pipeline/arm readiness required")
    return readiness


def paired_summary(rows):
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
        pairs.append(dict(zip(keys, key), clean_goal_difference=int(candidate["success"])-int(legacy["success"]),
            stale_fraction_difference=candidate["stale_navigation_fraction"]-legacy["stale_navigation_fraction"] if comparable else None,
            stale_comparison_available=comparable,
            legacy_stale_fraction=legacy["stale_navigation_fraction"], candidate_stale_fraction=candidate["stale_navigation_fraction"],
            legacy_clean_goal=legacy["success"], candidate_clean_goal=candidate["success"],
            both_full_horizon=legacy["full_horizon"] and candidate["full_horizon"]))
    stale_pairs = [p for p in pairs if p["stale_comparison_available"]]
    return dict(complete_pairs=len(pairs), stale_comparable_pairs=len(stale_pairs), incomplete_pairs=incomplete, pairs=pairs,
        mean_paired_stale_fraction_difference=float(np.mean([p["stale_fraction_difference"] for p in stale_pairs])) if stale_pairs else None,
        clean_goal_count_difference=sum(p["clean_goal_difference"] for p in pairs) if pairs else None,
        negative_difference_means="candidate fewer stale stops; positive clean-goal difference means candidate more goals",
        uncertainty="two seed clusters, descriptive paired outcomes only; no population confidence interval")


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


def aggregate_groups(rows):
    groups = {}
    for pipeline in PIPELINES:
        for arm in ARMS:
            values = [r for r in rows if r["method"] == arm and r["pipeline"] == pipeline]
            available = [r["stale_navigation_fraction"] for r in values if r["stale_navigation_fraction"] is not None]
            groups[gate_key(pipeline, arm)] = dict(episodes=len(values), clean_goals=sum(r["success"] for r in values),
                falls=sum(r["fell"] for r in values), contacts=sum(r["collision"] for r in values),
                mean_stale_fraction=float(np.mean(available)) if available else None)
    return groups


def collect_campaign(root, smoke_path, output):
    """Strict exact-cohort collection; skipped gates remain unrun, never zero."""
    root = Path(root).resolve()
    smoke = json.loads(Path(smoke_path).read_text())
    ready = validate_smoke(smoke, sources=smoke.get("source_sha256"), inputs=smoke.get("input_sha256"))
    expected = {tuple(c[k] for k in ("case", "seed", "condition", "arm", "pipeline")) for c in declared_protocol()["development"]["cells"]}
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
            key = tuple(cell[k] for k in ("case", "seed", "condition", "arm", "pipeline"))
            if key not in expected or key in observed or key in unrun or not ready[gate_key(cell["pipeline"], cell["arm"])]:
                raise ValueError("undeclared, duplicate or gated development cell")
            episode_path = (path.parent / relative).resolve()
            if not episode_path.is_relative_to(path.parent.resolve()):
                raise ValueError("episode file leaves its result directory")
            if sha256(episode_path) != result["episode_sha256"].get(relative):
                raise ValueError("retained episode differs from result hash")
            row = json.loads(episode_path.read_text())
            if (row.get("case"), row.get("seed"), row.get("condition"), row.get("method"), row.get("pipeline")) != key:
                raise ValueError("episode identities differ from declared cell")
            for name, field in (("inference_sensors.npz", "inference_sensors_sha256"),
                                ("evaluator_truth.npz", "evaluator_truth_sha256"),
                                ("native/stream_receipt.json", "stream_receipt_sha256"),
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
            full = row["elapsed_simulation_s"] >= 40. - 1e-6
            success = reached and full and not (contact or fell or row["nonfinite"])
            if any(bool(row[k]) != actual for k, actual in (("goal_reached_evaluator", reached), ("collision", contact),
                    ("fell", fell), ("full_horizon", full), ("success", success))):
                raise ValueError("reported outcome differs from retained evaluator trace")
            nav = [r for r in trace if r["decision_timestamp_s"] >= 10.-1e-9]
            stale = sum(r["reason"] == "native_stale_stop" for r in nav)
            fraction = stale/len(nav) if nav else None
            if row["navigation_decisions"] != len(nav) or row["stale_navigation_decisions"] != stale or row["stale_navigation_fraction"] != fraction:
                raise ValueError("reported stale stops differ from decision trace")
            rows.append(row)
            observed.add(key)
        for cell in result["unrun_cells"]:
            key = tuple(cell[k] for k in ("case", "seed", "condition", "arm", "pipeline"))
            if (key not in expected or key in observed or key in unrun or ready[gate_key(cell["pipeline"], cell["arm"])]
                    or cell["reason"] != "UNRUN_AFTER_NEGATIVE_NOMINAL_ARM_SMOKE"):
                raise ValueError("invalid unrun cell or gate")
            unrun.add(key)
        lineage.append({"path": str(path), "sha256": sha256(path)})
    missing = expected - observed - unrun
    if missing:
        raise ValueError(f"incomplete development collection: {len(missing)} undeclared-missing outcomes")
    groups = aggregate_groups(rows)
    result = dict(schema=SCHEMA + "-collection", status="PASS", declared_cells=48, measured_cells=len(observed),
        unrun_after_nominal_gate=len(unrun), missing_cells=[], arm_development_ready=ready, by_pipeline_arm=groups,
        paired=paired_summary(rows), source_results=lineage, qualified_smoke_sha256=sha256(smoke_path),
        scientific_status="COMPLETE_BOUNDED_DEVELOPMENT_NO_CONFIRMATION",
        scope=declared_protocol()["scope"])
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


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--write-protocol", type=Path)
    p.add_argument("--collect", type=Path)
    p.add_argument("--profile-acquisition", type=Path)
    p.add_argument("--phase", choices=("smoke", "development"))
    for name in ("upstream", "deploy", "checkpoint", "orb-runtime-archive", "lio-runtime-archive", "protocol", "output", "qualified-smoke"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--orb-runtime-sha256")
    p.add_argument("--lio-runtime-sha256")
    p.add_argument("--case", choices=CASES)
    p.add_argument("--seed", type=int, choices=DEVELOPMENT_SEEDS)
    p.add_argument("--condition", choices=CONDITIONS)
    p.add_argument("--arm", choices=ARMS)
    a = p.parse_args(argv)
    if a.profile_acquisition:
        if a.output is None:
            p.error("profile requires --output")
        profile_retained_sequence(a.profile_acquisition, a.output)
        return 0
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
    shard = any(x is not None for x in (a.case, a.seed, a.condition, a.arm))
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
        readiness = {gate_key(p, arm): True for p in PIPELINES for arm in ARMS}
        if a.phase == "development":
            if a.qualified_smoke is None:
                raise ValueError("complete same-source/input smoke is required")
            readiness = validate_smoke(json.loads(a.qualified_smoke.read_text()), sources=sources, inputs=inputs)
        work = Path(os.getenv("H34_WORK_DIR", tempfile.mkdtemp(prefix="bhl-native-sensor-controls-")))
        orb = STEREO.unpack_runtime(a.orb_runtime_archive, a.orb_runtime_sha256, work / "schedule-orb")
        lio = HYBRID.unpack_lio(a.lio_runtime_archive, a.lio_runtime_sha256, work / "schedule-lio")
        actor = {"name": "frozen_dr_default_s0", "deploy": str(a.deploy.resolve()), "checkpoint": str(a.checkpoint.resolve())}
        cells = protocol[a.phase]["cells"]
        if shard:
            cells = [c for c in cells if (a.case is None or c["case"] == a.case) and c["seed"] == a.seed and c["condition"] == a.condition and (a.arm is None or c["arm"] == a.arm)]
        rows = []
        for cell in cells:
            if not readiness[gate_key(cell["pipeline"], cell["arm"])]:
                result["unrun_cells"].append(dict(cell, reason="UNRUN_AFTER_NEGATIVE_NOMINAL_ARM_SMOKE"))
                continue
            name = f"{cell['pipeline']}-{cell['case']}-s{cell['seed']}-{cell['condition']}-{cell['arm']}"
            row = run_episode(a.upstream.resolve(), actor, {"orb": orb, "lio": lio}, cell["case"], cell["seed"], a.output / name,
                condition=cell["condition"], arm=cell["arm"], seconds=protocol[a.phase]["seconds"], pipeline=cell["pipeline"])
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
            total_declared_development_cells=48, arm_development_ready=nominal_arm_readiness(rows) if a.phase == "smoke" else readiness,
            arms={arm: {"episodes": sum(r["method"] == arm for r in rows),
                "clean_goals": sum(r["success"] for r in rows if r["method"] == arm),
                "falls": sum(r["fell"] for r in rows if r["method"] == arm),
                "contacts": sum(r["collision"] for r in rows if r["method"] == arm)} for arm in ARMS},
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
