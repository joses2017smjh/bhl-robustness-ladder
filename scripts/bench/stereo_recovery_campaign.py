#!/usr/bin/env python3
"""Predeclared stereo recovery comparison: permanent stop versus verified relocalization.

Native ORB-SLAM3 stereo navigates the processing-schedule ``bounded_latest``
pipeline with the Atlas-telemetry wrapper. Two recovery policies are paired by
seed, condition and route:

* ``permanent_stop`` (control): the unchanged v1 behaviour. The legacy tracker
  watches the tracked-map-point map id; any coordinate invalidation after the
  first accepted epoch stops navigation for the rest of the episode.
* ``resume_verified_relocalization``: Atlas-aware coordinate epochs (a transient
  loss in the same map incarnation is not an invalidation; id changes, same-id
  resets and big changes still are) plus a gate that resumes only after two
  tracked frames agree with the last accepted pose dead-reckoned through the
  applied commands; any failure stops permanently, as before.

Both arms run the same native runtime, sensors, schedule, actor, routes,
registration, raw LiDAR brake and 0.35 s capture-age gate. Each computes the
other policy's decision in shadow for analysis only. The completed v1 source,
cohorts and results remain untouched.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from itertools import product
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
import native_sensor_controls_campaign as BASE
import processing_schedule_campaign as SCHED
from bhl_robust.research.native_orb import TRACKING_STATES, sha256
from bhl_robust.research.navigation_sensor_controls import condition_spec, tracking_events
from bhl_robust.research.pose_metrics import transform
from bhl_robust.research.processing_schedule import service_timing
from bhl_robust.research.stereo_relocalization import (
    AtlasCoordinateEpochTracker, ResumeBound, VerifiedRelocalizationGate, classify_losses,
)

SCHEMA = "bhl-stereo-recovery-v1"
PIPELINE = "bounded_latest"
ESTIMATOR = "stereo"
ARMS = ("permanent_stop", "resume_verified_relocalization")
CONDITIONS = ("nominal", "camera_10hz", "causal_40ms_exposure")
CASES = ("straight", "occluders")
SMOKE_SEED, DEVELOPMENT_SEEDS = 759000, (760000, 760001, 760002, 760003)
# Reset seeds already claimed by other cohorts; ours must stay disjoint from all.
EXCLUDED_SEEDS = {
    "processing-schedule-order-v1-20261014 (branch order-confirmation-20261014, commit f86d753; lead reservation)":
        [749000, 749900, *range(751000, 751008)],
    "processing-schedule-v1-20261012 (frozen 152ccf4)": [749000, 750000, 750001],
    "native-sensor-controls-v2 (frozen f881a0b)": [720000, 730000, 730001],
}
SMOKE_SECONDS, DEVELOPMENT_SECONDS = 18., 40.
NAVIGATION_BEGINS_S = 10.
ATLAS_TELEMETRY = "bhl-orb-atlas-telemetry-v1"
BOUND = ResumeBound()
initialization_command = SCHED.initialization_command


class RecoveryNativeClient(BASE.MatchedNativeClient):
    """Unchanged v1 stereo client plus Atlas epochs, the resume gate and raw poses.

    The base class computes the legacy permanent-stop packet exactly as in v1.
    The resume arm then replaces only acceptance, pose, state and reset status
    with the gate decision; the control keeps the legacy packet and records the
    gate decision in shadow. Every unique camera packet keeps the raw native
    pose (``raw_T_W_I``) whether or not navigation accepted it.
    """

    def __init__(self, runtime, t_imu_lidar, directory, *, max_frames, policy, known_start, policy_dt):
        if policy not in ARMS:
            raise ValueError("unknown predeclared recovery policy")
        super().__init__(dict(runtime, arm=ESTIMATOR), t_imu_lidar, directory, max_frames=max_frames)
        self.policy = policy
        self.known_start = transform(known_start)
        self.t_body_imu = transform(runtime["t_body_imu"])
        self.atlas_epochs = AtlasCoordinateEpochTracker()
        self.gate = VerifiedRelocalizationGate(BOUND, step_s=policy_dt)
        self.gate_registration = None
        self.commands = []
        self.atlas_frames = 0

    def __enter__(self):
        super().__enter__()
        if self.orb.ready.get("atlas_telemetry") != ATLAS_TELEMETRY:
            self.__exit__(None, None, None)
            raise ValueError("native runtime lacks the read-only Atlas telemetry wrapper")
        return self

    def record_command(self, decision_s, command):
        """Applied command log: only the controller's own outputs, never evaluator state."""
        row = (float(decision_s), *map(float, command))
        if len(row) != 4 or not all(math.isfinite(v) for v in row) or (self.commands and row[0] <= self.commands[-1][0]):
            raise ValueError("finite strictly increasing applied commands required")
        self.commands.append(row)

    def body_pose(self, t_w_i):
        registration = self.gate_registration
        if registration is None:
            registration = self.known_start @ self.t_body_imu @ np.linalg.inv(transform(t_w_i))
        return registration, registration @ transform(t_w_i) @ np.linalg.inv(self.t_body_imu)

    def track(self, scan, imu):
        packet = super().track(scan, imu)
        if not packet["new_native_camera_frame"]:
            return packet
        raw = self.latest_orb
        self.atlas_frames += raw.get("atlas") is not None
        epoch = self.atlas_epochs.observe(raw)
        raw_pose = STEREO.native_camera_to_imu(raw["T_W_C"], self.runtime["true_t_imu_lidar"]) if raw["tracked"] else None
        registration = body = None
        if raw_pose is not None:
            registration, body = self.body_pose(raw_pose)
        decision = self.gate.observe(timestamp_s=raw["timestamp_s"], tracked=bool(raw["tracked"]), body_pose=body,
                                     epoch=epoch, commands=self.commands)
        if decision["gate_reason"] == "REGISTERED_INITIAL_EPOCH":
            self.gate_registration = registration
        legacy = {key: packet[key] for key in ("tracked", "state", "map_reset_id")}
        packet.update(raw_T_W_I=raw_pose.tolist() if raw_pose is not None else None,
                      native_ORB_atlas=raw.get("atlas"), atlas_epoch=epoch, relocalization_gate=decision,
                      legacy_permanent_stop_decision=legacy, recovery_policy=self.policy)
        if self.policy == "resume_verified_relocalization":
            accepted = decision["accepted"]
            state = ("MAP_CHANGED_STOP" if decision["gate_state"] == "PERMANENT_STOP" else
                     "RELOCALIZATION_CONFIRMING_STOP" if decision["gate_state"] == "CONFIRMING" else
                     TRACKING_STATES[raw["tracking_state"]])
            packet.update(tracked=bool(accepted), T_W_I=raw_pose.tolist() if accepted else None,
                          state=state, map_reset_id=decision["map_reset_id"])
        return packet

    def __exit__(self, kind, value, traceback):
        try:
            return super().__exit__(kind, value, traceback)
        finally:
            camera = [r for r in self.records if r.get("new_native_camera_frame")]
            NAV.write_json(self.directory / "recovery_receipt.json", {
                "schema": SCHEMA + "-native-recovery", "policy": self.policy, "estimator": ESTIMATOR,
                "atlas_telemetry_frames": self.atlas_frames, "native_camera_frames": len(camera),
                "atlas_epochs": self.atlas_epochs.receipt(), "gate": self.gate.receipt(),
                "gate_is_shadow_only": self.policy == "permanent_stop",
                "legacy_epochs": self.coordinate_epochs.receipt(),
                "applied_commands": [list(c) for c in self.commands],
                "ground_truth_inputs": [], "evaluator_inputs": []})


def stopped_navigation(traces, policy_dt):
    """Zero applied command at navigation decisions, excluding estimated goal stops."""
    nav = [t for t in traces if t["decision_timestamp_s"] >= NAVIGATION_BEGINS_S - 1e-9]
    stopped = [t for t in nav if not any(t["command"]) and t["reason"] != "estimated_goal_stop"]
    return {"navigation_decisions": len(nav), "stopped_navigation_decisions": len(stopped),
            "stopped_navigation_time_s": len(stopped) * policy_dt,
            "stopped_navigation_fraction": len(stopped) / len(nav) if nav else None,
            "stopped_scope": "decisions at or after 10 s with exactly zero applied command; estimated_goal_stop excluded; held nonzero latency commands count as moving"}


def accepted_pose_errors(native_records, truth_imu, registration, t_body_imu):
    """Evaluator-only: registered accepted-pose position error at each capture time."""
    if registration is None:
        return {"accepted_pose_errors_m": [], "resumed_pose_errors_m": []}
    times = np.asarray([x[0] for x in truth_imu])
    errors, resumed = [], []
    for record in native_records:
        if not record.get("new_native_camera_frame") or not record["tracked"] or not len(times):
            continue
        estimate = transform(registration) @ transform(record["T_W_I"]) @ np.linalg.inv(transform(t_body_imu))
        index = int(np.argmin(np.abs(times - record["timestamp_s"])))
        truth = truth_imu[index][1] @ np.linalg.inv(transform(t_body_imu))
        error = float(np.linalg.norm(estimate[:2, 3] - truth[:2, 3]))
        errors.append(error)
        if (record.get("recovery_policy") == "resume_verified_relocalization"
                and (record.get("relocalization_gate") or {}).get("gate_reason") == "VERIFIED_RELOCALIZATION_RESUME"):
            resumed.append({"timestamp_s": record["timestamp_s"], "position_error_m": error})
    return {"accepted_pose_errors_m": errors, "resumed_pose_errors_m": resumed,
            "accepted_pose_error_p95_m": float(np.quantile(errors, .95)) if errors else None,
            "accepted_pose_error_max_m": max(errors) if errors else None,
            "truth_matching": "nearest policy-step evaluator sample (0.04 s); evaluator-only, never fed back"}


def run_episode(upstream, actor, runtime, case, seed, output, *, condition, arm, seconds):
    import mujoco
    from bhl_robust.eval.multi_robot import _WORLDS
    from team_airlock import ContactRunner
    from team_airlock import CpuPolicy
    from omegaconf import OmegaConf
    from bhl_robust.eval.gait_clock import make_controller
    pipeline = PIPELINE
    cfg=OmegaConf.load(actor["deploy"]);cfg.policy_checkpoint_path=actor["checkpoint"]
    controller=make_controller(cfg);controller.policy=CpuPolicy(actor["checkpoint"])
    output.mkdir(parents=True,exist_ok=False)
    _WORLDS["native_navigation"]=STEREO.textured_world(case,seed,output/"visual_assets")
    cache=Path(tempfile.mkdtemp(prefix="bhl-native-nav-assets-"))
    model,slots=STEREO.build_stereo_model(upstream,cache,1,[ESTIMATOR],variant="biped",world="native_navigation")
    (output/"world.xml").write_text(_WORLDS["native_navigation"])
    slot=slots[0];runner=ContactRunner(model,slots,[cfg],[controller]);runner.reset(np.random.default_rng(seed))
    owners=np.array([0 if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or "").startswith(slot.prefix) else -1 for g in range(model.ngeom)])
    runner.configure_contacts(owners);sensors=SCHED.ScheduledSensors(runner,slot,owners,output=output/"sensors",condition=condition,pipeline=pipeline)
    known_start=np.eye(4);known_start[:3,3]=model.qpos0[slot.qpos_adr:slot.qpos_adr+3]
    runtime=dict(runtime, condition=condition, true_t_imu_lidar=sensors.t_imu_lidar.copy(), t_body_imu=sensors.t_body_imu.copy())
    policy_dt=float(cfg.policy_dt)
    registration=None;delivery=NAV.DelayedNativeState();waypoint=0;command=np.zeros(3)
    traces=[];native_records=[];client_latencies=[];additional_sensor_latencies=[];dropped_scans=0;goals=False;fell=collision=nonfinite=False
    latency_steps=0;goal_time=None;truth_imu=[];last_consumed_s=None
    initial_z=float(runner.d.qpos[slot.qpos_adr+2])
    maximum_steps=int(round(seconds/policy_dt))
    with ExitStack() as resources:
        resources.callback(sensors.close)
        client = resources.enter_context(RecoveryNativeClient(runtime,np.linalg.inv(sensors.t_body_imu)@sensors.reported_t_body_lidar,output/"native",
            max_frames=int(seconds*20)+10,policy=arm,known_start=known_start,policy_dt=policy_dt))
        for step in range(maximum_steps):
            now=float(runner.d.time)
            arrived=delivery.advance(step)
            latest=delivery.available
            if arrived and latest["tracked"] and registration is None:
                registration=known_start@sensors.t_body_imu@np.linalg.inv(transform(latest["T_W_I"]))
                if arm=="resume_verified_relocalization" and not np.allclose(registration, client.gate_registration, atol=1e-12):
                    raise ValueError("navigation registration differs from the gate registration")
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
                    computed["schedule_timing"]["preprocessing_scope"] = "selected PNG IPC only"
                    computed["client_wall_seconds"] = elapsed
                    computed["full_pipeline_wall_seconds"] = elapsed + extra
                    body_points=scan["planner_points_body"]
                    delivery.stage(computed,body_points,step,elapsed+extra,policy_dt)
                    reason="measured_native_latency_hold";latency_steps+=1
                else:
                    command,waypoint,reason,_=NAV.navigation_command(latest,registration,sensors.t_body_imu,NAV.ROUTES[case],waypoint,now,delivery.raw)
                    last_consumed_s=now if latest is not None else None
            # Loss/reset can affect commands only after the response arrives.
            if latest is None or not latest["tracked"] or latest.get("map_reset_id")!=0: command=np.zeros(3)
            if latest is not None and now-latest["timestamp_s"]>.35:command=np.zeros(3);reason="native_stale_stop"
            if now < NAVIGATION_BEGINS_S:
                command=initialization_command(now);reason="predeclared_simulation_initialization_excitation"
            client.record_command(now, command)
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
                "native_tracked":latest["tracked"] if latest else False,"native_state":latest["state"] if latest else None,
                "evaluator_xy":position[:2].tolist(),"evaluator_height_m":float(position[2]),
                "evaluator_goal_distance_m":distance,"evaluator_tilt_rad":tilt,"evaluator_contact":bool(runner.hit_wall)})
            if fell or collision:break
    sensors.save(output/"inference_sensors.npz")
    sensors.close()
    np.savez_compressed(output/"evaluator_truth.npz",timestamp_s=np.asarray([x[0] for x in truth_imu]),T_W_I=np.asarray([x[1] for x in truth_imu]))
    actual_seconds=float(runner.d.time)
    camera_records=[r for r in native_records if r.get("new_native_camera_frame")]
    result={"status":"PASS","scientific_status":"MEASURED_DEVELOPMENT_EPISODE","method":ESTIMATOR,"arm":arm,"pipeline":pipeline,
        "condition":condition,"case":case,"seed":seed,
        "success":bool(goals and not(fell or collision or nonfinite) and actual_seconds>=seconds-1e-6),
        "goal_reached_evaluator":goals,"first_goal_time_s":goal_time,"fell":bool(fell),"collision":bool(collision),"nonfinite":bool(nonfinite),
        "elapsed_simulation_s":actual_seconds,"full_horizon":actual_seconds>=seconds-1e-6,"horizon_s":seconds,"controller_goal_stop":"estimated_pose_only",
        "native_frames":len(native_records),"native_tracked_frames":sum(r["tracked"] for r in native_records),
        "native_states":{state:sum(r["state"]==state for r in native_records) for state in sorted({r["state"] for r in native_records})},
        "native_camera_frames":len(camera_records),"native_raw_ok_frames":sum(r["native_ORB_tracking_state"]==2 for r in camera_records),
        "atlas_telemetry_frames":sum(r.get("native_ORB_atlas") is not None for r in camera_records),
        "client_wall_p95_ms":float(np.quantile(client_latencies,.95)*1000) if client_latencies else None,
        "additional_sensor_compute_total_s":float(sum(additional_sensor_latencies)),
        "compute_latency_policy_steps":latency_steps,"latency_quantization_s":policy_dt,"dropped_scans":dropped_scans,
        "response_deliveries":delivery.history,"responses_still_pending_at_horizon":int(delivery.pending is not None),
        "initial_registration":registration.tolist() if registration is not None else None,"known_start_pose":known_start.tolist(),
        "T_body_imu":sensors.t_body_imu.tolist(),"T_body_lidar":sensors.t_body_lidar.tolist(),"route_external":NAV.ROUTES[case],
        "evaluator_initial_height_m":initial_z,
        "trace":traces,"inference_sensors_sha256":sha256(output/"inference_sensors.npz"),"evaluator_truth_sha256":sha256(output/"evaluator_truth.npz"),
        "stream_receipt_sha256":sha256(output/"native/stream_receipt.json"),"native_frames_sha256":sha256(output/"native/native_frames.jsonl"),
        "recovery_receipt_sha256":sha256(output/"native/recovery_receipt.json"),"ground_truth_inputs_to_estimator_or_planner":[],
        "scope":"paired recovery-policy simulation development; bounded-latest native stereo; shared raw LiDAR brake; no hardware or independent confirmation",
        "initialization_excitation":{"stationary_until_s":3.,"excitation_until_s":9.,"navigation_begins_s":NAVIGATION_BEGINS_S,"max_vx_command_m_s":.1,"max_yaw_command_rad_s":.3,"native_threshold_changes":False,"known_start_registration_limitation":"first accepted native pose is registered to nominal known station; initialization-motion error is not evaluator-corrected"},
        "full_pipeline_p95_ms":float(np.quantile([r["wall_seconds"] for r in delivery.history],.95)*1000) if delivery.history else None,
        "measured_brake_false_clear_packets":sum(r["false_clear"] for r in sensors.brake_measurements),
        "measured_brake_unknown_packets":sum(r["unknown_stop"] for r in sensors.brake_measurements),
        "measured_brake_obstacle_packets":sum(r["measured_obstacle_present"] for r in sensors.brake_measurements)}
    result["unique_estimator_updates"]=len(camera_records)
    result["accepted_unique_estimator_updates"]=sum(r["tracked"] for r in camera_records)
    result.update(tracking_events(camera_records, actual_seconds))
    result["recovery"]=classify_losses(camera_records, actual_seconds)
    receipt=json.loads((output/"native/recovery_receipt.json").read_text())
    result["verified_resumes"]=receipt["gate"]["verified_resumes"] if arm=="resume_verified_relocalization" else 0
    result["shadow_verified_resumes"]=receipt["gate"]["verified_resumes"]
    result["permanent_stop"]=(receipt["gate"]["permanent_stop"] if arm=="resume_verified_relocalization"
        else next(({"timestamp_s":r["timestamp_s"],"reason":"LEGACY_COORDINATE_INVALIDATION","invalidation_reasons":r["native_coordinate_invalidation_reasons"]}
                   for r in camera_records if r["state"]=="MAP_CHANGED_STOP"), None))
    result["atlas_coordinate_invalidations"]=receipt["atlas_epochs"]["coordinate_invalidations"]
    result.update(stopped_navigation(traces, policy_dt))
    result.update(accepted_pose_errors(native_records, truth_imu, registration, sensors.t_body_imu))
    nav=[t for t in traces if t["decision_timestamp_s"]>=NAVIGATION_BEGINS_S-1e-9]
    ages=[t["decision_timestamp_s"]-t["native_timestamp_s"] for t in nav if t["native_timestamp_s"] is not None]
    result["stale_navigation_decisions"]=sum(t["reason"]=="native_stale_stop" for t in nav)
    result["stale_navigation_fraction"]=result["stale_navigation_decisions"]/len(nav) if nav else None
    result["navigation_pose_age_p95_ms"]=float(np.quantile(ages,.95)*1000) if ages else None
    result["processing_schedule_sha256"]=sha256(output/"sensors/processing_schedule.json")
    result["schedule_timing"]=[r["schedule_timing"] for r in native_records]
    result["acquisition_schedule"]=sensors.schedule_receipt
    SCHED.validate_schedule_evidence(result, sensors.schedule_receipt)
    result["latency_semantics"]="same DelayedNativeState quantization and arrival-step hold as processing-schedule v1 bounded_latest; selected PNG IPC+native service"
    NAV.write_json(output/"episode.json",result)
    import shutil
    shutil.rmtree(cache)
    return result


def arm_order(seed_index, condition_index, case_index):
    """Counterbalanced within-pair order: each arm runs first in 12 of 24 pairs."""
    return ARMS if (seed_index + condition_index + case_index) % 2 == 0 else ARMS[::-1]


def declared_cells(seeds, cases, conditions=CONDITIONS):
    cells = []
    for seed_index, seed in enumerate(seeds):
        for condition_index, condition in enumerate(conditions):
            for case_index, case in enumerate(cases):
                for arm in arm_order(seed_index, condition_index, case_index):
                    cells.append(dict(case=case, seed=seed, condition=condition, arm=arm))
    return cells


def declared_protocol():
    b = BOUND
    return dict(
        schema=SCHEMA + "-protocol", estimator=ESTIMATOR, pipeline=PIPELINE, arms=list(ARMS),
        conditions={c: condition_spec(c) for c in CONDITIONS}, routes=list(CASES),
        smoke=dict(seconds=SMOKE_SECONDS, cells=declared_cells([SMOKE_SEED], ["straight"])),
        development=dict(seconds=DEVELOPMENT_SECONDS, cells=declared_cells(DEVELOPMENT_SEEDS, CASES)),
        seeds={"smoke": SMOKE_SEED, "development": list(DEVELOPMENT_SEEDS),
               "excluded_reserved_seeds": EXCLUDED_SEEDS,
               "novelty": "disjoint from every excluded seed above; smoke and development seeds are disjoint"},
        pairing="same seed/condition/route; one pair per shard in one GPU allocation; within-pair arm order counterbalanced by (seed index + condition index + route index) mod 2, 12 first positions per arm",
        initialization=BASE.declared_protocol()["initialization"],
        unchanged=("frozen actor, routes, native ORB settings/thresholds, bounded_latest schedule, PNG transport, 0.35 s capture-age stop, "
                   "known-start registration, raw LiDAR brake, delivery quantization, 40 s horizon, 0.30 m evaluator goal radius"),
        native_runtime="ORB-SLAM3 4452a3c wrapper relinked with read-only Atlas telemetry; libORB_SLAM3.so byte-identical to v1; one runtime for both arms",
        arm_definitions={
            "permanent_stop": "control = v1 behaviour: legacy tracked-map-point epochs; after the first accepted epoch any invalidation (including native_map_identity_unavailable during RECENTLY_LOST) stops for the rest of the episode",
            "resume_verified_relocalization": "Atlas epochs + verified relocalization gate; transient loss in the same Atlas map incarnation is not an invalidation; map id change, same-id active-map reset, big change, map count change, return to initialization, map-point conflict, inconsistent snapshot or missing telemetry stop permanently"},
        coordinate_epoch_rule="AtlasCoordinateEpochTracker: reference = Atlas current map id, init keyframe id, big-change index, map count, inertial stage at the first tracked frame on an initialized map",
        resume_rule=dict(
            reference="last accepted registered body pose P0 at its capture time t0",
            prediction="P0 dead-reckoned (first-order unicycle, 0.04 s policy steps) through the commands actually applied at decision times in [t0, t)",
            translation_bound_m=f"{b.base_translation_m} + {b.translation_per_commanded_path} * commanded path length",
            yaw_bound_rad=f"{b.base_yaw_rad:.6f} + {b.yaw_per_commanded_rotation} * commanded absolute rotation",
            vertical_bound_m=b.max_vertical_m, tilt_change_bound_rad=round(b.max_tilt_change_rad, 6),
            confirmation_frames=b.confirmation_frames,
            on_failure="one inconsistent tracked frame stops permanently; an untracked frame during confirmation restarts confirmation",
            calibration=("fixed before this campaign from processing-schedule v1 bounded episodes (seeds 750000/750001; evaluator truth versus applied commands, "
                         "61,944 windows 0.04-5 s): joint bound exceeded in 0.29% of windows, 0.008% of windows <= 3 s, 0% of zero-command windows; "
                         "body height/tilt changes <= 2.2 cm / 6.7 deg"),
            parameters=asdict(b)),
        hypotheses={
            "H1_primary": "resume_verified_relocalization completes more clean goals than permanent_stop",
            "H2_secondary": "resume_verified_relocalization reduces stopped navigation time",
            "H3_safety": "resume_verified_relocalization does not increase falls or collisions",
            "H4_mechanism_descriptive": "same-map relocalization losses resume and resumed poses agree with evaluator truth within the bound"},
        primary_outcome="clean goal (success): evaluator goal within 0.30 m, full 40 s horizon, no fall/collision/nonfinite state",
        primary_analysis="exact two-sided McNemar test on the 24 pairs: scipy.stats.binomtest(b, b+c, 0.5), b = resume-only goals, c = control-only goals; alpha 0.05; supported iff p < 0.05 and b > c (with c = 0, b >= 6 is required)",
        effect_estimate="paired difference in clean-goal proportion with a seed-cluster percentile bootstrap by exact enumeration of all 4^4 = 256 seed resamples (6 pairs per seed cluster); Wilson 95% intervals per arm",
        secondary_outcomes=["stopped navigation time (zero applied command at decisions >= 10 s, estimated goal stops excluded): mean paired difference, seed-cluster bootstrap interval, exact sign test",
                            "navigation recovery time per loss (loss start to next accepted pose; unresumed losses right-censored at the horizon) and raw native relocalization time",
                            "falls and collisions per arm",
                            "loss mechanisms from Atlas telemetry (same-map relocalization, active-map reset, new Atlas map, other invalidation, no native recovery)",
                            "evaluator-only position error of accepted and resumed poses"],
        safety_rule="H1 support counts as an improvement only if resume falls <= control falls and resume collisions <= control collisions",
        negative_result=("NEGATIVE if p >= 0.05 or b <= c (no detectable goal improvement); NEGATIVE_SAFETY if resume has more falls or collisions than control; "
                         "mechanism-negative if the gate rejects most same-map relocalizations or resumed poses exceed the bound against truth"),
        equivocal_result="UNAVAILABLE if any declared development cell is unrun (failed smoke gate or execution stop); no partial verdict",
        stopping_rules=["no interim efficacy, futility or safety analyses; all declared cells are analysed",
                        "the dispatcher stops at the first execution failure without retries; remaining cells stay UNRUN and the cohort is incomplete",
                        "a failed 6-cell smoke execution gate prevents all development; a failed per-arm nominal readiness gate leaves that arm's cells UNRUN"],
        exclusion_rules="none after outcomes: falls, collisions and early terminations remain failed episodes; no reruns, replacement seeds or retuning",
        smoke_execution_gate="all 6 smoke cells execute with finite physics, native I/O, verified evidence and Atlas telemetry on every native camera frame",
        nominal_readiness_gate="per arm: nominal 18 s smoke reaches full horizon with an accepted native pose, Atlas telemetry on every frame, zero falls/contacts/nonfinite",
        deferred_arms={"stereo_inertial": "not run: adds ~1 GPU-hour, scored 0/40 in native-sensor-controls-v2 and never initialized in the earlier stereo-IMU smoke; would also need IMU-refinement epoch rules"},
        scope="four-seed bounded simulation development; ideal simulated sensing; no hardware, population or confirmation claims",
        no_retuning_after_outcomes=True, source_change_requires_fresh_smoke=True)


def source_hashes():
    return BASE.source_hashes()


def nominal_arm_readiness(rows):
    ready = {}
    for arm in ARMS:
        candidates = [r for r in rows if r["arm"] == arm and r["condition"] == "nominal"]
        ready[arm] = bool(len(candidates) == 1 and candidates[0]["native_tracked_frames"] > 0 and candidates[0]["full_horizon"]
                          and candidates[0]["atlas_telemetry_frames"] == candidates[0]["native_camera_frames"] > 0
                          and not any(candidates[0][k] for k in ("fell", "collision", "nonfinite")))
    return ready


def validate_smoke(smoke, *, sources, inputs):
    expected = declared_protocol()["smoke"]["cells"]
    if (smoke.get("schema") != SCHEMA or smoke.get("phase") != "smoke" or smoke.get("status") != "PASS"
            or smoke.get("execution_gate") is not True or smoke.get("source_sha256") != sources
            or smoke.get("input_sha256") != inputs or smoke.get("completed_cells") != expected
            or smoke.get("problems") or smoke.get("unrun_cells") or smoke.get("episodes") != len(expected)):
        raise ValueError("complete matching-source/input six-cell smoke required")
    readiness = smoke.get("arm_development_ready")
    if not isinstance(readiness, dict) or set(readiness) != set(ARMS) or any(type(x) is not bool for x in readiness.values()):
        raise ValueError("explicit per-arm readiness required")
    return readiness


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [0. if successes == 0 else max(0., centre - half), 1. if successes == n else min(1., centre + half)]


def seed_cluster_bootstrap(pairs, field):
    """Exact enumeration of all seed-cluster resamples (with replacement)."""
    seeds = sorted({p["seed"] for p in pairs})
    clusters = [float(np.mean([p[field] for p in pairs if p["seed"] == s])) for s in seeds]
    resamples = [float(np.mean([clusters[i] for i in choice])) for choice in product(range(len(clusters)), repeat=len(clusters))]
    low, high = np.quantile(resamples, [.025, .975])
    return {"estimate": float(np.mean(clusters)), "seed_cluster_means": clusters,
            "percentile_interval_95": [float(low), float(high)], "resamples": len(resamples)}


def analyze(rows):
    """Predeclared paired analysis; frozen with the source before any outcome."""
    from scipy.stats import binomtest
    keys = ("case", "seed", "condition")
    grouped = {}
    for row in rows:
        pair = grouped.setdefault(tuple(row[k] for k in keys), {})
        if row["arm"] in pair:
            raise ValueError("duplicate paired arm")
        pair[row["arm"]] = row
    expected = {(c["case"], c["seed"], c["condition"]) for c in declared_protocol()["development"]["cells"]}
    if set(grouped) != expected or any(set(p) != set(ARMS) for p in grouped.values()):
        return {"status": "UNAVAILABLE", "verdict": "UNAVAILABLE_INCOMPLETE_COHORT",
                "complete_pairs": sum(set(p) == set(ARMS) for p in grouped.values()), "declared_pairs": len(expected)}
    control, resume = ARMS
    pairs = []
    for key, pair in sorted(grouped.items()):
        c, r = pair[control], pair[resume]
        pairs.append(dict(zip(keys, key), control_goal=bool(c["success"]), resume_goal=bool(r["success"]),
                          goal_difference=int(r["success"]) - int(c["success"]),
                          stopped_time_difference_s=r["stopped_navigation_time_s"] - c["stopped_navigation_time_s"],
                          control_stopped_s=c["stopped_navigation_time_s"], resume_stopped_s=r["stopped_navigation_time_s"]))
    b = sum(p["resume_goal"] and not p["control_goal"] for p in pairs)
    c_only = sum(p["control_goal"] and not p["resume_goal"] for p in pairs)
    p_value = float(binomtest(b, b + c_only, .5).pvalue) if b + c_only else 1.
    nonzero = [p["stopped_time_difference_s"] for p in pairs if p["stopped_time_difference_s"] != 0]
    sign_p = float(binomtest(sum(d < 0 for d in nonzero), len(nonzero), .5).pvalue) if nonzero else 1.
    arms = {}
    for arm in ARMS:
        values = [g[arm] for g in grouped.values()]
        goals = sum(v["success"] for v in values)
        completed = [d for v in values for d in v["recovery"]["completed_navigation_recovery_s"]]
        censored = [d for v in values for d in v["recovery"]["right_censored_navigation_loss_s"]]
        native = [d for v in values for d in v["recovery"]["native_relocalization_s"]]
        arms[arm] = dict(episodes=len(values), clean_goals=goals, clean_goal_rate=goals / len(values),
                         clean_goal_wilson_95=wilson(goals, len(values)),
                         falls=sum(v["fell"] for v in values), collisions=sum(v["collision"] for v in values),
                         mean_stopped_navigation_time_s=float(np.mean([v["stopped_navigation_time_s"] for v in values])),
                         tracking_losses=sum(v["recovery"]["tracking_loss_events"] for v in values),
                         completed_navigation_recoveries=len(completed),
                         navigation_recovery_median_s=float(np.median(completed)) if completed else None,
                         right_censored_losses=len(censored),
                         native_relocalization_median_s=float(np.median(native)) if native else None,
                         loss_mechanisms={m: sum(v["recovery"]["loss_mechanisms"][m] for v in values)
                                          for m in values[0]["recovery"]["loss_mechanisms"]},
                         verified_resumes=sum(v["verified_resumes"] for v in values),
                         resumed_pose_error_max_m=max((e["position_error_m"] for v in values for e in v["resumed_pose_errors_m"]), default=None))
    supported = p_value < .05 and b > c_only
    safety_regression = arms[resume]["falls"] > arms[control]["falls"] or arms[resume]["collisions"] > arms[control]["collisions"]
    verdict = "NEGATIVE_SAFETY" if safety_regression else "POSITIVE" if supported else "NEGATIVE"
    return {"status": "PASS", "verdict": verdict, "pairs": pairs, "arms": arms,
            "primary": {"test": "exact two-sided McNemar (binomial on discordant pairs)", "resume_only_goals_b": b,
                        "control_only_goals_c": c_only, "discordant_pairs": b + c_only, "p_value": p_value, "alpha": .05,
                        "supported": supported},
            "goal_difference": seed_cluster_bootstrap(pairs, "goal_difference"),
            "stopped_time_difference_s": dict(seed_cluster_bootstrap(pairs, "stopped_time_difference_s"),
                                              exact_sign_test_p=sign_p, nonzero_pairs=len(nonzero)),
            "safety_regression": safety_regression,
            "uncertainty_scope": "four seed clusters, one frozen actor, ideal simulated sensors; development evidence, not confirmation"}


EVIDENCE = (("inference_sensors.npz", "inference_sensors_sha256"), ("evaluator_truth.npz", "evaluator_truth_sha256"),
            ("native/stream_receipt.json", "stream_receipt_sha256"), ("native/native_frames.jsonl", "native_frames_sha256"),
            ("native/recovery_receipt.json", "recovery_receipt_sha256"), ("sensors/processing_schedule.json", "processing_schedule_sha256"))


def collect_campaign(root, smoke_path, output):
    """Strict exact-cohort collection; skipped gates remain unrun, never zero."""
    root = Path(root).resolve()
    smoke = json.loads(Path(smoke_path).read_text())
    ready = validate_smoke(smoke, sources=smoke.get("source_sha256"), inputs=smoke.get("input_sha256"))
    protocol = declared_protocol()
    seconds = protocol["development"]["seconds"]
    expected = {tuple(c[k] for k in ("case", "seed", "condition", "arm")) for c in protocol["development"]["cells"]}
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
            if not episode_path.is_relative_to(path.parent.resolve()) or sha256(episode_path) != result["episode_sha256"].get(relative):
                raise ValueError("retained episode differs from result hash")
            row = json.loads(episode_path.read_text())
            if (row.get("case"), row.get("seed"), row.get("condition"), row.get("arm")) != key or row.get("method") != ESTIMATOR:
                raise ValueError("episode identities differ from declared cell")
            for name, field in EVIDENCE:
                if sha256(episode_path.parent / name) != row[field]:
                    raise ValueError("retained raw/evaluator lineage differs")
            SCHED.validate_schedule_evidence(row, json.loads((episode_path.parent / "sensors/processing_schedule.json").read_text()))
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
            if stopped_navigation(trace, row["latency_quantization_s"])["stopped_navigation_decisions"] != row["stopped_navigation_decisions"]:
                raise ValueError("reported stopped time differs from decision trace")
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
    try:
        analysis = analyze(rows)
    except Exception as error:  # accounting stays valid; the frozen analysis can be rerun on retained evidence
        analysis = {"status": "ERROR", "verdict": "UNAVAILABLE_ANALYSIS_ERROR", "error": type(error).__name__ + ": " + str(error)}
    result = dict(schema=SCHEMA + "-collection", status="PASS", declared_cells=len(expected), measured_cells=len(observed),
        unrun_after_nominal_gate=len(unrun), missing_cells=[], arm_development_ready=ready,
        analysis=analysis, source_results=lineage, qualified_smoke_sha256=sha256(smoke_path),
        scientific_status="COMPLETE_BOUNDED_DEVELOPMENT_NO_CONFIRMATION", scope=protocol["scope"])
    NAV.write_json(output, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
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
    if shard and (a.phase != "development" or any(x is None for x in (a.case, a.seed, a.condition))):
        p.error("development shards require --case, --seed and --condition (one matched pair)")
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
        work = Path(os.getenv("H34_WORK_DIR", tempfile.mkdtemp(prefix="bhl-stereo-recovery-")))
        orb = STEREO.unpack_runtime(a.orb_runtime_archive, a.orb_runtime_sha256, work / "recovery-orb")
        lio = HYBRID.unpack_lio(a.lio_runtime_archive, a.lio_runtime_sha256, work / "recovery-lio")
        actor = {"name": "frozen_dr_default_s0", "deploy": str(a.deploy.resolve()), "checkpoint": str(a.checkpoint.resolve())}
        cells = protocol[a.phase]["cells"]
        if shard:
            cells = [c for c in cells if c["case"] == a.case and c["seed"] == a.seed and c["condition"] == a.condition]
        rows = []
        for cell in cells:
            if not readiness[cell["arm"]]:
                result["unrun_cells"].append(dict(cell, reason="UNRUN_AFTER_NEGATIVE_NOMINAL_ARM_SMOKE"))
                continue
            name = f"{cell['arm']}-{cell['case']}-s{cell['seed']}-{cell['condition']}"
            row = run_episode(a.upstream.resolve(), actor, {"orb": orb, "lio": lio}, cell["case"], cell["seed"], a.output / name,
                condition=cell["condition"], arm=cell["arm"], seconds=protocol[a.phase]["seconds"])
            if row["nonfinite"] or (row["native_frames"] < 1 and (a.phase == "smoke" or not (row["fell"] or row["collision"]))):
                raise ValueError("execution smoke/episode requires finite state and actual native input/output")
            if row["atlas_telemetry_frames"] != row["native_camera_frames"]:
                raise ValueError("every native camera response must carry Atlas telemetry")
            row["episode_file"] = name + "/episode.json"
            rows.append(row)
            result["completed_cells"].append(cell)
            print(json.dumps({k: row[k] for k in ("arm", "condition", "case", "seed", "success", "native_tracked_frames",
                                                  "verified_resumes", "stopped_navigation_time_s", "fell", "collision")}), flush=True)
        if source_hashes() != sources or any(sha256(getattr(a, k)) != v for k, v in inputs.items()):
            raise ValueError("source or frozen input mutated during the experiment")
        result.update(status="PASS", scientific_status="DEVELOPMENT_ONLY_NO_CONFIRMATION" if a.phase == "development" else "EXECUTION_SMOKE_WITH_EXPLICIT_SCIENTIFIC_ARM_GATES",
            execution_gate=True, source_sha256=sources, input_sha256=inputs, episodes=len(rows),
            episode_files=[r["episode_file"] for r in rows], expected_declared_cells=len(cells),
            episode_sha256={r["episode_file"]: sha256(a.output / r["episode_file"]) for r in rows},
            total_declared_development_cells=len(protocol["development"]["cells"]),
            arm_development_ready=nominal_arm_readiness(rows) if a.phase == "smoke" else readiness,
            arms={arm: {"episodes": sum(r["arm"] == arm for r in rows), "clean_goals": sum(r["success"] for r in rows if r["arm"] == arm),
                        "falls": sum(r["fell"] for r in rows if r["arm"] == arm), "contacts": sum(r["collision"] for r in rows if r["arm"] == arm)} for arm in ARMS},
            aggregate_improvement_verdict="UNAVAILABLE_UNTIL_COMPLETE_STRICT_PAIRED_COLLECTION", scope=protocol["scope"])
    except Exception as error:
        result["problems"].append(type(error).__name__ + ": " + str(error))
    result["wall_seconds"] = time.monotonic() - started
    NAV.write_json(a.output / "campaign_result.json", result)
    print(json.dumps({k: result[k] for k in ("schema", "phase", "status", "episodes", "problems")}), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
