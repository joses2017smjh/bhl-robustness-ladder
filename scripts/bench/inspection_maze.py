"""Single full-humanoid dogleg maze: two inspections, two turns, dead-end, exit.

Uses oracle waypoints/localization, frozen Isaac-trained locomotion, actual
simulated LiDAR/IMU and paired idealized ray depth. No RGB detector is present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from bhl_robust.eval.inspection_maze import InspectionMaze, OPEN_CELLS, CELL_M, world_xml
from bhl_robust.eval.multi_robot import _WORLDS, build_multi
from bhl_robust.eval.team_airlock import velocity_command
from bhl_robust.eval.team_sensors import TeamSensors
from bhl_robust.fusion.attitude import (FILTERS, GRAVITY_M_S2, ImuNoise, attitude_from_accel,
                                        body_gravity)
from team_airlock import ContactRunner, CpuPolicy


class EstimatedAttitude:
    """Attitude filter in the loop (SF-03, docs/SENSOR_FUSION.md).

    Replaces the oracle quaternion and gyro in the policy observation with a
    Mahony/Madgwick estimate driven by the *corrupted* MuJoCo gyro and
    accelerometer. The filter runs at the policy rate, which is slower than a
    real AHRS; that makes this a conservative test of the frozen gait's
    tolerance, not a model of any particular IMU.
    """

    def __init__(self, args, model, slot, seed):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, slot.prefix + "imu_acc")
        if sid < 0:
            raise ValueError(f"missing accelerometer sensor for {slot.prefix}")
        self.acc_adr = int(model.sensor_adr[sid])
        self.slot = slot
        self.args = args
        self.noise = ImuNoise(gyro_std=args.imu_gyro_std, accel_std=args.imu_accel_std,
                              gyro_bias=args.imu_gyro_bias, gyro_bias_walk=args.imu_gyro_bias_walk,
                              delay_steps=args.imu_delay_steps, seed=seed)
        self.filter = None
        self.errors = []
        self.pre_alignment_steps = 0
        self._still = []
        # Sensor-rate mode (SF-03b): the filter runs every `every` physics substeps
        # from a substep hook, with the delivery delay quantised in IMU samples;
        # apply() then only substitutes the latest estimate into the observation.
        self.sensor_rate_hz = float(getattr(args, "imu_rate_hz", 0.0) or 0.0)
        self.every = None
        self._substep = 0
        self._latest = None

    ALIGN_STEPS = 10         # stationary averaging window in policy steps (0.4 s at 25 Hz)

    def _make_filter(self, q_true, accel):
        yaw = float(np.arctan2(2*(q_true[0]*q_true[3] + q_true[1]*q_true[2]),
                               1 - 2*(q_true[2]**2 + q_true[3]**2)))
        q0 = q_true.copy() if self.args.imu_init == "truth" else attitude_from_accel(accel, yaw=yaw)
        if self.args.imu_filter == "mahony":
            return FILTERS["mahony"](kp=self.args.imu_kp, ki=self.args.imu_ki, q0=q0)
        return FILTERS["madgwick"](beta=self.args.imu_beta, q0=q0)

    def bind_sensor_rate(self, runner, physics_dt):
        """Run the filter from the runner's substep hook at `sensor_rate_hz`."""
        if self.sensor_rate_hz <= 0:
            return
        self.every = max(1, int(round(1.0 / (self.sensor_rate_hz * physics_dt))))
        self.ALIGN_STEPS = max(5, int(round(0.4 * self.sensor_rate_hz)))   # keep a 0.4 s window
        self.sensor_dt = self.every * physics_dt
        delay_ms = float(getattr(self.args, "imu_delay_ms", 0.0) or 0.0)
        self.noise.delay_steps = int(round(delay_ms / 1000.0 / self.sensor_dt))
        runner.substep_hook = self._substep_update

    def _substep_update(self, data):
        self._substep += 1
        if self._substep % self.every:
            return
        self._latest = self._update(data, self.sensor_dt)

    def _update(self, data, dt):
        """One filter update from the current MuJoCo sensors; returns (q_est, g_corrected) or None before alignment."""
        s = self.slot
        q_true = data.sensordata[s.quat_adr:s.quat_adr + 4].copy()
        gyro = data.sensordata[s.gyro_adr:s.gyro_adr + 3]
        accel = data.sensordata[self.acc_adr:self.acc_adr + 3]
        g, a = self.noise(gyro, accel, dt)
        if self.filter is None:
            # Stationary alignment gate. The robot spawns in free fall (specific
            # force ~0, direction = noise) and then lands (|a| passes through g
            # while pointing tens of degrees off). A deployment aligns while
            # standing still by AVERAGING a short window, so noise on single
            # samples does not block alignment: over the last ALIGN_STEPS
            # samples we require no free-fall sample (|a| > 0.5 g each), the
            # mean specific force within 10 % of g, and the mean gyro below
            # 0.3 rad/s. The oracle observation is passed through until then
            # (the mission target is zero for the first second anyway).
            self.pre_alignment_steps += 1
            self._still.append((a.copy(), g.copy()))
            if len(self._still) > self.ALIGN_STEPS:
                self._still.pop(0)
            if len(self._still) < self.ALIGN_STEPS:
                return None
            accs = np.array([x[0] for x in self._still]); gyrs = np.array([x[1] for x in self._still])
            if (np.linalg.norm(accs, axis=1).min() <= 0.5*GRAVITY_M_S2
                    or abs(np.linalg.norm(accs.mean(0)) - GRAVITY_M_S2) > 0.10*GRAVITY_M_S2
                    or np.linalg.norm(gyrs.mean(0)) >= 0.3):
                return None
            self.filter = self._make_filter(q_true, accs.mean(0))
        q_est = self.filter.update(g, a, dt)
        self.errors.append(float(np.linalg.norm(body_gravity(q_est) - body_gravity(q_true))))
        return q_est, g - self.filter.bias

    def apply(self, obs, data, dt):
        """Substitute the estimate into the policy observation (policy-rate mode
        updates the filter here; sensor-rate mode uses the latest hook result)."""
        est = self._latest if self.every else self._update(data, dt)
        if est is None:
            return obs
        q_est, g_corr = est
        obs = obs.copy()
        obs[0:4] = q_est
        obs[4:7] = g_corr
        return obs

    def summary(self):
        e = np.asarray(self.errors)
        return {"source": "estimated", "filter": self.args.imu_filter,
                "filter_rate_hz": self.sensor_rate_hz if self.every else 25.0,
                "imu_delay_ms": float(getattr(self.args, "imu_delay_ms", 0.0) or 0.0) if self.every else None,
                "aligned": self.filter is not None,
                "pre_alignment_steps": self.pre_alignment_steps,
                "gravity_rmse": float(np.sqrt(np.mean(e**2))) if e.size else None,
                "gravity_max": float(e.max()) if e.size else None,
                "gyro_bias_true_rad_s": self.noise.bias.tolist(),
                "gyro_bias_estimated_rad_s": self.filter.bias.tolist() if self.filter is not None else None,
                "settings": {k: getattr(self.args, k) for k in (
                    "imu_gyro_std", "imu_accel_std", "imu_gyro_bias", "imu_gyro_bias_walk",
                    "imu_delay_steps", "imu_init", "imu_kp", "imu_ki", "imu_beta")}}


def episode(args, model, slots, cfg, policy, seed, sensor_mode, route, frame_hook=None):
    """One mission. `frame_hook(step=, now=, runner=, sensors=, mission=, xy=, yaw=,
    command=, raw=)` is called after every policy step for recorders; it only
    reads state, and the run is the same with or without it."""
    controller = RlController(cfg)
    controller.policy = policy
    runner = ContactRunner(model, slots, [cfg], [controller])
    rng = np.random.default_rng(seed)
    runner.reset(rng)
    slot = slots[0]
    runner.d.qpos[slot.qpos_adr:slot.qpos_adr+2] = rng.normal(0, .025, 2)
    mujoco.mj_forward(model, runner.d)
    owners = np.array([0 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                             int(model.geom_bodyid[g])) or "").startswith(slot.prefix)
                       else -1 for g in range(model.ngeom)])
    runner.configure_contacts(owners)
    sensors = TeamSensors(model, slots, owners, mode=sensor_mode, seed=seed,
                           dropout_probability=args.dropout_probability)
    mission = InspectionMaze(route=route)
    trace, wall_contact_steps, path_m = [], 0, 0.0
    prior_xy = runner.d.xpos[slot.body_id, :2].copy()
    dt = float(cfg.policy_dt)
    imu = EstimatedAttitude(args, model, slot, seed) if args.imu_source == "estimated" else None
    if imu is not None:
        imu.bind_sensor_rate(runner, float(cfg.physics_dt))
    # SF-02: localization error applied ONLY to the pose the route controller uses;
    # the mission judge (mission.update) keeps the true pose.
    loc_rng = np.random.default_rng(10_000 + seed)
    loc_bias = args.pose_bias_m * (lambda v: v / np.linalg.norm(v))(loc_rng.normal(size=2)) if args.pose_bias_m else np.zeros(2)
    loc_yaw = np.deg2rad(args.pose_yaw_deg)
    renderer = video = None
    if (args.video and sensor_mode == args.video_sensor_mode
            and seed == args.seed_start and route == args.video_route):
        import imageio.v2 as imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        video = imageio.get_writer(str(args.video), fps=25)
        renderer = mujoco.Renderer(model, height=540, width=960)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [2.4, .2, .3]
        camera.distance, camera.azimuth, camera.elevation = 8.6, 120, -58
    try:
        for step in range(int(args.seconds / dt)):
            now = step*dt
            xy = runner.d.xpos[slot.body_id, :2].copy()
            tilt = runner.tilt(0)
            if not np.isfinite(runner.d.qpos).all() or not np.isfinite(runner.d.qvel).all():
                mission.failure = "nonfinite_state"
                break
            mission.update(xy, tilt < .78, dt, now)
            if mission.failure is not None or mission.completed_at is not None:
                break
            q = runner.d.qpos[slot.qpos_adr+3:slot.qpos_adr+7]
            yaw = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
            target = mission.target() if now >= 1 else np.zeros(2)
            xy_est = xy + loc_bias + args.pose_drift_mps * now * np.array([1.0, 0.0]) + (loc_rng.normal(size=2) * args.pose_noise_m if args.pose_noise_m else 0.0)
            raw = velocity_command(xy_est, yaw + loc_yaw, target, args.speed)
            command = sensors.filter_commands(runner.d, [raw], now)[0]
            obs = runner.observe(0, command)
            if imu is not None:
                obs = imu.apply(obs, runner.d, dt)
            target_q = controller.update(obs)
            if not np.isfinite(target_q).all():
                mission.failure = "nonfinite_action"
                break
            runner.step([target_q])
            wall_contact_steps += int(runner.hit_wall)
            after = runner.d.xpos[slot.body_id, :2].copy()
            path_m += float(np.linalg.norm(after-prior_xy))
            prior_xy = after
            if step % 25 == 0:
                trace.append({"time_s": now, "xy": xy.tolist(), "stage": mission.stage,
                              "target": target.tolist(), "label": mission.active_label,
                              "tilt_rad": tilt, "unfiltered_command": raw.tolist(),
                              "command": command.tolist(), "sensors": sensors.latest[0]})
            if renderer is not None:
                renderer.update_scene(runner.d, camera=camera)
                video.append_data(renderer.render())
            if frame_hook is not None:
                frame_hook(step=step, now=now, runner=runner, sensors=sensors, mission=mission,
                           xy=after, yaw=yaw, command=command, raw=raw)
            if runner.tilt(0) >= .78:
                mission.failure = "fall"
                break
    finally:
        if renderer is not None:
            renderer.close()
            video.close()
    success = mission.completed_at is not None and mission.failure is None and wall_contact_steps == 0
    failure = mission.failure
    if not success and failure is None:
        failure = "wall_collision" if wall_contact_steps else "mission_timeout"
    return {"seed": seed, "route": route, "sensor_mode": sensor_mode, "success": bool(success),
            "failure": failure, "elapsed_s": round(now, 3), "completion_s": mission.completed_at,
            "stations_completed": len(mission.station_times), "station_times": mission.station_times,
            "waypoints_completed": mission.stage, "wall_contact_steps": wall_contact_steps,
            "path_length_m": path_m, "final_xy": runner.d.xpos[slot.body_id, :2].tolist(),
            "sensor_stats": sensors.stats, "dropout_probability": sensors.dropout,
            "imu": imu.summary() if imu is not None else {"source": "truth"},
            "localization_error": {"pose_bias_m": args.pose_bias_m, "pose_noise_m": args.pose_noise_m,
                                   "pose_yaw_deg": args.pose_yaw_deg, "pose_drift_mps": args.pose_drift_mps},
            "trace": trace}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--speed", type=float, default=.4)
    parser.add_argument("--sensor-modes", nargs="+", choices=("record", "reactive", "reactive_dropout"),
                        default=["reactive", "reactive_dropout"])
    parser.add_argument("--dropout-probability", type=float, default=1.,
                        help="complete-outage control by default; uncalibrated stress-test parameter")
    parser.add_argument("--routes", nargs="+", choices=("ordered", "wrong_branch"), default=["ordered"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-route", choices=("ordered", "wrong_branch"), default="ordered",
                        help="Explicitly select a nominal or negative-control recording")
    parser.add_argument("--video-sensor-mode", choices=("record", "reactive", "reactive_dropout"),
                        default="reactive")
    parser.add_argument("--gate", action="store_true")
    imu = parser.add_argument_group("attitude filter in the loop (SF-03, docs/SENSOR_FUSION.md)")
    imu.add_argument("--imu-source", choices=("truth", "estimated"), default="truth",
                     help="'estimated' replaces the oracle quaternion/gyro the policy sees with a "
                          "filter estimate driven by corrupted MuJoCo gyro+accelerometer samples")
    imu.add_argument("--imu-filter", choices=tuple(FILTERS), default="mahony")
    imu.add_argument("--imu-init", choices=("accel", "truth"), default="accel",
                     help="initial alignment: tilt from the first accelerometer sample (yaw from truth), or truth")
    imu.add_argument("--imu-kp", type=float, default=1.0)
    imu.add_argument("--imu-ki", type=float, default=0.1)
    imu.add_argument("--imu-beta", type=float, default=0.1)
    imu.add_argument("--imu-gyro-std", type=float, default=0.0, help="rad/s white noise (stress setting, not a calibration)")
    imu.add_argument("--imu-accel-std", type=float, default=0.0, help="m/s^2 white noise")
    imu.add_argument("--imu-gyro-bias", type=float, default=0.0, help="rad/s constant bias magnitude, random direction per seed")
    imu.add_argument("--imu-gyro-bias-walk", type=float, default=0.0, help="rad/s/sqrt(s)")
    imu.add_argument("--imu-delay-steps", type=int, default=0, help="policy steps of IMU delivery delay")
    imu.add_argument("--imu-rate-hz", type=float, default=0.0, help="run the filter at this sensor rate from the physics substep hook (0 = policy rate)")
    imu.add_argument("--imu-delay-ms", type=float, default=0.0, help="IMU delivery delay in ms (sensor-rate mode only)")
    loc = parser.add_argument_group("localization error fed to the route controller (SF-02)")
    loc.add_argument("--pose-bias-m", type=float, default=0.0, help="constant position offset magnitude, random direction per seed")
    loc.add_argument("--pose-noise-m", type=float, default=0.0, help="white position noise std per policy step")
    loc.add_argument("--pose-yaw-deg", type=float, default=0.0, help="constant heading-estimate error")
    loc.add_argument("--pose-drift-mps", type=float, default=0.0, help="position drift rate along +x")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.video and (args.video_route not in args.routes or args.video_sensor_mode not in args.sensor_modes):
        parser.error("Requested video route/sensor mode must be included in this evaluation")
    if args.video and args.video.exists():
        parser.error("Video output already exists; choose a new path")
    if args.seeds < 1 or not 0 < args.speed <= .5 or not 0 <= args.dropout_probability <= 1:
        parser.error("positive seed count, speed in (0,.5], and dropout probability in [0,1] required")
    cfg = OmegaConf.load(args.deploy)
    if cfg.num_actions != 22 or cfg.num_joints != 22 or cfg.num_observations != 75:
        parser.error("requires full 22-DoF/75-observation humanoid locomotion checkpoint")
    if args.seconds < float(cfg.policy_dt):
        parser.error("episode duration must include one policy step")
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    _WORLDS["inspection_maze"] = world_xml()
    model, slots = build_multi(args.upstream, args.cache_dir, 1, ["inspector"], variant="humanoid",
                               world="inspection_maze")
    rows = []
    payload = {"task": "inspection_maze_v1", "simulator": f"MuJoCo {mujoco.__version__}",
               "control": "oracle_map_waypoints_localization+frozen_isaac_gait+optional_sensor_brake",
               "inspection": "ordered_proximity_dwell_not_object_detection_or_button_press",
               "heading": "fixed_east_holonomic_travel_with_two_90_degree_path_turns",
               "cell_size_m": CELL_M, "open_cells": OPEN_CELLS,
               "deploy": str(args.deploy.resolve()), "checkpoint": cfg.policy_checkpoint_path,
               "sensor_metadata": TeamSensors.metadata(), "episodes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for route in args.routes:
        for mode in args.sensor_modes:
            for seed in range(args.seed_start, args.seed_start+args.seeds):
                row = episode(args, model, slots, cfg, policy, seed, mode, route)
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "trace"}), flush=True)
                args.output.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    nominal = [r for r in rows if r["route"] == "ordered" and r["sensor_mode"] == "reactive"]
    outage = [r for r in rows if r["route"] == "ordered" and r["sensor_mode"] == "reactive_dropout"]
    rates = {mode: float(np.mean([r["success"] for r in rows if r["route"] == "ordered"
                                 and r["sensor_mode"] == mode]))
             for mode in args.sensor_modes if any(r["route"] == "ordered" and r["sensor_mode"] == mode for r in rows)}
    complete = bool(nominal and outage)
    gate = bool(complete and rates["reactive"] >= .8 and rates["reactive_dropout"] < rates["reactive"])
    payload["summary"] = {"gate_passed": gate, "controls_complete": complete, "success_rates": rates,
                          "seconds": args.seconds, "speed": args.speed, "contact_check": "every_physics_substep",
                          "imu_source": args.imu_source}
    if args.imu_source == "estimated":
        rmse = [r["imu"]["gravity_rmse"] for r in rows if r["imu"].get("gravity_rmse") is not None]
        payload["summary"]["imu_aligned_episodes"] = int(sum(bool(r["imu"].get("aligned")) for r in rows))
        payload["summary"]["imu_gravity_rmse_mean"] = float(np.mean(rmse)) if rmse else None
        payload["summary"]["imu_gravity_max"] = float(max(r["imu"]["gravity_max"] for r in rows if r["imu"].get("gravity_max") is not None)) if rmse else None
        payload["attitude_filter"] = {"filter": args.imu_filter, "rate": "policy step (conservative; a real AHRS runs faster)",
                                      "init": args.imu_init, "noise_is_calibration": False}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    verdict = "PASS" if gate else "FAIL" if complete else "INCOMPLETE_CONTROLS"
    print(f"INSPECTION_MAZE_GATE={verdict} output={args.output}", flush=True)
    if args.gate and not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
