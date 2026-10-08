"""Evaluate a 2/3-member BHL inspection-airlock team in one MuJoCo world.

This uses an oracle map/pose supervisor and a frozen Isaac-trained locomotion
policy. It is the feasibility baseline for navigation cooperation, not an
end-to-end sensor policy or a physical carrying demonstration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from bhl_robust.eval.multi_robot import MultiRunner, _WORLDS, build_multi
from bhl_robust.eval.team_airlock import TeamAirlock, velocity_command, world_xml


class CpuPolicy:
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.key = self.session.get_inputs()[0].name

    def forward(self, observation):
        return self.session.run(None, {self.key: observation})[0]


class ContactRunner(MultiRunner):
    """Check contacts at every physics step, including brief transient hits."""

    def configure_contacts(self, owners):
        self.owners = owners
        self.wall_geoms = np.array([
            (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
                ("wall_", "side_", "door")) for g in range(self.m.ngeom)])

    def step(self, targets_per_robot):
        self.hit_robot, self.hit_wall = False, False
        for _ in range(self.substeps):
            for i, slot in enumerate(self.slots):
                jp = self.d.sensordata[slot.jpos_adr]
                jv = self.d.sensordata[slot.jvel_adr]
                tau = self.kp * (targets_per_robot[i] - jp) - self.kd * jv
                self.d.ctrl[slot.ctrl] = np.clip(tau, -self.eff, self.eff)
            mujoco.mj_step(self.m, self.d)
            hook = getattr(self, "substep_hook", None)   # e.g. an IMU/attitude filter at sensor rate
            if hook is not None:
                hook(self.d)
            if self.d.ncon:
                g1, g2 = self.d.contact.geom1, self.d.contact.geom2
                a, b = self.owners[g1], self.owners[g2]
                self.hit_robot |= bool(np.any((a >= 0) & (b >= 0) & (a != b)))
                self.hit_wall |= bool(np.any(((a >= 0) & self.wall_geoms[g2])
                                             | ((b >= 0) & self.wall_geoms[g1])))


def evaluate(args, model, slots, cfg, policy, seed, mode):
    controllers = [RlController(cfg) for _ in slots]
    for controller in controllers:
        controller.policy = policy
    runner = ContactRunner(model, slots, [cfg] * len(slots), controllers)
    rng = np.random.default_rng(seed)
    runner.reset(rng)
    mission = TeamAirlock(len(slots), mode=mode)
    # Keep free joints at the exact designated station lanes. Other reset
    # perturbations still randomize the actual joint configurations.
    for i, slot in enumerate(slots):
        runner.d.qpos[slot.qpos_adr:slot.qpos_adr + 2] = mission.starts[i] + rng.normal(0, .025, 2)
    mujoco.mj_forward(model, runner.d)
    owners = np.full(model.ngeom, -1, dtype=int)
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        for i, slot in enumerate(slots):
            if name.startswith(slot.prefix):
                owners[g] = i
    runner.configure_contacts(owners)
    sensor_sampler = None
    if args.sensor_mode != "off":
        from bhl_robust.eval.team_sensors import TeamSensors
        sensor_sampler = TeamSensors(model, slots, owners, mode=args.sensor_mode, seed=seed,
                                     dropout_probability=args.sensor_dropout_probability)
    door_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "airlock")
    door_mid = model.body_mocapid[door_bid]
    dt = float(cfg.policy_dt)
    min_separation = float("inf")
    robot_contact_steps = 0
    wall_contact_steps = 0
    failed = None
    trace = []
    video = None
    renderer = None
    if args.video and mode == "coordinated" and seed == args.seed_start:
        import imageio.v2 as imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        video = imageio.get_writer(str(args.video), fps=25)
        renderer = mujoco.Renderer(model, height=540, width=960)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [1.8, 0.0, 0.35]
        camera.distance, camera.azimuth, camera.elevation = 6.3, 140, -32
    try:
        for step in range(int(args.seconds / dt)):
            now = step * dt
            positions = np.array([runner.d.xpos[s.body_id, :2] for s in slots])
            tilts = np.array([runner.tilt(i) for i in range(len(slots))])
            if not np.isfinite(runner.d.qpos).all() or not np.isfinite(runner.d.qvel).all():
                failed = "nonfinite_state"
                break
            upright = tilts < 0.78
            if not upright.all():
                failed = "fall"
                break
            mission.update(positions, upright, dt, now)
            runner.d.mocap_pos[door_mid, 2] = 2.0 if mission.door_open else .55
            targets = mission.targets()
            commands = []
            for i, slot in enumerate(slots):
                q = runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]
                yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                 1 - 2 * (q[2] ** 2 + q[3] ** 2))
                target = targets[i]
                # Independent start delays expose whether the first arriver
                # waits for the other roles before leaving its station.
                if now < 1.0 + i * .8:
                    target = mission.starts[i]
                commands.append(velocity_command(positions[i], yaw, target, args.speed))
            if sensor_sampler is not None:
                unfiltered_commands = [cmd.tolist() for cmd in commands]
                commands = sensor_sampler.filter_commands(runner.d, commands, now)
            targets_q = [c.update(runner.observe(i, commands[i])) for i, c in enumerate(controllers)]
            if not all(np.isfinite(q).all() for q in targets_q):
                failed = "nonfinite_action"
                break
            runner.step(targets_q)
            robot_contact_steps += int(runner.hit_robot)
            wall_contact_steps += int(runner.hit_wall)
            for i in range(len(slots)):
                for j in range(i):
                    min_separation = min(min_separation, float(np.linalg.norm(positions[i] - positions[j])))
            if step % 25 == 0:
                trace.append({"time_s": round(now, 3), "xy": positions.tolist(),
                              "door_open": mission.door_open, "active_member": int(mission.active),
                              "stages": mission.stages.tolist(), "tilt_rad": tilts.tolist(),
                              "commands": [cmd.tolist() for cmd in commands]})
                if sensor_sampler is not None:
                    trace[-1]["sensors"] = sensor_sampler.latest.copy()
                    trace[-1]["unfiltered_commands"] = unfiltered_commands
            if renderer is not None:
                renderer.update_scene(runner.d, camera=camera)
                video.append_data(renderer.render())
            if any(runner.tilt(i) >= .78 for i in range(len(slots))):
                failed = "fall"
                break
            if mission.completed_at is not None:
                break
    finally:
        if renderer is not None:
            renderer.close()
            video.close()
    success = mission.completed_at is not None and failed is None and robot_contact_steps == 0 and wall_contact_steps == 0
    if not success and failed is None:
        failed = ("robot_collision" if robot_contact_steps else "wall_collision" if wall_contact_steps
                  else "team_timeout")
    row = {"seed": seed, "mode": mode, "crew": len(slots), "success": bool(success),
            "failure": failed, "elapsed_s": round(now, 3), "release_s": mission.released_at,
            "completion_s": mission.completed_at, "stations_visited": int(mission.visited.sum()),
            "members_crossed": int(np.sum(mission.stages == 2)),
            "robot_contact_steps": robot_contact_steps, "wall_contact_steps": wall_contact_steps,
            "minimum_base_separation_m": min_separation if np.isfinite(min_separation) else None,
            "final_xy": positions.tolist(), "trace": trace}
    if sensor_sampler is not None:
        row["sensors"] = {"mode": args.sensor_mode, "stats": sensor_sampler.stats,
                          "dropout_probability": sensor_sampler.dropout,
                          "metadata": sensor_sampler.metadata()}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--crew", type=int, choices=(2, 3), default=2)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--speed", type=float, default=.4)
    parser.add_argument("--modes", nargs="+", choices=("coordinated", "no_wait", "withhold_last"),
                        default=["coordinated", "no_wait", "withhold_last"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--sensor-mode", choices=("off", "record", "reactive", "reactive_dropout"),
                        default="off", help="optional actual ray/IMU inputs; localization remains oracle")
    parser.add_argument("--sensor-dropout-probability", type=float, default=.35,
                        help="uncalibrated extero packet loss stress test in reactive_dropout mode")
    parser.add_argument("--gate", action="store_true", help="require >=80%% coordinated success, worse no-wait, and no withheld-role success")
    args = parser.parse_args()
    if args.seeds <= 0 or args.seconds <= 0 or not 0 < args.speed <= .5:
        parser.error("seeds/seconds must be positive, and speed must be in (0, .5]")
    if not 0 <= args.sensor_dropout_probability <= 1:
        parser.error("sensor dropout probability must be in [0,1]")
    cfg = OmegaConf.load(args.deploy)
    if cfg.num_actions != 22 or cfg.num_joints != 22 or cfg.num_observations != 75:
        parser.error("this task requires the full 22-DoF, 75-observation humanoid locomotion policy")
    if args.seconds < float(cfg.policy_dt):
        parser.error("episode duration must include at least one policy step")
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    _WORLDS["team_airlock"] = world_xml(args.crew)
    model, slots = build_multi(args.upstream, args.cache_dir, args.crew,
                               [f"member-{i}" for i in range(args.crew)], variant="humanoid",
                               spacing=1.8 / (args.crew - 1), world="team_airlock")
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            row = evaluate(args, model, slots, cfg, policy, seed, mode)
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "trace"}), flush=True)
            # Flush partial evidence after each episode so wall-time cancellation
            # does not discard the completed controls.
            payload = {"task": "team_airlock_v1", "control": "oracle_pose_map_supervisor+frozen_isaac_locomotion",
                       "simulator": f"MuJoCo {mujoco.__version__}", "physical_object_transport": False,
                       "deploy": str(args.deploy.resolve()), "checkpoint": cfg.policy_checkpoint_path,
                       "crew": args.crew, "episodes": rows}
            if args.sensor_mode != "off":
                payload["sensor_mode"] = args.sensor_mode
                if args.sensor_mode.startswith("reactive"):
                    payload["control"] += "+range_imu_brake"
            args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    coordinated = [r for r in rows if r["mode"] == "coordinated"]
    withheld = [r for r in rows if r["mode"] == "withhold_last"]
    no_wait = [r for r in rows if r["mode"] == "no_wait"]
    rates = {mode: float(np.mean([r["success"] for r in rows if r["mode"] == mode]))
             for mode in args.modes}
    complete = bool(coordinated and withheld and no_wait)
    gate = bool(complete and rates["coordinated"] >= .8
                and rates["no_wait"] < rates["coordinated"] and rates["withhold_last"] == 0)
    payload["summary"] = {"success_rates": rates, "gate_passed": gate,
                          "controls_complete": complete, "seconds": args.seconds, "speed": args.speed,
                          "policy_dt": float(cfg.policy_dt), "physics_dt": float(cfg.physics_dt),
                          "contact_check": "every_physics_substep"}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    verdict = "PASS" if gate else "FAIL" if complete else "INCOMPLETE_CONTROLS"
    print(f"TEAM_GATE={verdict} output={args.output}", flush=True)
    if args.gate and not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
