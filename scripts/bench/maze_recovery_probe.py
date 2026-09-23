"""Finite Isaac maze smoke / checkpoint rollout with real navigation metrics.

Run inside the existing v60 bhl_exec environment. Without --checkpoint this is
an integration smoke only; it cannot certify navigation success. With one,
--minimum-success is the explicit promotion gate. Writes JSON only at the end
and prints MAZE_PROBE_PASS, so an Isaac shutdown exit-code bug cannot pass it.
"""
from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Velocity-BHL-MazeRecovery-Approach-Blind-v0")
parser.add_argument("--checkpoint")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--minimum-success", type=float, default=0.0)
parser.add_argument("--output", required=True)
parser.add_argument("--settings", default=None,
                    help="JSON list of sensor-fusion settings to evaluate in ONE boot (docs/SENSOR_FUSION.md SF-01/SF-02). "
                         "Each item: {name, gyro_std, gravity_std, imu_delay_steps, pose_bias_m, pose_noise_m, pose_yaw_deg}. "
                         "Implies --keep-corruption for items with noise; each item is a full rollout after env.reset.")
parser.add_argument("--keep-corruption", action="store_true",
                    help="Keep the task's policy observation noise (training default) instead of "
                         "forcing enable_corruption=False. Off by default = published behaviour.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)

import gymnasium as gym
import torch
import bhl_robust.tasks  # noqa: F401
from bhl_robust import maze_recovery as recovery



def _imu_slices(u):
    """Column slices of the IMU terms inside obs['policy'] (base_ang_vel, projected_gravity)."""
    om = u.observation_manager
    names = om.active_terms["policy"]; dims = om.group_obs_term_dim["policy"]
    out, col = {}, 0
    for n, d in zip(names, dims):
        w = int(torch.tensor(d).prod()) if not isinstance(d, int) else d
        if n in ("base_ang_vel", "projected_gravity"):
            out[n] = slice(col, col + w)
        col += w
    return out


class _ImuDelay:
    """Policy-side FIFO delay of the IMU columns only (SF-01): the policy acts on
    IMU values from `steps` control periods ago, everything else is current."""

    def __init__(self, slices, steps):
        self.slices, self.steps, self.queue = slices, int(steps), []

    def __call__(self, obs):
        if self.steps <= 0 or not self.slices:
            return obs
        pol = obs["policy"] if isinstance(obs, dict) or hasattr(obs, "keys") else obs
        cur = torch.cat([pol[:, s] for s in self.slices.values()], dim=1).clone()
        self.queue.append(cur)
        if len(self.queue) > self.steps:
            old = self.queue.pop(0)
        else:
            old = self.queue[0]
        pol = pol.clone(); c = 0
        for s in self.slices.values():
            w = s.stop - s.start; pol[:, s] = old[:, c:c + w]; c += w
        if isinstance(obs, dict) or hasattr(obs, "keys"):
            obs = obs.clone() if hasattr(obs, "clone") else dict(obs)
            obs["policy"] = pol
            return obs
        return pol


def _apply_setting(u, setting):
    """Mutate observation-noise stds at runtime (the noise func reads cfg.std on every
    compute) and install a localization-error hook on recovery.local_xy (SF-02)."""
    om = u.observation_manager
    def term_cfg(term):
        try:
            return om.get_term_cfg("policy", term)
        except Exception:                                        # noqa: BLE001
            names = list(om.active_terms["policy"])
            return om._group_obs_term_cfgs["policy"][names.index(term)]
    for term, key in (("base_ang_vel", "gyro_std"), ("projected_gravity", "gravity_std")):
        if setting.get(key) is not None:
            tc = term_cfg(term)
            if getattr(tc, "noise", None) is None:
                raise RuntimeError(f"{term} has no noise cfg; run with --keep-corruption on a task that defines one")
            tc.noise.std = float(setting[key])
    bias_m = float(setting.get("pose_bias_m") or 0.0); noise_m = float(setting.get("pose_noise_m") or 0.0)
    yaw_deg = float(setting.get("pose_yaw_deg") or 0.0)
    cmd = u.command_manager.get_term("base_velocity")
    gen = torch.Generator(device=u.device).manual_seed(int(setting.get("seed", 0)) + 7)
    if bias_m:
        ang = torch.rand(u.num_envs, generator=gen, device=u.device) * 2 * math.pi
        cmd._sf02_bias = bias_m * torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
    else:
        cmd._sf02_bias = None
    cmd._sf02_noise_m = noise_m
    cmd._sf02_gen = gen
    # Yaw error: rotate the command the teacher produces (heading error) via the
    # command term's heading_target offset.
    cmd._sf02_yaw = math.radians(yaw_deg)
    return {"gyro_std": setting.get("gyro_std"), "gravity_std": setting.get("gravity_std"),
            "imu_delay_steps": int(setting.get("imu_delay_steps") or 0), "pose_bias_m": bias_m,
            "pose_noise_m": noise_m, "pose_yaw_deg": yaw_deg}


def _rollout(env, u, policy, obs, start, delay, steps):
    completed = success = dead_end = timeout = 0
    first_finished = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    first_success = torch.zeros_like(first_finished)
    minimum_dist = torch.full((u.num_envs,), float("inf"), device=u.device)
    max_displacement = torch.zeros(args.num_envs, device=u.device)
    for _ in range(steps):
        with torch.inference_mode():
            action = policy(delay(obs)) if policy else torch.zeros((u.num_envs, u.action_manager.total_action_dim), device=u.device)
            if policy:
                obs, _, done, _ = env.step(action)
            else:
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
        # The cached state is from the pre-reset terminal transition, unlike
        # scene.root_pos_w which has already been reset by ManagerBasedRLEnv.
        state = getattr(u, "_maze_recovery_state", None)
        if state is not None:
            minimum_dist = torch.minimum(minimum_dist, state["dist"])
            won = state["success"]
        else:
            won = u.termination_manager.get_term("button_reached")
            distance = (recovery.local_xy(u) - start.new_tensor((3.0, 0.0))).norm(dim=-1)
            minimum_dist = torch.minimum(minimum_dist, distance)
        first_success |= won & done.bool() & ~first_finished
        first_finished |= done.bool()
        success += int((won & done.bool()).sum())
        dead_end += int((u.termination_manager.get_term("dead_end") & done.bool()).sum())
        timeout += int((u.termination_manager.time_outs & done.bool()).sum())
        completed += int(done.sum())
        xy = recovery.local_xy(u)
        max_displacement = torch.maximum(max_displacement, (xy - start).norm(dim=-1))
    return dict(completed=completed, success=success, dead_end=dead_end, timeout=timeout,
                first_finished=first_finished, first_success=first_success,
                minimum_dist=minimum_dist, max_displacement=max_displacement)


def run():
    spec = gym.spec(args.task)
    cfg = spec.kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    if not args.keep_corruption:
        cfg.observations.policy.enable_corruption = False
    observation_corruption = bool(cfg.observations.policy.enable_corruption)
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    start = recovery.local_xy(u).clone()
    command = u.command_manager.get_command("base_velocity")
    if not torch.isfinite(obs["policy"]).all() or not (command[:, 0] > 0.0).all():
        raise RuntimeError("Non-finite observation or missing forward approach command at reset")
    sensors = {}
    for name in ("lidar", "stereo_l", "stereo_r"):
        if name not in u.scene.sensors:
            continue
        sensor = u.scene.sensors[name]
        if name == "lidar":
            from bhl_robust.sensors_rig import lidar_obs
            from isaaclab.managers import SceneEntityCfg
            values = lidar_obs(u, SceneEntityCfg(name))
        else:
            from bhl_robust.tasks.depth_env_cfg import depth_obs
            from isaaclab.managers import SceneEntityCfg
            values = depth_obs(u, SceneEntityCfg(name), pool=16)
        if not torch.isfinite(values).all() or float(values.std()) < 1.e-5:
            raise RuntimeError(f"{name}: non-finite or constant observation")
        sensors[name] = {"shape": list(values.shape), "min": float(values.min()), "max": float(values.max())}
    policy = None
    if args.checkpoint:
        from importlib.metadata import version
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        agent = spec.kwargs["rsl_rl_cfg_entry_point"]()
        try:
            from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
            agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
        except ImportError:
            pass
        env = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=u.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=u.device)
        obs = env.get_observations()
        obs = obs[0] if isinstance(obs, tuple) else obs
    settings = json.loads(args.settings) if args.settings else [{"name": "as_configured"}]
    slices = _imu_slices(u)
    print(f"policy terms {list(u.observation_manager.active_terms['policy'])} imu slices {{k: (v.start, v.stop) for k, v in slices.items()}}", flush=True)
    per_setting = []
    for si, setting in enumerate(settings):
        setting = dict(setting); setting.setdefault("seed", args.seed)
        try:
            applied = _apply_setting(u, setting)
            print(f"SETTING {setting.get('name')} applied {applied}", flush=True)
        except Exception:                                        # noqa: BLE001
            import traceback
            tb = traceback.format_exc()
            print("SETTING-ERROR\n" + tb, flush=True)             # stdout: Kit swallows stderr
            Path(args.output).with_suffix(".error.txt").write_text(tb)
            per_setting.append({"name": setting.get("name", f"setting{si}"), "error": tb[-800:]})
            print(json.dumps(per_setting[-1]), flush=True)
            continue
        try:
            if policy is None:
                obs, _ = env.reset()
            else:
                env.unwrapped.reset(seed=args.seed)
                obs = env.get_observations(); obs = obs[0] if isinstance(obs, tuple) else obs
            start = recovery.local_xy(u).clone()
            delay = _ImuDelay(slices, applied["imu_delay_steps"])
            r = _rollout(env, u, policy, obs, start, delay, args.steps)
        except Exception:                                        # noqa: BLE001
            import traceback
            tb = traceback.format_exc()
            print("ROLLOUT-ERROR\n" + tb, flush=True)
            Path(args.output).with_suffix(".error.txt").write_text(tb)
            per_setting.append({"name": setting.get("name", f"setting{si}"), "error": tb[-800:]})
            print(json.dumps(per_setting[-1]), flush=True)
            continue
        per_setting.append({"name": setting.get("name", f"setting{si}"), **applied,
                            "first_episodes_completed": int(r["first_finished"].sum()),
                            "first_episode_success_rate": float(r["first_success"].float().mean()),
                            "episodes_completed": r["completed"], "successes": r["success"],
                            "dead_ends": r["dead_end"], "timeouts": r["timeout"],
                            "min_goal_distance_m": float(r["minimum_dist"].min()),
                            "max_displacement_m": float(r["max_displacement"].max())})
        print(json.dumps(per_setting[-1]), flush=True)
    # The published single-setting fields refer to the first (or only) setting.
    r0 = per_setting[0]
    completed, success, dead_end, timeout = r0["episodes_completed"], r0["successes"], r0["dead_ends"], r0["timeouts"]
    first_finished = torch.tensor([r0["first_episodes_completed"] == u.num_envs] * u.num_envs, device=u.device)
    first_success = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    first_success[: round(r0["first_episode_success_rate"] * u.num_envs)] = True
    minimum_dist = torch.tensor([r0["min_goal_distance_m"]], device=u.device)
    max_displacement = torch.tensor([r0["max_displacement_m"]], device=u.device)
    result = {"settings": per_setting, "imu_terms": {k: [v.start, v.stop] for k, v in slices.items()},
              "task": args.task, "checkpoint": args.checkpoint, "seed": args.seed,
        "steps": args.steps, "num_envs": args.num_envs, "completed_episodes": completed,
        "successful_episodes": success, "success_rate": success / max(completed, 1),
        "first_episodes_completed": int(first_finished.sum()),
        "first_episode_success_rate": float(first_success.float().mean()),
        "dead_end_episodes": dead_end, "timeout_episodes": timeout,
        "mean_minimum_button_distance_m": float(minimum_dist.mean()),
        "mean_max_displacement_from_initial_spawn_m": float(max_displacement.mean()),
        "sensors": sensors, "gate_kind": "policy_evaluation" if policy else "integration_smoke",
        "navigation_source": "oracle_waypoint_teacher", "camera_source": "raycast_depth_not_rgb_stereo",
        "observation_corruption": observation_corruption}
    if policy and (not bool(first_finished.all()) or result["first_episode_success_rate"] < args.minimum_success):
        result["passed"] = False
    else:
        result["passed"] = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    env.close()
    if not result["passed"]:
        raise RuntimeError(f"Policy promotion rejected: first-success={result['first_episode_success_rate']:.3f}, "
                           f"first episodes={int(first_finished.sum())}/{args.num_envs}")
    print("MAZE_PROBE_PASS", flush=True)


try:
    run()
except BaseException:                                       # noqa: BLE001
    import traceback
    print("PROBE-ERROR\n" + traceback.format_exc(), flush=True)   # stdout: Kit swallows stderr and its close exits 0
    raise
finally:
    app.app.close()
