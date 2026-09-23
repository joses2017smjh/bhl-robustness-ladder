"""Finite Isaac maze smoke / checkpoint rollout with real navigation metrics.

Run inside the existing v60 bhl_exec environment. Without --checkpoint this is
an integration smoke only; it cannot certify navigation success. With one,
--minimum-success is the explicit promotion gate. Writes JSON only at the end
and prints MAZE_PROBE_PASS, so an Isaac shutdown exit-code bug cannot pass it.
"""
from __future__ import annotations

import argparse
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
    completed = success = dead_end = timeout = 0
    first_finished = torch.zeros(args.num_envs, dtype=torch.bool, device=u.device)
    first_success = torch.zeros_like(first_finished)
    minimum_dist = torch.full((args.num_envs,), float("inf"), device=u.device)
    max_displacement = torch.zeros(args.num_envs, device=u.device)
    for _ in range(args.steps):
        with torch.inference_mode():
            action = policy(obs) if policy else torch.zeros((args.num_envs, u.action_manager.total_action_dim), device=u.device)
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
    result = {"task": args.task, "checkpoint": args.checkpoint, "seed": args.seed,
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
finally:
    app.app.close()
