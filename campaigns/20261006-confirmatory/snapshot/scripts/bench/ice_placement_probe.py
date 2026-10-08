"""Is the ice where the robots are? Measured on the built B3 scene.

G-B3 proved the patches are flush -- analytically, from the constants. It never
asked where they are. The patches used to be `AssetBaseCfg` under `{ENV_REGEX_NS}`, so each sat at
its env's *grid* origin; with a terrain generator Isaac Lab resets each robot
onto a *terrain* origin instead (`scene.env_origins`), and GPU collision
filtering lets a robot touch only its own env's prims. That is `21328532`: a
median 72 m gap. They are kinematic rigid bodies now, teleported onto
`env_origins` at reset.

For every env this reads, from the live scene:

  robot    root xy after reset                    (where the robot is)
  patches  physics root xy of that env's ice      (not USD: fabric can lie)
  reach    episode length x the largest commanded speed

and reports how far each robot starts from its nearest own patch, and the
fraction of robots that could reach one inside an episode at full commanded
speed, walking straight at it -- a generous upper bound.

Verdict line, grepped by the batch script:
  ICE-PLACEMENT REACHABLE    most robots can reach their own ice
  ICE-PLACEMENT UNREACHABLE  most cannot: B3 trained on bumpy ground
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Velocity-BHL-Biped-Ice-Depth-v0")
parser.add_argument("--num_envs", type=int, default=4096)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import bhl_robust.tasks  # noqa: F401,E402


def main() -> None:
    cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=cfg, disable_env_checker=True)
    u = env.unwrapped
    env.reset()

    names = sorted(k for k in u.scene.keys() if str(k).startswith("ice_"))
    if not names:
        raise SystemExit("no ice_* assets in the built scene")
    n = u.num_envs
    robot_xy = u.scene["robot"].data.root_pos_w[:, :2].cpu()
    terrain_xy = u.scene.env_origins[:, :2].cpu()
    grid = getattr(u.scene, "_default_env_origins", None)
    grid_xy = grid[:, :2].cpu() if grid is not None else None

    # Physics pose, not the USD xform: fabric can leave USD at the clone-grid
    # origin after write_root_pose_to_sim, which is how a working placement
    # would still look like the 72 m bug.
    patches = torch.stack(
        [u.scene[name].data.root_pos_w[:, :2].cpu() for name in names], dim=1)
    print(f"ice assets {names}  type {type(u.scene[names[0]]).__name__}")

    nearest = (patches - robot_xy[:, None, :]).norm(dim=-1).min(dim=1).values
    vx = u.cfg.commands.base_velocity.ranges.lin_vel_x
    vy = u.cfg.commands.base_velocity.ranges.lin_vel_y
    vmax = max(abs(vx[0]), abs(vx[1])) + max(abs(vy[0]), abs(vy[1]))
    reach = float(u.cfg.episode_length_s) * vmax
    half = float(u.cfg.scene.ice_0.spawn.size[0]) / 2.0
    reachable = (nearest <= reach + half).float().mean().item()
    on_ice = (nearest <= half).float().mean().item()

    print(f"task {args_cli.task}  envs {n}  patches per env {len(names)}  "
          f"collision filtering {u.cfg.scene.filter_collisions}")
    print(f"robot->terrain origin, mean {float((robot_xy - terrain_xy).norm(dim=-1).mean()):.2f} m")
    if grid_xy is not None:
        own = (patches[:, 0, :] - grid_xy).norm(dim=-1)
        print(f"patch 0 -> its env grid origin, mean {float(own.mean()):.2f} m "
              f"(config offset {tuple(u.cfg.scene.ice_0.init_state.pos[:2])})")
        print(f"terrain origin -> grid origin, median {float((terrain_xy - grid_xy).norm(dim=-1).median()):.1f} m")
    q = torch.quantile(nearest, torch.tensor([0.1, 0.5, 0.9]))
    print(f"robot -> nearest own patch: p10 {q[0]:.1f} m  median {q[1]:.1f} m  p90 {q[2]:.1f} m")
    print(f"reach bound {reach:.1f} m ({u.cfg.episode_length_s:.0f} s x {vmax:.2f} m/s)  "
          f"on ice at spawn {on_ice:.3f}  could reach own ice {reachable:.3f}")
    verdict = "REACHABLE" if reachable >= 0.5 else "UNREACHABLE"
    print(f"ICE-PLACEMENT {verdict}  median distance {q[1]:.1f} m, reach {reach:.1f} m, "
          f"reachable fraction {reachable:.3f}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
