#!/usr/bin/env python3
"""Why does the cloth-sort robot fall in half a second?

`21233958` passed its training gate (mean episode length 12.8 against 1.0) and
then reported `Episode_Termination/fallen = 1.0000`: every episode ends with
the robot toppling at ~0.51 s, and `Episode_Reward/progress` never leaves
zero. Decision rule 1 in docs/CLOTH_SORT.md says fix the controller or the
task geometry before training anything, so this measures *how* it goes down
before anything is changed.

Three poses, same scene, zero action throughout:

  configured  the pinch/squat the task spawns with (hips -0.85, knees 1.45)
  default     the asset's own default joint pose, no squat
  half        the squat halved

For each it reports tilt from the spawn orientation, root height, and how many
bodies are below the floor, at reset and every few steps. No policy, no
checkpoint. It answers whether the robot cannot *hold* the squat or was never
standing to begin with.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--report-every", type=int, default=10)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import bhl_robust.tasks  # noqa: F401,E402
from bhl_robust.cloth.kinematics import relative_up_z  # noqa: E402
from bhl_robust.tasks.cloth_sort_env_cfg import build_cfg  # noqa: E402

TASK = "ClothSort-BHL-Rigid-Oracle-v0"

#: Leg joints only. The arms hold the pinch pose in every variant so the only
#: thing changing is how deep the robot is crouching.
_LEGS = (
    "leg_left_hip_pitch_joint", "leg_right_hip_pitch_joint",
    "leg_left_knee_pitch_joint", "leg_right_knee_pitch_joint",
    "leg_left_ankle_pitch_joint", "leg_right_ankle_pitch_joint",
)


def _clear() -> None:
    try:
        from isaaclab.sim import SimulationContext
        SimulationContext.clear_instance()
    except Exception:
        pass


def _cpu(x) -> torch.Tensor:
    """Everything onto the CPU as float before any arithmetic.

    Isaac Lab 3.x hands some of these back as warp ProxyArrays and some as cuda
    tensors; mixing them raises "Expected all tensors to be on the same device".
    """
    return torch.as_tensor(x[:]).detach().float().cpu()


def _measure(env) -> dict:
    art = env.unwrapped.scene["robot"]
    q = _cpu(art.data.root_quat_w)[:, :4]
    q0 = _cpu(art.data.default_root_state)[:, 3:7]
    up = relative_up_z(q, q0).clamp(-1.0, 1.0)
    tilt = torch.acos(up)
    origin_z = _cpu(env.unwrapped.scene.env_origins)[:, 2]
    root_z = _cpu(art.data.root_pos_w)[:, 2] - origin_z
    body = _cpu(art.data.body_pos_w)[..., 2] - origin_z.unsqueeze(1)
    below = (body < -1e-3).sum(dim=1)
    return {
        "tilt_rad": float(tilt.mean()),
        "tilt_max": float(tilt.max()),
        "root_z": float(root_z.mean()),
        "bodies_below_floor": float(below.float().mean()),
        "n_bodies": int(body.shape[1]),
    }


def _run(label: str, scale: float) -> dict:
    _clear()
    cfg = build_cfg(
        gym.spec(TASK).kwargs["env_cfg_entry_point"],
        num_envs=args_cli.num_envs, device=app_launcher.device,
    )
    jp = dict(cfg.scene.robot.init_state.joint_pos)
    for j in _LEGS:
        if j in jp:
            jp[j] = jp[j] * scale
    cfg.scene.robot.init_state.joint_pos = jp
    # Zero action holds the pose; nothing here is a policy.
    env = gym.make(TASK, cfg=cfg, disable_env_checker=True)
    env.reset()
    act = torch.zeros(
        (args_cli.num_envs, env.unwrapped.action_space.shape[-1]),
        device=env.unwrapped.device,
    )
    track = [{"step": 0, **_measure(env)}]
    for i in range(1, args_cli.steps + 1):
        env.step(act)
        if i % args_cli.report_every == 0 or i == args_cli.steps:
            track.append({"step": i, **_measure(env)})
    env.close()
    row = {
        "pose": label,
        "leg_scale": scale,
        "squat_joint_pos": {j: jp.get(j) for j in _LEGS},
        "track": track,
        "final_tilt_rad": track[-1]["tilt_rad"],
        "fell": track[-1]["tilt_rad"] > 0.78,
    }
    print(f"\n--- {label} (leg scale {scale}) ---")
    for t in track:
        print(f"  step {t['step']:3d}  tilt {t['tilt_rad']:.3f} rad  "
              f"root_z {t['root_z']:+.3f}  below {t['bodies_below_floor']:.1f}/{t['n_bodies']}")
    print(f"  fell (>0.78 rad): {row['fell']}")
    return row


def main() -> None:
    rows = []
    for label, scale in (("configured_squat", 1.0), ("half_squat", 0.5), ("default_pose", 0.0)):
        try:
            rows.append(_run(label, scale))
        except Exception as e:
            import traceback
            traceback.print_exc()
            rows.append({"pose": label, "leg_scale": scale, "error": f"{type(e).__name__}: {e}"[:200]})

    print(f"\n{'pose':20} {'final tilt':>11}  {'root_z':>8}  fell")
    for r in rows:
        if "error" in r:
            print(f"{r['pose']:20} {r['error']}")
            continue
        print(f"{r['pose']:20} {r['final_tilt_rad']:>10.3f}  "
              f"{r['track'][-1]['root_z']:>+8.3f}  {r['fell']}")

    out = args_cli.out or os.environ.get("BENCH_OUT")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
