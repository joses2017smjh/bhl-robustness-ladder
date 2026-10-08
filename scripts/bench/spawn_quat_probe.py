"""CubeToShelf spawn quaternions on the running Isaac Lab.

The cameras already go through ``native_quat``. Robot ``init_state.rot`` now
does too (documented yaw -90 / +90). Status 2026-09-16: ``21342562`` found
three STANDING 4-tuples; as-configured was BURIED. This re-probe is the gate
that as_configured stands after that wrap.

This does not change the task. It resets the configured env, reports that
spawn, then overwrites each robot's root rotation with a candidate and reports
bodies-below-ground and R[2,2] after two physics steps.

Verdict line, grepped by the batch script:
  SPAWN-QUAT <name> STANDING
  SPAWN-QUAT <name> BURIED
"""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="TaskV2-BHL-CubeToShelf-Blind-v0")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

import bhl_robust.tasks  # noqa: F401,E402
from bhl_robust.quat_order import native_quat, quat_order  # noqa: E402
from bhl_robust.tasks import coop_lift_env_cfg as coop  # noqa: E402

S2 = 0.70710678
# Literals are (w, x, y, z), the repo convention.
CANDIDATES = (
    ("as_configured", None, False),
    ("current_native", (S2, -S2, 0.0, 0.0), True),   # today's _YAW_M90, reordered
    ("current_raw", (S2, -S2, 0.0, 0.0), False),
    ("legacy_yaw_native", (S2, 0.0, 0.0, -S2), True),  # documented yaw -90
    ("legacy_yaw_raw", (S2, 0.0, 0.0, -S2), False),
    ("stand_up_native", (0.0, 0.0, 1.0, 0.0), True),
    ("stand_up_raw", (0.0, 0.0, 1.0, 0.0), False),
    ("identity_native", (1.0, 0.0, 0.0, 0.0), True),
    ("identity_raw", (1.0, 0.0, 0.0, 0.0), False),
)


def _measure(robot) -> dict:
    bodies = robot.data.body_pos_w[0]
    root = robot.data.root_pos_w[0]
    quat = robot.data.root_quat_w[0]
    r22 = float(matrix_from_quat(quat.unsqueeze(0))[0, 2, 2])
    z = bodies[:, 2]
    return {
        "root_z": float(root[2]),
        "min_z": float(z.min()),
        "n_below": int((z < 0).sum().item()),
        "n_bodies": int(z.numel()),
        "R22": r22,
        "quat": [float(v) for v in quat],
    }


def _apply(robot, wxyz, native: bool, origins: torch.Tensor) -> tuple[float, ...]:
    q = native_quat(wxyz) if native else tuple(float(v) for v in wxyz)
    state = robot.data.default_root_state.clone()
    state[:, :3] = state[:, :3] + origins
    state[:, 3:7] = torch.tensor(q, device=state.device, dtype=state.dtype).expand_as(state[:, 3:7])
    robot.write_root_state_to_sim(state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    return q


def _standing(row: dict) -> bool:
    # The locomotion control has 1 of 27 bodies below (the base frame). A
    # buried spawn has 15–27. R22 ~ -1 is an upside-down torso.
    return row["n_below"] <= 2 and row["R22"] > 0.5 and row["min_z"] > -0.08


def main() -> None:
    cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=cfg, disable_env_checker=True)
    u = env.unwrapped
    env.reset()
    order = quat_order()
    print(f"task {args_cli.task}  envs {u.num_envs}  quat_order {order}")
    print(f"configured robot_a rot {tuple(cfg.scene.robot_a.init_state.rot)}")
    print(f"configured robot_b rot {tuple(cfg.scene.robot_b.init_state.rot)}")
    print(f"coop _YAW_M90 {coop._YAW_M90}  _YAW_P90 {coop._YAW_P90}")

    rows = []
    for name, wxyz, native in CANDIDATES:
        env.reset()
        applied = None
        if wxyz is not None:
            origins = u.scene.env_origins
            applied = _apply(u.scene["robot_a"], wxyz, native, origins)
            w, x, y, z = wxyz
            _apply(u.scene["robot_b"], (w, -x, y, -z), native, origins)
            for _ in range(2):
                u.sim.step()
            u.scene.update(u.sim.get_physics_dt())
        a = _measure(u.scene["robot_a"])
        b = _measure(u.scene["robot_b"])
        standing = _standing(a)
        tag = "STANDING" if standing else "BURIED"
        print(
            f"SPAWN-QUAT {name} {tag}  native={int(native)}  applied={applied}  "
            f"a n_below {a['n_below']}/{a['n_bodies']} min_z {a['min_z']:+.3f} "
            f"R22 {a['R22']:+.3f}  b n_below {b['n_below']}/{b['n_bodies']} "
            f"min_z {b['min_z']:+.3f} R22 {b['R22']:+.3f}",
            flush=True,
        )
        rows.append({"name": name, "wxyz": wxyz, "native": native,
                     "applied": applied, "standing": standing, "a": a, "b": b})

    out = os.environ.get("BENCH_OUT") or os.path.join(
        os.environ.get("REPO", "."), "results", "spawn_quat_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {"task": args_cli.task, "quat_order": order, "rows": rows}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
