#!/usr/bin/env python3
"""Where are the hands, in the Isaac cloth scene, relative to table and garment?

Two things the cloth layout has been assuming without a measurement:

1. **Which way the robot faces.** The scene puts the table on +x and says the
   robot faces +x. It spawns under ``(0, 0, 1, 0)``, and the spawn record says a
   heading cannot be set on that pose -- so the facing direction is whatever
   that quaternion produces, and nobody has checked it against the table.
2. **How far the hand is from the garment.** MuJoCo FK over the full arm range
   puts the right hand's furthest forward point at 0.29 m from the root, and its
   lowest reach in the pinch squat at z = 0.339 -- above a 0.30 m table top.
   The garment spawns 0.74 m away. This reads the same distances in Isaac, from
   the articulation itself, so the redesign is placed in the frame the robot
   actually has rather than the one the config assumes.

No policy and no checkpoint: construct, reset, read, one zero-action step, read.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import bhl_robust.tasks  # noqa: F401,E402
from bhl_robust.cloth import layout as L  # noqa: E402
from bhl_robust.tasks.cloth_sort_env_cfg import build_cfg  # noqa: E402

TASK = "ClothSort-BHL-Rigid-Oracle-v0"


def _cpu(x) -> torch.Tensor:
    return torch.as_tensor(x[:]).detach().float().cpu()


def _snapshot(env, label: str) -> dict:
    scene = env.unwrapped.scene
    art = scene["robot"]
    names = list(art.body_names)
    origins = _cpu(scene.env_origins)
    root = _cpu(art.data.root_pos_w)[:, :3] - origins
    body = _cpu(art.data.body_pos_w) - origins.unsqueeze(1)
    garment = _cpu(scene["garment_0"].data.root_pos_w)[:, :3] - origins
    li, ri = names.index("arm_left_hand_link"), names.index("arm_right_hand_link")
    hands = body[:, [li, ri]]
    rows = []
    for e in range(root.shape[0]):
        lh, rh, rt, g = hands[e, 0], hands[e, 1], root[e], garment[e]
        mid = 0.5 * (lh + rh)
        rows.append({
            "root": [round(float(v), 4) for v in rt],
            "left_hand": [round(float(v), 4) for v in lh],
            "right_hand": [round(float(v), 4) for v in rh],
            "garment": [round(float(v), 4) for v in g],
            # Hands in front of the root along +x means the robot faces the table.
            "hands_minus_root_x": round(float(mid[0] - rt[0]), 4),
            "hands_minus_root_y": round(float(mid[1] - rt[1]), 4),
            "right_hand_to_garment_xy": round(float(torch.linalg.norm(rh[:2] - g[:2])), 4),
            "right_hand_above_table": round(float(rh[2] - L.TABLE_TOP_Z), 4),
        })
    e0 = rows[0]
    print(f"\n--- {label} (env 0 of {len(rows)}) ---")
    for k, v in e0.items():
        print(f"  {k:28} {v}")
    return {"label": label, "envs": rows}


def main() -> None:
    cfg = build_cfg(gym.spec(TASK).kwargs["env_cfg_entry_point"],
                    num_envs=args_cli.num_envs, device=app_launcher.device)
    env = gym.make(TASK, cfg=cfg, disable_env_checker=True)
    env.reset()
    snaps = [_snapshot(env, "after reset")]
    act = torch.zeros((args_cli.num_envs, env.unwrapped.action_space.shape[-1]),
                      device=env.unwrapped.device)
    env.step(act)
    snaps.append(_snapshot(env, "after one zero-action step"))
    env.close()

    s = snaps[-1]["envs"]
    fx = [r["hands_minus_root_x"] for r in s]
    reach = [r["right_hand_to_garment_xy"] for r in s]
    verdict = {
        "layout_assumes": {"robot_xy": L.ROBOT_XY, "table_center_xy": L.TABLE_CENTER_XY,
                           "table_top_z": L.TABLE_TOP_Z},
        "faces_plus_x": all(v > 0 for v in fx),
        "faces_minus_x": all(v < 0 for v in fx),
        "mean_hands_minus_root_x": round(sum(fx) / len(fx), 4),
        "mean_right_hand_to_garment_xy": round(sum(reach) / len(reach), 4),
    }
    print("\n=== verdict ===")
    for k, v in verdict.items():
        print(f"  {k:32} {v}")
    out = args_cli.out or os.environ.get("BENCH_OUT")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"verdict": verdict, "snapshots": snaps}, indent=2))
        print(f"wrote {out}")
    print("hand probe done")


if __name__ == "__main__":
    main()
    simulation_app.close()
