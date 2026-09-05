#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Which arm joint does Isaac mirror differently from MuJoCo?

MuJoCo mirrors every arm joint with opposite values, which is what PINCH_POSE
uses, and produces hands symmetric to four decimal places. Isaac, same values,
puts them 40 cm apart. This drives one pair at a time with everything else at
zero, so the offender is named rather than inferred.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True; a.enable_cameras = True
app = AppLauncher(a)

import gymnasium as gym, torch
import bhl_robust.tasks  # noqa: F401
from bhl_robust.tasks.coop_lift_mdp import _t

TASK = "TaskV2-BHL-CubeToShelf-Blind-v0"
PAIRS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll"]
# Magnitudes legal on both sides with either sign. shoulder_roll is
# [-0.262, +1.309] on the left and the mirror of that on the right, so 0.25 is
# the largest that stays valid for all four combinations.
THETA = {"shoulder_pitch": 0.6, "shoulder_roll": 0.25, "shoulder_yaw": 0.6,
         "elbow_pitch": 0.6, "elbow_roll": 0.6}

def measure(pose):
    cfg = gym.spec(TASK).kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = 2
    for r in (cfg.scene.robot_a, cfg.scene.robot_b):
        r.init_state = r.init_state.replace(joint_pos=dict(pose),
                                            pos=(r.init_state.pos[0], r.init_state.pos[1], 0.6))
    e = gym.make(TASK, cfg=cfg, disable_env_checker=True); e.reset()
    rr = e.unwrapped.scene["robot_a"]
    b = _t(rr.data.body_pos_w)[0]; nm = rr.body_names
    root = _t(rr.data.root_pos_w)[0]
    li, ri = nm.index("arm_left_hand_link"), nm.index("arm_right_hand_link")
    # z is yaw-invariant; so is horizontal distance from the root. A mirrored
    # pair matches on both. World y does not work here -- the robots carry a
    # 90 deg yaw, so the lateral axis is not the world's.
    import math
    rad = lambda i: math.hypot(float(b[i,0]-root[0]), float(b[i,1]-root[1]))
    out = (float(b[li,2]-root[2]), float(b[ri,2]-root[2]), rad(li), rad(ri))
    e.close(); return out

zero = {j: 0.0 for j in
        [f"arm_{s}_{k}_joint" for s in ("left","right") for k in
         ("shoulder_pitch","shoulder_roll","shoulder_yaw","elbow_pitch","elbow_roll")]}
print(f"{'joint':16} {'signs':>9}  {'Lz':>8} {'Rz':>8} {'|dz|':>7}  {'Lr':>8} {'Rr':>8}  {'|dr|':>7}")
for key in PAIRS:
    res = {}
    theta = THETA.get(key, 0.25)
    for label, rv in (("same", +theta), ("opposite", -theta)):
        pose = dict(zero)
        pose[f"arm_left_{key}_joint"] = theta
        pose[f"arm_right_{key}_joint"] = rv
        try:
            lz, rz, ly, ry = measure(pose)
        except Exception as exc:
            print(f"{key:16} {label:>9}  invalid: {str(exc)[:44]}"); continue
        # mirror symmetry: same height, opposite lateral offset
        print(f"{key:16} {label:>9}  {lz:8.4f} {rz:8.4f} {abs(lz-rz):7.4f}"
              f"  {ly:8.4f} {ry:8.4f}  {abs(ly-ry):7.4f}")
        res[label] = abs(lz-rz) + abs(ly-ry)
    if res:
        win = min(res, key=res.get)
        print(f"{'':16} -> Isaac mirrors with {win.upper()}  "
              f"(score {res[win]:.4f} vs {max(res.values()):.4f})   "
              f"MuJoCo says OPPOSITE\n")
app.app.close()
PY
