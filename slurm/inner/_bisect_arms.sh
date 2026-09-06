#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Bisect the 39 cm hand split: is it the pose, the asset, or the task?

Known: the bare articulation at zero arm angles is symmetric to four decimal
places, and robot_a inside the coop task under PINCH_POSE is 39 cm apart. The
case between them was never measured -- the bare articulation under PINCH_POSE.
That single cell says whether the pose alone does it, or whether something the
task adds does.

MuJoCo, same URDF, same pose: +0.5804 and +0.5804. So one of these rows is
wrong about the robot, and the point is to find which.
"""
import argparse, math
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True
app = AppLauncher(a)

import torch
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_CFG
from bhl_robust.tasks import coop_lift_env_cfg as C

ZERO_ARMS = {f"arm_{s}_{k}_joint": 0.0 for s in ("left", "right")
             for k in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                       "elbow_pitch", "elbow_roll")}

CASES = {
    "bare, zero arms":            {**dict(HUMANOID_LITE_CFG.init_state.joint_pos), **ZERO_ARMS},
    "bare, PINCH_POSE":           dict(C._PINCH_JOINT_POS),
    "bare, PINCH arms only":      {**dict(HUMANOID_LITE_CFG.init_state.joint_pos),
                                   **{k: v for k, v in C._PINCH_JOINT_POS.items() if k.startswith("arm_")}},
    "bare, PINCH legs only":      {**dict(HUMANOID_LITE_CFG.init_state.joint_pos),
                                   **{k: v for k, v in C._PINCH_JOINT_POS.items() if k.startswith("leg_")}},
}

sim = SimulationContext(SimulationCfg(dt=0.005, device="cuda:0"))
arts = {}
for i, (label, jp) in enumerate(CASES.items()):
    cfg = HUMANOID_LITE_CFG.replace(prim_path=f"/World/case{i}")
    cfg.init_state = cfg.init_state.replace(
        pos=(i * 2.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos=jp)
    arts[label] = Articulation(cfg)
sim.reset()

# Writing joint_pos into init_state is not enough for a bare Articulation: the
# earlier version of this probe returned byte-identical numbers for "zero arms"
# and "PINCH_POSE", which is how it was caught -- both were reporting the USD
# default pose. The state has to be pushed to the sim explicitly, then a
# kinematics update run, before body_pos_w means anything.
for label, art in arts.items():
    jp = art.data.default_joint_pos.clone()
    names = art.joint_names
    for j, v in CASES[label].items():
        if j in names:
            jp[:, names.index(j)] = v
    art.write_joint_state_to_sim(jp, torch.zeros_like(jp))
sim.step(render=False)

print(f"\n{'case':26} {'Lz':>9} {'Rz':>9} {'|dz|':>8}  {'Lr':>8} {'Rr':>8}  verdict")
for label, art in arts.items():
    art.update(dt=0.0)
    nm = art.body_names
    b = art.data.body_pos_w; b = b.torch if hasattr(b, "torch") else b
    r = art.data.root_pos_w; r = r.torch if hasattr(r, "torch") else r
    def q(sub):
        i = nm.index(sub)
        return (float(b[0, i, 2] - r[0, 2]),
                math.hypot(float(b[0, i, 0] - r[0, 0]), float(b[0, i, 1] - r[0, 1])))
    lz, lr = q("arm_left_hand_link"); rz, rr = q("arm_right_hand_link")
    v = "SYMMETRIC" if abs(lz - rz) < 0.01 else "<<< SPLIT"
    print(f"{label:26} {lz:+9.4f} {rz:+9.4f} {abs(lz-rz):8.4f}  {lr:8.4f} {rr:8.4f}  {v}")

print("\nMuJoCo, same URDF, PINCH_POSE: Lz=+0.5804 Rz=+0.5804 |dz|=0.0000")
app.close()
PY
