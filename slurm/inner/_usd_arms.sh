#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Is the arm asymmetry in the shipped USD, or in every USD from this URDF?

MuJoCo builds from the upstream URDF and gets hands symmetric to four decimal
places. Isaac, same joint values, puts them 39 cm apart. If our own URDF->USD
conversion (assets/gripper, written by scripts/add_gripper.py) is symmetric,
then the shipped USD is the problem and the fix is to convert it ourselves. If
it is asymmetric too, the converter is, and the fix is upstream of both.

Measured with the arms at zero, where a symmetric robot must be symmetric no
matter what any joint convention is.
"""
import argparse, math
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True
app = AppLauncher(a)

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_CFG

CANDIDATES = {"shipped berkeley_humanoid_lite_assets": HUMANOID_LITE_CFG}
try:
    from bhl_robust.gripper_asset import get_gripper_cfg
    CANDIDATES["our conversion (assets/gripper)"] = get_gripper_cfg()
except Exception as exc:
    print(f"(gripper cfg unavailable: {exc!r})")

sim = SimulationContext(SimulationCfg(dt=0.005, device="cuda:0"))
for slot, (label, base) in enumerate(CANDIDATES.items()):
    cfg = base.replace(prim_path=f"/World/probe{slot}")
    cfg.init_state = cfg.init_state.replace(
        pos=(0.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={**dict(base.init_state.joint_pos), **{f"arm_{s}_{k}_joint": 0.0
                   for s in ("left", "right")
                   for k in ("shoulder_pitch","shoulder_roll","shoulder_yaw",
                             "elbow_pitch","elbow_roll")}})
    art = Articulation(cfg)
    sim.reset()
    art.update(dt=0.0)
    nm, b = art.body_names, art.data.body_pos_w
    b = b.torch if hasattr(b, "torch") else b
    root = art.data.root_pos_w; root = root.torch if hasattr(root, "torch") else root
    def q(sub):
        i = nm.index(sub)
        return (float(b[0, i, 2] - root[0, 2]),
                math.hypot(float(b[0, i, 0] - root[0, 0]), float(b[0, i, 1] - root[0, 1])))
    lz, lr = q("arm_left_hand_link"); rz, rr = q("arm_right_hand_link")
    print(f"\n=== {label} ===")
    print(f"  arms at ZERO -> left hand  z={lz:+.4f} r={lr:.4f}")
    print(f"                  right hand z={rz:+.4f} r={rr:.4f}")
    print(f"  |dz| = {abs(lz-rz):.4f}   |dr| = {abs(lr-rr):.4f}"
          f"   {'SYMMETRIC' if abs(lz-rz) < 0.01 else '<<< ASYMMETRIC'}")
app.close()
PY
