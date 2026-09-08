#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Which way is up, measured where nothing can push back.

Every in-task attempt at this has been contaminated. Spawned 0.8 m underground,
PhysX's penetration response dominates the first step and washes the initial
orientation out -- two rotations 90 degrees apart came back with body heights
matching to 4 mm, twice, which is the tell that the measurement is not measuring
the rotation.

So: a bare articulation, five metres up, nothing to collide with. Free fall does
not change orientation, so body height *relative to the root* is pure kinematics.
A standing robot reads ankles at the bottom and shoulders at the top of that
list, and it needs no interpretation.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True
app = AppLauncher(a)

import torch
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_CFG
from bhl_robust.tasks import coop_lift_env_cfg as C

CANDS = {
    "identity          (1,0,0,0)":          (1.0, 0.0, 0.0, 0.0),
    "current _FACE_A   (0.7071,-0.7071,0,0)": (0.70710678, -0.70710678, 0.0, 0.0),
    "+90 about x       (0.7071,+0.7071,0,0)": (0.70710678, 0.70710678, 0.0, 0.0),
    "old yaw           (0.7071,0,0,-0.7071)": (0.7071, 0.0, 0.0, -0.7071),
    "180 about x       (0,1,0,0)":          (0.0, 1.0, 0.0, 0.0),
}

sim = SimulationContext(SimulationCfg(dt=0.005, device="cuda:0"))
arts = {}
for i, (label, rot) in enumerate(CANDS.items()):
    cfg = HUMANOID_LITE_CFG.replace(prim_path=f"/World/cand{i}")
    cfg.init_state = cfg.init_state.replace(
        pos=(i * 3.0, 0.0, 5.0), rot=rot, joint_pos=dict(C._PINCH_JOINT_POS))
    arts[label] = Articulation(cfg)
sim.reset()

for label, art in arts.items():
    jp = art.data.default_joint_pos.clone()
    names = art.joint_names
    for j, v in C._PINCH_JOINT_POS.items():
        if j in names:
            jp[:, names.index(j)] = v
    art.write_joint_state_to_sim(jp, torch.zeros_like(jp))
sim.step(render=False)

KEY = ("ankle_roll", "knee_pitch", "hip_pitch", "base", "shoulder_pitch", "hand_link")
print(f"\n{'candidate':34} " + " ".join(f"{k:>14}" for k in KEY))
for label, art in arts.items():
    art.update(dt=0.0)
    nm = art.body_names
    b = art.data.body_pos_w; b = b.torch if hasattr(b, "torch") else b
    r = art.data.root_pos_w; r = r.torch if hasattr(r, "torch") else r
    row, vals = [], {}
    for k in KEY:
        idx = [i for i, n in enumerate(nm) if k in n]
        v = float(b[0, idx, 2].mean() - r[0, 2]) if idx else float("nan")
        vals[k] = v
        row.append(f"{v:+14.4f}")
    upright = vals["ankle_roll"] < vals["hip_pitch"] < vals["shoulder_pitch"]
    print(f"{label:34} " + " ".join(row) + ("   UPRIGHT" if upright else "   not upright"))
print("\nUPRIGHT means ankles below hips below shoulders, relative to the root.")
app.close()
PY
