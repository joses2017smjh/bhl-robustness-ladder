#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Which body axis is up on this asset, and what does the fall check see?

The corrected spawn quats (0.7071, -+0.7071, 0, 0) make the hands symmetric and
survive reset -- verified, 21192782 -- and they also make training end every
episode on step one. `_tilt_from_quat` uses R[2,2] = 1 - 2(x^2+y^2), which
assumes the robot's up axis is body z. Under the corrected quat that evaluates
to 0, i.e. 1.5708 rad of tilt, past the 0.78 limit, at spawn.

If the asset is authored Y-up then R[2,2] is the wrong row and the tilt should
come from the body y axis instead -- and `projected_gravity_b` returning
[0, +-1, 0], which this repo recorded as an Isaac Lab 3.x defect, was correct.

Decides it from geometry rather than convention: with the robot spawned under
each candidate quat, is the head above the feet in world z?
"""
import argparse, math
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True; a.enable_cameras = True
app = AppLauncher(a)

import gymnasium as gym, torch
import bhl_robust.tasks  # noqa: F401
from bhl_robust.tasks import coop_lift_env_cfg as C
from bhl_robust.tasks.coop_lift_mdp import _t

TASK = "TaskV2-BHL-PlankToWall-Blind-v0"
# Rotation about x behaves as yaw on this Y-up asset: the cube pair uses
# -+90 deg about x and both stand. The plank pair faces along -+x instead of
# -+y, so it wants the two headings 90 deg away from those -- 0 and 180 about x.
# Measured rather than derived, because deriving is what cost five hypotheses.
CANDS = {
    "plank old a (1, 0, 0, 0)":      (1.0, 0.0, 0.0, 0.0),
    "plank old b (0, 0, 0, 1)":      (0.0, 0.0, 0.0, 1.0),
    "cand 180x   (0, 1, 0, 0)":      (0.0, 1.0, 0.0, 0.0),
    "cand -90x   (0.7071,-0.7071,0,0)": (0.70710678, -0.70710678, 0.0, 0.0),
}
print(f"{'spawn quat':34} {'R22':>7} {'R21':>7} {'tilt_now':>9} {'head-foot':>10} {'|dz|hands':>10}")
for label, rot in CANDS.items():
    cfg = gym.spec(TASK).kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = 2
    for r in (cfg.scene.robot_a, cfg.scene.robot_b):
        r.init_state = r.init_state.replace(rot=rot, joint_pos=dict(C._PINCH_JOINT_POS))
    e = gym.make(TASK, cfg=cfg, disable_env_checker=True); e.reset()
    rr = e.unwrapped.scene["robot_a"]; rr.update(dt=0.0)
    q = _t(rr.data.root_quat_w)[0]
    w, x, y, z = (float(v) for v in q)
    R22 = 1.0 - 2.0 * (x * x + y * y)          # world-z of body z
    R21 = 2.0 * (y * z + w * x)                # world-z of body y
    b = _t(rr.data.body_pos_w)[0]; nm = rr.body_names
    # geometry, not convention: is the torso above the ankles?
    foot = min(float(b[i, 2]) for i, n in enumerate(nm) if "ankle" in n)
    head = float(b[nm.index("base"), 2])
    li, ri = nm.index("arm_left_hand_link"), nm.index("arm_right_hand_link")
    dz = abs(float(b[li, 2]) - float(b[ri, 2]))
    print(f"{label:34} {R22:+7.3f} {R21:+7.3f} {math.acos(max(-1,min(1,R22))):9.4f} "
          f"{head-foot:+10.3f} {dz:10.4f}")
    e.close()
print("\nhead-foot > 0 means the torso is above the ankles, i.e. actually upright.")
print("tilt_now is what either_fallen currently computes; the limit is 0.78.")
app.app.close()
PY
