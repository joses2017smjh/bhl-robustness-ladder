#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Is the coop spawn quaternion a yaw, or is it tipping the robot on its side?

Every arm joint shows the same ~0.5 m left/right hand split in Isaac, whatever
signs it is driven with, while the horizontal distances stay equal. A constant
offset that no joint controls is not a joint bug -- it is the base orientation.
`_robot` passes rot=(0.7071, 0, 0, -0.7071), which is a -90 deg yaw in Isaac
Lab's (w, x, y, z) convention and a roll onto the side in (x, y, z, w).

So: measure the same robot at several rotations and see which one stands up.
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

TASK = "TaskV2-BHL-CubeToShelf-Blind-v0"
ROTS = {
    "as configured  (0.7071,0,0,-0.7071)": (0.7071, 0.0, 0.0, -0.7071),
    "identity       (1,0,0,0)":            (1.0, 0.0, 0.0, 0.0),
    "+90 yaw wxyz   (0.7071,0,0,+0.7071)": (0.7071, 0.0, 0.0, 0.7071),
    "180 yaw        (0,0,0,1)":            (0.0, 0.0, 0.0, 1.0),
}
print(f"{'rotation':38} {'Lz':>8} {'Rz':>8} {'|dz|':>7} {'up_z':>7} {'minz':>8} {'under':>6}")
for label, rot in ROTS.items():
    cfg = gym.spec(TASK).kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = 2
    for r in (cfg.scene.robot_a, cfg.scene.robot_b):
        r.init_state = r.init_state.replace(
            joint_pos=dict(C._PINCH_JOINT_POS), rot=rot,
            pos=(r.init_state.pos[0], r.init_state.pos[1], 0.6))
    e = gym.make(TASK, cfg=cfg, disable_env_checker=True); e.reset()
    rr = e.unwrapped.scene["robot_a"]
    b = _t(rr.data.body_pos_w)[0]; nm = rr.body_names
    root = _t(rr.data.root_pos_w)[0]
    q = _t(rr.data.root_quat_w)[0]
    # world-z of the body's own up axis: R[2,2] = 1 - 2(x^2 + y^2)
    up_z = 1.0 - 2.0 * (float(q[1])**2 + float(q[2])**2)
    li, ri = nm.index("arm_left_hand_link"), nm.index("arm_right_hand_link")
    lz, rz = float(b[li,2]-root[2]), float(b[ri,2]-root[2])
    minz = float((b[:,2]-root[2]).min())
    print(f"{label:38} {lz:8.4f} {rz:8.4f} {abs(lz-rz):7.4f} {up_z:7.3f} {minz:8.4f}")
    e.close()
print("\nup_z = +1 is upright, 0 is on its side, -1 is upside down.")
app.app.close()
PY
