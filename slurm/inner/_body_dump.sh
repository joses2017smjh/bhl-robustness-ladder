#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Every body, by name, at its absolute height. Nothing derived.

Two probes disagree about which way this robot faces. `_updir.sh` reported the
torso 0.224 above the ankles under the corrected quaternion; `_v2_penetration.sh`
with planting on reported the ankles at +0.42 and the shoulders at -0.13, which
is feet above shoulders. One of them is measuring the wrong thing and no further
inference is going to settle which.

So this prints the whole list: every body, its name, its height above the
terrain, sorted. A standing robot has feet at the bottom and head at the top and
needs no interpretation to see it. Run for each candidate quaternion, with the
joint state forced and a sim step taken first -- writing joint_pos and reading
body_pos_w without stepping returns the previous kinematics, which has already
voided two measurements in this investigation.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True; a.enable_cameras = True
app = AppLauncher(a)

import gymnasium as gym, torch
import bhl_robust.tasks  # noqa: F401
from bhl_robust.tasks import coop_lift_env_cfg as C
from bhl_robust.tasks.coop_lift_mdp import _t

TASK = "TaskV2-BHL-CubeToShelf-Blind-v0"
CANDS = {
    "current _FACE_A (0.7071,-0.7071,0,0)": (0.70710678, -0.70710678, 0.0, 0.0),
    "identity        (1,0,0,0)":            (1.0, 0.0, 0.0, 0.0),
    "old yaw         (0.7071,0,0,-0.7071)": (0.7071, 0.0, 0.0, -0.7071),
}

for label, rot in CANDS.items():
    cfg = gym.spec(TASK).kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = 2
    for r in (cfg.scene.robot_a, cfg.scene.robot_b):
        r.init_state = r.init_state.replace(rot=rot, joint_pos=dict(C._PINCH_JOINT_POS))
    e = gym.make(TASK, cfg=cfg, disable_env_checker=True)
    e.reset()
    u = e.unwrapped
    rr = u.scene["robot_a"]

    # Force the pose through the sim, then step, then read.
    #
    # Setting init_state.rot and calling reset() is not enough: the reset event
    # pipeline rewrites the root pose, so two different quaternions came back
    # with body heights identical to four decimal places -- which is how this
    # probe was caught. Writing the state directly and stepping is the only
    # sequence in this investigation that has not lied.
    root = _t(rr.data.root_state_w).clone()
    root[:, 3:7] = torch.tensor(rot, device=root.device, dtype=root.dtype)
    rr.write_root_state_to_sim(root)
    jp = _t(rr.data.default_joint_pos).clone()
    names = rr.joint_names
    for j, v in C._PINCH_JOINT_POS.items():
        if j in names:
            jp[:, names.index(j)] = v
    rr.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    u.sim.step(render=False)
    rr.update(dt=0.0)

    q = _t(rr.data.root_quat_w)[0]
    print(f"\n  asked for quat {tuple(round(v,4) for v in rot)}")
    print(f"  sim reports    {tuple(round(float(v),4) for v in q)}"
          f"   {'MATCHES' if max(abs(float(q[k])-rot[k]) for k in range(4)) < 1e-3 else '<<< DID NOT APPLY'}")
    nm = rr.body_names
    b = _t(rr.data.body_pos_w)[0]
    org = u.scene.env_origins[0]
    z = [(float(b[i, 2] - org[2]), nm[i]) for i in range(len(nm))]
    z.sort()
    print(f"\n{'='*64}\n{label}\n{'='*64}")
    print(f"  root z above terrain: {float(_t(rr.data.root_pos_w)[0][2] - org[2]):+.4f}")
    print(f"  {'height':>9}  body")
    for h, n in z:
        mark = "  <-- BELOW GROUND" if h < 0 else ""
        print(f"  {h:+9.4f}  {n}{mark}")
    below = sum(1 for h, _ in z if h < 0)
    print(f"  --> {below} of {len(z)} below ground; "
          f"lowest {z[0][1]} at {z[0][0]:+.3f}, highest {z[-1][1]} at {z[-1][0]:+.3f}")
    e.close()
app.app.close()
PY
