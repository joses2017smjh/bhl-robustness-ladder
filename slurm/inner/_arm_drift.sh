#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Do the arms actually hold PINCH_POSE in the task, or do they sag out of it?

The bare articulation under PINCH_POSE puts both hands at -0.6069 relative to
the root, symmetric to four decimal places. The same pose inside the coop task
measures -0.184 and +0.212. Neither task hand is where the pose puts them, which
means the arms are not in PINCH_POSE by the time anything reads them.

So the question is not "why is the asset asymmetric" -- it is not -- but "what
moves the arms between reset and the first measurement". This prints the actual
joint angles against the commanded ones.
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
cfg = gym.spec(TASK).kwargs["env_cfg_entry_point"]()
cfg.scene.num_envs = 4
env = gym.make(TASK, cfg=cfg, disable_env_checker=True)
env.reset(); u = env.unwrapped
r = u.scene["robot_a"]
names = r.joint_names
ARMS = [n for n in names if n.startswith("arm_")]

def show(tag):
    r.update(dt=0.0)
    q = _t(r.data.joint_pos)[0]
    print(f"\n--- {tag} ---")
    print(f"{'joint':34} {'commanded':>10} {'actual':>10} {'drift':>9}")
    worst = 0.0
    for n in ARMS:
        want = C._PINCH_JOINT_POS.get(n, 0.0)
        got = float(q[names.index(n)])
        d = got - want
        worst = max(worst, abs(d))
        flag = "  <<<" if abs(d) > 0.15 else ""
        print(f"{n:34} {want:+10.3f} {got:+10.3f} {d:+9.3f}{flag}")
    print(f"  worst arm drift: {worst:.3f} rad")

def hands(tag):
    """Hand height relative to root, with the pose forced and no jitter.

    Bare articulation under the same pose gives -0.6069 for both hands. If the
    task disagrees while the joint angles agree, the difference is in the frames
    and not in the pose -- which is the only place left for it to be.
    """
    jp = _t(r.data.default_joint_pos).clone()
    for j, v in C._PINCH_JOINT_POS.items():
        if j in names:
            jp[:, names.index(j)] = v
    r.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    r.update(dt=0.0)
    b = _t(r.data.body_pos_w)[0]
    root = _t(r.data.root_pos_w)[0]
    q = _t(r.data.root_quat_w)[0]
    nm = r.body_names
    li, ri = nm.index("arm_left_hand_link"), nm.index("arm_right_hand_link")
    lz, rz = float(b[li,2]-root[2]), float(b[ri,2]-root[2])
    print(f"\n=== {tag}: hands relative to root, pose forced, no jitter ===")
    print(f"  left  {lz:+.4f}    right {rz:+.4f}    |dz| {abs(lz-rz):.4f}")
    print(f"  bare articulation, same pose:  -0.6069 / -0.6069 / 0.0000")
    print(f"  root quat (w,x,y,z) = {[round(float(x),4) for x in q]}")
    print(f"  up_z = {1.0 - 2.0*(float(q[1])**2 + float(q[2])**2):.4f}")

hands("in-task robot_a")
show("immediately after env.reset()")
zero = torch.zeros((4, u.action_space.shape[-1]), device=u.device)
for i in range(5):
    env.step(zero)
show("after 5 zero-action steps")

print("\nArm actuators are 4 Nm, kp 10 (from the asset config). If the drift is"
      "\nlarge and downward, the arms cannot hold this pose against gravity and"
      "\nthe pinch pose is not a spawn bug but an unholdable target.")
app.app.close()
PY
