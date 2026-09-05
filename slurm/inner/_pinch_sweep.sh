#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Which pinch-pose signs actually produce a symmetric robot standing on the floor?

Reasoning from the URDF got this wrong once: every joint declares axis="0 0 1",
so the origin rotations looked like the mirroring convention, and by that reading
the elbows, knees and ankles were all backwards. Isaac rejected it -- the joint
*limits* are the authority, and they forced the elbow negative and the knee
positive, which is what the original pose already had.

So: dump the limits, work out which pairs are actually free to flip, and measure
every valid combination instead of arguing about it. The target is the geometry
the tasks were designed around -- hands level with each other, feet on the floor,
hands within reach of a payload at GRASP_Z = 0.30.
"""
import argparse, itertools
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True; a.enable_cameras = True
app = AppLauncher(a)

import gymnasium as gym, torch
import bhl_robust.tasks  # noqa: F401
from bhl_robust.tasks import coop_lift_env_cfg as C
from bhl_robust.tasks.coop_lift_mdp import _t

TASK = "TaskV2-BHL-CubeToShelf-Blind-v0"
POSED = ("shoulder_roll", "shoulder_pitch", "elbow_pitch",
         "hip_pitch", "knee_pitch", "ankle_pitch")

def build(pose, root_z):
    cfg = gym.spec(TASK).kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = 2
    for r in (cfg.scene.robot_a, cfg.scene.robot_b):
        r.init_state = r.init_state.replace(
            joint_pos=dict(pose), pos=(r.init_state.pos[0], r.init_state.pos[1], root_z))
    return gym.make(TASK, cfg=cfg, disable_env_checker=True)

# --- 1. the limits, which are the authority ---
env = build(C._PINCH_JOINT_POS, C._PINCH_ROOT_Z)
env.reset(); u = env.unwrapped
r = u.scene["robot_a"]
lim = _t(r.data.joint_pos_limits)[0]        # (joints, 2)
names = r.joint_names
print("=== joint limits for every posed joint ===")
free = []
for key in POSED:
    for side in ("left", "right"):
        pre = "arm" if key.startswith(("shoulder", "elbow")) else "leg"
        jn = f"{pre}_{side}_{key}_joint"
        if jn not in names: continue
        lo, hi = lim[names.index(jn)].tolist()
        both = lo < 0.0 < hi
        print(f"  {jn:38} [{lo:+.3f}, {hi:+.3f}]  {'both signs OK' if both else 'sign FORCED'}")
    # a pair is free to flip only if the right-hand joint admits both signs
    pre = "arm" if key.startswith(("shoulder", "elbow")) else "leg"
    jr = f"{pre}_right_{key}_joint"
    if jr in names:
        lo, hi = lim[names.index(jr)].tolist()
        if lo < 0.0 < hi: free.append(key)
print(f"\nfree to flip: {free}")
env.close()

# --- 2. measure every valid combination ---
base = dict(C._PINCH_JOINT_POS)
print(f"\n=== {2**len(free)} combinations x root_z, measured ===")
print(f"{'flipped':38} {'root_z':>7} {'|dz| hands':>11} {'min body z':>11} {'hand z':>8}")
best = None
for bits in itertools.product([False, True], repeat=len(free)):
    pose = dict(base)
    tag = []
    for key, flip in zip(free, bits):
        if not flip: continue
        pre = "arm" if key.startswith(("shoulder", "elbow")) else "leg"
        jr = f"{pre}_right_{key}_joint"
        pose[jr] = -pose[jr]; tag.append(key)
    label = "+".join(tag) or "(original)"
    for root_z in (C._PINCH_ROOT_Z, 0.0, 0.10):
        try:
            e = build(pose, root_z); e.reset(); uu = e.unwrapped
            rr = uu.scene["robot_a"]
            b = _t(rr.data.body_pos_w)[0]
            nm = rr.body_names
            li, ri = nm.index("arm_left_hand_link"), nm.index("arm_right_hand_link")
            dz = abs(b[li, 2] - b[ri, 2]).item()
            mz = b[:, 2].min().item()
            hz = ((b[li, 2] + b[ri, 2]) / 2).item()
            print(f"{label:38} {root_z:+7.3f} {dz:11.4f} {mz:11.4f} {hz:8.3f}")
            score = (dz, abs(mz))          # want level hands and feet on the floor
            if mz > -0.02 and (best is None or score < best[0]):
                best = (score, label, root_z, dz, mz, hz)
            e.close()
        except Exception as exc:
            print(f"{label:38} {root_z:+7.3f}  invalid: {str(exc)[:52]}")

print("\n=== best combination with the feet at or above the floor ===")
print(f"  {best[1]}  root_z={best[2]:+.3f}  |dz|={best[3]:.4f}  min_z={best[4]:+.4f}  hands@{best[5]:.3f}"
      if best else "  none cleared the floor")
print("  target: hands level, min body z >= 0, hands reachable to GRASP_Z = 0.30")
app.app.close()
PY
