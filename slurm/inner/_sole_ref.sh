#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""How high are the leg bodies when this robot is standing correctly?

plant_feet translated the root until the *lowest body* sat on the plane. With
the arm asymmetry still open the lowest body is a hand, so it hoisted the robot
until the hand cleared the floor and left the feet 8-20 cm in the air. Every
re-run arm then fell on every episode: mean episode length 5.0, fall rate 1.000.

The fix is to plant on the legs. This measures the reference the legs should
land at, from the configuration that is known to stand: the shipped locomotion
task, whose robot spawns at root z = 0 and walks.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True; a.enable_cameras = True
app = AppLauncher(a)

import gymnasium as gym
import bhl_robust.tasks  # noqa: F401
from bhl_robust.tasks.coop_lift_mdp import _t

TASK = "Velocity-BHL-Arms-PushAdaptive-v0"
cfg = gym.spec(TASK).kwargs["env_cfg_entry_point"]()
cfg.scene.num_envs = 4
env = gym.make(TASK, cfg=cfg, disable_env_checker=True)
env.reset(); u = env.unwrapped
r = u.scene["robot"]
r.update(dt=0.0)
nm = r.body_names
b = _t(r.data.body_pos_w)[0]
root = _t(r.data.root_pos_w)[0]
org = u.scene.env_origins[0]

leg = [i for i, n in enumerate(nm) if "leg" in n.lower()]
print(f"root z (above terrain): {float(root[2] - org[2]):+.4f}")
print("\nleg bodies, height above terrain:")
for i in sorted(leg, key=lambda j: float(b[j, 2])):
    print(f"  {nm[i]:34} {float(b[i,2] - org[2]):+.4f}")
lowest_leg = min(float(b[i, 2] - org[2]) for i in leg)
lowest_any = float((b[:, 2] - org[2]).min())
print(f"\nSOLE_REF (lowest leg body when standing) = {lowest_leg:+.4f}")
print(f"lowest body of any kind                  = {lowest_any:+.4f}")
print("\nThis is the number plant_feet should drive the legs to, for any pose.")
app.app.close()
PY
