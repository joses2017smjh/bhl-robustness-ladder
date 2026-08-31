#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Why does every v2 episode end on step one? Measure the spawn state.

Wrapping the ProxyArray reads did not fix it, so the theory was wrong or
incomplete. This prints what either_fallen actually sees at reset instead of
reasoning about what it ought to see.
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
cfg = gym.spec(TASK).kwargs["env_cfg_entry_point"]()
cfg.scene.num_envs = 4
env = gym.make(TASK, cfg=cfg, disable_env_checker=True)
obs, _ = env.reset()
u = env.unwrapped

for name in ("robot_a", "robot_b"):
    r = u.scene[name]
    pg = _t(r.data.projected_gravity_b)
    root = _t(r.data.root_pos_w)
    print(f"{name}:")
    print(f"  projected_gravity_b shape {tuple(pg.shape)}  row0 {pg[0].tolist()}")
    tilt = torch.acos((-pg[:, 2]).clamp(-1.0, 1.0))
    print(f"  tilt (rad) {tilt.tolist()}   limit 0.78")
    print(f"  root z {root[:, 2].tolist()}")

print("\ntermination terms at reset:")
tm = u.termination_manager
for term in tm.active_terms:
    val = tm.get_term(term)
    print(f"  {term:12} {val.float().mean().item():.4f}")

print("\nfurniture prims in the scene:")
print("  ", sorted(k for k in u.scene.keys() if k not in ("robot_a","robot_b","object","terrain")))
app.app.close()
PY
