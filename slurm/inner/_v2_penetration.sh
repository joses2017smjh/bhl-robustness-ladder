#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Does the v2 spawn put the robot through the floor?

The Isaac replay shows both robots flat on the ground within a second and
staying there, which is not what ~8-step episodes should look like -- at 8 steps
a 300-step clip should reset them upright dozens of times. `_PINCH_ROOT_Z` is
-0.07, and the shipped asset spawns at z = 0.0 with its root frame at ground
level, so the crouch pose is applied 7 cm *below* the plane.

This measures it instead of arguing about it: where every body starts, how far
the lowest one is under z = 0, and what the solver does about it over the first
few steps with a zero action. Depenetration at max_depenetration_velocity = 1.0
is what launched the plank payload 22 cm, so the same question is asked here.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True; a.enable_cameras = True
app = AppLauncher(a)

import gymnasium as gym, torch
import bhl_robust.tasks  # noqa: F401
from bhl_robust.tasks.coop_lift_mdp import _t

# The control. "19 of 27 bodies below z = 0" is a fact about this asset's frame
# convention until a task that demonstrably works reports a smaller number --
# the same mistake G-B2 made when it measured its own iteration budget and
# called it a terrain verdict.
TASKS = [
    ("v2 CubeToShelf (suspect)", "TaskV2-BHL-CubeToShelf-Blind-v0", ("robot_a", "robot_b")),
    ("22-DoF locomotion (control)", "Velocity-BHL-Arms-PushAdaptive-v0", ("robot",)),
]

def report(tag, u, robots):
    print(f"\n=== {tag} ===")
    for name in robots:
        r = u.scene[name]
        root = _t(r.data.root_pos_w)
        bodies = _t(r.data.body_pos_w)            # (envs, bodies, 3)
        # env origins are added into world pos; subtract to get z above terrain
        zmin, zi = bodies[..., 2].min(dim=1)
        names = r.body_names
        print(f"  {name}: root_z={root[0,2]:+.4f}  "
              f"lowest body={names[zi[0]]!r} at z={zmin[0]:+.4f}")
        under = (bodies[..., 2] < 0.0).sum(dim=1)
        print(f"          bodies below z=0: {under.tolist()} of {len(names)}")

for label, task, robots in TASKS:
    print(f"\n{'#'*66}\n# {label}: {task}\n{'#'*66}")
    if task not in gym.registry:
        print(f"  not registered -- skipped"); continue
    try:
        cfg = gym.spec(task).kwargs["env_cfg_entry_point"]()
        cfg.scene.num_envs = 4
        env = gym.make(task, cfg=cfg, disable_env_checker=True)
        env.reset()
        u = env.unwrapped
        report("at reset, before any step", u, robots)
        zero = torch.zeros((4, u.action_space.shape[-1]), device=u.device)
        for i in range(1, 11):
            env.step(zero)
            if i in (1, 2, 5, 10):
                report(f"after step {i} (zero action)", u, robots)
        print("\n  termination terms after 10 zero-action steps:")
        tm = u.termination_manager
        for term in tm.active_terms:
            print(f"    {term:14} {tm.get_term(term).float().mean().item():.4f}")
        env.close()
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"  FAILED: {exc!r}")
app.app.close()
PY
