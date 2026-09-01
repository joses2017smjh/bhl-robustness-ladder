#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Is plank_leaned firing at spawn?

planktowall reports success 0.39-0.43 at a mean episode length of 22 steps. The
task is to carry a 1.5 m plank to a wall and lean it at 50-80 degrees; doing
that four times out of ten within 22 steps is not credible. The likeliest
explanation is the predicate being satisfied by the spawn pose or by the plank
toppling off its supports, which would make the success rate a measurement of
the scene rather than of the policy.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True; a.enable_cameras = True
app = AppLauncher(a)

import gymnasium as gym, torch
import bhl_robust.tasks  # noqa: F401
from bhl_robust.tasks.task_v2_mdp import plank_leaned
from bhl_robust.tasks.coop_lift_mdp import _t

TASK = "TaskV2-BHL-PlankToWall-Blind-v0"
cfg = gym.spec(TASK).kwargs["env_cfg_entry_point"]()
cfg.scene.num_envs = 64
env = gym.make(TASK, cfg=cfg, disable_env_checker=True)
obs, _ = env.reset()
u = env.unwrapped

def report(tag):
    ok = plank_leaned(u, wall_x=1.0, contact_z=0.50)
    p = _t(u.scene["object"].data.root_pos_w)[:, :3] - u.scene.env_origins
    print(f"  {tag:12} success={ok.float().mean().item():.3f}  "
          f"plank z mean={p[:,2].mean().item():+.3f}  x mean={p[:,0].mean().item():+.3f}")

report("at reset")
act = torch.zeros((64, u.action_space.shape[-1]), device=u.device)
for i in range(1, 41):
    env.step(act)
    if i in (5, 10, 20, 40):
        report(f"step {i}")
print("\nIf success is already nonzero with a zero action, the predicate is")
print("measuring the scene settling, not the policy.")
app.app.close()
PY
