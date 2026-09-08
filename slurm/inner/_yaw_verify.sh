#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Confirm the new spawn 4-tuples survive env.reset(), not just a forced write.

21192773 found FACE+SYM at (0.707, -0.707, 0, 0) when that quat was pushed
through write_root_state_to_sim. Training goes through init_state.rot +
reset_root_state_uniform. If those write the 4-tuple the same way, reset
alone is FACE+SYM and the split is gone. If they do not, the forced-write
number was a local artefact and we have not fixed the spawn.
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

print("BHL_LEGACY_YAW:", __import__("os").environ.get("BHL_LEGACY_YAW", "<unset>"))
print("_YAW_M90:", C._YAW_M90)
print("_YAW_P90:", C._YAW_P90)

TASK = "TaskV2-BHL-CubeToShelf-Blind-v0"
cfg = gym.spec(TASK).kwargs["env_cfg_entry_point"]()
cfg.scene.num_envs = 2
for name in ("reset_joints_a", "reset_joints_b"):
    getattr(cfg.events, name).params["position_range"] = (0.0, 0.0)
for name in ("reset_root_a", "reset_root_b"):
    getattr(cfg.events, name).params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
env = gym.make(TASK, cfg=cfg, disable_env_checker=True)
env.reset()
u = env.unwrapped
u.sim.forward()
obj = _t(u.scene["object"].data.root_pos_w)[0, :2]

print(f"\n{'who':10} {'Lz':>8} {'Rz':>8} {'|dz|':>7} {'meanZ':>8}  toward  quat  verdict")
for who, want in (("robot_a", C._YAW_M90), ("robot_b", C._YAW_P90)):
    r = u.scene[who]
    r.update(dt=0.0)
    b = _t(r.data.body_pos_w)[0]
    root = _t(r.data.root_pos_w)[0]
    q = _t(r.data.root_quat_w)[0]
    nm = r.body_names
    li, ri = nm.index("arm_left_hand_link"), nm.index("arm_right_hand_link")
    lz = float(b[li, 2] - root[2]); rz = float(b[ri, 2] - root[2])
    mid = 0.5 * (b[li, :2] + b[ri, :2])
    to_obj = obj - root[:2]; to_h = mid - root[:2]
    toward = float(torch.dot(to_obj / (torch.norm(to_obj) + 1e-8),
                             to_h / (torch.norm(to_h) + 1e-8)))
    split = abs(lz - rz)
    ok = split < 0.01 and abs(0.5 * (lz + rz) + 0.6069) < 0.05 and toward > 0.5
    v = "FACE+SYM" if ok else ("SPLIT" if split >= 0.01 else "not facing / inverted")
    print(f"{who:10} {lz:+8.4f} {rz:+8.4f} {split:7.4f} {0.5*(lz+rz):+8.4f}  "
          f"{toward:+5.2f}  {[round(float(x),3) for x in q]}  {v}")
    print(f"           configured {list(want)}")

print("\nIf both rows are FACE+SYM the spawn fix holds through reset.")
print("If they split, init_state.rot and write_root_state_to_sim disagree.")
app.app.close()
PY
