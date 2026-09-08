#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Which 4-tuple is actually a yaw, once it is forced and forwarded?

21192744 closed the root-orientation question. Identity + PINCH is symmetric
to four decimal places (-0.6077 / -0.6077), matching the bare articulation.
The configured "yaw" (0.7071, 0, 0, -0.7071) puts the hands at -0.177 / +0.177
-- the identity pose's left/right Y, now on world Z. That is a roll, not a yaw.

up_z computed from the 4-tuple as wxyz is 1.000 either way, which is how the
earlier rotation check (21186364) concluded the robot was upright. It was
reading the label, not the bodies. This sweep scores each candidate on the
bodies: hand |dz|, mean hand height vs the -0.607 reference, and whether the
hands sit toward the cube (robot_a should face -Y).
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

S2 = 0.70710678
CASES = {
    "identity                 (1,0,0,0)":           (1.0, 0.0, 0.0, 0.0),
    "wxyz yaw -90  (current)  (.707,0,0,-.707)":    (S2, 0.0, 0.0, -S2),
    "wxyz roll +90            (.707,+.707,0,0)":    (S2,  S2, 0.0, 0.0),
    "wxyz roll -90            (.707,-.707,0,0)":    (S2, -S2, 0.0, 0.0),
    "wxyz pitch -90           (.707,0,-.707,0)":    (S2, 0.0, -S2, 0.0),
    "wxyz yaw 180             (0,0,0,1)":           (0.0, 0.0, 0.0, 1.0),
    "wxyz 180 X               (0,1,0,0)":           (0.0, 1.0, 0.0, 0.0),
    "wxyz 180 Y               (0,0,1,0)":           (0.0, 0.0, 1.0, 0.0),
    "xyzw-as-wxyz yaw -90     (0,0,-.707,.707)":    (0.0, 0.0, -S2,  S2),
}

TASK = "TaskV2-BHL-CubeToShelf-Blind-v0"
cfg = gym.spec(TASK).kwargs["env_cfg_entry_point"]()
cfg.scene.num_envs = 2
for name in ("reset_joints_a", "reset_joints_b"):
    getattr(cfg.events, name).params["position_range"] = (0.0, 0.0)
    getattr(cfg.events, name).params["velocity_range"] = (0.0, 0.0)
for name in ("reset_root_a", "reset_root_b"):
    getattr(cfg.events, name).params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
env = gym.make(TASK, cfg=cfg, disable_env_checker=True)
env.reset()
u = env.unwrapped
sim = u.sim
r = u.scene["robot_a"]
obj = u.scene["object"]
names = r.joint_names
nm = r.body_names
li, ri = nm.index("arm_left_hand_link"), nm.index("arm_right_hand_link")

def force(quat):
    jp = _t(r.data.default_joint_pos).clone()
    for j, v in C._PINCH_JOINT_POS.items():
        if j in names:
            jp[:, names.index(j)] = float(v)
    r.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    state = _t(r.data.root_state_w).clone()
    state[:, 3:7] = torch.tensor(list(quat), device=state.device, dtype=state.dtype)
    state[:, 7:] = 0
    r.write_root_state_to_sim(state)
    sim.forward()
    r.update(dt=0.0)

print(f"{'case':42} {'Lz':>8} {'Rz':>8} {'|dz|':>7} {'meanZ':>8}  "
      f"{'Lxy':>16} {'Rxy':>16}  toward  verdict")
obj_xy = _t(obj.data.root_pos_w)[0, :2]
for label, quat in CASES.items():
    force(quat)
    b = _t(r.data.body_pos_w)[0]
    root = _t(r.data.root_pos_w)[0]
    lz = float(b[li, 2] - root[2]); rz = float(b[ri, 2] - root[2])
    lxy = (float(b[li, 0] - root[0]), float(b[li, 1] - root[1]))
    rxy = (float(b[ri, 0] - root[0]), float(b[ri, 1] - root[1]))
    mid = 0.5 * (b[li, :2] + b[ri, :2])
    to_obj = obj_xy - root[:2]
    to_hands = mid - root[:2]
    toward = float(torch.dot(
        to_obj / (torch.norm(to_obj) + 1e-8),
        to_hands / (torch.norm(to_hands) + 1e-8),
    ))
    split = abs(lz - rz)
    meanz = 0.5 * (lz + rz)
    near_ref = abs(meanz + 0.6069) < 0.03
    v = "SYMMETRIC" if split < 0.01 else "<<< SPLIT"
    if split < 0.01 and near_ref and toward > 0.5:
        v = "<<< FACE+SYM"
    elif split < 0.01 and near_ref:
        v = "SYM, not facing"
    print(f"{label:42} {lz:+8.4f} {rz:+8.4f} {split:7.4f} {meanz:+8.4f}  "
          f"({lxy[0]:+.2f},{lxy[1]:+.2f}) ({rxy[0]:+.2f},{rxy[1]:+.2f})  "
          f"{toward:+5.2f}  {v}")

print("\nReference: bare Articulation identity PINCH  Lz=Rz=-0.6069")
print("FACE+SYM is the 4-tuple to put in _YAW_M90. Its negation / conjugate")
print("is _YAW_P90. Do not change the published default until that cell is")
print("this row -- the current spawn is what FINDINGS was trained on.")
app.app.close()
PY
