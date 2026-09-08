#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""What does the task do to the root between reset and the first read?

Established, with the measurement trap named:

* Bare Articulation + PINCH_POSE, identity quat, write then step:
  hands at -0.6069 / -0.6069. Asset and pose are symmetric.
* In-task joints hold PINCH (worst drift 0.077 rad = the jitter). Not sag.
* In-task "pose forced" without a sim step: hands -0.20 / +0.15, root quat
  (0.7059, -0.0419, -0.0419, -0.7059) against a configured pure yaw. That
  cell skipped the kinematics update, so it may be the same stale read that
  voided the first bisection.

This probe does four things the last one did not:

1. Zero the reset jitter, so we are not measuring noise.
2. Call sim.forward() after every write -- kinematics, no physics -- before
   any body_pos_w read. sim.step() is a separate row, labelled as physics.
3. Report hands in the root BODY frame, not just world-Z. A 90 deg yaw
   cannot split world-Z of a symmetric pose; a roll can. Body-frame is
   what decides whether the split is the root or the arms.
4. Force the root onto the configured yaw, and onto identity, so the one
   measured discrepancy (a tilt nothing asked for) is isolated.

Rule: force the state, forward, then read. Anything else is measuring the
previous pose.
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
cfg = gym.spec(TASK).kwargs["env_cfg_entry_point"]()
cfg.scene.num_envs = 4
# Zero jitter: the last in-task numbers mixed +/-0.08 joint offset and
# +/-0.12 yaw into every read. This probe is about the spawn, not the noise.
for name in ("reset_joints_a", "reset_joints_b"):
    getattr(cfg.events, name).params["position_range"] = (0.0, 0.0)
    getattr(cfg.events, name).params["velocity_range"] = (0.0, 0.0)
for name in ("reset_root_a", "reset_root_b"):
    getattr(cfg.events, name).params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    getattr(cfg.events, name).params["velocity_range"] = {}
if hasattr(cfg.events, "reset_object"):
    cfg.events.reset_object.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}

env = gym.make(TASK, cfg=cfg, disable_env_checker=True)
env.reset()
u = env.unwrapped
sim = u.sim

def qrot_inv(q, v):
    """Rotate v by the inverse of unit quaternion q (w,x,y,z)."""
    w, x, y, z = (float(a) for a in q)
    vx, vy, vz = (float(a) for a in v)
    # t = 2 * cross(qvec, v)
    tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
    # v' = v - w t + cross(qvec, t)   (rotation by q*)
    return (
        vx - w * tx + (y * tz - z * ty),
        vy - w * ty + (z * tx - x * tz),
        vz - w * tz + (x * ty - y * tx),
    )

def force_joints(art, jp_map):
    names = art.joint_names
    jp = _t(art.data.default_joint_pos).clone()
    for j, v in jp_map.items():
        if j in names:
            jp[:, names.index(j)] = float(v)
    art.write_joint_state_to_sim(jp, torch.zeros_like(jp))

def force_root_quat(art, quat, pos=None):
    state = _t(art.data.root_state_w).clone()
    q = torch.tensor(list(quat), device=state.device, dtype=state.dtype)
    state[:, 3:7] = q
    if pos is not None:
        state[:, 0:3] = torch.tensor(list(pos), device=state.device, dtype=state.dtype)
    state[:, 7:] = 0
    art.write_root_state_to_sim(state)

def refresh(art, physics=False):
    """Make body_pos_w mean the state we just wrote.

    forward() is kinematics only. step() integrates PhysX -- depenetration,
    gravity -- and is a different measurement. The last in-task probe called
    neither, which is why it is void.
    """
    if physics:
        sim.step(render=False)
    else:
        sim.forward()
    art.update(dt=0.0)

def report(tag, art, configured):
    art.update(dt=0.0)
    b = _t(art.data.body_pos_w)[0]
    root = _t(art.data.root_pos_w)[0]
    q = _t(art.data.root_quat_w)[0]
    nm = art.body_names
    li, ri = nm.index("arm_left_hand_link"), nm.index("arm_right_hand_link")
    lw = (float(b[li, 0] - root[0]), float(b[li, 1] - root[1]), float(b[li, 2] - root[2]))
    rw = (float(b[ri, 0] - root[0]), float(b[ri, 1] - root[1]), float(b[ri, 2] - root[2]))
    lb = qrot_inv(q, lw)
    rbod = qrot_inv(q, rw)
    up_z = 1.0 - 2.0 * (float(q[1]) ** 2 + float(q[2]) ** 2)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
    zs = [float(b[i, 2]) for i in range(len(nm))]
    n_under = sum(1 for z in zs if z < 0.0)
    ankles = []
    for name in ("leg_left_ankle_roll", "leg_right_ankle_roll"):
        if name in nm:
            ankles.append(f"{name} z={float(b[nm.index(name), 2]):+.3f}")
    print(f"\n=== {tag} ===")
    print(f"  root pos  {[round(float(x), 4) for x in root]}")
    print(f"  root quat {[round(float(x), 4) for x in q]}")
    print(f"  configured {list(configured)}")
    print(f"  up_z={up_z:.4f}  tilt={tilt:.2f} deg  bodies_z<0={n_under}/{len(nm)}")
    print(f"  world-Z hands  L={lw[2]:+.4f}  R={rw[2]:+.4f}  |dz|={abs(lw[2]-rw[2]):.4f}")
    print(f"  body-Z  hands  L={lb[2]:+.4f}  R={rbod[2]:+.4f}  |dz|={abs(lb[2]-rbod[2]):.4f}")
    print(f"  body-XY hands  L=({lb[0]:+.3f},{lb[1]:+.3f})  R=({rbod[0]:+.3f},{rbod[1]:+.3f})")
    print("  bare Articulation, same pose, identity quat: world-Z -0.6069 / -0.6069")
    if ankles:
        print("  " + "  ".join(ankles))
    v_w = "SYMMETRIC" if abs(lw[2] - rw[2]) < 0.01 else "<<< SPLIT"
    v_b = "SYMMETRIC" if abs(lb[2] - rbod[2]) < 0.01 else "<<< SPLIT"
    print(f"  verdict  world-Z {v_w}   body-Z {v_b}")

print("configured robot_a rot (wxyz):", C._YAW_M90)
print("configured robot_a pos:", cfg.scene.robot_a.init_state.pos)
print("configured _PINCH_ROOT_Z:", C._PINCH_ROOT_Z)
print("jitter: OFF")
print("plant_feet:", "ON" if hasattr(cfg.events, "plant_feet_a") else "OFF")

ra, rb = u.scene["robot_a"], u.scene["robot_b"]

# --- 1. reset as training sees it, kinematics only -------------------
refresh(ra, physics=False)
report("1 after reset, forward() only -- robot_a", ra, C._YAW_M90)
report("1 after reset, forward() only -- robot_b", rb, C._YAW_P90)

# --- 2. one physics step: depenetration, gravity ---------------------
refresh(ra, physics=True)
report("2 after reset + 1 physics step -- robot_a", ra, C._YAW_M90)

# --- 3. force PINCH + configured yaw, kinematics only ----------------
force_joints(ra, C._PINCH_JOINT_POS)
force_root_quat(ra, C._YAW_M90)
refresh(ra, physics=False)
report("3 force PINCH + configured yaw, forward() -- robot_a", ra, C._YAW_M90)

# --- 4. force PINCH + identity quat: does the yaw cause world-Z split?
force_joints(ra, C._PINCH_JOINT_POS)
force_root_quat(ra, (1.0, 0.0, 0.0, 0.0))
refresh(ra, physics=False)
report("4 force PINCH + identity quat, forward() -- robot_a", ra, (1.0, 0.0, 0.0, 0.0))

# --- 5. force PINCH + configured yaw, then let physics touch it ------
force_joints(ra, C._PINCH_JOINT_POS)
force_root_quat(ra, C._YAW_M90)
refresh(ra, physics=True)
report("5 force PINCH + configured yaw + 1 physics step -- robot_a", ra, C._YAW_M90)

print("\nHow to read this:")
print("  If 3 body-Z is symmetric at -0.607 and 3 world-Z is not,")
print("    the split is the root orientation. The arms are fine.")
print("  If 3 body-Z still splits, something the task scene does to FK")
print("    remains -- and that is no longer a stale-kinematics artefact.")
print("  If 3 is symmetric and 5 splits, physics (burial / depenetration)")
print("    is what moves the hands, and the spawn height is the fix.")
print("  4 matching the bare Articulation (-0.6069 both) is the control")
print("    that the last in-task probe never had.")

app.app.close()
PY
