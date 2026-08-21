#!/bin/bash
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
exec "$PY" - <<'PY'
"""Print hand positions at candidate squat/reach poses (kinematics only)."""
import os
from pathlib import Path
import mujoco
import numpy as np
from bhl_robust.eval.mjcf_assets import prepare_mjcf

JOINTS = [
    "arm_left_shoulder_pitch_joint", "arm_left_shoulder_roll_joint",
    "arm_left_shoulder_yaw_joint", "arm_left_elbow_pitch_joint",
    "arm_left_elbow_roll_joint",
    "arm_right_shoulder_pitch_joint", "arm_right_shoulder_roll_joint",
    "arm_right_shoulder_yaw_joint", "arm_right_elbow_pitch_joint",
    "arm_right_elbow_roll_joint",
    "leg_left_hip_roll_joint", "leg_left_hip_yaw_joint",
    "leg_left_hip_pitch_joint", "leg_left_knee_pitch_joint",
    "leg_left_ankle_pitch_joint", "leg_left_ankle_roll_joint",
    "leg_right_hip_roll_joint", "leg_right_hip_yaw_joint",
    "leg_right_hip_pitch_joint", "leg_right_knee_pitch_joint",
    "leg_right_ankle_pitch_joint", "leg_right_ankle_roll_joint",
]

def pose(**kw):
    q = np.zeros(22)
    # standing legs
    q[12] = q[18] = -0.2
    q[13] = q[19] = 0.4
    q[14] = q[20] = -0.3
    name = {n: i for i, n in enumerate(JOINTS)}
    for k, v in kw.items():
        q[name[k]] = v
    return q

POSES = {
    "stand": pose(),
    "squat": pose(
        leg_left_hip_pitch_joint=-0.85, leg_right_hip_pitch_joint=-0.85,
        leg_left_knee_pitch_joint=1.45, leg_right_knee_pitch_joint=1.45,
        leg_left_ankle_pitch_joint=-0.55, leg_right_ankle_pitch_joint=-0.55,
    ),
    "squat_deep": pose(
        leg_left_hip_pitch_joint=-1.4, leg_right_hip_pitch_joint=-1.4,
        leg_left_knee_pitch_joint=2.1, leg_right_knee_pitch_joint=2.1,
        leg_left_ankle_pitch_joint=-0.7, leg_right_ankle_pitch_joint=-0.7,
    ),
    "pick_low": pose(
        leg_left_hip_pitch_joint=-1.4, leg_right_hip_pitch_joint=-1.4,
        leg_left_knee_pitch_joint=2.1, leg_right_knee_pitch_joint=2.1,
        leg_left_ankle_pitch_joint=-0.7, leg_right_ankle_pitch_joint=-0.7,
        arm_left_shoulder_pitch_joint=-1.2, arm_right_shoulder_pitch_joint=1.2,
        arm_left_shoulder_roll_joint=0.5, arm_right_shoulder_roll_joint=-0.5,
        arm_left_elbow_pitch_joint=1.3, arm_right_elbow_pitch_joint=-1.3,
        arm_left_shoulder_yaw_joint=0.3, arm_right_shoulder_yaw_joint=-0.3,
    ),
    "pick_fwd": pose(
        leg_left_hip_pitch_joint=-1.1, leg_right_hip_pitch_joint=-1.1,
        leg_left_knee_pitch_joint=1.8, leg_right_knee_pitch_joint=1.8,
        leg_left_ankle_pitch_joint=-0.65, leg_right_ankle_pitch_joint=-0.65,
        arm_left_shoulder_pitch_joint=-1.35, arm_right_shoulder_pitch_joint=1.35,
        arm_left_shoulder_roll_joint=0.25, arm_right_shoulder_roll_joint=-0.25,
        arm_left_elbow_pitch_joint=0.85, arm_right_elbow_pitch_joint=-0.85,
    ),
    "reach_down": pose(
        leg_left_hip_pitch_joint=-1.4, leg_right_hip_pitch_joint=-1.4,
        leg_left_knee_pitch_joint=2.1, leg_right_knee_pitch_joint=2.1,
        leg_left_ankle_pitch_joint=-0.7, leg_right_ankle_pitch_joint=-0.7,
        arm_left_shoulder_pitch_joint=0.6, arm_right_shoulder_pitch_joint=-0.6,
        arm_left_shoulder_roll_joint=0.9, arm_right_shoulder_roll_joint=-0.9,
        arm_left_elbow_pitch_joint=1.5, arm_right_elbow_pitch_joint=-1.5,
        arm_left_shoulder_yaw_joint=0.5, arm_right_shoulder_yaw_joint=-0.5,
    ),
    "reach_in": pose(
        leg_left_hip_pitch_joint=-1.4, leg_right_hip_pitch_joint=-1.4,
        leg_left_knee_pitch_joint=2.1, leg_right_knee_pitch_joint=2.1,
        leg_left_ankle_pitch_joint=-0.7, leg_right_ankle_pitch_joint=-0.7,
        arm_left_shoulder_pitch_joint=-0.3, arm_right_shoulder_pitch_joint=0.3,
        arm_left_shoulder_roll_joint=1.2, arm_right_shoulder_roll_joint=-1.2,
        arm_left_elbow_pitch_joint=1.5, arm_right_elbow_pitch_joint=-1.5,
    ),
}

up = Path(os.environ["UPSTREAM"])
cache = Path(os.environ["CACHE_DIR"])
m = mujoco.MjModel.from_xml_path(str(prepare_mjcf(up, cache, "humanoid")))
d = mujoco.MjData(m)
lh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_left_hand_link")
rh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_right_hand_link")
base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
la = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "leg_left_ankle_roll")
ra = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "leg_right_ankle_roll")
jadr = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINTS}

STAND_ANKLE_Z = None
for name, q in POSES.items():
    mujoco.mj_resetData(m, d)
    for n, v in zip(JOINTS, q):
        d.qpos[jadr[n]] = v
    mujoco.mj_forward(m, d)
    if STAND_ANKLE_Z is None:
        STAND_ANKLE_Z = 0.5 * (d.xpos[la][2] + d.xpos[ra][2])
        print("stand ankle z", STAND_ANKLE_Z)
    # drop the free joint so feet stay on the floor
    dz = 0.5 * (d.xpos[la][2] + d.xpos[ra][2]) - STAND_ANKLE_Z
    d.qpos[2] -= dz
    mujoco.mj_forward(m, d)
    # brute-force a grasp pose at squat_deep legs
legs = dict(
    leg_left_hip_pitch_joint=-1.4, leg_right_hip_pitch_joint=-1.4,
    leg_left_knee_pitch_joint=2.1, leg_right_knee_pitch_joint=2.1,
    leg_left_ankle_pitch_joint=-0.7, leg_right_ankle_pitch_joint=-0.7,
)
print("--- search (x>0.10, z<0.40, |y|<0.18) ---")
hits = 0
for lp in (-1.4, -0.8, -0.3, 0.3, 0.6):
    for lr in (0.2, 0.6, 1.0, 1.25):
        for ly in (-0.5, 0.0, 0.5, 0.75):
            for le in (0.3, 0.9, 1.5):
                q = pose(
                    **legs,
                    arm_left_shoulder_pitch_joint=lp, arm_right_shoulder_pitch_joint=-lp,
                    arm_left_shoulder_roll_joint=lr, arm_right_shoulder_roll_joint=-lr,
                    arm_left_shoulder_yaw_joint=ly, arm_right_shoulder_yaw_joint=-ly,
                    arm_left_elbow_pitch_joint=le, arm_right_elbow_pitch_joint=-le,
                )
                mujoco.mj_resetData(m, d)
                for n, v in zip(JOINTS, q):
                    d.qpos[jadr[n]] = v
                mujoco.mj_forward(m, d)
                dz = 0.5 * (d.xpos[la][2] + d.xpos[ra][2]) - STAND_ANKLE_Z
                d.qpos[2] -= dz
                mujoco.mj_forward(m, d)
                L = d.xpos[lh]
                if L[0] > 0.10 and L[2] < 0.40 and abs(L[1]) < 0.18:
                    print(f"  hit pitch={lp:+.2f} roll={lr:.2f} yaw={ly:+.2f} elb={le:.2f}  "
                          f"L={np.round(L,3)}")
                    hits += 1
                    if hits >= 12:
                        break
            if hits >= 12:
                break
        if hits >= 12:
            break
    if hits >= 12:
        break
print("hits", hits)
PY
