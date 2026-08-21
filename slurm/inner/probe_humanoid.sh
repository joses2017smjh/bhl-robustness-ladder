#!/bin/bash
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
exec "$PY" - <<'PY'
import os
from pathlib import Path
import mujoco
import numpy as np
from bhl_robust.eval.mjcf_assets import prepare_mjcf

up = Path(os.environ["UPSTREAM"])
cache = Path(os.environ["CACHE_DIR"])
scene = prepare_mjcf(up, cache, "humanoid")
m = mujoco.MjModel.from_xml_path(str(scene))
d = mujoco.MjData(m)
print("nq", m.nq, "nu", m.nu, "nv", m.nv)
print("actuators:")
for i in range(m.nu):
    print(f"  {i:2d}", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

stand = {
    "leg_left_hip_pitch_joint": -0.2,
    "leg_left_knee_pitch_joint": 0.4,
    "leg_left_ankle_pitch_joint": -0.3,
    "leg_right_hip_pitch_joint": -0.2,
    "leg_right_knee_pitch_joint": 0.4,
    "leg_right_ankle_pitch_joint": -0.3,
}
mujoco.mj_resetData(m, d)
for name, q in stand.items():
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    d.qpos[m.jnt_qposadr[j]] = q
mujoco.mj_forward(m, d)
print("base xpos", np.round(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")], 3))
for b in ["arm_left_hand_link", "arm_right_hand_link", "arm_left_elbow_roll",
          "arm_right_elbow_roll", "leg_left_ankle_roll", "leg_right_ankle_roll"]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
    print(f"{b:28s} xpos={np.round(d.xpos[bid], 3)}")
print("freejoint qpos", np.round(d.qpos[:7], 3))
PY
