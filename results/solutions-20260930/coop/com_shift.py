import os, sys, json
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
import numpy as np, mujoco
REPO = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder"); sys.path.insert(0, str(REPO / "src"))
from bhl_robust.eval.mjcf_assets import prepare_mjcf
from bhl_robust.eval.scripted_carry import KEYFRAMES_LEFT, ARM_JOINTS_L, ARM_JOINTS_R
scene = prepare_mjcf(REPO / "external/Berkeley-Humanoid-Lite", Path("mjcf_cache"), "humanoid")
m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m)
def jid(n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
LEG = {"leg_left_hip_pitch_joint": -0.2, "leg_left_knee_pitch_joint": 0.4, "leg_left_ankle_pitch_joint": -0.3,
       "leg_right_hip_pitch_joint": -0.2, "leg_right_knee_pitch_joint": 0.4, "leg_right_ankle_pitch_joint": -0.3}
base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
M = float(m.body_subtreemass[base]); g = 9.81
def com(arms_left=None, arms_right=None):
    d.qpos[:] = 0; d.qpos[3] = 1
    for k, v in LEG.items(): d.qpos[m.jnt_qposadr[jid(k)]] = v
    if arms_left is not None:
        for n, v in zip(ARM_JOINTS_L, arms_left): d.qpos[m.jnt_qposadr[jid(n)]] = v
    if arms_right is not None:
        for n, v in zip(ARM_JOINTS_R, arms_right): d.qpos[m.jnt_qposadr[jid(n)]] = v
    mujoco.mj_forward(m, d)
    return d.subtree_com[base].copy(), d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_left_hand_link")].copy()
c0, h0 = com()
out = {"robot_mass_kg": round(M, 3)}
for k in ("squeeze", "lift"):
    c, h = com(KEYFRAMES_LEFT[k])
    dy, dx = c[1] - c0[1], c[0] - c0[0]
    # cube share 0.25 kg at the hand (~pad) position, moment about the stance centre
    out[k] = {"com_shift_lateral_m": round(float(dy), 4), "com_shift_fwd_m": round(float(dx), 4),
              "arm_self_moment_roll_Nm": round(float(M * g * dy), 3), "arm_self_moment_pitch_Nm": round(float(M * g * dx), 3),
              "hand_xyz_m": np.round(h, 3).tolist(),
              "payload_share_moment_roll_Nm": round(float(0.25 * g * h[1]), 3),
              "payload_share_moment_pitch_Nm": round(float(0.25 * g * h[0]), 3)}
# Isaac pinch arms (both arms forward)
c, h = com([-0.55, -0.26, 0.0, 0.90, 0.0], [0.55, 0.26, 0.0, -0.90, 0.0])
out["isaac_pinch_both_arms"] = {"com_shift_fwd_m": round(float(c[0] - c0[0]), 4), "arm_self_moment_pitch_Nm": round(float(M * g * (c[0] - c0[0])), 3),
                                "payload_moment_pitch_Nm_0.25kg_per_robot_at_cube_face": round(0.25 * g * 0.34, 3)}
out["training_disturbance_on_base"] = "constant force +/-2 N and torque +/-2 Nm per episode (reset); base mass -1..+2 kg (startup)"
print(json.dumps(out, indent=1)); json.dump(out, open("com_shift.json", "w"), indent=1)
