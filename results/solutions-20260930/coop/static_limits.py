"""Static torque / grip budget of the BHL arm at the postures the coop tasks used.

Read-only on the repo: builds the MJCF into a scratch cache. MuJoCo 3.3.5, CPU.
Outputs JSON to the scratch dir.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import mujoco
import numpy as np

REPO = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder")
sys.path.insert(0, str(REPO / "src"))
from bhl_robust.eval.mjcf_assets import prepare_mjcf  # noqa: E402
from bhl_robust.eval.scripted_carry import (  # noqa: E402
    ARM_JOINTS_L, ARM_JOINTS_R, KEYFRAMES_LEFT, add_hand_pads, hand_pad_frames, lowest_point)

OUT = Path(__file__).resolve().parent
CACHE = OUT / "mjcf_cache"
UP = REPO / "external" / "Berkeley-Humanoid-Lite"
G = 9.81
ARM_CAP = 4.0
LEG_CAP = 6.0
CUBE_M = 0.5

scene = prepare_mjcf(UP, CACHE, "humanoid")
robot_xml = scene.parent / "berkeley_humanoid_lite.xml"
spec = mujoco.MjSpec.from_file(str(scene))
add_hand_pads(spec, hand_pad_frames(robot_xml), 1.0)
m = spec.compile()
d = mujoco.MjData(m)


def jid(n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)


def bid(n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def gid(n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)


LEG_DEFAULT = {"leg_left_hip_pitch_joint": -0.2, "leg_left_knee_pitch_joint": 0.4,
               "leg_left_ankle_pitch_joint": -0.3, "leg_right_hip_pitch_joint": -0.2,
               "leg_right_knee_pitch_joint": 0.4, "leg_right_ankle_pitch_joint": -0.3}
LEG_CROUCH = {"leg_left_hip_pitch_joint": -0.85, "leg_right_hip_pitch_joint": -0.85,
              "leg_left_knee_pitch_joint": 1.45, "leg_right_knee_pitch_joint": 1.45,
              "leg_left_ankle_pitch_joint": -0.55, "leg_right_ankle_pitch_joint": -0.55}
ISAAC_PINCH_ARMS = {"arm_left_shoulder_roll_joint": -0.26, "arm_right_shoulder_roll_joint": 0.26,
                    "arm_left_shoulder_pitch_joint": -0.55, "arm_right_shoulder_pitch_joint": 0.55,
                    "arm_left_elbow_pitch_joint": 0.90, "arm_right_elbow_pitch_joint": -0.90}

ARM_DOF = {s: [m.jnt_dofadr[jid(n)] for n in (ARM_JOINTS_L if s == "left" else ARM_JOINTS_R)]
           for s in ("left", "right")}
ARM_NAMES = ["sh_pitch", "sh_roll", "sh_yaw", "elbow", "elbow_roll"]
coll_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] > 0]


def set_pose(legs: dict, arms: dict):
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0
    for k, v in {**legs, **arms}.items():
        d.qpos[m.jnt_qposadr[jid(k)]] = v
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    d.qpos[2] -= lowest_point(m, d, [g for g in coll_geoms if "pad" not in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")])
    mujoco.mj_forward(m, d)


def arms_from_left(q5):
    q5 = np.asarray(q5, float)
    return {**{n: v for n, v in zip(ARM_JOINTS_L, q5)}, **{n: -v for n, v in zip(ARM_JOINTS_R, q5)}}


def point_jac(body, point):
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jp, jr, np.asarray(point, float), body)
    return jp


def arm_budget(side, point, body, n_dir, f_vert):
    """Torques at the 5 arm joints for gravity alone, gravity + a vertical load
    f_vert (N, downward on the arm), and per newton of force along n_dir (the
    direction the payload pushes ON the arm). Largest N with all |tau| <= cap."""
    dofs = ARM_DOF[side]
    g = d.qfrc_bias[dofs].copy()                     # static torque to hold the arm (qvel = 0)
    J = point_jac(body, point)[:, dofs]
    t_vert = J.T @ np.array([0.0, 0.0, -f_vert])     # payload weight share pulls the arm down
    # tau_needed = g - J^T F_ext  (F_ext = force applied BY the payload ON the arm)
    t_load = g - t_vert
    t_perN = -(J.T @ np.asarray(n_dir, float))
    nmax = np.inf
    lim = None
    for i in range(5):
        a, b = t_load[i], t_perN[i]
        if abs(b) < 1e-9:
            if abs(a) > ARM_CAP:
                nmax, lim = 0.0, ARM_NAMES[i]
            continue
        hi = (ARM_CAP - a) / b if b > 0 else (-ARM_CAP - a) / b
        if hi < nmax:
            nmax, lim = hi, ARM_NAMES[i]
    return {"gravity_only_Nm": dict(zip(ARM_NAMES, np.round(g, 3).tolist())),
            "gravity_plus_weight_share_Nm": dict(zip(ARM_NAMES, np.round(t_load, 3).tolist())),
            "per_N_squeeze_Nm": dict(zip(ARM_NAMES, np.round(t_perN, 4).tolist())),
            "max_abs_gravity_Nm": round(float(np.abs(g).max()), 3),
            "max_abs_with_weight_Nm": round(float(np.abs(t_load).max()), 3),
            "sag_at_kp10_rad": dict(zip(ARM_NAMES, np.round(np.abs(t_load) / 10.0, 3).tolist())),
            "sag_at_kp30_rad": dict(zip(ARM_NAMES, np.round(np.abs(t_load) / 30.0, 3).tolist())),
            "N_max_within_cap": None if not np.isfinite(nmax) else round(float(max(nmax, 0.0)), 2),
            "N_limited_by": lim}


out = {}

# ---------------------------------------------------------------- arm mass
arm_mass = {}
for side in ("left", "right"):
    names = [f"arm_{side}_shoulder_pitch", f"arm_{side}_shoulder_roll", f"arm_{side}_shoulder_yaw",
             f"arm_{side}_elbow_pitch", f"arm_{side}_elbow_roll", f"arm_{side}_hand_link"]
    arm_mass[side] = {n: round(float(m.body_mass[bid(n)]), 4) for n in names}
out["arm_link_mass_kg"] = arm_mass
out["arm_total_mass_kg"] = round(sum(arm_mass["left"].values()), 3)
out["robot_total_mass_kg"] = round(float(m.body_subtreemass[bid("base")]), 3)

# ---------------------------------------------------------------- 1. MuJoCo scripted keyframes
# Layout (scripted_carry): robot faces +y; in the robot's own frame the cube is to the grasping
# side. Left grasping arm, robot frame x forward, y left. The cube face the LEFT pad presses is
# at y = +(side_off - 0.14) = +0.30 in robot frame; the payload pushes the pad back toward -y.
kf = {}
pad_l = gid("arm_left_hand_pad")
for name, q5 in KEYFRAMES_LEFT.items():
    set_pose(LEG_DEFAULT, arms_from_left(q5))
    c = d.geom_xpos[pad_l].copy()
    R = d.geom_xmat[pad_l].reshape(3, 3).copy()
    hand = bid("arm_left_hand_link")
    budget = arm_budget("left", c, hand, n_dir=[0.0, -1.0, 0.0], f_vert=CUBE_M * G / 2.0)
    # orientation of the pad about the robot's lateral axis (world y here = robot left; for the
    # carry layout robots face +y, so robot-frame y maps to world -x -- rotation about the pinch axis)
    kf[name] = {"q_left": list(map(float, q5)), "pad_centre_robot_frame_m": np.round(c, 3).tolist(),
                "pad_R": np.round(R, 3).tolist(), **budget}
# hand rotation about the pinch axis (robot-frame y) between squeeze and lift
def rot_about_y(Ra, Rb):
    Rrel = Rb @ Ra.T
    # angle of rotation projected on the y axis
    ang = np.arctan2(Rrel[0, 2] - Rrel[2, 0], Rrel[0, 0] + Rrel[2, 2])
    return float(ang)
Rs = np.array(kf["squeeze"]["pad_R"])
Rl = np.array(kf["lift"]["pad_R"])
kf["pad_rotation_about_pinch_axis_squeeze_to_lift_rad"] = round(rot_about_y(Rs, Rl), 3)
Rrel = Rl @ Rs.T
kf["pad_total_rotation_squeeze_to_lift_rad"] = round(float(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))), 3)
out["mujoco_scripted_keyframes"] = kf

# pad face alignment with the cube face (normal = robot-frame -y face of the pad should be +y)
# report the angle between each pad-box axis and the robot y axis at squeeze
set_pose(LEG_DEFAULT, arms_from_left(KEYFRAMES_LEFT["squeeze"]))
R = d.geom_xmat[pad_l].reshape(3, 3)
out["squeeze_pad_axes_vs_face_normal_deg"] = [round(float(np.degrees(np.arccos(abs(R[1, k])))), 1) for k in range(3)]
out["pad_half_sizes_m"] = np.round(m.geom_size[pad_l], 4).tolist()

# ---------------------------------------------------------------- 2. Isaac pinch arms, standing legs
# Robots face each other across the cube (y = +/-0.48 from its centre); each presses the near
# face with BOTH forearms (hands have no collider in Isaac). Payload pushes the arm backward
# (robot -x). Weight share: 0.5 kg over 4 forearm contacts.
isaac = {}
for legs_name, legs in (("standing", LEG_DEFAULT), ("crouch", LEG_CROUCH)):
    set_pose(legs, ISAAC_PINCH_ARMS)
    row = {}
    for side in ("left", "right"):
        er = bid(f"arm_{side}_elbow_roll")
        hl = bid(f"arm_{side}_hand_link")
        # forearm collider: cylinder r 0.03, half-length 0.06, centred 0.01 below the elbow-roll frame
        fore = [g for g in range(m.ngeom) if m.geom_bodyid[g] == er and m.geom_contype[g]]
        upper = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid(f"arm_{side}_shoulder_yaw") and m.geom_contype[g]]
        fc = d.geom_xpos[fore[0]].copy() if fore else None
        fa = d.geom_xmat[fore[0]].reshape(3, 3)[:, 2].copy() if fore else None
        hp = d.xpos[hl].copy()
        b = arm_budget(side, fc, er, n_dir=[-1.0, 0.0, 0.0], f_vert=CUBE_M * G / 4.0)
        row[side] = {"hand_link_origin_m": np.round(hp, 3).tolist(),
                     "forearm_cyl_centre_m": None if fc is None else np.round(fc, 3).tolist(),
                     "forearm_cyl_axis": None if fa is None else np.round(fa, 3).tolist(),
                     "upper_arm_cyl_centre_m": np.round(d.geom_xpos[upper[0]], 3).tolist() if upper else None,
                     "shoulder_m": np.round(d.xpos[bid(f'arm_{side}_shoulder_pitch')], 3).tolist(),
                     **b}
    isaac[legs_name] = row
out["isaac_pinch_arms"] = isaac

# ---------------------------------------------------------------- 3. support-from-below posture
# upper arm hanging, elbow bent 90 deg, forearm horizontal forward; payload rests ON the forearm
support = {}
for elbow in (1.2, 1.4, 1.57):
    for pitch in (0.0, -0.3):
        q5 = [pitch, 0.0, 0.0, elbow, 0.0]
        set_pose(LEG_DEFAULT, arms_from_left(q5))
        er = bid("arm_left_elbow_roll")
        fore = [g for g in range(m.ngeom) if m.geom_bodyid[g] == er and m.geom_contype[g]][0]
        fc = d.geom_xpos[fore].copy()
        hl = d.xpos[bid("arm_left_hand_link")].copy()
        # 0.5 kg tray on 4 forearms -> 1.23 N each; 2 kg tray -> 4.9 N each
        for payload in (0.5, 2.0):
            b = arm_budget("left", fc, er, n_dir=[0.0, 0.0, -1.0], f_vert=payload * G / 4.0)
            support[f"pitch{pitch}_elbow{elbow}_payload{payload}"] = {
                "forearm_centre_m": np.round(fc, 3).tolist(), "hand_m": np.round(hl, 3).tolist(),
                "max_abs_with_weight_Nm": b["max_abs_with_weight_Nm"],
                "gravity_plus_weight_share_Nm": b["gravity_plus_weight_share_Nm"]}
out["support_from_below"] = support

# ---------------------------------------------------------------- 4. crouch vs standing: statics
def stance_statics(legs, arms, label):
    set_pose(legs, arms)
    com = d.subtree_com[bid("base")].copy()
    W = float(m.body_subtreemass[bid("base")]) * G
    feet = {}
    for side in ("left", "right"):
        fb = bid(f"leg_{side}_ankle_roll")
        fg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == fb and m.geom_contype[g]][0]
        c = d.geom_xpos[fg].copy()
        R = d.geom_xmat[fg].reshape(3, 3)
        h = m.geom_size[fg]
        corners = np.array([c + R @ (h * np.array([sx, sy, sz])) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        bottom = corners[np.argsort(corners[:, 2])[:4]]
        feet[side] = {"centre": c, "xmin": float(bottom[:, 0].min()), "xmax": float(bottom[:, 0].max()),
                      "ymin": float(bottom[:, 1].min()), "ymax": float(bottom[:, 1].max()),
                      "zsole": float(bottom[:, 2].mean()), "body": fb}
    xmin = min(f["xmin"] for f in feet.values())
    xmax = max(f["xmax"] for f in feet.values())
    inside = xmin <= com[0] <= xmax
    # joint torques with W/2 per foot applied at (com_x, foot_y, sole) (COP under the COM)
    tau = d.qfrc_bias.copy()
    for side, f in feet.items():
        p = np.array([np.clip(com[0], f["xmin"], f["xmax"]), f["centre"][1], f["zsole"]])
        J = point_jac(f["body"], p)
        tau = tau - J.T @ np.array([0.0, 0.0, W / 2.0])
    base_resid = tau[:6]
    legs_t = {}
    for side in ("left", "right"):
        for j in ("hip_roll", "hip_yaw", "hip_pitch", "knee_pitch", "ankle_pitch", "ankle_roll"):
            n = f"leg_{side}_{j}_joint"
            legs_t[n] = round(float(tau[m.jnt_dofadr[jid(n)]]), 3)
    return {"label": label, "com_m": np.round(com, 3).tolist(), "support_x_m": [round(xmin, 3), round(xmax, 3)],
            "com_inside_support_x": bool(inside), "com_margin_to_heel_m": round(float(com[0] - xmin), 3),
            "com_margin_to_toe_m": round(float(xmax - com[0]), 3),
            "base_residual_after_cop_under_com": np.round(base_resid, 3).tolist(),
            "leg_torques_Nm": legs_t,
            "max_abs_leg_Nm": round(float(max(abs(v) for v in legs_t.values())), 3),
            "pelvis_z_m": round(float(d.qpos[2]), 4), "weight_N": round(W, 2)}

out["stance_statics"] = [
    stance_statics(LEG_DEFAULT, {}, "upstream standing, arms zero"),
    stance_statics(LEG_DEFAULT, ISAAC_PINCH_ARMS, "standing legs + pinch arms (Stand/Stand2/Stand3 spawn)"),
    stance_statics(LEG_CROUCH, ISAAC_PINCH_ARMS, "pinch crouch (CubeToShelf spawn), feet flat"),
]

(OUT / "static_limits.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
