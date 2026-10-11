"""The Waiter program's robot (docs/WAITER_PROGRAM.md, phase 1): 24 DoF, palm colliders, arm limits 8 Nm.

Opt-in, like `gripper_asset`: `HUMANOID_LITE_CFG` and every existing task stay as they were. The URDF is written by
`scripts/waiter/make_waiter_urdf.py` and converted to USD by `slurm/inner/waiter_asset.sh`.

Joint order: upstream's 22 (arms 0-9, legs 10-21) with the two grippers appended (22, 23), as `gripper_asset`.
"""

from __future__ import annotations

import os

WAITER_ASSET_DIR = os.environ.get("BHL_WAITER_ASSET_DIR", "/nfs/hpc/share/sanchej7/Humanoid_Lite/assets/waiter")
WAITER_URDF = os.path.join(WAITER_ASSET_DIR, "berkeley_humanoid_lite_waiter.urdf")
WAITER_USD_DIR = os.path.join(WAITER_ASSET_DIR, "usd")

ARM_JOINTS = [
    "arm_left_shoulder_pitch_joint", "arm_left_shoulder_roll_joint", "arm_left_shoulder_yaw_joint",
    "arm_left_elbow_pitch_joint", "arm_left_elbow_roll_joint",
    "arm_right_shoulder_pitch_joint", "arm_right_shoulder_roll_joint", "arm_right_shoulder_yaw_joint",
    "arm_right_elbow_pitch_joint", "arm_right_elbow_roll_joint",
]
LEG_JOINTS = [
    "leg_left_hip_roll_joint", "leg_left_hip_yaw_joint", "leg_left_hip_pitch_joint", "leg_left_knee_pitch_joint",
    "leg_left_ankle_pitch_joint", "leg_left_ankle_roll_joint",
    "leg_right_hip_roll_joint", "leg_right_hip_yaw_joint", "leg_right_hip_pitch_joint", "leg_right_knee_pitch_joint",
    "leg_right_ankle_pitch_joint", "leg_right_ankle_roll_joint",
]
GRIPPER_JOINTS = ["arm_left_gripper_joint", "arm_right_gripper_joint"]
JOINT_ORDER = ARM_JOINTS + LEG_JOINTS + GRIPPER_JOINTS            # 24
UPPER_BODY_JOINTS = ARM_JOINTS + GRIPPER_JOINTS                     # 12, the commanded ones

#: Arm joint limits (rad), from the URDF (identical to upstream's).
ARM_LIMITS = {
    "arm_left_shoulder_pitch_joint": (-1.5708, 0.785398), "arm_left_shoulder_roll_joint": (-0.261799, 1.309),
    "arm_left_shoulder_yaw_joint": (-0.785398, 0.785398), "arm_left_elbow_pitch_joint": (0.0, 1.5708),
    "arm_left_elbow_roll_joint": (-0.785398, 0.785398),
    "arm_right_shoulder_pitch_joint": (-0.785398, 1.5708), "arm_right_shoulder_roll_joint": (-1.309, 0.261799),
    "arm_right_shoulder_yaw_joint": (-0.785398, 0.785398), "arm_right_elbow_pitch_joint": (-1.5708, 0.0),
    "arm_right_elbow_roll_joint": (-0.785398, 0.785398),
}

#: Declared actuator changes (MODIFIED ASSET). Legs and ankles stay upstream's (6 Nm, 20 / 2).
ARM_EFFORT_NM = 8.0          # upstream training cap 4; motors peak ~25-32 Nm; upstream MJCF 20
ARM_STIFFNESS = 20.0         # upstream 10
ARM_DAMPING = 2.0
GRIPPER_EFFORT_NM = 2.0      # assumed (gripper_asset); replaced by a measured value when available
GRIPPER_STIFFNESS = 20.0
GRIPPER_DAMPING = 1.0
GRIPPER_OPEN_RAD = 0.0
GRIPPER_CLOSED_RAD = 1.20


def find_waiter_usd(root: str = WAITER_USD_DIR) -> str:
    """The converter nests its output and may write .usda for a .usd name: resolve by search (as gripper_asset)."""
    for ext in (".usda", ".usd"):
        for dirpath, _, files in os.walk(root):
            for f in sorted(files):
                if f.startswith("berkeley_humanoid_lite_waiter") and f.endswith(ext):
                    return os.path.join(dirpath, f)
    return os.path.join(root, "berkeley_humanoid_lite_waiter.usd")


def make_waiter_cfg(usd_path: str | None = None):
    """`HUMANOID_LITE_CFG` on the Waiter USD, with the declared arm and gripper actuators."""
    from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_CFG
    from isaaclab.actuators import ImplicitActuatorCfg

    path = usd_path or find_waiter_usd()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Waiter USD not found at {path}: run slurm/repo20260923/gpu_waiter_asset.sbatch")
    cfg = HUMANOID_LITE_CFG.copy()
    cfg.spawn = cfg.spawn.replace(usd_path=path)
    joint_pos = dict(cfg.init_state.joint_pos)
    for j in GRIPPER_JOINTS:
        joint_pos[j] = GRIPPER_OPEN_RAD
    cfg.init_state = cfg.init_state.replace(joint_pos=joint_pos)
    actuators = dict(cfg.actuators)
    arms = actuators["arms"]
    actuators["arms"] = arms.replace(effort_limit=ARM_EFFORT_NM, stiffness=ARM_STIFFNESS, damping=ARM_DAMPING)
    actuators["grippers"] = ImplicitActuatorCfg(
        joint_names_expr=["arm_.*_gripper_joint"], effort_limit=GRIPPER_EFFORT_NM, velocity_limit=6.0,
        stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING)
    cfg.actuators = actuators
    return cfg
