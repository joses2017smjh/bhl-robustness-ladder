"""Sweep plan → joint-position trajectory.

Reuses the pinch/squat pose the rest of this repo already measured, rather
than inventing a second whole-body controller. Arm joints are offset from that
pose by the planar sweep; legs stay in the squat that puts the hands at
``GRASP_Z``. Trajectories that leave the safety walls are marked invalid
instead of being clipped silently — an invalid rate is a metric, a quiet clip
is a lie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bhl_robust.cloth.kinematics import JOINT_LIMITS, clip_joint, is_valid_joints
from bhl_robust.cloth.sweep import SweepPlan, hand_pose_at
from bhl_robust.cloth.reach import HAND_DROP_MAX, can_reach_xy
from bhl_robust.limb_partition import JOINTS_22

#: Crouch + side-hold from ``coop_lift_env_cfg._PINCH_JOINT_POS``. Copied, not
#: imported: that module pulls in isaaclab and cannot run on a login node.
PINCH_JOINT_POS: dict[str, float] = {
    "leg_left_hip_pitch_joint": -0.85,
    "leg_right_hip_pitch_joint": -0.85,
    "leg_left_knee_pitch_joint": 1.45,
    "leg_right_knee_pitch_joint": 1.45,
    "leg_left_ankle_pitch_joint": -0.55,
    "leg_right_ankle_pitch_joint": -0.55,
    "arm_left_shoulder_roll_joint": -0.26,
    "arm_right_shoulder_roll_joint": 0.26,
    "arm_left_shoulder_pitch_joint": -0.55,
    "arm_right_shoulder_pitch_joint": 0.55,
    "arm_left_elbow_pitch_joint": 0.90,
    "arm_right_elbow_pitch_joint": -0.90,
}


def default_pose() -> dict[str, float]:
    """Standing-squat hold. Missing joints stay at zero."""
    pose = {name: 0.0 for name in JOINTS_22}
    pose.update(PINCH_JOINT_POS)
    return pose


@dataclass(frozen=True)
class JointWaypoint:
    t: float
    joints: dict[str, float]
    valid: bool


def _arm_offsets(xy: np.ndarray, z: float, plan: SweepPlan) -> dict[str, float]:
    """Map a hand waypoint onto small shoulder/elbow deltas.

    This is not inverse kinematics. It is a bounded offset from a measured
    hold pose, which is what the 4 Nm arms can actually track. The right
    arm does the sweep (the table is on +x, the robot faces +x); the left
    arm holds.
    """
    # Lateral (y) → right shoulder roll; forward progress → shoulder pitch.
    # Scaled so a 0.4 m sweep stays inside the safety walls.
    dy = float(xy[1])
    progress = float(np.dot(xy - plan.start_xy, plan.direction))
    roll = float(np.clip(0.35 * dy, -0.25, 0.25))
    pitch = float(np.clip(0.40 * progress, -0.35, 0.35))
    # Height error relative to contact: lift the elbow a little on approach/retract.
    lift = float(np.clip((z - plan.contact_height) * 1.2, -0.20, 0.35))
    return {
        "arm_right_shoulder_roll_joint": 0.26 + roll,
        "arm_right_shoulder_pitch_joint": 0.55 + pitch,
        "arm_right_shoulder_yaw_joint": float(np.clip(0.20 * dy, -0.30, 0.30)),
        "arm_right_elbow_pitch_joint": -0.90 + lift,
    }


def joints_at(plan: SweepPlan, t: float) -> dict[str, float]:
    """Joint targets at time ``t`` into ``plan``, clipped to the safety walls."""
    xy, z = hand_pose_at(plan, t)
    pose = default_pose()
    pose.update(_arm_offsets(xy, z, plan))
    return {name: clip_joint(name, value) for name, value in pose.items()}


def is_plan_valid(plan: SweepPlan, dt: float = 0.05, check_reach: bool = True) -> bool:
    """Sample the trajectory; False if any sample needed clipping past a wall,
    or if the hand cannot get over a waypoint at all.

    ``joints_at`` always clips. Validity is "the unclipped command was already
    inside the walls", which is the thing we want to log as invalid_trajectory.

    ``check_reach`` is the part this module went without. Joint walls alone
    say nothing about *where* the hand ends up, so every plan in the original
    layout passed while the garment it swept sat 0.74 m from a hand that
    reaches 0.29 m. Each waypoint must now lie over the measured workspace
    (``bhl_robust.cloth.reach``).
    """
    if plan.distance > plan.sweep_duration * 2.5:
        # Speed/distance combination the arms cannot track.
        return False
    if plan.distance > 0.95 or plan.speed > 1.25:
        return False
    t = 0.0
    while t <= plan.duration + 1e-9:
        xy, z = hand_pose_at(plan, t)
        pose = default_pose()
        pose.update(_arm_offsets(xy, z, plan))
        if not is_valid_joints(pose):
            return False
        if z < 0.05:
            return False
        if check_reach and not can_reach_xy(xy, z, z + HAND_DROP_MAX):
            return False
        t += dt
    return True


def sample_trajectory(plan: SweepPlan, dt: float = 0.05) -> list[JointWaypoint]:
    """Dense joint trajectory for replay or an Isaac action term."""
    out: list[JointWaypoint] = []
    t = 0.0
    valid = is_plan_valid(plan, dt)
    while t <= plan.duration + 1e-9:
        out.append(JointWaypoint(t=t, joints=joints_at(plan, t), valid=valid))
        t += dt
    return out


def assert_limits_cover_pinch() -> None:
    """The hold pose itself must sit inside the walls, or every plan is invalid."""
    pose = default_pose()
    for name, value in pose.items():
        lo, hi = JOINT_LIMITS[name]
        if not (lo <= value <= hi):
            raise AssertionError(f"{name} pinch {value} outside [{lo}, {hi}]")
