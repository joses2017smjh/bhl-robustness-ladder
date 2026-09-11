"""Joint safety limits for sweep trajectories.

These are conservative walls around the Berkeley Humanoid Lite's published
ranges, not a second copy of the URDF. The controller refuses a plan that
would command a joint past them rather than discovering the limit in contact.
The robot articulation itself is not changed here — see ``robot.py``.
"""

from __future__ import annotations

from bhl_robust.limb_partition import ARM_JOINTS, JOINTS_22, LEG_JOINTS

# Approximate URDF walls. Slightly inside the real stops so a numerical
# interpolation cannot sit on the limit and chatter.
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "arm_left_shoulder_pitch_joint": (-1.70, 1.70),
    "arm_left_shoulder_roll_joint": (-0.50, 1.70),
    "arm_left_shoulder_yaw_joint": (-1.40, 1.40),
    "arm_left_elbow_pitch_joint": (0.00, 2.20),
    "arm_left_elbow_roll_joint": (-1.40, 1.40),
    "arm_right_shoulder_pitch_joint": (-1.70, 1.70),
    "arm_right_shoulder_roll_joint": (-1.70, 0.50),
    "arm_right_shoulder_yaw_joint": (-1.40, 1.40),
    "arm_right_elbow_pitch_joint": (-2.20, 0.00),
    "arm_right_elbow_roll_joint": (-1.40, 1.40),
    "leg_left_hip_roll_joint": (-0.40, 0.40),
    "leg_left_hip_yaw_joint": (-0.70, 0.70),
    "leg_left_hip_pitch_joint": (-1.40, 0.60),
    "leg_left_knee_pitch_joint": (0.00, 2.00),
    "leg_left_ankle_pitch_joint": (-0.90, 0.70),
    "leg_left_ankle_roll_joint": (-0.40, 0.40),
    "leg_right_hip_roll_joint": (-0.40, 0.40),
    "leg_right_hip_yaw_joint": (-0.70, 0.70),
    "leg_right_hip_pitch_joint": (-1.40, 0.60),
    "leg_right_knee_pitch_joint": (0.00, 2.00),
    "leg_right_ankle_pitch_joint": (-0.90, 0.70),
    "leg_right_ankle_roll_joint": (-0.40, 0.40),
}

assert set(JOINT_LIMITS) == set(JOINTS_22)
assert set(ARM_JOINTS).issubset(JOINT_LIMITS)
assert set(LEG_JOINTS).issubset(JOINT_LIMITS)


def clip_joint(name: str, value: float) -> float:
    lo, hi = JOINT_LIMITS[name]
    return float(min(max(value, lo), hi))


def is_valid_joints(joint_pos: dict[str, float], margin: float = 0.0) -> bool:
    """False if any named joint is outside its wall, expanded by ``margin``."""
    for name, value in joint_pos.items():
        if name not in JOINT_LIMITS:
            continue
        lo, hi = JOINT_LIMITS[name]
        if value < lo - margin or value > hi + margin:
            return False
    return True


def missing_or_extra_joints(joint_pos: dict[str, float]) -> tuple[list[str], list[str]]:
    have = set(joint_pos)
    want = set(JOINTS_22)
    return sorted(want - have), sorted(have - want)


# --------------------------------------------------------------- orientation

def relative_up_z(q, q0):
    """``R[2, 2]`` of the rotation taking pose ``q0`` to pose ``q``.

    1.0 means "same tilt as the reference pose"; -1.0 means inverted. Both
    arguments are ``(..., 4)`` quaternions in ``(w, x, y, z)``.

    Deliberately plain arithmetic — indexing, multiply, add — so the *same*
    function serves the numpy tests here and the torch tensors in
    ``bhl_robust.tasks.cloth_sort_mdp``. A second copy of this algebra is
    exactly how a sign error survives.

    Why relative rather than absolute: this asset's identity orientation is not
    upright. The configured stand-up quaternion ``(0, 0, 1, 0)`` — the one the
    spawn photographs show standing — has ``R[2, 2] = -1``, so an absolute
    ``R[2,2] < 0.70`` fall test fires at reset and ends every episode on step
    one (21233866). Measured against the spawn pose this reads 0 by
    construction, whatever the identity frame means.
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w0, x0, y0, z0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
    # rel = q ⊗ conj(q0); only the x and y components are needed for R[2, 2].
    rx = -w * x0 + x * w0 - y * z0 + z * y0
    ry = -w * y0 + x * z0 + y * w0 - z * x0
    return 1.0 - 2.0 * (rx * rx + ry * ry)
