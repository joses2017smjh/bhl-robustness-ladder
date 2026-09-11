"""Articulation invariants for the cloth-sort robot.

The Berkeley Humanoid Lite keeps its normal 22-DoF welded-hand body. "Experimental
arm" in this project means an experimental *condition*, never a third physical
limb. These checks exist so a later edit cannot silently add joints or arms.
"""

from __future__ import annotations

from bhl_robust.limb_partition import ARM_JOINTS, JOINTS_22, LEG_JOINTS

EXPECTED_JOINTS = 22
EXPECTED_ARMS = 2
EXPECTED_ARM_JOINTS = 10
EXPECTED_LEG_JOINTS = 12

# The shipped asset welds both hands shut. Restoring grippers is a different
# experiment (``docs/GRIPPER.md``) and is not this task.
EXPECTED_PHYSICAL_HANDS = 2


def assert_articulation(
    joint_names: list[str] | tuple[str, ...] | None = None,
    n_arms: int = EXPECTED_ARMS,
) -> None:
    """Raise if the robot is not the standard 22-DoF, two-arm humanoid."""
    names = list(joint_names) if joint_names is not None else list(JOINTS_22)
    if len(names) != EXPECTED_JOINTS:
        raise AssertionError(
            f"cloth-sort robot must stay at {EXPECTED_JOINTS} DoF, got {len(names)}"
        )
    if set(names) != set(JOINTS_22):
        missing = sorted(set(JOINTS_22) - set(names))
        extra = sorted(set(names) - set(JOINTS_22))
        raise AssertionError(f"joint set changed: missing={missing} extra={extra}")
    n_arm_j = sum(1 for n in names if n.startswith("arm_"))
    n_leg_j = sum(1 for n in names if n.startswith("leg_"))
    if n_arm_j != EXPECTED_ARM_JOINTS or n_leg_j != EXPECTED_LEG_JOINTS:
        raise AssertionError(
            f"expected {EXPECTED_ARM_JOINTS} arm + {EXPECTED_LEG_JOINTS} leg "
            f"joints, got {n_arm_j} + {n_leg_j}"
        )
    if n_arms != EXPECTED_ARMS:
        raise AssertionError(
            f"cloth-sort must not add a physical arm; expected {EXPECTED_ARMS}, got {n_arms}"
        )
    if any("gripper" in n for n in names):
        raise AssertionError(
            "cloth-sort uses the welded-hand 22-DoF asset; grippers are a different experiment"
        )
    # Silence unused-import lints: the lists are the source of truth.
    assert len(ARM_JOINTS) == EXPECTED_ARM_JOINTS
    assert len(LEG_JOINTS) == EXPECTED_LEG_JOINTS
