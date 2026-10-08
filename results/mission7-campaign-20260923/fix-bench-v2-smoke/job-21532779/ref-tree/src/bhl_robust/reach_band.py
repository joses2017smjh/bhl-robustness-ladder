"""Measured reach geometry, and the one constant every redesigned task uses.

Lives at package top level, not under `tasks/`, on purpose: `tasks/__init__.py`
registers gym ids and pulls in isaaclab, which aborts unless Isaac Sim's
SimulationApp was instantiated first. These are plain floats and the bench
scripts that check them must not have to boot a simulator to read a number.

The original three lift tasks put the payload on the floor -- cube centre at
0.14 m, plank at 0.04 m. That is inside the reach of a robot that has already
fallen over, and since nothing in the reward or the terminations refers to base
height (both fall tests read orientation only), the cheapest way to earn the
reach terms was to collapse. Both cube arms drop ~41 cm within 0.2 s of the
episode starting, before touching anything, and hold that pose.

Raising the payload fixes that without adding a penalty term, provided the new
height is chosen against the actual kinematics rather than by eye. Measured by
forward kinematics over the shoulder pitch/roll and elbow pitch ranges, feet
planted (`slurm/inner/_reach.sh`):

    standing, arms only          hands span 0.404 .. 0.610 m
    collapsed 41 cm, arms only   hands span -0.006 .. 0.200 m

Which gives three bands:

    below 0.20 m      a collapsed robot can still reach it  <- the old tasks
    0.20 .. 0.40 m    standing is necessary AND the knees must bend
    0.40 .. 0.61 m    standing suffices, arms alone, no leg involvement
    above 0.61 m      unreachable

GRASP_Z sits in the middle of the second band. Ten centimetres below the
standing arm-only floor, so the legs have to do work; ten centimetres above the
collapsed ceiling, so falling over loses the payload instead of winning it.
Standing becomes instrumentally necessary, which is a stronger guarantee than a
height penalty tuned to outweigh a 15.0-weight lift bonus.
"""

from __future__ import annotations

# Hand heights above the floor, measured, feet planted.
STAND_HAND_LO = 0.404
STAND_HAND_HI = 0.610
COLLAPSE_HAND_HI = 0.200

#: Height every redesigned payload is grasped at.
GRASP_Z = 0.30

#: Margin either side, kept as named numbers so a later change is visible.
MARGIN_BELOW_STANDING = STAND_HAND_LO - GRASP_Z      # 0.104 m of knee bend
MARGIN_ABOVE_COLLAPSE = GRASP_Z - COLLAPSE_HAND_HI   # 0.100 m of clearance

#: One robot's hand-to-hand span in the pinch pose. The shoulders do not adduct
#: past this, so it caps the width of anything a single robot can bracket.
HAND_SPAN = 0.355


def plinth_height(payload_half_height: float) -> float:
    """Support height that puts a payload's centre at `GRASP_Z`."""
    return GRASP_Z - payload_half_height


def reachable_standing(z: float) -> bool:
    return STAND_HAND_LO <= z <= STAND_HAND_HI


def needs_knee_bend(z: float) -> bool:
    """True when the height is below what the arms alone can reach standing."""
    return COLLAPSE_HAND_HI < z < STAND_HAND_LO
