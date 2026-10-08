"""A payload with handles, sized so that one robot cannot lift it and two can.

Three measured facts drive every number here.

**Mass was never the constraint.** Static holding torque at the limiting arm
joint -- shoulder roll, moment arm 0.257 m in the pinch pose, 4 Nm limit -- puts
one arm at 1.59 kg hanging, one robot at 3.17 kg, a pair at 6.35 kg. The cube is
0.5 kg. It is twelve times inside budget and has never once been lifted.

**The constraint is the squeeze.** With no fingers, a side pinch holds the load
by friction, and friction needs inward normal force. That force's reaction
pushes each robot *outward*, away from its own base of support, so the harder a
policy grips the more it destabilises itself. Every arm that formed a good pinch
either never lifted or braced against the floor to survive the reaction. A
handle deletes that trade entirely: the load hangs off the limb, gravity does
the retaining, and the arms spend their torque budget on holding rather than on
fighting their own grip reaction.

**So mass becomes free to use as the cooperation requirement.** `TOTE_MASS` sits
between one robot's ceiling and a pair's: 4.0 kg is 1.26x what a single robot
can hold and 0.63x what two can. One robot physically cannot do it, however good
its policy; two have a 1.6x margin. That is a stronger guarantee than the ball's
geometry argument, which failed -- a 0.36 m ball fits inside one robot's 0.355 m
hand span, so the ball task's "cooperation" was never enforced by anything.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from bhl_robust.reach_band import GRASP_Z

#: Measured ceilings, from `slurm/inner/_payload_sizing.sh`.
ONE_ARM_KG = 1.59
ONE_ROBOT_KG = 3.17
PAIR_KG = 6.35

#: Between the two, so cooperation is a physical requirement and not a hint.
TOTE_MASS = 4.0

#: Body of the tote. Narrow enough in y that a robot's hands straddle it.
TOTE_SIZE = (0.30, 0.26, 0.24)

#: Handle: a bar standing proud of each x face, with clearance beneath it for
#: the hand link to pass under and bear upward. Radius is small relative to the
#: hand so the contact is a hook rather than a pinch.
HANDLE_RADIUS = 0.015
HANDLE_LENGTH = 0.20          # inside one robot's 0.355 m hand span
HANDLE_STANDOFF = 0.075       # clear space between bar and body, for the hand
HANDLE_RGBA = (0.92, 0.41, 0.20)


def tote_spawn() -> sim_utils.MultiAssetSpawnerCfg | sim_utils.CuboidCfg:
    """The tote body. Handles are separate rigid bodies welded on by the cfg.

    Kept as a plain cuboid rather than a mesh so the collision geometry is
    exactly what it looks like -- section 6 spent a job establishing that convex
    decomposition of a mesh changes contact behaviour, and a payload whose
    collider differs from its render is the last thing this task needs.
    """
    return sim_utils.CuboidCfg(
        size=TOTE_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=TOTE_MASS),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.165, 0.471, 0.839)),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=0.9),
    )


def hookable(hand_thickness: float = 0.05) -> bool:
    """Is there room under the bar for a hand to hook?

    The standoff has to exceed the hand's thickness or the bar is a bump on a
    wall rather than a handle, and the task silently reverts to the friction
    pinch this design exists to escape.
    """
    return HANDLE_STANDOFF > hand_thickness + 0.01


def cooperation_required() -> bool:
    """True when the mass is above one robot's ceiling and below a pair's."""
    return ONE_ROBOT_KG < TOTE_MASS < PAIR_KG


def margins() -> dict[str, float]:
    return {
        "vs one robot": TOTE_MASS / ONE_ROBOT_KG,
        "vs a pair": TOTE_MASS / PAIR_KG,
        "grasp height": GRASP_Z,
    }
