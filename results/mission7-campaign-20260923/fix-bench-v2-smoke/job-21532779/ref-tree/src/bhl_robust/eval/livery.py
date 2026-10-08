"""Paint schemes for rendered robots.

Two jobs, and they are different. `PALETTE` gives every robot in a race its own
flat tint, which is the right thing when the question is *which* robot fell —
hue is the label. The livery here is for the opposite case: one robot, or a crew
all doing the same task, where the question is *what the robot is doing* and a
flat tint hides every limb against every other limb.

Orange and black are not arbitrary. `#eb6834` is the second entry of the
categorical pair in `scripts/make_charts.py`, so a robot in this livery matches
the charts it appears next to in the README.

The shell/housing partition is read out of the asset rather than hand-listed.
Upstream gives every link exactly one visual mesh (group 2, `contype="0"`) and
gives only the structural segments a collision primitive (group 3): torso,
thigh, shank, upper arm, forearm, foot. A link with no collision primitive is a
bare actuator at a rotation axis. That means the rule survives an upstream
re-pin, and it fails loudly — if a future asset collides everything, the whole
robot comes out orange rather than quietly mislabelled.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Repo categorical orange (#eb6834) and a near-black that still shows a
# specular edge; pure 0,0,0 reads as a hole in the frame under the headlight.
SHELL_RGBA = (0.922, 0.408, 0.204, 1.0)
JOINT_RGBA = (0.094, 0.094, 0.106, 1.0)

# Repo categorical blue (#2a78d6). The payload has to contrast with the robot
# carrying it, and reusing the chart pair keeps the README internally
# consistent instead of introducing a third hue.
PAYLOAD_RGBA = (0.165, 0.471, 0.839, 1.0)

# Shells that are painted dark anyway, overriding the collision-primitive rule.
#
# The naming convention is that a body carries the name of the joint proximal to
# it, so `knee_pitch` is the shank and `base` is the torso. Painting those two
# dark and letting the foot come out orange puts the bright colour on the
# extremities -- feet, thighs, upper arms, forearms -- and the dark on the
# central mass, which is the read the crew clips want: at 880 px you are
# tracking where four feet land, not where a torso is.
#
# The cost is real and is the reason the first version did the opposite: an
# orange foot sits against an orange thigh across a dark shank, so the leg reads
# as two bright segments rather than one, and a foot at the bottom of its swing
# is close enough to its own contact shadow to lose an edge. Flip `_DARK_SHELLS`
# back to `("ankle_roll",)` for the old scheme.
_DARK_SHELLS = ("knee_pitch", "base")


def _shell_bodies(model: mujoco.MjModel) -> set[int]:
    """Body ids that carry a collision primitive, i.e. the printed shells."""
    out = set()
    for g in range(model.ngeom):
        if int(model.geom_contype[g]) != 0 and int(model.geom_type[g]) != mujoco.mjtGeom.mjGEOM_PLANE:
            out.add(int(model.geom_bodyid[g]))
    return out


def apply_livery(model: mujoco.MjModel, prefix: str = "") -> tuple[int, int]:
    """Paint one robot orange-on-black in place.

    Args:
        model: compiled model, mutated through `geom_rgba`.
        prefix: body-name prefix identifying the robot (`""`, `"r0_"`, ...).
            Only geoms under a body with this prefix are touched, so a crew
            can be painted one member at a time.

    Returns:
        (n_shell_geoms, n_joint_geoms) actually painted, so a caller can
        assert the partition was not empty.
    """
    shells = _shell_bodies(model)
    n_shell = n_joint = 0
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if b == 0 or not name.startswith(prefix):
            continue
        stem = name[len(prefix):]
        dark = any(stem.endswith(s) for s in _DARK_SHELLS)
        if b in shells and not dark:
            model.geom_rgba[g] = SHELL_RGBA
            n_shell += 1
        else:
            model.geom_rgba[g] = JOINT_RGBA
            n_joint += 1
    return n_shell, n_joint


def paint(model: mujoco.MjModel, geom_name: str, rgba) -> None:
    """Set one named geom's colour, ignoring absence."""
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if g >= 0:
        model.geom_rgba[g] = np.asarray(rgba, dtype=np.float32)
