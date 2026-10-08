"""Collision-representation overlay: convex-decomposition meshes vs primitives.

Upstream's URDF (and therefore the USD Isaac Lab trains from) uses primitive
boxes and cylinders for every colliding geom. The visual meshes exist, but they
are visual-only. That is an asset-optimization decision, not a bug — and it is
exactly the decision this overlay reverses, holding every other term of the
task identical to the s=1.0 biped baseline so the collision representation is
the only variable.

The converted USD is materialised outside `external/` (same rule as the MJCF
path repair): the source tree stays re-pinnable.
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass

from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped.env_cfg import (
    BerkeleyHumanoidLiteBipedEnvCfg,
)

# Written by scripts/convert_convex_usd.py. Lives next to the venv, not in git:
# a USD with convex hulls is a build product of a URDF, not a source file.
CONVEX_USD = os.environ.get(
    "BHL_CONVEX_USD",
    "/nfs/hpc/share/sanchej7/Humanoid_Lite/assets/bhl_biped_convex/"
    "berkeley_humanoid_lite_biped.usd",
)


@configclass
class BipedConvexCollisionCfg(BerkeleyHumanoidLiteBipedEnvCfg):
    """Identical to the repo-default biped task, mesh collision instead of primitives."""

    def __post_init__(self):
        super().__post_init__()
        if not os.path.isfile(CONVEX_USD):
            raise FileNotFoundError(
                f"convex-collision USD missing: {CONVEX_USD}  "
                "(run scripts/convert_convex_usd.py first)"
            )
        self.scene.robot.spawn.usd_path = CONVEX_USD
