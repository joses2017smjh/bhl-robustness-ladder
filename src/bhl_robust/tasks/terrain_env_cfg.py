"""Rough-terrain variants of the BHL biped locomotion task.

Upstream's scene is `terrain_type="plane"` with `terrain_generator=None`, and
its `terrain_levels_vel` curriculum helper is present but unreachable (it is
bound to `curriculums`, which Isaac Lab never reads). This wires both up: a
generated terrain with difficulty rows, and the level curriculum that promotes
an environment when its robot walks far enough and demotes it when it does not.

Domain randomization is held at upstream's default (s = 1.0) so that terrain is
the isolated variable relative to the flat-ground baseline.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped.env_cfg import (
    BerkeleyHumanoidLiteBipedEnvCfg,
)
from berkeley_humanoid_lite.tasks.locomotion.velocity import mdp

from bhl_robust.terrains.bumpy import BUMPY_TERRAINS_CFG, SMOOTH_TERRAINS_CFG


@configclass
class TerrainCurriculumCfg:
    """Promote/demote each environment's terrain difficulty by distance walked."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class BipedBumpyEnvCfg(BerkeleyHumanoidLiteBipedEnvCfg):
    """Rough ground: noise + slopes + low discrete obstacles."""

    # NOTE: `curriculum`, not upstream's `curriculums`. Isaac Lab reads the
    # singular name; the plural silently does nothing.
    curriculum: TerrainCurriculumCfg = TerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BUMPY_TERRAINS_CFG
        # Start every environment on the easiest row and let the curriculum
        # earn the harder ones; seeding higher would just mass-terminate.
        self.scene.terrain.max_init_terrain_level = 0
        # The Nucleus-hosted shingle material is a flat-ground decoration and
        # projects badly onto height fields; the generator supplies its own.
        self.scene.terrain.visual_material = None


@configclass
class BipedSmoothEnvCfg(BipedBumpyEnvCfg):
    """Ablation: same terrain menu minus the discrete obstacles."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = SMOOTH_TERRAINS_CFG
