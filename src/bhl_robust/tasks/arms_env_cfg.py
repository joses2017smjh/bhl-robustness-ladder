"""22-DoF (arms included) counterparts of the biped robustness tasks.

The biped experiments control 12 leg joints; this variant actuates all 22,
adding shoulders, elbows and the neck. Everything else -- reward terms, the
disturbance protocol, the terrain menu -- is held identical to the biped
overlays, so the arms are the only thing that changed.

Why it is worth running twice: arms are not decoration on a push-recovery task.
A humanoid rejects a lateral shove partly by swinging its arms to move angular
momentum away from the legs, and a 12-DoF biped simply cannot do that. If the
arms help, the push and terrain results should improve at matched settings; if
they do not, that is also informative, because upstream's reward set penalises
arm deviation (`joint_deviation_arms`) and may be suppressing exactly the
strategy that would help.

Upstream's humanoid config repeats the biped's `curriculums` naming bug, so
these bind to `curriculum` for the same reason.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from berkeley_humanoid_lite.tasks.locomotion.velocity.config.humanoid.env_cfg import (
    BerkeleyHumanoidLiteEnvCfg,
    CurriculumsCfg,
    EventsCfg,
)
from berkeley_humanoid_lite.tasks.locomotion.velocity import mdp

from bhl_robust.curricula.push import push_levels_adaptive
from bhl_robust.terrains.bumpy import BUMPY_TERRAINS_CFG

# Matched to the biped arms of the experiment so the two are comparable.
PUSH_INTERVAL_S = (5.0, 9.0)


@configclass
class ArmsPushEventsCfg(EventsCfg):
    """Upstream humanoid events plus the interval push it ships commented out."""

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=PUSH_INTERVAL_S,
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    )


@configclass
class ArmsPushCurriculumCfg(CurriculumsCfg):
    """Push magnitude gated on measured fall rate, identical to the biped rule."""

    push_levels = CurrTerm(
        func=push_levels_adaptive,
        params={
            "term_name": "push_robot",
            "step": 0.02,
            "min_magnitude": 0.0,
            "max_magnitude": 1.0,
            "fall_rate_target": 0.20,
        },
    )


@configclass
class ArmsTerrainCurriculumCfg(CurriculumsCfg):
    """Terrain level promotion by distance walked."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class HumanoidPushAdaptiveCfg(BerkeleyHumanoidLiteEnvCfg):
    """22 DoF, competence-gated push curriculum."""

    events: ArmsPushEventsCfg = ArmsPushEventsCfg()
    curriculum: ArmsPushCurriculumCfg = ArmsPushCurriculumCfg()


@configclass
class HumanoidBumpyEnvCfg(BerkeleyHumanoidLiteEnvCfg):
    """22 DoF on generated rough terrain with the level curriculum."""

    curriculum: ArmsTerrainCurriculumCfg = ArmsTerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BUMPY_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.visual_material = None
