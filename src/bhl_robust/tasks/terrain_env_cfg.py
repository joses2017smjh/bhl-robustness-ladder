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

from bhl_robust.terrains.bumpy import (BUMPY_TERRAINS_CFG, FLATFILL_TERRAINS_CFG,
                                       SMOOTH_TERRAINS_CFG)
from bhl_robust.terrains.stairs import STAIRS_TERRAINS_CFG


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


@configclass
class BipedFlatFillEnvCfg(BipedBumpyEnvCfg):
    """Corrected obstacle ablation: obstacle share replaced by flat ground.

    `BipedSmoothEnvCfg` removed the obstacles and redistributed their 20% into
    rough and slope, which made the ablation arm rougher on average and left the
    result unquotable. Here every other proportion is held fixed.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = FLATFILL_TERRAINS_CFG


@configclass
class BipedSlipperyEnvCfg(BipedBumpyEnvCfg):
    """Uniformly low friction over the standard bumpy menu.

    The material terrain of the pair. Geometry is byte-identical to
    `BipedBumpyEnvCfg` -- the same generator, the same seed, the same height
    field -- and only the contact friction moves. That is the design: a
    ray-cast depth camera returns geometry and nothing else, so a terrain whose
    only difficulty is material is a terrain where depth *must not* help. It is
    the negative control for the depth claim, not a difficulty setting.

    Friction is contact-combined by multiplication against the terrain's own
    1.0, so the robot-side range is the effective one. Upstream randomises it
    over [0.4, 1.2]; this pins it to [0.25, 0.35], strictly below upstream's
    floor, which makes it a different regime rather than the unlucky tail of the
    existing one. The first version stopped at 0.40 -- touching that floor
    exactly -- and the gate rejected it, correctly: a range whose top sample is
    the baseline's bottom sample is the tail, however it is described. Dynamic
    sits under static, as it does for real surfaces: sliding below breakaway.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.physics_material.params["static_friction_range"] = (0.25, 0.35)
        self.events.physics_material.params["dynamic_friction_range"] = (0.18, 0.30)


@configclass
class BipedStairsEnvCfg(BipedBumpyEnvCfg):
    """Stairs up and down, risers capped at 18% of leg length. Geometry terrain.

    The other half of the depth pair: `slippery` is difficulty a depth camera
    cannot see, this is difficulty that is nothing but geometry. If ray-cast
    depth does not help here it does not help anywhere.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG

@configclass
class BipedIceEnvCfg(BipedBumpyEnvCfg):
    """B3: low-friction patches on flat ground, invisible to a depth camera.

    The sharper version of the slippery rung. `BipedSlipperyEnvCfg` drops
    friction *uniformly*, so a policy can adapt its whole gait once and be done.
    Here the friction is patchy and the ground is flat, so the hazard has a
    location and no geometry -- there is nothing for a depth camera to see, by
    construction, and `scripts/bench/ice_gate.py` checks the patches sit flush
    rather than trusting the number.

    That makes it the strongest form of the negative control. Section 6 found
    depth helping most on uniform low friction, 2.5x at three seeds, which
    inverted the prediction that depth pays on geometry and not material. If
    depth also helps here, the mechanism cannot be "it sees the hazard" -- there
    is no hazard to see -- and the remaining explanation is that looking ahead
    buys foot placement needing less friction margin everywhere.
    """

    def __post_init__(self):
        super().__post_init__()
        from bhl_robust.terrains.ice import ice_patches
        for name, patch in ice_patches().items():
            setattr(self.scene, name, patch)


@configclass
class BipedIceVisibleEnvCfg(BipedIceEnvCfg):
    """The same patches, coloured so RGB can see them and depth still cannot.

    Runs only as the arm that separates "depth helps without seeing" from "any
    camera helps once the patch is visible". Colouring the ice by default would
    have quietly turned the whole rung into an RGB experiment.
    """

    def __post_init__(self):
        super().__post_init__()
        from bhl_robust.terrains.ice import ice_patches
        for name, patch in ice_patches(visible=True).items():
            setattr(self.scene, name, patch)
