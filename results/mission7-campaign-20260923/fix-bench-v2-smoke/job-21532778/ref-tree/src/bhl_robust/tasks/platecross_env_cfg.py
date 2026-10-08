"""Velocity-BHL-Arms-PlateCross-v0: Mission 7 learned crossing (workstream m7-platecross, 2026-10-02).

Frozen design: SLURM_JOBS.md, 'User approval recorded 2026-10-02 14:15', item (B). Both scripted crossing
plans failed Mission 7's plate bench (M2 8/64, M3 16/41); this task fine-tunes the qualified turning
checkpoint arms-turngait-clock-s2 (Velocity-BHL-Arms-TurnGaitClock-v0, "R1") on Mission 7's own plates.

    HumanoidPlateCrossCfg = arms_env_cfg.HumanoidTurnGaitClockCfg (R1) UNCHANGED -- rewards, command mix,
    observations (actor 77 with the gait clock, critic 80), events including the fixed +/-0.5 m/s pushes,
    domain randomization -- EXCEPT the terrain:

      scene.terrain.terrain_type           "plane" -> "generator"
      scene.terrain.terrain_generator      None -> PLATES_TERRAINS_CFG (flat ground + Mission 7's plates)
      scene.terrain.max_init_terrain_level 5 -> None (spawn on every tile; the tiles are identical, so
                                           levels mean nothing, and on a plane the field was unused)
      scene.terrain.visual_material        Nucleus shingle MDL -> None (visual only; a generated mesh gets
                                           no MDL, as HumanoidBumpyEnvCfg / BipedBumpyEnvCfg do)

The ground's physics material (static = dynamic friction 1.0, restitution 0, multiply) is unchanged and
covers the plates too, as Mission 7's MuJoCo floor and plates share MuJoCo's default friction. The plate
geometry and the declared field (pitch 1.2 m, 8.4 m tiles, checkerboard, centre clear, 48 plates per tile)
live in the Isaac-free `platecross_terrain.py`, which cites Mission 7's source lines; they are copied into
PlatesTerrainCfg fields so every run's params/env.yaml records the numbers it trained with.

Blind policy: no height scan, so the 77 actor observations are R1's and the export / deploy path is R1's
(bhl_robust.eval.gait_clock). Do NOT edit arms_env_cfg.py or gait_clock_mdp.py from here: this module
only imports them.
"""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from bhl_robust.tasks import platecross_terrain as _pt
from bhl_robust.tasks.arms_env_cfg import HumanoidTurnGaitClockCfg

TASK_ID = "Velocity-BHL-Arms-PlateCross-v0"


@configclass
class PlatesTerrainCfg(SubTerrainBaseCfg):
    """One tile of flat ground with Mission 7's plates on the declared lattice (platecross_terrain)."""

    function: Callable = _pt.plates_terrain
    round_radius_m: float = _pt.ROUND_RADIUS_M
    square_side_m: float = _pt.SQUARE_SIDE_M
    height_m: float = _pt.PLATE_HEIGHT_M
    pitch_m: float = _pt.PITCH_M
    disc_sections: int = _pt.DISC_SECTIONS
    clear_centre: bool = _pt.CLEAR_CENTRE


PLATES_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(_pt.TILE_M, _pt.TILE_M),
    border_width=_pt.BORDER_M,
    num_rows=_pt.NUM_ROWS,
    num_cols=_pt.NUM_COLS,
    curriculum=False,
    seed=_pt.TERRAIN_SEED,
    use_cache=False,
    sub_terrains={"plates": PlatesTerrainCfg(proportion=1.0)},
)


@configclass
class HumanoidPlateCrossCfg(HumanoidTurnGaitClockCfg):
    """R1 (Velocity-BHL-Arms-TurnGaitClock-v0) on flat ground scattered with Mission 7's plates."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = PLATES_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.visual_material = None
