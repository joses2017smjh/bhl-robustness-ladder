"""Rough-terrain generator: noise, slopes, and low discrete obstacles.

Deliberately NO stairs. BHL is 11.3 kg and 3D-printed with 6 Nm joint limits;
a step tall enough to be a stair is very likely past what it can lift a foot
over, so including them would spend the whole curriculum on an unlearnable
sub-terrain and contaminate the levels of everything else.

Difficulty is what Isaac Lab interpolates across terrain rows: at difficulty 0
every range collapses to its low end, at difficulty 1 to its high end. Note that
`obstacle_height_range` therefore starts at **0.0** -- the obstacles literally
are not there on the easy rows and grow to 4 cm at the hardest. Discrete
obstacles are the component most likely to be unlearnable at this robot's torque,
so making their height a curriculum axis rather than a constant means the
experiment degrades gracefully instead of failing outright: if the terrain levels
stall, the level at which they stall is itself the result.

BHL's observation is blind -- no height scanner -- so this is blind rough-terrain
locomotion. That is harder, but it keeps the observation vector byte-identical to
the flat-ground policies, so the same sim2sim harness and the same sim2real
deployment path apply without modification.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

BUMPY_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,      # difficulty levels the curriculum promotes through
    num_cols=20,      # variations per level
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # required for terrain_levels_vel to have levels to move between
    sub_terrains={
        # Broadband roughness: the "bumpy road" case. Amplitude tops out at 5 cm,
        # a real perturbation for a robot with ~0.1 m ground clearance.
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.40, noise_range=(0.0, 0.05), noise_step=0.005, border_width=0.25,
        ),
        # Slopes to ~15 deg. Continuous, so they exercise balance not foot placement.
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20, slope_range=(0.0, 0.26), platform_width=2.0, border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.20, slope_range=(0.0, 0.26), platform_width=2.0, border_width=0.25,
        ),
        # Low discrete obstacles: hardest component, hence the 0 -> 4 cm ramp.
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.20, obstacle_height_mode="choice",
            obstacle_width_range=(0.2, 0.6), obstacle_height_range=(0.0, 0.04),
            num_obstacles=12, platform_width=2.0, border_width=0.25,
        ),
    },
)

# Same menu without obstacles: the fallback arm if obstacles prove unlearnable,
# and a clean ablation isolating what the obstacles cost.
SMOOTH_TERRAINS_CFG = BUMPY_TERRAINS_CFG.replace(
    sub_terrains={
        "rough": BUMPY_TERRAINS_CFG.sub_terrains["rough"].replace(proportion=0.50),
        "slope_up": BUMPY_TERRAINS_CFG.sub_terrains["slope_up"].replace(proportion=0.25),
        "slope_down": BUMPY_TERRAINS_CFG.sub_terrains["slope_down"].replace(proportion=0.25),
    }
)

# The correct obstacle ablation. `SMOOTH_TERRAINS_CFG` above redistributed the
# obstacles' 20% share into more rough ground and steeper slopes, so "smooth" is
# also *rougher on average* -- the comparison confounds "no obstacles" with
# "more of everything else". This one holds every other proportion fixed and
# replaces the obstacle share with flat ground, so the only difference from
# BUMPY is that one fifth of the tiles have nothing to step over.
FLATFILL_TERRAINS_CFG = BUMPY_TERRAINS_CFG.replace(
    sub_terrains={
        "rough": BUMPY_TERRAINS_CFG.sub_terrains["rough"],
        "slope_up": BUMPY_TERRAINS_CFG.sub_terrains["slope_up"],
        "slope_down": BUMPY_TERRAINS_CFG.sub_terrains["slope_down"],
        # A pyramid slope with a zero slope range is a flat tile that still
        # participates in the curriculum's row/column layout, so the generator
        # produces the same grid shape as BUMPY with one component neutralised.
        "flat": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20, slope_range=(0.0, 0.0), platform_width=2.0, border_width=0.25,
        ),
    }
)
