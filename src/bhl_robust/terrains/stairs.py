"""Stairs, sized against this robot's leg rather than against a building.

`bumpy.py` excludes stairs on purpose, and its reasoning has only got stronger
since it was written: "a step tall enough to be a stair is very likely past what
it can lift a foot over, so including them would spend the whole curriculum on
an unlearnable sub-terrain and contaminate the levels of everything else."

Section 8 then measured exactly that. On the composed lab floor this robot falls
on a **2.5 cm cable** -- 9% of its 0.28 m leg -- and stalls at a **4 cm door
threshold**, 14%. Three of four 12-DoF policies ended their run at the cable.

A building-code stair is 17 cm. That is 61% of leg length. It is not a hard
curriculum level, it is a wall, and a terrain generator full of them produces a
curriculum pinned at zero that teaches nothing and tells you nothing.

So the riser range starts at zero and tops out at 5 cm -- 18% of leg length,
just above the threshold this robot already stalls on. The same graceful
degradation the obstacle height already uses: at difficulty 0 the stairs are
flat ground, and if the levels stall, the level they stall at *is* the
measurement.

Tread width is the other half and is usually forgotten. A 6 Nm biped that
cannot place a whole foot on a step is being asked to balance on an edge, which
is a different and much harder task than climbing. 0.32 m is wider than the
foot.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

# 0.12 m thigh + 0.16 m shank, the same number section 8 normalises on.
LEG_LENGTH = 0.28
MAX_RISER = 0.05          # 18% of leg length
TREAD_WIDTH = 0.32        # wider than the foot, so a step can be stood on

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # Up and down at equal proportion. Descending is not the easy direction
        # for a blind robot -- the foot arrives lower than the policy expects and
        # the first contact is a drop rather than a push -- so splitting evenly
        # measures both rather than assuming one is free.
        "stairs_up": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.0, MAX_RISER),
            step_width=TREAD_WIDTH,
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_down": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.0, MAX_RISER),
            step_width=TREAD_WIDTH,
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)

# Fallback if G-B2 shows the curriculum pinned at level 0: 3 cm is 11% of leg
# length, between the cable this robot falls on and the threshold it stalls at.
STAIRS_LOW_TERRAINS_CFG = STAIRS_TERRAINS_CFG.replace(
    sub_terrains={
        "stairs_up": STAIRS_TERRAINS_CFG.sub_terrains["stairs_up"].replace(
            step_height_range=(0.0, 0.03)),
        "stairs_down": STAIRS_TERRAINS_CFG.sub_terrains["stairs_down"].replace(
            step_height_range=(0.0, 0.03)),
    }
)
