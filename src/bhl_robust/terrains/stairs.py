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

**G-B2 settled the riser empirically and the first guess was still too tall.**
The riser is 5 cm, 18% of leg length. It was briefly 3 cm, on the strength of an
entry probe that ran 300 iterations, saw the terrain level pinned at 0.0000 and
concluded the first step was a wall. That probe had no control. Run the same
check against `depth-bumpy`, which is terrain this robot demonstrably walks, and
it is *also* at 0.0000 at iteration 300 -- it does not clear the probe's own 0.05
threshold until past iteration 1,500. The probe was measuring how long PPO had
been running, and 5 cm was never actually rejected.

The 3 cm menu is kept below as STAIRS_LOW_TERRAINS_CFG in case a longer probe
does reject 5 cm, so the fallback is one import away rather than a re-derivation.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

# 0.12 m thigh + 0.16 m shank, the same number section 8 normalises on.
LEG_LENGTH = 0.28
MAX_RISER = 0.05          # 18% of leg length -- see G-B2 below
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

# G-B2 rejected this 5 cm menu twice and was wrong both times. The probe ran 300
# iterations and called a terrain level of 0 proof that the first riser was a
# wall; the depth-bumpy control, on terrain the robot demonstrably walks, is
# also at 0.0000 at iteration 300 and does not clear the gate's own 0.05
# threshold until somewhere past iteration 1,500. The probe was measuring its
# own budget. 5 cm is restored and the probe now runs long enough to mean
# something -- see `scripts/bench/terrain_level.py`, which reads the trainer's
# event file and refuses to return a verdict without that control column.
#
# The 3 cm menu is kept as the fallback the probe should have had to argue for.
STAIRS_LOW_TERRAINS_CFG = STAIRS_TERRAINS_CFG.replace(
    sub_terrains={
        "stairs_up": STAIRS_TERRAINS_CFG.sub_terrains["stairs_up"].replace(
            step_height_range=(0.0, 0.03)),
        "stairs_down": STAIRS_TERRAINS_CFG.sub_terrains["stairs_down"].replace(
            step_height_range=(0.0, 0.03)),
    }
)
