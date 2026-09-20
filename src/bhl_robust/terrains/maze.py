"""One maze per generated terrain tile.

The first B5 scene cloned wall assets under ``{ENV_REGEX_NS}``.  Those clones
land at the regular scene-grid origins, while a terrain-generator locomotion
task resets each robot onto ``scene.env_origins`` (the selected terrain-tile
origins).  This is the same frame split that put B3's ice 72 m away.  It is
also invisible to both ray sensors: they cast only against ``/World/ground``.

This module makes the floor and every maze obstacle one sub-terrain mesh.
TerrainGenerator translates that mesh and its returned centre together, so a
robot, the collision geometry, lidar and stereo all use the same frame.
"""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils.configclass import configclass

from bhl_robust.terrains.maze_layout import maze_meshes


def maze_terrain(
    difficulty: float,
    cfg: "MazeTerrainCfg",
):
    """Return a flat tile with the complete maze fused into its mesh list.

    ``difficulty`` is unused: every tile is the same corridor, and the maze
    env disables terrain-level promotion so a locomotion curriculum cannot
    be reported as a navigation score.
    """
    del difficulty
    return maze_meshes(cfg.size)


@configclass
class MazeTerrainCfg(SubTerrainBaseCfg):
    function: Callable = maze_terrain


# Match BUMPY's tile/grid shape so env allocation stays comparable, but do
# not promote rows: every maze tile is the same corridor, so terrain level
# is not a navigation score.
MAZE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    sub_terrains={"maze": MazeTerrainCfg(proportion=1.0)},
)
