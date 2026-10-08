"""Clip-only coloured cuboids on the maze tile, in the terrain-origin frame.

The fused ``/World/ground`` mesh has no albedo RTX will paint, so camera-sensor
clips of mazenav policies recorded as R=G=B (21355466). These overlays sit at
``scene.env_origins`` — the same frame the robot is reset onto — with collision
off, so they do not change the MDP the policies trained on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def spawn_maze_clip_color(env: "ManagerBasedRLEnv") -> int:
    """Spawn PreviewSurface cuboids at each env's terrain origin. Returns count."""
    import isaaclab.sim as sim_utils

    from bhl_robust.terrains.maze_layout import visual_overlays

    origins = env.scene.env_origins
    origins = origins.torch if hasattr(origins, "torch") else origins
    n_env = int(origins.shape[0])
    spawned = 0
    for i in range(n_env):
        o = origins[i].detach().cpu().tolist()
        for name, size, centre, rgb in visual_overlays():
            path = f"/World/maze_clip_viz/e{i}_{name}"
            cfg = sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            )
            cfg.func(
                path,
                cfg,
                translation=(o[0] + centre[0], o[1] + centre[1], o[2] + centre[2]),
            )
            spawned += 1
    return spawned
