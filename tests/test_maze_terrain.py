"""Static gates for B5's generated maze geometry."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


_REPO = Path(__file__).resolve().parents[1]


class MazeTerrainTests(unittest.TestCase):
    def test_complete_maze_is_relative_to_returned_terrain_origin(self):
        from bhl_robust.terrains.maze_layout import (
            BUTTON_AT,
            JUNCTIONS,
            MAZE_WALLS,
            OBSTACLES,
            maze_meshes,
        )

        size = (8.0, 8.0)
        meshes, origin = maze_meshes(size)
        np.testing.assert_allclose(origin, (4.0, 4.0, 0.0))
        self.assertEqual(
            len(meshes),
            1 + len(MAZE_WALLS) + len(JUNCTIONS) + len(OBSTACLES) + 1,
        )

        # First box after the floor is the +y corridor wall.  Its centre must
        # be +0.45 m from *this tile's* origin, not +0.45 m from world zero.
        wall_center = meshes[1].centroid
        np.testing.assert_allclose(wall_center[:2] - origin[:2], (0.0, 0.45), atol=1e-9)

        button_center = meshes[-1].centroid
        np.testing.assert_allclose(button_center - origin, BUTTON_AT, atol=1e-9)
        from bhl_robust.terrains.maze_layout import visual_overlays
        overlays = visual_overlays()
        self.assertEqual(len(overlays), len(MAZE_WALLS) + len(JUNCTIONS) + len(OBSTACLES) + 1)
        names = [item[0] for item in overlays]
        self.assertIn("button", names)
        self.assertTrue(any(c[0] != c[1] or c[1] != c[2] for *_, c in overlays))
        for mesh in meshes:
            self.assertGreaterEqual(float(mesh.bounds[0, 0]), 0.0)
            self.assertGreaterEqual(float(mesh.bounds[0, 1]), 0.0)
            self.assertLessEqual(float(mesh.bounds[1, 0]), size[0])
            self.assertLessEqual(float(mesh.bounds[1, 1]), size[1])

    def test_maze_env_uses_ground_mesh_not_scene_grid_clones(self):
        src = (_REPO / "src/bhl_robust/tasks/maze_env_cfg.py").read_text()
        self.assertIn("self.scene.terrain.terrain_generator = MAZE_TERRAINS_CFG", src)
        self.assertNotIn("_build_maze(self.scene)", src)
        sensors = (_REPO / "src/bhl_robust/sensors_rig.py").read_text()
        self.assertGreaterEqual(sensors.count('mesh_prim_paths=mesh_paths or ["/World/ground"]'), 2)

    def test_path_reaches_the_button_and_stays_in_the_corridor(self):
        from bhl_robust.terrains.maze_layout import (
            BUTTON_AT,
            CORRIDOR_Y_LIMIT,
            PATH,
            button_xy_reached,
            dead_end_xy,
            junction_indicated_heading,
        )

        self.assertEqual(PATH[-1], BUTTON_AT[:2])
        for xy in PATH:
            self.assertFalse(dead_end_xy(xy), f"{xy} is on the commanded path")
            self.assertLessEqual(abs(xy[1]), CORRIDOR_Y_LIMIT)
        self.assertTrue(button_xy_reached(BUTTON_AT[:2]))
        self.assertFalse(button_xy_reached((0.0, 0.0)))
        self.assertTrue(dead_end_xy((0.0, 0.8)))
        self.assertTrue(dead_end_xy((-2.5, 0.0)))
        self.assertGreater(junction_indicated_heading("left"), 0.0)
        self.assertLess(junction_indicated_heading("right"), 0.0)

    def test_maze_env_commands_the_button_not_random_velocity(self):
        src = (_REPO / "src/bhl_robust/tasks/maze_env_cfg.py").read_text()
        self.assertIn("MazeWaypointCommandCfg", src)
        self.assertIn("class MazeRewardsCfg", src)
        self.assertIn("class MazeTerminationsCfg", src)
        self.assertIn("progress_to_button", src)
        self.assertIn("button_reached", src)
        self.assertIn("in_dead_end", src)
        self.assertIn('"y": (-0.15, 0.15)', src)
        self.assertIn("class MazeCurriculumCfg", src)
        self.assertNotIn("terrain_levels = CurrTerm", src)
        gen = (_REPO / "src/bhl_robust/terrains/maze.py").read_text()
        self.assertIn("curriculum=False", gen)
        mdp = (_REPO / "src/bhl_robust/tasks/maze_mdp.py").read_text()
        self.assertIn("class MazeWaypointCommand", mdp)
        self.assertIn("PATH", mdp)
        self.assertIn("CRUISE_SPEED", mdp)
        self.assertIn("episode_length_buf", mdp)


if __name__ == "__main__":
    unittest.main()
