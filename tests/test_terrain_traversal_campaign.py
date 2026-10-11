"""Outcome and inference contracts that prevent stopped-robot success claims."""
from dataclasses import replace
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bhl_robust.research.terrain_traversal import (TraversalSettings, gravity_aligned_points,
    infer_terrain, prepare_course, qualified_actor, terrain_command, traversal_outcome)


class TraversalContractTests(unittest.TestCase):
    def test_stopped_upright_is_not_a_goal(self):
        result = traversal_outcome(.159, False, False, False, False, 40.)
        self.assertFalse(result["success"])
        self.assertEqual(result["failure"], "timeout")

    def test_goal_followed_by_fall_fails(self):
        result = traversal_outcome(5.2, True, True, False, False, 37.)
        self.assertFalse(result["success"])
        self.assertEqual(result["failure"], "fall")

    def test_early_goal_requires_full_horizon_survival(self):
        self.assertFalse(traversal_outcome(5., True, False, False, False, 20.)["success"])
        self.assertTrue(traversal_outcome(5., True, False, False, False, 40.)["success"])

    def test_collision_is_failure_even_with_goal(self):
        self.assertFalse(traversal_outcome(5., True, False, True, False, 40.)["success"])

    def test_nonfinite_cannot_pass(self):
        self.assertFalse(traversal_outcome(float("nan"), True, False, False, False, 40.)["success"])

    def test_unknown_map_stops_without_certifying_free_space(self):
        settings = TraversalSettings()
        terrain = infer_terrain(np.empty((0, 3)), np.array([1., 0, 0, 0]), settings)
        command, diagnostic = terrain_command(terrain, settings)
        self.assertTrue(np.array_equal(command, np.zeros(3)))
        self.assertEqual(diagnostic["reason"], "unknown_support_stop")
        self.assertFalse(terrain["known"].any())
        self.assertTrue(np.isnan(terrain["height_m"]).all())

    def test_dense_flat_scan_moves_and_hazardous_plane_stops(self):
        settings = TraversalSettings()
        xx, yy = np.meshgrid(np.arange(.11, 1.60, .025), np.arange(-.49, .5, .025))
        points = np.column_stack((xx.ravel()-settings.lidar_mount_x_m,
                                  yy.ravel(), np.full(xx.size, -settings.lidar_mount_z_m)))
        terrain = infer_terrain(points, np.array([1., 0, 0, 0]), settings)
        command, diagnostic = terrain_command(terrain, settings)
        self.assertAlmostEqual(command[0], .30)
        self.assertEqual(diagnostic["reason"], "nominal")
        points[:, 2] += np.tan(np.deg2rad(12))*xx.ravel()
        terrain = infer_terrain(points, np.array([1., 0, 0, 0]), settings)
        command, diagnostic = terrain_command(terrain, settings)
        self.assertEqual(command[0], 0.)
        self.assertEqual(diagnostic["reason"], "observed_hazard_stop")

    def test_thin_line_cannot_claim_two_dimensional_plane_support(self):
        settings = TraversalSettings()
        points = np.column_stack((np.arange(.1, 1.5, .01), np.zeros(140), np.full(140, -.38)))
        terrain = infer_terrain(points, np.array([1., 0, 0, 0]), settings)
        command, _ = terrain_command(terrain, settings)
        self.assertEqual(command[0], 0.)
        self.assertFalse(terrain["known"].any())

    def test_orientation_compensation_recovers_level_ground(self):
        settings = TraversalSettings()
        angle = np.deg2rad(20)
        quaternion = np.array([np.cos(angle/2), 0, np.sin(angle/2), 0])
        from bhl_robust.research.terrain_traversal import quaternion_rotation
        rotation = quaternion_rotation(quaternion)
        expected = np.array([[.3, -.2, 0.], [.6, .1, 0.], [1., .2, 0.]])
        mount = np.array([settings.lidar_mount_x_m, 0, settings.lidar_mount_z_m])
        raw = expected @ rotation - mount
        recovered = gravity_aligned_points(raw, quaternion, settings)
        np.testing.assert_allclose(recovered, expected, atol=1e-12)

    def test_invalid_imu_and_two_dimensional_scan_rejected(self):
        with self.assertRaises(ValueError):
            infer_terrain(np.zeros((3, 2)), np.array([1., 0, 0, 0]))
        with self.assertRaises(ValueError):
            infer_terrain(np.zeros((3, 3)), np.zeros(4))

    def passing_screen(self):
        rows = []
        for terrain in ("flat", "small_steps", "ramp"):
            for group in (250000, 250001):
                rows.append({"terrain": terrain, "group": group,
                             **traversal_outcome(5.1, True, False, False, False, 40.)})
        return rows

    def test_qualification_requires_every_distinct_fixture_group(self):
        rows = self.passing_screen()
        self.assertTrue(qualified_actor(rows))
        self.assertFalse(qualified_actor(rows[:-1]))
        self.assertFalse(qualified_actor(rows[:-1]+[rows[0]]))

    def test_saved_success_flag_cannot_override_fall(self):
        rows = self.passing_screen()
        rows[0]["fell"] = True
        self.assertFalse(qualified_actor(rows))

    def test_one_timeout_disqualifies_actor(self):
        rows = self.passing_screen()
        rows[0].update(traversal_outcome(.2, False, False, False, False, 40.))
        self.assertFalse(qualified_actor(rows))

    def test_degenerate_sensor_or_threshold_configuration_rejected(self):
        for settings in (replace(TraversalSettings(), lidar_rings=1),
                         replace(TraversalSettings(), minimum_known_fraction=0.),
                         replace(TraversalSettings(), seconds=.5)):
            with self.assertRaises(ValueError):
                settings.validate()

    def test_fixtures_have_real_geometry_and_deterministic_group_variation(self):
        import xml.etree.ElementTree as ET
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flat = root/"flat.xml"
            flat.write_text('<mujoco><asset/><worldbody><geom name="floor" type="plane" size="0 0 .1"/></worldbody></mujoco>')
            steps, metadata = prepare_course(flat, root/"steps", "small_steps", 250000)
            geoms = ET.parse(steps).getroot().find("worldbody").findall("geom")
            self.assertEqual(sum(g.get("name", "").startswith("terrain_step") for g in geoms), 5)
            self.assertEqual(metadata["step_rise_m"], .01)
            ramp, ramp_meta = prepare_course(flat, root/"ramp", "ramp", 250000)
            self.assertEqual(ET.parse(ramp).getroot().find("asset/hfield").get("name"), "terrain_hfield")
            self.assertEqual(ramp_meta["grade_deg"], 3.)
            _, second = prepare_course(flat, root/"repeat", "small_steps", 250000)
            self.assertEqual(metadata, second)
            _, third = prepare_course(flat, root/"different", "small_steps", 250001)
            self.assertNotEqual(metadata["onset_m"], third["onset_m"])

    def test_frozen_protocol_resolves_only_named_source_and_model_inputs(self):
        from dataclasses import asdict
        script = Path(__file__).resolve().parents[1]/"scripts/bench/terrain_traversal_campaign.py"
        spec = importlib.util.spec_from_file_location("terrain_campaign_test", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"source"
            source.mkdir()
            (source/"fixture.py").write_text("frozen")
            inputs = root/"inputs/actors/a"
            inputs.mkdir(parents=True)
            (inputs/"deploy.yaml").write_text("config")
            (inputs/"policy.onnx").write_bytes(b"checkpoint")
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            config = {"schema": "bhl-terrain-traversal-v1", "settings": asdict(TraversalSettings()),
                      "screen_groups": [250000, 250001], "confirmation_groups": list(range(260000, 260008)),
                      "screen_terrains": ["flat", "small_steps", "ramp"],
                      "confirmation_terrains": ["flat", "small_steps", "ramp"],
                      "confirmation_arms": ["baseline", "lidar", "lidar50"], "packages": {},
                      "upstream_source": "external/Berkeley-Humanoid-Lite",
                      "source_sha256": {"fixture.py": sha(source/"fixture.py")},
                      "actors": [{"name": "a", "deploy_input": "actors/a/deploy.yaml",
                                  "checkpoint_input": "actors/a/policy.onnx", "checkpoint_sha256": sha(inputs/"policy.onnx")} ]}
            outer = {"schema_version": 1, "terrain_traversal": config,
                     "input_files": {"actors/a/deploy.yaml": {"sha256": sha(inputs/"deploy.yaml")},
                                     "actors/a/policy.onnx": {"sha256": sha(inputs/"policy.onnx")}}}
            protocol = root/"protocol.json"
            protocol.write_text(json.dumps(outer))
            with patch.dict(os.environ, {"REPO": str(source)}, clear=False):
                resolved = module.validate(protocol)
                self.assertEqual(resolved["actors"][0]["checkpoint"], str(inputs/"policy.onnx"))
                config["upstream_source"] = "../outside"
                protocol.write_text(json.dumps(outer))
                with self.assertRaisesRegex(ValueError, "unsafe relocated"):
                    module.validate(protocol)


if __name__ == "__main__":
    unittest.main()
