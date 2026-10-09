"""Geometry/causality checks for the native stereo navigation adapter."""
import importlib.util
from collections import deque
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/bench/native_stereo_navigation_campaign.py"
SPEC = importlib.util.spec_from_file_location("native_stereo_navigation_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NativeStereoNavigationTests(unittest.TestCase):
    def test_parallel_optical_baseline_is_positive_camera_x(self):
        left, right = MODULE.body_camera_transform(1), MODULE.body_camera_transform(-1)
        relative = np.linalg.inv(left)@right
        np.testing.assert_allclose(relative[:3, 3], [.12, 0, 0], atol=1e-12)
        np.testing.assert_allclose(relative[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(left[:3, :3]@np.array([0, 0, 1]), [1, 0, 0])

    def test_camera_to_imu_uses_fixed_extrinsics_in_correct_direction(self):
        t_body_imu = np.eye(4)
        t_body_imu[:3, 3] = [.02, 0, .69]
        t_body_lidar = np.eye(4)
        t_body_lidar[:3, 3] = [.10, 0, .38]
        t_imu_lidar = np.linalg.inv(t_body_imu)@t_body_lidar
        t_world_body = np.eye(4)
        t_world_body[:3, 3] = [2., -1., .1]
        t_world_camera = t_world_body@MODULE.body_camera_transform(1)
        actual = MODULE.native_camera_to_imu(t_world_camera, t_imu_lidar)
        np.testing.assert_allclose(actual, t_world_body@t_body_imu, atol=1e-12)

    def test_fixed_cameras_preserve_full_body_roll(self):
        import mujoco
        child = mujoco.MjSpec.from_string('<mujoco><worldbody><body name="base"><freejoint/><geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>')
        MODULE.add_stereo_cameras(child)
        model = child.compile()
        data = mujoco.MjData(model)
        angle = np.deg2rad(35)
        data.qpos[3:7] = [np.cos(angle/2), np.sin(angle/2), 0, 0]
        mujoco.mj_forward(model, data)
        body_rotation = data.xmat[1].reshape(3, 3)
        for index in range(2):
            np.testing.assert_allclose(data.cam_xmat[index].reshape(3, 3), body_rotation@MODULE.R_BODY_CAMERA_GL, atol=1e-12)
        np.testing.assert_allclose(data.cam_xpos[0]-data.cam_xpos[1], body_rotation@np.array([0, .12, 0]), atol=1e-12)

    def packet(self, map_id=0, tracked=True):
        return {"schema": "bhl-orb-native-frame-v1", "timestamp_s": 2.2, "tracked": tracked,
                "tracking_state": 2 if tracked else 4, "map_id": map_id,
                "T_W_C": np.eye(4).tolist() if tracked else None, "compute_seconds": .03}

    def bridge(self, responses):
        client = MODULE.OrbNavigationClient(Path("runtime"), np.eye(4), Path("work"))
        class FixtureTransport:
            """Explicit test transport fixture; never a campaign estimator."""
            def track(self, stamp, left, right):
                value = responses.pop(0)
                value["timestamp_s"] = stamp
                return value
        client.client = FixtureTransport()
        return client

    def scan(self, stamp=2.2):
        return {"stereo": {"timestamp_s": stamp, "left": "left.png", "right": "right.png",
                           "left_sha256": "left", "right_sha256": "right"},
                "additional_sensor_compute_seconds": .012}

    def test_lost_tracking_stays_null_in_generic_controller_packet(self):
        client = self.bridge([self.packet(tracked=False)])
        response = client.track(self.scan(), {"must_not_forward": True})
        self.assertFalse(response["tracked"])
        self.assertIsNone(response["T_W_I"])
        self.assertIsNone(response["native_ORB_T_W_C"])

    def test_missing_map_identifier_stops_even_when_native_pose_exists(self):
        client = self.bridge([self.packet(map_id=None)])
        response = client.track(self.scan(), {})
        self.assertFalse(response["tracked"])
        self.assertIsNone(response["T_W_I"])
        self.assertIsNotNone(response["native_ORB_T_W_C"])

    def test_map_change_permanently_stops_and_original_pose_is_retained(self):
        client = self.bridge([self.packet(map_id=4), self.packet(map_id=7), self.packet(map_id=4)])
        first = client.track(self.scan(), {})
        second = client.track(self.scan(2.4), {})
        third = client.track(self.scan(2.6), {})
        self.assertTrue(first["tracked"])
        for response in (second, third):
            self.assertFalse(response["tracked"])
            self.assertEqual(response["map_reset_id"], 1)
            self.assertIsNone(response["T_W_I"])
            self.assertIsNotNone(response["native_ORB_T_W_C"])
        self.assertEqual(second["native_ORB_map_id"], 7)
        self.assertEqual(first["additional_sensor_compute_seconds"], .012)

    def test_texture_is_deterministic_and_does_not_change_collision_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one = MODULE.textured_world("straight", 380000, root/"one")
            two = MODULE.textured_world("straight", 380000, root/"two")
            self.assertEqual((root/"one/scene_texture.png").read_bytes(), (root/"two/scene_texture.png").read_bytes())
            original = ET.fromstring(MODULE.NAV.world_xml("straight", 380000))
            changed = ET.fromstring(one)
            for before, after in zip(original.find("worldbody").findall("geom"), changed.find("worldbody").findall("geom")):
                for key in ("name", "type", "size", "pos", "contype", "conaffinity", "group"):
                    self.assertEqual(before.get(key), after.get(key))

    def test_settings_use_original_resolution_and_pinned_feature_recipe(self):
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory)/"stereo.yaml"
            MODULE.write_settings(settings)
            reader = cv2.FileStorage(str(settings), cv2.FILE_STORAGE_READ)
            try:
                self.assertEqual(reader.getNode("Camera.width").real(), 640)
                self.assertEqual(reader.getNode("Camera.height").real(), 480)
                self.assertEqual(reader.getNode("Camera.fps").real(), 5)
                self.assertEqual(reader.getNode("ORBextractor.nFeatures").real(), 1200)
                self.assertAlmostEqual(reader.getNode("Stereo.T_c1_c2").mat()[0, 3], .12)
            finally:
                reader.release()

    def test_discarded_camera_frames_keep_their_measured_capture_cost(self):
        sensors = MODULE.StereoSensors.__new__(MODULE.StereoSensors)
        sensors.ready = deque([{"frame_timestamp_s": .2}, {"frame_timestamp_s": .4}, {"frame_timestamp_s": .6}])
        sensors.imu = [(t, np.zeros(3), np.array([0., 0., 9.81])) for t in np.arange(0., .601, .005)]
        sensors.imu_cursor = 0
        sensors.unbilled_capture_seconds = .012+.013+.014
        scan, _, dropped = sensors.newest()
        self.assertEqual(dropped, 2)
        self.assertEqual(scan["frame_timestamp_s"], .6)
        self.assertAlmostEqual(scan["additional_sensor_compute_seconds"], .039)
        self.assertEqual(sensors.unbilled_capture_seconds, 0.)
        self.assertIsNone(sensors.newest())


if __name__ == "__main__":
    unittest.main()
