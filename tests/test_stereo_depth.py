"""Known-shift image and correspondence tests for real SGBM depth estimation."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
import numpy as np
from bhl_robust.stereo_depth import (
    RectifiedCalibration, disparities_to_depth, estimate_rectified_depth, load_depth_metadata,
)


class StereoDepthTests(unittest.TestCase):
    def setUp(self):
        # Synthetic camera ONLY for numeric tests, not a hardware calibration.
        self.calib = RectifiedCalibration(320, 96, 160., .10, 160., 160., True)

    def test_actual_sgbm_recovers_known_pixel_shift(self):
        import cv2
        rng = np.random.default_rng(717)
        left = cv2.GaussianBlur(rng.integers(0, 256, (96, 320), dtype=np.uint8), (3, 3), 0)
        right = rng.integers(0, 256, left.shape, dtype=np.uint8)
        right[:, :-8] = left[:, 8:]
        depth, valid, disparity = estimate_rectified_depth(left, right, self.calib, num_disparities=32)
        center = np.zeros_like(valid)
        center[10:-10, 50:-50] = True
        self.assertGreater(float(valid[center].mean()), .90)
        self.assertAlmostEqual(float(np.median(disparity[center & valid])), 8., places=2)
        self.assertAlmostEqual(float(np.median(depth[center & valid])), 2., places=2)
        self.assertTrue(np.isnan(depth[~valid]).all())
        self.assertFalse(valid[:, :8].any())

    def test_consistency_and_explicit_pixel_mask_reject_bad_correspondence(self):
        dl = np.full((96, 320), 8.)
        dr = np.full_like(dl, -8.)
        dr[:, 92:102] = -3.  # maps to left pixels100:110
        mask = np.ones_like(dl, dtype=bool)
        mask[:, 120:130] = False
        dl[0, 200] = np.nan
        depth, valid = disparities_to_depth(dl, dr, self.calib, left_valid=mask)
        self.assertFalse(valid[:, 100:110].any())
        self.assertFalse(valid[:, 120:130].any())
        self.assertFalse(valid[0, 200])
        np.testing.assert_allclose(depth[valid], 2.)

    def test_principal_point_offset_and_depth_bounds(self):
        dl = np.full((96, 320), 10.)
        dr = -dl
        c = replace(self.calib, cx_left_px=162.)
        depth, valid = disparities_to_depth(dl, dr, c)
        np.testing.assert_allclose(depth[valid], 2.)  #16/(10-2), not16/10
        _, bounded = disparities_to_depth(dl, dr, c, max_depth_m=1.)
        self.assertFalse(bounded.any())

    def test_untextured_images_and_invalid_calibration(self):
        image = np.full((96, 320), 128, dtype=np.uint8)
        depth, valid, _ = estimate_rectified_depth(image, image, self.calib, num_disparities=32)
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(depth).all())
        for update in ({"rectified": False}, {"fx_px": float("nan")}, {"baseline_m": -1}):
            with self.assertRaises(ValueError):
                replace(self.calib, **update)
        with self.assertRaises(ValueError):
            estimate_rectified_depth(image.astype(float), image, self.calib)

    def test_ssd_metadata_keeps_capture_time_and_invalid_depth(self):
        from bhl_robust.sensor_io import ssd_features
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "left.png"
            image.write_bytes(b"fixture-image-identity")
            depth = np.full((96, 320), 2., dtype=np.float32)
            mask = np.ones_like(depth, dtype=bool)
            mask[:, :50] = False
            np.save(root / "depth.npy", depth)
            np.save(root / "mask.npy", mask)
            record = {"schema": "bhl-stereo-depth-v1", "depth_source": "stereo_estimate",
                "depth_convention": "left_camera_optical_axis_z_m", "left_image": str(image),
                "left_image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "depth_capture_time": 100., "depth_path": str(root / "depth.npy"),
                "valid_mask_path": str(root / "mask.npy"),
                "calibration": {"image_height": 96, "image_width": 320}}
            meta = root / "metadata.json"
            meta.write_text(json.dumps(record))
            loaded, stamp, source = load_depth_metadata(meta, image, 100.)
            self.assertEqual((stamp, source), (100., "stereo_estimate"))
            self.assertTrue(np.isnan(loaded[:, :50]).all())
            pred = {"boxes": [[100,20,140,60]], "scores": [.9], "labels": [1]}
            out = ssd_features(pred, image_width=320, image_height=96,
                stamp_s=100., now_s=100.05, depth_m=loaded, depth_stamp_s=stamp)
            np.testing.assert_allclose(out[0, 6:9], [2.,1.,1.])
            with self.assertRaises(ValueError):
                load_depth_metadata(meta, image, 101.)

    def test_cli_writes_timestamped_artifacts_and_rejects_skew(self):
        import cv2
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(319)
            left = rng.integers(0, 256, (96, 320), dtype=np.uint8)
            right = np.zeros_like(left)
            right[:, :-8] = left[:, 8:]
            cv2.imwrite(str(root / "left.png"), left)
            cv2.imwrite(str(root / "right.png"), right)
            calibration = root / "camera.json"
            calibration.write_text(json.dumps({"schema": "bhl-rectified-stereo-v1", **asdict(self.calib)}))
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/perception/stereo_depth.py"),
                "--left", str(root / "left.png"), "--right", str(root / "right.png"),
                "--calibration", str(calibration), "--left-time", "100", "--right-time", "100.004",
                "--num-disparities", "32", "--out", str(root / "result")]
            subprocess.run(command, check=True, capture_output=True, text=True)
            depth, stamp, source = load_depth_metadata(root / "result/metadata.json", root / "left.png", 100.)
            self.assertEqual(stamp, 100.)
            self.assertEqual(source, "stereo_estimate")
            self.assertAlmostEqual(float(np.nanmedian(depth)), 2., places=2)
            command[command.index("--right-time") + 1] = "101"
            command[-1] = str(root / "skewed")
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((root / "skewed").exists())


if __name__ == "__main__":
    unittest.main()
