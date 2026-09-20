import unittest
import numpy as np
from bhl_robust.sensor_io import SensorTiming, imu_features, ssd_features


class SensorInputTests(unittest.TestCase):
    def test_imu_axes_and_staleness(self):
        out = imu_features([0, 0, 0, 1], [1, 2, 3], [0, 0, 9.81], stamp_s=1, now_s=1.02)
        np.testing.assert_allclose(out[:3], [0, 0, -1])
        np.testing.assert_allclose(out[3:6], [1, 2, 3])
        self.assertEqual(out[-1], 1)
        rolled = imu_features([2**-0.5, 0, 0, 2**-0.5], [0]*3, [0]*3, stamp_s=1, now_s=1)
        np.testing.assert_allclose(rolled[:3], [0, -1, 0], atol=1e-6)
        stale = imu_features([0, 0, 0, 1], [0]*3, [0]*3, stamp_s=1, now_s=2)
        self.assertFalse(stale.any())
        self.assertFalse(SensorTiming().fresh(2, 1))

    def test_detection_depth_masks_and_sorting(self):
        pred = {"boxes": [[0,0,8,8], [1,1,7,7], [2,2,2,3]],
                "scores": [.6,.9,.95], "labels": [1,3,4]}
        depth = np.full((8,8), 2.)
        depth[3,3] = np.nan
        out = ssd_features(pred, image_width=8, image_height=8, stamp_s=1,
                           now_s=1.02, depth_m=depth, depth_stamp_s=1.01, slots=3)
        np.testing.assert_equal(out[:,0], [3,1,0])
        np.testing.assert_equal(out[:2,6:9], [[2,1,1],[2,1,1]])
        skewed = ssd_features(pred, image_width=8, image_height=8, stamp_s=1,
                              now_s=1.1, depth_m=depth, depth_stamp_s=.95)
        np.testing.assert_equal(skewed[:2,7], 0)
        np.testing.assert_equal(skewed[:2,8], 1)

    def test_empty_and_invalid_inputs(self):
        out = ssd_features({"boxes":[],"labels":[],"scores":[]},
                           image_width=8,image_height=8,stamp_s=0,now_s=0)
        self.assertEqual(out.shape, (8,9))
        self.assertFalse(out.any())
        out = imu_features([0]*4, [0]*3, [0]*3, stamp_s=0,now_s=0)
        self.assertFalse(out.any())
        with self.assertRaises(ValueError):
            SensorTiming(max_age_s=-1)

    def test_large_finite_inputs_cannot_emit_valid_infinity(self):
        invalid = imu_features([0,0,0,1], [1e50,0,0], [0]*3, stamp_s=0, now_s=0)
        self.assertFalse(invalid.any())
        normalized = imu_features([1e200,0,0,0], [0]*3, [0]*3, stamp_s=0, now_s=0)
        np.testing.assert_allclose(normalized[:3], [0,0,1])
        self.assertTrue(np.isfinite(normalized).all())


if __name__ == "__main__":
    unittest.main()
