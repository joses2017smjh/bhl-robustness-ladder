import numpy as np
import pytest
from bhl_robust.research.stereo_consistency import photometric_confidence, project_lidar, fuse


def test_textured_correct_match_has_cost_margin_and_wrong_match_rejected():
    rng = np.random.default_rng(13)
    left = rng.integers(0, 256, (40, 100), dtype=np.uint8)
    right = np.zeros_like(left); right[:, :-8] = left[:, 8:]
    valid = np.ones(left.shape, bool); disparity = np.full(left.shape, 8., np.float32)
    confidence, accepted, _ = photometric_confidence(left, right, disparity, valid)
    assert accepted[8:-8, 20:-8].mean() > .99
    assert confidence[8:-8, 20:-8].mean() > .99
    _, wrong, _ = photometric_confidence(left, right, disparity+3., valid)
    assert wrong[8:-8, 20:-8].mean() < .1


def test_uniform_and_invalid_patches_cannot_gain_confidence():
    image = np.full((40,100), 128, np.uint8)
    for support in (np.ones(image.shape, bool), np.zeros(image.shape, bool)):
        confidence, accepted, _ = photometric_confidence(image, image, np.full(image.shape, 8.), support)
        assert not accepted.any() and not confidence.any()


def test_projection_uses_nearest_return_and_no_behind_camera_points():
    calibration = dict(T_C_L=np.eye(4), image_width=20, image_height=10, fx_px=10., fy_px=10., cx_left_px=10., cy_px=5.)
    depth, valid = project_lidar(np.array([[0,0,4], [0,0,2], [0,0,-1], [20,0,1]]), calibration)
    assert valid.sum() == 1 and depth[5,10] == 2.


def test_inconsistent_measurements_become_unknown_without_hole_fill():
    stereo = np.array([[1., 1., 1., np.nan]])
    lidar = np.array([[3., 1.1, np.nan, 2.]])
    confidence = np.array([[1., .5, 0., 0.]])
    depth, valid, conflict = fuse(stereo, np.isfinite(stereo), lidar, np.isfinite(lidar), confidence, gated=True)
    assert np.array_equal(valid, [[False, True, False, True]])
    assert np.array_equal(conflict, [[True, False, False, False]])
    assert np.isnan(depth[0,0]) and depth[0,3] == 2.
    assert depth[0,1] == pytest.approx((.5+1.1)/1.5)
    simple, simple_valid, _ = fuse(stereo, np.isfinite(stereo), lidar, np.isfinite(lidar), confidence, gated=False)
    assert simple_valid.all() and simple[0,0] == 2.


def test_occlusion_hole_invalidates_neighbor_cost_patch():
    rng = np.random.default_rng(7); left = rng.integers(0,256,(40,100),dtype=np.uint8)
    right = np.zeros_like(left); right[:, :-8] = left[:, 8:]
    valid = np.ones(left.shape,bool); valid[20,40] = False
    _, accepted, _ = photometric_confidence(left,right,np.full(left.shape,8.),valid)
    assert not accepted[18:23,38:43].any()
