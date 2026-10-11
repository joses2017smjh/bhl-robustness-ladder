import numpy as np
import pytest
from bhl_robust.research.sensor_map_methods import stereo_points, fuse_maps, hazard_coverage
from bhl_robust.research.terrain_methods import MapSettings, UncertainElevationMap


def snapshot(height, variance=.0001):
    model = UncertainElevationMap()
    model.height[:] = height
    model.variance[:] = variance
    model.scans[:] = 1
    model.last_seen[:] = 0.
    model.last_update = 0.
    return model.snapshot(0.)


def test_conflict_gating_does_not_turn_unknown_into_free_space():
    a, b = snapshot(0.), snapshot(.3)
    gated, naive = fuse_maps(a, b, gated=True), fuse_maps(a, b, gated=False)
    assert not gated['known'].any()
    assert gated['hazard'].all()
    assert naive['known'].all()
    assert np.allclose(naive['height_m'], .15)
    assert gated['sensor_conflict_cells'] == a['known'].size


def test_fusion_keeps_missing_returns_and_shared_variance_floor():
    a, b = snapshot(.01, .000001), snapshot(.012, .000001)
    fused = fuse_maps(a, b, gated=True)
    assert np.all(fused['variance_m2'] >= MapSettings().translation_sigma_m**2)
    a['known'][:] = False
    assert np.allclose(fuse_maps(a, b, gated=True)['height_m'], .012)
    b['known'][:] = False
    assert not fuse_maps(a, b, gated=True)['known'].any()


def test_stereo_variance_uses_optical_ray_projection_and_calibration():
    t = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, .75], [0, 0, 0, 1.]])
    c = dict(cx_left_px=0, cy_px=0, fx_px=10., fy_px=10., baseline_m=.1,
             T_M_L=t, T_C_L=np.eye(4))
    p, variance, transform = stereo_points(np.full((2, 2), 2.), np.ones((2, 2), bool), c, stride=1)
    assert p.shape == (4, 3)
    assert variance.tolist() == pytest.approx([0., 0., .16, .16])
    assert np.array_equal(transform, t)


def test_unknown_hazards_remain_in_recall_denominator():
    s = snapshot(0.)
    s['hazard_known'][:] = False
    metrics = hazard_coverage(s, np.ones_like(s['known']), np.ones_like(s['known']))
    assert metrics['hazard_recall_including_unresolved'] == 0.
    assert metrics['unresolved_ground_hazards'] == s['known'].size
