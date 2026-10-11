"""Measured stereo/LiDAR maps and explicit disagreement-gated sensor fusion."""
from __future__ import annotations
import numpy as np
from bhl_robust.research.terrain_methods import MapSettings, UncertainElevationMap


def stereo_points(depth, valid, calibration, *, stride=4):
    """Backproject predictions only; one-pixel disparity noise is an assumption."""
    depth, valid = np.asarray(depth), np.asarray(valid, bool)
    if depth.shape != valid.shape or stride < 1:
        raise ValueError("depth/support shape or sampling stride invalid")
    yy, xx = np.indices(depth.shape)
    keep = valid & np.isfinite(depth) & (depth >= .1) & (depth <= 12) & (yy % stride == 0) & (xx % stride == 0)
    z = depth[keep]
    ray = np.stack(((xx[keep]-calibration["cx_left_px"])/calibration["fx_px"],
                    (yy[keep]-calibration["cy_px"])/calibration["fy_px"], np.ones(len(z))), -1)
    points = z[:, None]*ray
    depth_sigma = z*z/(calibration["fx_px"]*calibration["baseline_m"])
    t_m_c = np.asarray(calibration["T_M_L"])@np.linalg.inv(np.asarray(calibration["T_C_L"]))
    # Project rank-one depth covariance along the backprojected ray into map z.
    variance_z = (depth_sigma*(ray@t_m_c[2, :3]))**2
    return points, variance_z, t_m_c


def measured_map(points, transform, *, point_variance=None, settings=MapSettings()):
    result = UncertainElevationMap(settings)
    result.update(points, transform, 0., point_variance_m2=point_variance)
    return result.snapshot(0.)


def fuse_maps(stereo, lidar, *, gated, settings=MapSettings(), gate_sigma=3.):
    """Single-frame same-grid fusion. Conflicting confident cells become unknown.

    Both methods use the same per-sensor ground extraction. Ungated fusion is
    an equal mean; gated fusion uses inverse variance, preserving the shared
    pose floor. This is an engineering ablation, not calibrated probability.
    """
    if stereo["known"].shape != lidar["known"].shape:
        raise ValueError("coincident map grids required")
    result = UncertainElevationMap(settings)
    a, b = stereo["known"], lidar["known"]
    both, only_a, only_b = a & b, a & ~b, b & ~a
    conflict = np.zeros_like(both)
    if gated:
        difference = np.abs(stereo["height_m"]-lidar["height_m"])
        scale = np.sqrt(stereo["variance_m2"]+lidar["variance_m2"])
        conflict = both & (difference > gate_sigma*scale)
    valid = (a | b) & ~conflict
    for mask, source in ((only_a, stereo), (only_b, lidar)):
        result.height[mask], result.variance[mask] = source["height_m"][mask], source["variance_m2"][mask]
    common = both & ~conflict
    va, vb = stereo["variance_m2"][common], lidar["variance_m2"][common]
    if gated:
        result.height[common] = (stereo["height_m"][common]/va+lidar["height_m"][common]/vb)/(1/va+1/vb)
        result.variance[common] = np.maximum(settings.translation_sigma_m**2, 1/(1/va+1/vb))
    else:
        result.height[common] = .5*(stereo["height_m"][common]+lidar["height_m"][common])
        result.variance[common] = np.maximum(settings.translation_sigma_m**2, .25*(va+vb))
    result.scans[valid], result.last_seen[valid] = 1, 0.
    result.blocked[conflict] = True
    result.last_update = 0.
    snapshot = result.snapshot(0.)
    snapshot["sensor_conflict_cells"] = int(conflict.sum())
    return snapshot


def hazard_coverage(snapshot, truth_hazard, eligible):
    positives = np.asarray(truth_hazard, bool) & np.asarray(eligible, bool)
    detected = positives & snapshot["hazard_known"] & snapshot["hazard"]
    unknown = positives & ~snapshot["hazard_known"]
    return dict(total_ground_hazard_cells=int(positives.sum()), detected_ground_hazards=int(detected.sum()),
        unresolved_ground_hazards=int(unknown.sum()),
        hazard_recall_including_unresolved=float(detected.sum()/positives.sum()) if positives.any() else None)
