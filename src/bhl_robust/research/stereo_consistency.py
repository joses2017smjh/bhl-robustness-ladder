"""Disclosed SGBM + photometric-cost confidence and sparse LiDAR consistency.

This is a project hypothesis: local image SAD costs, not OpenCV's inaccessible
aggregated SGM volume, and not a reproduction of three-view semidensification.
"""
from __future__ import annotations
import numpy as np
from bhl_robust.stereo_depth import estimate_rectified_depth
from .pose_metrics import transform

SETTINGS = {'block_size_px': 5, 'competitor_offset_px': 2., 'minimum_cost_margin': .10,
            'maximum_mean_sad': 20., 'agreement_absolute_m': .15, 'agreement_relative': .05}


def photometric_confidence(left, right, disparity, valid):
    """Continuous local SAD margin on already bidirectionally consistent SGBM."""
    import cv2
    left, right = np.asarray(left), np.asarray(right)
    shape = np.asarray(disparity).shape
    if left.shape != right.shape or left.shape not in (shape, (*shape, 3)) or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError('matched original uint8 gray/RGB images required')
    if np.asarray(valid).shape != shape or np.asarray(valid).dtype != np.bool_:
        raise ValueError('boolean SGBM validity required')
    l = left.astype(np.float32) if left.ndim == 2 else cv2.cvtColor(left, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = right.astype(np.float32) if right.ndim == 2 else cv2.cvtColor(right, cv2.COLOR_RGB2GRAY).astype(np.float32)
    yy, xx = np.indices(shape, dtype=np.float32)
    disparity = np.asarray(disparity, dtype=np.float32)
    safe = np.where(np.asarray(valid) & np.isfinite(disparity), disparity, 0.)
    radius = SETTINGS['block_size_px']//2
    support = np.asarray(valid) & np.isfinite(disparity) & (xx >= radius) & (xx < shape[1]-radius) & (yy >= radius) & (yy < shape[0]-radius)
    costs = []
    for offset in (0., -SETTINGS['competitor_offset_px'], SETTINGS['competitor_offset_px']):
        xr = xx-safe-offset
        inside = (xr >= radius) & (xr < shape[1]-radius)
        # Require the whole local patch to have in-bounds, valid disparities;
        # border/occlusion holes must not receive artificial good match costs.
        patch_support = cv2.erode((inside & valid).astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        warped = cv2.remap(r, xr, yy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cost = cv2.boxFilter(np.abs(l-warped), -1, (2*radius+1, 2*radius+1), normalize=True)
        cost[~patch_support] = np.inf
        costs.append(cost)
    winner, alternatives = costs[0], np.minimum(costs[1], costs[2])
    finite = support & np.isfinite(winner) & np.isfinite(alternatives)
    margin = np.zeros(shape, np.float32)
    margin[finite] = np.clip((alternatives[finite]-winner[finite])/(alternatives[finite]+1e-6), 0., 1.)
    confidence = margin*np.clip(1.-np.where(np.isfinite(winner), winner, SETTINGS['maximum_mean_sad'])/SETTINGS['maximum_mean_sad'], 0., 1.)
    accepted = finite & (margin >= SETTINGS['minimum_cost_margin']) & (winner <= SETTINGS['maximum_mean_sad'])
    confidence[~accepted] = 0.
    return confidence, accepted, {'mean_sad': winner, 'local_cost_margin': margin}


def project_lidar(points_xyz_m, calibration):
    points = np.asarray(points_xyz_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all(): raise ValueError('finite raw XYZ required')
    matrix = transform(calibration['T_C_L'])
    camera = points@matrix[:3, :3].T + matrix[:3, 3]
    h, w = calibration['image_height'], calibration['image_width']
    z = camera[:, 2]; keep = (z >= .1) & (z <= 12.)
    camera, z = camera[keep], z[keep]
    xx = np.rint(calibration['fx_px']*camera[:, 0]/z+calibration['cx_left_px']).astype(int)
    yy = np.rint(calibration['fy_px']*camera[:, 1]/z+calibration['cy_px']).astype(int)
    keep = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
    depth = np.full((h, w), np.inf, np.float32)
    np.minimum.at(depth, (yy[keep], xx[keep]), z[keep])
    valid = np.isfinite(depth); depth[~valid] = np.nan
    return depth, valid


def fuse(stereo, stereo_valid, lidar, lidar_valid, confidence, *, gated):
    arrays = [np.asarray(v) for v in (stereo, stereo_valid, lidar, lidar_valid, confidence)]
    if len({a.shape for a in arrays}) != 1 or any(a.dtype != np.bool_ for a in (arrays[1], arrays[3])):
        raise ValueError('same-shaped depths and boolean support required')
    stereo, a, lidar, b, confidence = arrays
    a = a & np.isfinite(stereo) & (stereo > 0)
    b = b & np.isfinite(lidar) & (lidar > 0)
    if gated: a &= np.isfinite(confidence) & (confidence > 0)
    both = a & b
    conflict = both & (np.abs(stereo-lidar) > np.maximum(SETTINGS['agreement_absolute_m'], SETTINGS['agreement_relative']*np.minimum(stereo, lidar))) if gated else np.zeros_like(a)
    valid = (a | b) & ~conflict
    result = np.full(stereo.shape, np.nan, np.float32)
    result[a & ~b] = stereo[a & ~b]; result[b & ~a] = lidar[b & ~a]
    # Confidence is an image-cost score, not a variance/probability. Disclosed
    # convex weights cap stereo influence at half on agreeing dual support.
    common = both & ~conflict
    weight = np.clip(confidence[common], 0., 1.) if gated else np.ones(common.sum())
    result[common] = (weight*stereo[common]+lidar[common])/(weight+1.)
    return result, valid, conflict


def predict(left, right, lidar_points, rectified_calibration, full_calibration):
    depth, valid, disparity = estimate_rectified_depth(left, right, rectified_calibration,
        num_disparities=64, block_size=5, lr_threshold_px=1.)
    confidence, accepted, costs = photometric_confidence(left, right, disparity, valid)
    sparse, sparse_valid = project_lidar(lidar_points, full_calibration)
    filtered = np.where(accepted, depth, np.nan).astype(np.float32)
    simple, simple_valid, _ = fuse(depth, valid, sparse, sparse_valid, confidence, gated=False)
    combined, combined_valid, conflict = fuse(depth, valid, sparse, sparse_valid, confidence, gated=True)
    return {'sgbm': (depth, valid), 'sgbm_local_cost': (filtered, accepted), 'lidar_sparse': (sparse, sparse_valid),
            'simple_projected_fusion': (simple, simple_valid), 'cost_consistency_fusion': (combined, combined_valid)}, \
           {'confidence': confidence, 'conflict': conflict, **costs}
