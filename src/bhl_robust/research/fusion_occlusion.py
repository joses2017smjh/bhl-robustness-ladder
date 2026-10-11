"""Matched projected stereo/LiDAR fusion and evaluator-only visibility controls.

Agreement and local SAD scores are engineering gates, not uncertainty
probabilities. No aggregated SGM cost volume or depth completion is exposed.
"""
from __future__ import annotations

import time
import numpy as np
from .stereo_consistency import photometric_confidence, project_lidar, fuse

ENGINES = ('sgbm', 'foundation')
ARMS = ('lidar_sparse', 'sgbm', 'foundation', 'sgbm_simple', 'sgbm_agreement',
        'sgbm_cost', 'foundation_simple', 'foundation_agreement', 'foundation_cost')
SETTINGS = {'near_obstacle_m': 3., 'hazard_tolerance_m': .15, 'visibility_tolerance_m': .03,
            'width': 320, 'height': 240, 'rings': 16, 'columns': 128,
            'local_cost': {'block_size_px': 5, 'competitor_offset_px': 2.,
                           'minimum_cost_margin': .10, 'maximum_mean_sad': 20.}}


def score(depth, valid, truth, obstacle, *, support=None):
    """Preserve denominators and coverage losses instead of rewarding abstention."""
    depth, valid, truth, obstacle = map(np.asarray, (depth, valid, truth, obstacle))
    if len({x.shape for x in (depth, valid, truth, obstacle)}) != 1 or valid.dtype != bool or obstacle.dtype != bool:
        raise ValueError('matched arrays and boolean masks required')
    eligible = np.isfinite(truth) & (truth >= .1) & (truth <= 12.)
    if support is not None:
        support = np.asarray(support)
        if support.shape != depth.shape or support.dtype != bool: raise ValueError('boolean support required')
        eligible &= support
    valid = valid & np.isfinite(depth) & (depth >= .1) & (depth <= 12.)
    evaluated = eligible & valid
    errors = np.abs(depth[evaluated] - truth[evaluated])
    near = eligible & obstacle & (truth <= SETTINGS['near_obstacle_m'])
    detected = near & valid & (np.abs(depth-truth) <= SETTINGS['hazard_tolerance_m'])
    false_free = near & valid & (depth > truth + SETTINGS['hazard_tolerance_m'])
    unknown = near & ~valid
    denom, count, near_count = int(eligible.sum()), int(evaluated.sum()), int(near.sum())
    return {'eligible_pixels': denom, 'predicted_pixels': count, 'mae_m': float(errors.mean()) if count else None,
            'absolute_error_sum_m': float(errors.sum(dtype=np.float64)), 'coverage': count/denom if denom else None,
            'near_obstacle_pixels': near_count, 'detected_obstacle_pixels': int(detected.sum()),
            'missed_obstacle_pixels': int((near & ~detected).sum()), 'unknown_obstacle_pixels': int(unknown.sum()),
            'false_free_obstacle_pixels': int(false_free.sum()),
            'false_free_fraction': float(false_free.sum()/near_count) if near_count else None,
            'obstacle_pixel_recall': float(detected.sum()/near_count) if near_count else None}


def visibility_mask(left_depth, other_depth, fx_px, translation_x_m, *, tolerance_m=.03):
    """Evaluator: visibility of left geometry in a parallel translated camera.

    Ground-truth z buffers only. This function must never feed an inference arm.
    Nearest-pixel correspondence; out-of-frustum points remain a separate mask.
    """
    left, other = np.asarray(left_depth), np.asarray(other_depth)
    if left.shape != other.shape or left.ndim != 2 or fx_px <= 0 or tolerance_m < 0:
        raise ValueError('matched depth views and positive calibration required')
    good = np.isfinite(left) & (left >= .1) & (left <= 12.)
    yy, xx = np.indices(left.shape)
    shifted = xx - fx_px*translation_x_m/np.where(good, left, 1.)
    u = np.rint(shifted).astype(np.int64)
    inside = good & (u >= 0) & (u < left.shape[1])
    observed = other[yy, np.clip(u, 0, left.shape[1]-1)]
    inside &= np.isfinite(observed) & (observed >= .1) & (observed <= 12.)
    occluded = inside & (observed + tolerance_m < left)
    return occluded, inside


def evaluate_predictions(predictions, truth, obstacle, right_occluded, lidar_occluded, visibility_support):
    if tuple(predictions) != ARMS: raise ValueError('exact matched arm order required')
    common = np.logical_and.reduce([v for _, v in predictions.values()])
    strata = {'both_visible': visibility_support & ~right_occluded & ~lidar_occluded,
              'right_occluded': visibility_support & right_occluded,
              'lidar_occluded': visibility_support & lidar_occluded,
              'either_occluded': visibility_support & (right_occluded | lidar_occluded)}
    result = {}
    for arm, (depth, valid) in predictions.items():
        result[arm] = {'native': score(depth, valid, truth, obstacle),
                       'all_arm_common': score(depth, valid, truth, obstacle, support=common),
                       'visibility_strata': {name: score(depth, valid, truth, obstacle, support=mask) for name, mask in strata.items()}}
    # Evaluate both sides on each pair's common mask, which does not require
    # sparse LiDAR support unless sparse LiDAR is itself a comparator.
    pairs = {}
    for engine in ENGINES:
        for suffix in ('simple', 'agreement', 'cost'):
            arm = engine+'_'+suffix
            for comparator in (engine, 'lidar_sparse', engine+'_simple', engine+'_agreement'):
                if arm == comparator: continue
                support = predictions[arm][1] & predictions[comparator][1]
                pairs[arm+'__vs__'+comparator] = {
                    name: score(*predictions[name], truth, obstacle, support=support) for name in (arm, comparator)}
    return {'arms': result, 'pairwise_common': pairs,
            'visibility_pixels': {name: int(mask.sum()) for name, mask in strata.items()},
            'visibility_support_pixels': int(visibility_support.sum())}


def predict(left, right, lidar_points, rectified, full_calibration, foundation):
    """Same input arrays for every arm; no evaluator data in signature.

    Timings are actual measured shared components summed to each arm's required
    pipeline. They are not independent end-to-end camera-live measurements.
    """
    from bhl_robust.stereo_depth import estimate_rectified_depth
    started = time.perf_counter_ns()
    sparse, sparse_valid = project_lidar(lidar_points, full_calibration)
    lidar_ms = (time.perf_counter_ns()-started)/1e6
    values, timings, diagnostics = {'lidar_sparse': (sparse, sparse_valid)}, {'lidar_sparse': lidar_ms}, {}
    for engine in ENGINES:
        start = time.perf_counter_ns()
        if engine == 'sgbm':
            depth, valid, disparity = estimate_rectified_depth(left, right, rectified, num_disparities=64, block_size=5, lr_threshold_px=1.)
        else:
            depth, valid = foundation(left, right)
            disparity = np.zeros_like(depth)
            disparity[valid] = rectified.fx_px*rectified.baseline_m/depth[valid] + rectified.cx_left_px-rectified.cx_right_px
        stereo_ms = (time.perf_counter_ns()-start)/1e6
        values[engine] = (depth, valid); timings[engine] = stereo_ms
        start = time.perf_counter_ns()
        confidence, accepted, _ = photometric_confidence(left, right, disparity, valid)
        cost_ms = (time.perf_counter_ns()-start)/1e6
        diagnostics[engine+'_cost_confidence'] = confidence
        diagnostics[engine+'_cost_accepted'] = accepted
        for suffix in ('simple', 'agreement', 'cost'):
            start = time.perf_counter_ns()
            conf = confidence if suffix == 'cost' else valid.astype(np.float32)
            combined, mask, conflict = fuse(depth, valid, sparse, sparse_valid, conf, gated=suffix!='simple')
            fusion_ms = (time.perf_counter_ns()-start)/1e6
            arm = engine+'_'+suffix
            values[arm] = (combined, mask)
            timings[arm] = stereo_ms + lidar_ms + fusion_ms + (cost_ms if suffix == 'cost' else 0.)
            diagnostics[arm+'_conflict'] = conflict
    return {name: values[name] for name in ARMS}, {name: timings[name] for name in ARMS}, diagnostics
