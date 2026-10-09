"""Geometry baselines for a calibrated stereo/3D-lidar research pilot.

Inference functions accept sensor inputs and calibration only. Ground truth is
accepted exclusively by the evaluation functions. Depth is optical-axis Z in
metres; unobserved/invalid entries remain NaN. Projected lidar endpoints never
certify a dense free-space image. Confidence values are consistency scores,
not calibrated probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Pinhole:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_calibration(cls, calibration: Mapping) -> "Pinhole":
        if calibration.get("rectified") is not True:
            raise ValueError("rectified calibration is required")
        camera = cls(int(calibration["image_width"]), int(calibration["image_height"]),
                     float(calibration["fx_px"]), float(calibration["fy_px"]),
                     float(calibration["cx_left_px"]), float(calibration["cy_px"]))
        camera.validate()
        return camera

    def validate(self) -> None:
        if not (1 <= self.width <= 8192 and 1 <= self.height <= 8192):
            raise ValueError("invalid image dimensions")
        if not all(math.isfinite(v) for v in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("non-finite intrinsics")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")


def rigid_transform(matrix: np.ndarray) -> np.ndarray:
    """Validate T_A_B, which maps coordinates in B into coordinates in A."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-7):
        raise ValueError("invalid homogeneous transform row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("transform rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError("transform must be a proper rotation")
    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    matrix = rigid_transform(matrix)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def stressed_extrinsic(matrix: np.ndarray, rotation_y_deg: float = 0.,
                       translation_x_m: float = 0.) -> np.ndarray:
    """Apply a real calibration perturbation about the camera's Y axis."""
    matrix = rigid_transform(matrix)
    if not all(math.isfinite(x) for x in (rotation_y_deg, translation_x_m)):
        raise ValueError("non-finite extrinsic stress")
    angle = math.radians(rotation_y_deg)
    c, s = math.cos(angle), math.sin(angle)
    delta = np.array([[c, 0, s, translation_x_m], [0, 1, 0, 0],
                      [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float64)
    return delta @ matrix


def project_lidar(points_l: np.ndarray, t_c_l: np.ndarray,
                  camera: Pinhole, max_depth_m: float = 30.) -> dict:
    """Nearest-pixel projection with a Z-buffer; no interpolation/completion."""
    camera.validate()
    if not math.isfinite(max_depth_m) or max_depth_m <= 0:
        raise ValueError("invalid maximum depth")
    points_c = transform_points(points_l, t_c_l)
    valid = np.isfinite(points_c).all(axis=1)
    valid &= (points_c[:, 2] > 0) & (points_c[:, 2] <= max_depth_m)
    points_c = points_c[valid]
    # Reject image-plane outliers before integer conversion to avoid overflow.
    u = camera.fx * points_c[:, 0] / points_c[:, 2] + camera.cx
    v = camera.fy * points_c[:, 1] / points_c[:, 2] + camera.cy
    inside = (u >= -.5) & (u < camera.width - .5)
    inside &= (v >= -.5) & (v < camera.height - .5)
    points_c, u, v = points_c[inside], u[inside], v[inside]
    ix, iy = np.floor(u + .5).astype(int), np.floor(v + .5).astype(int)
    flat = iy * camera.width + ix
    buffer = np.full(camera.height * camera.width, np.inf)
    counts = np.zeros(buffer.shape, dtype=np.int64)
    np.minimum.at(buffer, flat, points_c[:, 2])
    np.add.at(counts, flat, 1)
    buffer[~np.isfinite(buffer)] = np.nan
    return {"depth_z_m": buffer.reshape(camera.height, camera.width),
            "point_count": counts.reshape(camera.height, camera.width),
            "input_points": int(len(points_l)), "projected_points": int(len(flat)),
            "occupied_pixels": int(np.count_nonzero(counts))}


def retained_indices(n: int, retention: float, seed: int) -> np.ndarray:
    """Nested masks across densities; reuse the exact indices for every arm."""
    if n < 0 or not math.isfinite(retention) or not 0 <= retention <= 1:
        raise ValueError("invalid point retention")
    permutation = np.random.default_rng(seed).permutation(n)
    return np.sort(permutation[:int(math.floor(n * retention))])


def depth_array(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must have shape (height, width)")
    depth = depth.copy()
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return depth


@dataclass(frozen=True)
class FusionConfig:
    confidence_min: float = .6
    agreement_abs_m: float = .15
    agreement_relative: float = .05
    near_threshold_m: float = 3.

    def validate(self) -> None:
        values = (self.confidence_min, self.agreement_abs_m,
                  self.agreement_relative, self.near_threshold_m)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("non-finite fusion configuration")
        if not 0 <= self.confidence_min <= 1:
            raise ValueError("invalid confidence threshold")
        if self.agreement_abs_m < 0 or self.agreement_relative < 0 or self.near_threshold_m <= 0:
            raise ValueError("invalid fusion depth threshold")


def fuse_depth(stereo: np.ndarray, lidar: np.ndarray, confidence: np.ndarray,
               config: FusionConfig = FusionConfig()) -> dict:
    """Five fixed arms: each sensor, arithmetic fusion, gate, nearest endpoint.

    The gate rejects lidar behind a confident visible stereo surface. Other
    disagreement with confident stereo is unknown; low-confidence stereo is
    replaced only where lidar actually observed an endpoint. Agreement uses
    fixed inverse-variance weights. Variance models are heuristic baselines.
    This is informed by stereo/lidar consistency work, not a reproduction of
    the published DDC/semidensification CUDA implementation.
    """
    config.validate()
    stereo, lidar = depth_array(stereo), depth_array(lidar)
    confidence = np.asarray(confidence, dtype=np.float64)
    if stereo.shape != lidar.shape or stereo.shape != confidence.shape:
        raise ValueError("depth/confidence shapes disagree")
    sv, lv = np.isfinite(stereo), np.isfinite(lidar)
    if not np.isfinite(confidence[sv]).all() or np.any((confidence[sv] < 0) | (confidence[sv] > 1)):
        raise ValueError("valid stereo confidence must be finite and in [0,1]")
    confidence = np.where(sv, confidence, 0.)
    both = sv & lv
    simple = np.where(sv, stereo, lidar)
    simple[both] = (stereo[both] + lidar[both]) / 2
    conservative = np.where(sv, stereo, lidar)
    conservative[both] = np.minimum(stereo[both], lidar[both])
    high = sv & (confidence >= config.confidence_min)
    gate = np.full(stereo.shape, np.nan)
    gate[high & ~lv] = stereo[high & ~lv]
    gate[lv & ~high] = lidar[lv & ~high]
    tolerance = config.agreement_abs_m + config.agreement_relative * np.minimum(stereo, lidar)
    agree = high & lv & (np.abs(stereo - lidar) <= tolerance)
    # Sensor sigma grows with range. These are fixed heuristic weights, not
    # uncertainty calibration fitted using evaluation truth.
    ws = 1 / np.square(.05 + .03 * stereo[agree] ** 2)
    wl = 1 / np.square(.02 + .005 * lidar[agree] ** 2)
    gate[agree] = (ws * stereo[agree] + wl * lidar[agree]) / (ws + wl)
    lidar_occluded = high & lv & (lidar - stereo > tolerance)
    gate[lidar_occluded] = stereo[lidar_occluded]
    conflict_unknown = high & lv & (stereo - lidar > tolerance)
    # Keep unknown on the other side of a disagreement. Sparse projected lidar
    # can be incorrect under timing/extrinsic faults; a confident pixel alone
    # cannot adjudicate which endpoint belongs to the camera ray.
    arms = {"stereo": stereo, "lidar": lidar, "simple": simple,
            "confidence_gated": gate, "conservative_minimum": conservative}
    scores = {name: np.where(np.isfinite(depth), np.maximum(confidence, lv.astype(float)), 0.)
              for name, depth in arms.items()}
    scores["stereo"] = confidence
    scores["lidar"] = lv.astype(float)
    return {"arms": arms, "consistency_scores": scores,
            "diagnostics": {"input_stereo_valid": int(sv.sum()),
                "input_lidar_valid": int(lv.sum()), "agreement_pixels": int(agree.sum()),
                "lidar_occluded_pixels": int(lidar_occluded.sum()),
                "conflict_unknown_pixels": int(conflict_unknown.sum()),
                "stereo_confidence_rejections": int((sv & ~high).sum())}}


def depth_metrics(prediction: np.ndarray, truth: np.ndarray,
                  near_threshold_m: float = 3., obstacle_mask: np.ndarray | None = None) -> dict:
    """Evaluate against independently acquired/known geometry Z, not input lidar.

    False free space means a valid prediction beyond the near-obstacle
    threshold at a ground-truth near-obstacle pixel. Unknown is a miss, never
    a free-space declaration. Both counts are reported to expose rejection.
    """
    prediction, truth = depth_array(prediction), depth_array(truth)
    if prediction.shape != truth.shape:
        raise ValueError("prediction and independent truth shapes disagree")
    if not math.isfinite(near_threshold_m) or near_threshold_m <= 0:
        raise ValueError("invalid near obstacle threshold")
    tv, pv = np.isfinite(truth), np.isfinite(prediction)
    common = tv & pv
    labelled = obstacle_mask is not None
    if labelled:
        obstacle_mask = np.asarray(obstacle_mask)
        if obstacle_mask.shape != truth.shape or obstacle_mask.dtype != np.bool_:
            raise ValueError("independent obstacle mask must be a matching boolean image")
    else:
        obstacle_mask = np.ones(truth.shape, dtype=bool)
    near = tv & obstacle_mask & (truth <= near_threshold_m)
    false_free = near & pv & (prediction > near_threshold_m)
    detected = near & pv & (prediction <= near_threshold_m)
    predicted_near = tv & obstacle_mask & pv & (prediction <= near_threshold_m)
    false_obstacle = predicted_near & ~near
    missed, unknown = near & ~detected, near & ~pv
    error = prediction[common] - truth[common]
    n_truth, n_near, n_common = int(tv.sum()), int(near.sum()), int(common.sum())
    return {"gt_valid_pixels": n_truth, "common_valid_pixels": n_common,
            "native_valid_pixels": int(pv.sum()), "image_pixels": int(prediction.size),
            "coverage_on_gt": n_common / n_truth if n_truth else None,
            "native_coverage": float(pv.mean()),
            "mae_m": float(np.abs(error).mean()) if n_common else None,
            "rmse_m": float(np.sqrt(np.square(error).mean())) if n_common else None,
            "abs_rel": float((np.abs(error) / truth[common]).mean()) if n_common else None,
            "near_gt_pixels": n_near, "near_detected_pixels": int(detected.sum()),
            "near_missed_pixels": int(missed.sum()), "near_unknown_pixels": int(unknown.sum()),
            "false_free_space_pixels": int(false_free.sum()),
            "false_obstacle_pixels": int(false_obstacle.sum()),
            "predicted_near_pixels": int(predicted_near.sum()),
            "obstacle_metric_scope": "independent_obstacle_mask" if labelled else "near_surface_proxy",
            "obstacle_precision": float(detected.sum() / predicted_near.sum()) if labelled and predicted_near.any() else None,
            "obstacle_recall": float(detected.sum() / n_near) if labelled and n_near else None,
            "missed_obstacle_rate": float(missed.sum() / n_near) if labelled and n_near else None,
            "false_free_space_rate": float(false_free.sum() / n_near) if labelled and n_near else None,
            "near_surface_proxy_recall": float(detected.sum() / n_near) if not labelled and n_near else None,
            "near_surface_proxy_false_free_rate": float(false_free.sum() / n_near) if not labelled and n_near else None}


def risk_coverage(prediction: np.ndarray, score: np.ndarray, truth: np.ndarray,
                  thresholds=(0., .3, .6, .9), near_threshold_m=3., obstacle_mask=None) -> list:
    prediction = depth_array(prediction)
    score = np.asarray(score, dtype=float)
    if score.shape != prediction.shape:
        raise ValueError("risk score shape disagrees")
    return [{"consistency_threshold": float(threshold),
             **depth_metrics(np.where(score >= threshold, prediction, np.nan),
                             truth, near_threshold_m, obstacle_mask)} for threshold in thresholds]


def sector_minima(ranges_m: np.ndarray, angles_rad: np.ndarray,
                  sectors: int = 36) -> np.ndarray:
    """Use every ray, including remainder rays; unknown sectors stay NaN."""
    ranges_m, angles_rad = np.asarray(ranges_m, float), np.asarray(angles_rad, float)
    if ranges_m.ndim != 1 or ranges_m.shape != angles_rad.shape or sectors < 1:
        raise ValueError("invalid sector inputs")
    valid = np.isfinite(ranges_m) & (ranges_m > 0) & np.isfinite(angles_rad)
    indices = np.floor(np.mod(angles_rad[valid] + np.pi, 2 * np.pi) * sectors / (2 * np.pi)).astype(int)
    indices = np.minimum(indices, sectors - 1)
    result = np.full(sectors, np.inf)
    np.minimum.at(result, indices, ranges_m[valid])
    result[~np.isfinite(result)] = np.nan
    return result


@dataclass(frozen=True)
class TerrainConfig:
    x_min_m: float = -3.
    x_max_m: float = 3.
    y_min_m: float = -3.
    y_max_m: float = 3.
    resolution_m: float = .15
    min_points: int = 1
    slope_hazard_deg: float = 20.
    roughness_hazard_m: float = .05
    step_hazard_m: float = .12

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        values = (self.x_min_m, self.x_max_m, self.y_min_m, self.y_max_m,
                  self.resolution_m, self.slope_hazard_deg,
                  self.roughness_hazard_m, self.step_hazard_m)
        if not all(math.isfinite(x) for x in values) or self.resolution_m <= 0:
            raise ValueError("invalid terrain configuration")
        if self.x_max_m <= self.x_min_m or self.y_max_m <= self.y_min_m or self.min_points < 1:
            raise ValueError("invalid terrain bounds/support")
        nx = int(math.ceil((self.x_max_m - self.x_min_m) / self.resolution_m))
        ny = int(math.ceil((self.y_max_m - self.y_min_m) / self.resolution_m))
        if nx * ny > 1_000_000 or min(nx, ny) < 1:
            raise ValueError("terrain grid too large or empty")
        return (self.x_min_m + (np.arange(nx) + .5) * self.resolution_m,
                self.y_min_m + (np.arange(ny) + .5) * self.resolution_m)


def elevation_map(points_m: np.ndarray, config: TerrainConfig = TerrainConfig(),
                  lidar_dimension: int = 3) -> dict:
    """Rig-local z-up endpoint median, local plane slope/residual roughness.

    This is a geometric perception baseline. Obstacles/walls can contaminate
    cell medians; ground segmentation and traversability are separate studies.
    No interpolation fills unknown cells, and hazard-free requires supported
    height, plane slope and roughness. A 2D scan is rejected for this route.
    """
    if lidar_dimension != 3:
        raise ValueError("3D lidar is required for elevation research")
    points_m = np.asarray(points_m, dtype=np.float64)
    if points_m.ndim != 2 or points_m.shape[1] != 3:
        raise ValueError("terrain points must have shape (N,3)")
    x, y = config.axes()
    shape = (len(y), len(x))
    height = np.full(shape, np.nan)
    count = np.zeros(shape, dtype=int)
    points_m = points_m[np.isfinite(points_m).all(axis=1)]
    inside = (points_m[:, 0] >= config.x_min_m) & (points_m[:, 0] < config.x_max_m)
    inside &= (points_m[:, 1] >= config.y_min_m) & (points_m[:, 1] < config.y_max_m)
    points_m = points_m[inside]
    ix = np.floor((points_m[:, 0] - config.x_min_m) / config.resolution_m).astype(int)
    iy = np.floor((points_m[:, 1] - config.y_min_m) / config.resolution_m).astype(int)
    # Sparse scans produce at most a few thousand occupied cells; no dense
    # N-points x N-cells matrix is materialized.
    groups: dict[tuple[int, int], list[float]] = {}
    for xx, yy, zz in zip(ix, iy, points_m[:, 2]):
        groups.setdefault((int(yy), int(xx)), []).append(float(zz))
    for (yy, xx), heights in groups.items():
        count[yy, xx] = len(heights)
        if len(heights) >= config.min_points:
            height[yy, xx] = float(np.median(heights))
    slope, roughness, step = (np.full(shape, np.nan) for _ in range(3))
    for yy, xx in zip(*np.where(np.isfinite(height))):
        ys = slice(max(0, yy - 1), min(len(y), yy + 2))
        xs = slice(max(0, xx - 1), min(len(x), xx + 2))
        patch = height[ys, xs]
        gx, gy = np.meshgrid(x[xs], y[ys])
        valid = np.isfinite(patch)
        if valid.sum() < 3:
            continue
        design = np.column_stack((gx[valid] - x[xx], gy[valid] - y[yy], np.ones(valid.sum())))
        coefficients, _, rank, _ = np.linalg.lstsq(design, patch[valid], rcond=None)
        if rank < 3:
            continue
        slope[yy, xx] = math.degrees(math.atan(math.hypot(*coefficients[:2])))
        roughness[yy, xx] = float(np.sqrt(np.square(patch[valid] - design @ coefficients).mean()))
        step[yy, xx] = float(np.max(np.abs(patch[valid] - height[yy, xx])))
    known = np.isfinite(height) & np.isfinite(slope) & np.isfinite(roughness) & np.isfinite(step)
    hazard = known & ((slope >= config.slope_hazard_deg) |
                      (roughness >= config.roughness_hazard_m) | (step >= config.step_hazard_m))
    return {"height_m": height, "slope_deg": slope, "roughness_m": roughness,
            "neighbor_step_m": step, "point_count": count, "known": known,
            "hazard": hazard, "x_m": x, "y_m": y,
            "frame": "M_local_z_up", "traversal_outcomes": None}


def terrain_metrics(terrain: dict, truth_height_m: np.ndarray,
                    truth_hazard: np.ndarray | None = None) -> dict:
    truth = np.asarray(truth_height_m, dtype=float)
    height = np.asarray(terrain["height_m"], dtype=float)
    if truth.shape != height.shape:
        raise ValueError("terrain truth must match declared map coordinates")
    valid_truth = np.isfinite(truth)
    common = valid_truth & np.isfinite(height)
    error = height[common] - truth[common]
    result = {"gt_height_cells": int(valid_truth.sum()), "observed_height_cells": int(common.sum()),
              "height_coverage": float(common.sum() / valid_truth.sum()) if valid_truth.any() else None,
              "height_mae_m": float(np.abs(error).mean()) if common.any() else None,
              "height_rmse_m": float(np.sqrt(np.square(error).mean())) if common.any() else None,
              "hazard_known_cells": int(terrain["known"].sum()),
              "hazard_unknown_cells": int((~terrain["known"]).sum()),
              "traversal_outcomes": None}
    if truth_hazard is not None:
        truth_hazard = np.asarray(truth_hazard)
        if truth_hazard.shape != height.shape or truth_hazard.dtype != np.bool_:
            raise ValueError("terrain hazard truth must be an independent boolean grid")
        positives = truth_hazard & valid_truth
        detected = positives & terrain["known"] & terrain["hazard"]
        false_positive = valid_truth & ~truth_hazard & terrain["known"] & terrain["hazard"]
        result.update({"hazard_gt_cells": int(positives.sum()),
                       "hazard_detected_cells": int(detected.sum()),
                       "hazard_missed_cells": int((positives & ~detected).sum()),
                       "hazard_false_positive_cells": int(false_positive.sum()),
                       "hazard_recall": float(detected.sum() / positives.sum()) if positives.any() else None})
    return result
