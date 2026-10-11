"""Estimated-pose path control and uncertainty-aware ground elevation mapping.

The interfaces accept sensor measurements and estimated poses only. Evaluation
labels enter ``ground_map_metrics`` after inference. This is a small independent
implementation inspired by regulated pure pursuit and probabilistic elevation
mapping; it is not a port of Nav2 or ANYbotics' full system.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np

ARMS = ("baseline", "heading", "regulated")
TERRAINS = ("flat", "small_steps", "ramp")
DEVELOPMENT_SEEDS = tuple(range(410000, 410010))
CONFIRMATION_SEEDS = tuple(range(420000, 420020))


@dataclass(frozen=True)
class ControlSettings:
    desired_speed_m_s: float = .30
    minimum_speed_m_s: float = .08
    maximum_yaw_rate_rad_s: float = .50
    lookahead_m: float = .60
    curvature_speed_gain: float = .60
    lateral_clearance_slow_m: float = .50
    obstacle_stop_m: float = .30
    footprint_half_width_m: float = .18
    goal_tolerance_m: float = .20
    maximum_pose_age_s: float = .35

    def validate(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("control settings must be finite and positive")
        if self.minimum_speed_m_s > self.desired_speed_m_s:
            raise ValueError("minimum speed exceeds desired speed")


def path_command(native, registration, t_body_imu, raw_points_b, now_s, arm,
                 settings=ControlSettings(), goal_xy=(5., 0.)):
    """Straight prescribed-path pure pursuit, with shared estimated goal stop.

    The baseline has no heading correction. All arms share stationary startup,
    native-loss/staleness stops and estimated-pose goal stop. Raw-point stopping
    and speed regulation apply only to the regulated arm.
    """
    from bhl_robust.research.pose_metrics import transform
    settings.validate()
    if arm not in ARMS:
        raise ValueError("unknown matched controller arm")
    diagnostic = {"reason": "native_untracked_stop", "arm": arm}
    zero = np.zeros(3)
    if native is None or native.get("tracked") is not True:
        return zero, diagnostic, None
    if registration is None or native.get("map_reset_id") != 0:
        diagnostic["reason"] = "native_unregistered_or_reset_stop"
        return zero, diagnostic, None
    age = now_s - float(native["timestamp_s"])
    if not math.isfinite(age) or age < -1e-8 or age > settings.maximum_pose_age_s:
        diagnostic["reason"] = "native_stale_stop"
        return zero, diagnostic, None
    pose = transform(registration) @ transform(native["T_W_I"]) @ np.linalg.inv(transform(t_body_imu))
    x, y = pose[:2, 3]
    goal = np.asarray(goal_xy, dtype=float)
    if goal.shape != (2,) or not np.isfinite(goal).all() or goal[0] <= 0 or goal[1] != 0:
        raise ValueError("this declared experiment requires a positive straight x-axis route")
    diagnostic.update(estimated_cross_track_m=float(y), pose_age_s=age)
    if np.linalg.norm(goal - pose[:2, 3]) <= settings.goal_tolerance_m or (x >= goal[0] and abs(y) <= settings.goal_tolerance_m):
        diagnostic["reason"] = "estimated_goal_stop"
        return zero, diagnostic, pose
    if arm == "baseline":
        diagnostic["reason"] = "matched_blind_heading_baseline"
        return np.array([settings.desired_speed_m_s, 0., 0.]), diagnostic, pose
    yaw = math.atan2(pose[1, 0], pose[0, 0])
    target = np.array([min(goal[0], max(0., x) + settings.lookahead_m), 0.])
    delta = target - pose[:2, 3]
    local_y = -math.sin(yaw)*delta[0] + math.cos(yaw)*delta[1]
    curvature = 2*local_y/max(float(delta@delta), .01)
    speed = settings.desired_speed_m_s
    diagnostic.update(curvature_m_inv=float(curvature), target_xy_m=target.tolist())
    if arm == "regulated":
        points = np.asarray(raw_points_b, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 20 or not np.isfinite(points).all():
            diagnostic["reason"] = "unknown_obstacle_support_stop"
            return zero, diagnostic, pose
        obstacles = points[(points[:, 2] > .10) & (points[:, 2] < .80) & (points[:, 0] > 0) & (points[:, 0] < .80)]
        frontal = obstacles[np.abs(obstacles[:, 1]) < settings.footprint_half_width_m]
        if len(frontal) and float(frontal[:, 0].min()) < settings.obstacle_stop_m:
            diagnostic["reason"] = "observed_obstacle_stop"
            return zero, diagnostic, pose
        clearance = float(np.abs(obstacles[:, 1]).min()) if len(obstacles) else None
        curvature_scale = 1/(1+settings.curvature_speed_gain*abs(curvature))
        clearance_scale = 1. if clearance is None else float(np.clip(
            (clearance-settings.footprint_half_width_m)/
            (settings.lateral_clearance_slow_m-settings.footprint_half_width_m), 0., 1.))
        speed = max(settings.minimum_speed_m_s, speed*min(curvature_scale, clearance_scale))
        diagnostic.update(observed_lateral_clearance_m=clearance, curvature_scale=curvature_scale,
                          clearance_scale=clearance_scale)
    yaw_rate = float(np.clip(speed*curvature, -settings.maximum_yaw_rate_rad_s, settings.maximum_yaw_rate_rad_s))
    diagnostic.update(reason="regulated_pursuit" if arm == "regulated" else "heading_pursuit", speed_m_s=speed)
    return np.array([speed, 0., yaw_rate]), diagnostic, pose


@dataclass(frozen=True)
class MapSettings:
    x_min_m: float = -.5
    x_max_m: float = 5.8
    y_min_m: float = -1.1
    y_max_m: float = 1.1
    resolution_m: float = .10
    maximum_age_s: float = 2.
    process_variance_m2_s: float = 2.5e-5
    range_sigma_m: float = .005
    translation_sigma_m: float = .005
    attitude_sigma_rad: float = math.pi/360
    scan_motion_sigma_m: float = .010
    vertical_span_rejection_m: float = .08
    max_height_above_low_envelope_m: float = .18
    minimum_scan_points: int = 2
    hazard_slope_deg: float = 8.
    hazard_step_m: float = .025

    def axes(self):
        return (np.arange(self.x_min_m+self.resolution_m/2, self.x_max_m, self.resolution_m),
                np.arange(self.y_min_m+self.resolution_m/2, self.y_max_m, self.resolution_m))

    def validate(self):
        values = asdict(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("finite map settings required")
        if self.x_min_m >= self.x_max_m or self.y_min_m >= self.y_max_m:
            raise ValueError("empty map bounds")
        if any(v <= 0 for k, v in values.items() if k not in ("x_min_m", "x_max_m", "y_min_m", "y_max_m")):
            raise ValueError("positive map resolution, uncertainty and gates required")


class UncertainElevationMap:
    """Per-cell scalar Bayesian elevation fusion with explicit unobserved cells.

    Pose uncertainty is a predeclared conservative floor, not a covariance
    exported by FAST-LIO2. A scan's returns are correlated: one median per cell
    is fused and shared pose uncertainty never averages away. Vertical columns
    and points above the scan's low envelope are excluded from ground inference.
    """
    def __init__(self, settings=MapSettings()):
        settings.validate()
        self.settings = settings
        self.x, self.y = settings.axes()
        shape = (len(self.y), len(self.x))
        self.height = np.full(shape, np.nan)
        self.variance = np.full(shape, np.inf)
        self.last_seen = np.full(shape, -np.inf)
        self.scans = np.zeros(shape, np.int32)
        self.rejected_vertical = np.zeros(shape, np.int32)
        self.blocked = np.zeros(shape, bool)
        self.last_update = None

    def update(self, points_l, t_world_lidar, timestamp_s, *, pose_variance_m2=None, point_variance_m2=None):
        from bhl_robust.research.pose_metrics import transform
        points = np.asarray(points_l, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("finite raw lidar (N,3) points required")
        if not math.isfinite(timestamp_s) or (self.last_update is not None and timestamp_s <= self.last_update):
            raise ValueError("map updates require strictly increasing acquisition timestamps")
        s = self.settings
        floor = s.translation_sigma_m**2 if pose_variance_m2 is None else float(pose_variance_m2)
        if not math.isfinite(floor) or floor < 0:
            raise ValueError("finite nonnegative pose variance required")
        t = transform(t_world_lidar)
        if point_variance_m2 is None:
            point_variance = np.full(len(points), s.range_sigma_m**2)
        else:
            point_variance = np.asarray(point_variance_m2, dtype=float)
            if point_variance.ndim == 0:
                point_variance = np.full(len(points), float(point_variance))
            elif point_variance.shape == (len(points), 3):
                # Independent local x/y/z variances projected into world z.
                point_variance = point_variance@(t[2, :3]**2)
            if point_variance.shape != (len(points),) or not np.isfinite(point_variance).all() or (point_variance < 0).any():
                raise ValueError("point variances must be finite nonnegative scalar, N-vector or Nx3 diagonal local covariance")
        world = points@t[:3, :3].T+t[:3, 3]
        self.last_update = timestamp_s
        if not len(world):
            return {"input_points": 0, "fused_cells": 0, "rejected_wall_cells": 0}
        columns = np.floor((world[:, 0]-s.x_min_m)/s.resolution_m).astype(int)
        rows = np.floor((world[:, 1]-s.y_min_m)/s.resolution_m).astype(int)
        inside = (columns >= 0) & (columns < len(self.x)) & (rows >= 0) & (rows < len(self.y))
        low = float(np.quantile(world[:, 2], .10))
        ids = rows*len(self.x)+columns
        fused = rejected = 0
        for key in np.unique(ids[inside]):
            selected = inside & (ids == key)
            z = world[selected, 2]
            row, col = divmod(int(key), len(self.x))
            if (np.ptp(z) > s.vertical_span_rejection_m or
                    float(np.median(z)) > low+s.max_height_above_low_envelope_m):
                self.rejected_vertical[row, col] += 1
                self.blocked[row, col] = True
                rejected += 1
                continue
            if len(z) < s.minimum_scan_points:
                continue
            mean = float(np.median(z))
            # Jacobian range term + rotation lever arm + shared translation and
            # scan motion. No division of shared uncertainty by point count.
            lever = float(np.median(np.sum(points[selected, :2]**2, axis=1)))
            measurement = max(1e-10, floor+float(np.median(point_variance[selected]))+lever*s.attitude_sigma_rad**2+
                              s.scan_motion_sigma_m**2+float(np.var(z)))
            if self.scans[row, col] and timestamp_s-self.last_seen[row, col] <= s.maximum_age_s:
                prior = self.variance[row, col]+s.process_variance_m2_s*(timestamp_s-self.last_seen[row, col])
                innovation = mean-self.height[row, col]
                # Preserve discontinuities as new measurements when old support
                # disagrees by over 3 sigma; do not blur a step into a ramp.
                if abs(innovation) <= 3*math.sqrt(prior+measurement):
                    gain = prior/(prior+measurement)
                    mean = self.height[row, col]+gain*innovation
                    measurement = max(floor, (1-gain)*prior)
            self.blocked[row, col] = False
            self.height[row, col] = mean
            self.variance[row, col] = measurement
            self.last_seen[row, col] = timestamp_s
            self.scans[row, col] += 1
            fused += 1
        return {"input_points": len(points), "fused_cells": fused, "rejected_wall_cells": rejected}

    def snapshot(self, timestamp_s):
        if not math.isfinite(timestamp_s) or (self.last_update is not None and timestamp_s < self.last_update):
            raise ValueError("snapshot timestamp precedes acquired data")
        s = self.settings
        age = timestamp_s-self.last_seen
        observed = self.scans > 0
        known = observed & (age <= s.maximum_age_s) & ~self.blocked
        variance = self.variance+s.process_variance_m2_s*np.where(observed, age, 0.)
        slope = np.full(self.height.shape, np.nan)
        step = np.full(self.height.shape, np.nan)
        for row, col in zip(*np.where(known)):
            differences = []
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                yy, xx = row+dy, col+dx
                if 0 <= yy < len(self.y) and 0 <= xx < len(self.x) and known[yy, xx]:
                    differences.append(abs(self.height[yy, xx]-self.height[row, col]))
            if differences:
                step[row, col] = max(differences)
                slope[row, col] = math.degrees(math.atan(step[row, col]/s.resolution_m))
        hazard_known = known & np.isfinite(step)
        hazard = np.ones(self.height.shape, dtype=bool)
        hazard[hazard_known] = ((slope[hazard_known] >= s.hazard_slope_deg) |
                                (step[hazard_known] >= s.hazard_step_m))
        return {"x_m": self.x.copy(), "y_m": self.y.copy(), "height_m": self.height.copy(),
                "variance_m2": variance, "age_s": age, "known": known, "scan_count": self.scans.copy(),
                "rejected_vertical_count": self.rejected_vertical.copy(), "blocked": self.blocked.copy(), "slope_deg": slope,
                "neighbor_step_m": step, "hazard": hazard, "hazard_known": hazard_known,
                "pose_uncertainty_source": "declared conservative floor, not estimator covariance"}


def ground_map_metrics(snapshot, truth_height_m, ground_mask, *, truth_hazard=None):
    """Only independent evaluator ground cells enter any accuracy denominator."""
    truth = np.asarray(truth_height_m, dtype=float)
    ground = np.asarray(ground_mask, dtype=bool)
    if truth.shape != snapshot["known"].shape or ground.shape != truth.shape:
        raise ValueError("truth, ground mask and inferred grid shape mismatch")
    eligible = ground & np.isfinite(truth)
    scored = eligible & snapshot["known"] & np.isfinite(snapshot["height_m"])
    error = snapshot["height_m"][scored]-truth[scored]
    sigma = np.sqrt(snapshot["variance_m2"][scored])
    result = {"ground_cells": int(eligible.sum()), "covered_ground_cells": int(scored.sum()),
              "ground_coverage": float(scored.sum()/eligible.sum()) if eligible.any() else None,
              "ground_height_mae_m": float(np.abs(error).mean()) if len(error) else None,
              "ground_height_rmse_m": float(np.sqrt(np.mean(error**2))) if len(error) else None,
              "ground_height_p95_absolute_m": float(np.quantile(np.abs(error), .95)) if len(error) else None,
              "declared_95_percent_interval_coverage": float(np.mean(np.abs(error) <= 1.96*sigma)) if len(error) else None,
              "non_ground_cells_excluded": int((~ground).sum()),
              "truth_source": "independent downward rays restricted to named ground geoms; walls excluded",
              "uncertainty_scope": "heuristic sensor/pose uncertainty; empirical coverage is measured, not assumed calibrated"}
    if truth_hazard is not None:
        hazard = np.asarray(truth_hazard, dtype=bool)
        if hazard.shape != truth.shape:
            raise ValueError("hazard truth grid shape mismatch")
        comparable = scored & snapshot["hazard_known"]
        positives = comparable & hazard
        negatives = comparable & ~hazard
        missed = positives & ~snapshot["hazard"]
        result.update(hazard_evaluable_ground_cells=int(comparable.sum()),
                      ground_hazard_recall=float(np.mean(snapshot["hazard"][positives])) if positives.any() else None,
                      false_traversable_ground_cells=int(missed.sum()),
                      ground_false_positive_rate=float(np.mean(snapshot["hazard"][negatives])) if negatives.any() else None,
                      unknown_ground_hazard_cells=int((eligible & ~snapshot["hazard_known"]).sum()))
    return result
