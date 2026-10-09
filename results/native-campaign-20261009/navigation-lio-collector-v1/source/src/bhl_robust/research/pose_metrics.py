"""Metric-scale pose scoring and an estimated-pose navigation boundary.

This module does not estimate poses. Ground truth is accepted only by the
offline scorer; the navigation adapter accepts estimated packets and sensors.
Transforms T_A_B map coordinates in B into A. Time is in seconds.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


METHODS = ("orb_slam3_stereo", "fast_lio2_lidar_imu")


def transform(value) -> np.ndarray:
    t = np.asarray(value, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("pose must be a finite 4x4 transform")
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("invalid homogeneous transform row")
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not math.isclose(float(np.linalg.det(r)), 1.0, abs_tol=1e-5):
        raise ValueError("pose rotation must be in SO(3), without scale")
    return t.copy()


def _timestamps(values) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or np.any(np.diff(a) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    return a


@dataclass(frozen=True)
class Trajectory:
    timestamp_s: np.ndarray
    T_W_S: np.ndarray
    tracked: np.ndarray | None = None

    def __post_init__(self):
        ts = _timestamps(self.timestamp_s)
        poses = np.asarray(self.T_W_S, dtype=np.float64)
        if poses.shape != (len(ts), 4, 4):
            raise ValueError("one pose is required for each timestamp")
        poses = np.stack([transform(p) for p in poses])
        states = np.ones(len(ts), dtype=bool) if self.tracked is None else np.asarray(self.tracked)
        if states.dtype != np.bool_ or states.shape != ts.shape:
            raise ValueError("tracked must contain one explicit boolean per frame")
        object.__setattr__(self, "timestamp_s", ts.copy())
        object.__setattr__(self, "T_W_S", poses)
        object.__setattr__(self, "tracked", states.copy())


def associate_timestamps(estimated, truth, *, max_difference_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Nearest pairs, consuming each truth sample once and preserving time order."""
    e, g = _timestamps(estimated), _timestamps(truth)
    if not math.isfinite(max_difference_s) or max_difference_s < 0:
        raise ValueError("association tolerance must be finite and nonnegative")
    ei, gi, next_g = [], [], 0
    for i, stamp in enumerate(e):
        insertion = int(np.searchsorted(g, stamp))
        candidates = [j for j in (max(next_g, insertion - 1), max(next_g, insertion)) if j < len(g)]
        if not candidates:
            continue
        j = min(candidates, key=lambda k: (abs(float(g[k] - stamp)), k))
        if abs(float(g[j] - stamp)) <= max_difference_s:
            ei.append(i)
            gi.append(j)
            next_g = j + 1
    return np.asarray(ei, dtype=int), np.asarray(gi, dtype=int)


def _yaw(r) -> float:
    return math.atan2(float(r[1, 0]), float(r[0, 0]))


def _yaw_rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def align_metric_poses(estimated, truth, *, alignment: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Rigid alignment only; scale remains exactly 1 for both sensor routes."""
    e, g = np.asarray(estimated), np.asarray(truth)
    if e.shape != g.shape or e.ndim != 3 or e.shape[1:] != (4, 4) or len(e) < 2:
        raise ValueError("at least two matched pose pairs are required")
    e = np.stack([transform(p) for p in e])
    g = np.stack([transform(p) for p in g])
    x, y = e[:, :3, 3], g[:, :3, 3]
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    mode = alignment
    if alignment == "se3":
        if np.linalg.matrix_rank(xc, tol=1e-8) < 2 or np.linalg.matrix_rank(yc, tol=1e-8) < 2:
            r = g[0, :3, :3] @ e[0, :3, :3].T
            mode = "se3_initial_orientation_translation_fit"
        else:
            u, _, vt = np.linalg.svd(xc.T @ yc)
            sign = np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)])
            r = vt.T @ sign @ u.T
    elif alignment == "gravity_yaw_translation":
        dot = float(np.sum(xc[:, :2] * yc[:, :2]))
        cross = float(np.sum(xc[:, 0] * yc[:, 1] - xc[:, 1] * yc[:, 0]))
        if math.hypot(dot, cross) < 1e-12:
            angle = _yaw(g[0, :3, :3]) - _yaw(e[0, :3, :3])
            mode = "gravity_initial_yaw_translation_fit"
        else:
            angle = math.atan2(cross, dot)
        r = _yaw_rotation(angle)
    elif alignment == "none":
        return e.copy(), np.eye(4), mode
    else:
        raise ValueError("alignment must be se3, gravity_yaw_translation or none")
    a = np.eye(4)
    a[:3, :3], a[:3, 3] = r, y.mean(axis=0) - r @ x.mean(axis=0)
    return a[None] @ e, a, mode


def _rotation_error_degrees(r) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def _stats(values):
    v = np.asarray(values, dtype=float)
    return {"rmse": float(np.sqrt(np.mean(v * v))), "mean": float(v.mean()),
            "p95": float(np.quantile(v, .95)), "max": float(v.max())}


def score_trajectories(estimated: Trajectory, truth: Trajectory, *, alignment: str,
                       max_difference_s: float = .02, end_time_s: float | None = None) -> dict:
    """ATE, consecutive-associated-pair RPE, endpoint drift and tracking loss.

    RPE always includes its actual time intervals. Unmatched poses do not get
    interpolated or silently interpreted as tracking failures.
    """
    if not np.all(truth.tracked):
        raise ValueError("truth must explicitly cover all supplied reference poses")
    tracked_indices = np.flatnonzero(estimated.tracked)
    if len(tracked_indices) < 2:
        raise ValueError("not enough tracked poses for metric scoring")
    ei, gi = associate_timestamps(estimated.timestamp_s[tracked_indices], truth.timestamp_s,
                                  max_difference_s=max_difference_s)
    ei = tracked_indices[ei]
    if len(ei) < 2:
        raise ValueError("not enough timestamp-associated tracked poses")
    aligned, a, alignment_mode = align_metric_poses(estimated.T_W_S[ei], truth.T_W_S[gi], alignment=alignment)
    gt = truth.T_W_S[gi]
    ate = np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=1)
    rot = [_rotation_error_degrees(g[:3, :3].T @ e[:3, :3]) for e, g in zip(aligned, gt)]
    trans_rpe, rot_rpe = [], []
    for i in range(len(aligned) - 1):
        dg = np.linalg.inv(gt[i]) @ gt[i + 1]
        de = np.linalg.inv(aligned[i]) @ aligned[i + 1]
        error = np.linalg.inv(dg) @ de
        trans_rpe.append(np.linalg.norm(error[:3, 3]))
        rot_rpe.append(_rotation_error_degrees(error[:3, :3]))
    path = float(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1).sum())
    drift = np.linalg.inv(np.linalg.inv(gt[0]) @ gt[-1]) @ (np.linalg.inv(aligned[0]) @ aligned[-1])
    endpoint = float(np.linalg.norm(drift[:3, 3]))
    stamps = estimated.timestamp_s
    end = float(stamps[-1]) if end_time_s is None else float(end_time_s)
    if not math.isfinite(end) or end < stamps[-1]:
        raise ValueError("end_time_s must be at least the last output timestamp")
    durations = np.diff(np.append(stamps, end))
    lost = ~estimated.tracked
    failures = int(np.sum(lost & np.r_[True, ~lost[:-1]]))
    return {
        "schema": "bhl-pose-metrics-v1", "scale": 1.0, "alignment": alignment_mode,
        "T_truthworld_estimatedworld": a.tolist(), "associated_poses": len(ei),
        "estimate_frames": len(stamps), "tracked_frames": int(estimated.tracked.sum()),
        "unassociated_tracked_frames": int(estimated.tracked.sum()) - len(ei),
        "association_max_error_s": float(np.abs(stamps[ei] - truth.timestamp_s[gi]).max()),
        "ate_translation_m": _stats(ate), "ate_rotation_deg": _stats(rot),
        "rpe_translation_m": _stats(trans_rpe), "rpe_rotation_deg": _stats(rot_rpe),
        "rpe_definition": "consecutive associated tracked pose pairs",
        "rpe_interval_s": np.diff(stamps[ei]).tolist(), "reference_path_length_m": path,
        "endpoint_drift_m": endpoint, "endpoint_drift_m_per_m": endpoint / path if path > 1e-9 else None,
        "tracking_failure_segments": failures, "tracking_lost_frames": int(lost.sum()),
        "tracking_lost_time_s": float(durations[lost].sum()), "tracking_observation_end_s": end,
        "associated_estimate_indices": ei.tolist(), "associated_truth_indices": gi.tolist(),
    }


@dataclass(frozen=True)
class PosePacket:
    T_W_B: np.ndarray
    capture_time_s: float
    arrival_time_s: float
    confidence: float
    tracking: bool
    clock_domain: str
    source_method: str

    def __post_init__(self):
        object.__setattr__(self, "T_W_B", transform(self.T_W_B))
        if not all(math.isfinite(v) for v in (self.capture_time_s, self.arrival_time_s, self.confidence)):
            raise ValueError("pose packet times and confidence must be finite")
        if self.arrival_time_s < self.capture_time_s or not 0 <= self.confidence <= 1:
            raise ValueError("pose arrival must follow capture and confidence must be in [0,1]")
        if type(self.tracking) is not bool or not self.clock_domain or self.source_method not in METHODS:
            raise ValueError("explicit tracking, clock domain and supported estimator method are required")


class EstimatedPoseGate:
    """Fail closed before policy inference when estimated pose is unusable.

    A confidence score is a method-specific score, not a calibrated probability.
    Reusing an unchanged packet is allowed until its capture age expires.
    """
    def __init__(self, *, clock_domain: str, source_method: str, max_age_s=.15, min_confidence=.5):
        if source_method not in METHODS or not clock_domain:
            raise ValueError("supported method and clock domain required")
        if not math.isfinite(max_age_s) or max_age_s < 0 or not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("invalid pose gate limits")
        self.clock_domain, self.source_method = clock_domain, source_method
        self.max_age_s, self.min_confidence = max_age_s, min_confidence
        self.last_capture_s = -math.inf

    def accept(self, packet: PosePacket, now_s: float) -> dict:
        reason = None
        if not math.isfinite(now_s): reason = "invalid_consumption_time"
        elif packet.clock_domain != self.clock_domain: reason = "clock_domain_mismatch"
        elif packet.source_method != self.source_method: reason = "method_mismatch"
        elif packet.arrival_time_s > now_s: reason = "packet_not_arrived"
        elif packet.capture_time_s > now_s: reason = "capture_in_future"
        elif packet.capture_time_s < self.last_capture_s: reason = "out_of_order_capture"
        elif now_s - packet.capture_time_s > self.max_age_s: reason = "stale_capture"
        else:
            # A new lost-tracking packet invalidates older confident packets.
            self.last_capture_s = packet.capture_time_s
            if not packet.tracking: reason = "tracking_lost"
            elif packet.confidence < self.min_confidence: reason = "low_confidence"
        if reason:
            return {"accepted": False, "reason": reason, "safe_command_vx_vy_wz": [0., 0., 0.]}
        self.last_capture_s = packet.capture_time_s
        t = packet.T_W_B
        return {"accepted": True, "reason": "estimated_pose", "x_m": float(t[0, 3]),
                "y_m": float(t[1, 3]), "yaw_rad": _yaw(t[:3, :3]),
                "source_method": packet.source_method, "capture_time_s": packet.capture_time_s,
                "pose_age_s": now_s - packet.capture_time_s}


def estimated_navigation_observation(*, gate: EstimatedPoseGate, packet: PosePacket,
                                     now_s: float, ranges_m, scan_capture_time_s: float,
                                     goal_xy, previous_action, observation_keys, occupancy_map,
                                     visit=None, max_pose_scan_skew_s=.02, builder: Callable | None = None) -> dict:
    """Use estimated pose in the existing frozen NavGym observation builder.

    The caller must anchor its occupancy/visitation maps and known goal in the
    estimator world frame. There is deliberately no simulator-state argument.
    This interface alone does not constitute a closed-loop experiment.
    """
    if not math.isfinite(max_pose_scan_skew_s) or max_pose_scan_skew_s < 0:
        raise ValueError("pose/scan skew limit must be finite and nonnegative")
    goal, prev = np.asarray(goal_xy, dtype=float), np.asarray(previous_action, dtype=float)
    if (goal.shape != (2,) or prev.shape != (2,) or not np.isfinite(goal).all()
            or not np.isfinite(prev).all() or np.any(np.abs(prev) > 1)):
        raise ValueError("finite known goal and normalized previous action are required")
    decision = gate.accept(packet, now_s)
    if not decision["accepted"]:
        return {"gate": decision, "observation": None}
    ranges = np.asarray(ranges_m, dtype=float)
    if (ranges.shape != (108,) or not np.isfinite(ranges).all() or np.any(ranges <= 0)
            or np.any(ranges > 12.0) or not math.isfinite(scan_capture_time_s)
            or abs(scan_capture_time_s - packet.capture_time_s) > max_pose_scan_skew_s):
        return {"gate": {"accepted": False, "reason": "invalid_or_unaligned_scan",
                          "safe_command_vx_vy_wz": [0., 0., 0.]}, "observation": None}
    if builder is None:
        from bhl_robust.navgym.env import build_obs
        builder = build_obs
    obs = builder(observation_keys, ranges, occupancy_map, decision["x_m"], decision["y_m"],
                  decision["yaw_rad"], goal_xy, previous_action, visit=visit)
    return {"gate": decision, "observation": obs}


def calibrated_body_pose(T_W_S, T_B_S):
    """Convert native camera/IMU pose to body using the measured rigid mount.

    This changes the represented sensor frame without registering the world to
    ground truth. Offline evaluation alignment must never anchor policy goals.
    """
    return transform(T_W_S) @ np.linalg.inv(transform(T_B_S))
