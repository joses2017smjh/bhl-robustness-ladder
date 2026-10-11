"""Predeclared causal acquisition/fault controls for native navigation.

Image transformations accept acquired pixels, and IMU buffering accepts only
measurements that have become available. Evaluator state is not an input.
"""
from __future__ import annotations

from collections import deque
import copy
import math

import numpy as np

ARMS = ("stereo", "stereo_inertial", "lidar_inertial", "permanent_stop", "verified_lio_recovery")
CONDITIONS = {
    "nominal": {"camera_hz": 5.},
    "camera_2_5hz": {"camera_hz": 2.5},
    "camera_10hz": {"camera_hz": 10.},
    "dark_exposure_gain": {"camera_hz": 5., "exposure_gain": .25},
    "causal_40ms_exposure": {"camera_hz": 5., "exposure_s": .04, "exposure_samples": 3},
    "visual_blackout": {"camera_hz": 5., "blackout_start_s": 12., "blackout_end_s": 16.},
    "lidar_quarter_returns": {"camera_hz": 5., "sensor_fault": "lidar_keep_quarter"},
    "lidar_extrinsic_error": {"camera_hz": 5., "sensor_fault": "lidar_extrinsic_yaw2deg_x2cm"},
    "imu_clock_plus80ms": {"camera_hz": 5., "sensor_fault": "imu_clock_plus80ms"},
    "gyro_bias": {"camera_hz": 5., "sensor_fault": "gyro_z_bias_after5s"},
}


def condition_spec(name):
    if name not in CONDITIONS:
        raise ValueError("unknown predeclared navigation condition")
    return copy.deepcopy(CONDITIONS[name])


class NativeCoordinateEpochTracker:
    """Conservative causal registration epochs, including reused map IDs.

    A retained map in RECENTLY_LOST/LOST does not itself mean coordinates reset.
    The pinned shim observes a map through usable tracked map points, not the
    Atlas directly. Missing identity/initialization can therefore mean missing
    evidence rather than an actual native reset. We conservatively invalidate
    continuity and require a new epoch for such uncertainty, an ID change, or
    return to NOT_INITIALIZED. These events are not counted as confirmed resets.
    No poses, truth or future frames enter this detector. Observe each unique
    native camera response exactly once.
    """
    def __init__(self):
        self.previous_stamp = -math.inf
        self.previous_map = None
        self.previous_initialized = None
        self.previous_state = None
        self.ever_had_map = False
        self.pending_epoch = True
        self.epoch = -1
        self.events = []

    def observe(self, frame):
        stamp, native_map, state = frame["timestamp_s"], frame["map_id"], frame["tracking_state"]
        initialized = frame.get("inertial", {}).get("map_imu_initialized")
        if (isinstance(stamp, bool) or not isinstance(stamp, (float, int))
                or not math.isfinite(stamp) or stamp <= self.previous_stamp
                or (native_map is not None and (type(native_map) is not int or native_map < 0))
                or type(state) is not int or state not in (-1, 0, 1, 2, 3, 4, 5)
                or (initialized is not None and type(initialized) is not bool)):
            raise ValueError("unique ordered native response and explicit native diagnostics required")
        reasons = []
        if self.previous_map is not None and native_map is None:
            reasons.append("native_map_identity_unavailable")
        if self.previous_map is not None and native_map is not None and native_map != self.previous_map:
            reasons.append("native_map_id_changed")
        if self.previous_initialized is True and initialized is False:
            reasons.append("native_imu_initialization_unconfirmed")
        if self.ever_had_map and not self.pending_epoch and state == 1 and self.previous_state != 1:
            reasons.append("native_returned_to_not_initialized")
        if reasons:
            self.events.append({"timestamp_s": stamp, "previous_native_map_id": self.previous_map,
                                "native_map_id": native_map, "previous_epoch": self.epoch,
                                "next_epoch": self.epoch + 1, "reasons": reasons})
            self.pending_epoch = True
        if native_map is not None:
            if self.pending_epoch:
                self.epoch += 1
                self.pending_epoch = False
            self.ever_had_map = True
        elif self.ever_had_map:
            self.pending_epoch = True
        self.previous_stamp, self.previous_map = stamp, native_map
        self.previous_initialized, self.previous_state = initialized, state
        return {"epoch_id": self.epoch if native_map is not None else None,
                "native_map_id": native_map, "continuity_invalidated": bool(reasons),
                "invalidation_reasons": reasons, "coordinate_invalidations": len(self.events)}

    def receipt(self):
        return {"schema": "bhl-native-coordinate-epochs-v1", "epochs_observed": self.epoch + 1,
                "coordinate_invalidations": len(self.events), "events": self.events,
                "uses_native_diagnostics_only": True,
                "confirmed_native_reset_count": None,
                "observation_scope": "pinned wrapper observes map identity/IMU initialization through usable tracked map points; missing evidence conservatively invalidates registration, not proof of a native reset",
                "transient_tracking_loss_with_retained_map_is_not_reset": True}


def exposure_image(samples, timestamps, *, acquisition_s, exposure_s=0., gain=1., blackout=False):
    """Average actual past renders; never synthesize a future exposure sample."""
    values = [np.asarray(x) for x in samples]
    times = np.asarray(timestamps, dtype=float)
    if (not values or len(values) != len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or times[-1] > acquisition_s + 1e-9
            or times[0] < acquisition_s - exposure_s - 1e-9
            or not math.isfinite(gain) or gain <= 0
            or any(v.dtype != np.uint8 or v.ndim != 3 or v.shape[-1] != 3 or v.shape != values[0].shape for v in values)):
        raise ValueError("finite causal RGB uint8 exposure samples required")
    if blackout:
        return np.zeros_like(values[0])
    return np.clip(np.mean(np.stack(values), axis=0) * gain, 0, 255).astype(np.uint8)


class AvailableImu:
    """Faulted timestamps are also availability times; future samples stay queued."""
    def __init__(self, *, clock_offset_s=0., gyro_bias=(0., 0., 0.), bias_start_s=math.inf):
        if not math.isfinite(clock_offset_s) or clock_offset_s < 0:
            raise ValueError("nonnegative causal clock offset required")
        self.offset = clock_offset_s
        self.bias = np.asarray(gyro_bias, dtype=float)
        if self.bias.shape != (3,) or not np.isfinite(self.bias).all():
            raise ValueError("finite three-axis gyro bias required")
        self.bias_start = bias_start_s
        self.pending = deque()
        self.last_original = -math.inf
        self.delivered = []

    def append(self, timestamp, gyro, acceleration):
        if not math.isfinite(timestamp) or timestamp <= self.last_original:
            raise ValueError("strictly increasing original IMU required")
        g, a = np.asarray(gyro, dtype=float), np.asarray(acceleration, dtype=float)
        if g.shape != (3,) or a.shape != (3,) or not np.isfinite(g).all() or not np.isfinite(a).all():
            raise ValueError("finite original SI-unit IMU required")
        self.last_original = timestamp
        if timestamp >= self.bias_start:
            g = g + self.bias
        self.pending.append((timestamp + self.offset, g.copy(), a.copy(), timestamp))

    def consume(self, until_s):
        rows = []
        while self.pending and self.pending[0][0] <= until_s + 1e-9:
            rows.append(self.pending.popleft())
        self.delivered.extend(rows)
        return {"timestamp_s": np.asarray([r[0] for r in rows]),
                "gyro_rad_s": np.asarray([r[1] for r in rows]).reshape(-1, 3),
                "specific_force_m_s2": np.asarray([r[2] for r in rows]).reshape(-1, 3),
                "original_timestamp_s": np.asarray([r[3] for r in rows])}


def blocked_returns(points_body):
    p = np.asarray(points_body, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("finite body-frame cloud required")
    return ((p[:, 0] > .10) & (p[:, 0] < .45) & (np.abs(p[:, 1]) < .18)
            & (p[:, 2] > .10) & (p[:, 2] < .70))


def measured_brake_comparison(original_body, faulted_body):
    """Evaluator-only comparison on the exact surfaces hit by simulated rays.

    This does not claim complete occupancy, unseen geometry, or pixel depth.
    Fewer than 20 faulted returns cause an unknown stop, never safe free space.
    """
    truth = bool(blocked_returns(original_body).any())
    observed = bool(blocked_returns(faulted_body).any())
    known = len(faulted_body) >= 20
    return {"measured_obstacle_present": truth, "faulted_obstacle_present": observed,
            "unknown_stop": not known, "false_clear": bool(truth and known and not observed),
            "scope": "same-time raw-brake surface-return box only; not complete free-space occupancy"}


def tracking_events(records, horizon_s):
    """Observed acquisition-time loss/recovery durations with right censoring."""
    seen = False
    lost = None
    losses, recoveries, durations = 0, 0, []
    previous_map = None
    map_changes = 0
    for r in records:
        stamp = float(r["timestamp_s"])
        native_map = r.get("native_ORB_map_id")
        if native_map is not None:
            if previous_map is not None and native_map != previous_map:
                map_changes += 1
            previous_map = native_map
        if r["tracked"]:
            if lost is not None:
                durations.append(stamp - lost)
                recoveries += 1
                lost = None
            seen = True
        elif seen and lost is None:
            lost = stamp
            losses += 1
    return {"tracking_loss_events": losses, "tracking_recoveries": recoveries,
            "completed_recovery_durations_s": durations,
            "recovery_duration_mean_s": float(np.mean(durations)) if durations else None,
            "right_censored_loss_duration_s": horizon_s - lost if lost is not None else None,
            "native_map_changes": map_changes,
            "native_coordinate_invalidations": max((r.get("native_coordinate_invalidations", 0) for r in records), default=0),
            "native_coordinate_epochs": max((r.get("native_coordinate_epoch_id") for r in records
                                                if r.get("native_coordinate_epoch_id") is not None), default=-1) + 1,
            "recovery_time_scope": "accepted-output tracking at acquisition timestamps; delivery delay reported separately"}
