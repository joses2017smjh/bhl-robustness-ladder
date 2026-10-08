"""Timestamped sensor inputs independent of Isaac, MuJoCo, ROS and PyTorch.

The IMU boundary uses REP-145 units and ROS xyzw quaternions. SSD inputs use
torchvision's boxes/labels/scores convention. Depth must be registered to the
left RGB image; ray-cast ground truth must be identified as such by the caller.
No hardware calibration or detection accuracy is implied by these adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SensorTiming:
    max_age_s: float = 0.15
    stereo_skew_s: float = 0.035

    def __post_init__(self):
        if not all(math.isfinite(x) and x >= 0 for x in (self.max_age_s, self.stereo_skew_s)):
            raise ValueError("Timing limits must be finite and nonnegative")

    def fresh(self, stamp_s: float, now_s: float) -> bool:
        return bool(math.isfinite(stamp_s) and math.isfinite(now_s)
                    and 0 <= now_s - stamp_s <= self.max_age_s)


def imu_features(orientation_xyzw, angular_velocity_rad_s, specific_force_m_s2,
                 *, stamp_s: float, now_s: float, timing=SensorTiming(),
                 orientation_available: bool = True) -> np.ndarray:
    """Return [body gravity unit vector, gyro, accelerometer, valid] (10 floats).

    Stale/unavailable orientation returns an explicitly invalid zero vector.
    The quaternion maps body to an ENU world with gravity along negative z.
    Convert vendor NED/axis conventions upstream; do not pass degrees/s or g.
    ROS orientation_covariance[0] == -1 means orientation_available=False.
    """
    q = np.asarray(orientation_xyzw, dtype=float)
    gyro = np.asarray(angular_velocity_rad_s, dtype=float)
    accel = np.asarray(specific_force_m_s2, dtype=float)
    if q.shape != (4,) or gyro.shape != (3,) or accel.shape != (3,):
        raise ValueError("Expected xyzw quaternion and two 3-vectors")
    out = np.zeros(10, dtype=np.float32)
    if not orientation_available or not timing.fresh(stamp_s, now_s):
        return out
    if not all(np.isfinite(x).all() for x in (q, gyro, accel)):
        return out
    scale = np.max(np.abs(q))
    if scale < 1e-8 or any(np.any(np.abs(v) > np.finfo(np.float32).max) for v in (gyro, accel)):
        return out
    q = q / scale
    x, y, z, w = q / np.linalg.norm(q)
    # R(world <- body)^T @ [0, 0, -1].
    gravity = np.array([2 * (w*y - x*z), -2 * (y*z + w*x),
                        2 * (x*x + y*y) - 1])
    out[:] = np.r_[gravity, gyro, accel, 1.0]
    return out


def ssd_features(prediction, *, image_width: int, image_height: int,
                 stamp_s: float, now_s: float, depth_m=None,
                 depth_stamp_s: float | None = None, slots: int = 8,
                 score_threshold: float = 0.5, max_range_m: float = 12.0,
                 timing=SensorTiming()) -> np.ndarray:
    """Pack SSD predictions into fixed slots with explicit masks.

    Columns: class_id, score, normalized cx/cy/w/h, depth_z_m, depth_valid,
    detection_valid. Zero slots are padding. Missing depth never means zero
    distance. Depth is the median valid optical-axis depth in the box's central
    half, only when capture timestamps agree. This is not Euclidean range;
    converting to a slant range requires camera intrinsics. Box depth is an estimate, not
    a guarantee that pixels belong to the detected object.
    """
    if image_width <= 0 or image_height <= 0 or slots <= 0:
        raise ValueError("Image dimensions and slot count must be positive")
    if not 0 <= score_threshold <= 1 or not math.isfinite(max_range_m) or max_range_m <= 0:
        raise ValueError("Invalid score threshold or maximum range")
    out = np.zeros((slots, 9), dtype=np.float32)
    if not timing.fresh(stamp_s, now_s):
        return out
    boxes = np.asarray(prediction["boxes"], dtype=float).reshape(-1, 4)
    scores = np.asarray(prediction["scores"], dtype=float).reshape(-1)
    labels = np.asarray(prediction["labels"], dtype=float).reshape(-1)
    if len(boxes) != len(scores) or len(scores) != len(labels):
        raise ValueError("SSD boxes, scores, labels must have equal lengths")
    depth = None
    if depth_m is not None:
        candidate = np.asarray(depth_m, dtype=float)
        if candidate.shape != (image_height, image_width):
            raise ValueError("Depth must be registered to the RGB image dimensions")
        if (depth_stamp_s is not None and timing.fresh(depth_stamp_s, now_s)
                and abs(depth_stamp_s-stamp_s) <= timing.stereo_skew_s):
            depth = candidate
    slot = 0
    for idx in np.argsort(-scores, kind="stable"):
        box, score, label = boxes[idx], scores[idx], labels[idx]
        if (not np.isfinite(box).all() or not np.isfinite([score, label]).all()
                or not score_threshold <= score <= 1 or not 0 < label <= 2**31-1 or label != int(label)):
            continue
        x0, y0, x1, y1 = box
        x0, x1 = np.clip([x0, x1], 0, image_width)
        y0, y1 = np.clip([y0, y1], 0, image_height)
        if x1 <= x0 or y1 <= y0:
            continue
        cx, cy = (x0+x1)/2, (y0+y1)/2
        width, height = x1-x0, y1-y0
        out[slot] = [label, score, cx/image_width, cy/image_height,
                     width/image_width, height/image_height, 0, 0, 1]
        if depth is not None:
            xa, xb = int(cx-width/4), int(math.ceil(cx+width/4))
            ya, yb = int(cy-height/4), int(math.ceil(cy+height/4))
            values = depth[ya:yb, xa:xb]
            values = values[np.isfinite(values) & (values > 0) & (values <= max_range_m)]
            if values.size:
                out[slot, 6:8] = [np.median(values), 1]
        slot += 1
        if slot == slots:
            break
    return out
