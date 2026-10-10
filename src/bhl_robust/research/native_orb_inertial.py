"""Causal SI-unit IMU transport and calibration for native ORB-SLAM3.

The estimator remains the pinned upstream IMU_STEREO implementation. These
helpers never synthesize IMU from evaluator poses, interpolate measurements, or
turn pre-initialization visual poses into claims of inertial initialization.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np

from .pose_metrics import transform


@dataclass(frozen=True)
class ImuSettings:
    frequency_hz: int = 200
    noise_gyro: float = .001
    noise_acc: float = .01
    gyro_walk: float = .00001
    acc_walk: float = .0001

    def validate(self):
        if type(self.frequency_hz) is not int or not 50 <= self.frequency_hz <= 2000:
            raise ValueError("IMU frequency must be an integer in [50, 2000] Hz")
        for name in ("noise_gyro", "noise_acc", "gyro_walk", "acc_walk"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Native IMU covariance parameters must be finite and positive")
        return asdict(self)


def write_stereo_inertial_settings(path, *, t_imu_camera, imu=ImuSettings(), **stereo):
    """Write native legacy Tbc: left optical camera coordinates into IMU frame.

    Noise values are declared engineering settings, not sensor calibration claims.
    Upstream scales the continuous-time noise densities by sqrt(frequency).
    """
    from .native_orb import write_rectified_stereo_settings
    parameters = imu.validate()
    t_i_c = transform(t_imu_camera)
    write_rectified_stereo_settings(path, **stereo)
    rows = ["Tbc: !!opencv-matrix", "  rows: 4", "  cols: 4", "  dt: f",
            "  data: [" + ",".join(str(float(x)) for x in t_i_c.ravel()) + "]",
            "InsertKFsWhenLost: 0", f"IMU.Frequency: {imu.frequency_hz}"]
    for name, field in (("NoiseGyro", "noise_gyro"), ("NoiseAcc", "noise_acc"),
                        ("GyroWalk", "gyro_walk"), ("AccWalk", "acc_walk")):
        rows.append(f"IMU.{name}: {float(parameters[field])}")
    with Path(path).open("a") as stream:
        stream.write("\n".join(rows) + "\n")
    return {"T_imu_camera": t_i_c.tolist(), "imu": parameters,
            "noise_scope": "fixed engineering noise densities; not calibrated physical sensor estimates"}


def validate_imu_batch(samples, timestamp_s, *, previous_imu_s=-math.inf, previous_frame_s=-math.inf):
    """Accept original [time, wx, wy, wz, ax, ay, az] rows exactly once.

    Accelerometer values are specific force in m/s² including gravity response,
    not world-frame kinematic acceleration. Timestamps share the camera clock.
    """
    if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)) or not math.isfinite(timestamp_s):
        raise ValueError("A finite camera timestamp is required")
    try:
        values = np.asarray(samples)
    except (TypeError, ValueError) as error:
        raise ValueError("IMU rows must be numeric SI measurements") from error
    if (values.ndim != 2 or values.shape[1:] != (7,) or not 1 <= len(values) <= 10000
            or values.dtype.kind not in "fiu" or not np.isfinite(values).all()):
        raise ValueError("IMU requires bounded finite rows of time, gyro xyz and specific force xyz")
    values = values.astype(float)
    times = values[:, 0]
    if (times[0] <= previous_imu_s or np.any(np.diff(times) <= 0)
            or times[-1] > timestamp_s + 1e-12):
        raise ValueError("IMU samples must be strictly increasing, unused and causal")
    intervals = list(np.diff(times))
    if math.isfinite(previous_imu_s): intervals.append(times[0] - previous_imu_s)
    if math.isfinite(previous_frame_s): intervals.append(times[0] - previous_frame_s)
    intervals.append(timestamp_s - times[-1])
    if max(intervals, default=0) > .020000000001:
        raise ValueError("Original IMU coverage cannot have gaps above 20 ms")
    return values.tolist()


def partition_imu(camera_timestamps, samples):
    """Partition a high-rate recording without duplicates or future samples."""
    times = np.asarray(camera_timestamps)
    rows = np.asarray(samples)
    if (times.ndim != 1 or len(times) < 1 or times.dtype.kind not in "fiu"
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Camera timestamps must increase strictly")
    if (rows.ndim != 2 or rows.shape[1:] != (7,) or not len(rows)
            or rows.dtype.kind not in "fiu" or not np.isfinite(rows).all()
            or np.any(np.diff(rows[:, 0]) <= 0)):
        raise ValueError("Original IMU recording must be finite and strictly ordered")
    if rows[0, 0] > times[0] or rows[-1, 0] < times[-1] - .020000000001:
        raise ValueError("Original IMU must span the camera recording")
    batches, cursor, previous_imu, previous_frame = [], 0, -math.inf, -math.inf
    for stamp in times:
        end = int(np.searchsorted(rows[:, 0], stamp, side="right"))
        batch = validate_imu_batch(rows[cursor:end], float(stamp), previous_imu_s=previous_imu,
                                   previous_frame_s=previous_frame)
        batches.append(batch)
        previous_imu, previous_frame, cursor = batch[-1][0], float(stamp), end
    return batches, {"original_imu_samples": len(rows), "consumed_imu_samples": cursor,
                     "unconsumed_tail_samples": len(rows)-cursor, "duplicate_or_interpolated_samples": 0}


def validate_inertial_response(value, batch):
    data = value.get("inertial")
    if not isinstance(data, dict) or data.get("schema") != "bhl-orb-inertial-frame-v1":
        raise ValueError("Actual native inertial participation telemetry is required")
    if type(data.get("imu_samples")) is not int or data["imu_samples"] != len(batch):
        raise ValueError("Native estimator IMU count differs from original batch")
    if type(data.get("map_imu_initialized")) is not bool:
        raise ValueError("Native map inertial initialization must be explicit")
    if data["map_imu_initialized"] and value.get("map_id") is None:
        raise ValueError("Native inertial initialization requires an observed map")
    for name, expected in (("first_imu_timestamp_s", batch[0][0]), ("last_imu_timestamp_s", batch[-1][0])):
        stamp = data.get(name)
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or not math.isclose(stamp, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Native IMU timestamp acknowledgement differs from original batch")
    return dict(data)
