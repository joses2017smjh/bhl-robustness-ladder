"""Preserve malformed raw scan evidence without inventing time or measurements.

The original native guard remains unchanged. Invalid scans are unavailable
measurements. The existing 350 ms pose-age stop handles their absence, and IMU
samples remain queued until the next valid scan so no integration data is lost.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np


def scan_time_problem(scan, previous_tail=None):
    times = np.asarray(scan["point_time_s"], dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        return "invalid_original_point_times"
    beginning, tail = float(times.min()), float(times.max())
    if tail <= beginning:
        return "zero_original_time_span"
    if previous_tail is not None and beginning <= previous_tail:
        return "overlapping_or_repeated_original_scan"
    if not math.isclose(float(scan["frame_timestamp_s"]), tail, rel_tol=0, abs_tol=1e-8):
        return "original_tail_mismatch"
    return None


def validated_sensor_class(base):
    class ValidatedTerrainSensors(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.rejected_scans = []
            self.last_valid_scan_tail = None

        def newest(self):
            while self.ready:
                scan = self.ready[-1]
                reason = scan_time_problem(scan, self.last_valid_scan_tail)
                if reason is None:
                    break
                # Keep the exact raw packet. In particular do not spread a
                # single-column timestamp or fabricate an endpoint return.
                self.rejected_scans.append(dict(reason=reason,
                    previous_valid_tail_s=self.last_valid_scan_tail,
                    imu_cursor_before_drop=self.imu_cursor,
                    packet={key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in scan.items()}))
                self.ready.pop()
            if not self.ready:
                return None
            packet = super().newest()
            if packet is not None:
                self.last_valid_scan_tail = float(packet[0]["frame_timestamp_s"])
            return packet

        def save(self, path):
            super().save(path)
            evidence = dict(schema="bhl-original-scan-rejections-v1", rejected_count=len(self.rejected_scans),
                action="unavailable measurement; unchanged native pose-age stop; IMU retained until next valid scan",
                synthetic_points_or_timestamps=False, rejected_scans=self.rejected_scans)
            with Path(path).with_name("rejected-scans.json").open("x") as stream:
                json.dump(evidence, stream, indent=2, allow_nan=False)
                stream.write("\n")
    return ValidatedTerrainSensors
