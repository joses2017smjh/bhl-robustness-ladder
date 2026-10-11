"""Rectification diagnostics and explicit physical-sensor intake evidence.

These measurements do not recalibrate a camera or certify a robot. Feature
matches are selected without an epipolar filter so vertical error is observable.
"""
from __future__ import annotations

import numpy as np


def rectification_residuals(left_points, right_points, *, minimum_matches=20,
                            maximum_p95_px=2.0):
    left = np.asarray(left_points, dtype=float).reshape(-1, 2)
    right = np.asarray(right_points, dtype=float).reshape(-1, 2)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("matching finite Nx2 point arrays required")
    if minimum_matches < 1 or not np.isfinite(maximum_p95_px) or maximum_p95_px <= 0:
        raise ValueError("positive diagnostic thresholds required")
    n = len(left)
    if n == 0:
        return {"matches": 0, "status": "INSUFFICIENT_MATCHES", "vertical_p95_px": None,
                "vertical_rmse_px": None, "vertical_median_px": None,
                "signed_vertical_median_px": None, "positive_disparity_fraction": None,
                "fraction_within_2px": None}
    signed = left[:, 1] - right[:, 1]
    absolute = np.abs(signed)
    p95 = float(np.quantile(absolute, .95))
    return {"matches": n, "status": "INSUFFICIENT_MATCHES" if n < minimum_matches else
            ("WITHIN_DIAGNOSTIC_LIMIT" if p95 <= maximum_p95_px else "EXCEEDS_DIAGNOSTIC_LIMIT"),
            "vertical_p95_px": p95, "vertical_rmse_px": float(np.sqrt(np.mean(signed ** 2))),
            "vertical_median_px": float(np.median(absolute)),
            "signed_vertical_median_px": float(np.median(signed)),
            "positive_disparity_fraction": float(np.mean(left[:, 0] > right[:, 0])),
            "fraction_within_2px": float(np.mean(absolute <= 2.0))}


def match_rectified_pair(left, right, *, ratio=.75, max_features=2000):
    """Mutual descriptor-ratio SIFT matches, with no row/disparity preselection."""
    import cv2
    if left is None or right is None or left.shape != right.shape or left.ndim != 2:
        raise ValueError("same-size grayscale images required")
    if not 0 < ratio < 1 or max_features < 2:
        raise ValueError("invalid feature matching settings")
    detector = cv2.SIFT_create(nfeatures=max_features)
    kl, dl = detector.detectAndCompute(left, None)
    kr, dr = detector.detectAndCompute(right, None)
    if dl is None or dr is None or len(dl) < 2 or len(dr) < 2:
        return np.empty((0, 2)), np.empty((0, 2)), {"features_left": len(kl), "features_right": len(kr)}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def accepted(a, b):
        return {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance}
    forward, backward = accepted(dl, dr), accepted(dr, dl)
    pairs = [(i, j) for i, j in sorted(forward.items()) if backward.get(j) == i]
    lp = np.asarray([kl[i].pt for i, _ in pairs], dtype=float).reshape(-1, 2)
    rp = np.asarray([kr[j].pt for _, j in pairs], dtype=float).reshape(-1, 2)
    return lp, rp, {"features_left": len(kl), "features_right": len(kr)}


def hardware_evidence(manifest, *, lidar_packets_valid, epipolar_by_sequence,
                      provenance_files_verified=False):
    """Conservative readiness result; declarations alone never prove hardware use."""
    checks = {}
    sequences = manifest.get("sequences", [])
    checks["physical_origin"] = bool(sequences) and all(s.get("origin") == "physical" for s in sequences)
    rig = manifest.get("physical_rig", {})
    checks["recorded_device_identity"] = all(
        isinstance(rig.get(k), str) and bool(rig[k].strip())
        for k in ("robot_or_rig", "left_camera", "right_camera", "lidar", "imu", "capture_date_utc"))
    checks["clock_sync_evidence_declared"] = bool(rig.get("clock_sync_evidence"))
    checks["calibration_evidence_declared"] = bool(rig.get("calibration_evidence"))
    checks["provenance_files_hash_verified"] = provenance_files_verified is True
    stamp_pairs = [(f.get("left_timestamp_s"), f.get("right_timestamp_s"))
                   for s in sequences for f in s.get("frames", [])]
    checks["stereo_recorded_timestamps_within_1ms"] = bool(stamp_pairs) and all(
        all(isinstance(t, (int, float)) and not isinstance(t, bool) and np.isfinite(t) for t in pair)
        and abs(pair[0] - pair[1]) <= .001 for pair in stamp_pairs)
    timing_consistent = bool(sequences)
    for sequence in sequences:
        frames = sequence.get("frames", [])
        stamps = [[frame.get(k) for k in ("left_timestamp_s", "right_timestamp_s", "timestamp_s")]
                  for frame in frames]
        if len(stamps) < 2 or not all(
                isinstance(t, (int, float)) and not isinstance(t, bool) and np.isfinite(t)
                for row in stamps for t in row):
            timing_consistent = False
            continue
        times = np.asarray(stamps, dtype=float)
        timing_consistent &= bool(np.all(np.diff(times, axis=0) > 0)
                                  and np.all(np.abs(times[:, :2].mean(axis=1) - times[:, 2]) <= .001))
    checks["camera_clocks_match_canonical_monotonic_time"] = bool(timing_consistent)
    checks["timed_3d_lidar_and_imu_valid"] = lidar_packets_valid is True
    expected_ids = {s.get("id") for s in sequences}
    checks["all_sequences_rectification_measured"] = bool(expected_ids) and set(epipolar_by_sequence) == expected_ids
    checks["rectification_diagnostic_within_limit"] = bool(epipolar_by_sequence) and all(
        row.get("status") == "WITHIN_DIAGNOSTIC_LIMIT" for row in epipolar_by_sequence.values())
    missing = [k for k, passed in checks.items() if not passed]
    return {"schema": "bhl-physical-intake-evidence-v1", "checks": checks,
            "status": "READY_FOR_NATIVE_REPLAY" if not missing else "INCOMPLETE_PHYSICAL_EVIDENCE",
            "missing_requirements": missing, "hardware_validation_complete": False,
            "limitations": ["Intake measurements do not demonstrate physical closed-loop navigation.",
                            "Device, clock and calibration provenance still require independent inspection.",
                            "Vertical residual alone cannot identify baseline, focal-length or all extrinsic errors."]}
