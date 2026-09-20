#!/usr/bin/env python3
"""Compute actual calibrated rectified stereo depth, mask, disparity and metadata."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import cv2
import numpy as np
from bhl_robust.stereo_depth import RectifiedCalibration, estimate_rectified_depth


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", required=True, type=Path)
    p.add_argument("--right", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path)
    p.add_argument("--left-time", required=True, type=float, help="Original left capture timestamp, seconds")
    p.add_argument("--right-time", required=True, type=float, help="Original right capture timestamp, seconds")
    p.add_argument("--max-skew", default=0.035, type=float)
    p.add_argument("--num-disparities", default=64, type=int)
    p.add_argument("--min-disparity", default=0, type=int)
    p.add_argument("--block-size", default=5, type=int)
    p.add_argument("--lr-threshold", default=1.0, type=float)
    p.add_argument("--max-depth", default=12.0, type=float)
    p.add_argument("--left-valid-mask", type=Path)
    p.add_argument("--right-valid-mask", type=Path)
    p.add_argument("--out", required=True, type=Path, help="New output directory (will not overwrite)")
    a = p.parse_args()
    if not all(math.isfinite(v) for v in (a.left_time, a.right_time, a.max_skew)) or a.max_skew < 0:
        p.error("Capture timestamps and allowed skew must be finite")
    if abs(a.left_time - a.right_time) > a.max_skew:
        p.error("Stereo capture timestamp skew exceeds the allowed limit")
    if a.out.exists():
        raise FileExistsError(f"refusing to overwrite {a.out}")
    calibration = RectifiedCalibration.from_dict(json.loads(a.calibration.read_text()))
    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in (a.left, a.right)]
    if any(image is None for image in images):
        raise ValueError("Could not decode both input images")
    started = time.perf_counter()
    depth, valid, disparity = estimate_rectified_depth(*images, calibration,
        num_disparities=a.num_disparities, min_disparity=a.min_disparity,
        block_size=a.block_size, lr_threshold_px=a.lr_threshold, max_depth_m=a.max_depth,
        left_valid=np.load(a.left_valid_mask, allow_pickle=False) if a.left_valid_mask else None,
        right_valid=np.load(a.right_valid_mask, allow_pickle=False) if a.right_valid_mask else None)
    elapsed = time.perf_counter() - started
    a.out.mkdir(parents=True)
    np.save(a.out / "depth_z_m.npy", depth, allow_pickle=False)
    np.save(a.out / "valid.npy", valid, allow_pickle=False)
    np.save(a.out / "disparity_px.npy", disparity, allow_pickle=False)
    metadata = {"schema": "bhl-stereo-depth-v1", "method": "OpenCV StereoSGBM",
        "opencv_version": cv2.__version__, "depth_source": "stereo_estimate",
        "depth_convention": "left_camera_optical_axis_z_m", "invalid_depth": "NaN",
        "left_image": str(a.left.resolve()), "right_image": str(a.right.resolve()),
        "left_image_sha256": hashlib.sha256(a.left.read_bytes()).hexdigest(),
        "right_image_sha256": hashlib.sha256(a.right.read_bytes()).hexdigest(),
        "left_capture_time": a.left_time, "right_capture_time": a.right_time,
        "depth_capture_time": a.left_time, "capture_skew_s": abs(a.left_time - a.right_time),
        "computed_at": time.time(), "compute_s": elapsed, "valid_fraction": float(valid.mean()),
        "calibration": asdict(calibration), "calibration_path": str(a.calibration.resolve()),
        "calibration_sha256": hashlib.sha256(a.calibration.read_bytes()).hexdigest(),
        "depth_path": str((a.out / "depth_z_m.npy").resolve()),
        "valid_mask_path": str((a.out / "valid.npy").resolve()),
        "matcher": {"num_disparities": a.num_disparities, "min_disparity": a.min_disparity,
                    "block_size": a.block_size, "lr_threshold_px": a.lr_threshold,
                    "max_depth_m": a.max_depth}}
    (a.out / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(a.out.resolve()), "valid_fraction": float(valid.mean()),
                      "compute_s": elapsed}), flush=True)


if __name__ == "__main__":
    main()
