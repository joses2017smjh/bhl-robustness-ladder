"""Calibrated CPU stereo matching for already rectified image pairs.

No simulator ground truth is used. Rectification/calibration must be supplied;
this module cannot infer a physical module's baseline, focal length or timing.
Output is left-camera optical-axis z in meters, with NaN for invalid pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class RectifiedCalibration:
    image_width: int
    image_height: int
    fx_px: float
    baseline_m: float
    cx_left_px: float
    cx_right_px: float
    rectified: bool

    def __post_init__(self):
        if self.rectified is not True:
            raise ValueError("Input images must already be horizontally rectified")
        if (not isinstance(self.image_width, int) or not isinstance(self.image_height, int)
                or self.image_width < 16 or self.image_height < 8):
            raise ValueError("Calibration requires positive integer image dimensions")
        if not all(math.isfinite(x) for x in
                   (self.fx_px, self.baseline_m, self.cx_left_px, self.cx_right_px)):
            raise ValueError("Calibration values must be finite")
        if self.fx_px <= 0 or self.baseline_m <= 0:
            raise ValueError("Focal length in pixels and baseline in meters must be positive")

    @classmethod
    def from_dict(cls, value):
        if value.get("schema") != "bhl-rectified-stereo-v1":
            raise ValueError("Expected calibration schema bhl-rectified-stereo-v1")
        # Every field is required, including principal points and rectification.
        return cls(**{key: value[key] for key in cls.__dataclass_fields__})


def disparities_to_depth(left, right, calibration, *, left_valid=None,
                         right_valid=None, lr_threshold_px=1.0,
                         min_depth_m=0.10, max_depth_m=12.0):
    """Reject inconsistent correspondences before converting d to metric z.

    Left disparity is uL-uR; right disparity is uR-uL. Positive physical
    disparity is d-(cxL-cxR). Right disparity is sampled at the rounded uR;
    the default one-pixel tolerance includes subpixel/rounding differences.
    """
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    shape = (calibration.image_height, calibration.image_width)
    if left.shape != shape or right.shape != shape:
        raise ValueError("Disparity dimensions do not match calibration")
    if (not all(math.isfinite(v) for v in (lr_threshold_px, min_depth_m, max_depth_m))
            or lr_threshold_px < 0 or not 0 < min_depth_m < max_depth_m):
        raise ValueError("Invalid depth interval or consistency threshold")
    lv = np.isfinite(left)
    rv = np.isfinite(right)
    for mask, target in ((left_valid, lv), (right_valid, rv)):
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != shape or mask.dtype != np.bool_:
                raise ValueError("Pixel validity masks must be boolean and match calibration")
            target &= mask
    yy, xx = np.indices(shape)
    xr = xx - np.where(lv, left, 0.0)
    inside = np.isfinite(xr) & (xr >= 0) & (xr <= shape[1] - 1)
    ix = np.rint(np.clip(np.nan_to_num(xr), 0, shape[1] - 1)).astype(np.int64)
    paired = right[yy, ix]
    disparity = left - (calibration.cx_left_px - calibration.cx_right_px)
    valid = lv & inside & rv[yy, ix] & (disparity > 0)
    valid &= np.abs(left + paired) <= lr_threshold_px
    depth = np.full(shape, np.nan, dtype=np.float64)
    np.divide(calibration.fx_px * calibration.baseline_m, disparity, out=depth, where=valid)
    valid &= np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    depth[~valid] = np.nan
    return depth.astype(np.float32), valid


def estimate_rectified_depth(left, right, calibration, *, num_disparities=64,
                             min_disparity=0, block_size=5, lr_threshold_px=1.0,
                             min_depth_m=0.10, max_depth_m=12.0,
                             min_texture_std=2.0, left_valid=None, right_valid=None):
    """Return (depth_z_m, valid_mask, left_disparity_px) from real images.

    Images must be uint8 grayscale or RGB, with the calibrated dimensions.
    Optional masks exclude invalid rectification borders. No hole filling is
    performed: unsupported pixels remain explicitly invalid.
    """
    import cv2
    shape = (calibration.image_height, calibration.image_width)
    if (not isinstance(num_disparities, int) or num_disparities <= 0 or num_disparities % 16
            or not isinstance(min_disparity, int)
            or not isinstance(block_size, int) or block_size < 3 or block_size % 2 == 0
            or num_disparities + abs(min_disparity) + block_size >= shape[1]
            or block_size >= shape[0]):
        raise ValueError("Invalid SGBM disparity search or block size for image dimensions")
    if not math.isfinite(min_texture_std) or min_texture_std < 0:
        raise ValueError("Texture threshold must be finite and nonnegative")

    def grayscale(image):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.shape not in (shape, (*shape, 3)):
            raise ValueError("Images must be uint8 gray/RGB with calibrated dimensions")
        return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def textured(image, mask):
        values = image.astype(np.float32)
        mean = cv2.boxFilter(values, -1, (block_size, block_size))
        mean2 = cv2.boxFilter(values * values, -1, (block_size, block_size))
        result = np.maximum(mean2 - mean * mean, 0) >= min_texture_std**2
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != shape or mask.dtype != np.bool_:
                raise ValueError("Pixel validity masks must be boolean and match calibration")
            result &= mask
        return result

    left, right = grayscale(left), grayscale(right)
    params = dict(numDisparities=num_disparities, blockSize=block_size,
                  P1=8 * block_size**2, P2=32 * block_size**2,
                  disp12MaxDiff=1, preFilterCap=31, uniquenessRatio=10,
                  speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    right_min = -min_disparity - num_disparities + 1
    raw_l = cv2.StereoSGBM_create(minDisparity=min_disparity, **params).compute(left, right)
    raw_r = cv2.StereoSGBM_create(minDisparity=right_min, **params).compute(right, left)
    # OpenCV encodes subpixel disparity as signed fixed-point with scale16.
    dl, dr = raw_l.astype(np.float32) / 16, raw_r.astype(np.float32) / 16
    lv = (raw_l > (min_disparity - 1) * 16) & textured(left, left_valid)
    rv = (raw_r > (right_min - 1) * 16) & textured(right, right_valid)
    depth, valid = disparities_to_depth(dl, dr, calibration, left_valid=lv, right_valid=rv,
        lr_threshold_px=lr_threshold_px, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    dl[~valid] = np.nan
    return depth, valid, dl


def load_depth_metadata(metadata_path, left_image, capture_time):
    """Load a stereo artifact for SSD, checking registration and capture time."""
    import hashlib
    import json
    from pathlib import Path
    record = json.loads(Path(metadata_path).read_text())
    if (record.get("schema") != "bhl-stereo-depth-v1"
            or record.get("depth_source") != "stereo_estimate"
            or record.get("depth_convention") != "left_camera_optical_axis_z_m"):
        raise ValueError("Unsupported stereo depth metadata convention/source")
    left_image = Path(left_image).resolve()
    if (Path(record["left_image"]).resolve() != left_image
            or hashlib.sha256(left_image.read_bytes()).hexdigest() != record["left_image_sha256"]):
        raise ValueError("Depth is not registered to this left image")
    stamp = float(record["depth_capture_time"])
    if not math.isfinite(capture_time) or not math.isfinite(stamp) or abs(stamp - capture_time) > 1.e-6:
        raise ValueError("RGB capture timestamp does not match stereo metadata")
    depth = np.load(record["depth_path"], allow_pickle=False)
    valid = np.load(record["valid_mask_path"], allow_pickle=False)
    shape = (record["calibration"]["image_height"], record["calibration"]["image_width"])
    if depth.shape != shape or valid.shape != shape or valid.dtype != np.bool_:
        raise ValueError("Stereo depth/mask dimensions or mask dtype are invalid")
    if not np.isfinite(depth[valid]).all() or (depth[valid] <= 0).any():
        raise ValueError("Stereo metadata marks a nonpositive/nonfinite depth valid")
    depth = depth.astype(np.float32)
    depth[~valid] = np.nan
    return depth, stamp, record["depth_source"]
