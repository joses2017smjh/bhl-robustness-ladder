"""Auditable rectified stereo replay, without simulator truth in inference.

The official pretrained C-Fast-FoundationStereo ONNX checkpoint is the
commercially licensed member of the requested Fast-FoundationStereo family.
Model statistics in publications are never substituted for measurements here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time
import numpy as np

from bhl_robust.stereo_depth import RectifiedCalibration, estimate_rectified_depth

FFS_MODEL_REVISION = "53a84e28a368f37e699776250215fcb450d6a109"
FFS_UPSTREAM_COMMIT = "476f4249561f7c79ca707326954f9255643412a6"
FFS_MODEL_SHA256 = "2c79bbb274dc0aef687a8732b987077e9e3aa26ff9dd653141cda48c57dad5d3"
FFS_MODEL_BYTES = 103228473
FFS_MODEL_URL = (
    "https://huggingface.co/nvidia/c-fast-foundationstereo/resolve/"
    + FFS_MODEL_REVISION + "/onnx/c_fast_foundationstereo_bp2_dynbatch_dynhw.onnx"
)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_path(root, relative, *, namespace=None):
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Replay paths must be contained relative paths")
    if namespace is not None and (not rel.parts or rel.parts[0] != namespace):
        raise ValueError(f"Expected {namespace} namespace, received {relative}")
    cursor = root
    for component in rel.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError("Replay paths cannot follow symbolic links")
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Replay path escapes dataset or is missing")
    return path


def load_replay(root):
    """Validate group splits and every recorded digest before inference."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "bhl-sensor-replay-v1":
        raise ValueError("Unsupported replay schema")
    calibration = RectifiedCalibration.from_dict(manifest["calibration"])
    splits = {}
    seen = set()
    for sequence in manifest["sequences"]:
        scene = sequence["scene_id"]
        split = sequence["split"]
        if split not in {"dev", "validation", "test"}:
            raise ValueError("Invalid sequence split")
        if scene in splits and splits[scene] != split:
            raise ValueError("Scene crosses development and evaluation splits")
        splits[scene] = split
        previous = -np.inf
        for frame in sequence["frames"]:
            identity = (sequence["id"], frame["frame_id"])
            stamp = float(frame["timestamp_s"])
            if identity in seen or not np.isfinite(stamp) or stamp <= previous:
                raise ValueError("Duplicate frame or unordered/nonfinite timestamp")
            previous = stamp
            seen.add(identity)
            for key in ("left", "right", "lidar", "imu"):
                checked_path(root, frame[key], namespace="inference")
            checked_path(root, frame["truth"], namespace="evaluator")
    if not seen:
        raise ValueError("Empty replay")
    files = manifest.get("file_sha256", {})
    for sequence in manifest["sequences"]:
        for frame in sequence["frames"]:
            for key in ("left", "right", "lidar", "imu", "truth"):
                rel = frame[key]
                if files.get(rel) != sha256_file(checked_path(root, rel)):
                    raise ValueError(f"Replay digest mismatch: {rel}")
    return manifest, calibration


def load_rgb(path):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_and_pad(image):
    """Pixel-domain ImageNet normalization; pad only, preserving disparity."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB image")
    h, w = image.shape[:2]
    if h < 32 or w < 32:
        raise ValueError("Images must be at least 32 pixels per dimension")
    ph, pw = (-h) % 32, (-w) % 32
    image = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="edge")
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    value = (image.astype(np.float32) - mean) / std
    return np.ascontiguousarray(value.transpose(2, 0, 1)[None]), (h, w)


def disparity_depth(disparity, calibration, *, min_depth_m=0.1, max_depth_m=12.0):
    disparity = np.asarray(disparity, dtype=np.float32)
    shape = (calibration.image_height, calibration.image_width)
    if disparity.shape != shape:
        raise ValueError("Disparity does not match original calibration")
    effective = disparity - (calibration.cx_left_px - calibration.cx_right_px)
    xx = np.indices(shape)[1]
    valid = np.isfinite(effective) & (effective > 0) & (xx - disparity >= 0)
    depth = np.full(shape, np.nan, dtype=np.float32)
    np.divide(calibration.fx_px * calibration.baseline_m, effective, out=depth, where=valid)
    valid &= np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    depth[~valid] = np.nan
    return depth, valid


class SGBMMethod:
    name = "sgbm"

    def __init__(self, calibration):
        import cv2
        cv2.setNumThreads(1)
        self.calibration = calibration

    def __call__(self, left, right):
        depth, valid, _ = estimate_rectified_depth(left, right, self.calibration,
            num_disparities=64, block_size=5, lr_threshold_px=1.0)
        return depth, valid

    def provenance(self):
        import cv2
        return {"method": self.name, "opencv_version": cv2.__version__,
                "opencv_threads": cv2.getNumThreads(),
                "num_disparities": 64, "block_size": 5, "lr_threshold_px": 1.0,
                "confidence_kind": "binary_match_support_not_calibrated_probability"}


class FastFoundationStereoMethod:
    name = "c_fast_foundationstereo"

    def __init__(self, calibration, model_path, *, provider="CUDAExecutionProvider"):
        self.calibration = calibration
        self.model_path = Path(model_path)
        if (self.model_path.stat().st_size != FFS_MODEL_BYTES
                or sha256_file(self.model_path) != FFS_MODEL_SHA256):
            raise ValueError("Official pretrained checkpoint size/SHA256 mismatch")
        self.cuda_runtime = None
        if provider == "CUDAExecutionProvider":
            # ORT documents importing PyTorch BEFORE session construction as
            # the supported way to preload its bundled CUDA/cuDNN libraries.
            # In an isolated ORT wheel, its own package discovery need not find
            # the existing shared venv's NVIDIA packages under cleanenv.
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA runtime is unavailable; GPU inference required")
            self.cuda_runtime = {"loader_strategy": "import_torch_before_onnxruntime_session",
                "torch_version": str(torch.__version__), "torch_cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "device_index": 0, "device_name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "documentation": "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#compatibility-with-pytorch"}
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls") and provider == "CUDAExecutionProvider":
            ort.preload_dlls()
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"Required ONNX provider unavailable: {provider}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._profile_dir = None
        self.provider_evidence = None
        if provider == "CUDAExecutionProvider":
            # ORT legitimately assigns shape/control operators to CPU. Requiring
            # every node on CUDA would reject a valid exported graph. Instead,
            # prove actual CUDA kernel execution on the first prediction.
            self._profile_dir = tempfile.TemporaryDirectory(prefix="bhl-ffs-profile-")
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(self._profile_dir.name)/"execution")
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options,
                                            providers=[provider])
        if provider not in self.session.get_providers():
            raise RuntimeError("Requested provider did not initialize")
        self.session.disable_fallback()
        inputs = self.session.get_inputs()
        if ({v.name for v in inputs} != {"left_image", "right_image"}
                or any(v.type != "tensor(float)" for v in inputs)):
            raise ValueError("Unexpected official ONNX input contract")
        if "disparity" not in {v.name for v in self.session.get_outputs()}:
            raise ValueError("Unexpected official ONNX disparity output")
        self.provider = provider
        self.ort_version = ort.__version__

    def _verify_first_gpu_execution(self):
        if self._profile_dir is None:
            return
        try:
            profile_path = Path(self.session.end_profiling())
            if not profile_path.resolve().is_relative_to(Path(self._profile_dir.name).resolve()):
                raise RuntimeError("Unexpected ONNX profiling output path")
            events = json.loads(profile_path.read_text())
            counts, ops = {}, {}
            for event in events:
                args = event.get("args", {})
                provider = args.get("provider")
                if event.get("cat") == "Node" and provider and "kernel_time" in event.get("name", ""):
                    counts[provider] = counts.get(provider, 0) + 1
                    if provider == "CUDAExecutionProvider":
                        op = args.get("op_name", "unknown")
                        ops[op] = ops.get(op, 0) + 1
            if (counts.get("CUDAExecutionProvider", 0) < 1
                    or "CUDAExecutionProvider" not in self.session.get_providers()):
                raise RuntimeError("First prediction executed no profiled CUDA kernels")
            self.provider_evidence = {"status": "VERIFIED_CUDA_KERNEL_EXECUTION",
                "kernel_events_by_provider": counts, "cuda_operator_events": ops,
                "session_providers": self.session.get_providers(),
                "provider_options": self.session.get_provider_options(),
                "profiling_scope": "first_prediction_only_excluded_from_warm_latency",
                "cpu_shape_control_nodes_allowed": True}
        finally:
            self._profile_dir.cleanup()
            self._profile_dir = None

    def __call__(self, left, right):
        if left.shape != right.shape:
            raise ValueError("Stereo dimensions differ")
        l, (h, w) = normalize_and_pad(left)
        r, _ = normalize_and_pad(right)
        disparity = self.session.run(["disparity"], {"left_image": l, "right_image": r})[0]
        self._verify_first_gpu_execution()
        disparity = np.asarray(disparity).reshape(l.shape[2:])[:h, :w]
        # This network exports only left disparity. Its native visibility mask
        # is evaluated separately from SGBM's stricter LR-consistency mask.
        return disparity_depth(disparity, self.calibration)

    def provenance(self):
        return {"method": self.name, "family": "Fast-FoundationStereo",
                "model_sha256": FFS_MODEL_SHA256, "model_bytes": FFS_MODEL_BYTES,
                "model_revision": FFS_MODEL_REVISION, "model_url": FFS_MODEL_URL,
                "upstream_code_commit": FFS_UPSTREAM_COMMIT,
                "license": "NVIDIA Open Model Agreement", "provider": self.provider,
                "onnxruntime_version": self.ort_version,
                "cuda_runtime": self.cuda_runtime,
                "provider_execution_evidence": self.provider_evidence,
                "normalization": "ImageNet_RGB_pixel_domain", "padding": "edge_bottom_right_to_32",
                "confidence_kind": "binary_visibility_support_not_calibrated_probability",
                "left_right_consistency": "not_available_from_single_left_disparity_export"}


def depth_metrics(prediction, valid, truth, *, common_mask=None, obstacle_mask=None,
                  near_m=3.0, tolerance_m=0.15):
    """Evaluate depth/coverage and an explicit near-obstacle PIXEL recall proxy."""
    prediction, valid, truth = np.asarray(prediction), np.asarray(valid), np.asarray(truth)
    if prediction.shape != truth.shape or valid.shape != truth.shape or valid.dtype != np.bool_:
        raise ValueError("Metric shapes/mask disagree")
    gt = np.isfinite(truth) & (truth >= 0.1) & (truth <= 12.0)
    native = valid & np.isfinite(prediction) & (prediction > 0) & gt
    chosen = native if common_mask is None else native & np.asarray(common_mask, dtype=bool)
    error = prediction[chosen].astype(float) - truth[chosen].astype(float)
    near_surface = gt & (truth <= near_m)
    if obstacle_mask is not None:
        obstacle_mask = np.asarray(obstacle_mask)
        if obstacle_mask.shape != truth.shape or obstacle_mask.dtype != np.bool_:
            raise ValueError("Obstacle truth mask must be boolean with the depth shape")
        near = near_surface & obstacle_mask
    else:
        near = np.zeros_like(gt)
    recalled = near & native & (np.abs(prediction - truth) <= tolerance_m)
    recalled_surface = near_surface & native & (np.abs(prediction - truth) <= tolerance_m)
    metrics = {"gt_pixels": int(gt.sum()), "valid_gt_pixels": int(native.sum()),
        "scored_pixels": int(chosen.sum()), "coverage": float(native.sum()/gt.sum()) if gt.any() else None,
        "mae_m": float(np.abs(error).mean()) if error.size else None,
        "rmse_m": float(np.sqrt(np.square(error).mean())) if error.size else None,
        "abs_rel": float((np.abs(error)/truth[chosen]).mean()) if error.size else None,
        "near_obstacle_pixels": int(near.sum()), "recalled_near_obstacle_pixels": int(recalled.sum()),
        "near_obstacle_pixel_recall": float(recalled.sum()/near.sum()) if near.any() else None,
        "near_surface_pixel_recall_proxy": (float(recalled_surface.sum()/near_surface.sum())
                                             if near_surface.any() else None),
        "near_range_m": near_m, "recall_depth_tolerance_m": tolerance_m,
        "recall_kind": ("independently_labeled_near_obstacle_pixel_recall_not_instance_detection"
                        if obstacle_mask is not None else "obstacle_labels_missing_surface_proxy_only")}
    return metrics


def paired_sequence_bootstrap(sequence_differences, *, seed=20261008, samples=2000):
    """Block bootstrap over independent held-out scenes, never frame samples."""
    values = np.asarray(sequence_differences, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Differences must be finite sequence-level values")
    if values.size < 3:
        return {"status": "INSUFFICIENT_INDEPENDENT_GROUPS", "groups": int(values.size)}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(samples, values.size))].mean(axis=1)
    return {"status": "ESTIMATED", "groups": int(values.size), "paired_mean": float(values.mean()),
            "lower_95": float(np.quantile(means, .025)), "upper_95": float(np.quantile(means, .975)),
            "bootstrap_seed": seed, "samples": samples}


def timed_replay(method, pairs, *, warmup=30, timed=200):
    """Disk PNG decode, preprocessing, model, depth and mask; no GT or saving.

    Repeated fixed inputs are a warm-cache latency microbenchmark. Cold session
    creation is reported by the caller; latency is not a camera-live deadline.
    """
    if warmup < 0 or timed < 1 or not pairs:
        raise ValueError("Invalid timing counts/input pairs")
    times = []
    for i in range(warmup + timed):
        left, right = pairs[i % len(pairs)]
        start = time.perf_counter_ns()
        method(load_rgb(left), load_rgb(right))
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if i >= warmup:
            times.append(elapsed)
    return {"warmup_iterations": warmup, "timed_iterations": timed,
            "p50_ms": float(np.quantile(times, .5)), "p95_ms": float(np.quantile(times, .95)),
            "p99_ms": float(np.quantile(times, .99)), "samples_ms": times,
            "scope": "warm_cache_PNG_decode_preprocess_inference_depth_validity",
            "excluded": ["sensor_capture", "rectification_already_rectified", "fusion", "output_serialization"]}
