"""Scientific contract checks, including rejection of convenient fake results."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from bhl_robust.research.stereo_benchmark import (
    FastFoundationStereoMethod, SGBMMethod, checked_path, depth_metrics,
    disparity_depth, load_replay, normalize_and_pad, paired_sequence_bootstrap,
)
from bhl_robust.stereo_depth import RectifiedCalibration


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sensor_capture_test", ROOT/"scripts/bench/sensor_capture.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class StereoResearchTests(unittest.TestCase):
    def setUp(self):
        self.cal = RectifiedCalibration(160, 80, 160., .1, 79.5, 79.5, True)

    def test_real_sgbm_shift_has_known_metric_depth(self):
        # A textured plane image with a known 8-pixel baseline shift is not an
        # injected disparity. Actual SGBM must recover depth from the two RGBs.
        rng = np.random.default_rng(98)
        left = rng.integers(0, 256, size=(80, 160, 3), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, :-8] = left[:, 8:]
        depth, valid = SGBMMethod(self.cal)(left, right)
        central = valid[10:-10, 75:105]
        self.assertGreater(float(central.mean()), .5)
        self.assertAlmostEqual(float(np.nanmedian(depth[10:-10, 75:105])), 2., places=2)

    def test_disparity_offset_and_visibility(self):
        cal = RectifiedCalibration(160, 80, 160., .1, 82., 79., True)
        disparity = np.full((80, 160), 11., dtype=np.float32)
        disparity[0, 20] = np.nan
        depth, valid = disparity_depth(disparity, cal)
        self.assertFalse(valid[0, 20])
        self.assertFalse(valid[20, 10])
        self.assertTrue(valid[20, 11])
        self.assertAlmostEqual(float(depth[20, 100]), 2.)

    def test_padding_preserves_width_and_normalization(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        padded, shape = normalize_and_pad(image)
        self.assertEqual(padded.shape, (1, 3, 256, 320))
        self.assertEqual(shape, (240, 320))
        np.testing.assert_allclose(padded[0, :, 0, 0], -np.array([123.675, 116.28, 103.53])/np.array([58.395,57.12,57.375]), rtol=1.e-6)

    def test_coverage_reject_all_cannot_win_recall(self):
        truth = np.array([[1., 2.], [4., np.nan]], dtype=np.float32)
        prediction = np.full((2, 2), np.nan)
        metrics = depth_metrics(prediction, np.zeros((2, 2), dtype=bool), truth)
        self.assertEqual(metrics["coverage"], 0.)
        self.assertIsNone(metrics["near_obstacle_pixel_recall"])
        self.assertEqual(metrics["near_surface_pixel_recall_proxy"], 0.)
        self.assertIsNone(metrics["rmse_m"])

    def test_common_mask_does_not_replace_native_coverage(self):
        truth = np.ones((2, 2), dtype=np.float32)
        predicted = truth.copy()
        predicted[0, 0] = 3.
        common = np.array([[False, True], [False, False]])
        metrics = depth_metrics(predicted, np.ones((2, 2), dtype=bool), truth, common_mask=common,
                                obstacle_mask=np.ones((2, 2), dtype=bool))
        self.assertEqual(metrics["coverage"], 1.)
        self.assertEqual(metrics["rmse_m"], 0.)
        self.assertEqual(metrics["scored_pixels"], 1)
        self.assertEqual(metrics["near_obstacle_pixel_recall"], .75)

    def test_nearby_ground_is_excluded_from_obstacle_recall(self):
        truth = np.ones((2, 2), dtype=np.float32)
        predicted = truth.copy()
        predicted[0, 0] = 5.  # One independently labeled obstacle is missed.
        obstacles = np.array([[True, False], [False, False]])
        metrics = depth_metrics(predicted, np.ones((2, 2), dtype=bool), truth, obstacle_mask=obstacles)
        self.assertEqual(metrics["near_obstacle_pixel_recall"], 0.)
        self.assertEqual(metrics["near_surface_pixel_recall_proxy"], .75)

    def test_single_heldout_scene_has_no_fake_confidence_interval(self):
        result = paired_sequence_bootstrap([-.03])
        self.assertEqual(result["status"], "INSUFFICIENT_INDEPENDENT_GROUPS")
        self.assertNotIn("lower_95", result)

    def test_group_bootstrap_is_reproducible(self):
        self.assertEqual(paired_sequence_bootstrap([-.1,-.2,-.3]), paired_sequence_bootstrap([-.1,-.2,-.3]))

    def test_inference_cannot_request_ground_truth_namespace(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"evaluator"
            path.mkdir()
            (path/"truth.npz").write_bytes(b"gt")
            with self.assertRaises(ValueError):
                checked_path(folder, "evaluator/truth.npz", namespace="inference")
            with self.assertRaises(ValueError):
                checked_path(folder, "../truth.npz")

    def test_wrong_checkpoint_never_becomes_neural_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/"fake.onnx"
            checkpoint.write_bytes(b"fake_pretrained")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                FastFoundationStereoMethod(self.cal, checkpoint, provider="CPUExecutionProvider")

    def test_gpu_provider_claim_requires_actual_kernel_evidence(self):
        for actual_cuda in (True, False):
            with self.subTest(actual_cuda=actual_cuda):
                method = object.__new__(FastFoundationStereoMethod)
                method._profile_dir = tempfile.TemporaryDirectory()
                profile = Path(method._profile_dir.name)/"profile.json"
                profile.write_text(json.dumps([{"cat": "Node", "name": "Conv_kernel_time",
                    "args": {"provider": "CUDAExecutionProvider" if actual_cuda else "CPUExecutionProvider", "op_name": "Conv"}}]))
                method.session = SimpleNamespace(end_profiling=lambda: str(profile),
                    get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
                    get_provider_options=lambda: {"CUDAExecutionProvider": {"device_id": "0"}})
                if actual_cuda:
                    method._verify_first_gpu_execution()
                    self.assertEqual(method.provider_evidence["kernel_events_by_provider"]["CUDAExecutionProvider"], 1)
                else:
                    with self.assertRaisesRegex(RuntimeError, "no profiled CUDA"):
                        method._verify_first_gpu_execution()
                self.assertIsNone(method._profile_dir)

    def test_shared_torch_cuda_is_loaded_before_isolated_ort_session(self):
        events = []
        def available():
            events.append("torch_cuda_ready")
            return True
        torch = SimpleNamespace(__version__="2.7.0+cu128", version=SimpleNamespace(cuda="12.8"),
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90600)),
            cuda=SimpleNamespace(is_available=available, get_device_name=lambda index: "Test GPU",
                                 get_device_capability=lambda index: (8, 0)))
        def preload():
            self.assertIn("torch_cuda_ready", events)
            events.append("ort_preload")
        session = SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
            disable_fallback=lambda: None,
            get_inputs=lambda: [SimpleNamespace(name=name, type="tensor(float)") for name in ("left_image", "right_image")],
            get_outputs=lambda: [SimpleNamespace(name="disparity")])
        def construct(*args, **kwargs):
            self.assertIn("ort_preload", events)
            events.append("ort_session")
            return session
        ort = SimpleNamespace(__version__="1.22.0", preload_dlls=preload,
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
            SessionOptions=SimpleNamespace, InferenceSession=construct)
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)/"model.onnx"
            model.write_bytes(b"test")
            with patch.dict("sys.modules", {"torch": torch, "onnxruntime": ort}), \
                    patch("bhl_robust.research.stereo_benchmark.FFS_MODEL_BYTES", 4), \
                    patch("bhl_robust.research.stereo_benchmark.sha256_file", return_value="2c79bbb274dc0aef687a8732b987077e9e3aa26ff9dd653141cda48c57dad5d3"):
                method = FastFoundationStereoMethod(self.cal, model)
                self.assertEqual(events, ["torch_cuda_ready", "ort_preload", "ort_session"])
                self.assertEqual(method.provenance()["cuda_runtime"]["torch_cuda_version"], "12.8")
                method._profile_dir.cleanup()
                method._profile_dir = None

    def test_imu_is_uniform_across_camera_packet_boundaries(self):
        packets = [capture.imu_packet(i/15, 1/15) for i in range(10)]
        stamps = np.concatenate([packet["timestamp_s"] for packet in packets])
        np.testing.assert_allclose(np.diff(stamps), .005, atol=1.e-12)
        np.testing.assert_allclose(packets[0]["specific_force_m_s2"][:, 2], 9.81)

    def test_fixedrig_extrinsics_baseline_and_local_height(self):
        cal = capture.calibration()
        t_b_c, t_b_l, t_c_l = (np.array(cal[key]) for key in ("T_B_C", "T_B_L", "T_C_L"))
        np.testing.assert_allclose(t_b_c @ t_c_l, t_b_l)
        self.assertAlmostEqual(float(t_c_l[0, 3]), capture.BASELINE/2)
        self.assertAlmostEqual(float(np.array(cal["T_M_L"])[2, 3]), .75)

    def test_capture_cannot_render_outside_slurm(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                capture.capture(folder, frames_per_scene=1)


if __name__ == "__main__":
    unittest.main()
