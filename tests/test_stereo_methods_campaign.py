"""Synthetic transport fixtures exercise the real hybrid adapter boundary."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("bhl_stereo_methods_test",
    Path(__file__).resolve().parents[1]/"scripts/bench/stereo_methods_campaign.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def adapter(arm):
    client = MODULE.HybridOrbNavigationClient({"arm": arm}, np.eye(4), Path("unused"))
    class StereoFixture:
        def __init__(self): self.client = SimpleNamespace(responses=[])
        def track(self, scan, imu):
            native = {"timestamp_s": scan["timestamp_s"], "tracked": True, "map_id": scan["map_id"],
                      "T_W_C": np.eye(4).tolist(), "compute_seconds": .03,
                      "diagnostics": {"schema": "bhl-orb-frame-diagnostics-v1"}}
            self.client.responses.append(native)
            return {"timestamp_s": scan["timestamp_s"], "tracked": scan["map_id"] == 0,
                    "map_reset_id": int(scan["map_id"] != 0), "T_W_I": np.eye(4).tolist() if scan["map_id"] == 0 else None,
                    "state": "OK" if scan["map_id"] == 0 else "MAP_CHANGED_STOP", "compute_seconds": .03,
                    "additional_sensor_compute_seconds": .08, "native_ORB_T_W_C": native["T_W_C"]}
    class LioFixture:
        def __init__(self): self.inputs = []
        def track(self, scan, imu):
            self.inputs.append((scan, imu))
            return {"timestamp_s": scan["timestamp_s"], "tracked": True, "map_reset_id": 0,
                    "T_W_I": np.eye(4).tolist(), "effective_points": 100, "compute_seconds": .01}
    client.stereo, client.lio = StereoFixture(), LioFixture()
    return client


def test_permanent_stop_baseline_never_accepts_shadow_lio_after_map_change():
    client = adapter("permanent_stop")
    row = client.track({"timestamp_s": 1., "map_id": 4}, {"original_imu": "unit_fixture"})
    assert row["reference_native_LIO"]["tracked"]
    assert not row["tracked"]
    assert row["T_W_I"] is None
    assert row["lio_mode"] == "shadow_only"
    assert row["compute_seconds"] == pytest.approx(.04)
    assert row["additional_sensor_compute_seconds"] == .08


def test_hybrid_preserves_original_sensor_arguments_and_capture_latency():
    client = adapter("verified_lio_recovery")
    scan, imu = {"timestamp_s": 1., "map_id": 0}, {"original_imu": "unit_fixture"}
    row = client.track(scan, imu)
    assert client.lio.inputs[0][0] is scan
    assert client.lio.inputs[0][1] is imu
    assert row["tracked"]
    assert row["additional_sensor_compute_seconds"] == .08
    assert row["compute_seconds"] == pytest.approx(.04)
    assert row["ground_truth_inputs"] == []
    assert row["native_ORB_T_W_C"] == np.eye(4).tolist()


def test_map_changed_raw_pose_is_available_only_after_hybrid_verification():
    client = adapter("verified_lio_recovery")
    client.track({"timestamp_s": 1., "map_id": 0}, {})
    outputs = [client.track({"timestamp_s": 1.2+.2*i, "map_id": 7}, {}) for i in range(6)]
    assert not any(row["tracked"] for row in outputs[:-1])
    assert outputs[-1]["tracked"]
    assert outputs[-1]["map_reset_id"] == 0
    assert outputs[-1]["state"] == "VERIFIED_MAP_RECOVERY_RESUME"
    assert all(row["native_ORB_T_W_C"] is not None for row in outputs)
