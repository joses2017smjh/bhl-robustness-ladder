"""Scientific scoring boundaries for native replay, independent of runtimes."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("native_slam_test", Path(__file__).parents[1] / "scripts/bench/native_slam_campaign.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def pose(x, yaw=0.):
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.eye(4)
    p[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    p[0, 3] = x
    return p


def test_reference_interpolation_preserves_metric_scale_and_rotation():
    result = MOD.interpolate_truth([0., 1.], [pose(0., 0.), pose(2., math.pi/2)], .5)
    np.testing.assert_allclose(result, pose(1., math.pi/4), atol=1e-12)


@pytest.mark.parametrize("target", [-.1, 1.1])
def test_evaluator_does_not_extrapolate_missing_lidar_tail(target):
    assert MOD.interpolate_truth([0., 1.], [pose(0.), pose(1.)], target) is None


def test_map_reset_is_never_hidden_by_one_global_alignment():
    rows = [{"map_id": None}, {"map_id": 0}, {"map_id": None}, {"map_id": 0}, {"map_id": 1}]
    parts = MOD.segments(rows, method="orb")
    assert [len(part) for part in parts] == [4, 1]
    assert parts[1][0]["map_id"] == 1


def test_lost_initialization_is_a_negative_not_a_missing_metric_crash(tmp_path):
    (tmp_path / "evaluator").mkdir()
    frames, hashes = [], {}
    for index in range(3):
        name = f"evaluator/{index}.npz"
        np.savez(tmp_path / name, T_W_C=pose(index), T_W_I=pose(index))
        hashes[name] = MOD.sha256(tmp_path / name)
        frames.append({"truth": name, "timestamp_s": float(index)})
    rows = [{"timestamp_s": float(i), "tracked": False, "T_W_C": None,
             "compute_seconds": .01, "map_id": None} for i in range(3)]
    gate = {"initialization_exclusion_s": 0., "minimum_tracking_fraction": .9,
            "maximum_ate_rmse_m": .1, "maximum_rotation_p95_deg": 5.,
            "maximum_native_compute_p95_s": .1}
    manifest = {"calibration": {"T_B_C": np.eye(4).tolist()},
                "file_sha256": hashes, "frame_rate_hz": 1.}
    result = MOD.evaluate(rows, method="orb", replay_root=tmp_path,
                          manifest=manifest, sequence={"frames": frames}, gate=gate)
    assert result["replay_readiness_gate"] == "NEGATIVE"
    assert result["segments"][0]["metrics"] is None
    assert result["post_initialization_tracked_fraction"] == 0.


def test_original_evaluator_hash_required(tmp_path):
    np.savez(tmp_path / "truth.npz", T_W_C=pose(0.))
    with pytest.raises(ValueError, match="hash mismatch"):
        MOD.evaluate([], method="orb", replay_root=tmp_path,
                     manifest={"calibration": {"T_B_C": np.eye(4)}, "file_sha256": {"truth.npz": "0"*64}},
                     sequence={"frames": [{"truth": "truth.npz", "timestamp_s": 0.}]}, gate={})
