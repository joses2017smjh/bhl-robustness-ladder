"""Native evidence collection must preserve negatives and reject hidden losses."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("native_slam_collect_test", Path(__file__).parents[1] / "scripts/bench/native_slam_collect.py")
COLLECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECT)


def negative_run():
    run = {"method": "lio", "sequence": "scene", "split": "test", "frames": 2, "tracked_frames": 0,
           "measurement": {"native_compute_p95_ms": 10., "segments": [{"metrics": None}], "replay_readiness_gate": "NEGATIVE"}}
    rows = [{"timestamp_s": i / 5., "tracked": False, "T_W_I": None, "compute_seconds": .01} for i in range(2)]
    return run, rows, {"ground_truth_inputs": []}


def test_all_lost_negative_remains_valid_with_unscorable_metrics():
    result = COLLECT.audit_run(*negative_run())
    assert result["tracked_frames"] == 0
    assert result["measurement"]["segments"][0]["metrics"] is None
    assert result["measurement"]["replay_readiness_gate"] == "NEGATIVE"


@pytest.mark.parametrize("fault", ["hidden_identity", "fake_count", "fake_timing", "truth_input"])
def test_scientific_evidence_mismatch_is_rejected(fault):
    run, rows, receipt = negative_run()
    if fault == "hidden_identity":
        rows[0]["T_W_I"] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    elif fault == "fake_count":
        run["tracked_frames"] = 1
    elif fault == "fake_timing":
        run["measurement"]["native_compute_p95_ms"] = 1.
    else:
        receipt["ground_truth_inputs"] = ["truth.npz"]
    with pytest.raises(ValueError):
        COLLECT.audit_run(run, rows, receipt)
