"""Evidence-preservation and corruption tests on a retained MuJoCo trace.

These tests mutate a real published snapshot. They do not generate a benchmark
dataset, claim hardware calibration, or count mutations as new sensor results.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/bench/sensor_dataset_audit.py"
spec = importlib.util.spec_from_file_location("sensor_dataset_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
SOURCE = ROOT / "results/inspection_maze_probe.json"


@pytest.fixture
def observed_trace():
    record = json.loads(SOURCE.read_text())
    record["episodes"] = [deepcopy(record["episodes"][0])]
    record["episodes"][0]["trace"] = [deepcopy(record["episodes"][0]["trace"][0])]
    return record


def test_real_retained_arrays_do_not_establish_hardware_or_scan_replay():
    result = audit.audit_paths([SOURCE], repo_root=ROOT)
    entry = result["artifacts"][0]
    assert entry["input_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert entry["data_quality"]["verdict"] == "PASS"
    assert entry["source_classification"]["depth"] == "simulated_paired_ray_depth"
    assert entry["source_classification"]["lidar"] == "sector_minima_only"
    assert entry["source_classification"]["oracle_localization_declared"]
    assert entry["replay_readiness"]["verdict"] == "NOT_READY"
    assert entry["logged_snapshot_spacing_s"]["median"] == 1.0
    assert entry["declared_sensor_metadata"]["lidar"]["nominal_hz"] == 10.0
    # Missing packets contain zero policy features. Do not relabel these as
    # received metric depth/lidar frames or count them as a sensor measurement.
    assert entry["observed"]["missing_extero_snapshots"] > 0
    assert entry["observed"]["lidar_sector_snapshots"] < entry["observed"]["sensor_snapshots"]


@pytest.mark.parametrize("mutation,expected", [
    ("future", "future_extero_timestamp"),
    ("stale_marked_fresh", "freshness_flag_disagrees_with_logged_age"),
    ("missing_marked_fresh", "missing_packet_marked_fresh"),
    ("wrong_depth_shape", "invalid_paired_idealized_depth_m_shape_or_values"),
    ("infinite_lidar", "invalid_lidar_sector_m_shape_or_values"),
    ("physical_source_claim", "ray_depth_source_not_explicitly_declared"),
])
def test_corrupted_real_snapshot_is_refused(observed_trace, mutation, expected):
    packet = observed_trace["episodes"][0]["trace"][0]["sensors"]
    if mutation == "future":
        packet["extero_stamp_s"] = .01
    elif mutation == "stale_marked_fresh":
        packet["extero_stamp_s"] = -.3
        # A newly received packet must keep its old measurement timestamp.
        packet["receive_stamp_s"] = 0.
    elif mutation == "missing_marked_fresh":
        packet["extero_stamp_s"] = None
    elif mutation == "wrong_depth_shape":
        packet["paired_idealized_depth_m"] = packet["paired_idealized_depth_m"][0]
    elif mutation == "infinite_lidar":
        packet["lidar_sector_m"][0] = float("inf")
    elif mutation == "physical_source_claim":
        observed_trace["sensor_metadata"]["stereo"]["kind"] = "physical_rgb_stereo"
    entry = audit.audit_record(observed_trace)
    assert entry["data_quality"]["verdict"] == "FAIL"
    assert expected in {issue["code"] for issue in entry["data_quality"]["issues"]}
    assert entry["replay_readiness"]["verdict"] == "NOT_READY"


def test_maze_video_outcome_trace_cannot_be_recovered_as_sensor_data():
    path = ROOT / "results/maze-humanoid-20260928/humanoid-hard-6x6/seed12_sensors.json"
    entry = audit.audit_record(json.loads(path.read_text()))
    assert entry["artifact_kind"] == "maze_episode_summary"
    assert entry["observed"]["sensor_snapshots"] == 0
    assert entry["source_classification"]["depth"] == "no_retained_depth_frames"
    assert entry["replay_readiness"]["verdict"] == "NOT_READY"


def test_cli_audits_bytes_and_refuses_to_replace_a_prior_report(tmp_path):
    output = tmp_path / "audit.json"
    command = [sys.executable, str(SCRIPT), str(SOURCE), "--out", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    before = output.read_bytes()
    result = json.loads(before)
    assert result["schema"] == audit.SCHEMA
    assert result["summary"]["replay_ready"] == 0
    attempt = subprocess.run(command, capture_output=True, text=True)
    assert attempt.returncode != 0
    assert output.read_bytes() == before


def test_unknown_payload_cannot_be_assumed_to_be_sensor_data():
    with pytest.raises(ValueError, match="Unsupported artifact"):
        audit.audit_record({"success": True, "stereo": "calibrated"})


def test_cli_preserves_diagnostic_report_and_fails_for_corrupted_data(tmp_path, observed_trace):
    observed_trace["episodes"][0]["trace"][0]["sensors"]["extero_stamp_s"] = .01
    source = tmp_path / "corrupted-real-snapshot.json"
    source.write_text(json.dumps(observed_trace))
    output = tmp_path / "failed-audit.json"
    command = [sys.executable, str(SCRIPT), str(source), "--out", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    report = json.loads(output.read_text())
    assert report["summary"]["integrity_pass"] == 0
    entry = report["artifacts"][0]
    assert entry["data_quality"]["verdict"] == "FAIL"
    assert "future_extero_timestamp" in {issue["code"] for issue in entry["data_quality"]["issues"]}
    assert entry["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
