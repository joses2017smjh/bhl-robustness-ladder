"""Stereo recovery campaign: protocol, paired clients and predeclared analysis (no native binaries)."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bhl_robust.research.stereo_relocalization import ResumeBound

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from test_stereo_relocalization import active_map_reset_same_id, frame, new_atlas_map, same_map_relocalization


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    previous = sys.path.copy()
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous
    return module


@pytest.fixture(scope="module")
def campaign():
    return load("stereo_recovery_campaign_test", "scripts/bench/stereo_recovery_campaign.py")


def test_protocol_declares_new_seeds_paired_cells_and_counterbalanced_order(campaign):
    p = campaign.declared_protocol()
    smoke, development = p["smoke"]["cells"], p["development"]["cells"]
    assert len(smoke) == 6 and len(development) == 48
    assert len({tuple(sorted(c.items())) for c in development}) == 48
    seeds = {c["seed"] for c in development}
    assert seeds == {760000, 760001, 760002, 760003} and {c["seed"] for c in smoke} == {759000}
    assert seeds.isdisjoint({749000, 750000, 750001, 750002, 751000, 752000})
    pairs = [development[i:i + 2] for i in range(0, 48, 2)]
    for a, b in pairs:  # each consecutive pair is one matched (seed, condition, route)
        assert {k: a[k] for k in ("case", "seed", "condition")} == {k: b[k] for k in ("case", "seed", "condition")}
        assert {a["arm"], b["arm"]} == set(campaign.ARMS)
    first = [a["arm"] for a, _ in pairs]
    assert first.count("permanent_stop") == first.count("resume_verified_relocalization") == 12
    for condition in campaign.CONDITIONS:
        for case in campaign.CASES:
            cell_first = [a["arm"] for a, _ in pairs if a["condition"] == condition and a["case"] == case]
            assert sorted(cell_first) == sorted(["permanent_stop", "resume_verified_relocalization"] * 2)
    assert p["smoke"]["seconds"] == 18. and p["development"]["seconds"] == 40.
    assert p["resume_rule"]["parameters"] == {k: getattr(ResumeBound(), k) for k in p["resume_rule"]["parameters"]}
    for field in ("hypotheses", "primary_outcome", "primary_analysis", "negative_result", "stopping_rules", "exclusion_rules"):
        assert p[field]
    assert json.loads(json.dumps(p)) == p  # exact frozen-protocol equality survives JSON


def mock_client(campaign, policy, frames, monkeypatch, *, ready=None):
    monkeypatch.setattr(campaign.STEREO, "native_camera_to_imu", lambda pose, _: np.asarray(pose))
    client = campaign.RecoveryNativeClient({"condition": "nominal", "true_t_imu_lidar": np.eye(4), "t_body_imu": np.eye(4)},
                                           np.eye(4), Path("unused"), max_frames=1000, policy=policy,
                                           known_start=np.eye(4), policy_dt=.04)
    remaining = iter(frames)
    client.orb = SimpleNamespace(track=lambda *args, **kwargs: next(remaining),
                                 ready=ready or {"atlas_telemetry": campaign.ATLAS_TELEMETRY})
    return client


def send(client, stamp):
    scan = {"frame_timestamp_s": stamp, "event_timestamp_s": stamp,
            "stereo": {"timestamp_s": stamp, "left": "unused.png", "right": "unused.png"},
            "additional_sensor_compute_seconds": .002}
    imu = {"timestamp_s": np.array([stamp]), "gyro_rad_s": np.zeros((1, 3)), "specific_force_m_s2": np.zeros((1, 3))}
    return client.track(scan, imu)


def run(client, frames):
    packets = []
    for f in frames:
        client.record_command(f["timestamp_s"] - .01, np.zeros(3))
        packets.append(send(client, f["timestamp_s"]))
    return dict(zip([f["timestamp_s"] for f in frames], packets))


def test_control_arm_reproduces_v1_permanent_stop_and_keeps_raw_poses(campaign, monkeypatch):
    frames = same_map_relocalization()
    packets = run(mock_client(campaign, "permanent_stop", frames, monkeypatch), frames)
    assert packets[14.0]["tracked"] and packets[14.0]["state"] == "OK"
    assert packets[14.3]["native_coordinate_invalidation_reasons"] == ["native_map_identity_unavailable"]
    for stamp in (14.5, 14.6, 14.7):
        assert not packets[stamp]["tracked"] and packets[stamp]["state"] == "MAP_CHANGED_STOP"
        assert packets[stamp]["map_reset_id"] == 1 and packets[stamp]["T_W_I"] is None
        assert packets[stamp]["raw_T_W_I"] == np.eye(4).tolist()  # raw pose retained although rejected
    # The resume gate is evaluated in shadow only.
    assert packets[14.6]["relocalization_gate"]["gate_reason"] == "VERIFIED_RELOCALIZATION_RESUME"
    assert packets[14.6]["recovery_policy"] == "permanent_stop"


def test_resume_arm_accepts_verified_same_map_relocalization(campaign, monkeypatch):
    frames = same_map_relocalization()
    packets = run(mock_client(campaign, "resume_verified_relocalization", frames, monkeypatch), frames)
    assert [packets[t]["state"] for t in (14.2, 14.3, 14.5)] == ["RECENTLY_LOST", "RECENTLY_LOST", "RELOCALIZATION_CONFIRMING_STOP"]
    assert not any(packets[t]["tracked"] for t in (14.2, 14.3, 14.5))
    assert packets[14.6]["tracked"] and packets[14.6]["map_reset_id"] == 0 and packets[14.6]["T_W_I"] == np.eye(4).tolist()
    assert packets[14.6]["legacy_permanent_stop_decision"]["state"] == "MAP_CHANGED_STOP"
    assert packets[14.7]["tracked"] and packets[14.7]["state"] == "OK"


@pytest.mark.parametrize("archetype,stop", [(active_map_reset_same_id, 14.4), (new_atlas_map, 25.0)])
def test_resume_arm_stops_on_reset_and_new_map(campaign, monkeypatch, archetype, stop):
    frames = archetype()
    packets = run(mock_client(campaign, "resume_verified_relocalization", frames, monkeypatch), frames)
    later = [p for t, p in packets.items() if t >= stop]
    assert all(not p["tracked"] and p["map_reset_id"] == 1 and p["state"] == "MAP_CHANGED_STOP" for p in later)
    assert packets[stop]["relocalization_gate"]["gate_reason"] == "COORDINATE_INVALIDATION"


def test_client_requires_atlas_runtime_and_monotone_command_log(campaign, monkeypatch):
    client = mock_client(campaign, "permanent_stop", [], monkeypatch, ready={"schema": "bhl-orb-native-ready-v1"})
    monkeypatch.setattr(campaign.BASE.MatchedNativeClient, "__enter__", lambda self: self)
    monkeypatch.setattr(campaign.BASE.MatchedNativeClient, "__exit__", lambda self, *a: False)
    monkeypatch.setattr(campaign.NAV, "write_json", lambda *a, **k: None)
    with pytest.raises(ValueError, match="Atlas telemetry"):
        client.__enter__()
    client.record_command(1., [0., 0., 0.])
    with pytest.raises(ValueError):
        client.record_command(1., [0., 0., 0.])
    with pytest.raises(ValueError):
        campaign.RecoveryNativeClient({"condition": "nominal", "t_body_imu": np.eye(4)}, np.eye(4), Path("x"), max_frames=1,
                                      policy="stereo", known_start=np.eye(4), policy_dt=.04)


def test_stopped_time_counts_zero_commands_after_navigation_start_only(campaign):
    trace = [{"decision_timestamp_s": t, "command": c, "reason": r} for t, c, r in [
        (9.96, [0, 0, 0], "predeclared_simulation_initialization_excitation"),
        (10.0, [0, 0, 0], "native_untracked_stop"), (10.04, [.3, 0, 0], "estimated_route"),
        (10.08, [0., 0., .1], "measured_native_latency_hold"), (10.12, [0, 0, 0], "measured_native_latency_hold"),
        (10.16, [0, 0, 0], "estimated_goal_stop")]]
    result = campaign.stopped_navigation(trace, .04)
    assert result["navigation_decisions"] == 5 and result["stopped_navigation_decisions"] == 2
    assert math.isclose(result["stopped_navigation_time_s"], .08)


def readiness_row(**changes):
    row = {"arm": "permanent_stop", "condition": "nominal", "native_tracked_frames": 10, "full_horizon": True,
           "atlas_telemetry_frames": 90, "native_camera_frames": 90, "fell": False, "collision": False, "nonfinite": False}
    row.update(changes)
    return row


@pytest.mark.parametrize("failure", [dict(native_tracked_frames=0), dict(full_horizon=False), dict(atlas_telemetry_frames=89),
                                     dict(fell=True), dict(collision=True), dict(nonfinite=True)])
def test_nominal_readiness_requires_pose_horizon_safety_and_telemetry(campaign, failure):
    rows = [readiness_row(), readiness_row(arm="resume_verified_relocalization", **failure)]
    assert campaign.nominal_arm_readiness(rows) == {"permanent_stop": True, "resume_verified_relocalization": False}


def test_smoke_cannot_be_reused_after_source_change_or_partial_cohort(campaign):
    smoke = {"schema": campaign.SCHEMA, "phase": "smoke", "status": "PASS", "execution_gate": True,
             "source_sha256": {"a": "b"}, "input_sha256": {"c": "d"}, "problems": [], "unrun_cells": [],
             "completed_cells": campaign.declared_protocol()["smoke"]["cells"], "episodes": 6,
             "arm_development_ready": {arm: True for arm in campaign.ARMS}}
    assert campaign.validate_smoke(smoke, sources={"a": "b"}, inputs={"c": "d"})
    for mutation in ({"source_sha256": {"a": "new"}}, {"episodes": 5}, {"problems": ["x"]},
                     {"completed_cells": smoke["completed_cells"][:-1]}):
        with pytest.raises(ValueError):
            campaign.validate_smoke({**smoke, **mutation}, sources={"a": "b"}, inputs={"c": "d"})


def analysis_rows(campaign, resume_goals=0, control_goals=0, resume_falls=0):
    rows = []
    cells = campaign.declared_protocol()["development"]["cells"]
    counts = {"resume_verified_relocalization": resume_goals, "permanent_stop": control_goals}
    falls = {"resume_verified_relocalization": resume_falls, "permanent_stop": 0}
    for cell in cells:
        arm = cell["arm"]
        success = counts[arm] > 0
        counts[arm] -= success
        fell = falls[arm] > 0 and not success
        falls[arm] -= fell
        rows.append(dict(cell, success=success, fell=fell, collision=False,
                         stopped_navigation_time_s=5. if arm == "resume_verified_relocalization" else 25.,
                         verified_resumes=int(arm == "resume_verified_relocalization"), resumed_pose_errors_m=[],
                         recovery={"tracking_loss_events": 1, "completed_navigation_recovery_s": [.4] if success else [],
                                   "right_censored_navigation_loss_s": [] if success else [20.],
                                   "native_relocalization_s": [.3],
                                   "loss_mechanisms": {"same_map_relocalization": 1, "active_map_reset": 0, "new_atlas_map": 0,
                                                       "other_invalidation": 0, "no_native_recovery": 0}}))
    return rows


def test_primary_exact_mcnemar_thresholds_and_verdicts(campaign):
    six = campaign.analyze(analysis_rows(campaign, resume_goals=6))
    assert six["primary"]["resume_only_goals_b"] == 6 and math.isclose(six["primary"]["p_value"], .03125)
    assert six["verdict"] == "POSITIVE"
    five = campaign.analyze(analysis_rows(campaign, resume_goals=5))
    assert math.isclose(five["primary"]["p_value"], .0625) and five["verdict"] == "NEGATIVE"
    assert campaign.analyze(analysis_rows(campaign))["verdict"] == "NEGATIVE"
    reverse = campaign.analyze(analysis_rows(campaign, control_goals=7))
    assert reverse["primary"]["control_only_goals_c"] == 7 and reverse["verdict"] == "NEGATIVE"
    unsafe = campaign.analyze(analysis_rows(campaign, resume_goals=8, resume_falls=1))
    assert unsafe["primary"]["supported"] and unsafe["verdict"] == "NEGATIVE_SAFETY"
    bootstrap = six["goal_difference"]
    assert bootstrap["resamples"] == 256 and len(bootstrap["seed_cluster_means"]) == 4
    assert math.isclose(bootstrap["estimate"], 6 / 24)
    assert six["stopped_time_difference_s"]["estimate"] == -20.
    assert six["arms"]["permanent_stop"]["clean_goal_wilson_95"][0] == 0.


def test_incomplete_cohort_has_no_verdict(campaign):
    rows = analysis_rows(campaign, resume_goals=10)[:-1]
    result = campaign.analyze(rows)
    assert result["status"] == "UNAVAILABLE" and result["verdict"] == "UNAVAILABLE_INCOMPLETE_COHORT"


def test_collector_rejects_missing_trials_instead_of_fabricating_zeros(campaign, tmp_path):
    smoke = {"schema": campaign.SCHEMA, "phase": "smoke", "status": "PASS", "execution_gate": True,
             "source_sha256": {"a": "b"}, "input_sha256": {"c": "d"}, "problems": [], "unrun_cells": [],
             "completed_cells": campaign.declared_protocol()["smoke"]["cells"], "episodes": 6,
             "arm_development_ready": {arm: True for arm in campaign.ARMS}}
    (tmp_path / "smoke.json").write_text(json.dumps(smoke))
    with pytest.raises(ValueError, match="incomplete development collection: 48"):
        campaign.collect_campaign(tmp_path, tmp_path / "smoke.json", tmp_path / "collection.json")


@pytest.fixture(scope="module")
def dispatch():
    return load("stereo_recovery_dispatch_test", "scripts/bench/stereo_recovery_dispatch.py")


def test_dispatch_inventory_matches_campaign_cells_and_rtx8000_resources(dispatch, campaign):
    jobs = dispatch.declared_jobs(dispatch.SHARED_ROOT / "unit-campaign")
    assert len(jobs) == 26 and jobs[0]["kind"] == "smoke" and jobs[-1]["kind"] == "evaluate"
    identities = set()
    for job in jobs[1:-1]:
        a = job["args"]
        identities.add(tuple(a[a.index(k) + 1] for k in ("--seed", "--condition", "--case")))
        assert job["resources"] == dispatch.RESOURCES and job["entrypoint"] == "scripts/bench/stereo_recovery_campaign.py"
    assert identities == {(str(c["seed"]), c["condition"], c["case"]) for c in campaign.declared_protocol()["development"]["cells"]}
    assert (dispatch.CONDITIONS, dispatch.CASES, dispatch.ARMS, dispatch.SEEDS) == (
        campaign.CONDITIONS, campaign.CASES, campaign.ARMS, campaign.DEVELOPMENT_SEEDS)
    assert dispatch.DECLARED_CELLS == 48 and dispatch.SMOKE_EPISODES == len(campaign.declared_protocol()["smoke"]["cells"])
    assert dispatch.RESOURCES["constraint"] == "rtx8000" and dispatch.RESOURCES["partition"] == "gpu"
    launcher = load("h34_for_recovery_test", "scripts/bench/h34_campaign.py")
    protocol = dict(schema_version=1, campaign_id="unit-campaign",
                    runtime={"python": "/x/python", "share_root": "/x", "sif": "/x/bhl.sif"}, jobs=jobs)
    assert launcher.validate_protocol(protocol) is protocol
    command = dispatch.submit_command(dispatch.SHARED_ROOT / "unit-campaign", "a" * 64)
    for flag in ("--constraint=rtx8000", "--gres=gpu:1", "--no-requeue", "--time=06:00:00", "--partition=gpu",
                 "--cpus-per-task=2", "--mem=16G", "--export=NONE", "--account=eecs"):
        assert flag in command
    text = dispatch.bootstrap_text()
    for name in ("stereo_recovery_dispatch.py", "processing_schedule_dispatch.py", "h34_campaign.py"):
        assert name in text


def test_dispatch_smoke_gate_and_orb_runtime_receipt(dispatch, tmp_path):
    good = dict(status="PASS", phase="smoke", episodes=6, execution_gate=True)
    dispatch.smoke_gate(good)
    for change in ({"episodes": 12}, {"status": "NEGATIVE"}, {"execution_gate": False}):
        with pytest.raises(ValueError):
            dispatch.smoke_gate({**good, **change})
    archive = tmp_path / "runtime.tar.gz"
    archive.write_bytes(b"not the runtime")
    receipt = tmp_path / "build-result.json"
    receipt.write_text(json.dumps({"status": "PASS", "runtime_archive_sha256": dispatch.ORB_SHA, "provenance": {}}))
    with pytest.raises(ValueError, match="Atlas-telemetry"):
        dispatch.verified_orb_runtime(archive, receipt)


def test_dispatch_retains_native_frames_and_recovery_receipt(dispatch, tmp_path):
    output, episode = tmp_path / "output", tmp_path / "output/ep"
    fields = {}
    for name, field in dispatch.EVIDENCE:
        path = episode / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        fields[field] = dispatch.digest(path)
    (episode / "sensors/00001-left.png").write_bytes(b"png")
    dispatch.write(episode / "episode.json", fields)
    result = dict(episode_files=["ep/episode.json"], episode_sha256={"ep/episode.json": dispatch.digest(episode / "episode.json")})
    dispatch.write(output / "campaign_result.json", result)
    inventory = dispatch.retain_collector_inputs(output, tmp_path / "compact", result)
    assert {"ep/native/native_frames.jsonl", "ep/native/recovery_receipt.json"} <= set(inventory)
    assert len(inventory) == 8 and not (tmp_path / "compact/ep/sensors/00001-left.png").exists()
