"""Held-out identity, geometry, evidence and independent outcome checks."""
import copy
import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from bhl_robust.research.terrain_confirmation import (QUALIFIED, SMOKE_SEED, arms_for_actor,
    expected_cells, geometry_parameters, make_world, summarize, validate_development_summary)

SPEC = importlib.util.spec_from_file_location("terrain_confirmation_campaign", Path(__file__).resolve().parents[1] / "scripts/bench/terrain_confirmation_campaign.py")
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def synthetic_rows(phase="confirmation"):
    return [dict(actor=a, terrain=t, seed=s, arm=arm, success=arm != "baseline", fell=False,
                 collision=False, nonfinite=False, native_tracked_frames=4, full_horizon=True)
            for a, t, s, arm in sorted(expected_cells(phase))]


def episode(terrain="flat", seed=440000):
    geometry = geometry_parameters(terrain, seed)
    return dict(schema="bhl-terrain-confirmation-episode-v2", ground_truth_inputs_to_controller=[],
        required_horizon_s=40., terrain=terrain, seed=seed, geometry_evaluator=geometry,
        geometry_sha256=geometry["geometry_sha256"], known_start_pose=np.eye(4).tolist(),
        elapsed_simulation_s=40., goal_reached_evaluator=True, collision=False, fell=False,
        full_horizon=True, success=True, nonfinite=False,
        trace=[dict(time_s=40., evaluator_xyz_m=[5., 0., .1], evaluator_tilt_rad=0.,
                    evaluator_goal_distance_m=0., evaluator_wall_contact=False)])


def test_reserved_identity_counts_and_excluded_failed_controller():
    assert len(expected_cells("confirmation")) == 480
    assert len(expected_cells("smoke")) == 24
    assert {s for _, _, s, _ in expected_cells("confirmation")} == set(range(440000, 440020))
    assert arms_for_actor("dr-default-s1") == ("baseline", "heading")
    assert not any(a == "dr-default-s1" and arm == "regulated" for a, _, _, arm in expected_cells("confirmation"))
    with pytest.raises(ValueError):
        arms_for_actor("unqualified")


@pytest.mark.parametrize("terrain", ("flat", "small_steps", "ramp"))
def test_new_geometry_is_seeded_and_different_from_development(tmp_path, terrain):
    first = tmp_path / "one"
    first.mkdir()
    world, params = make_world(terrain, 440000, first)
    assert params == geometry_parameters(terrain, 440000)
    assert params["onset_m"] >= .95  # development onset was .775--.825
    assert geometry_parameters(terrain, 440001) != params
    root = ET.fromstring(world)
    landmarks = [g for g in root.find("worldbody").findall("geom") if g.get("name", "").startswith("wall_confirmation_")]
    assert len(landmarks) == 6
    assert all(abs(float(g.get("pos").split()[1])) - .08 > .75 for g in landmarks)
    if terrain == "small_steps":
        steps = [g for g in root.find("worldbody").findall("geom") if g.get("name", "").startswith("terrain_step")]
        assert len(steps) == 6
        assert params["step_levels"] == [1, 2, 2, 3, 2, 1]
    if terrain == "ramp":
        assert (first / "course/ramp_height.png").stat().st_size > 0
        assert params["rise_length_m"] < 1.8  # development rise length was 1.8 m


def test_development_and_arbitrary_seeds_cannot_enter_confirmation():
    for seed in (400000, 410000, 420000, 430000, 440020, 450001):
        with pytest.raises(ValueError):
            geometry_parameters("flat", seed)


def test_qualification_requires_all_270_and_frozen_selection():
    receipt = dict(status="PASS", episodes=270, complete_declared_cells=270, protocol_sha256="hash",
        qualified_for_separately_preregistered_confirmation=[dict(actor=a, arm=arm) for a, arm in QUALIFIED])
    validate_development_summary(receipt, "hash")
    for field, value in (("episodes", 269), ("status", "NEGATIVE"), ("protocol_sha256", "other"),
                         ("qualified_for_separately_preregistered_confirmation", [])):
        with pytest.raises(ValueError):
            validate_development_summary(dict(receipt, **{field: value}), "hash")


def test_complete_confirmation_reports_shared_baseline_once():
    summary = summarize(synthetic_rows(), "confirmation")
    assert summary["episodes"] == 480
    assert summary["baseline_unique_episodes"] == 180
    assert summary["candidate_unique_episodes"] == 300
    assert summary["confirmed_combinations"] == 5
    assert all(r["paired_goal_rate_difference"] == 1 for r in summary["qualified_comparisons"])


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "wrong_seed", "unqualified"))
def test_partial_duplicate_or_consumed_cohort_cannot_claim_confirmation(mutation):
    rows = synthetic_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "wrong_seed":
        rows[0]["seed"] = 410000
    else:
        rows[0].update(actor="dr-default-s1", arm="regulated")
    with pytest.raises(ValueError):
        summarize(rows, "confirmation")


def test_contact_breaks_qualification_even_with_many_goals():
    rows = synthetic_rows()
    target = next(r for r in rows if r["actor"] == "dr-default-s0" and r["arm"] == "heading")
    target.update(success=False, collision=True)
    result = summarize(rows, "confirmation")
    first = result["qualified_comparisons"][0]
    assert first["candidate"]["clean_goals"] == 59
    assert not first["confirmed"]


def test_smoke_is_execution_gate_and_separate_from_confirmation():
    rows = synthetic_rows("smoke")
    assert summarize(rows, "smoke")["gate_to_confirmation"]
    rows[0]["native_tracked_frames"] = 0
    result = summarize(rows, "smoke")
    assert result["status"] == "NEGATIVE"
    assert not result["gate_to_confirmation"]


def test_new_scene_factory_is_restored_on_failure():
    original = campaign.development.make_world
    with pytest.raises(RuntimeError):
        with campaign.confirmation_world():
            assert campaign.development.make_world is make_world
            raise RuntimeError("deliberate")
    assert campaign.development.make_world is original


def test_success_independently_rescored_from_xyz():
    assert campaign.rescore_episode(episode())
    value = episode()
    value["trace"][0]["evaluator_xyz_m"] = [0., 0., .1]
    with pytest.raises(ValueError, match="distances"):
        campaign.rescore_episode(value)


@pytest.mark.parametrize("change", ("contact", "fall", "sink", "elapsed", "geometry", "claimed_failure", "future_trace"))
def test_optimistic_or_tampered_result_rejected(change):
    value = episode()
    if change == "contact":
        value["trace"][0]["evaluator_wall_contact"] = True
    elif change == "fall":
        value["trace"][0]["evaluator_tilt_rad"] = .8
    elif change == "sink":
        value["trace"][0]["evaluator_xyz_m"][2] = -.3
    elif change == "elapsed":
        value["elapsed_simulation_s"] = 39.
    elif change == "geometry":
        value["geometry_evaluator"]["onset_m"] = .8
    elif change == "claimed_failure":
        value["success"] = False
    else:
        value["trace"][0]["time_s"] = 41.
    with pytest.raises(ValueError):
        campaign.rescore_episode(value)


def test_actual_fall_is_retained_as_negative_not_execution_error():
    value = episode()
    value["trace"][0]["evaluator_tilt_rad"] = .8
    value.update(fell=True, success=False)
    assert campaign.rescore_episode(value) is False


def test_empty_trace_needs_explicit_nonfinite_failure():
    value = episode()
    value["trace"] = []
    with pytest.raises(ValueError):
        campaign.rescore_episode(value)
    value.update(nonfinite=True, success=False)
    assert campaign.rescore_episode(value) is False


def test_augmented_receipt_preserves_exclusive_base_episode(tmp_path, monkeypatch):
    import json
    def base_runner(*args):
        output = args[-2]
        output.mkdir()
        row = dict(episode(), files_sha256={})
        (output / "episode.json").write_text(json.dumps(row))
        (output / "rejected-scans.json").write_text(json.dumps({"rejected_count": 0}))
        return row
    monkeypatch.setattr(campaign.development, "run_episode", base_runner)
    output = tmp_path / "episode"
    value = campaign.run_episode({}, {}, None, "flat", 440000, "heading", output, 40.)
    assert (output / "episode.json").exists()
    assert (output / "confirmation-episode.json").exists()
    assert value["files_sha256"]["episode.json"] == campaign.sha256(output / "episode.json")
    assert "independent_confirmation" not in json.loads((output / "episode.json").read_text())
