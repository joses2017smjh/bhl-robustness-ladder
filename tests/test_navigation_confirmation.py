"""Campaign verdict controls using explicit synthetic episode records.

These test the evidence gate and paired accounting, not robot performance.
No physics, trained model, GPU or shared campaign output is touched.
"""
from __future__ import annotations

from copy import deepcopy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("navigation_confirmation", ROOT / "scripts/bench/navigation_confirmation.py")
nav = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nav)


def save(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def campaign(tmp_path):
    """All controls present; each candidate actor is exactly at both thresholds."""
    c = tmp_path / "campaign"
    c.mkdir()
    cells = [{"name": f"{actor}-{mode}-{condition}", "actor": actor, "mode": mode,
              "condition": condition, "sensor_mode": "reactive" if condition == "nominal" else "reactive_dropout"}
             for actor in nav.ACTORS for mode in ("none", "wz-lpf") for condition in ("nominal", "drop35")]
    p = {"input_sha256": {}, "packages": {}, "source_gait_sha256": "synthetic-gait",
         "seeds": list(nav.SEEDS), "cells": cells, "candidate_rule": "the declared per-actor gates",
         "scope": "Synthetic evidence-gate fixture; no robot result", "statistical_scope": "shared layouts"}
    save(c / "protocol.json", p)
    for cell in cells:
        d = c / "runs" / cell["name"]
        d.mkdir(parents=True)
        threshold = 40 if cell["condition"] == "nominal" else 36
        for index, seed in enumerate(nav.SEEDS):
            success = index < threshold
            e = {"seed": seed, "gait": {"checkpoint_sha256": "synthetic-gait"},
                 "sensor_mode": cell["sensor_mode"],
                 "cmd_filter": {"mode": cell["mode"], "tau_s": .2 if cell["mode"] == "wz-lpf" else None,
                                "gait_wz": {"flips_per_s": 1.0}},
                 "policy": str(c / "models" / f"{cell['actor']}.onnx"),
                 "maze": {"n": 6, "m": 6, "extra_openings": 1, "layout_sha256": f"layout-{seed}"},
                 "policy_map_integration": {"mode": "capture_pose"}, "success": success,
                 "outcome": "goal" if success else "time_out", "clean_success": success,
                 "completion_s": 60.0 if success else None, "wall_contact_steps": 0}
            save(d / f"seed{seed}.json", e)
        save(d / "process.json", {"returncode": 0, "inputs_still_valid": True})
        save(d / "receipt.json", {"protocol_sha256": nav.digest(c / "protocol.json")})
    return c


def episode(c, actor="armV5-s8", mode="wz-lpf", condition="nominal", seed=120000):
    return c / "runs" / f"{actor}-{mode}-{condition}" / f"seed{seed}.json"


def edit(path, function):
    e = json.loads(path.read_text())
    function(e)
    save(path, e)


def score(c):
    with contextlib.redirect_stdout(io.StringIO()):
        rc = nav.summary(SimpleNamespace(campaign=c))
    return rc, json.loads((c / "confirmation_report.json").read_text())


def test_complete_threshold_campaign_passes_without_claiming_fall_reduction(campaign):
    rc, r = score(campaign)
    assert rc == 0 and r["status"] == "PASS"
    assert all(r["per_actor_pass"].values())
    assert r["fall_comparison"] == {"baseline_falls": 0, "candidate_falls": 0,
                                    "fewer_observed_candidate_falls": False}


def test_one_weak_actor_cannot_be_hidden_by_two_perfect_actors(campaign):
    edit(episode(campaign), lambda e: e.update(success=False, clean_success=False, outcome="time_out", completion_s=None))
    for actor in ("armV5-s9", "armV5-s10"):
        for path in (campaign / "runs" / f"{actor}-wz-lpf-nominal").glob("seed*.json"):
            edit(path, lambda e: e.update(success=True, clean_success=True, outcome="goal", completion_s=60.0))
    rc, r = score(campaign)
    assert rc == 0 and r["status"] == "NEGATIVE"
    assert r["per_actor_pass"]["armV5-s8"] is False
    assert r["per_actor_pass"]["armV5-s9"] is True


def test_a_single_candidate_fall_fails_even_when_all_goal_thresholds_pass(campaign):
    edit(episode(campaign, seed=120047), lambda e: e.update(outcome="fall"))
    rc, r = score(campaign)
    assert rc == 0 and r["status"] == "NEGATIVE"
    assert r["per_cell"]["armV5-s8-wz-lpf-nominal"]["goals"] == 40
    assert r["fall_comparison"]["candidate_falls"] == 1


@pytest.mark.parametrize("condition,seed", [("nominal", 120000), ("drop35", 120047)])
def test_missing_baseline_episode_blocks_pass_despite_complete_candidate(campaign, condition, seed):
    episode(campaign, mode="none", condition=condition, seed=seed).unlink()
    rc, r = score(campaign)
    assert rc == 1 and r["status"] == "INCOMPLETE"
    assert all(r["per_actor_pass"].values())
    assert r["fall_comparison"]["fewer_observed_candidate_falls"] is False


def test_swapped_maze_records_cannot_pass_with_unchanged_aggregate_counts(campaign):
    edit(episode(campaign), lambda e: e["maze"].update(layout_sha256="different-layout"))
    rc, r = score(campaign)
    assert rc == 1 and r["status"] == "INCOMPLETE"
    assert any("layout mismatch" in p for p in r["problems"])


@pytest.mark.parametrize("change", [
    lambda e: e.update(seed=119999),  # smoke leakage
    lambda e: e.update(sensor_mode="reactive"),  # dropout mislabeled
    lambda e: e["gait"].update(checkpoint_sha256="wrong-gait"),
    lambda e: e.update(policy="wrong-actor.onnx"),
    lambda e: e["cmd_filter"].update(tau_s=.3),  # post-declaration tuning
    lambda e: e.update(outcome="time_out"),  # success/outcome disagreement
    lambda e: e.update(wall_contact_steps=1),  # clean-success disagreement
    lambda e: e["policy_map_integration"].update(mode="loop_top_pose"),
])
def test_mislabeled_actual_input_or_outcome_evidence_blocks_pass(campaign, change):
    edit(episode(campaign, condition="drop35"), change)
    rc, r = score(campaign)
    assert rc == 1 and r["status"] == "INCOMPLETE"
    assert any("contract mismatch" in p for p in r["problems"])


@pytest.mark.parametrize("name,change", [
    ("process.json", lambda e: e.update(returncode=1)),
    ("process.json", lambda e: e.update(inputs_still_valid=False)),
    ("receipt.json", lambda e: e.update(protocol_sha256="different-protocol")),
])
def test_failed_process_or_unbound_receipt_blocks_pass(campaign, name, change):
    edit(campaign / "runs/armV5-s8-wz-lpf-nominal" / name, change)
    rc, r = score(campaign)
    assert rc == 1 and r["status"] == "INCOMPLETE"
    assert any("process/integrity mismatch" in p for p in r["problems"])


def test_paired_accounting_preserves_discordant_goals_falls_and_completion_times(campaign):
    # A recovered baseline fall, one candidate-only failure and a faster matched
    # successful completion must all remain visible, irrespective of the verdict.
    edit(episode(campaign, mode="none", seed=120000),
         lambda e: e.update(success=False, clean_success=False, outcome="fall", completion_s=None))
    edit(episode(campaign, seed=120001),
         lambda e: e.update(success=False, clean_success=False, outcome="fall", completion_s=None))
    edit(episode(campaign, seed=120002), lambda e: e.update(completion_s=57.0))
    _, r = score(campaign)
    p = r["paired"]["armV5-s8-nominal"]
    assert p["matched_layouts"] == 48 and p["both_goal"] == 38
    assert p["baseline_only_goal"] == 1 and p["candidate_only_goal"] == 1
    assert p["baseline_only_fall"] == 1 and p["candidate_only_fall"] == 1
    assert p["matched_success_time_delta_s"].count(-3.0) == 1
    assert len(p["matched_success_time_delta_s"]) == 38


def test_final_report_cannot_be_rewritten_after_it_is_recorded(campaign):
    _, first = score(campaign)
    with pytest.raises(FileExistsError):
        score(campaign)
    assert json.loads((campaign / "confirmation_report.json").read_text()) == first


verify_spec = importlib.util.spec_from_file_location("verify_navigation_confirmation", ROOT / "scripts/bench/verify_navigation_confirmation.py")
strict = importlib.util.module_from_spec(verify_spec)
verify_spec.loader.exec_module(strict)


@pytest.fixture
def strict_campaign(campaign):
    """Synthetic campaign shaped like a real capture, with fixture asset hashes."""
    p = json.loads((campaign / "protocol.json").read_text())
    p.update(schema="bhl-h1-confirmation-v1", max_episodes=576, max_concurrent_evaluators=2,
             tau_s=.2, time_limit_s=180., upstream="/fixture/upstream", source_actors={})
    models = campaign / "models"
    models.mkdir()
    (models / "gait.onnx").write_bytes(b"synthetic asset integrity fixture; never used for inference")
    p["source_gait_sha256"] = strict.digest(models / "gait.onnx")
    for actor in nav.ACTORS:
        f = models / f"{actor}.onnx"
        f.write_bytes(actor.encode() + b" synthetic fixture; not a trained model")
        p["source_actors"][actor] = {"sha256": strict.digest(f)}
    save(campaign / "protocol.json", p)
    source = ROOT / "results/task-closure-20261008/h1-navigation/runtime-smoke/armV5-s9-wz-lpf-nominal/seed119999.json"
    base = json.loads(source.read_text())
    for cell in p["cells"]:
        d = campaign / "runs" / cell["name"]
        for path in d.glob("seed*.json"):
            minimal = json.loads(path.read_text())
            e = deepcopy(base)
            e.update(minimal)
            e["gait"].update(checkpoint_sha256=p["source_gait_sha256"], checkpoint="models/gait.onnx",
                             deploy=str(campaign / "models/deploy.yaml"))
            e["maze"]["layout_sha256"] = f"{e['seed']:016x}"
            e["cmd_filter"] = deepcopy(base["cmd_filter"])
            e["cmd_filter"].update(mode=cell["mode"], tau_s=.2 if cell["mode"] == "wz-lpf" else None)
            e["policy_map_integration"] = deepcopy(base["policy_map_integration"])
            e["elapsed_s"] = 59.96 if e["success"] else 179.96
            e["goal_check"]["judge"]["reached"] = e["success"]
            e["goal_check"]["judge"]["final_true_dist_m"] = .25 if e["success"] else 2.0
            save(path, e)
        command = ["/fixture/bin/python", str(campaign / "snapshot/scripts/bench/maze_explore.py"),
                   "--upstream", p["upstream"], "--gait", str(models / "deploy.yaml"),
                   "--policy", str(models / f"{cell['actor']}.onnx"), "--policy-capture-pose",
                   "--policy-cmd-filter", cell["mode"], "--policy-cmd-tau", "0.2",
                   "--sensor-mode", cell["sensor_mode"], "--dropout-probability", ".35",
                   "--seeds", "48", "--seed-start", "120000", "--n", "6", "--m", "6",
                   "--extra-openings", "1", "--time-limit", "180.0", "--out-dir", str(d),
                   "--no-overwrite", "--cache-dir", "/tmp/bhl-nav-confirm-fixture"]
        save(d / "receipt.json", {"protocol_sha256": strict.digest(campaign / "protocol.json"),
                                  "cell": cell, "cwd": str(campaign), "scope": "confirmation", "command": command})
    return campaign


def strict_score(c):
    return strict.verify(c, strict.digest(c / "protocol.json"))


def test_independent_verifier_accepts_bound_complete_threshold_fixture(strict_campaign):
    r = strict_score(strict_campaign)
    assert r["status"] == "PASS", r["problems"]
    assert r["valid_episode_records"] == 576 and len(r["raw_episode_sha256"]) == 576


@pytest.mark.parametrize("change", [
    lambda e: e.update(success=1),
    lambda e: e.update(outcome="operator_skip"),
    lambda e: e.update(wall_seconds=float("nan")),
    lambda e: e.update(wall_contact_steps=-1),
    lambda e: e.update(completion_s=None),
    lambda e: e.update(completion_s=175.),
    lambda e: e.update(elapsed_s=181.),
    lambda e: e["cmd_filter"]["gait_wz"].update(saturated_share=1.5),
    lambda e: e["goal_check"]["judge"].update(reached=False),
    lambda e: e["goal_check"]["judge"].update(final_true_dist_m=2.0),
])
def test_independent_verifier_refuses_invalid_telemetry_in_high_scoring_campaign(strict_campaign, change):
    edit(episode(strict_campaign), change)
    r = strict_score(strict_campaign)
    assert r["status"] == "INCOMPLETE" and r["valid_episode_records"] < 576


@pytest.mark.parametrize("change", [
    lambda e: e.update(scope="UNSCORED runtime smoke"),
    lambda e: e["cell"].update(actor="armV5-s9"),
    lambda e: e["command"].__setitem__(e["command"].index("--seed-start") + 1, "119999"),
    lambda e: e["command"].__setitem__(e["command"].index("--dropout-probability") + 1, ".15"),
    lambda e: e["command"].__setitem__(e["command"].index("--time-limit") + 1, "300.0"),
])
def test_independent_verifier_rejects_receipt_scope_or_science_argv_changes(strict_campaign, change):
    edit(strict_campaign / "runs/armV5-s8-wz-lpf-nominal/receipt.json", change)
    assert strict_score(strict_campaign)["status"] == "INCOMPLETE"


def test_layout_pairing_also_spans_actors_and_sensing_conditions(strict_campaign):
    # Change both arms together; within-actor pairing alone would miss this.
    for mode in ("none", "wz-lpf"):
        edit(episode(strict_campaign, actor="armV5-s10", mode=mode),
             lambda e: e["maze"].update(layout_sha256="ffffffffffffffff"))
    r = strict_score(strict_campaign)
    assert r["status"] == "INCOMPLETE"
    assert any("across actors/arms/conditions" in p for p in r["problems"])


def test_actual_fixture_asset_corruption_and_wrong_trusted_protocol_hash_are_refused(strict_campaign):
    model = strict_campaign / "models/armV5-s9.onnx"
    model.write_bytes(model.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="wrong actor bytes"):
        strict_score(strict_campaign)
    with pytest.raises(ValueError, match="trusted protocol hash mismatch"):
        strict.verify(strict_campaign, "0" * 64)
