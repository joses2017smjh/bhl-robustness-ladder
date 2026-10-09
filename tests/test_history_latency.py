"""Semantic controls for causal delay, matched capacity and H3 qualification."""
from __future__ import annotations

import importlib.util
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bhl_robust.latency_history import CausalImuHistory, expanded_teacher_state, actor_from_state
from bhl_robust.latency_history import file_sha256

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("h3_eval_test", ROOT / "scripts" / "bench" / "history_latency_eval.py")
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def queue(mode="history", n=1, delay=2):
    state = CausalImuHistory(n, noise=False, mode=mode, max_delay_steps=3)
    state.force_delay = delay
    state.reset()
    return state


def packet(value, n=1):
    return torch.full((n, 6), float(value))


@pytest.mark.parametrize("delay", [0, 1, 2, 3])
def test_delayed_packets_are_causal_and_exact(delay):
    state = queue(delay=delay)
    for tick in range(8):
        output = state.update(packet(tick), tick)[0]
        source = [max(0, tick - delay - old) for old in (3, 2, 1, 0)]
        assert state.source_ticks()[0].tolist() == source
        assert output[:, 0].tolist() == source


def test_feedforward_has_equal_capacity_but_only_current_delivered_information():
    state = queue("feedforward", delay=1)
    for tick in range(8):
        output = state.update(packet(tick), tick)
        assert output.shape == (1, 4, 6)
        assert output.unique().tolist() == [float(max(0, tick - 1))]
        assert state.source_ticks().unique().tolist() == [max(0, tick - 1)]


def test_multiple_observation_terms_do_not_advance_capture_or_resample_noise():
    state = CausalImuHistory(1, noise=True, generator=torch.Generator().manual_seed(12))
    state.force_delay = 0
    state.reset()
    first = state.update(packet(1), 0)
    assert torch.equal(first, state.update(packet(999), 0))
    assert state.source_ticks().tolist() == [[0, 0, 0, 0]]
    second = state.update(packet(2), 1)
    assert torch.equal(second[0, -2], first[0, -1])


def test_selective_reset_does_not_leak_old_episode_or_touch_other_environment():
    state = queue(n=2, delay=0)
    state.update(packet(1, 2), 0)
    before = state.update(packet(2, 2), 1)
    state.reset([0])
    after = state.update(packet(77, 2), 1)
    assert after[0].unique().tolist() == [77.]
    assert torch.equal(after[1], before[1])


@pytest.mark.parametrize("tick", [-1, 1.5, True])
def test_invalid_tick_rejected(tick):
    with pytest.raises(ValueError):
        queue().update(packet(1), tick)


def test_skipped_or_backward_tick_cannot_invent_packet_history():
    state = queue()
    state.update(packet(1), 0)
    with pytest.raises(ValueError, match="skipped"):
        state.update(packet(3), 2)
    state.update(packet(2), 1)
    with pytest.raises(ValueError, match="backwards"):
        state.update(packet(1), 0)


def test_invalid_shape_and_nonfinite_packet_rejected():
    state = queue()
    with pytest.raises(ValueError, match="shape"):
        state.update(torch.zeros(1, 5), 0)
    with pytest.raises(ValueError, match="finite"):
        state.update(packet(float("nan")), 0)


def test_noise_matches_teacher_imu_bounds_and_is_captured_once():
    state = CausalImuHistory(500, noise=True, generator=torch.Generator().manual_seed(19))
    state.force_delay = 0
    state.reset()
    result = state.update(packet(0, 500), 0)
    assert (result[..., :3].abs() <= .3).all()
    assert (result[..., 3:].abs() <= .05).all()
    assert result.unique().numel() > 10
    assert torch.equal(result[:, 0], result[:, -1])


def teacher_state():
    torch.manual_seed(99)
    actor = torch.nn.Sequential(torch.nn.Linear(45, 256), torch.nn.ELU(),
                                torch.nn.Linear(256, 128), torch.nn.ELU(),
                                torch.nn.Linear(128, 128), torch.nn.ELU(), torch.nn.Linear(128, 12))
    result = {"actor." + k: v for k, v in actor.state_dict().items()}
    result["critic.0.weight"] = torch.randn(256, 48)
    result["std"] = torch.ones(12)
    return result


def test_warmstart_keeps_every_teacher_tensor_and_zeroes_only_added_columns():
    teacher = teacher_state()
    expanded = expanded_teacher_state(teacher)
    assert expanded["actor.0.weight"].shape == (256, 63)
    assert torch.equal(expanded["actor.0.weight"][:, :45], teacher["actor.0.weight"])
    assert expanded["actor.0.weight"][:, 45:].count_nonzero() == 0
    for key in teacher:
        if key != "actor.0.weight":
            assert torch.equal(expanded[key], teacher[key])
            assert expanded[key].data_ptr() != teacher[key].data_ptr()
    examples = torch.randn(123, 63)
    with torch.no_grad():
        original_actions = actor_from_state(teacher, width=45)(examples[:, :45])
        history_actions = actor_from_state(expanded)(examples)
    torch.testing.assert_close(history_actions, original_actions, atol=1e-6, rtol=1e-6)


def test_wrong_teacher_shape_is_rejected():
    state = teacher_state()
    state["actor.0.weight"] = torch.zeros(256, 44)
    with pytest.raises(ValueError, match="45"):
        expanded_teacher_state(state)


def test_dynamic_batch_onnx_uses_declared_input_name(tmp_path):
    from bhl_robust.eval.history_latency import load_onnx_policy

    actor = actor_from_state(expanded_teacher_state(teacher_state()))
    example = torch.randn(1, 63)
    path = tmp_path / "policy.onnx"
    torch.onnx.export(actor, example, str(path), input_names=["custom_dynamic_input"],
                      output_names=["action"], dynamic_axes={"custom_dynamic_input": {0: "batch"},
                                                            "action": {0: "batch"}}, opset_version=17)
    actual = load_onnx_policy(path).forward(example.numpy())
    np.testing.assert_allclose(actual, actor(example).detach().numpy(), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("command", EVAL.COMMANDS)
def test_standing_still_is_not_qualified_under_any_active_command(command):
    assert not EVAL.qualification(command, [0, 0, 0], fell=False, completed_steps=250)["qualified"]


@pytest.mark.parametrize("command", EVAL.COMMANDS)
def test_exact_half_signed_command_integral_is_qualified(command):
    progress = [c * 9 * .5 for c in command]
    assert EVAL.qualification(command, progress, fell=False, completed_steps=250)["qualified"]


def test_arc_must_satisfy_translation_and_yaw_without_tradeoff():
    command = (.3, 0, .5)
    assert not EVAL.qualification(command, [100, 0, 0], fell=False, completed_steps=250)["qualified"]
    assert not EVAL.qualification(command, [0, 0, 100], fell=False, completed_steps=250)["qualified"]


def test_wrong_direction_fall_or_partial_rollout_is_never_qualified():
    assert not EVAL.qualification((-.2, 0, 0), [10, 0, 0], fell=False, completed_steps=250)["qualified"]
    assert not EVAL.qualification((.3, 0, 0), [10, 0, 0], fell=True, completed_steps=250)["qualified"]
    assert not EVAL.qualification((.3, 0, 0), [10, 0, 0], fell=False, completed_steps=249)["qualified"]


def valid_grid():
    rows = []
    for d in EVAL.DELAY_STEPS:
        for c, command in enumerate(EVAL.COMMANDS):
            for reset in EVAL.RESET_SEEDS:
                progress = [v * 9 for v in command]
                row = dict(arm="history", training_seed=0, delay_steps=d, delay_ms=d * 40,
                           command_index=c, command=list(command), reset_seed=reset, scored=True,
                           expected_steps=250, completed_steps=250, progress=progress,
                           fell=False, qualified=True, active_samples=225)
                rows.append(row)
    return dict(arm="history", training_seed=0, phase="run", policy_dt=.04, physics_dt=.002,
                episode_s=10., warmup_s=1., num_episodes=120, episodes=rows)


def test_exact_grid_passes_validation():
    assert EVAL.validate_evaluation(valid_grid(), arm="history", seed=0) == []


@pytest.mark.parametrize("mutation", ["duplicate", "omit", "unseen_seed", "delay_units", "false_gate", "partial", "nonfinite"])
def test_grid_validation_fails_on_scientific_record_corruption(mutation):
    data = valid_grid()
    row = data["episodes"][0]
    if mutation == "duplicate":
        data["episodes"][1] = row.copy()
    elif mutation == "omit":
        data["episodes"].pop()
    elif mutation == "unseen_seed":
        row["reset_seed"] = 12
    elif mutation == "delay_units":
        row["delay_ms"] = 20
    elif mutation == "false_gate":
        row["progress"] = [0, 0, 0]
    elif mutation == "partial":
        row["completed_steps"] = 249
    else:
        row["progress"][0] = float("nan")
    assert EVAL.validate_evaluation(data, arm="history", seed=0)


def test_paired_summary_requires_all_six_cells_and_no_smoke_substitution(tmp_path):
    result = EVAL.paired_summary([tmp_path])
    assert result["status"] == "INCOMPLETE"
    assert result["paired_seed_passes"] == 0


def make_summary_cells(tmp_path, monkeypatch):
    # Raw replay verification has dedicated controls below; this fixture
    # isolates the six-cell decision rule from physics and neural inference.
    monkeypatch.setattr(EVAL, "verify_trace", lambda directory, data: [])
    paths = []
    for arm in EVAL.ARMS:
        for seed in EVAL.TRAINING_SEEDS:
            directory = tmp_path / f"{arm}-{seed}"
            directory.mkdir()
            data = valid_grid()
            data.update(arm=arm, training_seed=seed)
            for row in data["episodes"]:
                row.update(arm=arm, training_seed=seed)
            # Exactly six qualified gains on primary80ms; survival is not
            # substituted for tracking qualification.
            if arm == "repeated_current":
                for row in [r for r in data["episodes"] if r["delay_steps"] == 2][:6]:
                    row.update(qualified=False, progress=[0, 0, 0])
            evaluation_path = directory / "latency-evaluation.json"
            evaluation_path.write_text(json.dumps(data))
            (directory / "final-checkpoint.pt").write_bytes(b"synthetic test fixture checkpoint")
            (directory / "policy.onnx").write_bytes(b"synthetic test fixture export")
            receipt = dict(arm=arm, seed=seed, phase="run", execution_status="COMPLETE",
                           completed_iterations=500, checkpoint_iteration=499,
                           protocol_sha256="one frozen protocol", teacher_sha256="one teacher",
                           evaluation_sha256=file_sha256(evaluation_path),
                           checkpoint_sha256=file_sha256(directory / "final-checkpoint.pt"),
                           export=dict(policy_sha256=file_sha256(directory / "policy.onnx")))
            (directory / "campaign_result.json").write_text(json.dumps(receipt))
            paths.append(directory)
    return paths


def modify_summary_cell(path, mutate):
    file = path / "latency-evaluation.json"
    data = json.loads(file.read_text())
    mutate(data)
    file.write_text(json.dumps(data))
    receipt_path = path / "campaign_result.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["evaluation_sha256"] = file_sha256(file)
    receipt_path.write_text(json.dumps(receipt))


def test_complete_paired_gate_requires_joint_checks_in_at_least_two_training_seeds(tmp_path, monkeypatch):
    paths = make_summary_cells(tmp_path, monkeypatch)
    result = EVAL.paired_summary(paths)
    assert result["status"] == "PASS"
    assert result["num_episodes"] == 720
    assert result["paired_seed_passes"] == 3
    assert all(g["primary_qualified_delta"] == 6 for g in result["seed_gates"])

    def lose_seven(data):
        for row in [r for r in data["episodes"] if r["delay_steps"] == 2][:7]:
            row.update(qualified=False, progress=[0, 0, 0])

    modify_summary_cell(paths[0], lose_seven)
    assert EVAL.paired_summary(paths)["status"] == "PASS"  # two remaining paired seeds
    modify_summary_cell(paths[1], lose_seven)
    result = EVAL.paired_summary(paths)
    assert result["status"] == "NEGATIVE"
    assert result["paired_seed_passes"] == 1


def test_nominal_survival_and_qualified_retention_are_both_required(tmp_path, monkeypatch):
    paths = make_summary_cells(tmp_path, monkeypatch)

    def lose_two_nominal_qualifications(data):
        for row in [r for r in data["episodes"] if r["delay_steps"] == 0][:2]:
            row.update(qualified=False, progress=[0, 0, 0])

    for path in paths[:2]:
        modify_summary_cell(path, lose_two_nominal_qualifications)
    result = EVAL.paired_summary(paths)
    assert result["status"] == "NEGATIVE"
    assert result["seed_gates"][0]["checks"]["nominal_survival"]
    assert not result["seed_gates"][0]["checks"]["nominal_qualified_retention"]


def test_changed_protocol_or_incomplete_training_is_incomplete_not_negative(tmp_path, monkeypatch):
    paths = make_summary_cells(tmp_path, monkeypatch)
    receipt_path = paths[0] / "campaign_result.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["protocol_sha256"] = "post-score changed protocol"
    receipt["completed_iterations"] = 499
    receipt_path.write_text(json.dumps(receipt))
    result = EVAL.paired_summary(paths)
    assert result["status"] == "INCOMPLETE"
    assert any("frozen protocol" in p for p in result["problems"])
    assert any("incomplete" in p for p in result["problems"])


def test_raw_trace_reconstructs_causal_actor_packets_and_progress(tmp_path):
    delay = 2
    trace_rows = []
    packets = []
    for tick in range(250):
        current = np.full(6, tick / 10, dtype=np.float32)
        packets.append(current)
        source = [max(0, tick - delay - old) for old in (3, 2, 1, 0)]
        history = np.array([packets[s] for s in source])
        features = np.zeros(63, dtype=np.float32)
        features[3:9] = history[-1]
        features[45:] = history[:-1].reshape(-1)
        trace_rows.append(dict(episode_id="test-episode", tick=tick, captured_imu=current.tolist(),
                               actor_obs=features.tolist(), source_ticks=source,
                               velocity=[.3, 0, 0], tilt_rad=0, sink_m=0, fell=False))
    episode = dict(episode_id="test-episode", delay_steps=delay, arm="history", command=[.3, 0, 0],
                   completed_steps=250, fell=False, progress=[2.7, 0, 0], tracking_rmse=[0, 0, 0])
    trace_path = tmp_path / "trace.jsonl.gz"

    def write_trace():
        with gzip.open(trace_path, "wt") as stream:
            for row in trace_rows:
                stream.write(json.dumps(row) + "\n")
        return dict(episodes=[episode], trace_path=trace_path.name, trace_sha256=file_sha256(trace_path))

    data = write_trace()
    assert EVAL.verify_trace(tmp_path, data) == []
    trace_rows[25]["actor_obs"][3] = 999  # hash refreshed; semantic future/wrong packet check must fail
    assert EVAL.verify_trace(tmp_path, write_trace())
    trace_rows[25]["actor_obs"][3] = 2.3
    data = write_trace()
    episode["progress"][0] = 999
    assert EVAL.verify_trace(tmp_path, data)
