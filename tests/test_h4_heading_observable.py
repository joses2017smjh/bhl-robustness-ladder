"""Causal heading-latch parity, weight migration and immutable H4 gates."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from bhl_robust.eval.heading_observable import (
    FEATURE_CONTRACT, HEADING_KEY, HeadingLatch, expand_parent_state,
    heading_features, make_controller, torch_latched_features,
)

REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/bench" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("wz,active", [(0, True), (0.049, True), (0.05, False), (-0.05, False), (0.6, False)])
def test_mask_is_observable_and_strict(wz, active):
    actual = heading_features(0.4, 0.0, wz)
    expected = [math.sin(0.4), math.cos(0.4) - 1] if active else [0, 0]
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_latch_retains_same_command_and_resets_on_change():
    latch = HeadingLatch()
    np.testing.assert_array_equal(latch.update(1.0, [0, 0, 0]), [0, 0])
    np.testing.assert_allclose(latch.update(1.4, [0, 0, 0]), [math.sin(.4), math.cos(.4) - 1], atol=1e-7)
    np.testing.assert_array_equal(latch.update(1.4, [.35, 0, 0]), [0, 0])
    assert latch.reference == 1.4
    latch.update(1.7, [.35, 0, 0])
    latch.reset()
    np.testing.assert_array_equal(latch.update(-2.0, [.35, 0, 0]), [0, 0])


def test_command_change_uses_exact_float32_components():
    latch = HeadingLatch()
    latch.update(.0, [.35, 0, 0])
    same = float(np.float32(.35)) + 1e-10
    assert latch.update(.4, [same, 0, 0])[0] != 0
    changed = np.nextafter(np.float32(.35), np.float32(1))
    np.testing.assert_array_equal(latch.update(.5, [changed, 0, 0]), [0, 0])


def test_torch_deploy_parity_for_fixed_command_yaw_trace():
    latch = HeadingLatch()
    reference = torch.zeros(1)
    previous = torch.zeros(1, 3)
    ready = torch.zeros(1, dtype=torch.bool)
    trace = [(-3.10, [0, 0, 0], True), (3.10, [0, 0, 0], False),
             (2.9, [.35, 0, 0], False), (2.7, [.35, 0, 0], False),
             (2.5, [0, 0, .6], False), (2.3, [0, 0, 0], False),
             (2.0, [0, 0, 0], False), (-3.0, [0, 0, 0], True)]
    for yaw, command, reset in trace:
        if reset:
            latch.reset()
        expected = latch.update(float(np.float32(yaw)), command)
        actual, reference, previous, ready = torch_latched_features(
            torch.tensor([yaw]), torch.tensor([command]), reference, previous, ready, torch.tensor([reset]))
        np.testing.assert_allclose(actual[0].numpy(), expected, atol=5e-7)


def test_periodic_wrap_has_no_pi_discontinuity():
    a = heading_features(-math.pi + .01, math.pi - .01, 0)
    b = heading_features(.02, 0, 0)
    np.testing.assert_allclose(a, b, atol=1e-7)


@pytest.mark.parametrize("change", ["missing_key", "actor_width", "critic_width", "action_width"])
def test_parent_migration_rejects_wrong_shapes(change):
    source = {"actor.0.weight": torch.ones(256, 77), "critic.0.weight": torch.ones(256, 80),
              "actor.6.weight": torch.ones(22, 128)}
    target = {"actor.0.weight": torch.zeros(256, 79), "critic.0.weight": torch.zeros(256, 82),
              "actor.6.weight": torch.zeros(22, 128)}
    if change == "missing_key":
        del source["critic.0.weight"]
    elif change == "actor_width":
        source["actor.0.weight"] = torch.zeros(256, 75)
    elif change == "critic_width":
        target["critic.0.weight"] = torch.zeros(256, 83)
    else:
        target["actor.6.weight"] = torch.zeros(24, 128)
    with pytest.raises(ValueError):
        expand_parent_state(source, target)


def test_zero_columns_preserve_actor_for_nonzero_new_features():
    torch.manual_seed(100)
    original = torch.nn.Sequential(torch.nn.Linear(77, 256), torch.nn.ELU(), torch.nn.Linear(256, 22))
    wider = copy.deepcopy(original)
    wider[0] = torch.nn.Linear(79, 256)
    parent = {"actor." + k: v for k, v in original.state_dict().items()}
    parent["critic.0.weight"] = torch.randn(256, 80)
    target = {"actor." + k: v for k, v in wider.state_dict().items()}
    target["critic.0.weight"] = torch.empty(256, 82)
    state = expand_parent_state(parent, target)
    wider.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("actor.")})
    obs = torch.randn(256, 79)
    torch.testing.assert_close(original(obs[:, :77]), wider(obs), atol=2e-6, rtol=1e-5)
    assert torch.count_nonzero(state["actor.0.weight"][:, -2:]) == 0
    assert torch.count_nonzero(state["critic.0.weight"][:, -2:]) == 0


def controller_config():
    cfg = OmegaConf.load(REPO / "tests/fixtures/arms-dr1.0-s0/deploy.yaml")
    cfg.num_observations = 79
    cfg.gait_clock = {"period_s": .8, "phase_offset": 0.0}
    cfg[HEADING_KEY] = FEATURE_CONTRACT
    return cfg


def test_actual_controller_observation_order_and_reset():
    cfg = controller_config()
    controller = make_controller(cfg)
    class Recorder:
        def forward(self, obs):
            self.obs = obs.copy()
            return np.zeros((1, 22), dtype=np.float32)
    controller.policy = Recorder()
    def packet(yaw, cmd):
        raw = np.zeros(55, dtype=np.float32)
        raw[:4] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        raw[7:29] = cfg.default_joint_positions
        raw[52:55] = cmd
        return raw
    controller.update(packet(.0, [.35, 0, 0]))
    controller.update(packet(.4, [.35, 0, 0]))
    assert controller.policy.obs.shape == (1, 79)
    np.testing.assert_allclose(controller.policy.obs[0, :3], [.35, 0, 0])
    np.testing.assert_allclose(controller.policy.obs[0, 77:], [math.sin(.4), math.cos(.4)-1], atol=1e-7)
    controller.policy_observations[:] = 0
    controller.update(packet(-2, [.35, 0, 0]))
    np.testing.assert_array_equal(controller.policy.obs[0, 77:], [0, 0])
    np.testing.assert_allclose(controller.policy.obs[0, 75:77], [0, 1], atol=1e-7)


@pytest.mark.parametrize("key,value", [("num_joints", 24), ("num_actions", 12), ("num_observations", 77)])
def test_controller_rejects_wrong_robot(key, value):
    cfg = controller_config()
    cfg[key] = value
    with pytest.raises(ValueError):
        make_controller(cfg)


def test_frozen_protocol_and_gate_guards():
    runner = load_script("h4_train")
    protocol = json.loads((REPO / "results/h34-campaign-20261008/h4/protocol.json").read_text())
    runner.check_protocol(protocol, protocol["cells"][0], "run")
    for key, value in (("actor_observations", 77), ("reference_command_tolerance", .01), ("joints", 24)):
        changed = copy.deepcopy(protocol)
        changed["h4"][key] = value
        with pytest.raises(ValueError):
            runner.check_protocol(changed, changed["cells"][0], "run")
    changed = copy.deepcopy(protocol)
    changed["h4"]["gate"]["qualification_turn_min"] = 8
    with pytest.raises(ValueError):
        runner.check_protocol(changed, changed["cells"][0], "run")


def synthetic_gate_records():
    def trial(seed, sign):
        return {"seed": seed, "cmd": [0., 0., .6 * sign], "warm_s": 3., "seconds": 6.,
                "fell_at_s": None, "yaw_deg": 160. * sign}
    def walk(seed):
        return {"seed": seed, "cmd": [.35, 0., 0.], "warm_s": 1., "seconds": 6.,
                "fell_at_s": None, "yaw_deg": 15.}
    v2 = {"protocol": "v2", "turns": [trial(s, a) for s in range(3) for a in (1, -1)], "walk": walk(0)}
    rule = {"turn_min_deg": 150., "drift_max_deg": 15., "seconds": 6., "turn_seeds": [10,11,12,13,14],
            "walk_seeds": [10,11,12], "min_turn_ok": 9, "min_walk_ok": 2, "turn_wz": .6,
            "turn_warm_s": 3., "walk_cmd": [.35,0.,0.], "walk_warm_s": 1.}
    v2x = {"protocol": "v2x", "rule": rule, "turns": [trial(s,a) for s in range(10,15) for a in (1,-1)],
           "walks": [walk(s) for s in (10,11,12)]}
    from bhl_robust.eval.harness import EvalConfig
    rows = [{"command_vx": c[0], "command_vy": c[1], "command_wz": c[2], "seed": s, "fell": False}
            for c in EvalConfig().commands for s in range(10)]
    return v2, v2x, rows


def test_unchanged_joint_rule_boundaries_and_missing_data():
    qualifier = load_script("h4_qualification")
    v2, v2x, rows = synthetic_gate_records()
    v2x["turns"][0]["yaw_deg"] = 149.9  # exactly one allowed miss
    v2x["walks"][0]["yaw_deg"] = 15.1  # exactly one allowed miss
    for row in rows[:9]:
        row["fell"] = True
    assert qualifier.judge_records(v2, v2x, rows)["joint_pass"]
    rows[9]["fell"] = True
    assert not qualifier.judge_records(v2, v2x, rows)["joint_pass"]
    with pytest.raises(ValueError):
        qualifier.judge_records(v2, v2x, rows[:-1])
    changed = copy.deepcopy(rows)
    changed[-1] = changed[0]
    with pytest.raises(ValueError):
        qualifier.judge_records(v2, v2x, changed)
    rows[9]["fell"] = False
    v2["walk"]["yaw_deg"] = 15.1
    assert not qualifier.judge_records(v2, v2x, rows)["joint_pass"]
