"""Waiter phase 4 gate scorer (scripts/waiter/dsrl/phase4_gate.py): the verdict from a synthetic scored evaluation,
the input checks (rule hash, seeds, executed steps, base model, contract env, training run, final checkpoint) and the
write-once rule. No simulator, no torch."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "waiter" / "dsrl"))
import phase4_gate as G  # noqa: E402

SEEDS = list(range(6000, 6100))


def _setup(tmp_path, decisions=1000):
    run = tmp_path / "dsrl1"
    (run / "ckpt").mkdir(parents=True)
    (run / "ckpt" / "sac.pt").write_bytes(b"final checkpoint")
    (run / "config.json").write_text(json.dumps({
        "labels": [], "exec_horizon": 20,
        "args": {"reward": "sparse", "shaping": "none", "fall_penalty": "0.0", "train_seeds": "5000-5999",
                 "n_envs": "16", "seed": "0"}}))
    (run / "summary.json").write_text(json.dumps({"counters": {"decisions": decisions}}))
    rule = {"rule": "test rule", "seed0": 6000, "seed1": 6099, "alpha": 0.05, "exec_horizon": 20, "noise_seed": 0,
            "base_model_sha256": "abc", "deploy": "results/x/waiter-wbc1c-s7.deploy.yaml", "run_dir": str(run),
            "train": {"seeds": "5000-5999", "max_decisions": 1000, "n_envs": 16, "seed": 0}}
    rp = tmp_path / "rule.json"
    rp.write_text(json.dumps(rule))
    return run, rp, hashlib.sha256(rp.read_bytes()).hexdigest()


def _eval(run, sha, base_ok, dsrl_ok, **over):
    def eps(ok):
        return [{"seed": s, "success": s in ok, "fell": False} for s in SEEDS]
    r = {"labels": [], "scored_phase4_eval": True, "env": "waiter", "gate": {"sha256": sha}, "seeds": SEEDS,
         "exec_horizon": 20, "noise_seed": 0, "policy_model_sha256": "abc", "run_dir": "/elsewhere/dsrl1",
         "env_kwargs": {"deploy": "/abs/waiter-wbc1c-s7.deploy.yaml", "upstream": "/u", "cache_dir": "/tmp/c",
                        "render": True},
         "sac_ckpt_sha256": hashlib.sha256((run / "ckpt" / "sac.pt").read_bytes()).hexdigest(),
         "episodes": {"base": eps(base_ok), "dsrl": eps(dsrl_ok), "zero": eps(set(SEEDS[:7]))}}
    r.update(over)
    return r


def _run(tmp_path, rule, res, run=None):
    (tmp_path / "scored.json").write_text(json.dumps(res))
    return G.main(["--rule", str(rule), "--eval", str(tmp_path / "scored.json"), "--out", str(tmp_path / "verdict.json")]
                  + (["--run-dir", str(run)] if run else []))


def test_mcnemar_matches_the_evaluation_code():
    assert G.mcnemar_one_sided(0, 0) == 1.0
    assert G.mcnemar_one_sided(5, 0) == pytest.approx(1 / 32)
    assert G.mcnemar_one_sided(8, 2) == pytest.approx(sum((45, 10, 1)) / 1024)
    sys.path.insert(0, str(REPO / "scripts" / "waiter" / "dsrl"))
    rollout = pytest.importorskip("rollout")                      # needs torch; the same number either way
    for b, c in ((8, 2), (27, 15), (3, 9), (0, 4)):
        assert G.mcnemar_one_sided(b, c) == pytest.approx(rollout.mcnemar_exact(b, c)[0])


def test_pass_fail_and_write_once(tmp_path):
    run, rule, sha = _setup(tmp_path)
    base = set(SEEDS[:30])
    dsrl = set(SEEDS[10:55])                                      # 45 vs 30: DSRL only 25, base only 10
    assert _run(tmp_path, rule, _eval(run, sha, base, dsrl)) == 0
    v = json.loads((tmp_path / "verdict.json").read_text())
    assert v["verdict"] == "PASS" and (v["base"]["successes"], v["dsrl"]["successes"]) == (30, 45)
    assert v["paired"] == {"both": 20, "dsrl_only": 25, "base_only": 10, "neither": 45,
                           "mcnemar_p_one_sided": pytest.approx(G.mcnemar_one_sided(25, 10)), "alpha": 0.05}
    assert v["reported_not_gated"] == {"zero": {"successes": 7, "n": 100}} and v["rule_sha256"] == sha
    with pytest.raises(SystemExit):                               # never overwritten
        _run(tmp_path, rule, _eval(run, sha, base, dsrl))
    (tmp_path / "verdict.json").unlink()
    # more successes but not significantly more: 34 vs 30 with 14 / 10 discordant
    assert _run(tmp_path, rule, _eval(run, sha, base, set(SEEDS[10:44]))) == 0
    v = json.loads((tmp_path / "verdict.json").read_text())
    assert v["verdict"] == "FAIL" and v["paired"]["mcnemar_p_one_sided"] > 0.05
    (tmp_path / "verdict.json").unlink()
    assert _run(tmp_path, rule, _eval(run, sha, dsrl, base)) == 0  # DSRL worse than the base
    assert json.loads((tmp_path / "verdict.json").read_text())["verdict"] == "FAIL"


@pytest.mark.parametrize("change", [
    {"scored_phase4_eval": False}, {"labels": ["TOY"]}, {"labels": ["TRAINED_WITH_SHAPING"]}, {"env": "stub"},
    {"gate": {"sha256": "0" * 64}}, {"seeds": SEEDS[:50]}, {"exec_horizon": 10}, {"noise_seed": 1},
    {"policy_model_sha256": "zzz"}, {"run_dir": "/elsewhere/dsrl2"}, {"sac_ckpt_sha256": "0" * 64},
    {"env_kwargs": {"deploy": "/abs/waiter-wbc1c-s7.deploy.yaml", "stance_lock": False}},
    {"env_kwargs": {"deploy": "/abs/waiter-wbc1c-s7.deploy.yaml", "time_limit_s": 120.0}},
    {"env_kwargs": {"deploy": "/abs/other.deploy.yaml"}},
])
def test_invalid_inputs_write_nothing(tmp_path, change):
    run, rule, sha = _setup(tmp_path)
    with pytest.raises(SystemExit, match="INVALID"):
        _run(tmp_path, rule, _eval(run, sha, set(SEEDS[:30]), set(SEEDS[10:55]), **change))
    assert not (tmp_path / "verdict.json").exists()


def test_episode_lists_and_training_run_are_checked(tmp_path):
    run, rule, sha = _setup(tmp_path)
    res = _eval(run, sha, set(SEEDS[:30]), set(SEEDS[10:55]))
    res["episodes"]["dsrl"] = res["episodes"]["dsrl"][:99]
    with pytest.raises(SystemExit, match="dsrl: 99 episodes"):
        _run(tmp_path, rule, res)
    good = _eval(run, sha, set(SEEDS[:30]), set(SEEDS[10:55]))
    cfg = json.loads((run / "config.json").read_text())
    for key, value, msg in (("shaping", "stage", "unshaped"), ("train_seeds", "1000-1899", "training seeds")):
        bad = json.loads(json.dumps(cfg))
        bad["args"][key] = value
        (run / "config.json").write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match=msg):
            _run(tmp_path, rule, good)
    (run / "config.json").write_text(json.dumps(cfg))
    ruled = json.loads(rule.read_text())                          # a rule that also pins design arguments
    ruled["train"]["args"] = {"reward": "sparse", "target_entropy": 0.0}
    pinned = tmp_path / "rule_pinned.json"
    pinned.write_text(json.dumps(ruled))
    with pytest.raises(SystemExit, match="training argument target_entropy = None, the rule says 0.0"):
        _run(tmp_path, pinned, dict(good, gate={"sha256": hashlib.sha256(pinned.read_bytes()).hexdigest()}))
    (run / "summary.json").write_text(json.dumps({"counters": {"decisions": 5000}}))
    with pytest.raises(SystemExit, match="over the 1000 budget"):
        _run(tmp_path, rule, good)
    (run / "summary.json").unlink()                               # a run that never reached its end
    with pytest.raises(SystemExit, match="summary.json missing"):
        _run(tmp_path, rule, good)
    (run / "summary.json").write_text(json.dumps({"counters": {"decisions": 1000}}))
    (run / "ckpt" / "sac.pt").write_bytes(b"a later checkpoint")  # not the file that was evaluated
    with pytest.raises(SystemExit, match="final checkpoint"):
        _run(tmp_path, rule, good)
    assert not (tmp_path / "verdict.json").exists()
