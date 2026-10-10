"""Waiter phase 4 (DSRL) plumbing tests: noise map, TOY stub env/policy, SAC, replay, vec env, stats, train smoke.

Pure torch/numpy (no lerobot, no MuJoCo). The real-SmolVLA wrapper check is a GPU script:
``scripts/waiter/dsrl/smolvla_noise.py --check`` (see phase4/PROGRESS.md for the job record).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
_DSRL = _REPO / "scripts" / "waiter" / "dsrl"
sys.path.insert(0, str(_DSRL))

from chunk_vec_env import ChunkVecEnv  # noqa: E402
from env_worker import dec, enc  # noqa: E402
from rollout import (check_seeds, mcnemar_exact, paired_summary, parse_seeds, run_episodes)  # noqa: E402
from sac_noise import Actor, NoiseSAC, ReplayBuffer, SACConfig, random_shift  # noqa: E402
from smolvla_noise import NoiseMap  # noqa: E402
from stub_env import StubWaiterEnv  # noqa: E402
from stub_policy import StubFlowPolicy  # noqa: E402

INFO_KEYS = {"success", "fell", "stage", "tumbler_on_tray", "tray_on_target", "t"}


# ------------------------------------------------------------------------------------------ noise map
def test_noise_map_repeat_rows_equal():
    nm = NoiseMap(50, 32, "repeat")
    assert nm.latent_dim == 32
    w = torch.randn(3, 32)
    z = nm(w)
    assert z.shape == (3, 50, 32)
    assert torch.allclose(z, w[:, None, :].expand(3, 50, 32))


def test_noise_map_dct_k1_is_repeat_and_unit_variance():
    w = torch.randn(4, 32)
    assert torch.allclose(NoiseMap(50, 32, "dct", 1)(w), NoiseMap(50, 32, "repeat")(w))
    nm = NoiseMap(50, 32, "dct", 4)
    assert nm.latent_dim == 128
    z = nm(torch.randn(4000, nm.latent_dim))
    assert abs(float(z.var(0).mean()) - 1.0) < 0.05     # average marginal variance ~1


def test_noise_map_full_and_active_dims():
    nm = NoiseMap(50, 32, "full")
    w = torch.randn(2, 50 * 32)
    assert torch.equal(nm(w), w.reshape(2, 50, 32))
    na = NoiseMap(50, 32, "repeat", active_dims=15)
    z = na(torch.randn(2, 15))
    assert na.latent_dim == 15 and torch.all(z[:, :, 15:] == 0)


# ------------------------------------------------------------------------------------------ stub env
def test_stub_env_contract_shapes_and_purity():
    env = StubWaiterEnv()
    o1 = env.reset(1234)
    assert o1["image"].shape == (224, 224, 3) and o1["image"].dtype == np.uint8
    assert o1["state"].shape == (30,) and o1["state"].dtype == np.float32
    assert isinstance(o1["instruction"], str)
    o2 = StubWaiterEnv().reset(1234)
    assert np.array_equal(o1["image"], o2["image"]) and o1["instruction"] == o2["instruction"]
    obs, r, done, info = env.step(np.zeros(15, np.float32))
    assert INFO_KEYS <= set(info) and r in (0.0, 1.0) and isinstance(done, bool)
    assert abs(info["t"] - 0.08) < 1e-9


def test_stub_env_oracle_success_reward_once_and_time_limit():
    env = StubWaiterEnv(time_limit_s=4.0)
    env.reset(1001)
    goal = env.oracle()["goal"]
    a = np.zeros(15, np.float32)
    a[3:5] = goal
    rewards = []
    for _ in range(50):
        _, r, done, info = env.step(a)
        rewards.append(r)
        if done:
            break
    assert info["success"] and rewards[-1] == 1.0 and sum(rewards) == 1.0
    env.reset(1002)
    steps = 0
    done = False
    while not done:
        _, r, done, info = env.step(np.zeros(15, np.float32))   # parks between... maybe not on goal
        steps += 1
    assert steps <= 50 and (info["success"] or steps == 50)


def test_stub_env_fall_is_terminal():
    env = StubWaiterEnv()
    env.reset(1003)
    a = np.zeros(15, np.float32)
    a[3:5] = (0.95, 0.0)
    _, r, done, info = env.step(a)
    assert done and info["fell"] and r == 0.0


# ------------------------------------------------------------------------------------------ stub policy
def test_stub_policy_deterministic_in_noise():
    pol, env = StubFlowPolicy(), StubWaiterEnv()
    obs = [env.reset(1000 + i) for i in range(3)]
    n1, n2 = torch.randn(3, 50, 32), torch.randn(3, 50, 32)
    assert pol.noise_shape == (50, 32)
    c1, c1b, c2 = pol(obs, n1), pol(obs, n1), pol(obs, n2)
    assert c1.shape == (3, 50, 15)
    assert torch.equal(c1, c1b)
    assert (c1 - c2).abs().max() > 1e-2


def _episode(env, pol, seed, noise_fn, H=10):
    obs = env.reset(seed)
    k = 0
    while True:
        chunk = pol([obs], noise_fn(k))[0, :H].numpy()
        for a in chunk:
            obs, r, done, info = env.step(a)
            if done:
                return info["success"]
        k += 1


def test_toy_base_rate_nontrivial_and_a_noise_direction_succeeds():
    pol, env = StubFlowPolicy(), StubWaiterEnv()
    g = torch.Generator().manual_seed(0)
    seeds = list(range(1000, 1040))
    base = [_episode(env, pol, s, lambda k: torch.randn(1, 50, 32, generator=g)) for s in seeds]
    assert 0.05 <= np.mean(base) <= 0.8, np.mean(base)
    # "a specific noise direction yields success": invert the frozen noise->offset map per layout
    wins = []
    for s in seeds[:10]:
        env.reset(s)
        goal = torch.as_tensor(env.oracle()["goal"], dtype=torch.float32)
        mid = pol.encode([env._obs()]).mid[0]
        z = torch.zeros(32, requires_grad=True)
        opt = torch.optim.Adam([z], lr=0.05)
        for _ in range(400):
            loss = ((mid + pol.offset(z[None])[0] - goal) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        zz = z.detach().clamp(-1.5, 1.5)
        wins.append(_episode(env, pol, s, lambda k: zz[None, None, :].expand(1, 50, 32).clone()))
    assert np.mean(wins) >= 0.9, wins


# ------------------------------------------------------------------------------------------ SAC
def test_tanh_logprob_matches_torch_distributions():
    cfg = SACConfig(vec_dim=5, act_dim=3, hidden=16, layers=2)
    actor = Actor(cfg)
    obs = {"vec": torch.randn(7, 5)}
    torch.manual_seed(1)
    u, logp = actor(obs)
    mu, log_std = actor.net(actor.enc(obs)).chunk(2, -1)
    log_std = log_std.clamp(-5, 2)
    dist = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(mu, log_std.exp()), [torch.distributions.TanhTransform(cache_size=1)])
    ref = dist.log_prob(u.clamp(-1 + 1e-6, 1 - 1e-6)).sum(-1)
    assert torch.allclose(logp, ref, atol=2e-3), (logp, ref)
    assert torch.all(u.abs() <= 1)


def test_replay_save_load_roundtrip(tmp_path):
    buf = ReplayBuffer(10, 4, 2, img_hw=8)
    for i in range(13):
        o = {"vec": np.full(4, i, np.float32), "img": np.full((8, 8, 3), i, np.uint8)}
        buf.add(o, np.array([i, -i], np.float32), float(i), 0.99, i % 2 == 0, o)
    assert buf.size == 10 and buf.ptr == 3
    buf.save(tmp_path / "b.npz")
    b2 = ReplayBuffer(10, 4, 2, img_hw=8)
    b2.load(tmp_path / "b.npz")
    for k in buf.data:
        assert np.array_equal(buf.data[k], b2.data[k])
    obs, act, rew, disc, term, nobs = b2.sample(5, "cpu")
    assert obs["vec"].shape == (5, 4) and obs["img"].dtype == torch.uint8 and act.shape == (5, 2)


def test_random_shift_shape():
    x = torch.randint(0, 255, (4, 56, 56, 3), dtype=torch.uint8)
    y = random_shift(x)
    assert y.shape == x.shape and y.dtype == torch.uint8


def test_sac_learns_contextual_bandit_and_checkpoint_roundtrip(tmp_path):
    """One-step episodes: reward = 1 - ||u - c(obs)||^2 with c(obs) = 0.6*tanh(obs[:2]). SAC must condition on obs."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = SACConfig(vec_dim=4, act_dim=2, hidden=64, layers=2, batch_size=128, target_entropy=-2.0,
                    actor_lr=1e-3, critic_lr=1e-3)
    sac = NoiseSAC(cfg, seed=0)
    buf = ReplayBuffer(5000, 4, 2)

    def target(v):
        return 0.6 * np.tanh(v[:, :2])

    for it in range(1500):
        v = rng.standard_normal((4, 4)).astype(np.float32)
        u = sac.act({"vec": v}) if it > 50 else rng.uniform(-1, 1, (4, 2)).astype(np.float32)
        rew = 1.0 - ((u - target(v)) ** 2).sum(1)
        for j in range(4):
            buf.add({"vec": v[j]}, u[j], rew[j], 0.99, True, {"vec": np.zeros(4, np.float32)})
        if it > 50:
            sac.update(buf)
    v = rng.standard_normal((256, 4)).astype(np.float32)
    err = float(((sac.act({"vec": v}, deterministic=True) - target(v)) ** 2).sum(1).mean())
    err_zero = float((target(v) ** 2).sum(1).mean())          # the best obs-independent answer (u = 0)
    assert err < 0.5 * err_zero, (err, err_zero)   # job 21589524: 0.097 vs 0.276
    sac.save(tmp_path / "s.pt")
    s2 = NoiseSAC.load(tmp_path / "s.pt")
    assert np.allclose(sac.act({"vec": v}, True), s2.act({"vec": v}, True))


# ------------------------------------------------------------------------------------------ vec env / wire
def test_wire_roundtrip():
    x = {"image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3), "state": np.ones(3, np.float32),
         "b": np.bool_(True), "f": np.float32(1.5), "l": [np.int64(3)], "s": "hi"}
    y = dec(enc(x))
    assert np.array_equal(y["image"], x["image"]) and y["image"].dtype == np.uint8
    assert y["b"] is True and y["f"] == 1.5 and y["l"] == [3] and y["s"] == "hi"


def test_vec_env_subprocess_matches_inprocess(tmp_path):
    spec = {"kind": "stub", "kwargs": {"time_limit_s": 2.0}}
    a = ChunkVecEnv(spec, 2)
    b = ChunkVecEnv(spec, 2, python=sys.executable, log_dir=str(tmp_path))
    try:
        oa = a.reset({0: (1010, None), 1: (1011, None)})
        ob = b.reset({0: (1010, None), 1: (1011, None)})
        assert np.array_equal(oa[1]["image"], ob[1]["image"])
        rng = np.random.default_rng(0)
        for _ in range(3):
            ch = {i: rng.uniform(-0.3, 0.3, (10, 15)).astype(np.float32) for i in (0, 1)}
            a.step_async(ch)
            b.step_async(ch)
            ra, rb = a.step_wait(), b.step_wait()
            for i in (0, 1):
                assert ra[i]["n"] == rb[i]["n"] and ra[i]["done"] == rb[i]["done"]
                assert np.allclose(ra[i]["obs"]["state"], rb[i]["obs"]["state"])
    finally:
        b.close()


# ------------------------------------------------------------------------------------------ seeds / stats
def test_seed_parsing_and_guards():
    assert parse_seeds("1000-1003,1010") == [1000, 1001, 1002, 1003, 1010]
    check_seeds([1000, 1999], {"dev"}, "x")
    for bad in ([6000], [5000], [4000], [999]):
        with pytest.raises(SystemExit):
            check_seeds(bad, {"dev"}, "x")
    check_seeds([6000, 6099], {"p4_eval"}, "x")


def test_mcnemar_and_paired_summary():
    one, two = mcnemar_exact(5, 0)
    assert math.isclose(one, 1 / 32) and math.isclose(two, 2 / 32)
    s = paired_summary([1, 1, 0, 1, 0], [0, 1, 0, 0, 0])
    assert (s["both"], s["treat_only"], s["ctrl_only"], s["neither"]) == (1, 2, 0, 2)
    assert math.isclose(s["diff"], 0.4)


def test_run_episodes_one_per_seed_in_order():
    pol = StubFlowPolicy()
    venv = ChunkVecEnv({"kind": "stub", "kwargs": {"time_limit_s": 2.0}}, 3)
    nm = NoiseMap(50, 32, "repeat")
    eps = run_episodes(venv, pol, [1100, 1101, 1102, 1103, 1104], "base", 10, nm, 1.5)
    assert [e["seed"] for e in eps] == [1100, 1101, 1102, 1103, 1104]
    again = run_episodes(venv, pol, [1100, 1101, 1102, 1103, 1104], "base", 10, nm, 1.5)
    assert [e["success"] for e in eps] == [e["success"] for e in again]   # paired, reproducible base noise


# ------------------------------------------------------------------------------------------ train smoke
def test_train_and_eval_smoke(tmp_path):
    out = tmp_path / "run"
    cmd = [sys.executable, str(_DSRL / "train_dsrl.py"), "--out", str(out), "--device", "cpu",
           "--n-envs", "2", "--total-decisions", "60", "--warmup-transitions", "20", "--hidden", "32",
           "--utd", "1", "--batch-size", "16", "--eval-seeds", "1900-1901", "--env-kwargs",
           json.dumps({"time_limit_s": 2.0}), "--ckpt-every", "30"]
    subprocess.run(cmd, check=True, timeout=600)
    summ = json.loads((out / "summary.json").read_text())
    assert summ["labels"][0] == "TOY" and summ["counters"]["decisions"] >= 60
    assert (out / "ckpt" / "sac.pt").exists() and (out / "eval_final.json").exists()
    subprocess.run(cmd + ["--resume", "--total-decisions", "80"], check=True, timeout=600)
    assert json.loads((out / "summary.json").read_text())["counters"]["decisions"] >= 80
    ev = tmp_path / "eval.json"
    subprocess.run([sys.executable, str(_DSRL / "eval_dsrl.py"), "--run-dir", str(out), "--device", "cpu",
                    "--seeds", "1950-1953", "--arms", "base,zero,dsrl", "--n-envs", "2",
                    "--env-kwargs", json.dumps({"time_limit_s": 2.0}), "--out", str(ev)], check=True, timeout=600)
    res = json.loads(ev.read_text())
    assert set(res["arms"]) == {"base", "zero", "dsrl"} and "dsrl_vs_base" in res["paired"]
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([sys.executable, str(_DSRL / "eval_dsrl.py"), "--seeds", "6000-6001", "--arms", "base",
                        "--device", "cpu", "--out", str(tmp_path / "x.json")], check=True, timeout=600,
                       capture_output=True)
