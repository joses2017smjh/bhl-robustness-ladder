"""Waiter phase 3 data layer (scripts/waiter/vla): contract validation, splits, stats, chunks, caching, stub env.

CPU only, numpy only (runs in the main venv). The lerobot-side checks live in test_waiter_vla_lerobot.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

VLA = Path(__file__).resolve().parents[1] / "scripts" / "waiter" / "vla"
sys.path.insert(0, str(VLA))

import vla_env_worker as env_worker  # noqa: E402
import rollout_vla  # noqa: E402
import waiter_vla_data as wd  # noqa: E402
from vla_stub_env import (SYNTHETIC_HELD_OUT, SYNTHETIC_SLOTS, SYNTHETIC_TEMPLATES, StubExpert,  # noqa: E402
                      StubWaiterEnv, templates_json)


def write_episode(path, T=6, seed=900000, success=True, instruction="pick up the red tumbler, put it on the tray "
                  "and bring the tray to the left table", **overrides):
    rng = np.random.default_rng(seed)
    fields = {
        "image": rng.integers(0, 256, size=(T, 224, 224, 3), dtype=np.uint8),
        "state": rng.standard_normal((T, 30)).astype(np.float32),
        "action": np.clip(rng.standard_normal((T, 15)).astype(np.float32) * 0.1, -0.2, 0.2),
        "stage": np.zeros(T, dtype=np.int8),
        "instruction": np.asarray(instruction), "seed": np.asarray(seed), "success": np.asarray(success),
        "synthetic": np.asarray(True),
    }
    fields["action"][:, 13:15] = 0.5
    fields.update(overrides)
    np.savez(path, **fields)
    return path


# --------------------------------------------------------------------------------------------------- chunks
def test_action_chunk_pads_with_last_action_and_flags_padding():
    actions = np.arange(5 * 15, dtype=np.float32).reshape(5, 15)
    chunk, pad = wd.action_chunk(actions, 3, 4)
    assert chunk.shape == (4, 15) and chunk.dtype == np.float32
    np.testing.assert_array_equal(chunk[0], actions[3])
    np.testing.assert_array_equal(chunk[1], actions[4])
    np.testing.assert_array_equal(chunk[2], actions[4])
    assert pad.tolist() == [False, False, True, True]
    chunk, pad = wd.action_chunk(actions, 0, 3)
    np.testing.assert_array_equal(chunk, actions[:3])
    assert not pad.any()
    with pytest.raises(IndexError):
        wd.action_chunk(actions, 5, 3)


def test_early_sampling_mode():
    lengths = [600, 30, 640, 500]
    assert wd.parse_early_mode("early:40:0.4") == (40, 0.4)
    for bad in ("early", "early:0:0.5", "early:40:1.0", "early:40:0", "late:40:0.4", "early:x:0.4"):
        with pytest.raises(ValueError):
            wd.parse_early_mode(bad)
    a = wd.batch_indices(7, 0, 64, lengths, "early:40:0.4")
    assert a == wd.batch_indices(7, 0, 64, lengths, "early:40:0.4") != wd.batch_indices(8, 0, 64, lengths, "early:40:0.4")
    assert all(0 <= e < 4 and 0 <= f < lengths[e] for e, f in a)
    many = [s for step in range(400) for s in wd.batch_indices(step, 0, 64, lengths, "early:40:0.4")]
    early = sum(f < 40 for _, f in many) / len(many)
    uniform = sum(min(40, n) for n in lengths) / sum(lengths)       # share of first-40 frames under 'frame'
    assert abs(early - (0.4 + 0.6 * uniform)) < 0.02 and uniform < 0.09
    short = [f for e, f in many if e == 1]                           # the 30-frame episode never yields frame >= 30
    assert short and max(short) < 30
    with pytest.raises(ValueError):
        wd.batch_indices(0, 0, 4, lengths, "nonsense")


def test_batch_indices_are_a_pure_function_of_seed_and_step():
    lengths = [7, 3, 40, 1]
    a = wd.batch_indices(12, 0, 64, lengths)
    assert a == wd.batch_indices(12, 0, 64, lengths)  # resume reproduces the batch
    assert a != wd.batch_indices(13, 0, 64, lengths)
    assert a != wd.batch_indices(12, 1, 64, lengths)
    for mode in ("frame", "episode"):
        for e, f in wd.batch_indices(3, 5, 500, lengths, mode):
            assert 0 <= e < len(lengths) and 0 <= f < lengths[e]
    # frame mode is uniform over frames: the 40-frame episode dominates
    eps = [e for e, _ in wd.batch_indices(0, 0, 5000, lengths, "frame")]
    assert abs(np.mean(np.asarray(eps) == 2) - 40 / 51) < 0.03
    eps = [e for e, _ in wd.batch_indices(0, 0, 5000, lengths, "episode")]
    assert abs(np.mean(np.asarray(eps) == 2) - 0.25) < 0.03


def test_validation_samples_cover_episode_ends_deterministically():
    keys = wd.validation_samples([10, 1, 3], 4)
    assert (0, 0) in keys and (0, 9) in keys and (1, 0) in keys and (2, 2) in keys
    assert len(keys) == len(set(keys)) == 4 + 1 + 3
    assert keys == wd.validation_samples([10, 1, 3], 4)


# --------------------------------------------------------------------------------------------------- stats
def test_norm_stats_and_std_floor():
    a = np.zeros((100, 3), np.float32)
    a[:, 0] = np.linspace(-1, 1, 100)
    a[:, 1] = 5.0  # constant (e.g. fixed-base gravity)
    a[:, 2] = 5.0 + 1e-6 * np.arange(100)  # near-constant
    s = wd.compute_norm_stats([a[:50], a[50:]], std_floor=1e-2)
    assert s["count"] == 100
    np.testing.assert_allclose(s["mean"], a.mean(0), atol=1e-6)
    np.testing.assert_allclose(s["raw_std"], a.astype(np.float64).std(0), atol=1e-9)
    assert s["std"][1] == s["std"][2] == 1e-2
    assert s["floored_dims"] == [1, 2]
    x = np.random.default_rng(0).standard_normal((4, 3))
    np.testing.assert_allclose(wd.unnormalize(wd.normalize(x, s), s), x, atol=1e-12)
    # a small deviation on a constant dim stays bounded thanks to the floor
    assert abs(wd.normalize(np.array([0.0, 5.05, 5.0]), s)[1]) == pytest.approx(5.0)


# --------------------------------------------------------------------------------------------------- templates/splits
def test_template_matching_and_held_out():
    ts = wd.TemplateSet(dict(SYNTHETIC_TEMPLATES), {k: list(v) for k, v in SYNTHETIC_SLOTS.items()},
                        list(SYNTHETIC_HELD_OUT))
    for tid, tmpl in SYNTHETIC_TEMPLATES.items():
        for colour in SYNTHETIC_SLOTS["colour"]:
            text = tmpl.format(colour=colour, side="right")
            assert ts.match(text) == tid
            assert ts.match("  " + text.upper() + " ") == tid  # case/whitespace-insensitive
    with pytest.raises(wd.EpisodeError):
        ts.match("pick up the purple tumbler, put it on the tray and bring the tray to the left table")
    with pytest.raises(ValueError):
        wd.TemplateSet({"A": "x {c}"}, {}, ["B"])
    with pytest.raises(ValueError):
        wd.TemplateSet({"A": "x {c}"}, {}, ["A"])
    ambiguous = wd.TemplateSet({"A": "go to the {side} table", "B": "go to the {x}"}, {}, [])
    with pytest.raises(wd.EpisodeError):
        ambiguous.match("go to the left table")


def test_split_spec():
    spec = wd.SplitSpec(10, 9, wd.parse_seed_list("3100-3102,3200"), frozenset({"T3"}))
    assert spec.assign(3019, "T0") == "val_layout"
    assert spec.assign(3018, "T0") == "train"
    assert spec.assign(3101, None) == "val_layout"
    assert spec.assign(3200, "T1") == "val_layout"
    assert spec.assign(3018, "T3") == "val_template"
    assert spec.assign(3019, "T3") == "val_template"
    assert wd.parse_seed_list("1, 3-5") == frozenset({1, 3, 4, 5})
    with pytest.raises(ValueError):
        wd.parse_seed_list("5-3")


def test_seed_guards_never_admit_scored_ranges():
    wd.check_training_seed(3000, synthetic=False, allow_dev_seeds=False)
    wd.check_training_seed(3999, synthetic=False, allow_dev_seeds=False)
    for bad in (4000, 4099, 6000, 2000, 5000, 7000, 7099):       # 7000-7099: phase 3b's scored range (2026-10-11)
        with pytest.raises(wd.EpisodeError):
            wd.check_training_seed(bad, synthetic=False, allow_dev_seeds=True)
        with pytest.raises(wd.EpisodeError):
            wd.check_training_seed(bad, synthetic=True, allow_dev_seeds=True)
    with pytest.raises(wd.EpisodeError):
        wd.check_training_seed(1500, synthetic=False, allow_dev_seeds=False)
    wd.check_training_seed(1500, synthetic=False, allow_dev_seeds=True)
    wd.check_training_seed(900000, synthetic=True, allow_dev_seeds=False)
    with pytest.raises(wd.EpisodeError):
        wd.check_training_seed(900000, synthetic=False, allow_dev_seeds=True)


# --------------------------------------------------------------------------------------------------- episode files
def test_load_episode_validates_contract(tmp_path):
    ok = write_episode(tmp_path / "ok.npz")
    ep = wd.load_episode(ok)
    assert ep["seed"] == 900000 and ep["success"] is True and ep["synthetic"] is True
    assert ep["image"].shape == (6, 224, 224, 3)
    bad = {
        "dtype": {"state": np.zeros((6, 30), np.float64)},
        "shape": {"action": np.zeros((6, 14), np.float32)},
        "image": {"image": np.zeros((6, 128, 128, 3), np.uint8)},
        "stage": {"stage": np.zeros(6, np.int32)},
        "nan": {"state": np.full((6, 30), np.nan, np.float32)},
        "length": {"stage": np.zeros(5, np.int8)},
    }
    for name, override in bad.items():
        path = write_episode(tmp_path / f"{name}.npz", **override)
        with pytest.raises(wd.EpisodeError):
            wd.load_episode(path)
    fast = np.zeros((6, 15), np.float32)
    fast[:, 0] = 0.6  # |vx| > 0.4: not an action the env could have applied
    path = write_episode(tmp_path / "fast.npz", action=fast)
    with pytest.raises(wd.EpisodeError):
        wd.load_episode(path)
    assert wd.load_episode(path, strict=False)["action"][0, 0] == pytest.approx(0.6)
    np.savez(tmp_path / "missing.npz", image=np.zeros((1, 224, 224, 3), np.uint8))
    with pytest.raises(wd.EpisodeError):
        wd.load_episode(tmp_path / "missing.npz")


def make_synthetic(out: Path, n: int, mode="expert", first_seed=900000):
    subprocess.run([sys.executable, str(VLA / "make_synthetic_episodes.py"), "--out", str(out), "--episodes",
                    str(n), "--mode", mode, "--first-seed", str(first_seed), "--random-length", "12"],
                   check=True, capture_output=True)


def test_build_dataset_splits_cache_and_stats(tmp_path):
    data, cache = tmp_path / "data", tmp_path / "cache"
    make_synthetic(data, 8)  # templates cycle T0..T3; T3 held out
    templates = wd.TemplateSet.load(data / "templates.json")
    spec = wd.SplitSpec(0, 0, wd.parse_seed_list("900001"), frozenset(templates.held_out))
    ds = wd.build_dataset(data, cache, spec, templates=templates, log=lambda *_: None)
    manifest = json.loads((data / "manifest.json").read_text())["episodes"]
    tmpl = {r["seed"]: r["template_id"] for r in manifest}
    for e, split, t in zip(ds.episodes, ds.split_of, ds.template_of):
        seed = e.meta["seed"]
        assert t == tmpl[seed]
        expected = "val_template" if t == "T3" else ("val_layout" if seed == 900001 else "train")
        assert split == expected
    summary = ds.summary()
    assert summary["splits"]["val_template"]["templates"] == ["T3"]
    assert "T3" not in summary["splits"]["train"]["templates"]
    # stats are computed on training episodes only
    stats = wd.dataset_norm_stats(ds)
    train_states = np.concatenate([ds.episodes[i].arrays()["state"] for i in ds.indices("train")])
    np.testing.assert_allclose(stats[wd.STATE_KEY]["mean"], train_states.mean(0), atol=1e-5)
    assert stats[wd.STATE_KEY]["count"] == train_states.shape[0]
    # cache is content addressed and reused
    h = ds.data_hash()
    mtimes = {p: p.stat().st_mtime_ns for p in cache.rglob("*.npy")}
    ds2 = wd.build_dataset(data, cache, spec, templates=templates, log=lambda *_: None)
    assert ds2.data_hash() == h
    assert {p: p.stat().st_mtime_ns for p in cache.rglob("*.npy")} == mtimes
    # sample shapes
    e0 = ds.indices("train")[0]
    s = wd.make_sample(ds.episodes[e0], ds.episodes[e0].length - 1, 50)
    assert s["image"].shape == (224, 224, 3) and s["state"].shape == (30,) and s["action"].shape == (50, 15)
    assert s["action_is_pad"].sum() == 49
    # a changed file never hits the stale cache: manifest sha256 mismatch is refused
    rows = json.loads((data / "manifest.json").read_text())
    rows["episodes"][0]["sha256"] = "0" * 64
    (data / "manifest.json").write_text(json.dumps(rows))
    with pytest.raises(wd.EpisodeError):
        wd.build_dataset(data, tmp_path / "cache2", spec, templates=templates, log=lambda *_: None)


def test_build_dataset_excludes_failed_and_refuses_scored_seeds(tmp_path):
    data = tmp_path / "d"
    data.mkdir()
    rows = []
    for i, (seed, success) in enumerate(((3000, True), (3001, False), (3009, True))):
        p = write_episode(data / f"e{i}.npz", seed=seed, success=success, synthetic=np.asarray(False))
        rows.append({"file": p.name, "seed": seed, "success": success, "sha256": wd.sha256_file(p)})
    (data / "manifest.json").write_text(json.dumps({"episodes": rows}))
    ds = wd.build_dataset(data, tmp_path / "c", wd.SplitSpec(), log=lambda *_: None)
    assert [e.meta["seed"] for e in ds.episodes] == [3000, 3009]
    assert ds.split_of == ["train", "val_layout"]
    assert ds.excluded[0]["seed"] == 3001
    ds = wd.build_dataset(data, tmp_path / "c", wd.SplitSpec(), include_failed=True, log=lambda *_: None)
    assert len(ds.episodes) == 3
    p = write_episode(data / "leak.npz", seed=4005, synthetic=np.asarray(False))
    rows.append({"file": p.name, "seed": 4005, "success": True, "sha256": wd.sha256_file(p)})
    (data / "manifest.json").write_text(json.dumps({"episodes": rows}))
    with pytest.raises(wd.EpisodeError, match="4005"):
        wd.build_dataset(data, tmp_path / "c", wd.SplitSpec(), log=lambda *_: None)


# --------------------------------------------------------------------------------------------------- stub env
def test_stub_env_contract_surface_and_expert():
    env = StubWaiterEnv()
    successes = 0
    for seed in range(900100, 900106):
        obs = env.reset(seed, instruction_id="T1")
        assert obs["image"].shape == (224, 224, 3) and obs["image"].dtype == np.uint8
        assert obs["state"].shape == (30,) and obs["state"].dtype == np.float32
        assert obs["instruction"] == SYNTHETIC_TEMPLATES["T1"].format(colour=env.layout["colour"],
                                                                      side=env.layout["side"])
        expert = StubExpert(env)
        for _ in range(750):
            obs, reward, done, info = env.step(expert.act())
            assert set(info) >= {"success", "fell", "stage", "tumbler_on_tray", "tray_on_target", "t"}
            if done:
                break
        assert reward == float(info["success"])
        successes += info["success"]
    assert successes == 6
    # layout is a pure function of the seed
    a = StubWaiterEnv().reset(900100)
    b = StubWaiterEnv().reset(900100)
    np.testing.assert_array_equal(a["image"], b["image"])
    # a jerky base command trips the synthetic robot
    env.reset(900100)
    act = np.zeros(15, np.float32)
    act[0] = 0.4
    env.step(act)
    act[0] = -0.4
    *_, done, info = env.step(act)
    assert info["fell"] and done
    with pytest.raises(ValueError):
        env.reset(1)
        env.step(np.zeros(14, np.float32))


def test_env_worker_codec_and_subprocess_roundtrip():
    msg = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": [np.int64(3), "x", (np.bool_(True),)]}
    out = env_worker.decode(env_worker.encode(msg))
    np.testing.assert_array_equal(out["a"], msg["a"])
    assert out["a"].dtype == np.float32 and out["b"][0] == 3 and out["b"][2] == (True,)
    env = rollout_vla.SubprocessEnv("stub", {}, sys.executable, {})
    try:
        assert env.synthetic
        obs = env.reset(900200, "T0")
        assert obs["image"].shape == (224, 224, 3)
        ref = StubWaiterEnv()
        ref_obs = ref.reset(900200, instruction_id="T0")
        np.testing.assert_array_equal(obs["image"], ref_obs["image"])
        a = np.zeros(15, np.float32)
        env.step_async(a)
        obs, reward, done, info = env.step_wait()
        np.testing.assert_array_equal(obs["state"], ref.step(a)[0]["state"])
    finally:
        env.close()


def test_rollout_helpers(tmp_path):
    a = np.zeros(15, np.float32)
    a[0], a[1], a[2], a[13], a[14] = 1.0, -1.0, 0.1, 1.5, -0.2
    c, clipped = rollout_vla.clip_action(a)
    assert clipped and c[0] == pytest.approx(0.4) and c[1] == pytest.approx(-0.3) and c[2] == pytest.approx(0.1)
    assert c[13] == 1.0 and c[14] == 0.0
    assert rollout_vla.clip_action(np.zeros(15, np.float32))[1] is False
    lo, hi = rollout_vla.wilson(5, 10)
    assert lo < 0.5 < hi
    assert rollout_vla.wilson(0, 0) is None
    rollout_vla.check_seeds([1000, 1999], "dev", "waiter")
    rollout_vla.check_seeds([910000], "dev", "stub")
    for seeds, purpose, env in (([4000], "dev", "waiter"), ([3000], "dev", "waiter"), ([4000], "dev", "stub"),
                                ([1000], "scored", "waiter"), ([4000], "scored", "stub")):
        with pytest.raises(SystemExit):
            rollout_vla.check_seeds(seeds, purpose, env)
    rollout_vla.check_seeds([4000, 4099], "scored", "waiter")
    rollout_vla.check_seeds([7000, 7099], "scored", "waiter")       # phase 3b's range
    for seeds in ([4000, 7000], [7100], [6000]):                    # mixed ranges, outside, another phase's range
        with pytest.raises(SystemExit):
            rollout_vla.check_seeds(seeds, "scored", "waiter")
    gate = tmp_path / "rule.json"
    gate.write_text(json.dumps({"seed0": 7000, "seed1": 7099}))
    rollout_vla.check_gate_seeds(list(range(7000, 7100)), gate)
    for seeds in (list(range(4000, 4100)), list(range(7000, 7050))):  # the gate file names its own seeds
        with pytest.raises(SystemExit):
            rollout_vla.check_gate_seeds(seeds, gate)
    assert rollout_vla.parse_instruction_ids("T0, 3") == ["T0", 3]
    assert rollout_vla.parse_instruction_ids("") == [None]


def test_templates_json_roundtrip(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(templates_json()))
    ts = wd.TemplateSet.load(path)
    assert ts.held_out == SYNTHETIC_HELD_OUT and set(ts.templates) == set(SYNTHETIC_TEMPLATES)


def test_template_parse_and_counterfactual():
    ts = wd.TemplateSet(dict(SYNTHETIC_TEMPLATES), {k: list(v) for k, v in SYNTHETIC_SLOTS.items()},
                        list(SYNTHETIC_HELD_OUT))
    text = SYNTHETIC_TEMPLATES["T2"].format(colour="green", side="left")
    assert ts.parse(text) == ("T2", {"colour": "green", "side": "left"})
    swapped = ts.counterfactual(text, "side")
    assert swapped == SYNTHETIC_TEMPLATES["T2"].format(colour="green", side="right")
    assert ts.parse(swapped)[0] == "T2"
    assert ts.counterfactual(swapped, "side") == text
    assert ts.parse(ts.counterfactual(text, "colour"))[1]["colour"] == "yellow"
    with pytest.raises(ValueError):
        ts.counterfactual(text, "size")
