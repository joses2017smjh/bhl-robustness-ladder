"""Tests for scripts/bench/stand3_replay.py (Stand3 replay diagnostic, report-only).

Everything here runs on the login node without Isaac: the predeclared
thresholds against stand_mdp.py, the over-deck classification (exhaustive,
exclusive, bit-identical to stand_mdp's torch masks in float32), run lengths,
the funding reading, synthetic traces end to end, the config check on the two
real Stand3 params/env.yaml files, and the preflight on the real run dirs.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("stand3_replay", _REPO / "scripts" / "bench" / "stand3_replay.py")
sr = importlib.util.module_from_spec(_spec)
sys.modules["stand3_replay"] = sr
_spec.loader.exec_module(sr)

_RUNS = _REPO / "external" / "Berkeley-Humanoid-Lite" / "logs" / "rsl_rl" / "task_v2"
_S0 = _RUNS / sr.RUNS["s0"]
_S1 = _RUNS / sr.RUNS["s1"]
_TRAIN_COMMIT = "493c123"
_have_runs = pytest.mark.skipif(not (_S0 / "params" / "env.yaml").is_file()
                                or not (_S1 / "params" / "env.yaml").is_file(),
                                reason="Stand3 run dirs not present")
f32 = np.float32


def _stand_mdp(path: Path):
    spec = importlib.util.spec_from_file_location(f"stand_mdp_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git_show(rev_path: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(_REPO), "show", rev_path], check=True,
                              capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


@pytest.fixture(scope="module")
def sm():
    return _stand_mdp(_REPO / "src" / "bhl_robust" / "tasks" / "stand_mdp.py")


@pytest.fixture(scope="module")
def pinned_src(tmp_path_factory):
    """The six pinned task files as committed at the training commit."""
    root = tmp_path_factory.mktemp("pinned") / "src"
    for f in sr.PINNED_FILES:
        txt = _git_show(f"{_TRAIN_COMMIT}:src/{f}")
        if txt is None:
            pytest.skip("git / training commit unavailable")
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(txt)
    return root


# ------------------------------------------------------------------ thresholds
def test_literals_equal_current_stand_mdp(sm):
    assert sr.constants_mismatch(sm) == {}


def test_literals_equal_training_commit_stand_mdp(pinned_src):
    assert sr.constants_mismatch(_stand_mdp(pinned_src / "bhl_robust/tasks/stand_mdp.py")) == {}


def test_derived_literals_are_consistent():
    assert math.isclose(sr.SEAT_X_LO, sr.DECK_EDGE + 0.02)
    assert math.isclose(sr.REQUIRED_SHIFT, sr.SEAT_X_LO)
    assert math.isclose(sr.DECK_SEATED_Z, 0.41 + 0.02 + sr.CUBE_HALF)
    assert sr.MEASURED_SHIFT_ARMS < sr.MEASURED_SHIFT_ARMS_TWIST < sr.REQUIRED_SHIFT
    assert (sr.NUM_ENVS, sr.EPISODES_PER_ENV, sr.REPLAY_SEED) == (32, 3, 1000)
    assert sr.REPLAY_SEED not in sr.TRAIN_SEEDS
    assert sr.CHECKPOINT == "model_7999.pt"


# ---------------------------------------------------------------- classification
def _boundary_values():
    """Every threshold, its float32 neighbours, and a few interior points."""
    xs, ys, zs, vs = [], [], [], []
    for t in (sr.DECK_EDGE, sr.SEAT_X_LO, sr.SEAT_X_HI):
        c = f32(t)
        xs += [c, np.nextafter(c, f32(0)), np.nextafter(c, f32(1)), -c]
    xs += [f32(0.0), f32(0.25), f32(-0.3), f32(0.6)]
    c = f32(sr.SEAT_Y_HALF)
    ys += [f32(0.0), c, -c, np.nextafter(c, f32(0)), np.nextafter(c, f32(1)), f32(0.2)]
    for t in (sr.OFF_FLOOR_Z, sr.DECK_SEATED_Z - sr.SEAT_Z_TOL, sr.DECK_SEATED_Z,
              sr.DECK_SEATED_Z + sr.SEAT_Z_TOL):
        c = f32(t)
        zs += [c, np.nextafter(c, f32(0)), np.nextafter(c, f32(1))]
    zs += [f32(0.2), f32(0.62), f32(0.45)]
    # z values whose float32 difference to 0.57 sits exactly at +/- the tolerance
    zs += [f32(0.57) + f32(0.02), f32(0.57) - f32(0.02)]
    c = f32(sr.SEAT_SPEED)
    vs += [f32(0.0), c, np.nextafter(c, f32(0)), np.nextafter(c, f32(1)), f32(0.3)]
    g = np.array(np.meshgrid(xs, ys, zs, vs, indexing="ij"), dtype=np.float32).reshape(4, -1).T
    return g[:, :3].copy(), g[:, 3].copy()


def _random_points(n=20000, seed=3):
    rng = np.random.default_rng(seed)
    p = np.stack([rng.uniform(-0.5, 0.5, n), rng.uniform(-0.15, 0.15, n), rng.uniform(0.3, 0.65, n)], 1)
    v = rng.uniform(0.0, 0.1, n)
    return p.astype(np.float32), v.astype(np.float32)


@pytest.mark.parametrize("points", ["boundary", "random"])
def test_classes_exhaustive_and_exclusive(points):
    p, v = _boundary_values() if points == "boundary" else _random_points()
    cls, reason = sr.classify_steps(p, v)
    over = sr.over_deck_np(p)
    assert set(np.unique(cls)) <= {sr.NOT_OVER, sr.MOVING, sr.HELD_ABOVE, sr.SEATED, sr.RESTING}
    assert np.array_equal(cls != sr.NOT_OVER, over)
    assert np.array_equal(cls == sr.SEATED, sr.seated_np(p, v))
    assert np.all((reason != 0) == (cls == sr.RESTING))
    # height tests partition every point
    d = p[:, 2] - f32(sr.DECK_SEATED_Z)
    k = (d >= f32(sr.SEAT_Z_TOL)).astype(int) + (np.abs(d) < f32(sr.SEAT_Z_TOL)) + (d <= -f32(sr.SEAT_Z_TOL))
    assert np.all(k == 1)


@pytest.mark.parametrize("points", ["boundary", "random"])
def test_numpy_masks_bit_identical_to_stand_mdp_torch(sm, points):
    torch = pytest.importorskip("torch")
    p, v = _boundary_values() if points == "boundary" else _random_points()
    tp, tv = torch.from_numpy(p), torch.from_numpy(v)
    assert np.array_equal(sm.over_deck_mask(tp).numpy(), sr.over_deck_np(p))
    assert np.array_equal(sm.seated_mask(tp, tv).numpy(), sr.seated_np(p, v))


@pytest.mark.parametrize("pt, speed, want, reason", [
    ((0.25, 0.0, 0.60), 0.0, sr.HELD_ABOVE, 0),
    ((0.25, 0.0, 0.57), 0.0, sr.SEATED, 0),
    ((-0.30, 0.02, 0.575), 0.01, sr.SEATED, 0),
    ((0.25, 0.0, 0.54), 0.0, sr.RESTING, sr.REASON_LOW),
    ((0.18, 0.0, 0.57), 0.0, sr.RESTING, sr.REASON_X_BAND),
    ((0.46, 0.0, 0.57), 0.0, sr.RESTING, sr.REASON_X_BAND),
    ((0.25, 0.12, 0.57), 0.0, sr.RESTING, sr.REASON_Y_BAND),
    ((0.18, 0.12, 0.50), 0.0, sr.RESTING, sr.REASON_LOW | sr.REASON_X_BAND | sr.REASON_Y_BAND),
    ((0.25, 0.0, 0.57), 0.05, sr.MOVING, 0),
    ((0.25, 0.0, 0.62), 0.2, sr.MOVING, 0),
    ((0.16, 0.0, 0.62), 0.0, sr.NOT_OVER, 0),
    ((0.30, 0.0, 0.30), 0.0, sr.NOT_OVER, 0),
])
def test_named_points(pt, speed, want, reason):
    cls, r = sr.classify_steps(np.array([pt], np.float32), np.array([speed], np.float32))
    assert int(cls[0]) == want and int(r[0]) == reason


def test_run_lengths_and_counter():
    m = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1], bool)
    assert sr.run_lengths(m) == [2, 3, 1]
    assert sr.consecutive_count(m).tolist() == [0, 1, 2, 0, 1, 2, 3, 0, 0, 1]
    assert sr.run_lengths(np.zeros(4, bool)) == []


def test_episode_segments():
    valid = np.array([[1, 1], [1, 1], [1, 1], [1, 0]], bool)
    ep = np.array([[0, 0], [0, 1], [1, 2], [1, 3]], np.int8)
    segs = sr.episode_segments(valid, ep, 3)
    assert [(n, e, list(i)) for n, e, i in segs] == [(0, 0, [0, 1]), (0, 1, [2, 3]), (1, 0, [0]), (1, 1, [1]), (1, 2, [2])]
    bad = np.array([[0], [1], [0]], np.int8)
    with pytest.raises(ValueError):
        sr.episode_segments(np.ones((3, 1), bool), bad, 2)


# ---------------------------------------------------------------- torch helpers
def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float32)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def test_hand_box_distance_and_face_tilt():
    torch = pytest.importorskip("torch")
    flip = _rot_x(math.pi)                         # the as-trained spawn: 180 deg about x
    rolled = _rot_y(math.pi / 2) @ flip            # rolled 90 deg in the hands
    tipped = _rot_y(math.radians(20.0)) @ flip
    rc = torch.from_numpy(np.stack([np.eye(3, dtype=np.float32), flip, rolled, tipped]))
    cube = torch.tensor([[0.0, 0.0, 0.55]] * 4)
    hands = cube[:, None, :] + torch.tensor([[0.0, 0.16, 0.0], [0.0, 0.0, 0.0]])[None].repeat(4, 1, 1)
    d = sr.hand_box_distance(rc, hands, cube)
    assert torch.allclose(d[:3, 0], torch.tensor(0.02), atol=1e-5)      # 2 cm off the face, any flip/roll
    assert torch.allclose(d[:, 1], torch.tensor(-sr.CUBE_HALF), atol=1e-5)
    deg, axis = sr.cube_face_tilt(rc)
    assert deg[:3].abs().max() < 1e-2 and abs(float(deg[3]) - 20.0) < 1e-2
    assert axis.tolist()[:3] == [2, 2, 0]


def test_robot_mechanics_lean_and_twist():
    L = 5
    lean = math.radians(10.0)
    tr = {"base_pos_a": np.zeros((L, 1, 3), np.float32), "tilt_a": np.zeros((L, 1), np.float32),
          "up_a": np.tile(np.array([math.sin(lean), 0.0, math.cos(lean)], np.float32), (L, 1, 1)),
          "fwd_a": np.zeros((L, 1, 3), np.float32)}
    tr["base_pos_a"][:, 0, 0] = np.linspace(0, 0.04, L)
    yaws = np.radians([-90, -85, -80, -75, -70])
    tr["fwd_a"][:, 0, 0], tr["fwd_a"][:, 0, 1] = np.cos(yaws), np.sin(yaws)
    m = sr._robot_mechanics(tr, "a", np.arange(L), 0, k=4, side=-1.0)
    assert abs(m["max_abs_lean_x_deg"] - 10.0) < 1e-3
    assert abs(m["max_abs_yaw_rel_deg"] - 20.0) < 1e-3
    assert abs(m["at_carried_max"]["lean_x_toward_deck_deg"] + 10.0) < 1e-3   # cube on the -x side
    assert abs(m["at_carried_max"]["base_dx_toward_deck_m"] + 0.04) < 1e-6


# ---------------------------------------------------------------- funding reading
def _summ(S, R, M, D, complete=True, as_trained=True):
    return {"complete": complete, "as_trained": as_trained, "episodes_finished": 96, "episodes_expected": 96,
            "S_success_share": S, "R_reach_share": R, "M_median_carried_max_abs_x": M,
            "D_setdown_share_of_reaching": D}


@pytest.mark.parametrize("args, code", [
    ((0.10, 0.0, 0.05, None), "R1"),
    ((0.5, 1.0, 0.3, 1.0), "R1"),
    ((0.0, 0.0, 0.119, None), "R2"),
    ((0.0, 0.09, 0.18, 0.0), "R2"),
    ((0.0, 0.0, 0.118, None), "R2"),
    ((0.0, 0.0, 0.044, None), "R2"),
    ((0.0, 0.0, 0.0439, None), "R3"),
    ((0.09, 0.05, 0.02, 1.0), "R3"),
    ((0.0, 0.10, 0.12, 0.49), "R4"),
    ((0.0, 0.8, 0.25, 0.0), "R4"),
    ((0.0, 0.10, 0.12, 0.50), "R5"),
    ((0.05, 1.0, 0.3, 1.0), "R5"),
])
def test_funding_reading_table(args, code):
    assert sr.funding_reading(_summ(*args))["code"] == code


def test_r2_reach_limit_subreading():
    inside = sr.funding_reading(_summ(0.0, 0.0, 0.10, None))
    beyond = sr.funding_reading(_summ(0.0, 0.0, 0.15, None))
    assert inside["code"] == beyond["code"] == "R2"
    assert inside["within_twist_envelope"] is True and beyond["within_twist_envelope"] is False


def test_funding_reading_incomplete():
    assert sr.funding_reading(_summ(0.5, 1, 0.3, 1, complete=False))["code"] == "INCOMPLETE"
    assert sr.funding_reading(_summ(0.5, 1, 0.3, 1, as_trained=False))["code"] == "INCOMPLETE"


@pytest.mark.parametrize("a, b, funded", [
    ("R4", "R4", True), ("R2", "R2", True), ("R5", "R5", True),
    ("R1", "R1", False), ("R3", "R3", False), ("R4", "R5", False), ("R2", "R3", False),
    ("R4", "INCOMPLETE", False),
])
def test_pair_reading(a, b, funded):
    pr = sr.pair_reading({"s0": {"code": a, "kind": "k"}, "s1": {"code": b, "kind": "k"}})
    assert (pr["funding"] == "STAND4_JUSTIFIED") is funded


# ---------------------------------------------------------------- synthetic end to end
@pytest.mark.parametrize("scenario, code", [
    ("no_shift", "R3"), ("short_shift", "R2"), ("reach_hold", "R4"),
    ("setdown_jitter", "R5"), ("place", "R1"),
])
def test_synthetic_scenarios(scenario, code):
    tr = sr.synthetic_trace(scenario, num_envs=4, episodes=3)
    res = sr.summarize(tr, 4, 3)
    s = res["summary"]
    assert s["complete"] and s["consistent"], s["mismatch_totals"]
    assert sr.funding_reading(s)["code"] == code
    if scenario == "place":
        assert s["S_success_share"] == 1.0 and s["success_recount_share"] == 1.0
        assert s["longest_seated_run_max"] == sr.SEAT_HOLD_STEPS
    if scenario == "reach_hold":
        assert s["class_steps"]["HELD_ABOVE"] > 0 and s["class_steps"]["SEATED"] == 0
        assert s["held_above_pair_touch_share"] == 1.0
        assert s["moving_above_seat_share_of_moving"] is not None


def test_mismatch_is_detected():
    tr = sr.synthetic_trace("place", num_envs=2, episodes=3)
    i = np.nonzero(tr["seated_env"][:, 0])[0][0]
    tr["seated_env"][i, 0] = False
    s = sr.summarize(tr, 2, 3)["summary"]
    assert not s["consistent"] and s["mismatch_totals"]["seated"] == 1


def test_unfinished_episodes_make_it_incomplete():
    tr = sr.synthetic_trace("reach_hold", num_envs=2, episodes=3)
    last = np.nonzero(tr["done"][:, 1])[0][-1]
    tr["done"][last, 1] = False
    tr["term_time_out"][last, 1] = False
    s = sr.summarize(tr, 2, 3)["summary"]
    assert s["episodes_finished"] == 5 and not s["complete"]
    assert sr.funding_reading(s)["code"] == "INCOMPLETE"


def test_cli_synth_classify_report(tmp_path, capsys):
    for tag, sc in (("s0", "reach_hold"), ("s1", "reach_hold")):
        pre = str(tmp_path / tag)
        assert sr.main(["synth", "--scenario", sc, "--out-prefix", pre]) == 0
        assert sr.main(["classify", "--trace", pre + ".trace.npz", "--meta", pre + ".meta.json",
                        "--out", str(tmp_path / f"replay_{tag}.json"), "--tag", tag]) == 0
    assert sr.main(["report", "--replays", str(tmp_path / "replay_s0.json"), str(tmp_path / "replay_s1.json"),
                    "--out", str(tmp_path / "reading.json")]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC FUNDING" in out and "STAND4_JUSTIFIED" in out
    d = json.loads((tmp_path / "reading.json").read_text())
    assert d["synthetic"] and d["pair"]["codes"] == {"s0": "R4", "s1": "R4"}
    # never overwrite
    with pytest.raises(SystemExit):
        sr.main(["report", "--replays", str(tmp_path / "replay_s0.json"), str(tmp_path / "replay_s1.json"),
                 "--out", str(tmp_path / "reading.json")])
    with pytest.raises(SystemExit):
        sr.main(["synth", "--scenario", "place", "--out-prefix", str(tmp_path / "s0")])


def test_report_with_missing_replay_is_incomplete(tmp_path, capsys):
    pre = str(tmp_path / "s0")
    sr.main(["synth", "--scenario", "reach_hold", "--out-prefix", pre])
    sr.main(["classify", "--trace", pre + ".trace.npz", "--meta", pre + ".meta.json",
             "--out", str(tmp_path / "replay_s0.json")])
    sr.main(["report", "--replays", str(tmp_path / "replay_s0.json"), str(tmp_path / "nope.json"),
             "--out", str(tmp_path / "reading.json")])
    d = json.loads((tmp_path / "reading.json").read_text())
    assert d["pair"]["funding"] == "NO_STAND4_ON_THIS_EVIDENCE"
    assert d["readings"]["s1"]["code"] == "INCOMPLETE"


def test_not_as_trained_meta_gives_no_reading(tmp_path):
    pre = tmp_path / "s0"
    tr = sr.synthetic_trace("reach_hold", num_envs=2, episodes=3)
    np.savez_compressed(str(pre) + ".trace.npz", **tr)
    meta = {"synthetic": False, "num_envs": 2, "episodes_per_env": 3, "trace_complete": True,
            "config_check": {"verdict": "FAIL"}, "source": {"pre_quat_fix": True}, "label": "x"}
    (tmp_path / "s0.meta.json").write_text(json.dumps(meta))
    res = sr.classify_files(Path(str(pre) + ".trace.npz"), tmp_path / "s0.meta.json")
    assert res["reading"]["code"] == "INCOMPLETE" and not res["as_trained"]
    meta["config_check"]["verdict"] = "PASS"
    meta["source"]["pre_quat_fix"] = False
    (tmp_path / "s0.meta.json").write_text(json.dumps(meta))
    assert sr.classify_files(Path(str(pre) + ".trace.npz"), tmp_path / "s0.meta.json")["reading"]["code"] == "INCOMPLETE"


# ---------------------------------------------------------------- config check
def test_config_diff_semantics():
    a = {"scene": {"object": {"init_state": {"rot": (1.0, 0.0, 0.0, 0.0)}}}, "seed": 0,
         "rewards": {"x": {"weight": 1.0}}, "decimation": 8}
    b = json.loads(json.dumps(a))                           # tuples become lists
    b["seed"] = 1000
    assert sr.config_check(a, b)["verdict"] == "PASS"
    b["rewards"]["x"]["weight"] = 2.0
    cc = sr.config_check(a, b)
    assert cc["verdict"] == "PASS" and cc["n_reported"] == 1
    b["scene"]["object"]["init_state"]["rot"] = [0.0, 0.0, 0.0, 1.0]
    cc = sr.config_check(a, b)
    assert cc["verdict"] == "FAIL" and cc["gated_diffs"][0]["path"] == "scene.object.init_state.rot[0]"
    assert sr.config_diff({"a": True}, {"a": 1}) != []
    assert sr.config_diff({"a": 1}, {"a": 1.0}) == []


@_have_runs
def test_config_check_real_params_s0_vs_s1():
    e0 = sr.load_params_yaml(_S0 / "params" / "env.yaml")
    e1 = sr.load_params_yaml(_S1 / "params" / "env.yaml")
    cc = sr.config_check(e0, e1)
    assert cc["verdict"] == "PASS" and [r["path"] for r in cc["allowlisted"]] == ["seed"]
    assert cc["n_reported"] == 0
    ac = sr.agent_check(sr.load_params_yaml(_S0 / "params" / "agent.yaml"),
                        sr.load_params_yaml(_S1 / "params" / "agent.yaml"))
    assert ac["n_diffs"] == 0
    assert e0["scene"]["object"]["init_state"]["rot"] == [1.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------- preflight / args
def _args(tmp_path, pinned_src, label="s0", **kw):
    a = {"label": label, "run_dir": str(_RUNS / sr.RUNS[label]), "checkpoint": sr.CHECKPOINT,
         "src_root": str(pinned_src), "out_prefix": str(tmp_path / label), "seed": sr.REPLAY_SEED,
         "num_envs": sr.NUM_ENVS, "episodes": sr.EPISODES_PER_ENV, "enable_cameras": True,
         "config_check_mode": "gate"}
    a.update(kw)
    return a


@_have_runs
@pytest.mark.parametrize("label", ["s0", "s1"])
def test_preflight_passes_on_real_runs(tmp_path, pinned_src, label):
    plan = sr.preflight(_args(tmp_path, pinned_src, label))
    assert plan["trained_seed"] == int(label[1]) and plan["replay_seed"] == 1000
    assert len(plan["checkpoint_sha256"]) == 64 and plan["trained_object_rot"] == [1.0, 0.0, 0.0, 0.0]


@_have_runs
@pytest.mark.parametrize("override", [
    {"seed": 0}, {"seed": 1}, {"num_envs": 16}, {"episodes": 2}, {"checkpoint": "model_7800.pt"},
    {"label": "s1"},
])
def test_preflight_refuses(tmp_path, pinned_src, override):
    a = _args(tmp_path, pinned_src)
    if override.get("label") == "s1":           # s1 label with s0's run dir
        a["label"] = "s1"
    else:
        a.update(override)
    with pytest.raises(SystemExit):
        sr.preflight(a)


@_have_runs
def test_preflight_refuses_fixed_source_and_existing_outputs(tmp_path, pinned_src):
    fixed = tmp_path / "fixed" / "src"
    for f in sr.PINNED_FILES:
        (fixed / f).parent.mkdir(parents=True, exist_ok=True)
        (fixed / f).write_text((pinned_src / f).read_text())
    cfg = fixed / "bhl_robust/tasks/coop_lift_env_cfg.py"
    cfg.write_text(cfg.read_text().replace("rot=(1.0, 0.0, 0.0, 0.0)", "rot=native_quat(_OBJECT_ROT_WXYZ)"))
    with pytest.raises(SystemExit, match="pre-fix"):
        sr.preflight(_args(tmp_path, fixed))
    a = _args(tmp_path, pinned_src)
    Path(a["out_prefix"] + ".meta.json").write_text("{}")
    with pytest.raises(SystemExit, match="overwrite"):
        sr.preflight(a)


def test_args_file_validation(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"label": "s0"}))
    with pytest.raises(SystemExit, match="missing"):
        sr.read_args_file(str(p))
    full = {k: 1 for k in ("label", "run_dir", "checkpoint", "src_root", "out_prefix", "seed", "num_envs",
                           "episodes", "enable_cameras")}
    full["config_check_mode"] = "loose"
    p.write_text(json.dumps(full))
    with pytest.raises(SystemExit, match="gate"):
        sr.read_args_file(str(p))


def test_module_imports_without_isaac_or_torch():
    code = ("import importlib.util, sys; s = importlib.util.spec_from_file_location('m', sys.argv[1]); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "print(any(k.startswith('isaac') for k in sys.modules), 'torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code, str(_REPO / "scripts" / "bench" / "stand3_replay.py")],
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


def test_make_args_uses_predeclared_protocol(tmp_path):
    out = tmp_path / "s1.args.json"
    assert sr.main(["make-args", "--label", "s1", "--run-dir", str(_S1), "--src-root", "/x/src",
                    "--out-prefix", str(tmp_path / "s1"), "--out", str(out)]) == 0
    a = sr.read_args_file(str(out))
    assert (a["seed"], a["num_envs"], a["episodes"], a["checkpoint"]) == (1000, 32, 3, "model_7999.pt")
    assert a["config_check_mode"] == "gate" and a["enable_cameras"] is True
    with pytest.raises(SystemExit):
        sr.main(["make-args", "--label", "s1", "--run-dir", str(_S1), "--src-root", "/x/src",
                 "--out-prefix", str(tmp_path / "s1"), "--out", str(out)])


# ---------------------------------------------------------------- fake-env rollout
# The Isaac path's recorder and episode loop, run against a duck-typed env that
# mimics ManagerBasedRLEnv.step's order (physics, episode_length_buf += 1,
# termination compute, auto-reset of done envs, observations). The cube follows
# the synthetic 'place' path (success after 12 seated steps) or 'reach_hold'
# (time-out); the env's hold counter and success use stand_mdp's seated_mask.
def _mat_xyzw(q):
    import torch
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


class _Obj:
    def __init__(self, **kw):
        self.data = type("D", (), {})()
        for k, v in kw.items():
            setattr(self.data, k, v)


class _Scene(dict):
    env_origins = None


class _FakeTM:
    def __init__(self, u, sm):
        self.u, self.sm, self._terms = u, sm, {}

    def compute(self):
        import torch
        u = self.u
        p, sp = _fake_obj_state(u)
        now = self.sm.seated_mask(p, sp)
        hold = getattr(u, "_bhl_deck_hold", None)
        hold = torch.zeros(u.num_envs, dtype=torch.long) if hold is None else hold
        hold = torch.where(now, hold + 1, torch.zeros_like(hold))
        u._bhl_deck_hold = hold
        self._terms = {"success": hold >= sr.SEAT_HOLD_STEPS,
                       "fallen": torch.zeros(u.num_envs, dtype=torch.bool),
                       "time_out": u.episode_length_buf >= u.max_episode_length}
        return self._terms["success"] | self._terms["time_out"]

    def get_term(self, name):
        return self._terms[name]

    @property
    def time_outs(self):
        return self._terms["time_out"]


def _fake_obj_state(u):
    o = u.scene["object"].data
    return o.root_pos_w - u.scene.env_origins, o.root_lin_vel_w.norm(dim=-1)


class _FakeEnv:
    def __init__(self, sm, scenarios, L=60):
        import torch
        self.num_envs, self.device, self.max_episode_length = len(scenarios), "cpu", L
        self.scenarios, self.rng = scenarios, np.random.default_rng(0)
        N = self.num_envs
        self.episode_length_buf = torch.zeros(N, dtype=torch.long)
        self.k = [0] * N
        self.eps = [0] * N
        self.paths = [None] * N
        self.scene = _Scene()
        self.scene.env_origins = torch.tensor([[4.0 * n, 0.0, 0.0] for n in range(N)])
        s2 = 0.70710678
        qa = torch.tensor([[0.0, 0.0, -s2, s2]] * N)             # xyzw yaw -90 (robot_a faces -y)
        qb = torch.tensor([[0.0, 0.0, s2, s2]] * N)
        self.scene["object"] = _Obj(root_pos_w=torch.zeros(N, 3), root_lin_vel_w=torch.zeros(N, 3),
                                    root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * N))   # as trained
        for r, q, y in (("a", qa, 0.48), ("b", qb, -0.48)):
            self.scene[f"robot_{r}"] = _Obj(root_pos_w=self.scene.env_origins + torch.tensor([0.0, y, 0.0]),
                                            root_quat_w=q, body_pos_w=torch.zeros(N, 3, 3))
            self.scene[f"contact_{r}"] = _Obj(net_forces_w=torch.full((N, 3, 3), 2.0))
        self.termination_manager = _FakeTM(self, sm)

    def _new_path(self, n):
        sign = 1.0 if (n + self.eps[n]) % 2 == 0 else -1.0
        p, v, sp = sr._scenario_path(self.scenarios[n], self.max_episode_length + 1, self.rng, sign)
        self.paths[n] = (p, v)
        self.k[n] = 0

    def _apply(self):
        import torch
        o = self.scene["object"].data
        for n in range(self.num_envs):
            p, v = self.paths[n]
            o.root_pos_w[n] = torch.from_numpy(p[self.k[n]]) + self.scene.env_origins[n]
            o.root_lin_vel_w[n] = torch.from_numpy(v[self.k[n]])
        for r, side in (("a", 1.0), ("b", -1.0)):
            bp = self.scene[f"robot_{r}"].data.body_pos_w
            bp[:, 1] = o.root_pos_w + torch.tensor([0.0, 0.16 * side, 0.0])
            bp[:, 2] = o.root_pos_w + torch.tensor([0.0, 0.16 * side, 0.0])

    def reset(self, seed=None):
        for n in range(self.num_envs):
            self._new_path(n)
        self.episode_length_buf.zero_()
        self._apply()

    def reset_idx(self, ids):
        for n in ids:
            self.eps[n] += 1
            self._new_path(n)
            self.episode_length_buf[n] = 0
        self._apply()


class _FakeWrapped:
    def __init__(self, u):
        self.u = u

    def get_observations(self):
        import torch
        return torch.zeros(self.u.num_envs, 5)

    def step(self, actions):
        import torch
        u = self.u
        for n in range(u.num_envs):
            u.k[n] += 1
        u._apply()
        u.episode_length_buf += 1
        dones = u.termination_manager.compute()
        ids = [int(i) for i in torch.nonzero(dones).flatten()]
        if ids:
            u.reset_idx(ids)
        return self.get_observations(), torch.zeros(u.num_envs), dones.long(), {}


def _fake_run(sm, scenarios, T=None, episodes=3):
    torch = pytest.importorskip("torch")
    u = _FakeEnv(sm, scenarios)
    api = sr.SimpleNamespace(
        obj_state=_fake_obj_state, over_deck_mask=sm.over_deck_mask, seated_mask=sm.seated_mask,
        t=lambda v: v, matrix_from_quat=_mat_xyzw,
        tilt_from_quat=lambda robot: torch.acos(_mat_xyzw(robot.data.root_quat_w)[:, 2, 2].clamp(-1, 1)),
        unpack_wxyz=lambda q: (q[..., 3], q[..., 0], q[..., 1], q[..., 2]))
    T = T or u.max_episode_length * episodes + 50
    buf = sr.make_buffers(T, u.num_envs, "cpu")
    state = {"t": 0, "conv_dev": torch.zeros(())}
    contact = {r: (u.scene[f"contact_{r}"], [1, 2]) for r in ("a", "b")}
    record = sr.make_recorder(u, buf, api, {"a": [1, 2], "b": [1, 2]}, contact, state)
    policy = lambda obs: torch.zeros(obs.shape[0], 4)                  # noqa: E731
    roll = sr.run_episodes(u, _FakeWrapped(u), policy, buf, record, episodes, 5, state)
    return u, roll, sr.trace_from_buffers(buf, roll["t_end"]), state


def test_fake_rollout_records_terminal_steps(sm):
    u, roll, tr, state = _fake_run(sm, ["place", "place", "reach_hold", "reach_hold"])
    assert roll["all_done"] and roll["episodes_done_min"] == 3 and roll["determinism"]["repeat_equal"]
    assert u.termination_manager.compute.__func__ is _FakeTM.compute             # hook removed
    res = sr.summarize(tr, 4, 3)
    s = res["summary"]
    assert s["complete"] and s["consistent"], s["mismatch_totals"]
    by_env = {}
    for ep in res["episodes"]:
        by_env.setdefault(ep["env"], []).append(ep)
    assert all(len(v) == 3 for v in by_env.values())
    for n in (0, 1):
        assert all(ep["ended_by"] == "success" and ep["success_recount"] for ep in by_env[n])
    for n in (2, 3):
        assert all(ep["ended_by"] == "time_out" and ep["steps"] == 60 for ep in by_env[n])
    # the terminal (pre-reset) row of a success episode is seated with the counter at 12
    last = np.nonzero(tr["done"][:, 0])[0][0]
    assert tr["seated_env"][last, 0] and tr["hold_counter"][last, 0] == sr.SEAT_HOLD_STEPS
    assert tr["term_success"][last, 0] and tr["ep_step"][last, 0] == last + 1
    # the row after it is the next episode's first step, back near the plinth
    assert tr["ep_idx"][last + 1, 0] == 1 and tr["ep_step"][last + 1, 0] == 1
    assert abs(tr["cube_pos"][last + 1, 0, 0]) < 0.05
    # geometry and pose columns
    assert np.allclose(tr["hand_sdf"][tr["valid"]], 0.02, atol=1e-5)
    assert np.allclose(tr["hand_force"][tr["valid"]], np.sqrt(12.0), atol=1e-5)
    assert np.allclose(tr["cube_face_tilt_deg"][tr["valid"]], 0.0, atol=1e-2)       # 180 deg flip reads upright
    assert np.all(tr["cube_up_axis"][tr["valid"]] == 2)
    assert np.allclose(tr["tilt_a"][tr["valid"]], 0.0, atol=1e-3)
    assert float(state["conv_dev"]) < 1e-5
    mech = res["episodes"][0]["robot_a"]
    assert mech["max_abs_lean_x_deg"] < 1e-3 and mech["max_base_disp_m"] < 1e-6
    fwd = tr["fwd_a"][0, 0]
    assert abs(math.degrees(math.atan2(fwd[1], fwd[0])) + 90.0) < 1e-3
    # no row past an env's third episode is valid
    assert tr["valid"].sum() == sum(ep["steps"] for ep in res["episodes"])


def test_fake_rollout_buffer_too_short_is_incomplete(sm):
    u, roll, tr, _ = _fake_run(sm, ["reach_hold", "reach_hold"], T=100)
    assert not roll["all_done"] and roll["t_end"] == 100
    s = sr.summarize(tr, 2, 3)["summary"]
    assert not s["complete"] and sr.funding_reading(s)["code"] == "INCOMPLETE"


def test_terrain_num_envs_mirror_is_allowlisted():
    """Job 21463687 stopped on scene.terrain.num_envs (1024 vs 32), a copy of scene.num_envs."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "stand3_replay_allow", Path(__file__).resolve().parents[1] / "scripts/bench/stand3_replay.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert "scene.terrain.num_envs" in m.CONFIG_ALLOWLIST and "scene.num_envs" in m.CONFIG_ALLOWLIST
    assert not any(p.startswith(("rewards", "observations", "actions", "terminations", "events")) for p in m.CONFIG_ALLOWLIST)
