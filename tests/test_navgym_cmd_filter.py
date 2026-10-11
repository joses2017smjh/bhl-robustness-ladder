"""NavGym v5 command filter (2026-10-07): the low-pass is the gym's own lag, it removes control-rate chatter and passes
sustained turns; the predeclared dev and confirmation verdicts of scripts/bench/navgym_cmd_filter.py on synthetic
per-seed JSONs; the launcher's frozen header and seed blocks; the maze_explore.py guards. No MuJoCo episode is run."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from bhl_robust.eval.cmd_filter import (CMD_FILTER_MODES, DEFAULT_TAU_S, GAIT_MEASURED_TAU_S, GYM_NOMINAL_TAU_S,  # noqa: E402
                                        CommandLowPass, CommandStats, make_filter)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cf = _load("navgym_cmd_filter_under_test", "scripts/bench/navgym_cmd_filter.py")
LAUNCHER = _REPO / "slurm/repo20260923/cpu_navgym_cmd_filter.sbatch"
UPSTREAM = _REPO / "external/Berkeley-Humanoid-Lite"
DT = 0.04


# ------------------------------------------------------------------ the filter
def test_tau_is_the_gym_nominal_lag_minus_the_measured_gait_lag():
    from bhl_robust.navgym.env import Dynamics
    assert GYM_NOMINAL_TAU_S == Dynamics().tau == 0.25
    assert DEFAULT_TAU_S == pytest.approx(round(GYM_NOMINAL_TAU_S - GAIT_MEASURED_TAU_S, 1)) == pytest.approx(0.2)


def test_low_pass_is_the_gym_lag_update():
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, size=(200, 3))
    f = CommandLowPass(DT, 0.2, channels=(0, 2))
    alpha = DT / max(0.2, DT)
    x = np.zeros(3)
    for k in range(200):
        out = f(u[k])
        x[[0, 2]] += alpha * (u[k, [0, 2]] - x[[0, 2]])
        assert out[0] == pytest.approx(x[0]) and out[2] == pytest.approx(x[2])
        assert out[1] == u[k, 1]                                   # an unfiltered channel passes through


def test_wz_mode_leaves_vx_untouched_and_post_brake_filters_both():
    assert make_filter("none", DT) is None
    assert make_filter("wz-lpf", DT).channels == (2,)
    assert make_filter("lpf-post-brake", DT).channels == (0, 2)
    f = make_filter("wz-lpf", DT)
    out = f([0.35, 0.0, 1.0])
    assert out[0] == 0.35 and out[2] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        make_filter("median", DT)
    with pytest.raises(ValueError):
        CommandLowPass(DT, 0.0)
    assert CMD_FILTER_MODES == ("none", "wz-lpf", "lpf-post-brake")


def test_control_rate_chatter_is_removed_and_a_sustained_turn_passes():
    f = CommandLowPass(DT, 0.2)
    outs = [f([0.3, 0.0, 1.0 if k % 2 else -1.0])[2] for k in range(200)]
    # alternating +-1 every step: steady-state ripple alpha / (2 - alpha) = 0.111 instead of 1.0
    assert max(abs(o) for o in outs[50:]) <= 0.12
    g = CommandLowPass(DT, 0.2)
    turn = [g([0.0, 0.0, 0.6])[2] for _ in range(40)]           # 1.6 s of a constant 0.6 rad/s
    assert turn[-1] == pytest.approx(0.6, rel=0.01) and turn[21] >= 0.99 * 0.6


def test_command_stats_count_flips_and_saturation():
    s = CommandStats(DT, 1.0)
    for w in (1.0, -1.0, 1.0, 0.5, 0.0, -0.2):
        s.update(w)
    out = s.summary()
    assert out["steps"] == 6 and out["saturated_share"] == pytest.approx(3 / 6, abs=1e-4)
    assert out["flips_per_s"] == pytest.approx(2 / (6 * DT), abs=1e-3)   # +1 -> -1 -> +1; 0 never counts as a flip


# ------------------------------------------------------------------ synthetic episodes
def _ep(seed, actor, cond, arm, outcome="goal", walls=0, layout=None):
    r = {"seed": seed, "outcome": outcome, "success": outcome == "goal", "clean_success": outcome == "goal" and walls == 0,
         "completion_s": 60.0 if outcome == "goal" else None, "elapsed_s": 60.0, "wall_contact_steps": walls,
         "maze": {"n": 6, "m": 6, "extra_openings": 1, "layout_sha256": layout or f"L{seed}"},
         "sensor_mode": cf.CONDITIONS[cond]}
    if actor is not None:
        r["policy"] = f"/x/results/navgym-v5-20261002/{actor}/actor.onnx"
        r["policy_map_integration"] = {"mode": "capture_pose"}
        r["cmd_filter"] = {"mode": arm, "tau_s": None if arm == "none" else 0.2,
                           "gait_wz": {"flips_per_s": 0.5, "saturated_share": 0.1}, "actor_wz": {"flips_per_s": 15.0},
                           "max_tilt_rad": 0.2}
    else:
        r["policy"] = None
    return r


def _write(d: Path, rows):
    d.mkdir(parents=True, exist_ok=True)
    for r in rows:
        (d / f"seed{r['seed']}.json").write_text(json.dumps(r))


def _reference(tmp_path):
    ref = tmp_path / "ref"
    for actor in cf.ACTORS:
        for cond in cf.CONDITIONS:
            _write(ref / f"{actor}-{cond}", [_ep(s, actor, cond, "x", outcome="fall" if s == 72003 else "goal")
                                             for s in cf.DEV_SEEDS])
    return ref


def _dev(tmp_path, outcomes=None, skip=None, layout_bad=None):
    """outcomes: {(arm, actor, cond, seed): outcome}; every other episode reaches its goal."""
    root = tmp_path / "dev"
    for arm in cf.ARMS:
        for actor in cf.ACTORS:
            for cond in cf.CONDITIONS:
                rows = [_ep(s, actor, cond, arm, outcome=(outcomes or {}).get((arm, actor, cond, s), "goal"),
                            layout="BAD" if layout_bad == (arm, actor, cond, s) else None)
                        for s in cf.DEV_SEEDS if skip != (arm, actor, cond, s)]
                _write(cf.cell_dir(root, arm, actor, cond), rows)
    return root


def test_dev_selects_the_tie_break_arm_when_both_qualify(tmp_path):
    v = cf.dev_verdict(_dev(tmp_path), _reference(tmp_path))
    assert v["verdict"] == "SELECTED" and v["selected"] == "wz-lpf" and not v["problems"]
    cell = v["arms"]["wz-lpf"]["cells"]["armV5-s8-nominal"]
    assert cell["vs_2026_10_06"] == {"goals_won": 1, "goals_lost": 0, "reference_falls": 1}


def test_dev_one_fall_disqualifies_an_arm(tmp_path):
    v = cf.dev_verdict(_dev(tmp_path, outcomes={("wz-lpf", "armV5-s9", "drop35", 72010): "fall"}), _reference(tmp_path))
    assert v["arms"]["wz-lpf"]["qualifies"] is False and v["selected"] == "lpf-post-brake"


def test_dev_more_goals_wins_and_a_short_actor_disqualifies(tmp_path):
    out = {("wz-lpf", "armV5-s8", "nominal", s): "time_out" for s in cf.DEV_SEEDS[:3]}
    v = cf.dev_verdict(_dev(tmp_path, outcomes=out), _reference(tmp_path))
    assert v["selected"] == "lpf-post-brake"                      # 285 vs 288 goals
    out = {("lpf-post-brake", "armV5-s10", "drop35", s): "time_out" for s in cf.DEV_SEEDS[:13]}   # 35/48 < 36
    v = cf.dev_verdict(_dev(tmp_path / "b", outcomes=out), _reference(tmp_path))
    assert v["arms"]["lpf-post-brake"]["actors_meet_goal_clauses"]["armV5-s10"] is False
    assert v["selected"] == "wz-lpf"


def test_dev_negative_when_no_arm_qualifies(tmp_path):
    out = {("wz-lpf", "armV5-s8", "nominal", 72001): "fall", ("lpf-post-brake", "armV5-s8", "nominal", 72001): "fall"}
    v = cf.dev_verdict(_dev(tmp_path, outcomes=out), _reference(tmp_path))
    assert v["verdict"] == "DEV NEGATIVE" and v["selected"] is None


def test_dev_incomplete_on_a_missing_or_unpaired_episode(tmp_path):
    v = cf.dev_verdict(_dev(tmp_path, skip=("lpf-post-brake", "armV5-s9", "nominal", 72047)), _reference(tmp_path))
    assert v["verdict"] == "INCOMPLETE" and v["selected"] is None
    v = cf.dev_verdict(_dev(tmp_path / "b", layout_bad=("wz-lpf", "armV5-s8", "drop35", 72002)), _reference(tmp_path))
    assert v["verdict"] == "INCOMPLETE" and any("layout differs" in p for p in v["problems"])


def test_dev_incomplete_on_a_mislabelled_filter(tmp_path):
    root = _dev(tmp_path)
    f = cf.cell_dir(root, "wz-lpf", "armV5-s8", "nominal") / "seed72000.json"
    r = json.loads(f.read_text())
    r["cmd_filter"]["tau_s"] = 0.3
    f.write_text(json.dumps(r))
    v = cf.dev_verdict(root, _reference(tmp_path))
    assert v["verdict"] == "INCOMPLETE" and any("tau" in p for p in v["problems"])


def _confirm(tmp_path, arm="wz-lpf", outcomes=None, astar=True):
    root = tmp_path / "confirm"
    for actor in cf.ACTORS:
        for cond in cf.CONDITIONS:
            _write(cf.cell_dir(root, arm, actor, cond),
                   [_ep(s, actor, cond, arm, outcome=(outcomes or {}).get((actor, cond, s), "goal")) for s in cf.CONFIRM_SEEDS])
    if astar:
        for cond in cf.CONDITIONS:
            _write(cf.cell_dir(root, "astar", "astar", cond), [_ep(s, None, cond, None) for s in cf.CONFIRM_SEEDS])
    return root


def _dev_file(tmp_path, selected="wz-lpf", final=True, verdict="SELECTED"):
    p = tmp_path / "dev_verdict.json"
    p.write_text(json.dumps({"verdict": verdict, "selected": selected, "final": final}))
    return p


def test_confirm_pass_and_one_fall_is_negative(tmp_path):
    assert cf.confirm_verdict(_confirm(tmp_path), _dev_file(tmp_path))["verdict"] == "PASS"
    v = cf.confirm_verdict(_confirm(tmp_path / "b", outcomes={("armV5-s10", "drop35", 78040): "fall"}), _dev_file(tmp_path))
    assert v["verdict"] == "NEGATIVE" and v["per_actor"]["armV5-s10"]["meets"] is False


def test_confirm_goal_clauses_at_their_boundaries(tmp_path):
    ok = {("armV5-s9", "nominal", s): "time_out" for s in cf.CONFIRM_SEEDS[:8]}                 # 40/48: meets
    assert cf.confirm_verdict(_confirm(tmp_path, outcomes=ok), _dev_file(tmp_path))["verdict"] == "PASS"
    short = {("armV5-s9", "drop35", s): "time_out" for s in cf.CONFIRM_SEEDS[:13]}              # 35/48: fails
    assert cf.confirm_verdict(_confirm(tmp_path / "b", outcomes=short), _dev_file(tmp_path))["verdict"] == "NEGATIVE"


def test_confirm_incomplete_without_a_final_selection_or_the_reference(tmp_path):
    assert cf.confirm_verdict(_confirm(tmp_path), _dev_file(tmp_path, final=False))["verdict"] == "INCOMPLETE"
    assert cf.confirm_verdict(_confirm(tmp_path), _dev_file(tmp_path, selected=None, verdict="DEV NEGATIVE"))["verdict"] == "INCOMPLETE"
    assert cf.confirm_verdict(_confirm(tmp_path / "b", astar=False), _dev_file(tmp_path))["verdict"] == "INCOMPLETE"
    # episodes run with the other arm do not count for the selected one
    assert cf.confirm_verdict(_confirm(tmp_path / "c", arm="lpf-post-brake"), _dev_file(tmp_path))["verdict"] == "INCOMPLETE"


def test_cli_never_overwrites_and_does_not_record_incomplete(tmp_path, capsys):
    root = _dev(tmp_path, skip=("wz-lpf", "armV5-s8", "nominal", 72000))
    ref = _reference(tmp_path)
    assert cf.main(["dev-verdict", str(root), "--reference", str(ref)]) == 0
    assert not (root / "verdict.json").exists()                   # INCOMPLETE without --final: printed only
    assert cf.main(["dev-verdict", str(root), "--reference", str(ref), "--final"]) == 0
    first = (root / "verdict.json").read_text()
    assert json.loads(first)["verdict"] == "INCOMPLETE" and json.loads(first)["final"] is True
    assert cf.main(["dev-verdict", str(root), "--reference", str(ref), "--final"]) == 0
    assert (root / "verdict.json").read_text() == first
    assert cf.main(["selected-arm", str(root / "verdict.json")]) == 1


# ------------------------------------------------------------------ launcher and guards
def _header(path):
    lines = []
    for line in path.read_text().splitlines():
        if line.startswith("set -euo"):
            break
        if line.startswith("#") and not line.startswith("#!") and not line.startswith("#SBATCH"):
            lines.append(line.lstrip("#").strip())
    return re.sub(r"\s+", " ", " ".join(lines))


def test_launcher_header_holds_the_frozen_rules_verbatim():
    h = _header(LAUNCHER)
    assert cf.DEV_RULE in h and cf.CONFIRM_RULE in h


def test_launcher_seed_blocks_and_arms():
    body = LAUNCHER.read_text()
    assert "learned_cell" in body and '72000 "$ROOT/dev"' in body and '78000 "$ROOT/confirm"' in body
    assert "--seed-start 78000" in body                            # the A* reference on the confirmation mazes
    assert "ARMS=(wz-lpf lpf-post-brake)" in body and "--policy-cmd-tau 0.2" in body
    for actor, cond, seed in cf.REPRO:
        assert f"{actor}:{cond}:{seed}" in body
    assert cf.CONFIRM_SEEDS == tuple(range(78000, 78048)) and cf.DEV_SEEDS == tuple(range(72000, 72048))
    assert not set(cf.CONFIRM_SEEDS) & set(cf.DEV_SEEDS)


@pytest.mark.parametrize("extra,message", [(["--policy-cmd-filter", "wz-lpf"], "it needs --policy"),
                                           (["--policy-cmd-tau", "0"], "--policy-cmd-tau must be > 0")])
def test_maze_explore_guards(monkeypatch, tmp_path, capsys, extra, message):
    me = _load("maze_explore_cmd_filter_under_test", "scripts/bench/maze_explore.py")
    argv = ["maze_explore.py", "--upstream", str(UPSTREAM), "--cache-dir", str(tmp_path)] + extra
    if extra[0] == "--policy-cmd-tau":
        argv += ["--policy", str(tmp_path / "actor.onnx"), "--policy-cmd-filter", "wz-lpf"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as e:
        me.main()
    assert e.value.code == 2 and message in capsys.readouterr().err
