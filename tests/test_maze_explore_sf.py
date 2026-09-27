"""Pure parts of the maze_explore.py sensor-fusion stress options.

Pose-error application (SF-02 conventions of scripts/bench/inspection_maze.py),
the stall tracker, outcome classification (arrived_not_judged / stuck never
count as reached), the predeclared sweep verdict, the launcher's tags, and the
observation slots EstimatedAttitude overwrites. No MuJoCo episode is run.
"""
import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
_spec = importlib.util.spec_from_file_location("maze_explore_sf_under_test", _REPO / "scripts/bench/maze_explore.py")
me = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(me)


# ------------------------------------------------------------------ PoseError
def test_inactive_pose_error_is_the_identity_and_draws_nothing():
    pe = me.PoseError(3)
    state = pe.rng.bit_generator.state
    xy = np.array([1.25, -0.5])
    out_xy, out_yaw = pe.apply(xy, 0.7)
    assert out_xy is xy and out_yaw == 0.7
    assert not pe.active
    assert pe.rng.bit_generator.state == state
    assert np.array_equal(pe.bias, np.zeros(2))


def test_bias_follows_sf02_draw():
    for seed in range(5):
        pe = me.PoseError(seed, bias_m=0.15)
        ref = np.random.default_rng(10_000 + seed).normal(size=2)
        assert np.allclose(pe.bias, 0.15 * ref / np.linalg.norm(ref))
        assert math.isclose(np.linalg.norm(pe.bias), 0.15)
        xy = np.array([2.0, 3.0])
        out, yaw = pe.apply(xy, 0.1)
        assert np.allclose(out, xy + pe.bias) and yaw == 0.1
        assert np.array_equal(xy, [2.0, 3.0])            # input not mutated
    dirs = {tuple(np.round(me.PoseError(s, bias_m=1.0).bias, 6)) for s in range(5)}
    assert len(dirs) == 5                                 # random direction per seed


def test_noise_stream_matches_inspection_maze_order():
    """inspection_maze: loc_rng = default_rng(10_000+seed); bias draw first (if set); one normal(2) per step."""
    seed, b, s = 7, 0.05, 0.10
    pe = me.PoseError(seed, bias_m=b, noise_m=s)
    rng = np.random.default_rng(10_000 + seed)
    v = rng.normal(size=2)
    bias = b * v / np.linalg.norm(v)
    xy = np.zeros(2)
    for _ in range(50):
        got, _ = pe.apply(xy, 0.0)
        assert np.allclose(got, xy + bias + rng.normal(size=2) * s)
    # noise only: no bias draw, the first step uses the first normal
    pe = me.PoseError(seed, noise_m=s)
    rng = np.random.default_rng(10_000 + seed)
    assert np.allclose(pe.apply(xy, 0.0)[0], rng.normal(size=2) * s)


def test_heading_error_is_a_constant_offset_in_radians():
    pe = me.PoseError(0, yaw_deg=10.0)
    for yaw in (-3.0, 0.0, 1.2):
        xy = np.array([0.3, 0.4])
        out_xy, out_yaw = pe.apply(xy, yaw)
        assert math.isclose(out_yaw - yaw, math.radians(10.0))
        assert np.allclose(out_xy, xy)                    # heading only: position untouched
    assert pe.active and pe.summary()["pose_yaw_deg"] == 10.0


# --------------------------------------------------------------- StallTracker
def test_stall_tracker_measures_time_within_the_radius():
    st = me.StallTracker([0.0, 0.0], 0.0)
    for k in range(1, 501):                               # 20 s at 25 Hz, jitter within 0.1 m
        st.update([0.1 * math.sin(k), 0.05], k * 0.04)
    assert st.current_s == pytest.approx(20.0)
    st.update([0.5, 0.0], 20.04)                          # leaves the disc: stall resets
    assert st.current_s == 0.0 and st.longest_s == pytest.approx(20.0)
    for k in range(1, 26):                                # walking 0.3 m/s never stalls for long
        st.update([0.5 + 0.012 * k * 25, 0.0], 20.04 + k)
    assert st.current_s < 1.0 + 1e-9


# ---------------------------------------------------------- classification
@pytest.mark.parametrize("outcome,arrived,stall,expected", [
    ("goal", True, 30.0, "goal"),
    ("fall", True, 30.0, "fall"),
    ("nonfinite_state", False, 0.0, "nonfinite_state"),
    ("time_out", True, 0.0, "arrived_not_judged"),
    ("time_out", True, 99.0, "arrived_not_judged"),       # arrival claim takes precedence over stuck
    ("time_out", False, 20.0, "stuck"),
    ("time_out", False, 19.9, "time_out"),
])
def test_classify_outcome(outcome, arrived, stall, expected):
    assert me.classify_outcome(outcome, arrived, stall) == expected


# ---------------------------------------------------------------- verdict
def _rows(reached, falls=0, stuck=0, anj=0, n=12):
    rows = []
    for s in range(n):
        if s < reached:
            rows.append({"seed": s, "outcome": "goal", "outcome_class": "goal", "success": True, "clean_success": True})
        elif s < reached + falls:
            rows.append({"seed": s, "outcome": "fall", "outcome_class": "fall", "success": False, "clean_success": False})
        elif s < reached + falls + stuck:
            rows.append({"seed": s, "outcome": "time_out", "outcome_class": "stuck", "success": False, "clean_success": False})
        elif s < reached + falls + stuck + anj:
            rows.append({"seed": s, "outcome": "time_out", "outcome_class": "arrived_not_judged", "success": False, "clean_success": False})
        else:
            rows.append({"seed": s, "outcome": "time_out", "outcome_class": "time_out", "success": False, "clean_success": False})
    return rows


def _all_tags():
    tags = []
    for name in ("heading", "imu"):
        tags += [me.SF_SWEEP[name]["control"]] + [t for _, t in me.SF_SWEEP[name]["levels"]]
    for name in ("bias", "noise"):
        tags += [t for _, t in me.SF_SWEEP[name]["levels"]]
    return tags + me.SF_SWEEP["exploratory"]


def _sweep(**over):
    rows = {t: _rows(12) for t in _all_tags()}
    rows.update(over)
    return rows


def test_verdict_all_pass():
    v = me.sf_sweep_verdict(_sweep())
    assert v["heading"]["verdict"] == "MEASURED" and v["heading"]["tolerance_deg"] == 10.0
    assert v["imu"]["verdict"] == "MEASURED" and v["imu"]["budget_ms"] == 60.0
    assert set(v["bias"].values()) == {"PASS"} and v["noise"] == {"noise0.10": "PASS"}
    assert not v["heading"]["non_monotone"] and not v["imu"]["non_monotone"]


def test_heading_needs_a_perfect_control_and_takes_the_largest_passing_level():
    v = me.sf_sweep_verdict(_sweep(yaw0=_rows(11)))
    assert v["heading"]["verdict"] == "INVALID_CONTROL" and v["heading"]["tolerance_deg"] is None
    v = me.sf_sweep_verdict(_sweep(yaw10=_rows(9)))
    assert v["heading"]["tolerance_deg"] == 3.0
    v = me.sf_sweep_verdict(_sweep(yaw3=_rows(11, falls=1)))           # a fall fails the level
    assert v["heading"]["tolerance_deg"] == 10.0 and v["heading"]["non_monotone"]
    v = me.sf_sweep_verdict(_sweep(yaw1=_rows(9), yaw3=_rows(9), yaw10=_rows(9)))
    assert v["heading"]["verdict"] == "BELOW_SMALLEST_LEVEL" and v["heading"]["tolerance_deg"] is None


def test_imu_control_threshold_is_eleven():
    v = me.sf_sweep_verdict(_sweep(**{"imu-est0": _rows(11), "imu-est40": _rows(9), "imu-est60": _rows(0, falls=12)}))
    assert v["imu"]["verdict"] == "MEASURED" and v["imu"]["budget_ms"] == 30.0
    v = me.sf_sweep_verdict(_sweep(**{"imu-est0": _rows(10)}))
    assert v["imu"]["verdict"] == "INVALID_CONTROL" and v["imu"]["budget_ms"] is None


def test_stuck_and_arrived_not_judged_never_count_as_reached():
    v = me.sf_sweep_verdict(_sweep(**{"bias0.30": _rows(9, stuck=1, anj=2), "bias0.15": _rows(10, anj=2)}))
    assert v["bias"]["bias0.30"] == "FAIL" and v["bias"]["bias0.15"] == "PASS"
    c = v["counts"]["bias0.30"]
    assert (c["reached"], c["stuck"], c["arrived_not_judged"], c["time_out"]) == (9, 1, 2, 0)
    v = me.sf_sweep_verdict(_sweep(**{"bias0.05": _rows(11, falls=1)}))
    assert v["bias"]["bias0.05"] == "FAIL"


def test_missing_seeds_are_incomplete_not_pass():
    v = me.sf_sweep_verdict(_sweep(**{"noise0.10": _rows(11, n=11), "yaw3": _rows(11, n=11)}))
    assert v["noise"]["noise0.10"] == "INCOMPLETE"
    assert v["heading"]["verdict"] == "INCOMPLETE" and v["heading"]["tolerance_deg"] is None
    rows = _sweep()
    del rows["imu-est30"]
    assert me.sf_sweep_verdict(rows)["imu"]["verdict"] == "INCOMPLETE"


def test_launcher_tags_match_the_predeclared_sweep():
    sb = (_REPO / "slurm/repo20260923/cpu_maze_sf.sbatch").read_text()
    tags = re.findall(r"^\s*\d+\)\s+run\s+(\S+)", sb, flags=re.M)
    assert sorted(tags) == sorted(_all_tags())
    n_tasks = int(re.search(r"#SBATCH --array=0-(\d+)", sb).group(1)) + 1
    assert n_tasks == len(tags)
    assert "--seeds 12" in sb and "--n 6 --m 6 --extra-openings 1 --time-limit 180" in sb


# ------------------------------------------------ attitude substitution slots
def test_estimated_attitude_overwrites_quat_and_gyro_only_in_the_biped_layout():
    """MultiRunner.observe = [quat 4, gyro 3, jpos n, jvel n, mode 1, cmd 3]; RlController.update parses
    [0:4] and [4:7] for any num_actions, so the 12-DoF biped vector (35) takes the same substitution."""
    sys.path.insert(0, str(_REPO / "scripts/bench"))
    from inspection_maze import EstimatedAttitude
    imu = object.__new__(EstimatedAttitude)
    q_est, g_est = np.array([0.9, 0.1, -0.1, 0.4]), np.array([0.01, -0.02, 0.03])
    imu.every, imu._latest = 10, (q_est, g_est)
    obs = np.arange(4 + 3 + 12 + 12 + 1 + 3, dtype=np.float32)
    out = imu.apply(obs, None, 0.04)
    assert np.allclose(out[0:4], q_est) and np.allclose(out[4:7], g_est)
    assert np.array_equal(out[7:], obs[7:]) and np.array_equal(obs, np.arange(35, dtype=np.float32))
    imu._latest = None                                     # before alignment the oracle passes through
    assert imu.apply(obs, None, 0.04) is obs
