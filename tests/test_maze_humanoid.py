"""maze_explore.py --variant humanoid: the pure parts and the launcher's predeclared rule.

The fall rule equals run_eval's (harness constants), the honest labels (LEARNED gait
arms-turn-turnboth-s0, one checkpoint / SCRIPTED A* + turn-then-walk / ORACLE pose and goal),
controller overrides default to the constructor's values, the verdict of
slurm/repo20260923/cpu_maze_humanoid.sbatch (PASS iff >= 10/12 reached and 0 falls; INCOMPLETE
on any mismatch), and that the launcher's flags are the frozen settings. No MuJoCo episode is run.
"""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
_spec = importlib.util.spec_from_file_location("maze_explore_humanoid_under_test", _REPO / "scripts/bench/maze_explore.py")
me = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(me)

from bhl_robust.eval import random_maze as rm  # noqa: E402

LAUNCHER = _REPO / "slurm/repo20260923/cpu_maze_humanoid.sbatch"
UPSTREAM = _REPO / "external/Berkeley-Humanoid-Lite"
TURNBOTH = "external/Berkeley-Humanoid-Lite/logs/rsl_rl/humanoid/2026-09-25_18-54-44_arms-turn-turnboth-s0/exported/deploy.yaml"
BIPED = "external/Berkeley-Humanoid-Lite/logs/rsl_rl/biped/2026-08-17_09-54-10_dr-default-s0/exported/deploy.yaml"


# ------------------------------------------------------------------ fall rule
def test_fall_constants_are_run_evals():
    from bhl_robust.eval import harness
    assert me.HUMANOID_TILT_LIMIT_RAD == harness.TILT_LIMIT_RAD
    assert me.HUMANOID_MAX_SINK_M == harness.MAX_SINK_M


@pytest.mark.parametrize("tilt,sink,fell", [
    (0.10, 0.00, False), (0.78, 0.00, False), (0.781, 0.00, True),      # strict, as harness: tilt > 0.78
    (0.10, 0.25, False), (0.10, 0.251, True), (0.10, -0.05, False),     # sink > 0.25 m below spawn
    (1.20, 0.40, True),
])
def test_humanoid_fell(tilt, sink, fell):
    assert me.humanoid_fell(tilt, sink) is fell


# --------------------------------------------------------------------- labels
def test_run_name_from_the_export_path():
    assert me.gait_run_name(TURNBOTH) == "arms-turn-turnboth-s0" == me.HUMANOID_RUN
    assert me.gait_run_name(BIPED) == "dr-default-s0"


def test_labels_say_learned_scripted_oracle():
    lab = me.humanoid_labels(me.HUMANOID_RUN)
    assert lab["gait"].startswith("LEARNED gait arms-turn-turnboth-s0 (one checkpoint")
    assert "QUALIFIED" in lab["gait"] and "not a recipe" in lab["gait"]
    assert lab["planner"].startswith("SCRIPTED A*") and "turn-then-walk" in lab["planner"]
    assert lab["pose"].startswith("ORACLE") and lab["goal"].startswith("ORACLE")
    foot = me.humanoid_footer(me.HUMANOID_RUN)
    for part in ("LEARNED gait arms-turn-turnboth-s0 (one checkpoint)", "SCRIPTED A* planner + turn-then-walk", "ORACLE pose and goal"):
        assert part in foot
    assert len(foot) < 200                                   # one footer line at 13 px on a 1600 px frame
    assert "injected error" in me.humanoid_footer(me.HUMANOID_RUN, pose_error=True)
    other = me.humanoid_labels("arms-turn-turnrest-ft-s0")
    assert "QUALIFIED" not in other["gait"] and other["gait"].startswith("LEARNED gait arms-turn-turnrest-ft-s0 (one checkpoint")


# ----------------------------------------------------------------- controller
def test_controller_overrides_default_to_the_constructor():
    class A:
        turn_enter = turn_exit = wz_walk = waypoint_radius = None
    assert me.controller_overrides(A()) == {}
    base = rm.TurnWalkController(cruise=0.3, turn_rate=0.6)
    same = rm.TurnWalkController(cruise=0.3, turn_rate=0.6, **me.controller_overrides(A()))
    assert vars(base) == vars(same)
    a = A()
    a.turn_exit = 0.2
    assert me.controller_overrides(a) == {"turn_exit": 0.2}
    assert "goal_radius" not in me.CONTROLLER_KNOBS          # the judge's radius is not a knob


def test_frozen_settings_are_well_formed():
    f = me.HUMANOID_FROZEN
    assert set(f) == {"cruise", "turn_rate", "inflate", "settle_s", *me.CONTROLLER_KNOBS}
    # the planner rounds the inflation to whole 0.10 m cells: the frozen number must be exactly what runs
    assert abs(f["inflate"] / 0.10 - round(f["inflate"] / 0.10)) < 1e-9
    assert f["inflate"] >= me.HUMANOID_RADIUS_M              # the arms clear the wall band
    assert rm.CELL - rm.WALL_T - 2 * f["inflate"] >= 0.3     # and a free channel is left in a 1.32 m corridor
    assert me.HUMANOID_SCORED == {"n": 6, "m": 6, "extra_openings": 1, "time_limit": 180.0, "seeds": list(range(12, 24))}


# -------------------------------------------------------------------- verdict
def _row(seed, outcome="goal", t=80.0, wall=0, humanoid=True, maze=(6, 6, 1)):
    ok = outcome == "goal"
    r = {"seed": seed, "outcome": outcome, "outcome_class": outcome, "success": ok, "clean_success": ok and wall == 0,
         "completion_s": t if ok else None, "wall_contact_steps": wall,
         "maze": {"n": maze[0], "m": maze[1], "extra_openings": maze[2]},
         "gait": {"deploy": "/x/" + (TURNBOTH if humanoid else BIPED)}}
    if humanoid:
        r["variant"] = "humanoid"
    return r


def _summary(humanoid=True, **over):
    s = {"n": 6, "m": 6, "extra_openings": 1, "time_limit": 180.0}
    s.update(me.HUMANOID_FROZEN if humanoid else me.BIPED_REFERENCE_SETTINGS)
    s.update(over)
    return {"settings": s}


def _arm(outcomes, humanoid=True, seeds=range(12, 24), **kw):
    rows = [_row(s, o, t=60.0 + i, humanoid=humanoid, **kw) for i, (s, o) in enumerate(zip(seeds, outcomes))]
    return rows


def _biped_ok():
    return _arm(["goal"] * 12, humanoid=False), _summary(False)


def test_verdict_pass_at_ten_with_no_fall():
    h = _arm(["goal"] * 10 + ["time_out"] * 2)
    v = me.humanoid_maze_verdict(h, _summary(), *_biped_ok())
    assert v["verdict"] == "PASS" and v["humanoid"]["reached"] == 10 and v["reference_status"] == "COMPLETE"
    assert v["biped_reference"]["reached"] == 12 and v["biped_reference"]["clean"] == 12
    assert v["humanoid"]["median_completion_s"] == sorted(60.0 + i for i in range(10))[5]   # summary.json's convention
    assert "LEARNED" in v["labels"]["humanoid"]["gait"] and v["rule"] == me.HUMANOID_MAZE_RULE


def test_verdict_fails_on_nine_or_on_any_fall():
    assert me.humanoid_maze_verdict(_arm(["goal"] * 9 + ["stuck"] * 3), _summary(), *_biped_ok())["verdict"] == "FAIL"
    v = me.humanoid_maze_verdict(_arm(["goal"] * 11 + ["fall"]), _summary(), *_biped_ok())
    assert v["verdict"] == "FAIL" and v["humanoid"]["falls"] == 1


def test_clean_is_reported_not_gated():
    h = [_row(s, "goal", wall=5) for s in range(12, 24)]
    v = me.humanoid_maze_verdict(h, _summary(), *_biped_ok())
    assert v["verdict"] == "PASS" and v["humanoid"]["clean"] == 0


@pytest.mark.parametrize("mutate", ["missing_seed", "wrong_seeds", "no_summary", "settings", "time_limit", "gait", "maze"])
def test_verdict_incomplete_on_any_mismatch(mutate):
    h, s = _arm(["goal"] * 12), _summary()
    if mutate == "missing_seed":
        h = h[:-1]
    elif mutate == "wrong_seeds":
        h = _arm(["goal"] * 12, seeds=range(0, 12))
    elif mutate == "no_summary":
        s = None
    elif mutate == "settings":
        s = _summary(inflate=0.30)
    elif mutate == "time_limit":
        s = _summary(time_limit=150.0)
    elif mutate == "gait":
        h[3]["gait"]["deploy"] = "/x/" + TURNBOTH.replace("turnboth-s0", "turnrest-ft-s0")
    elif mutate == "maze":
        h = _arm(["goal"] * 12, maze=(5, 5, 2))
    v = me.humanoid_maze_verdict(h, s, *_biped_ok())
    assert v["verdict"] == "INCOMPLETE" and v["problems"]


def test_biped_reference_gaps_never_change_the_verdict():
    h = _arm(["goal"] * 12)
    v = me.humanoid_maze_verdict(h, _summary(), _arm(["goal"] * 11, humanoid=False, seeds=range(12, 23)), _summary(False))
    assert v["verdict"] == "PASS" and v["reference_status"].startswith("INCOMPLETE")
    v = me.humanoid_maze_verdict(h, _summary(), [], None)
    assert v["verdict"] == "PASS" and v["biped_reference"] is None
    # the biped arm must be the published configuration and the biped gait
    v = me.humanoid_maze_verdict(h, _summary(), _arm(["goal"] * 12, humanoid=True), _summary(False))
    assert v["reference_status"].startswith("INCOMPLETE")


def test_verdict_subcommand_never_overwrites(tmp_path):
    import json
    root = tmp_path / "res"
    for name, rows, summ in (("humanoid-hard-6x6", _arm(["goal"] * 12), _summary()),
                             ("biped-hard-6x6", *_biped_ok())):
        d = root / name
        d.mkdir(parents=True)
        for r in rows:
            (d / f"seed{r['seed']}.json").write_text(json.dumps(r))
        (d / "summary.json").write_text(json.dumps(summ))
    assert me.humanoid_verdict_main([str(root)]) == 0
    v = json.loads((root / "verdict.json").read_text())
    assert v["verdict"] == "PASS"
    (root / "verdict.json").write_text(json.dumps({**v, "verdict": "SENTINEL"}))
    me.humanoid_verdict_main([str(root)])
    assert json.loads((root / "verdict.json").read_text())["verdict"] == "SENTINEL"
    # an INCOMPLETE reading is printed, never recorded (a completed rerun can still be judged)
    (root / "humanoid-hard-6x6" / "seed23.json").unlink()
    assert me.humanoid_verdict_main([str(root), "--out", str(root / "v2.json")]) == 0
    assert not (root / "v2.json").exists()


# ------------------------------------------------------------------- launcher
def _header_text():
    lines = [ln[1:].strip() for ln in LAUNCHER.read_text().splitlines() if ln.startswith("#") and not ln.startswith(("#!", "#SBATCH"))]
    return " ".join(lines)


def test_launcher_header_holds_the_predeclared_rule():
    head = re.sub(r"\s+", " ", _header_text())
    assert re.sub(r"\s+", " ", me.HUMANOID_MAZE_RULE) in head
    assert "LEARNED gait arms-turn-turnboth-s0 (one checkpoint)" in head
    assert "SCRIPTED A* planner + turn-then-walk" in head and "ORACLE pose and goal" in head
    assert "PILOT_REST" not in LAUNCHER.read_text()           # no placeholder ships
    assert "Freeze rule" in head and "maze seeds 12-23" in head


def _flag(cmd, name):
    m = re.search(rf"--{name}\s+(\S+)", cmd)
    return m.group(1) if m else None


def test_launcher_flags_are_the_frozen_settings_and_the_declared_block():
    sb = LAUNCHER.read_text()
    hum = re.search(r"^HUMANOID_FLAGS=\((.*?)\)", sb, flags=re.M | re.S).group(1)
    for k, v in me.HUMANOID_FROZEN.items():
        got = _flag(hum, k.replace("_", "-"))
        if v is None:
            assert got is None, k
        else:
            assert got is not None and float(got) == v, k
    assert "--variant humanoid" in hum
    common = re.search(r"^COMMON=\((.*?)\)", sb, flags=re.M | re.S).group(1)
    assert "--seed-start 12 --seeds 12" in common and "--n 6 --m 6 --extra-openings 1 --time-limit 180" in common
    assert "--no-overwrite" in common
    bip = re.search(r"^BIPED_FLAGS=\((.*?)\)", sb, flags=re.M | re.S).group(1)
    for name in ("cruise", "turn-rate", "inflate", "variant", "settle-s"):
        assert _flag(bip, name) is None                      # the biped reference runs the published defaults
    assert "humanoid-verdict" in sb and "results/maze-humanoid-20260928" in sb


# ------------------------------------------------------------- argument guards
def test_policy_is_refused_with_the_humanoid(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["maze_explore.py", "--upstream", str(UPSTREAM), "--cache-dir", str(tmp_path),
                                      "--variant", "humanoid", "--policy", str(tmp_path / "a.onnx")])
    with pytest.raises(SystemExit) as e:
        me.main()
    assert e.value.code == 2


@pytest.mark.skipif(not (UPSTREAM / "logs").is_dir(), reason="upstream checkout (with logs/) not present")
def test_no_overwrite_refuses_before_running(monkeypatch, tmp_path):
    (tmp_path / "seed12.json").write_text("{}")
    for extra in ([], ["--variant", "humanoid"]):
        monkeypatch.setattr(sys, "argv", ["maze_explore.py", "--upstream", str(UPSTREAM), "--cache-dir", str(tmp_path / "c"),
                                          "--seed-start", "12", "--seeds", "2", "--out-dir", str(tmp_path), "--no-overwrite", *extra])
        with pytest.raises(SystemExit) as e:
            me.main()
        assert "refusing to run" in str(e.value.code)


@pytest.mark.skipif(not (UPSTREAM / "logs").is_dir(), reason="upstream checkout (with logs/) not present")
def test_the_default_humanoid_gait_is_the_one_turnboth_export():
    import glob
    hits = glob.glob(str(UPSTREAM / me.HUMANOID_GAIT_GLOB))
    assert len(hits) == 1 and me.gait_run_name(hits[0]) == me.HUMANOID_RUN


# ------------------------------------------------------ map from the sensor's pose
def test_integrate_scan_is_occupancygrid_update_for_a_level_scan_from_the_base():
    import numpy as np
    maze = rm.generate(6, 6, 100, extra_openings=1)
    rng = np.random.default_rng(0)
    angles = np.linspace(-np.pi, np.pi, 108, endpoint=False)
    a, b = rm.OccupancyGrid(maze.bounds()), rm.OccupancyGrid(maze.bounds())
    for _ in range(5):
        x, y, yaw = rng.uniform(0, 5), rng.uniform(0, 5), rng.uniform(-3, 3)
        r = rng.uniform(0.2, 14.0, 108)                     # includes misses beyond the 12 m range
        a.update(x, y, yaw, angles, r, 12.0)
        me.integrate_scan(b, (x, y), yaw + angles, np.minimum(r, 12.0), r < 12.0 - 1e-3)
    assert np.array_equal(a.l, b.l) and a.updates == b.updates


def test_scan_rays_from_sensor_geometry():
    import math
    import numpy as np
    from bhl_robust.eval.team_sensors import ray_pattern
    angles, dirs, _, _ = ray_pattern()
    # level body at the origin, mount 0.12 m forward and 0.72 m up: origin shifted, azimuths = angles, reach = r
    o, az, reach, hits = me.scan_rays_from_sensor(np.zeros(3), np.eye(3), (0.12, 0.0, 0.72), dirs, np.full(108, 2.0), 12.0)
    assert np.allclose(o, [0.12, 0.0]) and np.allclose(np.cos(az), np.cos(angles)) and np.allclose(reach, 2.0) and hits.all()
    # body yawed 90 deg: the forward mount points along +y
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    o, az, _, _ = me.scan_rays_from_sensor(np.array([1.0, 2.0, 0.0]), rz, (0.12, 0.0, 0.72), dirs, np.full(108, 2.0), 12.0)
    assert np.allclose(o, [1.0, 2.12]) and np.allclose(np.cos(az), np.cos(angles + math.pi / 2))
    # pitched forward 0.15 rad: the forward ray (angle 0) travelling 6 m ends below the floor -> not a wall hit,
    # the backward ray rises -> a wall hit; horizontal reach is r cos(pitch)
    p = 0.15
    ry = np.array([[math.cos(p), 0.0, math.sin(p)], [0.0, 1.0, 0.0], [-math.sin(p), 0.0, math.cos(p)]])
    o, az, reach, hits = me.scan_rays_from_sensor(np.zeros(3), ry, (0.0, 0.0, 0.66), dirs, np.full(108, 6.0), 12.0)
    fwd, back = int(np.argmin(np.abs(angles))), int(np.argmin(np.abs(np.abs(angles) - math.pi)))
    assert not hits[fwd] and hits[back] and math.isclose(reach[fwd], 6.0 * math.cos(p), rel_tol=1e-9)
    # a miss (max range) is never a hit
    _, _, _, hits = me.scan_rays_from_sensor(np.zeros(3), np.eye(3), (0.0, 0.0, 0.66), dirs, np.full(108, 12.0), 12.0)
    assert not hits.any()
    # pose error: the estimate replaces the base xy and rotates the mount offset and every azimuth
    o, az, _, _ = me.scan_rays_from_sensor(np.zeros(3), np.eye(3), (0.12, 0.0, 0.72), dirs, np.full(108, 2.0), 12.0,
                                           xy_est=(0.5, 0.0), yaw_err=math.pi / 2)
    assert np.allclose(o, [0.5, 0.12]) and np.allclose(np.cos(az), np.cos(angles + math.pi / 2))
