"""NavGym v4 capture-pose map integration (N0, docs/SOLUTIONS_2026-10-01.md): the predeclared verdicts of
scripts/bench/navgym_capture_pose.py on synthetic per-seed JSONs, the launcher's header and flags, and the
maze_explore.py guard. No MuJoCo episode is run.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cp = _load("navgym_capture_pose_under_test", "scripts/bench/navgym_capture_pose.py")
me = _load("maze_explore_capture_pose_under_test", "scripts/bench/maze_explore.py")
LAUNCHER = _REPO / "slurm/repo20260923/cpu_navgym_capture_pose.sbatch"
UPSTREAM = _REPO / "external/Berkeley-Humanoid-Lite"


def _row(seed, success, flag, outcome=None, walls=0, layout=None):
    r = {"seed": seed, "success": success, "clean_success": success and walls == 0,
         "outcome": outcome or ("goal" if success else "time_out"), "wall_contact_steps": walls,
         "maze": {"layout_sha256": layout or f"L{seed}"}, "policy": "actor.onnx"}
    if flag:
        r["policy_map_integration"] = {"mode": "capture_pose", "updates_at_capture_pose": 100, "updates_at_loop_top_pose": 0}
    return r


def _write_arm(d: Path, rows, summary=True):
    d.mkdir(parents=True, exist_ok=True)
    for r in rows:
        (d / f"seed{r['seed']}.json").write_text(json.dumps(r))
    if summary:
        (d / "summary.json").write_text(json.dumps({"n": len(rows)}))


SEEDS = list(range(9120, 9144))


def _screen(tmp_path, s5_base, s5_fix, s6_base=None, s6_fix=None, fix_outcomes=None, fix_walls=None):
    """sN_base / sN_fix: sets of seeds that reach the goal."""
    s6_base = set(SEEDS[:20]) if s6_base is None else s6_base
    s6_fix = set(SEEDS[:20]) if s6_fix is None else s6_fix
    for actor, base, fix in (("armV4-s5", s5_base, s5_fix), ("armV4-s6", s6_base, s6_fix)):
        _write_arm(tmp_path / f"{actor}-looptop", [_row(s, s in base, False) for s in SEEDS])
        _write_arm(tmp_path / f"{actor}-capture",
                   [_row(s, s in fix, True, outcome=(fix_outcomes or {}).get((actor, s)), walls=(fix_walls or {}).get((actor, s), 0))
                    for s in SEEDS])
    return cp.screen_verdict(tmp_path)


def test_mcnemar_exact_p():
    assert cp.mcnemar_exact_p(0, 0) == 1.0
    assert cp.mcnemar_exact_p(5, 0) == pytest.approx(0.0625)
    assert cp.mcnemar_exact_p(4, 1) == pytest.approx(0.375)
    assert cp.mcnemar_exact_p(1, 4) == cp.mcnemar_exact_p(4, 1)


def test_screen_pass_at_plus_four_with_one_lost(tmp_path):
    base = set(SEEDS[:12])
    fix = (base - {SEEDS[0]}) | set(SEEDS[12:17])           # won 5, lost 1 -> net +4
    v = _screen(tmp_path, base, fix)
    assert v["verdict"] == "PASS", v
    s5 = v["per_actor"]["armV4-s5"]
    assert (len(s5["won"]), len(s5["lost"]), s5["net"]) == (5, 1, 4)


@pytest.mark.parametrize("won,lost", [(3, 0), (6, 2)])
def test_screen_negative_below_plus_four_or_two_lost(tmp_path, won, lost):
    base = set(SEEDS[:12])
    fix = (base - set(SEEDS[:lost])) | set(SEEDS[12:12 + won])
    assert _screen(tmp_path, base, fix)["verdict"] == "NEGATIVE"


def test_screen_negative_when_s6_nets_minus_two(tmp_path):
    base5, fix5 = set(SEEDS[:12]), set(SEEDS[:17])
    v = _screen(tmp_path, base5, fix5, s6_base=set(SEEDS[:20]), s6_fix=set(SEEDS[:18]))
    assert v["verdict"] == "NEGATIVE" and v["clauses"]["s6_net"] is False


def test_screen_negative_on_a_fall_or_more_wall_contacts(tmp_path):
    base5, fix5 = set(SEEDS[:12]), set(SEEDS[:17])
    v = _screen(tmp_path / "a", base5, fix5, fix_outcomes={("armV4-s6", SEEDS[23]): "fall"})
    assert v["verdict"] == "NEGATIVE" and v["clauses"]["no_falls"] is False
    v = _screen(tmp_path / "b", base5, fix5, fix_walls={("armV4-s5", SEEDS[0]): 3})
    assert v["verdict"] == "NEGATIVE" and v["clauses"]["wall_contacts_not_increased"] is False


def test_screen_incomplete_on_missing_or_mislabelled_episodes(tmp_path):
    base5, fix5 = set(SEEDS[:12]), set(SEEDS[:17])
    _screen(tmp_path, base5, fix5)
    (tmp_path / "armV4-s5-capture" / "seed9130.json").unlink()
    v = cp.screen_verdict(tmp_path)
    assert v["verdict"] == "INCOMPLETE" and any("missing seeds [9130]" in p for p in v["problems"])
    # the flag must be recorded on every capture episode and absent from every default episode
    _write_arm(tmp_path / "armV4-s5-capture", [_row(s, s in fix5, False) for s in SEEDS])
    v = cp.screen_verdict(tmp_path)
    assert v["verdict"] == "INCOMPLETE" and any("not recorded" in p for p in v["problems"])
    _write_arm(tmp_path / "armV4-s5-capture", [_row(s, s in fix5, True) for s in SEEDS])
    _write_arm(tmp_path / "armV4-s6-looptop", [_row(s, True, True) for s in SEEDS])
    v = cp.screen_verdict(tmp_path)
    assert v["verdict"] == "INCOMPLETE" and any("override" in p for p in v["problems"])


def test_screen_incomplete_when_paired_layouts_differ(tmp_path):
    base5, fix5 = set(SEEDS[:12]), set(SEEDS[:17])
    _screen(tmp_path, base5, fix5)
    _write_arm(tmp_path / "armV4-s6-capture", [_row(s, s in set(SEEDS[:20]), True, layout="other") for s in SEEDS])
    v = cp.screen_verdict(tmp_path)
    assert v["verdict"] == "INCOMPLETE" and any("different layouts" in p for p in v["problems"])


FRESH = list(range(60000, 60012))


def _fresh(tmp_path, s5_goals, s6_goals, s6_outcome=None, summaries=True):
    for actor, goals in (("armV4-s5", s5_goals), ("armV4-s6", s6_goals)):
        _write_arm(tmp_path / f"{actor}-capture",
                   [_row(s, k < goals, True, outcome=s6_outcome if (actor == "armV4-s6" and k == 11) else None)
                    for k, s in enumerate(FRESH)], summary=summaries)
    _write_arm(tmp_path / "astar", [_row(s, True, False) for s in FRESH])
    return cp.fresh_verdict(tmp_path)


def test_fresh_rule_is_ten_of_twelve_each_with_no_fall(tmp_path):
    assert _fresh(tmp_path / "a", 10, 10)["verdict"] == "PASS"
    assert _fresh(tmp_path / "b", 10, 9)["verdict"] == "NEGATIVE"
    assert _fresh(tmp_path / "c", 12, 11, s6_outcome="fall")["verdict"] == "NEGATIVE"
    v = _fresh(tmp_path / "d", 12, 12, summaries=False)
    assert v["verdict"] == "INCOMPLETE"
    assert _fresh(tmp_path / "e", 12, 12)["astar_reference"] == {"goals": 12, "n": 12}


def test_cli_never_overwrites_and_does_not_record_incomplete(tmp_path, capsys):
    root = tmp_path / "screen"
    root.mkdir()
    assert cp.main(["screen", str(root)]) == 0                 # nothing there: INCOMPLETE, printed only
    assert not (root / "verdict.json").exists()
    (root / "verdict.json").write_text(json.dumps({"verdict": "NEGATIVE", "detail": "kept"}))
    assert cp.main(["screen", str(root)]) == 0
    assert json.loads((root / "verdict.json").read_text())["detail"] == "kept"
    assert "not overwritten" in capsys.readouterr().out


def _header_text():
    lines = [ln[1:].strip() for ln in LAUNCHER.read_text().splitlines() if ln.startswith("#") and not ln.startswith(("#!", "#SBATCH"))]
    return re.sub(r"\s+", " ", " ".join(lines))


def test_launcher_header_holds_the_predeclared_rules():
    head = _header_text()
    assert "PREDECLARED (2026-10-01, before any of these episodes has run" in head
    for s in ("maze seeds 9120-9143", "nets >= +4", "<= 1 lost", "nets >= -1", "0 falls with the flag",
              "wall-contact steps with the flag are <= its total without", "McNemar exact p reported, not gated",
              "run only if the screen PASSES", "maze seeds 60000-60011", ">= 10/12 goals with 0 falls",
              "LEARNED biped gait", "LEARNED NavGym v4 actors", "ORACLE: pose and goal coordinate"):
        assert s in head, s
    assert cp.SCREEN_RULE["seeds"] == list(range(9120, 9144)) and cp.FRESH_RULE["seeds"] == list(range(60000, 60012))
    assert (cp.SCREEN_RULE["s5_net_min"], cp.SCREEN_RULE["s5_lost_max"], cp.SCREEN_RULE["s6_net_min"]) == (4, 1, -1)
    assert cp.FRESH_RULE["min_goals"] == 10


def test_launcher_arms_and_seed_blocks():
    sb = LAUNCHER.read_text()
    body = sb.split("set -euo pipefail", 1)[1]
    common = re.search(r"^COMMON=\((.*?)\)", body, flags=re.M | re.S).group(1)
    assert "--n 6 --m 6 --extra-openings 1 --time-limit 180" in common and "--no-overwrite" in common
    assert body.count("--seeds 24 --seed-start 9120") == 2          # each actor: without and with the flag
    assert body.count("--seeds 12 --seed-start 60000") == 2         # the fresh actor loop and the A* reference
    assert body.count("--policy-capture-pose") == 2                 # the screen's flag arm and the fresh arm
    assert "40000" not in body and "50000" not in body
    assert 'if [ "$SCREEN" != "PASS" ]' in body


def test_capture_pose_flag_needs_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["maze_explore.py", "--upstream", str(UPSTREAM), "--cache-dir", str(tmp_path),
                                      "--policy-capture-pose"])
    with pytest.raises(SystemExit) as e:
        me.main()
    assert e.value.code == 2
