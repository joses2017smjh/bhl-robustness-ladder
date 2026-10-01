"""Mission 7 gate predicates (scripts/mission7_gates.py) and their three call sites.

Pins the 2026-10-01 corrections (docs/SOLUTIONS_2026-10-01.md section 3, M1):
the route gate counts successes, so 21398514's saved summaries read FAIL; a
replay gate's crossings count real clears read from the raw trace, so 21405537
and 21405541 read 4/7 and 1/7 instead of 7/7 hand-backs; the follow-up's
embedded summarize and the route-handoff probe both use them.  A trace that is
partial, in another layout, stripped of samples or not the trace of its gate's
result.json raises instead of counting, and the follow-up reports null for it
(or for a failed mission7_gates import) while still writing the verdict.  No
MuJoCo episode runs here.  The saved-trace tests stream the untracked raw traces
and skip when they are not on disk.
"""

from __future__ import annotations

import ast
import gzip
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import mission7_gates as gates  # noqa: E402
from bhl_robust.mission.layout import generate  # noqa: E402

CAMPAIGN = _REPO / "results/mission7-campaign-20260923"
ROUTE_EVAL = _REPO / "results/mission7-approach-followup-20260922/route-eval-cn-c22-v2/result.json"
VERIFY = _REPO / "results/solutions-20260930/verify-mission7/verify_crossings.json"
FOLLOWUP = _REPO / "slurm/repo20260923/m7_replay_gate_followup.sbatch"
PLATE_SAFE = _REPO / "scripts/mission7_plate_safe.py"
PROBE = _REPO / "scripts/mission7_route_handoff_probe.py"
DT = .04


# ---- route gate: mission7_plate_safe.gate() --------------------------------------------------------

def _plate_safe():
    # the runner's third-party import chain; a broken import of the repo's own modules still fails
    for module in ("mujoco", "torch", "onnxruntime", "omegaconf", "rsl_rl", "tensordict",
                   "berkeley_humanoid_lite_lowlevel"):
        pytest.importorskip(module)
    import mission7_plate_safe
    return mission7_plate_safe


def _stage(successes, episodes=16):
    return {"episodes": episodes, "successes": successes}


def test_route_stage_passed_21398514_without_the_simulator_stack():
    # the predicate gate() applies, through mission7_gates alone: this cannot skip where MuJoCo is absent
    routes = json.loads(ROUTE_EVAL.read_text())["routes"]
    assert not gates.route_stage_passed(routes["doors"]) and not gates.route_stage_passed(routes["transport"])


@pytest.mark.parametrize("successes, episodes, passed", [(16, 16, True), (15, 16, False), (1, 1, False), (0, 16, False)])
def test_route_stage_passed_counts_successes(successes, episodes, passed):
    assert gates.route_stage_passed(_stage(successes, episodes)) is passed


def test_route_gate_21398514_reads_fail():
    saved = json.loads(ROUTE_EVAL.read_text())
    routes = saved["routes"]
    assert (routes["doors"]["episodes"], routes["doors"]["successes"]) == (16, 1)
    assert (routes["transport"]["episodes"], routes["transport"]["successes"]) == (16, 0)
    assert _plate_safe().gate(saved["replay"], routes) == {
        "replay_10_no_falls": False, "doors_16": False, "transport_16": False}


@pytest.mark.parametrize("doors, transport", [(16, 16), (15, 16), (16, 15), (0, 0)])
def test_route_gate_counts_successes(doors, transport):
    gate = _plate_safe().gate({"no_fall": 10}, {"doors": _stage(doors), "transport": _stage(transport)})
    assert gate == {"replay_10_no_falls": True, "doors_16": doors == 16, "transport_16": transport == 16}


def test_route_gate_smoke_run_cannot_pass():
    gate = _plate_safe().gate({"no_fall": 1}, {"doors": _stage(1, 1), "transport": _stage(1, 1)})
    assert gate == {"replay_10_no_falls": False, "doors_16": False, "transport_16": False}


def test_plate_safe_result_json_uses_gate():
    source = PLATE_SAFE.read_text()
    assert '["episodes"] >= 16' not in source
    writes = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == "write"
              and isinstance(node.args[0], ast.BinOp) and getattr(node.args[0].right, "value", None) == "result.json"]
    assert len(writes) == 1
    entries = {key.value: value for key, value in zip(writes[0].args[1].keys, writes[0].args[1].values)}
    assert isinstance(entries["gate"], ast.Call) and entries["gate"].func.id == "gate"
    assert "gate_rule" in entries


# ---- route gate: the handoff probe's gate block --------------------------------------------------

def _rows(stage, layouts, successes):
    return [{"stage": stage, "layout_index": index, "success": n < successes} for n, index in enumerate(layouts)]


def test_probe_gate_block_16_of_16_passes():
    block = gates.route_gate_block(_rows("doors", range(16), 16), ("doors",))
    assert block["stages"]["doors"]["verdict"] == "PASS" and block["doors_16"] is True
    assert block["stages"]["transport"] == {"verdict": "NOT_RUN", "passed": False}
    assert block["transport_16"] is False


def test_probe_gate_block_15_of_16_fails():
    block = gates.route_gate_block(_rows("transport", range(16), 15), ("transport",))
    entry = block["stages"]["transport"]
    assert (entry["episodes"], entry["successes"], entry["protocol_complete"]) == (16, 15, True)
    assert entry["verdict"] == "FAIL" and block["transport_16"] is False


@pytest.mark.parametrize("layouts", [[1], list(range(8)), list(range(16, 32)), [0] + list(range(15)), []])
def test_probe_gate_block_other_or_fewer_layouts_never_pass(layouts):
    # every episode a success, but not the documented validation 0-15 (fewer, other, repeated, none)
    block = gates.route_gate_block(_rows("doors", layouts, len(layouts)), ("doors",))
    assert block["stages"]["doors"]["verdict"] == "NOT_GATE_PROTOCOL"
    assert block["stages"]["doors"]["passed"] is False and block["doors_16"] is False


def test_probe_gate_block_both_stages():
    rows = _rows("doors", range(16), 16) + _rows("transport", range(16), 16)
    block = gates.route_gate_block(rows, ("doors", "transport"))
    assert block["doors_16"] is True and block["transport_16"] is True
    block = gates.route_gate_block(_rows("doors", range(16), 16) + _rows("transport", range(16), 0),
                                   ("doors", "transport"))
    assert block["doors_16"] is True and block["transport_16"] is False


@pytest.mark.parametrize("name, verdict, successes", [
    ("local-stage-v2/doors-1-kick-center", "NOT_GATE_PROTOCOL", 1),   # the only end-to-end Doors success
    ("instage-doors-F1F2F3", "FAIL", 3),
    ("candidate1-doors-16-31", "NOT_GATE_PROTOCOL", 1),
])
def test_probe_gate_block_on_saved_probe_results(name, verdict, successes):
    saved = json.loads((CAMPAIGN / name / "result.json").read_text())
    block = gates.route_gate_block(saved["episodes"], saved["stages"])
    assert block["stages"]["doors"]["verdict"] == verdict
    assert block["stages"]["doors"]["successes"] == successes
    assert block["doors_16"] is False and block["transport_16"] is False


def test_probe_result_carries_gate_block():
    tree = ast.parse(PROBE.read_text())
    assert any(isinstance(node, ast.ImportFrom) and node.module == "mission7_gates"
               and "route_gate_block" in [alias.name for alias in node.names] for node in tree.body)
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    result = next(node.value for node in ast.walk(run) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets))
    entries = {key.value: value for key, value in zip(result.keys, result.values) if isinstance(key, ast.Constant)}
    call = entries["gate"]
    assert isinstance(call, ast.Call) and call.func.id == "route_gate_block"
    assert [arg.id for arg in call.args] == ["rows", "stages"]
    assert '(args.out / "result.json").write_text(json.dumps(result' in ast.get_source_segment(PROBE.read_text(), run)


# ---- crossing clears: synthetic rows ---------------------------------------------------------------

def _row(crossings, n=320, fall_at=None, index=0):
    """A replay-gate episode row on validation layout `index`.

    Each crossing is (door, side, k0, k1, along0, along1): the stage crosses at
    sample k0 and hands back at sample k1 (None: the trace ends first) while the
    base moves linearly from along0 to along1 metres past the plate centre along
    the door direction.  Outside crossings the base stands at the route start.
    fall_at is the first sample with tilt >= 0.78; it stays fallen.
    """
    layout = generate("validation", index)
    t = (np.arange(n) + 1) * DT
    xy = np.tile(layout.xy(layout.route[0]), (n, 1))
    history = []
    for door, side, k0, k1, along0, along1 in crossings:
        plate = np.asarray(layout.plate(door, side), dtype=float)
        direction = np.asarray(layout.door(door)[1], dtype=float)
        end = n - 1 if k1 is None else k1
        for k in range(k0, end + 1):
            xy[k] = plate + direction * (along0 + (along1 - along0) * (k - k0) / (end - k0))
        history += [{"time_s": float(t[k0 - 30]), "door": door, "side": side, "phase": "approach"},
                    {"time_s": float(t[k0 - 10]), "door": door, "side": side, "phase": "settle"},
                    {"time_s": float(t[k0]), "door": door, "phase": "cross"},
                    {"time_s": float(t[k0]), "door": door, "phase": "cross_kick", "prev_actions_before_norm": 1.}]
        if k1 is not None:
            history.append({"time_s": float(t[k1]), "door": door, "side": side, "phase": "recorded"})
    tilt = np.full(n, .1)
    if fall_at is not None:
        tilt[fall_at:] = 1.2
    return {"layout_index": index, "layout_seed": layout.seed, "replay_elapsed_s": float(t[-1]),
            "fall": fall_at is not None, "first_fall_s": None if fall_at is None else float(t[fall_at]),
            "maximum_tilt": float(tilt.max()), "stage_history": history,
            "samples": [{"time_s": float(t[k]), "xy": xy[k].tolist(), "tilt": float(tilt[k])} for k in range(n)]}


def _side(index=0, door=0):
    return generate("validation", index).correct_sides[door]


def _handed_back(row):
    """The pre-2026-10-01 follow-up count for one episode: 'recorded' after 'cross'."""
    phases = [h["phase"] for h in row["stage_history"]]
    return "cross" in phases and "recorded" in phases[phases.index("cross"):]


def test_clear_is_counted():
    (c,) = gates.crossing_outcomes(_row([(0, _side(), 100, 170, -.45, .36)]))
    assert c["cleared"] and c["handed_back"] and c["real_clear"] and c["correct_side"]
    assert not c["fell_in_crossing"] and not c["fell_before_handback"]
    assert c["start_s"] == pytest.approx(101 * DT) and c["handback_s"] == pytest.approx(171 * DT)
    assert (c["along_start_m"], c["along_max_m"], c["along_end_m"]) == pytest.approx((-.45, .36, .36))


def test_timeout_is_not_a_clear():
    (c,) = gates.crossing_outcomes(_row([(0, _side(), 100, 201, -.45, -.30)]))   # 4.04 s, 0.30 m short
    assert c["handed_back"] and not c["cleared"] and not c["real_clear"]
    assert c["along_max_m"] == pytest.approx(-.30)


def test_fall_in_crossing_is_not_a_clear():
    # falls half way, at along -0.025; the fallen body is carried to +0.40 by hand-back
    (c,) = gates.crossing_outcomes(_row([(0, _side(), 100, 200, -.45, .40)], fall_at=150))
    assert c["fell_in_crossing"] and c["fell_before_handback"] and c["handed_back"]
    assert not c["cleared"] and not c["real_clear"]
    assert c["along_max_m"] == pytest.approx(-.025) and c["along_end_m"] == pytest.approx(.40)


def test_fall_after_handback_keeps_the_clear():
    (c,) = gates.crossing_outcomes(_row([(0, _side(), 100, 170, -.45, .36)], fall_at=250))
    assert c["real_clear"] and not c["fell_in_crossing"] and not c["fell_before_handback"]


def test_fall_before_crossing_start_is_not_a_clear():
    (c,) = gates.crossing_outcomes(_row([(0, _side(), 100, 170, -.45, .36)], fall_at=50))
    assert c["cleared"] and not c["fell_in_crossing"]
    assert c["fell_before_handback"] and not c["real_clear"]


def test_crossing_without_handback_is_not_a_clear():
    (c,) = gates.crossing_outcomes(_row([(0, _side(), 100, None, -.45, .36)]))
    assert c["handback_s"] is None and not c["handed_back"]
    assert c["cleared"] and not c["real_clear"]


def test_wrong_side_crossing_and_two_doors():
    # the definition is the base clearing the plate it staged on; the side is reported, not gated
    door0, door1 = gates.crossing_outcomes(_row([(0, -_side(0, 0), 60, 130, -.45, .37),
                                                 (1, _side(0, 1), 200, 301, -.45, -.33)]))
    assert (door0["door"], door0["correct_side"], door0["real_clear"]) == (0, False, True)
    assert (door1["door"], door1["correct_side"], door1["real_clear"]) == (1, True, False)


def test_phase_count_overstates_real_clears():
    rows = [_row([(0, _side(), 100, 170, -.45, .36)]),                 # clear
            _row([(0, _side(), 100, 201, -.45, -.30)]),                # 4.04 s time-out
            _row([(0, _side(), 100, 200, -.45, .40)], fall_at=150),    # fell in the crossing
            _row([])]                                                  # never reached cross
    assert sum(_handed_back(row) for row in rows) == 3
    assert sum(c["real_clear"] for row in rows for c in gates.crossing_outcomes(row)) == 1


def test_inconsistent_rows_raise():
    row = _row([(0, _side(), 100, 170, -.45, .36)], fall_at=250)
    with pytest.raises(ValueError, match="seed"):
        gates.crossing_outcomes({**row, "layout_seed": row["layout_seed"] + 1})
    with pytest.raises(ValueError, match="first fall"):
        gates.crossing_outcomes({**row, "first_fall_s": None})
    with pytest.raises(ValueError, match="maximum_tilt"):
        gates.crossing_outcomes({**row, "maximum_tilt": .5})
    unstaged = [h for h in row["stage_history"] if h["phase"] not in ("approach", "settle")]
    with pytest.raises(ValueError, match="no staged side"):
        gates.crossing_outcomes({**row, "stage_history": unstaged})


def test_rows_without_samples_raise():
    # a samples-stripped trace read as 10 crossings, 0 clears and no error before 2026-10-01 (review)
    row = _row([(0, _side(), 100, 170, -.45, .36)])
    for stripped in (row, _row([])):
        with pytest.raises(ValueError, match="no samples"):
            gates.crossing_outcomes({**stripped, "samples": []})
    # a crossing that starts after the last sample: nothing to score, not a 'not cleared'
    samples = row["samples"][:90]
    with pytest.raises(ValueError, match="no samples between the crossing"):
        gates.crossing_outcomes({**row, "samples": samples, "replay_elapsed_s": samples[-1]["time_s"]})


# ---- crossing clears: the streaming reader ---------------------------------------------------------

def _dress(row, extra_key_after_samples=False):
    """Give a synthetic row the shape mission7_plate_stage writes: per-sample velocity, contact lists
    and contact events (whose nested keys sit deeper than the sample keys), episode-level extras."""
    samples = [{"time_s": s["time_s"], "xy": s["xy"], "velocity": [.1, 0.], "command": [.3, 0., 0.],
                "yaw": .2, "tilt": s["tilt"], "phase": "recorded", "contacts": ["wall_m7_3"],
                "contact_events": [{"world_geom": "plate_0_1", "body1": "world", "normal_force_N": 80.,
                                    "time_s": -1., "tilt": 9., "xy": [9., 9.]}]}
               for s in row["samples"]]
    dressed = {key: value for key, value in row.items() if key != "samples"}
    dressed["baseline"] = {"fall_time_s": None, "minimum_clearances_m": {"robot_plate_edge_clearance_m": -.07}}
    dressed["samples"] = samples
    if extra_key_after_samples:
        dressed["written_after_samples"] = {"time_s": 1.}
    return dressed


def _write_trace(path, rows, complete=True):
    # byte-for-byte what mission7_overnight.write() produces (json.dumps indent=2 + newline)
    text = json.dumps({"complete": complete, "episodes": rows}, indent=2, allow_nan=False) + "\n"
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as stream:
            stream.write(text)
    else:
        path.write_text(text)
    return path


def _synthetic_gate_rows():
    return [_dress(_row([(0, _side(0), 100, 170, -.45, .36)], index=0)),
            _dress(_row([(0, _side(1), 100, 201, -.45, -.30)], index=1), extra_key_after_samples=True),
            _dress(_row([(0, _side(4), 100, 200, -.45, .40)], fall_at=150, index=4)),
            _dress(_row([], index=9))]


@pytest.mark.parametrize("name", ["episodes.json", "episodes.json.gz"])
def test_reader_matches_json_load(tmp_path, name):
    rows = _synthetic_gate_rows()
    path = _write_trace(tmp_path / name, rows)
    with (gzip.open(path, "rt") if name.endswith(".gz") else open(path)) as stream:
        full = json.load(stream)["episodes"]
    lean = list(gates.read_replay_episodes(path))
    assert [row["layout_index"] for row in lean] == [0, 1, 4, 9]
    for a, b in zip(lean, full):
        assert a["stage_history"] == b["stage_history"] and a["baseline"] == b["baseline"]
        assert a["samples"] == [{k: s[k] for k in ("time_s", "xy", "tilt")} for s in b["samples"]]
        assert gates.crossing_outcomes(a) == gates.crossing_outcomes(b)
    assert lean[1]["written_after_samples"] == {"time_s": 1.}
    traced = gates.replay_gate_crossings(path)
    assert (traced["episodes"], traced["crossings"], traced["real_clears"]) == (4, 3, 1)


def test_reader_rejects_a_truncated_trace(tmp_path):
    path = _write_trace(tmp_path / "episodes.json", _synthetic_gate_rows()[:2])
    text = path.read_text()
    path.write_text(text[:text.rindex('"tilt"')])   # inside the second episode's samples
    with pytest.raises(ValueError, match="ends inside an episode"):
        list(gates.read_replay_episodes(path))
    sample_xy = "\n" + " " * 10 + '"xy": [\n'
    path.write_text(text[:text.rindex(sample_xy) + len(sample_xy) + 4])   # inside a sample's xy pair
    with pytest.raises(ValueError, match="could not convert"):
        list(gates.read_replay_episodes(path))


def test_reader_rejects_other_layouts_and_partial_traces(tmp_path):
    rows = _synthetic_gate_rows()
    path = tmp_path / "episodes.json"
    # unindented (the review's silent '0 episodes, 0 crossings') and otherwise indented files
    for text in (json.dumps({"complete": True, "episodes": rows}),
                 json.dumps({"complete": True, "episodes": rows}, indent=4)):
        path.write_text(text + "\n")
        with pytest.raises(ValueError, match="without an episode list"):
            list(gates.read_replay_episodes(path))
    # a running gate's trace: PlateStage rewrites it with complete: false after every episode
    _write_trace(path, rows, complete=False)
    with pytest.raises(ValueError, match='"complete": false'):
        gates.replay_gate_crossings(path)
    assert [row["layout_index"] for row in gates.read_replay_episodes(path, require_complete=False)] == [0, 1, 4, 9]
    path.write_text(json.dumps({"episodes": rows}, indent=2) + "\n")
    with pytest.raises(ValueError, match='no top-level "complete" flag'):
        list(gates.read_replay_episodes(path))
    text = _write_trace(path, rows).read_text()
    first = text.index("\n    },\n") + len("\n    },\n")
    for cut, message in ((text[:first], "inside its episode list"),           # cut between two episodes
                         (text[:text.rindex("}")], "before its closing brace"),
                         (text + "{}\n", "after the closing brace")):
        path.write_text(cut)
        with pytest.raises(ValueError, match=message):
            list(gates.read_replay_episodes(path))
    _write_trace(path, [])   # an empty but complete episode list is read as such
    assert list(gates.read_replay_episodes(path)) == []


def test_trace_must_be_the_trace_of_result_json(tmp_path):
    rows = _synthetic_gate_rows()
    path = _write_trace(tmp_path / "episodes.json", rows)
    comparison = [{key: value for key, value in row.items() if key != "samples"} for row in rows]
    traced = gates.replay_gate_crossings(path, comparison)
    assert (traced["episodes"], traced["crossings"], traced["real_clears"]) == (4, 3, 1)
    changed = [dict(row) for row in comparison]
    changed[1]["stage_history"] = changed[1]["stage_history"][:-1]   # its hand-back dropped
    # fewer rows, more rows, other order, another run's history: never counted
    for other in (comparison[:3], comparison + comparison[:1], comparison[::-1], changed):
        with pytest.raises(ValueError, match="result.json"):
            gates.replay_gate_crossings(path, other)


def test_trace_path_prefers_the_uncompressed_file(tmp_path):
    assert gates.replay_trace_path(tmp_path) is None
    _write_trace(tmp_path / "episodes.json.gz", [])
    assert gates.replay_trace_path(tmp_path).name == "episodes.json.gz"
    _write_trace(tmp_path / "episodes.json", [])
    assert gates.replay_trace_path(tmp_path).name == "episodes.json"


# ---- crossing clears: the saved replay gates -------------------------------------------------------

SAVED = {   # job: (gate dir, verification label, real clears, layouts that cleared, of them on the wrong plate)
    "21405537": ("replay-gate-v2-align", "v2align_21405537", 4, [0, 1, 5, 15], [5, 15]),
    "21405541": ("replay-gate-v4-settle080", "v4s080_21405541", 1, [1], []),
}


@pytest.fixture(scope="module")
def saved_gate():
    cache = {}

    def load(job):
        if job not in cache:
            gate_dir = CAMPAIGN / SAVED[job][0]
            trace = gates.replay_trace_path(gate_dir)
            if trace is None:
                pytest.skip(f"raw trace of {job} not on disk (untracked): {gate_dir}/episodes.json.gz")
            # as the follow-up reads it: complete, and episode for episode the trace of result.json
            rows = json.loads((gate_dir / "result.json").read_text())["episode_comparison"]
            cache[job] = gates.replay_gate_crossings(trace, rows)
        return cache[job]
    return load


@pytest.mark.parametrize("job", sorted(SAVED))
def test_saved_gate_real_clears(job, saved_gate):
    name, _, clears, layouts, wrong_plate = SAVED[job]
    traced = saved_gate(job)
    assert (traced["episodes"], traced["crossings"], traced["real_clears"]) == (10, 7, clears)
    cleared = [(t["layout_index"], c) for t in traced["layouts"] for c in t["crossings"] if c["real_clear"]]
    assert [index for index, _ in cleared] == layouts
    # a clear is the base clearing the plate it staged on; the square (wrong-side) plate counts too
    assert [index for index, c in cleared if not c["correct_side"]] == wrong_plate
    # the count the follow-up reported as crossings_completed until 2026-10-01: 7 of 7
    rows = json.loads((CAMPAIGN / name / "result.json").read_text())["episode_comparison"]
    assert sum(_handed_back(row) for row in rows) == 7


@pytest.mark.parametrize("job", sorted(SAVED))
def test_saved_gate_matches_the_verification(job, saved_gate):
    if not VERIFY.is_file():
        pytest.skip("verify_crossings.json not present")
    label = SAVED[job][1]
    verified = {row["layout"]: row["crossings"] for row in json.loads(VERIFY.read_text())[label]}
    for layout in saved_gate(job)["layouts"]:
        mine, theirs = layout["crossings"], verified[layout["layout_index"]]
        assert len(mine) == len(theirs)
        for c, v in zip(mine, theirs):
            assert (c["door"], c["side"], round(c["start_s"], 2), round(c["handback_s"], 2)) == \
                (v["door"], v["side"], v["t0"], v["t1"])
            assert (round(c["along_start_m"], 3), round(c["along_max_m"], 3), round(c["along_end_m"], 3)) == \
                (v["along0"], v["along_max"], v["along_end"])
            assert (c["cleared"], c["fell_in_crossing"]) == (v["cleared"], v["fell_in_cross"])


# ---- the follow-up sbatch's embedded summarize ----------------------------------------------------

def _summarize_heredoc():
    text = FOLLOWUP.read_text()
    match = re.search(r'^"\$PY" - "\$gate_dir" "\$tag" "\$campaign" "\$verdict_file" "\$REPO" "\$DRY_RUN" '
                      r'<<\'PYEOF\'\n(.*?)\nPYEOF$', text, re.S | re.M)
    assert match, "step-1 heredoc (with $REPO and $DRY_RUN) not found"
    return match.group(1)


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | \
        {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)} | \
        {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def test_followup_summarize_counts_real_clears():
    subprocess.run(["bash", "-n", str(FOLLOWUP)], check=True)
    tree = ast.parse(_summarize_heredoc())
    # mission7_gates is imported once, under a try that catches Exception (a SyntaxError in the live
    # copy included), so a failed import can null the clears but never stop the verdict
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)
               and "mission7_gates" in [alias.name for alias in node.names]
               or isinstance(node, ast.ImportFrom) and node.module == "mission7_gates"]
    guards = [node for node in tree.body if isinstance(node, ast.Try) and imports and imports[0] in node.body]
    assert len(imports) == 1 and len(guards) == 1
    assert [handler.type.id for handler in guards[0].handlers] == ["Exception"]
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "crossings")
    assert "replay_gate_crossings" in _names(helper) and "stage_history" not in _names(helper)
    # the trace is bound to result.json's rows, and everything that reads it sits under the helper's try
    call = next(node for node in ast.walk(helper) if isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "replay_gate_crossings")
    assert [arg.id for arg in call.args] == ["trace", "rows"]
    guarded = next(node for node in helper.body if isinstance(node, ast.Try))
    assert call in list(ast.walk(guarded)) and "replay_trace_path" in _names(guarded)
    summarize = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "summarize")
    returned = next(node.value for node in ast.walk(summarize) if isinstance(node, ast.Return))
    entries = {key.value: value for key, value in zip(returned.keys, returned.values)}
    completed = _names(entries["crossings_completed"])
    assert "real_clears" in completed
    assert not completed & {"stage_history", "phase", "recorded", "cross", "rows"}
    assert {"recorded", "stage_history"} <= _names(entries["crossings_handed_back"])
    for key in ("crossings_traced", "crossings_completed_definition", "crossings_trace_error"):
        assert key in entries


def _followup_gate(tmp_path, rows):
    campaign = tmp_path / "campaign"
    gate = campaign / "replay-gate-synthetic"
    gate.mkdir(parents=True)
    comparison = [{key: value for key, value in row.items() if key != "samples"} for row in rows]
    upright = sum(not row["fall"] for row in rows)
    (gate / "result.json").write_text(json.dumps({   # as mission7_plate_stage.run() writes it
        "complete": True, "status": "COMPLETED_INTERVENTION", "episodes": len(rows),
        "falls": len(rows) - upright, "upright": upright,
        "exact_replay_gate_passed": len(rows) == 10 and upright == 10, "episode_comparison": comparison}))
    (gate / "submission.json").write_text(json.dumps({
        "job_id": "0", "probe_args": ["--cross-clear=0.35"], "source_sha256": {}}))
    return campaign, gate


def _run_followup(tmp_path, campaign, gate, verdict="INCOMPLETE", repo=_REPO, env=None):
    """Run the step-1 heredoc as the sbatch does, at DRY_RUN=1 (its trace path is then the one read)."""
    script = tmp_path / "summarize.py"
    script.write_text(_summarize_heredoc())
    verdict_file = tmp_path / "verdict"
    out = subprocess.run([sys.executable, str(script), str(gate), "synthetic", str(campaign), str(verdict_file),
                          str(repo), "1"], cwd=_REPO, capture_output=True, text=True, timeout=120, check=True, env=env)
    summary = json.loads((gate / "summary.json").read_text())
    matrix = json.loads((campaign / "replay-gate-matrix-summary.json").read_text())
    assert verdict_file.read_text().strip() == summary["verdict"] == verdict
    assert [g["tag"] for g in matrix["gates"]] == ["synthetic"] and "layouts" not in matrix["gates"][0]
    assert matrix["gates"][0]["crossings_completed"] == summary["crossings_completed"]
    return summary, out.stdout


def test_followup_summarize_runs_on_a_synthetic_gate(tmp_path):
    rows = _synthetic_gate_rows()
    campaign, gate = _followup_gate(tmp_path, rows)
    _write_trace(gate / "episodes.json", rows)
    summary, stdout = _run_followup(tmp_path, campaign, gate)
    assert (summary["crossings_started"], summary["crossings_handed_back"]) == (3, 3)
    assert (summary["crossings_completed"], summary["crossings_traced"]) == (1, 3)
    assert summary["crossings_trace_error"] is None and summary["crossings_trace"] == str(gate / "episodes.json")
    assert [len(layout["crossings"]) for layout in summary["layouts"]] == [1, 1, 1, 0]
    assert "handed back 3, real clears 1 of 3" in stdout
    # after step 2 gzips the trace in place
    with open(gate / "episodes.json", "rb") as source, gzip.open(gate / "episodes.json.gz", "wb") as target:
        target.write(source.read())
    (gate / "episodes.json").unlink()
    summary, _ = _run_followup(tmp_path, campaign, gate)
    assert (summary["crossings_completed"], summary["crossings_traced"]) == (1, 3)


def test_followup_summarize_without_a_trace_reports_null(tmp_path):
    rows = _synthetic_gate_rows()
    campaign, gate = _followup_gate(tmp_path, rows)
    summary, stdout = _run_followup(tmp_path, campaign, gate)
    assert summary["crossings_handed_back"] == 3
    assert summary["crossings_completed"] is None and summary["crossings_traced"] is None
    assert "missing" in summary["crossings_trace_error"]
    assert all(layout["crossings"] is None for layout in summary["layouts"])
    assert "real clears None of None" in stdout
    # an unreadable trace is reported the same way and does not stop the follow-up
    (gate / "episodes.json").write_text('{\n  "complete": true,\n  "episodes": [\n    {\n      "layout_index": 0,\n')
    summary, _ = _run_followup(tmp_path, campaign, gate)
    assert summary["crossings_completed"] is None and "ValueError" in summary["crossings_trace_error"]


def test_followup_summarize_rejects_a_partial_or_foreign_trace(tmp_path):
    rows = _synthetic_gate_rows()
    campaign, gate = _followup_gate(tmp_path, rows)
    # the matrix race: a later gate still running, its trace rewritten with complete: false per episode
    _write_trace(gate / "episodes.json", rows, complete=False)
    summary, _ = _run_followup(tmp_path, campaign, gate)
    assert summary["crossings_completed"] is None and '"complete": false' in summary["crossings_trace_error"]
    # complete traces that are not the trace of result.json: fewer episodes, other order, a repeated one
    for traced in (rows[:3], rows[:2] + rows[3:] + rows[2:3], rows[:1] * 4):
        _write_trace(gate / "episodes.json", traced)
        summary, stdout = _run_followup(tmp_path, campaign, gate)
        assert summary["crossings_completed"] is None and summary["crossings_traced"] is None
        assert "result.json" in summary["crossings_trace_error"] and "real clears None of None" in stdout
        assert summary["crossings_handed_back"] == 3   # read from result.json, as before


@pytest.mark.parametrize("live_copy", [None, "def broken(:\n"])
def test_followup_writes_the_verdict_when_mission7_gates_cannot_import(tmp_path, live_copy):
    # A genuine 10/10 is still written (so the route gate is still released) when the import fails:
    # the module missing, or a broken live copy (scripts/mission7_gates.py is read live, not snapshotted).
    campaign, gate = _followup_gate(tmp_path, [_row([], index=index) for index in range(10)])
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    if live_copy is not None:
        (repo / "scripts/mission7_gates.py").write_text(live_copy)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}   # only the heredoc's own path
    summary, stdout = _run_followup(tmp_path, campaign, gate, verdict="PASS", repo=repo, env=env)
    assert summary["exact_replay_gate_passed"] is True and summary["crossings_handed_back"] == 0
    assert summary["crossings_completed"] is None and summary["crossings_completed_definition"] is None
    error = "ModuleNotFoundError" if live_copy is None else "SyntaxError"
    assert summary["crossings_trace_error"].startswith(f"import mission7_gates: {error}")
    assert "PASS 10/10 upright" in stdout


def test_followup_summary_names_the_trace_step_2_leaves(tmp_path):
    # Steps 1 and 2 in bash at DRY_RUN=0 on an INCOMPLETE synthetic gate: nothing past step 2 runs (no lock,
    # no route gate; a stub sbatch first on PATH would record any call), and summary.json names the
    # episodes.json.gz that step 2 leaves, not the episodes.json it gzipped away.
    rows = _synthetic_gate_rows()
    campaign, gate = _followup_gate(tmp_path, rows)
    _write_trace(gate / "episodes.json", rows)
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "sbatch").write_text(f"#!/bin/bash\ntouch {tmp_path / 'sbatch-called'}\nexit 1\n")
    (stub / "sbatch").chmod(0o755)
    env = {**os.environ, "DRY_RUN": "0", "PYTHON": sys.executable, "M7_REPO": str(_REPO),
           "PATH": f"{stub}:{os.environ['PATH']}"}
    out = subprocess.run(["bash", str(FOLLOWUP), str(gate), "synthetic", "1", str(campaign)], env=env,
                         capture_output=True, text=True, timeout=180, check=True)
    assert "ROUTE_GATE_NOT_RELEASED" in out.stdout and not (tmp_path / "sbatch-called").exists()
    assert not (gate / "episodes.json").exists() and (gate / "episodes.json.gz").is_file()
    summary = json.loads((gate / "summary.json").read_text())
    matrix = json.loads((campaign / "replay-gate-matrix-summary.json").read_text())
    assert summary["crossings_trace"] == matrix["gates"][0]["crossings_trace"] == str(gate / "episodes.json.gz")
    assert (summary["crossings_completed"], summary["crossings_traced"]) == (1, 3)
