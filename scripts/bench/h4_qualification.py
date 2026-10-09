"""Run the unchanged H4 instruments with the experiment's 79D controller.

The only temporary substitution is the controller factory. Original turn
simulation, commands, reset seeds, physics and predicates are called directly.
No generic evaluator file is changed and all failed episodes are retained.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]


def _load_turn_module():
    spec = importlib.util.spec_from_file_location("h4_original_turn_test", REPO / "scripts/bench/turn_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def judge_records(v2: dict, v2x: dict, rows: list[dict]) -> dict:
    """Recompute the original joint rule, requiring complete declared cohorts."""
    turn = _load_turn_module()
    from bhl_robust.eval.harness import EvalConfig
    if v2.get("protocol") != "v2" or v2x.get("protocol") != "v2x":
        raise ValueError("missing original v2/v2x protocol")
    expected_v2 = {(s, sign * 0.6) for s in (0, 1, 2) for sign in (1, -1)}
    expected_v2x = {(s, sign * 0.6) for s in (10, 11, 12, 13, 14) for sign in (1, -1)}
    for records, expected in ((v2["turns"], expected_v2), (v2x["turns"], expected_v2x)):
        if len(records) != len(expected) or {(r["seed"], r["cmd"][2]) for r in records} != expected:
            raise ValueError("turn records do not match the declared cohort")
        if any(r["cmd"][:2] != [0.0, 0.0] or r["warm_s"] != 3.0 or r["seconds"] != 6.0 for r in records):
            raise ValueError("turn command or timing changed")
    if v2["walk"]["seed"] != 0 or {r["seed"] for r in v2x["walks"]} != {10, 11, 12} or len(v2x["walks"]) != 3:
        raise ValueError("walk records do not match the declared cohort")
    walks = [v2["walk"], *v2x["walks"]]
    if any(r["cmd"] != [0.35, 0.0, 0.0] or r["warm_s"] != 1.0 or r["seconds"] != 6.0 for r in walks):
        raise ValueError("walk command or timing changed")
    rule = v2x.get("rule", {})
    expected_rule = {"turn_min_deg": 150.0, "drift_max_deg": 15.0, "seconds": 6.0,
                     "turn_seeds": [10, 11, 12, 13, 14], "walk_seeds": [10, 11, 12],
                     "min_turn_ok": 9, "min_walk_ok": 2, "turn_wz": 0.6, "turn_warm_s": 3.0,
                     "walk_cmd": [0.35, 0.0, 0.0], "walk_warm_s": 1.0}
    if rule != expected_rule:
        raise ValueError("v2x rule differs from the unchanged qualification")
    expected_push = {(tuple(c), s) for c in EvalConfig().commands for s in range(10)}
    pairs = [(tuple(float(r[k]) for k in ("command_vx", "command_vy", "command_wz")), int(r["seed"])) for r in rows]
    if len(rows) != 60 or len(set(pairs)) != 60 or set(pairs) != expected_push:
        raise ValueError("push rows do not match all 60 declared command/reset pairs")
    if any(str(r["fell"]).lower() not in ("true", "false") for r in rows):
        raise ValueError("push fall field is not boolean")
    falls = sum(str(r["fell"]).lower() == "true" for r in rows)
    a = turn.v2_judge(v2["turns"], v2["walk"], 150.0, 15.0)
    b = turn.v2x_judge(v2x["turns"], v2x["walks"], 150.0, 15.0, 9, 2)
    qualified = b["verdict"] == "PASS" and falls <= 9
    return {"v2": a["verdict"], "qualification": "QUALIFIED" if qualified else "NOT QUALIFIED",
            "turn_ok": b["n_turn_ok"], "turn_n": 10, "walk_ok": b["n_walk_ok"], "walk_n": 3,
            "push_falls": falls, "push_n": 60,
            "joint_pass": a["verdict"] == "PASS" and qualified}


def evaluate(deploy: Path, upstream: Path, work: Path, output: Path, label: str, smoke: bool = False) -> dict:
    from bhl_robust.eval import gait_clock
    from bhl_robust.eval.heading_observable import make_controller
    from bhl_robust.eval import run_eval
    turn = _load_turn_module()
    old_clock, old_eval = gait_clock.make_controller, run_eval.make_controller
    output.mkdir(parents=True, exist_ok=True)
    gait_clock.make_controller = make_controller
    run_eval.make_controller = make_controller
    try:
        if smoke:
            records = [turn.run_command(deploy, upstream, work / "mjcf-smoke", "humanoid", cmd,
                                        seconds=0.8, warm=0.2, seed=100)
                       for cmd in ((0.0, 0.0, 0.6), (0.0, 0.0, -0.6), (0.35, 0.0, 0.0))]
            result = {"status": "PASS", "non_scored_reset_seed": 100,
                      "note": "plumbing smoke only; episode outcomes are not qualification evidence",
                      "records": records}
            (output / "smoke_rollouts.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        common = dict(deploy=deploy, upstream=upstream, cache_dir=work / "mjcf", variant="humanoid",
                      seconds=6.0, warm=1.0, seed=0, turn_min_deg=150.0, drift_max_deg=15.0,
                      turn_seeds=[10, 11, 12, 13, 14], walk_seeds=[10, 11, 12],
                      min_turn_ok=9, min_walk_ok=2)
        turn.main_v2(SimpleNamespace(**common, out=output / "turn-v2.json"))
        turn.main_v2x(SimpleNamespace(**common, out=output / "turn-v2x.json"))
        push_csv = output / "push-60.csv"
        rc = run_eval.main(["--deploy-cfg", str(deploy), "--upstream", str(upstream),
                           "--cache-dir", str(work / "push-mjcf"), "--out", str(push_csv),
                           "--label", label, "--variant", "humanoid", "--episode-s", "12",
                           "--seeds", "10", "--push-speed", "0.5", "--terrain-difficulty", "0"])
        if rc != 0:
            raise RuntimeError("original push instrument did not finish")
        v2 = json.loads((output / "turn-v2.json").read_text())
        v2x = json.loads((output / "turn-v2x.json").read_text())
        with push_csv.open() as handle:
            rows = list(csv.DictReader(handle))
        result = judge_records(v2, v2x, rows)
        result["instrument_sha256"] = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in (REPO / "scripts/bench/turn_test.py",
                                                 REPO / "src/bhl_robust/eval/run_eval.py",
                                                 REPO / "src/bhl_robust/eval/harness.py")}
        (output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        gait_clock.make_controller = old_clock
        run_eval.make_controller = old_eval
