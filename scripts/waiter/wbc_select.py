#!/usr/bin/env python3
"""Waiter phase 1 selection (docs/WAITER_PROGRAM.md; frozen 2026-10-05).

A seed is qualified iff turn_test v2 PASS AND cpu_turn_qualify QUALIFIED (variant waiter, protocols unchanged) AND
Q2 PASS. The qualified seed with the lowest push-fall count (qualify's push clause, n = 60) is selected; tie: the
lowest seed. None qualified -> NEGATIVE (phase 2 then falls back to clock-s2 with scripted arm targets). Every seed
needs its three JSONs, else INCOMPLETE and nothing is written. The selection is written once (never overwritten).
usage: wbc_select.py --res <results/waiter-20261005/wbc> [--seeds 0 1 2] [--min-walk-m 1.5]

--min-walk-m (phase 1c onward, 2026-10-07; predeclared before any 1c checkpoint exists): a seed qualifies only if the
turn_test v2 walk AND each of the three cpu_turn_qualify (v2x) walks cover at least this many metres in their 6 s at
0.35 m/s. The phase 1 / 1b walk clauses scored heading drift only, and every one of those six checkpoints covered
0.07-0.41 m (marching in place) against 1.87-2.02 m for clock-s2. Omitted: the phase 1 / 1b rule, unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RULE = ("qualified = turn_test v2 PASS and cpu_turn_qualify QUALIFIED (variant waiter) and Q2 PASS; the qualified "
        "seed with the lowest push-fall count (tie: lowest seed) is selected; none -> NEGATIVE")
PROGRESS_RULE = ("and walk progress: the turn_test v2 walk and each of the 3 cpu_turn_qualify walks cover >= {m} m in "
                 "their 6 s at 0.35 m/s")


def walk_progress(v2: dict, v2x: dict, min_m: float) -> dict:
    """The walk-progress clause from the turn_test v2 JSON and the qualify v2x JSON (pure, so it is testable)."""
    walks = [v2["walk"]] + list(v2x["walks"])
    disp = [round(float(w["displacement_m"]), 3) for w in walks]
    cmds_ok = all(list(w["cmd"]) == [0.35, 0.0, 0.0] and float(w["seconds"]) == 6.0 for w in walks)
    return {"min_m": min_m, "displacement_m": disp, "n_walks": len(walks), "protocol_ok": cmds_ok,
            "pass": cmds_ok and len(walks) == 4 and all(d >= min_m for d in disp)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--prefix", default="waiter-wbc")
    ap.add_argument("--min-walk-m", type=float, default=None)
    a = ap.parse_args(argv)
    out = a.res / "selection.json"
    if out.exists():
        print("WAITER-WBC SELECTION:", json.loads(out.read_text())["line"], "(already recorded)")
        return 0
    per, missing = {}, []
    for s in a.seeds:
        run = f"{a.prefix}-s{s}"
        failed = a.res / "training" / f"{run}.failed.json"
        if failed.is_file():                  # training failed (e.g. diverged): recorded, cannot qualify
            per[f"s{s}"] = {"run": run, "training": "FAILED", "failure": json.loads(failed.read_text()),
                            "qualified": False, "v2": None, "qualify": None, "q2": None, "q2_falls": None,
                            "push_falls": None, "inputs": {"failed": str(failed)}}
            continue
        paths = {"v2": a.res / "turn-test-v2" / f"{run}.json", "qualify": a.res / "qualify" / f"{run}__qualify.json",
                 "q2": a.res / "q2" / f"{run}.json"}
        if a.min_walk_m is not None:
            paths["v2x"] = a.res / "qualify" / f"{run}__v2x.json"
        if not all(p.is_file() for p in paths.values()):
            missing.append(f"s{s}: " + ", ".join(k for k, p in paths.items() if not p.is_file()))
            continue
        v2, q, q2 = (json.loads(paths[k].read_text()) for k in ("v2", "qualify", "q2"))
        push = q["clauses"]["push"]
        per[f"s{s}"] = {"run": run, "v2": v2["verdict"], "qualify": q["verdict"], "q2": q2["verdict"],
                        "q2_falls": q2["falls"], "q2_drift_ok": q2["drift_ok"], "push_falls": push["falls"],
                        "push_n": push["n"], "qualify_detail": q.get("detail"),
                        "push_with_load": q2.get("push_with_load"),
                        "qualified": v2["verdict"] == "PASS" and q["verdict"] == "QUALIFIED" and q2["verdict"] == "PASS",
                        "inputs": {k: str(p) for k, p in paths.items()}}
        if a.min_walk_m is not None:
            wp = walk_progress(v2, json.loads(paths["v2x"].read_text()), a.min_walk_m)
            per[f"s{s}"]["walk_progress"] = wp
            per[f"s{s}"]["qualified"] = per[f"s{s}"]["qualified"] and wp["pass"]
    if missing:
        print("WAITER-WBC SELECTION: INCOMPLETE (" + "; ".join(missing) + "); nothing written")
        return 3
    ok = sorted((d["push_falls"], int(k[1:]), k) for k, d in per.items() if d["qualified"])
    if ok:
        sel = ok[0][2]
        line = f"SELECTED {per[sel]['run']} (push {per[sel]['push_falls']}/{per[sel]['push_n']})"
        verdict = "SELECTED"
    else:
        sel, verdict = None, "NEGATIVE"
        line = "NEGATIVE: no seed qualified (" + "; ".join(
            (f"{k} training FAILED" if d.get("training") == "FAILED" else
             f"{k} v2 {d['v2']} / {d['qualify']} / Q2 {d['q2']} (push {d['push_falls']}/60, Q2 falls {d['q2_falls']}"
             + (f", walk {min(d['walk_progress']['displacement_m'])}-{max(d['walk_progress']['displacement_m'])} m"
                if d.get("walk_progress") else "") + ")")
            for k, d in per.items()) + ")"
    rule = RULE if a.min_walk_m is None else RULE + "; " + PROGRESS_RULE.format(m=a.min_walk_m)
    rec = {"verdict": verdict, "selected": sel, "line": line, "rule": rule, "per_seed": per}
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(rec, f, indent=1)
    print("WAITER-WBC SELECTION:", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
