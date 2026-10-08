#!/usr/bin/env python3
"""Verdicts for the NavGym v4 capture-pose map-integration test (N0 in docs/SOLUTIONS_2026-10-01.md).

`maze_explore.py --policy` integrates each lidar packet into the learned ego map at the loop-top pose of the step AFTER
the packet was captured (as scored in 21484212). `--policy-capture-pose` integrates it at the estimated pose it was
captured at, as the humanoid branch already does for its grid. Two predeclared stages, rules frozen here and in
slurm/repo20260923/cpu_navgym_capture_pose.sbatch before any episode ran:

  screen  both gate-passing final actors (armV4-s5, armV4-s6), each with and without the flag, on training-range hard
          6x6 maze seeds 9120-9143 (never run by any physics or gym job), 180 s. SCREEN_RULE below.
  fresh   only if the screen PASSES: both actors WITH the flag on never-used maze seeds 60000-60011, 180 s, under the
          unchanged physics-transfer rule of 21484212 (each actor >= 10/12 goals with 0 falls). A* reported, not gated.

Labels: LEARNED biped gait (frozen) + LEARNED NavGym v4 actor; ORACLE pose and goal. A verdict file is never
overwritten (an existing one is re-printed); an INCOMPLETE reading is printed, not recorded.

usage: navgym_capture_pose.py screen <root>   |   navgym_capture_pose.py fresh <root>
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ACTORS = ("armV4-s5", "armV4-s6")
SCREEN_RULE = {
    "seeds": list(range(9120, 9144)),
    "time_limit_s": 180.0,
    "arms": {a: {"loop_top": f"{a}-looptop", "capture_pose": f"{a}-capture"} for a in ACTORS},
    "s5_net_min": 4, "s5_lost_max": 1, "s6_net_min": -1,
    "text": ("PASS iff armV4-s5 nets >= +4 goals with the flag (won minus lost, paired by maze) with <= 1 lost, armV4-s6 "
             "nets >= -1, each actor has 0 falls with the flag, and each actor's total wall-contact steps with the flag "
             "are <= its total without; on training-range hard 6x6 maze seeds 9120-9143, 180 s. McNemar exact p reported, "
             "not gated. Otherwise NEGATIVE; INCOMPLETE if any arm is missing an episode or a pairing check fails."),
}
FRESH_RULE = {
    "seeds": list(range(60000, 60012)),
    "time_limit_s": 180.0,
    "arms": {a: f"{a}-capture" for a in ACTORS},
    "min_goals": 10,
    "text": ("Run only if the screen PASSES. PASS iff EACH actor (armV4-s5, armV4-s6, final actors, with "
             "--policy-capture-pose) reaches >= 10/12 goals with 0 falls on never-used hard 6x6 maze seeds 60000-60011, "
             "180 s (21484212's rule, unchanged). Otherwise NEGATIVE; INCOMPLETE if any summary is missing or not 12 "
             "episodes. A* on the same mazes reported, not gated."),
}
LABELS = {"learned": "frozen biped gait (PPO) + NavGym v4 actor (vx, wz) commands",
          "scripted": "A* reference only (fresh stage)", "oracle": "pose and goal coordinate"}


def load_rows(d: Path) -> dict:
    """{seed: per-seed JSON} from <d>/seed<k>.json."""
    d = Path(d)
    if not d.is_dir():
        return {}
    return {int(m.group(1)): json.loads(f.read_text())
            for f in sorted(d.iterdir()) if (m := re.fullmatch(r"seed(\d+)\.json", f.name))}


def mcnemar_exact_p(won: int, lost: int) -> float:
    """Two-sided exact McNemar p on the discordant pairs (binomial, p = 0.5)."""
    n = won + lost
    if n == 0:
        return 1.0
    k = min(won, lost)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def _arm_problems(rows: dict, seeds, flag_on: bool, name: str) -> list:
    p = []
    missing = [s for s in seeds if s not in rows]
    if missing:
        p.append(f"{name}: missing seeds {missing}")
    extra = sorted(set(rows) - set(seeds))
    if extra:
        p.append(f"{name}: unexpected seeds {extra}")
    for s in seeds:
        r = rows.get(s)
        if r is None:
            continue
        mi = r.get("policy_map_integration")
        if flag_on and (mi is None or mi.get("mode") != "capture_pose" or not mi.get("updates_at_capture_pose")):
            p.append(f"{name} seed {s}: capture-pose integration not recorded")
        if not flag_on and mi is not None:
            p.append(f"{name} seed {s}: the default arm carries a map-integration override")
        if r.get("policy") is None:
            p.append(f"{name} seed {s}: not a learned --policy episode")
    return p


def screen_verdict(root: Path, rule: dict = SCREEN_RULE) -> dict:
    root, seeds = Path(root), rule["seeds"]
    problems, per = [], {}
    for actor in ACTORS:
        names = rule["arms"][actor]
        base, fix = load_rows(root / names["loop_top"]), load_rows(root / names["capture_pose"])
        problems += _arm_problems(base, seeds, False, names["loop_top"]) + _arm_problems(fix, seeds, True, names["capture_pose"])
        won, lost, both, neither = [], [], 0, 0
        for s in seeds:
            if s not in base or s not in fix:
                continue
            if base[s]["maze"]["layout_sha256"] != fix[s]["maze"]["layout_sha256"]:
                problems.append(f"{actor} seed {s}: the paired episodes ran different layouts")
            b, f = bool(base[s]["success"]), bool(fix[s]["success"])
            if f and not b:
                won.append(s)
            elif b and not f:
                lost.append(s)
            elif b and f:
                both += 1
            else:
                neither += 1
        per[actor] = {
            "goals_loop_top": sum(bool(r["success"]) for r in base.values()),
            "goals_capture_pose": sum(bool(r["success"]) for r in fix.values()),
            "won": won, "lost": lost, "net": len(won) - len(lost), "both": both, "neither": neither,
            "mcnemar_exact_p": round(mcnemar_exact_p(len(won), len(lost)), 4),
            "falls_capture_pose": sum(r["outcome"] == "fall" for r in fix.values()),
            "falls_loop_top": sum(r["outcome"] == "fall" for r in base.values()),
            "wall_contact_steps_capture_pose": sum(int(r["wall_contact_steps"]) for r in fix.values()),
            "wall_contact_steps_loop_top": sum(int(r["wall_contact_steps"]) for r in base.values()),
        }
    if problems:
        return {"verdict": "INCOMPLETE", "problems": problems, "rule": rule["text"], "per_actor": per, "labels": LABELS}
    s5, s6 = per["armV4-s5"], per["armV4-s6"]
    clauses = {
        "s5_net": s5["net"] >= rule["s5_net_min"],
        "s5_lost": len(s5["lost"]) <= rule["s5_lost_max"],
        "s6_net": s6["net"] >= rule["s6_net_min"],
        "no_falls": all(per[a]["falls_capture_pose"] == 0 for a in ACTORS),
        "wall_contacts_not_increased": all(per[a]["wall_contact_steps_capture_pose"] <= per[a]["wall_contact_steps_loop_top"]
                                           for a in ACTORS),
    }
    verdict = "PASS" if all(clauses.values()) else "NEGATIVE"
    detail = " | ".join(f"{a}: {per[a]['goals_loop_top']}->{per[a]['goals_capture_pose']}/24, won {len(per[a]['won'])}, "
                        f"lost {len(per[a]['lost'])}, net {per[a]['net']:+d}, McNemar p {per[a]['mcnemar_exact_p']}, "
                        f"falls {per[a]['falls_capture_pose']}, wall steps {per[a]['wall_contact_steps_loop_top']}->"
                        f"{per[a]['wall_contact_steps_capture_pose']}" for a in ACTORS)
    return {"verdict": verdict, "clauses": clauses, "per_actor": per, "detail": detail, "rule": rule["text"],
            "labels": LABELS}


def fresh_verdict(root: Path, rule: dict = FRESH_RULE) -> dict:
    root, seeds = Path(root), rule["seeds"]
    problems, per = [], {}
    for actor in ACTORS:
        name = rule["arms"][actor]
        rows = load_rows(root / name)
        problems += _arm_problems(rows, seeds, True, name)
        summ = root / name / "summary.json"
        if not summ.is_file():
            problems.append(f"{name}: no summary.json")
        per[actor] = {"goals": sum(bool(r["success"]) for r in rows.values()), "n": len(rows),
                      "falls": sum(r["outcome"] == "fall" for r in rows.values()),
                      "time_outs": sum(r["outcome"] == "time_out" for r in rows.values()),
                      "clean": sum(bool(r["clean_success"]) for r in rows.values()),
                      "failed_seeds": [s for s in seeds if s in rows and not rows[s]["success"]]}
    astar = load_rows(root / "astar")
    ref = {"goals": sum(bool(r["success"]) for r in astar.values()), "n": len(astar)}
    if problems:
        return {"verdict": "INCOMPLETE", "problems": problems, "rule": rule["text"], "per_actor": per,
                "astar_reference": ref, "labels": LABELS}
    meets = {a: per[a]["goals"] >= rule["min_goals"] and per[a]["falls"] == 0 for a in ACTORS}
    verdict = "PASS" if all(meets.values()) else "NEGATIVE"
    detail = " | ".join(f"{a}: {per[a]['goals']}/12 (falls {per[a]['falls']}, time-outs {per[a]['time_outs']})" for a in ACTORS)
    return {"verdict": verdict, "meets": meets, "per_actor": per, "astar_reference": ref,
            "detail": detail + f" | A* reference {ref['goals']}/{ref['n']}", "rule": rule["text"], "labels": LABELS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=("screen", "fresh"))
    ap.add_argument("root", type=Path, help="the stage's output directory (one sub-directory per arm)")
    a = ap.parse_args(argv)
    out = a.root / "verdict.json"
    if out.exists():
        v = json.loads(out.read_text())
        print(f"verdict already recorded, not overwritten: {out}")
    else:
        v = (screen_verdict if a.stage == "screen" else fresh_verdict)(a.root)
        if v["verdict"] != "INCOMPLETE":
            a.root.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(v, indent=2) + "\n")
            print(f"wrote {out}")
        else:
            print("INCOMPLETE problems: " + "; ".join(v["problems"]))
    print(f"NAVGYM-CAPTURE-POSE {a.stage.upper()} VERDICT: {v['verdict']} -- {v.get('detail', '')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
