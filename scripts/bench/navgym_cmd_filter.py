"""NavGym v5 command filter (2026-10-07): the predeclared development screen and fresh confirmation of a SCRIPTED
deployment low-pass on the learned navigator's command (bhl_robust/eval/cmd_filter.py), and their verdicts.

Why. The 2026-10-06 confirmation (Computer_Vision/project_results_upgrade/humanoid_confirmatory/verdict.json) was
NEGATIVE only on its zero-fall clause: 6 falls in 288 learned episodes (1-2 in four of six cells), 0 in 96 for A*.
In every fall the commanded wz was flipping between -1 and +1 rad/s at walking speed. The gym hides this chatter
behind its first-order lag (tau 0.15-0.45 s); the physical gait answers in about 0.06 s. The filter restores the
gym's nominal lag on the physics robot (tau 0.2 s, derived in cmd_filter.py, not tuned on maze outcomes).

DEV_RULE and CONFIRM_RULE below are frozen before any filtered episode exists; the launcher
(slurm/repo20260923/cpu_navgym_cmd_filter.sbatch) repeats them verbatim and a test checks that it does.

    navgym_cmd_filter.py dev-verdict <root> --reference <2026-10-06 runs dir> [--final]
    navgym_cmd_filter.py selected-arm <dev verdict.json>
    navgym_cmd_filter.py confirm-verdict <root> --dev-verdict <dev verdict.json> [--final]

A verdict file is never overwritten. Without --final an INCOMPLETE reading is printed, not recorded.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ARMS = ("wz-lpf", "lpf-post-brake")
TIE_BREAK_ARM = "wz-lpf"
TAU_S = 0.2
ACTORS = ("armV5-s8", "armV5-s9", "armV5-s10")
CONDITIONS = {"nominal": "reactive", "drop35": "reactive_dropout"}
MIN_GOALS = {"nominal": 40, "drop35": 36}
N_PER_CELL = 48
DEV_SEEDS = tuple(range(72000, 72048))       # scored on 2026-10-06: development data from here on, no claim
CONFIRM_SEEDS = tuple(range(78000, 78048))   # never used (checked 2026-10-07: no seed file, no launcher or ledger reference)
REPRO = (("armV5-s8", "nominal", 72014), ("armV5-s9", "nominal", 72001), ("armV5-s9", "drop35", 72000),
         ("armV5-s9", "drop35", 72005), ("armV5-s10", "nominal", 72044), ("armV5-s10", "nominal", 72045))

DEV_RULE = ("DEV (maze seeds 72000-72047, scored on 2026-10-06, so development data with no claim): arms wz-lpf and "
            "lpf-post-brake (tau 0.2 s) x actors armV5-s8/s9/s10 x {nominal, drop35}, 48 episodes per cell. An arm "
            "QUALIFIES iff its 6 cells are complete (48 valid episodes each, layouts paired with 2026-10-06), it has 0 "
            "falls over all 288 episodes, and every actor reaches >= 40/48 nominal and >= 36/48 drop35 goals. Selected: "
            "the qualifying arm with the most goals over its 288 episodes; tie -> wz-lpf. No arm qualifies -> DEV "
            "NEGATIVE and no confirmation runs. The 6 fall episodes of 2026-10-06 are rerun unfiltered (reported, not gated).")
CONFIRM_RULE = ("CONFIRM (never-used maze seeds 78000-78047): the dev-selected arm only, actors armV5-s8/s9/s10 x "
                "{nominal, drop35}, plus the SCRIPTED A* reference (unfiltered) x {nominal, drop35}, not gated. PASS iff "
                "EACH actor reaches >= 40/48 nominal and >= 36/48 drop35 goals with 0 falls (the 2026-10-06 rule); "
                "otherwise NEGATIVE; INCOMPLETE if any cell is missing, short or mislabelled.")
LABELS = {"learned": "frozen biped gait (PPO, dr-default-s0) + NavGym v5 final actor (vx, wz)",
          "scripted": "first-order command low-pass (cmd_filter.py) + team_sensors speed brake; A* reference only",
          "oracle": "pose and goal coordinate", "scope": "simulation (MuJoCo 3.3.5), no hardware"}


def cell_dir(root: Path, arm: str, actor: str, cond: str) -> Path:
    return Path(root) / arm / f"{actor}-{cond}"


def _load_cell(d: Path, seeds) -> tuple[dict, list[str]]:
    eps, problems = {}, []
    for s in seeds:
        f = d / f"seed{s}.json"
        if not f.is_file():
            problems.append(f"missing {f}")
            continue
        try:
            eps[s] = json.loads(f.read_text())
        except (OSError, ValueError) as e:
            problems.append(f"unreadable {f}: {e}")
    return eps, problems


def _check_episode(r: dict, seed: int, arm: str | None, actor: str | None, cond: str) -> list[str]:
    p = []
    if r.get("seed") != seed:
        p.append(f"seed {r.get('seed')} != {seed}")
    m = r.get("maze") or {}
    if (m.get("n"), m.get("m"), m.get("extra_openings")) != (6, 6, 1):
        p.append(f"seed {seed}: maze {m.get('n')}x{m.get('m')} +{m.get('extra_openings')}")
    if r.get("sensor_mode") != CONDITIONS[cond]:
        p.append(f"seed {seed}: sensor_mode {r.get('sensor_mode')} != {CONDITIONS[cond]}")
    if actor is None:                                   # the A* reference
        if r.get("policy") is not None or "cmd_filter" in r:
            p.append(f"seed {seed}: the A* reference ran a policy or a filter")
        return p
    if not str(r.get("policy") or "").endswith(f"{actor}/actor.onnx"):
        p.append(f"seed {seed}: policy {r.get('policy')} is not {actor}'s final actor")
    if (r.get("policy_map_integration") or {}).get("mode") != "capture_pose":
        p.append(f"seed {seed}: not run with --policy-capture-pose")
    cf = r.get("cmd_filter") or {}
    if cf.get("mode") != arm:
        p.append(f"seed {seed}: cmd_filter mode {cf.get('mode')} != {arm}")
    want_tau = None if arm == "none" else TAU_S
    if cf.get("tau_s") != want_tau:
        p.append(f"seed {seed}: cmd_filter tau {cf.get('tau_s')} != {want_tau}")
    return p


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def cell_stats(eps: dict) -> dict:
    rs = [eps[s] for s in sorted(eps)]
    outcomes = [r.get("outcome") for r in rs]
    cf = [r.get("cmd_filter") or {} for r in rs]
    return {"n": len(rs), "goals": sum(bool(r.get("success")) for r in rs),
            "clean": sum(bool(r.get("clean_success")) for r in rs),
            "falls": outcomes.count("fall"), "time_outs": outcomes.count("time_out"),
            "other": sum(o not in ("goal", "fall", "time_out") for o in outcomes),
            "fall_seeds": [r["seed"] for r in rs if r.get("outcome") == "fall"],
            "median_completion_s": _median([r.get("completion_s") for r in rs]),
            "gait_wz_flips_per_s_median": _median([(c.get("gait_wz") or {}).get("flips_per_s") for c in cf]),
            "actor_wz_flips_per_s_median": _median([(c.get("actor_wz") or {}).get("flips_per_s") for c in cf]),
            "gait_wz_saturated_share_median": _median([(c.get("gait_wz") or {}).get("saturated_share") for c in cf]),
            "max_tilt_rad_max": (max(c["max_tilt_rad"] for c in cf if c.get("max_tilt_rad") is not None)
                                 if any(c.get("max_tilt_rad") is not None for c in cf) else None)}


def _paired(eps: dict, ref_dir: Path) -> tuple[dict, list[str]]:
    """Layout pairing with, and outcome change against, the 2026-10-06 unfiltered episode of the same seed."""
    problems, won, lost, falls_ref = [], 0, 0, 0
    for s, r in eps.items():
        f = ref_dir / f"seed{s}.json"
        if not f.is_file():
            problems.append(f"reference episode missing: {f}")
            continue
        ref = json.loads(f.read_text())
        if (ref.get("maze") or {}).get("layout_sha256") != (r.get("maze") or {}).get("layout_sha256"):
            problems.append(f"seed {s}: layout differs from the 2026-10-06 episode")
        won += int(bool(r.get("success")) and not ref.get("success"))
        lost += int(bool(ref.get("success")) and not r.get("success"))
        falls_ref += int(ref.get("outcome") == "fall")
    return {"goals_won": won, "goals_lost": lost, "reference_falls": falls_ref}, problems


def _actor_meets(cells: dict, actor: str) -> bool:
    return all(cells[f"{actor}-{c}"]["goals"] >= MIN_GOALS[c] for c in CONDITIONS)


def dev_verdict(root: Path, reference: Path) -> dict:
    root, reference = Path(root), Path(reference)
    out = {"rule": DEV_RULE, "labels": LABELS, "tau_s": TAU_S, "seeds": [DEV_SEEDS[0], DEV_SEEDS[-1]], "arms": {},
           "problems": []}
    for arm in ARMS:
        cells, problems = {}, []
        for actor in ACTORS:
            for cond in CONDITIONS:
                d = cell_dir(root, arm, actor, cond)
                eps, pr = _load_cell(d, DEV_SEEDS)
                problems += pr
                for s, r in eps.items():
                    problems += _check_episode(r, s, arm, actor, cond)
                st = cell_stats(eps)
                st["vs_2026_10_06"], pp = _paired(eps, reference / f"{actor}-{cond}")
                problems += pp
                cells[f"{actor}-{cond}"] = st
        complete = not problems and all(c["n"] == N_PER_CELL for c in cells.values())
        falls = sum(c["falls"] for c in cells.values())
        goals = sum(c["goals"] for c in cells.values())
        meets = {a: _actor_meets(cells, a) for a in ACTORS}
        out["arms"][arm] = {"complete": complete, "falls": falls, "goals": goals, "actors_meet_goal_clauses": meets,
                            "qualifies": complete and falls == 0 and all(meets.values()), "cells": cells}
        out["problems"] += problems
    q = [a for a in ARMS if out["arms"][a]["qualifies"]]
    if any(not out["arms"][a]["complete"] for a in ARMS):
        out["verdict"], out["selected"] = "INCOMPLETE", None
    elif not q:
        out["verdict"], out["selected"] = "DEV NEGATIVE", None
    else:
        best = max(out["arms"][a]["goals"] for a in q)
        top = [a for a in q if out["arms"][a]["goals"] == best]
        out["verdict"], out["selected"] = "SELECTED", (TIE_BREAK_ARM if TIE_BREAK_ARM in top else top[0])
    out["repro"] = repro_report(root / "repro", reference)
    return out


def repro_report(repro_root: Path, reference: Path) -> dict:
    rows = []
    for actor, cond, seed in REPRO:
        f = Path(repro_root) / f"{actor}-{cond}" / f"seed{seed}.json"
        ref = json.loads((Path(reference) / f"{actor}-{cond}" / f"seed{seed}.json").read_text()) if (
            Path(reference) / f"{actor}-{cond}" / f"seed{seed}.json").is_file() else None
        if not f.is_file():
            rows.append({"cell": f"{actor}-{cond}", "seed": seed, "status": "missing"})
            continue
        r = json.loads(f.read_text())
        same = ref is not None and r.get("outcome") == ref.get("outcome") and r.get("elapsed_s") == ref.get("elapsed_s")
        rows.append({"cell": f"{actor}-{cond}", "seed": seed, "outcome": r.get("outcome"), "elapsed_s": r.get("elapsed_s"),
                     "reference_outcome": ref.get("outcome") if ref else None,
                     "reference_elapsed_s": ref.get("elapsed_s") if ref else None, "reproduced_exactly": same,
                     "gait_wz": (r.get("cmd_filter") or {}).get("gait_wz"),
                     "max_tilt_rad": (r.get("cmd_filter") or {}).get("max_tilt_rad")})
    return {"note": "reported, not gated: the 2026-10-06 fall episodes rerun unfiltered (--policy-cmd-filter none)",
            "episodes": rows, "reproduced_exactly": sum(bool(x.get("reproduced_exactly")) for x in rows)}


def selected_arm(dev_verdict_path: Path) -> str | None:
    v = json.loads(Path(dev_verdict_path).read_text())
    if not v.get("final") or v.get("verdict") != "SELECTED" or v.get("selected") not in ARMS:
        return None
    return v["selected"]


def confirm_verdict(root: Path, dev_verdict_path: Path) -> dict:
    root = Path(root)
    arm = selected_arm(dev_verdict_path)
    out = {"rule": CONFIRM_RULE, "labels": LABELS, "seeds": [CONFIRM_SEEDS[0], CONFIRM_SEEDS[-1]],
           "dev_verdict": str(dev_verdict_path), "arm": arm, "tau_s": TAU_S, "cells": {}, "problems": []}
    if arm is None:
        out.update(verdict="INCOMPLETE", problems=["the dev verdict is not a final SELECTED verdict"])
        return out
    for actor in ACTORS:
        for cond in CONDITIONS:
            eps, pr = _load_cell(cell_dir(root, arm, actor, cond), CONFIRM_SEEDS)
            out["problems"] += pr
            for s, r in eps.items():
                out["problems"] += _check_episode(r, s, arm, actor, cond)
            out["cells"][f"{actor}-{cond}"] = cell_stats(eps)
    astar = {}
    for cond in CONDITIONS:
        eps, pr = _load_cell(cell_dir(root, "astar", "astar", cond), CONFIRM_SEEDS)
        out["problems"] += pr
        for s, r in eps.items():
            out["problems"] += _check_episode(r, s, None, None, cond)
        astar[f"astar-{cond}"] = cell_stats(eps)
    out["astar_reference"] = astar
    complete = not out["problems"] and all(c["n"] == N_PER_CELL for c in [*out["cells"].values(), *astar.values()])
    per_actor = {a: {"meets": _actor_meets(out["cells"], a)
                     and all(out["cells"][f"{a}-{c}"]["falls"] == 0 for c in CONDITIONS),
                     **{c: {k: out["cells"][f"{a}-{c}"][k] for k in ("goals", "clean", "falls", "time_outs")}
                        for c in CONDITIONS}} for a in ACTORS}
    out["per_actor"] = per_actor
    if not complete:
        out["verdict"] = "INCOMPLETE"
    else:
        out["verdict"] = "PASS" if all(v["meets"] for v in per_actor.values()) else "NEGATIVE"
    return out


def _write_once(path: Path, v: dict, final: bool) -> int:
    path = Path(path)
    if path.exists():
        print(f"verdict already recorded, not overwritten: {path}")
        print(json.dumps({k: v.get(k) for k in ("verdict", "selected", "arm")}))
        return 0
    if v["verdict"] == "INCOMPLETE" and not final:
        print("INCOMPLETE (not recorded without --final): " + "; ".join(v["problems"][:12]))
        return 0
    v["final"] = bool(final)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(v, indent=1) + "\n")
    print(f"wrote {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dev-verdict")
    d.add_argument("root", type=Path)
    d.add_argument("--reference", type=Path, required=True)
    d.add_argument("--final", action="store_true")
    s = sub.add_parser("selected-arm")
    s.add_argument("dev_verdict", type=Path)
    c = sub.add_parser("confirm-verdict")
    c.add_argument("root", type=Path)
    c.add_argument("--dev-verdict", type=Path, required=True)
    c.add_argument("--final", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "dev-verdict":
        v = dev_verdict(a.root, a.reference)
        rc = _write_once(a.root / "verdict.json", v, a.final)
        print(f"NAVGYM-CMDFILTER DEV VERDICT: {v['verdict']} (selected {v['selected']}) "
              + " | ".join(f"{k}: falls {x['falls']}, goals {x['goals']}, qualifies {x['qualifies']}" for k, x in v["arms"].items()))
        return rc
    if a.cmd == "selected-arm":
        arm = selected_arm(a.dev_verdict)
        if arm is None:
            print("NONE")
            return 1
        print(arm)
        return 0
    v = confirm_verdict(a.root, a.dev_verdict)
    rc = _write_once(a.root / "verdict.json", v, a.final)
    print(f"NAVGYM-CMDFILTER CONFIRM VERDICT: {v['verdict']} (arm {v['arm']}) "
          + " | ".join(f"{k}: nominal {x['nominal']['goals']}/48 ({x['nominal']['falls']} falls), drop35 "
                       f"{x['drop35']['goals']}/48 ({x['drop35']['falls']} falls)" for k, x in v.get("per_actor", {}).items()))
    return rc


if __name__ == "__main__":
    sys.exit(main())
