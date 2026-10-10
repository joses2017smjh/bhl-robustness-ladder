#!/usr/bin/env python3
"""Waiter phase 2 scored expert gate (rule predeclared in SLURM_JOBS.md before the scored run).

Runs the SCRIPTED expert (scripts/waiter/expert.py, ORACLE poses) in `WaiterEnv` on held-out layouts and scores the
frozen success definition (docs/WAITER_INTERFACES.md). One JSON per episode (`ep_<seed>.json`), so several jobs can
share a directory and a killed job resumes; `--summarize` writes `gate.json` with the verdict and refuses to
overwrite it.

  episodes:  expert_gate.py --deploy D --upstream U --cache-dir C --out DIR --seeds 2000-2049 [--free-base]
  verdict:   expert_gate.py --out DIR --summarize --seeds 2000-2099 --min-success 80

Seeds are guarded to the gate range 2000-2099 (`--range dev` allows 1000-1999 for development runs).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

RANGES = {"gate": (2000, 2099), "dev": (1000, 1999)}


def parse_seeds(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 4), round((c + h) / d, 4))


def run_episodes(a, seeds) -> None:
    from bhl_robust.eval.waiter_env import STAGES, WaiterEnv
    from expert import ExpertParams, WaiterExpert
    env = WaiterEnv(a.deploy, a.upstream, a.cache_dir, render=False, stance_lock=not a.free_base,
                    time_limit_s=a.time_limit, lock_start_only=a.lock_start_only)
    for s in seeds:
        path = a.out / f"ep_{s}.json"
        if path.exists():
            print(f"GATE: seed {s} exists, skipped", flush=True)
            continue
        t0 = time.time()
        env.reset(s)
        ex = WaiterExpert(env, ExpertParams())
        ex.reset()
        done, info, locked_steps, steps = False, {}, 0, 0
        while not done:
            _, _, done, info = env.step(ex.act())
            locked_steps += int(info["base_locked"])
            steps += 1
        ev = env.evaluate()
        r = {"seed": s, "success": bool(info["success"]), "fell": bool(info["fell"]), "stage": int(info["stage"]),
             "stage_name": STAGES[int(info["stage"])], "t": round(float(info["t"]), 2),
             "deploy": str(a.deploy), "target_colour": env.layout.target_colour, "target_side": env.layout.target_side,
             "template_id": env.template_id, "cup_tilt_deg": round(float(ev["cup_tilt_deg"]), 1),
             "tray_on_target": bool(ev["tray_on_target"]), "stance_lock": bool(env.stance_lock),
             "lock_start_only": bool(env.lock_start_only),
             "locked_fraction": round(locked_steps / max(1, steps), 3), "time_limit_s": env.time_limit_s,
             "expert_notes": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ex.notes.items()},
             "wall_s": round(time.time() - t0, 1)}
        tmp = a.out / f".ep_{s}.tmp.json"
        tmp.write_text(json.dumps(r, indent=1))
        tmp.rename(path)
        print(f"GATE: seed {s} success {r['success']} stage {r['stage_name']} t {r['t']} wall {r['wall_s']} s", flush=True)
    env.close()


def summarize(a, seeds) -> int:
    out = a.out / "gate.json"
    if out.exists():
        raise SystemExit(f"GATE: REFUSED ({out} exists; never overwritten)")
    eps, missing = [], []
    for s in seeds:
        p = a.out / f"ep_{s}.json"
        if p.exists():
            eps.append(json.loads(p.read_text()))
        else:
            missing.append(s)
    if missing:
        raise SystemExit(f"GATE: {len(missing)} episodes missing, e.g. {missing[:5]}; no verdict written")
    k = sum(e["success"] for e in eps)
    n = len(eps)
    stages = {}
    for e in eps:
        stages[e["stage_name"]] = stages.get(e["stage_name"], 0) + 1
    verdict = "PASS" if k >= a.min_success else "FAIL"
    times = sorted(e["t"] for e in eps if e["success"])
    res = {"gate": "Waiter phase 2 scripted expert", "verdict": verdict,
           "rule": f"PASS iff >= {a.min_success} successes over seeds {seeds[0]}-{seeds[-1]} (n = {n})",
           "successes": k, "n": n, "rate": round(k / n, 4), "wilson95": wilson(k, n), "falls": sum(e["fell"] for e in eps),
           "final_stage_counts": stages, "stance_lock": sorted({e["stance_lock"] for e in eps}),
           "lock_start_only": sorted({bool(e.get("lock_start_only", False)) for e in eps}),
           "deploy": sorted({e.get("deploy", "") for e in eps}),
           "time_limit_s": sorted({e["time_limit_s"] for e in eps}),
           "success_time_s": {"median": times[len(times) // 2] if times else None, "max": times[-1] if times else None},
           "failed_seeds": [e["seed"] for e in eps if not e["success"]]}
    tmp = a.out / ".gate.tmp.json"
    tmp.write_text(json.dumps(res, indent=1))
    tmp.rename(out)
    print(f"WAITER PHASE 2 GATE: {verdict} ({k}/{n}; need >= {a.min_success}) -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", type=Path)
    ap.add_argument("--upstream", type=Path)
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--range", choices=sorted(RANGES), default="gate")
    ap.add_argument("--free-base", action="store_true", help="stance_lock=False (the reported free-base baseline)")
    ap.add_argument("--lock-start-only", action="store_true",
                    help="base held only through the reset settle, free from t = 0 (the lock-removal test)")
    ap.add_argument("--time-limit", type=float, default=None, help="default: the env's TIME_LIMIT_S")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--min-success", type=int, default=80)
    a = ap.parse_args(argv)
    seeds = parse_seeds(a.seeds)
    lo, hi = RANGES[a.range]
    bad = [s for s in seeds if not lo <= s <= hi]
    if bad:
        raise SystemExit(f"GATE: REFUSED seeds outside the {a.range} range {lo}-{hi}: {bad[:5]}")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        return summarize(a, seeds)
    if a.time_limit is None:
        from bhl_robust.eval.waiter_env import TIME_LIMIT_S
        a.time_limit = TIME_LIMIT_S
    run_episodes(a, seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
