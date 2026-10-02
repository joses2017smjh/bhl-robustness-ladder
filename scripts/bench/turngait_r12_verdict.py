"""Per-arm verdict of the R1 / R2 turning launcher (slurm/repo20260923/gpu_turngait_r12.sbatch).

PREDECLARED RULE (frozen 2026-10-01; v5's joint rule verbatim, SLURM_JOBS.md line 3055), per arm,
each arm judged separately:
  PASS iff >= 2/3 seeds both PASS turn_test v2 AND are QUALIFIED by cpu_turn_qualify's unchanged
  rule (v2x >= 9/10 on reset seeds 10-14, walk <= 15 deg on >= 2/3, push <= 9/60); else FAIL;
  INCOMPLETE if any JSON is missing. A seed that fails v2 does not count even if it qualifies
  through v2x. Training counts only if the run dir holds model_5999.pt and the training log shows
  the arm's task id, the feet_gait term and the push event.
Labels: LEARNED gait; MuJoCo gates. The two arms are new tasks trained from scratch, never
fine-tunes of TurnBoth-s0.

Read from JSON only (never exit codes): per seed <res>/training/<run>.json (written by the
launcher from the fresh training log), <res>/turn-test-v2/<run>.json (turn_test.py --protocol v2)
and <res>/qualify/<run>__qualify.json (cpu_turn_qualify.sbatch). An unreadable JSON, or one that is
not the declared protocol / not this run, counts as missing. The verdict JSON is written only when
the arm is complete, and an existing one is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RULE = ("PASS iff >= 2/3 seeds both PASS turn_test v2 AND are QUALIFIED by cpu_turn_qualify's unchanged rule "
        "(v2x >= 9/10 on reset seeds 10-14, walk <= 15 deg on >= 2/3, push <= 9/60); else FAIL; INCOMPLETE if "
        "any JSON is missing. A seed that fails v2 does not count even if it qualifies through v2x.")
TRAINING_RULE = ("Training counts only if the run dir holds model_5999.pt and the training log shows the arm's "
                 "task id, the feet_gait term and the push event.")
LABEL = "LEARNED gait; MuJoCo gates"
ARMS = {
    "R1": {"task": "Velocity-BHL-Arms-TurnGaitClock-v0", "prefix": "arms-turngait-clock",
           "what": "gait clock in the actor (77 obs) and critic"},
    "R2": {"task": "Velocity-BHL-Arms-TurnGaitCritic-v0", "prefix": "arms-turngait-critic",
           "what": "gait clock in the critic only (actor 75 obs)"},
}
SEEDS = (0, 1, 2)
FINAL_CKPT = "model_5999.pt"
# turn_test.py --protocol v2 as the launcher runs it (defaults; v5 identical)
V2_RULE = {"turn_min_deg": 150.0, "drift_max_deg": 15.0, "seconds": 6.0, "turn_seeds": [0, 1, 2], "turn_wz": 0.6,
           "turn_warm_s": 3.0, "walk_cmd": [0.35, 0.0, 0.0], "walk_warm_s": 1.0, "walk_seed": 0}
NEED = 2


def training_counts(t: dict) -> bool:
    """The frozen training clause, from the launcher's training JSON."""
    return bool(t.get("final_ckpt") == FINAL_CKPT and t.get("final_ckpt_present") and t.get("task_id_in_log")
                and t.get("feet_gait_in_log") and t.get("push_robot_in_log"))


def seed_counts(training_ok: bool, v2: str, qualify: str) -> bool:
    """A seed counts iff its training counts, it PASSES v2 AND it is QUALIFIED."""
    return bool(training_ok) and v2 == "PASS" and qualify == "QUALIFIED"


def arm_verdict(seeds: list[dict | None]) -> dict:
    """Pure rule. seeds: one entry per seed (exactly 3), each {"training": bool | None, "v2": "PASS" | "FAIL" | None,
    "qualify": "QUALIFIED" | "NOT QUALIFIED" | None}; None = that JSON is missing (or the whole seed is None)."""
    if len(seeds) != len(SEEDS):
        raise ValueError(f"expected {len(SEEDS)} seeds, got {len(seeds)}")
    rows = [s or {"training": None, "v2": None, "qualify": None} for s in seeds]
    missing = [i for i, s in enumerate(rows) if s.get("training") is None or s.get("v2") is None or s.get("qualify") is None]
    k = sum(seed_counts(s.get("training"), s.get("v2"), s.get("qualify")) for s in rows)
    if missing:
        return {"verdict": "INCOMPLETE", "n_counted": k, "need": NEED, "of": len(SEEDS), "missing_seeds": missing}
    return {"verdict": "PASS" if k >= NEED else "FAIL", "n_counted": k, "need": NEED, "of": len(SEEDS),
            "missing_seeds": []}


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:                                               # noqa: BLE001
        return None


def read_seed(res: Path, run: str) -> dict:
    """{training, v2, qualify} for one run from its three JSONs (None = missing / unreadable / not this run)."""
    t = _read(res / "training" / f"{run}.json")
    training = training_counts(t) if (isinstance(t, dict) and t.get("run") == run) else None
    d = _read(res / "turn-test-v2" / f"{run}.json")
    v2 = None
    if isinstance(d, dict) and d.get("protocol") == "v2" and f"_{run}/" in str(d.get("deploy", "")) \
            and all(d.get("rule", {}).get(k) == v for k, v in V2_RULE.items()) and d.get("verdict") in ("PASS", "FAIL"):
        v2 = d["verdict"]
    q = _read(res / "qualify" / f"{run}__qualify.json")
    qualify = q["verdict"] if (isinstance(q, dict) and q.get("run") == run
                               and q.get("verdict") in ("QUALIFIED", "NOT QUALIFIED")) else None
    return {"run": run, "training": training, "v2": v2, "qualify": qualify,
            "push": (q or {}).get("clauses", {}).get("push") if isinstance(q, dict) else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--res", type=Path, required=True, help="the launcher's results directory")
    ap.add_argument("--out", type=Path, default=None, help="verdict JSON (default <res>/verdict/<arm>.json)")
    args = ap.parse_args(argv)
    arm = ARMS[args.arm]
    out = args.out or args.res / "verdict" / f"{args.arm}.json"
    if out.exists():
        d = _read(out) or {}
        print(f"ARM-R12 {args.arm}: {d.get('verdict')} (already recorded, not overwritten: {out})")
        return 0
    rows = [read_seed(args.res, f"{arm['prefix']}-s{s}") for s in SEEDS]
    v = arm_verdict(rows)
    rec = {"arm": args.arm, "task": arm["task"], "what": arm["what"], **v, "seeds": rows, "rule": RULE,
           "training_rule": TRAINING_RULE, "label": LABEL,
           "note": "new task trained from scratch (no parent checkpoint); not a fine-tune of TurnBoth-s0"}
    print(f"ARM-R12 {args.arm} ({arm['task']}): {v['verdict']} ({v['n_counted']}/3 seeds count, need {NEED}; "
          f"per seed (training, v2, qualify) {[(r['training'], r['v2'], r['qualify']) for r in rows]}) [{LABEL}]")
    if v["verdict"] != "INCOMPLETE":
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(out, "x") as f:                # O_EXCL: two array tasks finishing together cannot both write
                f.write(json.dumps(rec, indent=2) + "\n")
            print(f"wrote {out}")
        except FileExistsError:
            print(f"verdict already recorded by another task, not overwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
