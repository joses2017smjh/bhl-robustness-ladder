#!/usr/bin/env python3
"""Option 3 of the stance-lock work (2026-10-14; SLURM_JOBS.md "Predeclared now ... more standing-only seeds"): pick
the candidate pair from the dev lock-removal checks by the predeclared rule.

Rule: the candidate with the most dev successes (lock_start_only, dev seeds 1000-1031); ties -> the smaller mean Q4
displacement over its non-fall episodes; then the lower seed. It goes to the scored test iff its dev count is
>= --min-dev. A candidate whose dev gate.json is missing is out. Writes the record once (never overwritten) and prints
`SELECT: <run> <deploy>` or `SELECT: none`.

usage: lockremoval_select.py --candidate RUN=DEPLOY=Q4_JSON=DEV_GATE_JSON [...] --min-dev 26 --out <selection.json>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def q4_mean_disp(q4: dict) -> float:
    eps = q4.get("episodes") or q4.get("rows") or []
    d = [float(e["disp_m"]) for e in eps if e.get("fell_at_s") is None and e.get("disp_m") is not None]
    return sum(d) / len(d) if d else float("inf")


def seed_of(run: str) -> int:
    m = re.search(r"-s(\d+)$", run)
    return int(m.group(1)) if m else 10 ** 9


def choose(rows: list[dict], min_dev: int) -> dict:
    """rows: {run, deploy, dev_successes (None = missing), q4_mean_disp_m}. Returns {candidate, scored, rule}."""
    ok = [r for r in rows if r.get("dev_successes") is not None]
    if not ok:
        return {"candidate": None, "scored": False}
    best = sorted(ok, key=lambda r: (-int(r["dev_successes"]), float(r["q4_mean_disp_m"]), seed_of(r["run"])))[0]
    return {"candidate": best["run"], "deploy": best["deploy"], "dev_successes": int(best["dev_successes"]),
            "scored": int(best["dev_successes"]) >= int(min_dev)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", action="append", required=True)
    ap.add_argument("--min-dev", type=int, default=26)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"SELECT: REFUSED ({a.out} exists; never overwritten)")
    rows = []
    for c in a.candidate:
        run, deploy, q4p, devp = c.split("=")
        dev = json.loads(Path(devp).read_text()) if Path(devp).exists() else None
        q4 = json.loads(Path(q4p).read_text()) if Path(q4p).exists() else {}
        rows.append({"run": run, "deploy": deploy, "dev_gate": devp, "q4": q4p,
                     "dev_successes": None if dev is None else int(dev["successes"]),
                     "dev_n": None if dev is None else int(dev["n"]), "q4_mean_disp_m": q4_mean_disp(q4)})
    pick = choose(rows, a.min_dev)
    rec = {"rule": "most dev successes (lock_start_only, dev 1000-1031); ties: smaller mean Q4 displacement, then lower "
                   f"seed; scored test iff >= {a.min_dev}/32", "min_dev": a.min_dev, "candidates": rows, **pick,
           "slurm_job": os.environ.get("SLURM_JOB_ID")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(rec, f, indent=1, default=lambda x: None if x == float("inf") else x)
    for r in rows:
        print(f"candidate {r['run']}: dev {r['dev_successes']}/{r['dev_n']}, Q4 mean displacement {r['q4_mean_disp_m']:.4f} m")
    print(f"SELECT: {pick['candidate']} {pick.get('deploy')}" if pick.get("scored") else
          f"SELECT: none (best {pick.get('candidate')} at {pick.get('dev_successes')}/32, needs {a.min_dev})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
