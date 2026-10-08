"""Give the flat race a denominator (LOC-07).

``docs/gifs/multi_race.gif`` is one 12 s rollout per policy: four robots in one
MuJoCo world, matched 0.45 m/s shoves every 3.5 s, seed 0. Exactly one robot
fell in it. A clip of n=1 says who fell once, not how often; this turns the
per-seed outcomes of ``scripts/render_multi.py --no-video --json`` into

* ``results/multi-race-20260923/multi_race_10seed.json`` -- one row per seed
  and policy (fell, first-fall time, peak x, the push times that seed drew) and
  a per-policy aggregate (falls over N seeds, first-fall times);
* ``results/multi-race-20260923/multi_race_seeds.csv`` -- the same rows as a
  flat CSV (seed, policy, fell, fell_at_s, peak_x_m, run_dir), the per-robot
  table LOC-07 asked for;
* a table on stdout and one line ``RACE RESULT: <label> falls/N | ...`` that
  the Slurm log and the ledger can quote verbatim.

Input is a directory of ``seed_<k>.json`` files written by ``--json``. As a
fallback (``--log``) it also parses the per-robot outcome lines render_multi
prints, in both the current ``peak x=`` and the older ``x=`` spelling, so a
run whose JSON was lost is still scoreable from its Slurm log. The two paths
produce the same rows (peak x to 4 dp from JSON, 2 dp from the log); the JSON
one additionally carries the push times.

Seed semantics, so the denominator is stated honestly: the seed drives the
initial joint-angle noise and the direction of every shove. ``push_all`` draws
one angle per event and applies it to all four robots, so within a seed the
shoves are matched across policies; across seeds they differ. Ten seeds are
ten matched races, not ten independent trials per robot.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

# The GIF run (Slurm 20951392, seed 0) recorded exactly one fall: the
# no-randomization robot at 4.8 s. Reported beside seed 0 of the rerun as an
# informational check only -- render_multi has changed since that log.
GIF_REFERENCE = {"seed": 0, "label": "no randomization", "fell_at_s": 4.8,
                 "source": "/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/multi-gif-20951392.out "
                           "(docs/gifs/multi_race.gif)"}

CSV_FIELDS = ["seed", "policy", "fell", "fell_at_s", "peak_x_m", "run_dir"]

_LINE = re.compile(
    r"^\s{2}(?P<label>.{1,22}?)\s+(?P<status>FELL @ (?P<t>[\d.]+)s|upright)\s+"
    r"(?:peak )?x=(?P<x>[+-]?[\d.]+) m\s*$")
_SEED = re.compile(r"(?:=== seed (?P<a>\d+) ===|--no-video\): \d+ policy steps simulated, seed (?P<b>\d+))")


def load_json_dir(d: Path) -> list[dict]:
    recs = []
    for f in sorted(d.glob("seed_*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or -1)):
        try:
            recs.append(json.loads(f.read_text()))
        except json.JSONDecodeError as e:
            print(f"warning: {f}: {e}", file=sys.stderr)
    return recs


def parse_log(path: Path) -> list[dict]:
    """Recover per-seed records from render_multi's printed outcome lines."""
    recs: list[dict] = []
    seed = None
    cur: dict | None = None
    for line in path.read_text(errors="replace").splitlines():
        m = _SEED.search(line)
        if m:
            seed = int(m.group("a") or m.group("b"))
            cur = None
            continue
        m = _LINE.match(line)
        if not m:
            continue
        if cur is None:
            cur = {"seed": seed, "flags": {}, "push_times_s": None, "robots": []}
            recs.append(cur)
        t = m.group("t")
        cur["robots"].append({
            "label": m.group("label").strip(),
            "fell": t is not None,
            "fell_at_s": float(t) if t is not None else None,
            "peak_x_m": float(m.group("x")),
        })
    return recs


def summarise(recs: list[dict], expect_seeds: int) -> dict:
    labels: "OrderedDict[str, None]" = OrderedDict()
    rows = []
    seeds_seen = []
    flags = None
    for r in recs:
        if r.get("seed") is None:
            raise SystemExit("a record has no seed; refusing to aggregate anonymous rollouts")
        if r["seed"] in seeds_seen:
            raise SystemExit(f"seed {r['seed']} appears twice; one race per seed")
        seeds_seen.append(r["seed"])
        if r.get("flags"):
            if flags is None:
                flags = r["flags"]
            elif r["flags"] != flags:
                raise SystemExit(f"seed {r['seed']} ran with different flags: {r['flags']} vs {flags}")
        for rb in r["robots"]:
            labels.setdefault(rb["label"], None)
            rows.append({"seed": r["seed"], "policy": rb["label"], "fell": bool(rb["fell"]),
                         "fell_at_s": rb["fell_at_s"], "peak_x_m": rb["peak_x_m"],
                         "run_dir": rb.get("run_dir"), "push_times_s": r.get("push_times_s")})

    per_policy = OrderedDict()
    n = len(seeds_seen)
    for lab in labels:
        mine = [x for x in rows if x["policy"] == lab]
        falls = [x for x in mine if x["fell"]]
        times = [x["fell_at_s"] for x in falls]
        per_policy[lab] = {
            "seeds": len(mine),
            "falls": len(falls),
            "fall_rate": round(len(falls) / len(mine), 3) if mine else None,
            "first_fall_times_s": times,
            "mean_first_fall_s": round(sum(times) / len(times), 2) if times else None,
            "fell_on_seeds": [x["seed"] for x in falls],
            "mean_peak_x_m": round(sum(x["peak_x_m"] for x in mine) / len(mine), 3) if mine else None,
            "run_dir": next((x["run_dir"] for x in mine if x.get("run_dir")), None),
        }

    ref = None
    s0 = [x for x in rows if x["seed"] == GIF_REFERENCE["seed"]]
    if s0:
        # The GIF log printed the fall time with %.1f, so compare at that precision.
        fallen = [(x["policy"], round(x["fell_at_s"], 1)) for x in s0 if x["fell"]]
        ref = {
            "reference": GIF_REFERENCE,
            "rerun_seed0_falls": fallen,
            "matches_reference": fallen == [(GIF_REFERENCE["label"], GIF_REFERENCE["fell_at_s"])],
            "note": "informational: same seed, but render_multi has changed since the GIF run",
        }

    complete = n >= expect_seeds and all(p["seeds"] == n for p in per_policy.values())
    return {
        "protocol": ("scripts/render_multi.py --no-video: four policies in one flat MuJoCo world, "
                     "matched shoves (one direction per event for all robots), same flags as "
                     "docs/gifs/multi_race.gif; one race per seed"),
        "flags": flags,
        "seeds": sorted(seeds_seen),
        "n_seeds": n,
        "expected_seeds": expect_seeds,
        "complete": complete,
        "per_policy": per_policy,
        "gif_seed0_check": ref,
        "rows": rows,
    }


def print_table(s: dict) -> None:
    n = s["n_seeds"]
    print(f"=== flat race, {n} seed(s) (expected {s['expected_seeds']}), flags {s['flags']} ===")
    print(f"{'policy':22s} {'falls':>7s} {'rate':>6s} {'mean 1st fall':>14s} {'mean peak x':>12s}  first-fall times (s)")
    for lab, p in s["per_policy"].items():
        mf = "-" if p["mean_first_fall_s"] is None else f"{p['mean_first_fall_s']:.1f} s"
        rate = "-" if p["fall_rate"] is None else f"{p['fall_rate']:.2f}"
        px = "-" if p["mean_peak_x_m"] is None else f"{p['mean_peak_x_m']:+.2f} m"
        times = ", ".join(f"{t:.1f}" for t in p["first_fall_times_s"]) or "-"
        print(f"{lab:22s} {p['falls']:>3d}/{p['seeds']:<3d} {rate:>6s} {mf:>14s} {px:>12s}  {times}")
    if s["gif_seed0_check"]:
        c = s["gif_seed0_check"]
        print(f"seed 0 vs GIF log: rerun falls {c['rerun_seed0_falls']} ; GIF recorded "
              f"[('{GIF_REFERENCE['label']}', {GIF_REFERENCE['fell_at_s']})] -> "
              f"{'same' if c['matches_reference'] else 'different'} ({c['note']})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("results/multi-race-20260923"),
                    help="directory holding seed_<k>.json from render_multi --json")
    ap.add_argument("--log", type=Path, default=None,
                    help="fallback: parse render_multi outcome lines from this log instead")
    ap.add_argument("--out", type=Path, default=Path("results/multi-race-20260923/multi_race_10seed.json"))
    ap.add_argument("--csv", type=Path, default=None,
                    help="per-robot rows as CSV (default: multi_race_seeds.csv beside --out)")
    ap.add_argument("--expect-seeds", type=int, default=10)
    ap.add_argument("--job", default=None, help="Slurm job id to record")
    args = ap.parse_args()

    recs = parse_log(args.log) if args.log else load_json_dir(args.dir)
    if not recs:
        print("RACE RESULT: INCOMPLETE (no rollouts found)")
        return 1
    s = summarise(recs, args.expect_seeds)
    s["job"] = args.job
    s["source"] = str(args.log) if args.log else str(args.dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(s, indent=2) + "\n")
    csv_path = args.csv or args.out.with_name("multi_race_seeds.csv")
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in s["rows"]:
            w.writerow({**row, "fell": int(row["fell"]),
                        "fell_at_s": "" if row["fell_at_s"] is None else row["fell_at_s"],
                        "run_dir": row["run_dir"] or ""})

    print_table(s)
    print(f"summary -> {args.out}")
    print(f"rows    -> {csv_path}")
    falls = " | ".join(f"{lab} {p['falls']}/{p['seeds']}" for lab, p in s["per_policy"].items())
    status = "" if s["complete"] else f"INCOMPLETE ({s['n_seeds']}/{s['expected_seeds']} seeds) "
    print(f"RACE RESULT: {status}{falls}")
    return 0 if s["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
