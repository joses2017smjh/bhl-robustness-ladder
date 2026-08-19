"""Aggregate per-episode CSVs into the tables the README and charts report.

Two products:
  results/terrain_retention.csv  fall rate vs MuJoCo roughness, per policy group
  results/flat_summary.csv       flat-ground sim2sim score, per policy group

Seeds are pooled within a group (e.g. all three terrain-bumpy seeds) because a
single seed's fall rate over 30 episodes is too noisy to plot.
"""
import csv, glob, os, re
from collections import defaultdict
from pathlib import Path

def group_of(label):
    return re.sub(r"-s\d+$", "", label)

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

# --- terrain sweep -------------------------------------------------------
acc = defaultdict(list)
for f in glob.glob("results/terrain/*.csv"):
    m = re.match(r"(.+)__d([0-9.]+)\.csv$", os.path.basename(f))
    if not m:
        continue
    acc[(group_of(m.group(1)), float(m.group(2)))].extend(load(f))

rows = []
for (grp, d), r in sorted(acc.items()):
    if not r:
        continue
    rows.append({
        "policy": grp, "difficulty": d, "n_episodes": len(r),
        "fall_rate": round(sum(x["fell"] == "True" for x in r) / len(r), 4),
        "survival_s": round(sum(float(x["survival_s"]) for x in r) / len(r), 3),
        "distance_m": round(sum(float(x["distance_m"]) for x in r) / len(r), 3),
    })
with open("results/terrain_retention.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"terrain_retention.csv : {len(rows)} rows, {len({r['policy'] for r in rows})} policies")

# --- flat ground ---------------------------------------------------------
flat = defaultdict(list)
for f in glob.glob("results/raw/*.csv"):
    base = os.path.basename(f)[:-4]
    if base.startswith(("demo-", "cmp-", "hi-", "tc-", "terraincheck", "_val")):
        continue
    flat[group_of(base)].extend(load(f))

frows = []
for grp, r in sorted(flat.items()):
    if not r:
        continue
    surv = [x for x in r if x["fell"] != "True"]
    frows.append({
        "policy": grp, "n_episodes": len(r),
        "fall_rate": round(sum(x["fell"] == "True" for x in r) / len(r), 4),
        "survival_s": round(sum(float(x["survival_s"]) for x in r) / len(r), 3),
        "lin_vel_err": round(sum(float(x["lin_vel_err"]) for x in surv) / len(surv), 4) if surv else None,
        "distance_m": round(sum(float(x["distance_m"]) for x in r) / len(r), 3),
    })
with open("results/flat_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(frows[0].keys())); w.writeheader(); w.writerows(frows)
print(f"flat_summary.csv      : {len(frows)} policies")
for r in frows:
    print(f"  {r['policy']:22s} n={r['n_episodes']:4d} fall={r['fall_rate']:.3f} "
          f"lin_err={r['lin_vel_err']} dist={r['distance_m']}")
