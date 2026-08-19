"""Downsample the tidy curve dump and summarise each run -> curves.json."""
import csv, json, statistics as st
from collections import defaultdict
from pathlib import Path

rows = list(csv.DictReader(open("results/curves/training_curves.csv")))
by = defaultdict(list)
for r in rows:
    by[(r["label"], r["metric"])].append((int(r["step"]), float(r["value"])))
for k in by:
    by[k].sort()

def downsample(series, n=120):
    if not series:
        return []
    step = max(1, len(series) // n)
    out = []
    for i in range(0, len(series), step):
        chunk = series[i:i + step]
        out.append([chunk[-1][0], round(sum(v for _, v in chunk) / len(chunk), 4)])
    return out

curves = {}
for (label, metric), s in by.items():
    curves.setdefault(label, {})[metric] = downsample(s)

def meta(label):
    if label.startswith("dr-"):
        rest = label[3:].rsplit("-s", 1)[0]
        scale = {"off": 0.0, "default": 1.0, "aggressive": 2.0}.get(rest)
        if scale is None:
            scale = float(rest[1:]) if rest.startswith("s") else None
        return ("DR ladder", scale)
    if label.startswith("push-"):
        return ("Push recovery", label[5:].rsplit("-s", 1)[0])
    if label.startswith("terrain-"):
        return ("Rough terrain", label[8:].rsplit("-s", 1)[0])
    if label.startswith("arms-"):
        return ("Arms (22 DoF)", label[5:].rsplit("-s", 1)[0])
    if label.startswith("coll-"):
        return ("Collision representation", label[5:].rsplit("-s", 1)[0])
    return ("other", None)

summary = []
for label in sorted(curves):
    rw, fl = by.get((label, "reward"), []), by.get((label, "fall_frac"), [])
    grp, key = meta(label)
    tail = lambda s: round(st.mean(v for _, v in s[-25:]), 3) if s else None
    summary.append({"label": label, "group": grp, "key": key,
                    "iters": rw[-1][0] if rw else 0,
                    "final_reward": tail(rw), "final_fall": tail(fl),
                    "complete": bool(rw and rw[-1][0] >= 5900)})

Path("results/curves/curves.json").write_text(json.dumps({"curves": curves, "summary": summary}))
print(f"{len(curves)} runs -> results/curves/curves.json")
