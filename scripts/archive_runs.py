"""Flatten every training run into a portable record: numbers, config, provenance.

Written against losing the cluster. `extract_curves.py` pulls a curated tag list
for plotting; this pulls *everything*, because the point is that the CSV has to
answer questions nobody has asked yet once the event files are gone.

Three outputs, all small enough to live in git:

    runs/manifest.csv    one row per run: name, experiment, iterations, wall
                         time, final and tail-mean value of every scalar
    runs/scalars.csv.gz  every (run, tag, step, value) triple
    runs/params/         each run's params/*.yaml, which is what says how the
                         numbers were produced

Checkpoints are deliberately not included -- they are 31 GB and a different
problem. This is the evidence, not the weights.
"""
from __future__ import annotations

import csv
import gzip
import shutil
import statistics
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOTS = [Path(a) for a in sys.argv[1:-1]]
OUT = Path(sys.argv[-1])
(OUT / "params").mkdir(parents=True, exist_ok=True)


def tail_mean(vals: list[float], frac: float = 0.05) -> float:
    n = max(1, int(len(vals) * frac))
    return statistics.fmean(vals[-n:])


runs: list[Path] = []
for root in ROOTS:
    if root.is_dir():
        runs += sorted(d for d in root.iterdir() if d.is_dir())

manifest_rows, all_tags = [], set()
scalars_path = OUT / "scalars.csv.gz"
n_ok = n_skip = 0

with gzip.open(scalars_path, "wt", newline="") as gz:
    w = csv.writer(gz)
    w.writerow(["run", "experiment", "tag", "step", "value"])
    for d in runs:
        if not any(d.glob("events.out.tfevents.*")):
            n_skip += 1
            continue
        try:
            ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
            ea.Reload()
            tags = ea.Tags()["scalars"]
        except Exception as exc:                                 # noqa: BLE001
            print(f"  unreadable: {d.name}: {exc!r}")
            n_skip += 1
            continue
        exp = d.parent.name
        row = {"run": d.name, "experiment": exp}
        iters = 0
        for tag in tags:
            pts = ea.Scalars(tag)
            vals = [p.value for p in pts]
            iters = max(iters, len(vals))
            for p in pts:
                w.writerow([d.name, exp, tag, p.step, f"{p.value:.6g}"])
            row[f"{tag}|final"] = f"{vals[-1]:.6g}"
            row[f"{tag}|tail5"] = f"{tail_mean(vals):.6g}"
            row[f"{tag}|max"] = f"{max(vals):.6g}"
            all_tags.add(tag)
        row["iterations"] = iters
        row["checkpoints"] = len(list(d.glob("model_*.pt")))
        manifest_rows.append(row)
        # provenance: what config produced these numbers
        pdir = d / "params"
        if pdir.is_dir():
            dest = OUT / "params" / d.name
            dest.mkdir(parents=True, exist_ok=True)
            for y in pdir.glob("*.yaml"):
                shutil.copy2(y, dest / y.name)
        n_ok += 1
        print(f"  {d.name:52} {iters:6d} iters  {len(tags):3d} tags")

cols = ["run", "experiment", "iterations", "checkpoints"]
for t in sorted(all_tags):
    cols += [f"{t}|final", f"{t}|tail5", f"{t}|max"]
with open(OUT / "manifest.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=cols, restval="")
    wr.writeheader()
    wr.writerows(manifest_rows)

print(f"\n{n_ok} runs archived, {n_skip} skipped (no event file)")
print(f"  {OUT/'manifest.csv'}  ({len(cols)} columns)")
print(f"  {scalars_path}  ({scalars_path.stat().st_size/1048576:.1f} MB)")
