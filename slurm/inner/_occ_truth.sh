#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
import glob, statistics
from tensorboard.backend.event_processing import event_accumulator
ROOT = ("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/"
        "external/Berkeley-Humanoid-Lite/logs/rsl_rl/biped/")
TAG = "Curriculum/terrain_levels"
cells = {}
for t in ("slippery", "stairs"):
    for v in ("depth", "blind"):
        vals = []
        for s in (0, 1):
            runs = sorted(glob.glob(f"{ROOT}*_ppo-{t}-{v}-s{s}"))
            if not runs:
                print(f"  missing ppo-{t}-{v}-s{s}"); continue
            ev = sorted(glob.glob(runs[-1] + "/events.out.tfevents.*"))
            ea = event_accumulator.EventAccumulator(ev[-1], size_guidance={"scalars": 0})
            ea.Reload()
            if TAG not in ea.Tags()["scalars"]:
                print(f"  ppo-{t}-{v}-s{s}: no {TAG}"); continue
            c = [x.value for x in ea.Scalars(TAG)]
            tail = c[int(0.95*len(c)):]          # last 5%, not the last line
            m = sum(tail)/len(tail)
            vals.append(m)
            print(f"  ppo-{t}-{v}-s{s:<2} n={len(c):5d} max={max(c):.4f} "
                  f"tail-mean={m:.4f}")
        if vals:
            cells[(t, v)] = vals
print()
print(f"{'terrain':10} {'depth':>16} {'blind':>16} {'ratio':>8}")
for t in ("slippery", "stairs"):
    d = cells.get((t, "depth")); b = cells.get((t, "blind"))
    if not d or not b: continue
    dm, bm = statistics.mean(d), statistics.mean(b)
    print(f"{t:10} {dm:8.3f} (n={len(d)}) {bm:8.3f} (n={len(b)}) {dm/bm:8.2f}x")
PY
