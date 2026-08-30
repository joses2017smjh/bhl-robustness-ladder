#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Read the finished v2 cells from their event files, not from a log tail."""
import glob
from tensorboard.backend.event_processing import event_accumulator
ROOT = ("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/"
        "external/Berkeley-Humanoid-Lite/logs/rsl_rl/task_v2/")
runs = sorted(glob.glob(ROOT + "*"))
if not runs:
    print("no task_v2 runs under", ROOT); raise SystemExit(0)
print(f"{'run':40} {'iters':>6} {'reward':>9} {'success':>9}")
for r in runs:
    ev = sorted(glob.glob(r + "/events.out.tfevents.*"))
    if not ev: continue
    ea = event_accumulator.EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    ea.Reload(); tags = set(ea.Tags()["scalars"])
    def tail(tag):
        if tag not in tags: return None
        v = [x.value for x in ea.Scalars(tag)]
        return sum(v[int(0.95*len(v)):]) / max(1, len(v[int(0.95*len(v)):]))
    rew = tail("Train/mean_reward")
    n = len(ea.Scalars("Train/mean_reward")) if "Train/mean_reward" in tags else 0
    succ = None
    for t in tags:
        if "success" in t.lower() or "placed" in t.lower() or "scored" in t.lower():
            succ = tail(t); break
    print(f"{r.split('/')[-1][:40]:40} {n:6d} "
          f"{rew if rew is not None else float('nan'):9.3f} "
          f"{succ if succ is not None else float('nan'):9.4f}")
PY
