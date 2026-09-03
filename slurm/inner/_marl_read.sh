#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
import glob
from tensorboard.backend.event_processing import event_accumulator
for run in sorted(glob.glob("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/logs/skrl/marl/*")):
    ev = sorted(glob.glob(run + "/**/events.out.tfevents.*", recursive=True))
    if not ev: continue
    ea = event_accumulator.EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    ea.Reload()
    t = "Reward / Total reward (mean)"
    if t not in ea.Tags()["scalars"]: continue
    v = [x.value for x in ea.Scalars(t)]
    k = max(1, int(0.05*len(v)))
    print(f"{run.split('/')[-1][:46]:46} n={len(v):5d} first={v[0]:+8.3f} tail={sum(v[-k:])/k:+8.3f}")
PY
