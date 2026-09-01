#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Tier 1 results, from skrl's event files."""
import glob
from tensorboard.backend.event_processing import event_accumulator
for run in sorted(glob.glob("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/logs/skrl/marl/*")):
    ev = sorted(glob.glob(run + "/**/events.out.tfevents.*", recursive=True))
    if not ev: print(f"{run.split('/')[-1]}: no events"); continue
    ea = event_accumulator.EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    ea.Reload(); tags = ea.Tags()["scalars"]
    rew = [t for t in tags if "eward" in t and "total" in t.lower()] or \
          [t for t in tags if "eward" in t]
    print(f"\n{run.split('/')[-1]}")
    for t in rew[:3]:
        v = [x.value for x in ea.Scalars(t)]
        k = max(1, int(0.05*len(v)))
        print(f"  {t:46} n={len(v):5d} first={v[0]:+9.3f} tail={sum(v[-k:])/k:+9.3f}")
PY
