#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Is the flat reward a hard task or a broken one? Break it into terms."""
import glob
from tensorboard.backend.event_processing import event_accumulator
ROOT = ("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/"
        "external/Berkeley-Humanoid-Lite/logs/rsl_rl/task_v2/")
runs = [r for r in sorted(glob.glob(ROOT + "*")) if "smoke" not in r]
for r in runs[:3]:
    ev = sorted(glob.glob(r + "/events.out.tfevents.*"))
    if not ev: continue
    ea = event_accumulator.EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    ea.Reload(); tags = sorted(ea.Tags()["scalars"])
    print(f"\n=== {r.split('/')[-1]}")
    def tail(t):
        v = [x.value for x in ea.Scalars(t)]
        k = max(1, int(0.05*len(v)))
        return sum(v[-k:]) / k, v[0]
    for t in tags:
        if t.startswith(("Episode_Reward/", "Train/mean_episode_length",
                         "Episode_Termination/")):
            last, first = tail(t)
            print(f"  {t:44} first={first:+9.4f} last={last:+9.4f}")
PY
