#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
import glob
from tensorboard.backend.event_processing import event_accumulator
ROOT = ("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/"
        "external/Berkeley-Humanoid-Lite/logs/rsl_rl/coop_lift/")
TAGS = ["Curriculum/lift_height", "Episode_Reward/lifting_object",
        "Curriculum/stage_lift"]
print(f"{'run':44} {'iters':>7} {'max h':>8} {'tail h':>8} {'stage':>7}")
for pat in ["2026-08-24_23-55-08*occluded-blind-s0", "*coop-occluded-blind-s1",
            "*coop-occluded-blind-s2", "2026-08-27*coop-occluded-depth-s0"]:
    for run in sorted(glob.glob(ROOT + pat)):
        ev = sorted(glob.glob(run + "/events.out.tfevents.*"))
        if not ev: continue
        ea = event_accumulator.EventAccumulator(ev[-1], size_guidance={"scalars": 0})
        ea.Reload(); tags = set(ea.Tags()["scalars"])
        def g(t):
            if t not in tags: return None
            v = [x.value for x in ea.Scalars(t)]
            return v
        h = g(TAGS[0]); s = g(TAGS[2])
        if h is None: continue
        tail = h[int(0.95*len(h)):]
        print(f"{run.split('/')[-1][:44]:44} {len(h):7d} {max(h):8.4f} "
              f"{sum(tail)/len(tail):8.4f} {(max(s) if s else 0):7.3f}")
PY
