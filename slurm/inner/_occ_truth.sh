#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Did the collapse happen in training, or only in the MuJoCo replay?

The replay shows both cube arms dropping ~40 cm in the first 0.2 s. If that is
a reward hack it must be visible on the PhysX side too, and base_contact is the
tell: it fires only when the base geom touches the ground. A robot squatting in
free space scores 0 on it; a robot sitting on the floor does not.
"""
import glob
from tensorboard.backend.event_processing import event_accumulator
ROOT = ("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/"
        "external/Berkeley-Humanoid-Lite/logs/rsl_rl/coop_lift/")
TAGS = ["Episode_Reward/base_contact_a", "Episode_Reward/base_contact_b",
        "Episode_Reward/still_alive", "Episode_Reward/flat_a",
        "Episode_Reward/lift_progress", "Episode_Termination/fallen"]
for pat in ["*coop-cube-staged-r2-s0", "*coop-cube-pinch-s0",
            "2026-08-24_23-55-08*occluded-blind-s0"]:
    runs = sorted(glob.glob(ROOT + pat))
    if not runs: print(f"{pat}: none"); continue
    run = runs[-1]
    ev = sorted(glob.glob(run + "/events.out.tfevents.*"))
    ea = event_accumulator.EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    ea.Reload(); avail = set(ea.Tags()["scalars"])
    print(f"\n{run.split('/')[-1]}")
    for t in TAGS:
        if t not in avail:
            print(f"    {t.split('/')[-1]:16} absent"); continue
        v = [x.value for x in ea.Scalars(t)]
        tail = v[int(0.95*len(v)):]
        print(f"    {t.split('/')[-1]:16} tail-mean={sum(tail)/len(tail):+9.4f}  "
              f"min={min(v):+8.4f}")
PY
