#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
import glob, statistics
from tensorboard.backend.event_processing import event_accumulator
ROOT=("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/"
      "external/Berkeley-Humanoid-Lite/logs/rsl_rl/biped/")
TAG="Curriculum/terrain_levels"
cells={}
for t in ("slippery","stairs"):
    for v in ("depth","blind"):
        vals=[]
        for s in (0,1,2):
            runs=sorted(glob.glob(f"{ROOT}*_ppo-{t}-{v}-s{s}"))
            if not runs: continue
            ev=sorted(glob.glob(runs[-1]+"/events.out.tfevents.*"))
            if not ev: continue
            ea=event_accumulator.EventAccumulator(ev[-1],size_guidance={"scalars":0}); ea.Reload()
            if TAG not in ea.Tags()["scalars"]: continue
            c=[x.value for x in ea.Scalars(TAG)]
            tail=c[int(0.95*len(c)):]
            vals.append((s,len(c),max(c),sum(tail)/len(tail)))
        cells[(t,v)]=vals
        for s,n,mx,tm in vals:
            print(f"  ppo-{t}-{v}-s{s}  n={n:5d} max={mx:.4f} tail={tm:.4f}")
print()
print(f"{'terrain':10} {'depth (n)':>16} {'blind (n)':>16} {'ratio':>7}")
for t in ("slippery","stairs"):
    d=[x[3] for x in cells.get((t,"depth"),[])]; b=[x[3] for x in cells.get((t,"blind"),[])]
    if not d or not b: continue
    dm,bm=statistics.mean(d),statistics.mean(b)
    print(f"{t:10} {dm:9.3f} (n={len(d)}) {bm:9.3f} (n={len(b)}) {dm/bm:7.2f}x")
PY
