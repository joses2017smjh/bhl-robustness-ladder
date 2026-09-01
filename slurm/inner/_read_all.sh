#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""B3 ice and the v2 cells, from event files."""
import glob, statistics
from tensorboard.backend.event_processing import event_accumulator
ROOT = "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/external/Berkeley-Humanoid-Lite/logs/rsl_rl/"

def tail(ea, tag):
    if tag not in ea.Tags()["scalars"]: return None
    v = [x.value for x in ea.Scalars(tag)]
    k = max(1, int(0.05*len(v)))
    return sum(v[-k:])/k, len(v)

print("=== B3 ice: terrain level, blind vs depth vs visible ===")
cells = {}
for run in sorted(glob.glob(ROOT + "biped/*ppo-ice-*")):
    ev = sorted(glob.glob(run + "/events.out.tfevents.*"))
    if not ev: continue
    ea = event_accumulator.EventAccumulator(ev[-1], size_guidance={"scalars": 0}); ea.Reload()
    r = tail(ea, "Curriculum/terrain_levels")
    if r is None: continue
    name = run.split("/")[-1]
    arm = "blind" if "-blind-" in name else ("depth" if "-depth-" in name else "visible")
    cells.setdefault(arm, []).append(r[0])
    print(f"  {name[:52]:52} n={r[1]:5d} tail={r[0]:.4f}")
print()
for a in ("blind", "depth", "visible"):
    if a in cells:
        print(f"  {a:8} mean={statistics.mean(cells[a]):.4f}  (n={len(cells[a])})")

print("\n=== v2 cells: welded hands vs grippers ===")
print(f"{'run':44} {'iters':>6} {'reward':>9} {'ep_len':>8} {'success':>8}")
for run in sorted(glob.glob(ROOT + "task_v2/*")):
    if "smoke" in run: continue
    ev = sorted(glob.glob(run + "/events.out.tfevents.*"))
    if not ev: continue
    ea = event_accumulator.EventAccumulator(ev[-1], size_guidance={"scalars": 0}); ea.Reload()
    rew = tail(ea, "Train/mean_reward"); el = tail(ea, "Train/mean_episode_length")
    su = tail(ea, "Episode_Termination/success")
    print(f"{run.split('/')[-1][:44]:44} {rew[1] if rew else 0:6d} "
          f"{rew[0] if rew else float('nan'):9.3f} {el[0] if el else float('nan'):8.2f} "
          f"{su[0] if su else float('nan'):8.4f}")
PY
