#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Does the descent come before the pinch, or after it?

If the policy squats to reach a low object, the pinch and the descent arrive
together and the descent is task-appropriate. If it drops first and only then
closes on the payload, the descent is what *buys* the pinch -- and nothing in
the reward or the terminations charges for it: flat_orientation_l2 and
either_fallen both read orientation only, and there is no base-height term.
"""
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/src")
from bhl_robust.eval.coop_replay import (build_crew, CoopActor, CrewRunner,
                                         POLICY_DT)
PINCH_GATE_M = 0.20   # same gate the carry clips light their pinch lamp on
UP = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/external/Berkeley-Humanoid-Lite")
CD = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/mjcf_cache")
L = UP/"logs/rsl_rl/coop_lift"
def ck(pat):
    r = sorted(L.glob(pat))[-1]
    return sorted(r.glob("model_*.pt"), key=lambda q:int(q.stem.split("_")[1]))[-1]

for label, pat, pay, seed in (("cube staged", "*coop-cube-staged-r2-s0", "cube", 11),
                              ("cube pinch-only", "*coop-cube-pinch-s0", "cube", 0)):
    try:
        c = ck(pat)
    except IndexError:
        print(f"{label}: no run"); continue
    m, slots, crates = build_crew(UP, CD, 2, ego_camera=True, payload=pay)
    r = CrewRunner(m, slots, crates, CoopActor(c))
    r.reset(np.random.default_rng(seed))
    z0 = np.array([float(r.d.xpos[s.body_id][2]) for s in slots])
    t_half_sink = t_pinch = t_lift = None
    total = None
    print(f"\n=== {label}  ({c.parent.name})")
    print(f"{'t':>6} {'sink cm':>8} {'pinch m':>8} {'lift cm':>8}")
    for t in range(300):
        r.step()
        sink = float(np.max(z0 - np.array([r.d.xpos[s.body_id][2] for s in slots])))
        d = r.pinch_distance(crates[0]); lift = r.crate_lift(crates[0])
        if total is None and t == 299: pass
        if t_pinch is None and d < PINCH_GATE_M: t_pinch = t*POLICY_DT
        if t_lift is None and lift > 0.01: t_lift = t*POLICY_DT
        if t % 30 == 0 or t == 299:
            print(f"{t*POLICY_DT:6.2f} {100*sink:8.1f} {d:8.3f} {100*lift:8.1f}")
    final_sink = float(np.max(z0 - np.array([r.d.xpos[s.body_id][2] for s in slots])))
    # Time to reach half the eventual descent.
    r.reset(np.random.default_rng(seed))
    for t in range(300):
        r.step()
        s = float(np.max(z0 - np.array([r.d.xpos[x.body_id][2] for x in slots])))
        if t_half_sink is None and s > 0.5*final_sink: t_half_sink = t*POLICY_DT; break
    print(f"  half of the {100*final_sink:.0f} cm descent at t={t_half_sink}s; "
          f"first pinch at t={t_pinch}s; first 1 cm of lift at t={t_lift}s")
    r.close()
PY
