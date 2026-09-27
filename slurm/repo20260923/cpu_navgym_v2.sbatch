#!/bin/bash
#SBATCH --job-name=navgym-v2
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=26:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-09-26: NavGym v2 -- the v1 post-mortem fixes (round 0.22 m footprint,
# route-scaled time limit, -5 crash / -0.002 step / continuous potential-based
# shaping, a 2 m near-field lidar key) in MazeNavEnv(version=2), trained with
# SB3 PPO on CPU (scripts/bench/navgym_train.py --env-version 2).
#
#   NAVGYM_PPO=v2  -> Arm A: v2 env + v2 PPO (target_kl 0.02, LR 3e-4 -> 3e-5,
#                    ent 0.01, log_std_init -0.5, std floor 0.2, gamma 0.998)
#   NAVGYM_PPO=v1  -> Arm B: v2 env + v1 PPO settings (the ablation; the env's
#                    shaping gamma follows the learner, 0.995)
#   NAVGYM_SEED    -> 0 or 1 (default 0);  NAVGYM_STEPS default 30 000 000
#   output: results/navgym-v2-20260926/arm{A,B}-s<seed>/
#
# Acceptance gate E0 (run on the login node before any submission, 2026-09-26,
# scripts/bench/navgym_scripted_gate.py): the scripted A* (inflate 0.45) +
# turn-then-walk driver inside the v2 gym, randomized dynamics, held-out seeds:
# 5x5 24/24 goals, 6x6 22/24 goals (2 time-outs), 0 collisions -> PASS.
#
# Wall time: v1 trained at ~1 070 steps/s on 16 CPUs (incl. its serial eval). On
# the 2-CPU login node (load 6-11) the interleaved v2/v1 training-throughput ratio
# was 0.67-1.94 (median 1.78; target_kl early-stops epochs); sized on the worst,
# 0.67 -> ~715 steps/s -> 30 M steps ~11.7 h, + <= 1.1 h of parallel held-out
# evals (16 eval workers) -> ~12.8 h; x2 margin -> 26 h.
#
# PREDECLARED (2026-09-26, before any v2 training):
#   Arm A PASSES iff for BOTH seeds (0 and 1) the FINAL actor's 48-maze held-out
#   evaluation (maze seeds 10000-10047, v2 time limit, deterministic actions)
#   gives 5x5 success >= 0.80 AND 6x6 success >= 0.65 AND 6x6 collision rate
#   <= 0.15, AND train/std at the end >= 0.2 (the last update's std BEFORE the
#   floor clamp, summary.json "train_std_last_update"). Anything else is
#   NEGATIVE at this budget. The rule is judged on the final actor only: the
#   best checkpoint is selected on seeds 10000-10023, a subset of the 48, so
#   its 48-maze numbers are reported beside it as selection-biased.
#   Arm B is the ablation: reported with the same numbers, not gated.
#   If Arm A passes, its best-checkpoint actor (actor_best.onnx) is transferred
#   to the physics robot on the 12 hard-6x6 seeds of
#   results/maze-explore-20260924/hard-6x6 via maze_explore.py --policy and
#   reported beside the A* table (not gated here).
# The per-seed verdict below is computed from summary.json, never from the exit
# code; the job that finishes second also prints the arm verdict for both seeds.
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
SEED=${NAVGYM_SEED:-0}
PPO=${NAVGYM_PPO:-v2}
STEPS=${NAVGYM_STEPS:-30000000}
case "$PPO" in
  v2) ARM=A ;;
  v1) ARM=B ;;
  *) echo "NAVGYM_PPO must be v1 or v2, got '$PPO'"; exit 2 ;;
esac
ROOT=${NAVGYM_ROOT:-$REPO/results/navgym-v2-20260926}
OUT=$ROOT/arm${ARM}-s${SEED}
mkdir -p "$OUT"
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | arm $ARM (env v2, ppo $PPO) | seed $SEED | steps $STEPS | $(date) ==="
set +e
"$PY" scripts/bench/navgym_train.py --out "$OUT" --seed "$SEED" --total-steps "$STEPS" --env-version 2 --ppo "$PPO" \
    --n-envs "${NAVGYM_ENVS:-16}" --eval-every "${NAVGYM_EVAL_EVERY:-1000000}" 2>&1 | grep --line-buffered -vE "Warning|warnings.warn"
RC=${PIPESTATUS[0]}
set -e
echo "trainer exit code $RC (informational; the verdict is computed from summary.json)"
"$PY" - "$ROOT" "$ARM" "$SEED" <<'PYEOF'
import json, sys
from pathlib import Path
root, arm, seed = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
BAR = {"5x5_success_min": 0.80, "6x6_success_min": 0.65, "6x6_collision_max": 0.15, "std_min": 0.2, "n": 48}

def judge(s):
    ev = s["final_eval_heldout_48"]
    a, b = ev["5x5"], ev["6x6"]
    std = s.get("train_std_last_update")
    checks = {"n48": a["n"] == BAR["n"] and b["n"] == BAR["n"], "5x5_success": a["success"] >= BAR["5x5_success_min"],
              "6x6_success": b["success"] >= BAR["6x6_success_min"], "6x6_collision": b["collision"] <= BAR["6x6_collision_max"],
              "std": std is not None and std >= BAR["std_min"]}
    nums = (f"5x5 {a['success']:.3f} (coll {a['collision']:.3f}, t/o {a['time_out']:.3f}) | 6x6 {b['success']:.3f} "
            f"(coll {b['collision']:.3f}, t/o {b['time_out']:.3f}) | 4x4 {ev['4x4']['success']:.3f} | std {std}")
    return all(checks.values()), checks, nums

def load(sd):
    p = root / f"arm{arm}-s{sd}" / "summary.json"
    return json.loads(p.read_text()) if p.is_file() else None

s = load(seed)
if s is None:
    print(f"NAVGYM-V2 VERDICT arm {arm} seed {seed}: NO SUMMARY -> the run did not finish; NEGATIVE (no result)")
    sys.exit(0)
ok, checks, nums = judge(s)
best = s.get("best") or {}
if best.get("final_eval_heldout_48"):
    be = best["final_eval_heldout_48"]
    print(f"best checkpoint @ {best.get('timesteps')} steps (selection-biased): 5x5 {be['5x5']['success']:.3f} | 6x6 {be['6x6']['success']:.3f} "
          f"(coll {be['6x6']['collision']:.3f})")
if arm == "A":
    print(f"NAVGYM-V2 VERDICT arm A seed {seed}: {'MEETS' if ok else 'MISSES'} the per-seed bar | {nums} | checks {checks}")
    other = load(1 - seed)
    if other is not None:
        ok2, _, nums2 = judge(other)
        print(f"NAVGYM-V2 ARM A (both seeds): {'PASS' if ok and ok2 else 'NEGATIVE at this budget'} | s{seed}: {nums} | s{1 - seed}: {nums2}")
    else:
        print(f"NAVGYM-V2 ARM A: waiting for seed {1 - seed}; the arm passes only if both seeds meet the bar")
else:
    print(f"NAVGYM-V2 ARM B (ablation, not gated) seed {seed}: {nums} | would-meet-A-bar {ok}")
PYEOF
echo "=== done: $(date) ==="
