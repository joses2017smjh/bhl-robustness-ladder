#!/bin/bash
#SBATCH -J waiter-vla-smoke
#SBATCH -A eecs
#SBATCH -p gpu,dgxh,ampere
#SBATCH --constraint=a40|rtx8000|l40s|h100|h200|h200-70g
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 0-01:00:00
# SYNTHETIC end-to-end plumbing test of waiter phase 3 (no waiter data, no MuJoCo scene; stub env only):
#   1 synthetic contract episodes (stub expert) + random-array episodes (shape-only audit)
#   2 fine-tune smolvla_base: 20 steps, then a second invocation resumes from step 20 to 30
#   3 closed-loop rollouts on the stub: subprocess env workers under the MuJoCo venv's interpreter (numpy 1.26),
#     and in-process envs; held-out seeds, seen (T0) and held-out (T3) templates
#   4 automatic checks -> <out>/smoke_summary.json
# Usage: slurm_clean sbatch -o <out>/slurm-%j.out scripts/waiter/vla/plumbing_smoke.sbatch <out> [steps1 steps2]
set -euo pipefail
OUT=${1:?output dir}
S1=${2:-20}
S2=${3:-30}
ROOT=${WAITER_VLA_REPO:-$SLURM_SUBMIT_DIR}
[ -f "$ROOT/scripts/waiter/vla/train_smolvla.py" ] || { echo "ROOT=$ROOT is not the repo root" >&2; exit 2; }
source "$ROOT/scripts/waiter/vla/vla_env.sh"
trap vla_cleanup EXIT
vla_banner
mkdir -p "$OUT"
DATA="$OUT/data"
t0=$(date +%s)

echo "[smoke] 1/4 synthetic episodes"
"$VLA_PY" "$VLA_DIR/make_synthetic_episodes.py" --out "$DATA" --episodes 16 --mode expert --first-seed 900000
"$VLA_PY" "$VLA_DIR/make_synthetic_episodes.py" --out "$OUT/data_random" --episodes 4 --mode random --first-seed 900100
"$VLA_PY" "$VLA_DIR/train_smolvla.py" --data "$OUT/data_random" --out "$OUT/audit_random" --cache "$VLA_CACHE" \
    --val-seed-mod 0 --audit-only

COMMON=(--data "$DATA" --out "$OUT/train" --cache "$VLA_CACHE" --val-seed-mod 0 --val-seeds 900001,900005
        --batch-size 8 --save-every 10 --log-every 5 --val-per-episode 2 --openloop-samples 4 --num-workers 2
        --chunk-size 50 --n-action-steps 10 --seed 0)
echo "[smoke] 2/4 train to step $S1, then resume to step $S2"
"$VLA_PY" "$VLA_DIR/train_smolvla.py" "${COMMON[@]}" --steps "$S1"
"$VLA_PY" "$VLA_DIR/train_smolvla.py" "${COMMON[@]}" --steps "$S2" --resume auto

echo "[smoke] 3/4 rollouts on the stub"
"$VLA_PY" "$VLA_DIR/rollout_vla.py" --checkpoint "$OUT/train/best.json" --env stub --seeds 910000-910003 \
    --instruction-ids T0,T3 --templates "$DATA/templates.json" --batch-envs 4 --workers 1 \
    --env-python "$ENV_PY" --out "$OUT/rollout_subproc.json"
"$VLA_PY" "$VLA_DIR/rollout_vla.py" --checkpoint "$OUT/train/latest.json" --env stub --seeds 910004-910005 \
    --instruction-ids T1 --templates "$DATA/templates.json" --batch-envs 2 --workers 0 --n-action-steps 5 \
    --max-steps 100 --out "$OUT/rollout_inproc.json"

echo "[smoke] 4/4 checks"
"$VLA_PY" - "$OUT" "$S1" "$S2" "$(( $(date +%s) - t0 ))" <<'EOF'
import json, sys
from pathlib import Path
out, s1, s2, wall = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
checks = {}
done = json.loads((out / "train/completed.json").read_text())
checks["train_completed_all_steps"] = done["steps_done"] == s2 and done["complete"]
log = (out / "train/train.log").read_text()
checks["resumed_from_step_s1"] = f"at step {s1}" in log and "resuming from" in log
rows = [json.loads(l) for l in (out / "train/metrics.jsonl").read_text().splitlines()]
val = [r for r in rows if "validation" in r]
checks["validation_at_0_and_saves"] = [r["step"] for r in val][0] == 0 and val[-1]["step"] == s2
checks["val_splits_present"] = set(val[-1]["validation"]) >= {"val_layout", "val_template", "train_probe"}
train_rows = [r for r in rows if "loss" in r]
checks["finite_train_loss"] = all(r["loss"] == r["loss"] and r["loss"] < 1e6 for r in train_rows)
split = json.loads((out / "train/data_split.json").read_text())
checks["held_out_template_not_trained"] = "T3" not in split["splits"]["train"]["templates"] and \
    split["splits"]["val_template"]["templates"] == ["T3"]
checks["norm_stats_train_only"] = json.loads((out / "train/norm_stats.json").read_text())["data_hash"] == split["data_hash"]
audit = json.loads((out / "audit_random/data_split.json").read_text())
checks["random_shape_audit"] = sum(v["episodes"] for v in audit["splits"].values()) == 4
r1 = json.loads((out / "rollout_subproc.json").read_text())
r2 = json.loads((out / "rollout_inproc.json").read_text())
checks["rollout_subproc_complete"] = r1["completed"] and len(r1["episodes"]) == 8
checks["rollout_inproc_complete"] = r2["completed"] and len(r2["episodes"]) == 2
checks["rollouts_labelled_synthetic"] = r1["synthetic"] and r2["synthetic"]
checks["latency_measured"] = r1["latency"]["inference_call"] is not None
checks["no_render_suspect"] = r1["summary"]["render_suspect"] == 0 and r2["summary"]["render_suspect"] == 0
checks["held_out_templates_labelled"] = r1["summary"]["held_out_templates"]["episodes"] == 4
summary = {"SYNTHETIC": "plumbing only; stub env; nothing here is a waiter result", "checks": checks,
           "all_pass": all(checks.values()), "wall_s": wall,
           "train": {"best_metric": done["best_metric"], "elapsed_s": done["elapsed_s"], "gpu": done["env"]["gpu"]},
           "last_validation": val[-1]["validation"], "first_validation": val[0]["validation"],
           "rollout_subproc": {"summary": r1["summary"], "latency": r1["latency"]},
           "rollout_inproc": {"summary": r2["summary"], "latency": r2["latency"]}}
(out / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(checks, indent=1))
print("[smoke] ALL PASS" if summary["all_pass"] else "[smoke] FAILURES")
sys.exit(0 if summary["all_pass"] else 1)
EOF
