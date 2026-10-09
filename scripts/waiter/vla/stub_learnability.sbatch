#!/bin/bash
#SBATCH -J waiter-vla-learn
#SBATCH -A eecs
#SBATCH -p gpu,dgxh,ampere
#SBATCH --constraint=a40|rtx8000|l40s|h100|h200|h200-70g
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 0-01:00:00
# SYNTHETIC learnability check (stub env; not a waiter result): can the pipeline learn a task whose solution is
# observable from instruction + image? Steps: N expert demos -> fine-tune -> closed-loop rollouts on held-out
# synthetic layouts with a seen template (T0), the held-out template (T3), and a counterfactual control (T0 with
# the side slot swapped: success there means the policy ignores the instruction).
# Usage: slurm_clean sbatch -o <out>/slurm-%j.out scripts/waiter/vla/stub_learnability.sbatch <out> [steps] [demos]
set -euo pipefail
OUT=${1:?output dir}
STEPS=${2:-3000}
DEMOS=${3:-96}
ROOT=${WAITER_VLA_REPO:-$SLURM_SUBMIT_DIR}
[ -f "$ROOT/scripts/waiter/vla/train_smolvla.py" ] || { echo "ROOT=$ROOT is not the repo root" >&2; exit 2; }
source "$ROOT/scripts/waiter/vla/vla_env.sh"
DATA=/tmp/waiter-stub-data-${SLURM_JOB_ID:-$$}
trap 'vla_cleanup; rm -rf "$DATA"' EXIT
vla_banner
mkdir -p "$OUT"
t0=$(date +%s)
"$VLA_PY" "$VLA_DIR/make_synthetic_episodes.py" --out "$DATA" --episodes "$DEMOS" --mode expert --first-seed 900000 \
    --no-compress | tail -2
cp "$DATA/manifest.json" "$DATA/templates.json" "$OUT/"
"$VLA_PY" "$VLA_DIR/train_smolvla.py" --data "$DATA" --out "$OUT/train" --cache "$VLA_CACHE" \
    --val-seed-mod 8 --val-seed-rem 7 --steps "$STEPS" --batch-size 32 --lr 1e-4 --warmup 200 \
    --save-every 500 --log-every 50 --keep-last 1 --val-per-episode 6 --openloop-samples 16 --num-workers 6 \
    --chunk-size 50 --n-action-steps 10 --seed 0 --max-minutes 36
echo "[learn] train wall $(( $(date +%s) - t0 )) s"
R=("$VLA_PY" "$VLA_DIR/rollout_vla.py" --checkpoint "$OUT/train/best.json" --env stub --seeds 910000-910019
   --templates "$OUT/templates.json" --batch-envs 10 --workers 1 --env-python "$ENV_PY")
"${R[@]}" --instruction-ids T0,T3 --out "$OUT/rollout_seen_heldout.json"
"${R[@]}" --instruction-ids T0 --counterfactual-slot side --out "$OUT/rollout_counterfactual_side.json"
"${R[@]}" --instruction-ids T0 --n-action-steps 50 --out "$OUT/rollout_seen_exec50.json"
"$VLA_PY" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
res = {}
for name in ("rollout_seen_heldout", "rollout_counterfactual_side", "rollout_seen_exec50"):
    r = json.loads((out / f"{name}.json").read_text())
    res[name] = {k: r["summary"].get(k) for k in ("all", "seen_templates", "held_out_templates", "terminations")}
    res[name]["latency_ms"] = r["latency"]["inference_call"]
    res[name]["completed"] = r["completed"]
done = json.loads((out / "train/completed.json").read_text()) if (out / "train/completed.json").exists() else \
    json.loads((out / "train/interrupted.json").read_text())
rows = [json.loads(l) for l in (out / "train/metrics.jsonl").read_text().splitlines()]
val = [r for r in rows if "validation" in r]
res["SYNTHETIC"] = "stub env learnability check; not a waiter result"
res["train"] = {"steps_done": done["steps_done"], "elapsed_s": done["elapsed_s"], "gpu": done["env"]["gpu"],
                "first_val": val[0]["validation"], "last_val": val[-1]["validation"]}
(out / "learnability_summary.json").write_text(json.dumps(res, indent=2) + "\n")
for name in ("rollout_seen_heldout", "rollout_counterfactual_side", "rollout_seen_exec50"):
    print(name, json.dumps(res[name]["all"]), "held-out:", json.dumps(res[name].get("held_out_templates")))
EOF
echo "[learn] total wall $(( $(date +%s) - t0 )) s"
