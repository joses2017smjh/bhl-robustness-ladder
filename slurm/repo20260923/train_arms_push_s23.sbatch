#!/bin/bash
#SBATCH --job-name=arms-push-s23
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --array=2-3
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Repo-wide campaign (2026-09-23): two more training seeds of the 22-DoF
# arms-push arm (seeds 2 and 3), each followed by its ONNX/deploy.yaml export.
#
# DEVIATION FROM THE BRIEF, stated up front: the brief asked for seeds 1 and 2
# on the premise that the arms-push n=60 result came from one seed. It did not.
# logs/rsl_rl/humanoid/ holds 2026-08-18_20-57-50_arms-push-s0 AND
# 2026-08-18_20-58-05_arms-push-s1 (61 checkpoints each, both exported), and
# the n=60 in docs/REPO_TASKS.md LOC-05 / docs/gifs/arms_push_pair.json is
# arms-push-s0 (7/30) + arms-push-s1 (2/30) from results/arms. Training seed 1
# again would create a second *_arms-push-s1 directory that every resolver in
# the repo (`ls -d *_<run> | head -1`) would ignore in favour of the old one:
# wasted and invisible. So this array trains seeds 2 and 3, taking the arm to
# four seeds (n = 120 under the matched protocol).
#
# Training: slurm/30_arms.sbatch, condition 2 (COND = IDX / 2 = 2, the
# arms-push arm) replicated exactly -- TASK=Velocity-BHL-Arms-PushAdaptive-v0,
# RUN_NAME=arms-push-s${SEED}, MAX_ITER=6000, no Hydra overrides, v51 stack
# (default), partition gpu, 4 CPU / 48 GB / 1 GPU / 20 h. The array index IS the
# seed here (30_arms derived SEED = IDX % 2; this file uses --array=2-3 and
# SEED = SLURM_ARRAY_TASK_ID). s0 and s1 each took ~4 h 47 min
# (model_0.pt -> model_5999.pt), so 20 h leaves ample room for the export.
#
# Export: the path the s0/s1 exports took (slurm/31_arms_eval.sbatch ->
# slurm/inner/export_one_humanoid.sh). 31_arms_eval set
# TASK=Velocity-Berkeley-Humanoid-Lite-v0 for the export; export_one_humanoid
# reads ${TASK:-...}, so TASK must be re-exported here or the export would run
# through the PushAdaptive overlay instead of the same code path s0/s1 used.
# train_play.py writes configs/policy_latest.yaml inside a try/except that only
# warns, and export_one_humanoid.sh's sed then repoints policy_checkpoint_path,
# so "points at this run" cannot tell a stale YAML from a fresh one. The
# verification below therefore also requires deploy.yaml to be config-equivalent
# (all but the checkpoint path and the sampled command) to arms-push-s0's, and
# fails the task if it is not. The rescore is submitted with afterany, so a
# FAIL here surfaces as INCOMPLETE there, never as a score.
#
# Resubmit-safe: training is skipped when arms-push-s<SEED> already has its final
# checkpoint (model_5999.pt, MAX_ITER=6000); the export runs only when
# exported/deploy.yaml is absent. A deploy.yaml that failed verification is
# moved aside (.bad-<jobid>), so a resubmit re-exports (no retrain) rather than
# skipping straight back to the failed check. A partial run (no model_5999.pt)
# is retrained into a new timestamped dir; find_run picks the newest.
set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

SEED=$SLURM_ARRAY_TASK_ID
export SEED MAX_ITER=6000
export TASK=Velocity-BHL-Arms-PushAdaptive-v0 RUN_NAME="arms-push-s${SEED}"
export OVERRIDE_FILE=""
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS RUN_DIR"

LOGROOT=$UPSTREAM/logs/rsl_rl/humanoid

find_run() {  # newest run dir for RUN_NAME (train.py prefixes a timestamp)
    ls -d "$LOGROOT"/*_"$RUN_NAME" 2>/dev/null | sort | tail -1
}

echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | $(date) ==="
existing=$(find_run || true)
if [ -n "$existing" ] && [ -f "$existing/model_5999.pt" ]; then
    echo "already trained (model_5999.pt present): $existing -- skipping training"
else
    bhl_exec "$REPO/slurm/inner/train.sh"
    echo "=== training done: $(date) ==="
fi

run=$(find_run || true)
[ -n "$run" ] || { echo "EXPORT: FAIL -- no run directory matching *_$RUN_NAME under $LOGROOT"; exit 1; }
ls "$run"/model_*.pt >/dev/null 2>&1 || { echo "EXPORT: FAIL -- no checkpoint in $run"; exit 1; }
export RUN_DIR=$(basename "$run")

if [ ! -f "$run/exported/deploy.yaml" ]; then
    echo "=== export $RUN_DIR $(date) ==="
    export TASK=Velocity-Berkeley-Humanoid-Lite-v0   # as 31_arms_eval.sbatch did for s0/s1
    bhl_exec "$REPO/slurm/inner/export_one_humanoid.sh" 2>&1 | grep -vE "pthread_setaffinity|EGLError" || true
fi

# Verify the export is this run's and is config-equivalent to the seeds already
# in the pool. The path grep alone is NOT enough: export_one_humanoid.sh's sed repoints
# policy_checkpoint_path on the copied YAML, so a stale configs/policy_latest.yaml
# (today's is a 12-joint mazenav config with num_observations 593) would pass
# it. The normalised diff against arms-push-s0's deploy.yaml (everything except
# the checkpoint path and the sampled command_velocity) is what catches that.
# A missing reference is a FAIL (cannot verify), not a skip.
norm() { awk '/^policy_checkpoint_path:/{next} /^command_velocity:/{skip=1;next} skip&&/^- /{next} {skip=0; print}' "$1"; }
cfg="$run/exported/deploy.yaml"
ref=$(ls -d "$LOGROOT"/*_arms-push-s0 2>/dev/null | sort | tail -1 || true)
ref_cfg="${ref:+$ref/exported/deploy.yaml}"
ok=1
[ -f "$cfg" ] && [ -f "$run/exported/policy.onnx" ] || ok=0
[ -n "$ref_cfg" ] && [ -f "$ref_cfg" ] || { echo "EXPORT: cannot verify -- reference $LOGROOT/*_arms-push-s0/exported/deploy.yaml not found"; ok=0; }
if [ "$ok" = 1 ]; then
    grep -q "^policy_checkpoint_path:.*$RUN_DIR/exported/policy.onnx" "$cfg" || ok=0
    grep -qE "^num_joints: 22$" "$cfg" || ok=0
    if ! diff <(norm "$ref_cfg") <(norm "$cfg"); then
        echo "deploy.yaml differs from $ref_cfg (above: < reference, > this run)"; ok=0
    fi
fi
if [ "$ok" = 1 ]; then
    echo "EXPORT: OK $cfg (config-equivalent to $(basename "$ref"))"
else
    echo "EXPORT: FAIL $cfg (missing, points elsewhere, num_joints != 22, or config differs from arms-push-s0)"
    if [ -f "$cfg" ]; then
        grep -E "^(policy_checkpoint_path|num_joints|num_observations):" "$cfg" || true
        # Quarantine the bad export: training is skipped on model_5999.pt, export
        # runs when deploy.yaml is absent, so a resubmit re-exports (no retrain)
        # instead of skipping straight back to this check. The rescore's score()
        # ignores runs without deploy.yaml, so .bad-* is never scored.
        mv "$cfg" "$cfg.bad-${SLURM_JOB_ID:-local}"
    fi
    exit 1
fi
echo "=== done: $(date) ==="
