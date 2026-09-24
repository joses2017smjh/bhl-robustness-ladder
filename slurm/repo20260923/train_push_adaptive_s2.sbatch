#!/bin/bash
#SBATCH --job-name=push-adapt-s2
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Repo-wide campaign (2026-09-23): third training seed of the adaptive push arm,
# then its ONNX/deploy.yaml export, in one job.
#
# Why: the matched push protocol (SLURM_JOBS.md, results/push-matched-20260923/
# push_matched_summary.json) gave push-adaptive 6/60 vs dr-default 19/90,
# one-sided Fisher p = 0.056, with only two push-adaptive seeds (s0, s1) against
# three dr-default seeds. This seed brings the arm to n = 90, matching the
# control's pool size. The rescore (cpu_push_rescore.sbatch) is a separate CPU
# job that depends on this one.
#
# Training: a verbatim copy of slurm/15_push_adaptive.sbatch (the launcher that
# produced s0 and s1 on 2026-08-17/18) with the array replaced by SEED=2:
#   TASK=Velocity-BHL-Biped-PushAdaptive-v0, MAX_ITER=6000, v51 stack (default),
#   partition gpu, 4 CPU / 48 GB / 1 GPU / 16 h. s0 and s1 each took ~3 h 45 min
#   (model_0.pt -> model_5999.pt timestamps), so 16 h leaves ample room for the
#   export step below.
#
# Export: exactly the path the s0/s1 exports took (slurm/20_evaluate_all.sbatch
# -> slurm/inner/export_one.sh, which runs scripts/train_play.py on the upstream
# Biped task, copies configs/policy_latest.yaml to <run>/exported/deploy.yaml
# and repoints policy_checkpoint_path). train_play.py writes policy_latest.yaml
# inside a try/except that only warns, so a stale YAML from an earlier export
# could be copied silently. export_one.sh's sed then repoints
# policy_checkpoint_path, so "points at this run" cannot tell a stale YAML from
# a fresh one; the verification below therefore also requires deploy.yaml to be
# config-equivalent (all but the checkpoint path and the sampled command) to
# push-adaptive-s0's, and fails the job if it is not. The rescore is submitted
# with afterany, so a FAIL here surfaces as INCOMPLETE there, never as a score.
#
# Resubmit-safe: training is skipped when push-adaptive-s2 already has its final
# checkpoint (model_5999.pt, MAX_ITER=6000); the export runs only when
# exported/deploy.yaml is absent. A deploy.yaml that failed verification is
# moved aside (.bad-<jobid>), so a resubmit re-exports (no retrain) rather than
# skipping straight back to the failed check. A partial run (no model_5999.pt)
# is retrained into a new timestamped dir; find_run picks the newest.
set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

export TASK=Velocity-BHL-Biped-PushAdaptive-v0
export SEED=2
export RUN_NAME="push-adaptive-s${SEED}"
export MAX_ITER=6000
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS RUN_DIR"

LOGROOT=$UPSTREAM/logs/rsl_rl/biped

find_run() {  # newest run dir for RUN_NAME (train.py prefixes a timestamp)
    ls -d "$LOGROOT"/*_"$RUN_NAME" 2>/dev/null | sort | tail -1
}

echo "=== node: $(hostname) | adaptive push | seed: $SEED | $(date) ==="
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
    bhl_exec "$REPO/slurm/inner/export_one.sh" 2>&1 | grep -vE "pthread_setaffinity|EGLError" || true
fi

# Verify the export is this run's and is config-equivalent to the seeds already
# in the pool. The path grep alone is NOT enough: export_one.sh's sed repoints
# policy_checkpoint_path on the copied YAML, so a stale configs/policy_latest.yaml
# (today's is a 12-joint mazenav config with num_observations 593) would pass
# it. The normalised diff against push-adaptive-s0's deploy.yaml (everything except
# the checkpoint path and the sampled command_velocity) is what catches that.
# A missing reference is a FAIL (cannot verify), not a skip.
norm() { awk '/^policy_checkpoint_path:/{next} /^command_velocity:/{skip=1;next} skip&&/^- /{next} {skip=0; print}' "$1"; }
cfg="$run/exported/deploy.yaml"
ref=$(ls -d "$LOGROOT"/*_push-adaptive-s0 2>/dev/null | sort | tail -1 || true)
ref_cfg="${ref:+$ref/exported/deploy.yaml}"
ok=1
[ -f "$cfg" ] && [ -f "$run/exported/policy.onnx" ] || ok=0
[ -n "$ref_cfg" ] && [ -f "$ref_cfg" ] || { echo "EXPORT: cannot verify -- reference $LOGROOT/*_push-adaptive-s0/exported/deploy.yaml not found"; ok=0; }
if [ "$ok" = 1 ]; then
    grep -q "^policy_checkpoint_path:.*$RUN_DIR/exported/policy.onnx" "$cfg" || ok=0
    grep -qE "^num_joints: 12$" "$cfg" || ok=0
    if ! diff <(norm "$ref_cfg") <(norm "$cfg"); then
        echo "deploy.yaml differs from $ref_cfg (above: < reference, > this run)"; ok=0
    fi
fi
if [ "$ok" = 1 ]; then
    echo "EXPORT: OK $cfg (config-equivalent to $(basename "$ref"))"
else
    echo "EXPORT: FAIL $cfg (missing, points elsewhere, num_joints != 12, or config differs from push-adaptive-s0)"
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
