#!/bin/bash
#SBATCH --job-name=er-obs1-ow
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint=a40|rtx8000|l40s|h100|h200|h200-70g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# ER-OBS-1-OW (2026-10-04, the user: "let's do this since we can just use the hpc compute"): the ER-OBS-1 observer
# check with OPEN-WEIGHT vision-language models on our own GPUs, a separate arm from the Gemini Robotics-ER 2 arm
# (design G). Code: scripts/bench/er_obs_local.py (new); the Gemini arm's prompt, parser, frames, labels and pure
# scoring functions are reused unchanged.
#   Models (frozen, Hugging Face, Apache-2.0, pinned commits; downloaded to the project HF cache, 35 GB):
#     molmo2-er    allenai/Molmo2-ER          dab22564403d2607855bb1fffb0721285b445081  (5B, Qwen3-4B + SigLIP2)
#     qwen3-vl-8b  Qwen/Qwen3-VL-8B-Instruct  0c351dd01ed87e9c1b53cbc748cba10e6187ff3b  (9B)
#   Inference (frozen): loaded as each model card does (AutoProcessor + AutoModelForImageTextToText, remote code for
#   Molmo2) from the pinned local snapshot, offline; bf16; batch 1; greedy (do_sample False); max_new_tokens 256; the
#   processor's own chat template with one user turn = [the PNG frame, the Gemini arm's exact text part]
#   (client.TEXT, prompt+schema sha256 873d16aed67f87f36fff88dbe7deb2ad6ec55abd0ce71c1a9e581f059191c20b). The answer
#   text goes through client.classify (the Gemini arm's parser) unchanged.
#   Frames (scored): er-obs1-data/frames-run from job 21544356 (seeds 300-319, 400 frames, every class >= 20).
#   Smoke frames: er-obs1-data/frames-smoke/21544355 (throw-away seeds 900/901).
# RULE (design G's, verbatim, applied to each model separately): "ACCURACY PASS iff balanced accuracy >= 0.90 on all
#   4 booleans, the point hit rate is >= 0.95 on cube-visible frames, and control C1 holds (same frames with the cube
#   rendered invisible must score <= 0.60 balanced accuracy). INCOMPLETE if any class has fewer than 20 frames.
#   IN-LOOP-ELIGIBLE is a separate verdict: p95 latency <= 1.0 s over 50 calls." As applied in the Gemini arm: a
#   failed or malformed answer scores WRONG on main frames and CORRECT on C1 frames; C1 booleans are read on their own
#   (whatever the point holds); point hit = the [y, x] 0-1000 point lands on a cube pixel of the segmentation mask;
#   a missing outcome makes the run INCOMPLETE. Latency: 50 sequential calls on the first 50 scored main frames after
#   one warm-up call, wall time from the chat template to the decoded text, on ONE FULL GPU (the latency mode is
#   submitted with --partition=gpu,ampere: no MIG slice); the GPU is recorded, and the verdict holds for that GPU only.
# Labels: LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms + OPEN-WEIGHT VLM observer (local GPU, rendered RGB,
#   offline); ORACLE labels for scoring only.
# Development before freezing (never scored): probe 21549431 on 3 smoke frames per model (all answers parsed; 1.1 s /
#   0.9 s per call on an H100 MIG slice); two failed probe attempts (21549333: a V100 has no kernels in this CUDA 13
#   build; Molmo2's processor did not load through the generic pipeline, so loading is explicit).
#
# usage: sbatch --export=ALL,ER_OW_MODEL=<molmo2-er|qwen3-vl-8b> gpu_er_obs1_ow.sbatch smoke|run
#        sbatch --partition=gpu,ampere --export=ALL,ER_OW_MODEL=<m> gpu_er_obs1_ow.sbatch latency
# Outputs (never overwritten): er-obs1-data/ow/<model>/{calls-smoke-<job>,calls-run,latency}/ ; verdicts and calls.json
# copied to results/er-obs1-20261004/ow/<model>/.
set -euo pipefail
MODE=${1:?usage: gpu_er_obs1_ow.sbatch smoke|run|latency}
MODEL=${ER_OW_MODEL:?set ER_OW_MODEL=molmo2-er|qwen3-vl-8b}
case "$MODEL" in molmo2-er|qwen3-vl-8b) ;; *) echo "ER-OBS-1-OW: REFUSED (unknown model $MODEL)"; exit 2;; esac
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv-isaac60/bin/python     # transformers 4.57.6 (Qwen3-VL needs >= 4.57)
DATA=/nfs/hpc/share/sanchej7/Humanoid_Lite/er-obs1-data
OUT=$DATA/ow/$MODEL
RES=$REPO/results/er-obs1-20261004/ow/$MODEL
export TMPDIR=/tmp HF_HOME=/nfs/hpc/share/sanchej7/Humanoid_Lite/.cache/huggingface HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONPATH=$REPO/src PYTHONUNBUFFERED=1
cd "$REPO"
echo "=== ER-OBS-1-OW $MODE $MODEL | job ${SLURM_JOB_ID:-local} | $(hostname) | HEAD $(git rev-parse --short HEAD) | $(date)"
nvidia-smi -L || true
L="$PY scripts/bench/er_obs_local.py"
case "$MODE" in
  smoke)
    case "${SLURM_JOB_NAME:-}" in *smoke*) ;; *) echo "ER-OBS-1-OW: REFUSED (a smoke job's name must contain 'smoke')"; exit 1;; esac
    S=$OUT/calls-smoke-${SLURM_JOB_ID:-local}
    $L calls --model "$MODEL" --mode smoke --frames "$DATA/frames-smoke/21544355/frames" --out "$S"
    $L verdict accuracy --frames "$DATA/frames-smoke/21544355/frames" --calls "$S/calls.json" --out "$S/verdict_accuracy.json"
    echo "ER-OBS-1-OW SMOKE: PASS ($MODEL; a pipeline check on throw-away frames, not a result)" ;;
  run)
    [ -e "$OUT/calls-run" ] && { echo "ER-OBS-1-OW: REFUSED ($OUT/calls-run exists; never overwritten)"; exit 1; }
    $L calls --model "$MODEL" --mode run --frames "$DATA/frames-run" --out "$OUT/calls-run"
    mkdir -p "$RES"
    $L verdict accuracy --frames "$DATA/frames-run" --calls "$OUT/calls-run/calls.json" --out "$RES/verdict_accuracy.json"
    cp "$OUT/calls-run/calls.json" "$RES/calls.json"
    echo "ER-OBS-1-OW ACCURACY RESULT ($MODEL): $($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['verdict'])" "$RES/verdict_accuracy.json") (read from $RES/verdict_accuracy.json)" ;;
  latency)
    nvidia-smi -L | grep -q "MIG" && { echo "ER-OBS-1-OW: REFUSED (latency needs a full GPU, not a MIG slice)"; exit 1; }
    [ -e "$OUT/latency" ] && { echo "ER-OBS-1-OW: REFUSED ($OUT/latency exists)"; exit 1; }
    $L latency --model "$MODEL" --frames "$DATA/frames-run" --out "$OUT/latency"
    mkdir -p "$RES"
    $L verdict latency --latency "$OUT/latency/latency.json" --out "$RES/verdict_latency.json"
    cp "$OUT/latency/latency.json" "$RES/latency.json"
    echo "ER-OBS-1-OW LATENCY RESULT ($MODEL): $($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['verdict'])" "$RES/verdict_latency.json") (read from $RES/verdict_latency.json)" ;;
  *) echo "ER-OBS-1-OW: REFUSED (mode $MODE)"; exit 2 ;;
esac
echo "=== done $(date)"
