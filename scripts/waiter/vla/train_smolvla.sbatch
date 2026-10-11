#!/bin/bash
#SBATCH -J waiter-vla-train
#SBATCH -A eecs
#SBATCH -p gpu,dgxh,ampere
#SBATCH --constraint=a40|rtx8000|l40s|h100|h200|h200-70g
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
# Waiter phase 3: SmolVLA fine-tune on one GPU (venv-lehome-train: torch 2.7.0+cu128, no V100).
#
# Submit from the repo/worktree root (or set WAITER_VLA_REPO):
#   source slurm/_env.sh
#   slurm_clean sbatch -o <out>/slurm-%j.out scripts/waiter/vla/train_smolvla.sbatch \
#       --data <contract npz dir> --out <out> --steps 20000 --max-minutes 1380 [train_smolvla.py args]
# Re-submitting the same command resumes from <out>/latest.json (--resume auto). --max-minutes should sit below
# the -t limit so the last checkpoint is written cleanly.
set -euo pipefail
ROOT=${WAITER_VLA_REPO:-$SLURM_SUBMIT_DIR}
[ -f "$ROOT/scripts/waiter/vla/train_smolvla.py" ] || { echo "ROOT=$ROOT is not the repo root" >&2; exit 2; }
source "$ROOT/scripts/waiter/vla/vla_env.sh"
trap vla_cleanup EXIT
vla_banner
"$VLA_PY" "$VLA_DIR/train_smolvla.py" --cache "$VLA_CACHE" "$@"
