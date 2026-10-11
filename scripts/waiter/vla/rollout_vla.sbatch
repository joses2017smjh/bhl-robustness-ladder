#!/bin/bash
#SBATCH -J waiter-vla-rollout
#SBATCH -A eecs
#SBATCH -p gpu,dgxh,ampere
#SBATCH --constraint=a40|rtx8000|l40s|h100|h200|h200-70g
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgxh-1
#SBATCH -c 12
#SBATCH --mem=48G
#SBATCH -t 0-06:00:00
# Waiter phase 3: closed-loop rollouts. The policy runs on the GPU in venv-lehome-train; each env runs in its own
# worker process under ENV_PY (main venv, MuJoCo 3.3.5, MUJOCO_GL=egl), so -c should be >= --batch-envs + 2.
#
#   source slurm/_env.sh
#   slurm_clean sbatch -o <dir>/slurm-%j.out scripts/waiter/vla/rollout_vla.sbatch \
#       --checkpoint <train out>/best.json --env waiter --deploy ... --upstream ... --cache-dir ... \
#       --seeds 1000-1019 --templates <templates.json> --instruction-ids ... --batch-envs 8 --out <dir>/dev.json
# Dev seeds only (1000-1999). The scored run (4000-4099) needs --purpose scored --gate <frozen gate file>.
# dgxh-1 is excluded (MIG slices have no EGL: software-rendered camera images that do not match the demonstrations);
# a run with --env waiter first runs scripts/waiter/render_check.py and stops if this node renders differently.
set -euo pipefail
ROOT=${WAITER_VLA_REPO:-$SLURM_SUBMIT_DIR}
[ -f "$ROOT/scripts/waiter/vla/rollout_vla.py" ] || { echo "ROOT=$ROOT is not the repo root" >&2; exit 2; }
source "$ROOT/scripts/waiter/vla/vla_env.sh"
trap vla_cleanup EXIT
vla_banner
if [ "${WAITER_SKIP_RENDER_CHECK:-0}" != 1 ] && printf '%s\n' "$@" | grep -qx waiter; then
    "$ENV_PY" "$ROOT/scripts/waiter/render_check.py"
fi
"$VLA_PY" "$VLA_DIR/rollout_vla.py" --env-python "$ENV_PY" "$@"
