#!/bin/bash
# Inside the container (via bhl_exec). One Full-stage checkpoint per Python
# process (Isaac Lab allows one SimulationContext per process), so the sbatch
# calls this once per arm x seed. Reads VARIANT (arm: Blind|Lidar|Stereo|Both),
# SEED (training seed 0|1|2), CKPT (checkpoint path) and MAZE_NOISE_OUT from the
# environment.
#
# Identical to the published Full evaluation in slurm/inner/weekend_maze.sh
# (--headless --enable_cameras --num-envs 32 --steps 600 --seed 100+SEED, same
# checkpoint) except for --keep-corruption. No --minimum-success and no
# MAZE_PROBE_PASS grep: this measures, it does not gate. The probe writes its
# JSON before it raises on "not every first episode finished", so a non-zero
# exit still leaves the data; the sbatch judges PASS/FAIL from the JSON.
set -uo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
: "${VARIANT:?VARIANT (arm) must be forwarded}"
: "${SEED:?SEED must be forwarded}"
: "${CKPT:?CKPT must be forwarded}"
case "$VARIANT" in Blind|Lidar|Stereo|Both) ;; *) echo "Invalid arm: $VARIANT" >&2; exit 2;; esac
case "$SEED" in 0|1|2) ;; *) echo "Invalid seed: $SEED" >&2; exit 2;; esac
[ -f "$CKPT" ] || { echo "checkpoint missing: $CKPT" >&2; exit 2; }
OUT="${MAZE_NOISE_OUT:-$REPO/results/repo-gpu-20260923/maze-noise}"
mkdir -p "$OUT"
task="Velocity-BHL-MazeRecovery-Full-${VARIANT}-v0"
json="$OUT/${VARIANT}-s${SEED}.json"
log="$OUT/${VARIANT}-s${SEED}.log"
echo "=== inner: $task | seed $((100 + SEED)) | ckpt $CKPT | stack ${BHL_STACK:-?} | $(date) ==="
timeout 540 "$PY" "$REPO/scripts/bench/maze_recovery_probe.py" --headless --enable_cameras \
    --task "$task" --num-envs 32 --steps 600 --seed "$((100 + SEED))" \
    --checkpoint "$CKPT" --keep-corruption --output "$json" > "$log" 2>&1
rc=$?
grep -E "first_episode|observation_corruption|MAZE_PROBE_PASS|RuntimeError|Traceback" "$log" | tail -20 || true
echo "=== inner exit $rc for $task s$SEED ==="
exit "$rc"
