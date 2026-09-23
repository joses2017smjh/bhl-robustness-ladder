#!/bin/bash
# Gate-only rerun for an already-trained BothRobust stage (the first Approach
# gate failed only because the probe built an MLP runner for a recurrent
# checkpoint). Mirrors the gate section of slurm/inner/weekend_maze.sh.
# Usage (inside bhl_exec): inner_sf04_gate.sh <Stage>
set -euo pipefail
stage=${1:?Approach Corridor Full}; arm=BothRobust; seed=0
export BHL_POLICY=recurrent
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
task="Velocity-BHL-MazeRecovery-${stage}-${arm}-v0"
outdir="$REPO/results/weekend-20260919"
RUN_NAME="wknd-${stage,,}-${arm,,}-s${seed}"
checkpoint=$("$PY" "$REPO/scripts/bench/maze_checkpoint.py" resolve --log-root "$UPSTREAM/logs/rsl_rl/biped" --run-name "$RUN_NAME")
echo "gate: $task <- $checkpoint"
output="$outdir/maze-eval-${stage}-${arm}-s${seed}.json"
log=$(mktemp "${TMPDIR:-/tmp}/bhl-maze-eval-XXXXXX.log")
"$PY" "$REPO/scripts/bench/maze_recovery_probe.py" --headless --enable_cameras \
    --task "$task" --num-envs 32 --steps 600 --seed "$((100 + seed))" \
    --checkpoint "$checkpoint" --minimum-success 0.30 --output "$output" 2>&1 | tee "$log"
grep -q MAZE_PROBE_PASS "$log"
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"] and d["gate_kind"] == "policy_evaluation" and d["first_episode_success_rate"] >= .3 and d["checkpoint"] == sys.argv[2], d' "$output" "$checkpoint"
echo "MAZE_STAGE_PASS stage=$stage arm=$arm seed=$seed checkpoint=$checkpoint"
