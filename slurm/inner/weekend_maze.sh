#!/bin/bash
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
mode=${1:?smoke or train}
stage=${2:?stage}
arm=${3:?arm}
seed=${4:?seed}
shift 4
case "$stage" in Approach|Corridor|Full) ;; *) echo "Invalid stage: $stage" >&2; exit 2;; esac
case "$arm" in Blind|Lidar|Stereo|Both|BothRobust) ;; *) echo "Invalid arm: $arm" >&2; exit 2;; esac
task="Velocity-BHL-MazeRecovery-${stage}-${arm}-v0"
outdir="$REPO/results/weekend-20260919"
mkdir -p "$outdir"
if [ "$mode" = smoke ]; then
    output="$outdir/maze-smoke-${stage}-${arm}-s${seed}.json"
    log=$(mktemp "${TMPDIR:-/tmp}/bhl-maze-gate-XXXXXX.log")
    "$PY" "$REPO/scripts/bench/maze_recovery_probe.py" --headless --enable_cameras \
        --task "$task" --num-envs 16 --steps 600 --seed "$seed" --output "$output" 2>&1 | tee "$log"
    grep -q MAZE_PROBE_PASS "$log"
    "$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' "$output"
elif [ "$mode" = train ]; then
    export TASK=$task RUN_NAME="wknd-${stage,,}-${arm,,}-s${seed}"
    export SEED=$seed NUM_ENVS=${1:-1024} MAX_ITER=${2:-500} ENABLE_CAMERAS=1
    resume_stage=${3:-}
    case "$stage" in
        Approach)
            [ -z "$resume_stage" ] || { echo "Approach pilot starts from scratch" >&2; exit 2; }
            unset OVERRIDE_FILE
            ;;
        Corridor|Full)
            expected=Approach
            [ "$stage" != Full ] || expected=Corridor
            [ "$resume_stage" = "$expected" ] || {
                echo "$stage requires passed $expected checkpoint; pass $expected after the iteration count" >&2
                exit 2
            }
            export OVERRIDE_FILE="$outdir/maze-resume-${stage}-${arm}-s${seed}.txt"
            "$PY" "$REPO/scripts/bench/maze_checkpoint.py" resume \
                --gate "$outdir/maze-eval-${resume_stage}-${arm}-s${seed}.json" \
                --task "Velocity-BHL-MazeRecovery-${resume_stage}-${arm}-v0" \
                --output "$OVERRIDE_FILE"
            ;;
    esac
    started=$(date +%s)
    bash "$REPO/slurm/inner/train.sh"
    checkpoint=$("$PY" "$REPO/scripts/bench/maze_checkpoint.py" resolve \
        --log-root "$UPSTREAM/logs/rsl_rl/biped" --run-name "$RUN_NAME" --newer-than "$started")
    output="$outdir/maze-eval-${stage}-${arm}-s${seed}.json"
    log=$(mktemp "${TMPDIR:-/tmp}/bhl-maze-eval-XXXXXX.log")
    "$PY" "$REPO/scripts/bench/maze_recovery_probe.py" --headless --enable_cameras \
        --task "$task" --num-envs 32 --steps 600 --seed "$((100 + seed))" \
        --checkpoint "$checkpoint" --minimum-success 0.30 --output "$output" 2>&1 | tee "$log"
    grep -q MAZE_PROBE_PASS "$log"
    "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"] and d["gate_kind"] == "policy_evaluation" and d["first_episode_success_rate"] >= .3 and d["first_episodes_completed"] == d["num_envs"]' "$output"
    echo "MAZE_STAGE_PASS stage=$stage arm=$arm seed=$seed checkpoint=$checkpoint"
else
    echo "Unknown maze mode: $mode" >&2; exit 2
fi
