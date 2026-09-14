#!/bin/bash
# G-T3: two real iterations of each Tier 3 row at 4,096 envs, one Isaac process
# at a time, then scripts/bench/arms_tier3_check.py on what they printed.
set -uo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
GATE_DIR="$REPO/results/arms_tier3_gate"
rm -rf "$GATE_DIR"; mkdir -p "$GATE_DIR"
T=Velocity-BHL-Arms-Stairs-Depth-v0

echo "=== ppo on stairs $(date +%T) ==="
( export TASK=$T RUN_NAME=gate-ppoarms-stairs SEED=0 NUM_ENVS=${NUM_ENVS:-4096} MAX_ITER=2 OVERRIDE_FILE=""
  timeout 900 "$REPO/slurm/inner/train.sh" ) > "$GATE_DIR/ppo-stairs.log" 2>&1
echo "  exit $?"
for ROW in mappo limb1; do
  case $ROW in mappo) A=mappo P=limb4 ;; limb1) A=ippo P=limb1 ;; esac
  echo "=== $ROW on stairs $(date +%T) ==="
  timeout 900 "$PY" scripts/train_marl.py --task "$T" --num_envs "${NUM_ENVS:-4096}" --seed 0 \
      --max_iterations 2 --write-interval 12 --run_name "gate-${ROW}arms-stairs" \
      --partition "$P" --algo "$A" --headless > "$GATE_DIR/$ROW-stairs.log" 2>&1
  echo "  exit $? | $(grep -a '\[marl-gate\] task' "$GATE_DIR/$ROW-stairs.log" | cut -c1-160)"
done
"$PY" scripts/bench/arms_tier3_check.py "$GATE_DIR" "$REPO" | tee "$REPO/results/arms_tier3_gate.txt"
grep -q "^G-T3 PASS" "$REPO/results/arms_tier3_gate.txt"
