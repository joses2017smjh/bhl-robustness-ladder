#!/bin/bash
# G-B4t: two real iterations of every Tier 1 MARL row, at the full 4,096 envs,
# one Isaac process at a time; then the verdict from what each run printed and
# wrote (scripts/bench/marl_terrain_check.py). A run that hangs in teardown is
# cut off by `timeout` -- its logs and event file are already written.
set -uo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
GATE_DIR="$REPO/results/marl_terrain_gate"
rm -rf "$GATE_DIR"; mkdir -p "$GATE_DIR"
# Ice is out until B3's patches are where the robots are (ice_placement_probe:
# median 72 m away, 4.4% reachable in an episode). Its grid rows stay unqueued.
TERRAINS=(stairs slippery rough)
declare -A TASK=([rough]=Velocity-BHL-Biped-Depth-v0 [slippery]=Velocity-BHL-Biped-Slippery-Depth-v0
                 [stairs]=Velocity-BHL-Biped-Stairs-Depth-v0 [ice]=Velocity-BHL-Biped-Ice-Depth-v0)
for T in "${TERRAINS[@]}"; do
  for ROW in mappo ippo limb1; do
    case $ROW in mappo) A=mappo P=legs2 ;; ippo) A=ippo P=legs2 ;; limb1) A=ippo P=limb1 ;; esac
    echo "=== $ROW on $T (${TASK[$T]}) $(date +%T) ==="
    timeout 900 "$PY" scripts/train_marl.py --task "${TASK[$T]}" --num_envs "${NUM_ENVS:-4096}" \
        --seed 0 --max_iterations 2 --write-interval 12 --run_name "gate-$ROW-$T" \
        --partition "$P" --algo "$A" --headless > "$GATE_DIR/$ROW-$T.log" 2>&1
    echo "  exit $? | $(grep -a '\[marl-gate\]' "$GATE_DIR/$ROW-$T.log" | cut -c1-160 | tr '\n' ' ')"
  done
done
"$PY" scripts/bench/marl_terrain_check.py "$GATE_DIR" "$REPO" $(( ${#TERRAINS[@]} * 3 )) | tee "$REPO/results/marl_terrain_gate.txt"
grep -q "^G-B4t PASS" "$REPO/results/marl_terrain_gate.txt"
