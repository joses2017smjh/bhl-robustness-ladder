#!/bin/bash
# Kinematic C0 / C1 / C5. No Isaac, no GPU.
set -euo pipefail
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
# Login-node python3.14 has no numpy; the v51/v60 venvs and /usr/bin/python3 do.
if [ -z "${PY:-}" ]; then
    for cand in /usr/bin/python3 /nfs/hpc/share/$USER/Humanoid_Lite/venv/bin/python python3; do
        if [ -x "$cand" ] && "$cand" -c "import numpy" 2>/dev/null; then
            PY=$cand
            break
        fi
    done
fi
: "${PY:?no python with numpy}"
mkdir -p "$REPO/results/cloth"
"$PY" "$REPO/tests/test_cloth_sort.py" -v
"$PY" "$REPO/scripts/cloth/make_meshes.py"
"$PY" "$REPO/scripts/cloth/eval_scripted.py" --rung C0 --episodes 64 \
    --out "$REPO/results/cloth/c0_scripted.json"
"$PY" "$REPO/scripts/cloth/train_rigid.py" --episodes 200 --eval-episodes 64 \
    --out "$REPO/results/cloth/c1_learned.json"
"$PY" "$REPO/scripts/cloth/eval_five.py" --episodes 32 \
    --out "$REPO/results/cloth/c5_scripted.json"
"$PY" "$REPO/scripts/cloth/eval_transfer.py" \
    --rigid "$REPO/results/cloth/c0_scripted.json" \
    --out "$REPO/results/cloth/c3_transfer.json"
"$PY" "$REPO/scripts/cloth/adapt.py" --collect 64 --eval-episodes 64 \
    --out "$REPO/results/cloth/c4_bc.json"
"$PY" "$REPO/scripts/bench/cloth_sort_bench.py"
"$PY" "$REPO/scripts/cloth/estimate_cost.py" --physics deformable --num-envs 2048 || true
echo "=== cloth kinematic ladder done ==="
