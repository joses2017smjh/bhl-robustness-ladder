#!/bin/bash
#SBATCH --job-name=ice-depth-freeze
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#
# 2026-09-26: WHY did the depth-camera biped beat the blind one on the placed-ice
# rung and its no-ice control? The ice terrain is 100% flat tiles and the Isaac
# depth camera ray-casts only /World/ground, so the 16x16 image can only encode
# camera height/pitch/roll. H1: the depth actor uses the image as a clean
# height/tilt sensor. Closed-loop MuJoCo test (scripts/bench/ice_depth_freeze.py),
# flat biped scene, one robot, EGL render on a GPU node:
#   A        live depth, rendered every step (MultiRunner.enable_depth_obs/depth_for)
#   B        depth frozen to the image rendered at t=0 (upright, nominal spawn height)
#   Bsettle  (secondary) live for the 1 s settle window, then frozen at that frame
#   C        (ORACLE) depth synthesised every step by ray-casting the z=0 plane
#            from the simulator's true camera pose = nominal + height/tilt change only
#   blind    matched blind checkpoint (same ring placed/control, same training seed)
# Checkpoints: model_5999.pt of the 4 ppo-ice-{placed,control}-depth-s{0,1} and the
# 4 matched blind runs, loaded directly (numpy MLP). Only placed-{blind,depth}-s0
# have exported/policy.onnx; for those the loader must match the ONNX within 1e-3
# (login node: 7.6e-6 depth, 1.1e-5 blind) or that checkpoint aborts.
# Protocol (run_eval / harness.run_episode): the 6 harness commands (incl. vx 0.3),
# seeds 0-9, 12 s, pushes off, 1 s settle, fall = tilt > 0.78 rad or sink > 0.25 m,
# lin-vel error in the yaw frame, displacement |xy_end - xy_start|. 60 episodes per
# arm, paired: every arm of a (command, seed) starts from the identical state.
#
# PREDECLARED (decided 2026-09-26, before any arm-A number):
#   gate   A fall rate > 0.50 -> INCONCLUSIVE, B not interpreted ("sim2sim gap" if
#          the matched blind fall rate <= 0.20, else "harness").
#   (i)    "uses depth as a live sensor"  iff fall(B)-fall(A) >= 0.30 or
#          disp(B) <= 0.50*disp(A)
#   (ii)   "image content not needed"     iff |fall(B)-fall(A)| <= 0.10 and
#          |disp(B)-disp(A)| <= 0.20*disp(A)
#   (iii)  otherwise inconclusive.
#   C secondary: C within 0.10 fall and 20% disp of A -> "height/tilt image
#          reproduces live depth"; else the MuJoCo render carries more.
#   Bsettle secondary: rules (i)-(iii) with Bsettle in place of B; never
#          replaces the primary B outcome.
#   pooled: episodes of gate-passing checkpoints only; none -> INCONCLUSIVE.
#   render check (reported): per-cell |rendered - analytic| <= 0.02 at t=0.
# Disclosure: login-node runs (no GL) of B/Bsettle/C (analytic nominal) and the
# blind reference were observed before this submission (placed-depth-s0: B fall
# 0.85 disp 0.29 m, Bsettle 0.70/0.62, C 0.00/2.29, blind 0.00/2.38). No rule or
# threshold above was changed after seeing them; arm A has never been run.
# Caveat: the t=0 camera is at 0.300 m, the robot settles at ~0.25 m, so B's
# frozen image is biased ~5 cm -- hence the secondary Bsettle arm.
# Cannot discriminate: the critic also saw depth in training (env.yaml critic
# group has the depth term); this actor-only test does not separate that.
# Labels: LEARNED actor MLP; SCRIPTED commands + frozen/analytic images; ORACLE
# arm C (true camera pose). MuJoCo floor has uniform friction: no ice here.
# Verdict lines are computed from verdict.json, never from exit codes.
set -uo pipefail
export MUJOCO_GL=egl
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
L=$U/logs/rsl_rl/biped
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
OUT=$REPO/results/repo-gpu-20260926/ice-depth-freeze
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "ICE-DEPTH-FREEZE RESULT: NOT RUN (refusing to overwrite published $OUT)"; exit 1
fi
mkdir -p "$OUT/parts"
CACHE=${TMPDIR:-/tmp}/bhl-ice-freeze-$SLURM_JOB_ID; mkdir -p "$CACHE"
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | $(date) ===" | tee "$OUT/provenance.txt"
md5sum scripts/bench/ice_depth_freeze.py src/bhl_robust/eval/multi_robot.py \
       src/bhl_robust/eval/depth_controller.py src/bhl_robust/eval/harness.py \
       results/deploy_ice_blind.yaml | tee -a "$OUT/provenance.txt"
"$PY" -c "import mujoco; r = mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'), 64, 64); r.enable_depth_rendering(); r.render(); r.close(); print('EGL depth render preflight OK', mujoco.__version__)" 2>&1 | tee -a "$OUT/provenance.txt"
# Unit tests (no-GL ones; logged, not a gate -- the verdict comes from JSON).
"$PY" -m pytest -q -p no:cacheprovider tests/test_ice_depth_freeze.py 2>&1 | tail -3 | tee "$OUT/pytest.txt"

# One background process per checkpoint (8 total, 8 CPUs, one GPU shared for EGL).
pids=()
for ring in placed control; do
  for s in 0 1; do
    for kind in depth blind; do
      run=$(ls -d "$L"/*_ppo-ice-${ring}-${kind}-s${s} 2>/dev/null | tail -1 || true)
      if [ -z "$run" ] || [ ! -f "$run/model_5999.pt" ]; then
          echo "MISSING ppo-ice-${ring}-${kind}-s${s} (or model_5999.pt) -- part skipped"; continue
      fi
      name=$(basename "$run")
      if [ "$kind" = depth ]; then arms="A B Bsettle C"; else arms="blind"; fi
      echo "launch $name arms=[$arms]"
      "$PY" scripts/bench/ice_depth_freeze.py run --run-dir "$run" --arms $arms \
          --nominal-source rendered --seeds 10 --episode-s 12 --commands all \
          --cache-dir "$CACHE/$name" --out "$OUT/parts/$name.json" \
          > "$OUT/parts/$name.log" 2>&1 &
      pids+=($!)
    done
  done
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "=== parts done: $(date) ==="
for f in "$OUT"/parts/*.log; do echo "--- $(basename "$f")"; grep -E "^\[parity\]|^\[nominal\]|^== |^ERROR|Traceback|Error" "$f" | head -12; done

"$PY" scripts/bench/ice_depth_freeze.py summarise --parts "$OUT/parts/*.json" --out "$OUT/verdict.json" \
    2>&1 | tee "$OUT/verdict.txt"

# Final line from the JSON only.
"$PY" - "$OUT/verdict.json" <<'EOF' 2>&1 | tee -a "$OUT/verdict.txt"
import json, sys
try:
    v = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"ICE-DEPTH-FREEZE RESULT: NO VERDICT (verdict.json unreadable: {e})"); sys.exit(0)
rows = v["per_checkpoint"]
if not rows:
    print("ICE-DEPTH-FREEZE RESULT: NO VERDICT (no depth checkpoint parts)"); sys.exit(0)
per = ", ".join(f"{r['run'].split('ppo-ice-')[-1]}={r['outcome']}" for r in rows)
rc = [r.get("render_check") or {} for r in rows]
t0 = [c.get("t0_max_abs") for c in rc if c.get("t0_max_abs") is not None]
p = v["pooled"]
print(f"ICE-DEPTH-FREEZE RESULT: pooled {p['outcome']} ({p['reason']}; "
      f"{p['n_checkpoints']}/{len(rows)} checkpoints pass the gate) | per-checkpoint: {per} | "
      f"render-vs-analytic t0 max {max(t0) if t0 else 'n/a'} (tol 0.02) | "
      f"C_vs_A pooled: {p.get('C_vs_A', 'n/a')} | Bsettle pooled: {p.get('Bsettle_outcome', 'n/a')}")
EOF
rm -rf "$CACHE"
ls -la "$OUT" "$OUT/parts"
echo "=== done: $(date) ==="
