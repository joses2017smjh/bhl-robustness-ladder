#!/bin/bash
#SBATCH --job-name=maze-sf01
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Repo-wide campaign (2026-09-23): maze recovery Full stage, re-evaluated with
# policy observation corruption ON.
#
# The published Full result (379/384 first-episode successes,
# docs/WEEKEND_RESULTS_2026-09-20.md) was scored by maze_recovery_probe.py with
# cfg.observations.policy.enable_corruption forced False, although the policy
# group trained with it True (params/env.yaml of every wknd-full run). This job
# re-runs exactly that evaluation -- same 12 checkpoints (read from the
# published results/weekend-20260919/maze-eval-Full-<arm>-s<seed>.json
# "checkpoint" field, all model_5997.pt), --num-envs 32 --steps 600,
# --seed 100+training seed, --headless --enable_cameras -- with only
# --keep-corruption added. One JSON per arm x seed under
#   results/repo-gpu-20260923/maze-sf01/<arm>-s<seed>.json
#
# Stack: v60 (Isaac Sim 6.0 / Lab 3.0b2). The weekend Full array 21359510 ran
# weekend_gpu.sbatch v60 (its log prints stack=v60) and the run dirs' logs sit
# under Kit/IsaacLab/3.0.
#
# Time: 12 sequential Kit boots + 32-env/600-step rollouts (~1-2 min each on the
# weekend nodes). Each boot goes through v60_boot_gate; this job releases its own
# lock as soon as its rollout exits (its Kit is gone, so no startup to protect)
# instead of idling out the 150 s window 12 times. A ~48-min budget check (plus the inner 540 s per-rollout timeout) stops
# launching new checkpoints so the summary / PASS-FAIL line always prints.
#
# PASS means all 12 noise-on evaluations ran to a complete, correctly
# configured JSON -- NOT that the policies stay above any success level. The
# pooled noise-on success vs the published noise-off 379/384 is the result.
set -uo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MAZE_SF01_OUT=${MAZE_SF01_OUT:-$REPO/results/repo-gpu-20260923/maze-sf01}
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS MAZE_SF01_OUT"
PUB_DIR=$REPO/results/weekend-20260919
BUDGET_S=${BUDGET_S:-9600}
mkdir -p "$MAZE_SF01_OUT"
source "$REPO/slurm/inner/v60_boot_gate.sh"
BOOT_LOCK="${V60_BOOT_LOCK:-/nfs/hpc/share/$USER/Humanoid_Lite/.v60-boot.lock}"
PYH=$WORKSPACE/venv/bin/python; [ -x "$PYH" ] || PYH=$(command -v python3)

ARMS=(Blind Lidar Stereo Both)
SEEDS=(0)

echo "=== node: $(hostname) | maze noise-on re-eval | stack $BHL_STACK | job ${SLURM_JOB_ID:-?} | $(date) ==="
cd "$REPO" && echo "repo HEAD $(git rev-parse --short HEAD 2>/dev/null) branch $(git branch --show-current 2>/dev/null)"
grep -q -- "--keep-corruption" "$REPO/scripts/bench/maze_recovery_probe.py" \
    || { echo "MAZE-SF01 RESULT: FAIL (probe lacks --keep-corruption)"; exit 1; }

# Resolve every checkpoint from the published JSONs before the first Kit boot.
declare -A CKPTS
for a in "${ARMS[@]}"; do for s in "${SEEDS[@]}"; do
    pub="$PUB_DIR/maze-eval-Full-$a-s$s.json"
    ck=$("$PYH" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$pub" 2>/dev/null) || ck=""
    lower=${a,,}
    case "$ck" in
        */logs/rsl_rl/biped/2026-09-19_*_wknd-full-$lower-s$s/model_*.pt) ;;
        *) echo "MAZE-SF01 RESULT: FAIL (bad/unreadable published checkpoint for $a s$s: '$ck')"; exit 1;;
    esac
    [ -f "$ck" ] || { echo "MAZE-SF01 RESULT: FAIL (checkpoint missing: $ck)"; exit 1; }
    CKPTS[$a-$s]=$ck
    echo "ckpt $a s$s -> $ck"
    # Remove stale outputs so PASS can only come from this job's JSONs.
    rm -f "$MAZE_SF01_OUT/$a-s$s.json" "$MAZE_SF01_OUT/$a-s$s.log"
done; done

for s in "${SEEDS[@]}"; do for a in "${ARMS[@]}"; do
    if [ "$SECONDS" -gt "$BUDGET_S" ]; then
        echo "=== budget ${BUDGET_S}s exceeded at ${SECONDS}s: skipping $a s$s ==="
        continue
    fi
    v60_boot_gate
    gate_pid=${!:-}   # the gate's release subshell (unset if the gate gave up)
    echo "=== $a s$s | t=${SECONDS}s | $(date) ==="
    VARIANT="$a" SEED="$s" CKPT="${CKPTS[$a-$s]}" bhl_exec "$REPO/slurm/repo20260923/inner_maze_sf01.sh"
    echo "=== $a s$s exit $? | t=${SECONDS}s ==="
    # This job's Kit has exited; drop our boot lock now rather than after 150 s.
    kill "$gate_pid" 2>/dev/null || true
    _v60_release "$BOOT_LOCK"
done; done

# PASS/FAIL and the table from the JSONs, not from exit codes.
"$PYH" - "$MAZE_SF01_OUT" "$PUB_DIR" <<'PY'
import json, os, sys
out, pub = sys.argv[1], sys.argv[2]
arms, seeds = ("Blind", "Lidar", "Stereo", "Both"), (0,)
ok, problems = True, []
pub_total = on_total = on_denom = 0
print(f"{'arm':7} {'seed':>4} {'noise-off (pub)':>16} {'noise-on':>9} {'first done':>10} "
      f"{'all-ep succ':>12} {'dead_end':>8} {'timeout':>7}  checkpoint")
per_arm = {a: [0, 0, 0] for a in arms}  # pub, on, denom
for a in arms:
    for s in seeds:
        pj = json.load(open(os.path.join(pub, f"maze-eval-Full-{a}-s{s}.json")))
        p_succ = round(pj["first_episode_success_rate"] * pj["num_envs"])
        pub_total += p_succ
        per_arm[a][0] += p_succ
        path = os.path.join(out, f"{a}-s{s}.json")
        if not os.path.exists(path):
            ok = False; problems.append(f"MISSING {a}-s{s}.json")
            print(f"{a:7} {s:>4} {p_succ:>13}/32 {'--':>9}"); continue
        try:
            d = json.load(open(path))
        except Exception as exc:
            ok = False; problems.append(f"UNREADABLE {a}-s{s}.json: {exc}"); continue
        checks = {
            "observation_corruption is True": d.get("observation_corruption") is True and isinstance(d.get("settings"), list) and len(d["settings"]) >= 14,
            "gate_kind == policy_evaluation": d.get("gate_kind") == "policy_evaluation",
            "task": d.get("task") == f"Velocity-BHL-MazeRecovery-Full-{a}-v0",
            "num_envs == 32": d.get("num_envs") == 32,
            "steps == 600": d.get("steps") == 600,
            f"seed == {100 + s}": d.get("seed") == 100 + s,
            "checkpoint == published": d.get("checkpoint") == pj["checkpoint"],
            "first_episodes_completed == 32": d.get("first_episodes_completed") == 32,
        }
        bad = [k for k, v in checks.items() if not v]
        if bad:
            ok = False; problems.append(f"{a}-s{s}: failed {bad}")
        n = d.get("num_envs") or 32
        on = round(float(d.get("first_episode_success_rate", 0.0)) * n)
        on_total += on; on_denom += n
        per_arm[a][1] += on; per_arm[a][2] += n
        print(f"{a:7} {s:>4} {p_succ:>13}/32 {on:>6}/{n:<2} {str(d.get('first_episodes_completed')):>7}/{n:<2} "
              f"{str(d.get('successful_episodes')) + '/' + str(d.get('completed_episodes')):>12} "
              f"{str(d.get('dead_end_episodes')):>8} {str(d.get('timeout_episodes')):>7}  "
              f"{os.path.basename(os.path.dirname(d.get('checkpoint', '')))}/{os.path.basename(d.get('checkpoint', ''))}")
print()
for a in arms:
    p, o, n = per_arm[a]
    print(f"{a:7} noise-off {p}/96   noise-on {o}/{n}")
if pub_total != 379:
    ok = False; problems.append(f"published pool recomputed as {pub_total}/384, expected 379")
print(f"\nPOOLED first-episode success: noise-OFF (published) {pub_total}/384 = {pub_total/384:.3%} | "
      f"noise-ON {on_total}/{on_denom} = {(on_total/on_denom if on_denom else float('nan')):.3%} | "
      f"delta {on_total - pub_total:+d} episodes")
for p in problems:
    print("PROBLEM:", p)
print("MAZE-SF01 RESULT: " + ("PASS (12/12 noise-on evaluations complete and correctly configured)"
                               if ok and on_denom == 384 else "FAIL (see PROBLEM lines)"))
sys.exit(0 if ok and on_denom == 384 else 1)
PY
rc=$?
echo "=== done: t=${SECONDS}s | $(date) ==="
exit $rc
