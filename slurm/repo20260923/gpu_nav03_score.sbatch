#!/bin/bash
#SBATCH --job-name=nav03-score
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# NAV-03 provenance (docs/REPO_TASKS.md NAV-03, isaac/mazenav_seed0): the four
# superseded B5 maze-NAVIGATION seed-0 policies were only ever scored from
# their training event files (Episode_Termination/button_reached = 0.0, last-50
# and max, in every run). This job rollout-scores them.
#
#   sbatch slurm/repo20260923/gpu_nav03_score.sbatch      (no arguments)
#
# Policies (LEARNED PPO MLP actors, 6000 iterations, seed 0, pinned by path):
#   Blind   2026-09-17_18-12-07_mazenav-blind-s0/model_5999.pt   on Velocity-BHL-Maze-Blind-v0
#   Lidar   2026-09-17_18-14-59_mazenav-lidar-s0/model_5999.pt   on Velocity-BHL-Maze-Lidar-v0
#   Stereo  2026-09-17_23-27-31_mazenav-stereo-s0/model_5999.pt  on Velocity-BHL-Maze-Stereo-v0
#   Both    2026-09-17_23-55-36_mazenav-both-s0/model_5999.pt    on Velocity-BHL-Maze-Both-v0
# Those are the legacy ids the runs trained on (slurm/93_maze_ppo.sbatch TASK=
# Velocity-BHL-Maze-<Arm>-v0, RUN_NAME mazenav-<arm>-s0; slurm/inner/mazenav_video.sh),
# still registered unchanged in src/bhl_robust/tasks/__init__.py. The route
# command is the task's ORACLE waypoint command (maze_mdp.MazeWaypointCommand);
# the policy is the learned walking layer.
#
# Probe (scripts/bench/maze_recovery_probe.py), one Kit boot per policy:
#   --num-envs 32 --steps 600 --seed 100, NO --keep-corruption (observation
#   noise OFF), NO --settings (the single 'as_configured' row). On these legacy
#   tasks the probe has no _maze_recovery_state and falls back to the
#   termination manager's 'button_reached' and 'dead_end' terms (both in the
#   legacy MDP). An episode is at most 20 s / (0.005 s x decimation 8) = 500
#   control steps (run params/env.yaml), so 600 steps finish every first episode.
#
# Score: button_reached on the first episode, out of 32
#   (round(first_episode_success_rate * 32)).
#
# PREDECLARED RULE (2026-09-26, before any rollout):
#   CONSISTENT  button_reached = 0/32 on all four (what the event files show)
#   REOPEN      any of the four > 0/32 -> NAV-03 is reopened
#   INCOMPLETE  a JSON missing/unreadable/errored, or an identity check fails
#               (task id, checkpoint path, num_envs 32, observation_corruption false)
# Info columns (not gated): first episodes finished, dead ends, timeouts, and
# button successes over ALL episodes in the 600 steps.
# Final line 'NAV03 RESULT: CONSISTENT|REOPEN|INCOMPLETE', computed from the
# JSONs, never from exit codes. Exit 0 for CONSISTENT and REOPEN, 1 otherwise.
#
# Provenance caveat, written to provenance.txt beside the JSONs: the runs were
# trained at commit 1f1740f plus an uncommitted diff (each run's git/ dir).
# Rebuilt from that diff, maze_env_cfg.py differs from today's only by the
# terrain visual_material (RTX colour; ray sensors cast distance, not albedo).
# depth_env_cfg.py (depth_obs, the stereo term) and terrain_env_cfg.py
# (BipedBumpyEnvCfg) at training equal today's WORKING TREE (the author's
# uncommitted install_ice edit was already present) and differ from HEAD only in
# the ice classes. maze_mdp.py, maze_layout.py and terrains/maze.py were
# UNTRACKED at training time and cannot be diffed; this job replays today's
# versions (maze_mdp's later SF-02 hooks are no-ops unless the probe sets them).
#
# Output: results/repo-gpu-20260926/nav03-score/<Arm>-s0.{json,log},
# provenance.txt, verdict.json. If that directory already holds a probe JSON
# it is left untouched and this job writes to nav03-score-rerun-<jobid>/.
set -uo pipefail
[ -n "${SLURM_JOB_ID:-}" ] || { echo "NAV03 RESULT: INCOMPLETE (not inside a Slurm allocation; submit with sbatch)"; exit 2; }

export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
unset BHL_POLICY BHL_RND BHL_SYMMETRY   # MLP actors: never forward the train.py overlays
source "$REPO/slurm/inner/v60_boot_gate.sh"
BOOT_LOCK="${V60_BOOT_LOCK:-/nfs/hpc/share/$USER/Humanoid_Lite/.v60-boot.lock}"
PYH=$WORKSPACE/venv/bin/python; [ -x "$PYH" ] || PYH=$(command -v python3)
LOG_ROOT=$UPSTREAM/logs/rsl_rl/biped
declare -A RUNS=(
    [Blind]=2026-09-17_18-12-07_mazenav-blind-s0
    [Lidar]=2026-09-17_18-14-59_mazenav-lidar-s0
    [Stereo]=2026-09-17_23-27-31_mazenav-stereo-s0
    [Both]=2026-09-17_23-55-36_mazenav-both-s0
)
ARMS=(Blind Lidar Stereo Both)
WALL_S=${WALL_S:-14400}   # must match #SBATCH --time
PROBE="$REPO/scripts/bench/maze_recovery_probe.py"
OUT=$REPO/results/repo-gpu-20260926/nav03-score
if ls "$OUT"/*-s0.json >/dev/null 2>&1; then
    OUT="$OUT-rerun-$SLURM_JOB_ID"; echo "=== earlier results present; writing to $OUT ==="
fi
mkdir -p "$OUT"

echo "=== node: $(hostname) | NAV-03 rollout score | stack $BHL_STACK | job $SLURM_JOB_ID | $(date) ==="
cd "$REPO" && echo "repo HEAD $(git rev-parse --short HEAD 2>/dev/null) branch $(git branch --show-current 2>/dev/null)"

# Preflight (no Kit): checkpoints exist, the legacy ids are still registered.
for a in "${ARMS[@]}"; do
    [ -f "$LOG_ROOT/${RUNS[$a]}/model_5999.pt" ] || { echo "NAV03 RESULT: INCOMPLETE (checkpoint missing: ${RUNS[$a]}/model_5999.pt)"; exit 1; }
    grep -q "\"Velocity-BHL-Maze-$a-v0\"" "$REPO/src/bhl_robust/tasks/__init__.py" \
        || { echo "NAV03 RESULT: INCOMPLETE (Velocity-BHL-Maze-$a-v0 not registered in tasks/__init__.py)"; exit 1; }
done
{
    echo "NAV-03 rollout score, job $SLURM_JOB_ID, $(date -Is)"
    echo "repo HEAD $(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
    echo "working-tree changes in files these tasks import (git status --short):"
    git -C "$REPO" status --short -- src/bhl_robust/tasks/maze_env_cfg.py src/bhl_robust/tasks/maze_mdp.py \
        src/bhl_robust/terrains src/bhl_robust/sensors_rig.py src/bhl_robust/tasks/terrain_env_cfg.py \
        src/bhl_robust/tasks/depth_env_cfg.py src/bhl_robust/tasks/__init__.py scripts/bench/maze_recovery_probe.py 2>&1 | sed 's/^/  /'
    for a in "${ARMS[@]}"; do
        d=$LOG_ROOT/${RUNS[$a]}
        echo "$a: $d/model_5999.pt  trained at commit $(sed -n 2p "$d/git/bhl-robustness-ladder.diff" 2>/dev/null) + uncommitted diff in $d/git/"
    done
    echo "maze_env_cfg.py at training (1f1740f + run diff) vs HEAD: differs only by the terrain visual_material block"
    echo "  (checked 2026-09-26 on the login node); depth_env_cfg.py and terrain_env_cfg.py at training equal the"
    echo "  2026-09-26 working tree and differ from HEAD only in the ice classes; maze_mdp.py, maze_layout.py,"
    echo "  terrains/maze.py were untracked at training time and cannot be diffed -- this job replays today's versions."
} > "$OUT/provenance.txt"
cat "$OUT/provenance.txt"

# One probe per bhl_exec (one Kit boot each), behind v60_boot_gate; inputs via forwarded env vars.
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS PROBE_TASK PROBE_CKPT PROBE_JSON PROBE_LOG PROBE_TMO"
INNER_PROBE='set -uo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
echo "=== probe $PROBE_TASK <- $PROBE_CKPT | noise off | timeout ${PROBE_TMO}s | $(date) ==="
timeout -k 60 "$PROBE_TMO" "$PY" "$REPO/scripts/bench/maze_recovery_probe.py" --headless --enable_cameras \
    --task "$PROBE_TASK" --num-envs 32 --steps 600 --seed 100 --checkpoint "$PROBE_CKPT" \
    --output "$PROBE_JSON" > "$PROBE_LOG" 2>&1
rc=$?
grep -E "^\{\"name\"|SETTING-ERROR|ROLLOUT-ERROR|PROBE-ERROR|^(ValueError|RuntimeError|TypeError|KeyError|AttributeError)" "$PROBE_LOG" | cut -c1-240 || true
echo "=== probe exit $rc (informational) | json $([ -s "$PROBE_JSON" ] && echo written || echo MISSING) ==="'

boot() { v60_boot_gate; GATE_PID=${!:-}; }
release() { kill "${GATE_PID:-}" 2>/dev/null || true; _v60_release "$BOOT_LOCK"; }
for a in "${ARMS[@]}"; do
    left=$(( WALL_S - SECONDS - 300 ))
    if [ "$left" -lt 420 ]; then echo "=== $a: only ${left}s of wall left; skipped ==="; continue; fi
    boot
    left=$(( WALL_S - SECONDS - 300 )); [ "$left" -gt 2400 ] && left=2400; [ "$left" -lt 60 ] && left=60
    rm -f "$OUT/$a-s0.json"   # the verdict may only come from this job's rollout
    PROBE_TASK="Velocity-BHL-Maze-$a-v0" PROBE_CKPT="$LOG_ROOT/${RUNS[$a]}/model_5999.pt" \
        PROBE_JSON="$OUT/$a-s0.json" PROBE_LOG="$OUT/$a-s0.log" PROBE_TMO=$left bhl_exec -c "$INNER_PROBE"
    echo "=== $a done | t=${SECONDS}s ==="; release
done

# Table and verdict from the JSONs, never from exit codes.
"$PYH" - "$OUT" "$LOG_ROOT" "${RUNS[Blind]}" "${RUNS[Lidar]}" "${RUNS[Stereo]}" "${RUNS[Both]}" <<'PY'
import json, os, sys
out, log_root = sys.argv[1], sys.argv[2]
arms = dict(zip(("Blind", "Lidar", "Stereo", "Both"), sys.argv[3:7]))
problems, rows = [], {}
for arm, run in arms.items():
    task, ckpt = f"Velocity-BHL-Maze-{arm}-v0", os.path.join(log_root, run, "model_5999.pt")
    path = os.path.join(out, f"{arm}-s0.json")
    if not os.path.exists(path):
        problems.append(f"{arm}: missing {arm}-s0.json"); rows[arm] = None; continue
    try:
        d = json.load(open(path))
    except Exception as exc:
        problems.append(f"{arm}: unreadable: {exc}"); rows[arm] = None; continue
    if d.get("num_envs") != 32:
        problems.append(f"{arm}: num_envs {d.get('num_envs')} != 32")
    if d.get("task") != task:
        problems.append(f"{arm}: task {d.get('task')!r} != {task}")
    if d.get("checkpoint") != ckpt:
        problems.append(f"{arm}: checkpoint {d.get('checkpoint')!r} != {ckpt}")
    if d.get("observation_corruption") is not False:
        problems.append(f"{arm}: observation_corruption is not False (noise must be off)")
    s = (d.get("settings") or [{}])[0]
    if "error" in s or "first_episode_success_rate" not in s:
        problems.append(f"{arm}: rollout recorded an error"); rows[arm] = None; continue
    rows[arm] = {"button_first": round(float(s["first_episode_success_rate"]) * 32),
                 "first_done": s.get("first_episodes_completed"), "episodes": s.get("episodes_completed"),
                 "button_all": s.get("successes"), "dead_ends": s.get("dead_ends"), "timeouts": s.get("timeouts"),
                 "min_goal_distance_m": s.get("min_goal_distance_m"), "checkpoint": ckpt, "task": task}
print("\nNAV-03: first-episode button_reached, 32 envs, seed 100, 600 steps, observation noise OFF;")
print("walking policy LEARNED (PPO MLP, 6000 it), route command from the task's ORACLE waypoint command")
print(f"{'arm':8}{'button 1st':>12}{'1st done':>10}{'episodes':>10}{'button all':>12}{'dead ends':>11}{'timeouts':>10}")
for arm, r in rows.items():
    if r is None:
        print(f"{arm:8}{'--':>12}  (no result)"); continue
    print(f"{arm:8}{str(r['button_first']) + '/32':>12}{str(r['first_done']):>10}{str(r['episodes']):>10}"
          f"{str(r['button_all']):>12}{str(r['dead_ends']):>11}{str(r['timeouts']):>10}")
for p in problems:
    print("PROBLEM:", p)
verdict = "INCOMPLETE"
if not problems and all(r is not None for r in rows.values()):
    verdict = "CONSISTENT" if all(r["button_first"] == 0 for r in rows.values()) else "REOPEN"
    print("check: button_reached == 0/32 on all four first episodes:", "ok" if verdict == "CONSISTENT" else "FAILED")
json.dump({"verdict": verdict, "rows": rows, "problems": problems,
           "rule": "CONSISTENT iff first-episode button_reached == 0/32 on all four; any > 0 reopens NAV-03 (predeclared 2026-09-26)",
           "labels": {"policy": "learned PPO MLP walking policy (legacy mazenav seed 0)",
                      "navigation": "oracle waypoint command of the legacy maze task", "noise": "observation corruption off"}},
          open(os.path.join(out, "verdict.json"), "w"), indent=2)
print(f"\nNAV03 RESULT: {verdict}")
sys.exit(0 if verdict != "INCOMPLETE" else 1)
PY
rc=$?
echo "=== done: t=${SECONDS}s | $(date) ==="
exit $rc
