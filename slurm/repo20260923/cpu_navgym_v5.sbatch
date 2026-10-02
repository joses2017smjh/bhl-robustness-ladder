#!/bin/bash
#SBATCH --job-name=navgym-v5
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --array=8-10
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# 2026-10-02: NavGym v5 -- N2 of docs/SOLUTIONS_2026-10-01.md (fully learned navigation; the user chose it over the A*
# sub-goal hybrid N1, which is not built). CONFIRMATORY run, 3 NEW training seeds (8, 9, 10; 0-1 were v2, 2-4 v3/v2ctl,
# 5-7 v4).
#
# Why (verified findings, results/solutions-20260930/navgym/, verify-navgym/, sysid/): armV4-s6's physics misses are its
# own (it fails the same mazes in the gym); maze 50009's dead-end pocket traps it because the policy has no memory of where
# it has been, and the route's decisive turn lies 2.8 m away, past its 2.4 m crop edge; yaw is bang-bang (|wz| saturated on
# 95-99 % of steps; final policy std 9.4 and 13.7). The physics capture-pose fix (N0, 21501509) was NEGATIVE as a single fix
# but harmless.
#
# v5 = the v4 configuration exactly (MazeNavEnv dynamics and rewards of env version 2, `--ppo v2`, the frozen v4 stall
# price, 16 envs, 30 M steps, the v4 curriculum and periodic evaluation; see slurm/repo20260923/cpu_navgym_v4.sbatch and
# scripts/bench/navgym_train.py V4_RULE) PLUS these changes ONLY (FROZEN), as a new env version (3):
#   - Visitation channel: a world-fixed grid at the ego map's 0.2 m resolution holding a recency value r in [0, 1]: every
#     step, cells within the robot's footprint radius are set to 1, and all cells decay as r *= exp(-dt / 60 s). It is
#     cropped and rotated exactly like the existing egocentric map crop and added to it as an extra channel.
#   - Coarse wide map: the same ego map (all existing channels) plus the visitation channel, downsampled to 0.6 m cells
#     and cropped 24 x 24 (14.4 m) around the robot in the same egocentric frame, as a second map input.
#   - Yaw-change penalty: reward -= 0.005 * |a_yaw(t) - a_yaw(t-1)| (actions in [-1, 1]); chosen, not tuned.
#   - The deployment speed brake in the gym, as the physics stack applies it (team_sensors.brake_command:
#     v *= clip((clearance - 0.42) / 0.48, 0, 1) within +-25 deg of travel, refreshed at 10 Hz);
#     results/solutions-20260930/sysid/sysid_env.py has an implementation to port. (Ported: env.py brake_scale, the
#     range term only; see "brake" below.)
#   - Nothing else changes. Env versions 1 and 2 and the v3/v4 training configurations must reproduce exactly: the
#     existing tests in tests/test_navgym.py (test_v1_is_reproduced_exactly etc.) must still pass, and the v2 gold test
#     must pass. (They do: tests/test_navgym.py incl. the v2 + v4-stall-price gold, and tests/test_navgym_v5.py.)
# How it is built (bhl_robust/navgym/env.py, version 3 -- every choice below was made before any v5 run, none from results):
#   - visitation: VisitMap on EgoMap's own lattice (same origin, 0.2 m cells, shape); each step all cells decay FIRST, then
#     the cells whose centres lie within ROBOT_RADIUS (0.22 m) of the robot are set to 1, so the current footprint reads
#     exactly 1; the start pose is marked at reset. Obs key "map_visit" (4, 24, 24) = occupied, free, unknown, visitation
#     (it replaces v2's 3-channel "map"; same crop and rotation code path as EgoMap.crop; outside the grid = 0).
#   - coarse map: obs key "map_coarse" (4, 24, 24): the four fine channels block-averaged over 3 x 3 fine cells (0.6 m
#     cells anchored at the fine grid's origin; padding past the grid edge counts as unknown, unvisited), then sampled
#     egocentrically exactly as EgoMap.crop samples (forward = up); outside the grid = unknown (0, 0, 1, 0).
#   - yaw change: on every step of any outcome, on the clipped actions; a_yaw before the first step = 0 (the reset's
#     previous action, which the "goal" key also shows).
#   - brake: the 10 Hz packet is captured on team_sensors' schedule (steps 0, 3, 5, 8, 10, ...) at the step's start pose,
#     i.e. it is the last scan; clearance = min of its sector minima whose centres lie within +-25 deg of straight ahead
#     (the gait has no vy); the scale multiplies the forward command before the latency queue. As in sysid_env.py, the
#     brake's stereo-depth tightening and IMU limit are not modelled (the gym has no depth camera or IMU). The stall price
#     still reads the actor's own a[0].
#     Disclosed (review, 2026-10-02): the omitted depth term is NOT minor for a wall straight ahead (the sysid notes'
#     "likely minor" looked only at flat-floor false braking). The biped's stereo centre sits 0.12 m ahead of the lidar
#     and is pitched 20 deg, so its horizontal pixel row reads ~0.94 x (d - 0.12) for a wall at lidar distance d: in
#     physics the brake starts at d ~ 1.08 m and stops the robot at d ~ 0.57 m; in the gym at 0.90 m and 0.42 m.
#   - policy: a second map input needs a second CNN, so v5 uses NavExtractorV5 (scripts/bench/navgym_train.py): NavExtractor's
#     map CNN widened to 4 input channels, an identical CNN with its own weights for "map_coarse", the same lidar + near +
#     goal MLP, and the 256-d output layer over the three 128-d embeddings. NavExtractor (v1-v4) is unchanged.
#   - ONNX inputs: lidar, near, map_visit, map_coarse, goal. scripts/bench/maze_explore.py --policy builds them with the
#     gym's own build_obs, keyed by those names, with a VisitMap stepped once per control step at the pose the observation
#     is built at (oracle pose); v1, v2 and v4 actors keep their path byte-identical.
#   - reported, not gated: summary.json "yaw_stats_not_gated" (yaw flips per second, |wz| saturation share = share of steps
#     with |a_yaw| >= 0.99, brake share) for the final and best actors on both evaluation sets.
#
# PREDECLARED RULES (2026-10-02, FROZEN before any v5 training; no v5 episode had run on any set):
#   - Gym gate: V4_RULE's clauses unchanged (5x5 success >= 0.80, 6x6 success >= 0.70, 6x6 collision <= 0.15, n = 48
#     per size, the run reached its last step (num_timesteps >= 30 000 000), and it is the declared v5 configuration),
#     judged on each seed's FINAL actor (deterministic) on the NEVER-USED set maze seeds 61000-61047 (dynamics
#     reset(seed = 61000 + k)). v5 PASSES iff >= 2 of the 3 seeds (8, 9, 10) meet every clause; otherwise NEGATIVE.
#     Seeds 10000-10047 and the best checkpoint are reported, not gated.
#   - Physics transfer, only if the gym gate PASSES, only for gate-passing final actors: MuJoCo 3.3.5, the frozen learned
#     biped gait, LEARNED v5 (vx, wz) commands, ORACLE pose and goal, --policy-capture-pose; hard 6x6, 1 extra opening,
#     maze seeds 63000-63011, 180 s. PASS iff EACH gate-passing final actor reaches >= 10/12 goals with 0 falls; otherwise
#     NEGATIVE; INCOMPLETE if a summary is missing or not 12 episodes. A* on the same mazes reported, not gated.
#     (slurm/repo20260923/cpu_navgym_v5_transfer.sbatch)
#   - Reported, not gated: yaw flips per second and |wz| saturation share in the gym evaluation. Maze seeds 40000-40047
#     and 50000-50011 are scored seeds of earlier tests: never run them.
#   "The declared v5 configuration" = summary.json ppo "v2", env_version 3, no std ceiling, stall_price == the frozen v4
#   price, v5_config == navgym_train.V5_CONFIG. Code: navgym_train.py V5_RULE / V5_TEXT / judge_v5 / verdict_v5 (the gate is
#   read from summary.json, never from an exit code; a verdict JSON is never overwritten).
#   Labels: LEARNED actors (deterministic Gaussian mean); pose and goal coordinate are ORACLE inputs.
#   Never-used sets, checked 2026-10-02 by grep over SLURM_JOBS.md, docs, scripts, slurm, src, tests, results,
#   solutions-20260930/ and logs/: maze seeds 61000-61047 and 63000-63011 appear only as declarations
#   (docs/SOLUTIONS_2026-10-01.md N2 row, navgym_train.py rule text, the campaign specs) or inside unrelated numbers (a hex
#   constant, a crash-dump file name, MARL progress counters); no job, script or result has used them.
#
# Modes (NAVGYM_MODE):
#   train   (default) one array task = one seed, output results/navgym-v5-20261002/armV5-s<seed>/ (refuses if it exists)
#   verdict the final arm verdict after every task has ended (a missing summary is a miss):
#           sbatch --dependency=afterany:<array job> --array=0 --cpus-per-task=1 --mem=2G --time=00:10:00 \
#                  --export=ALL,NAVGYM_MODE=verdict cpu_navgym_v5.sbatch
#           Only this mode (--final) writes results/navgym-v5-20261002/verdict_v5.json, the file the transfer reads. A
#           status check (`navgym_train.py verdict-v5 <root>` without --final) only prints; navgym_train.py never writes a
#           non-final verdict to a file named verdict_v5.json.
#   smoke   (<= 1 h, not scored) ~50 000 v5 training steps (seed 99, throw-away evaluation sets 10000-10001 and
#           90000-90001) + ONNX export + one 30 s physics episode of the exported actor on training-range maze 9150
#           (maze_explore.py --policy --policy-capture-pose) + the verdict and transfer-gate code paths; exits non-zero
#           on any failure. Outputs under solutions-20260930/campaign-navgym-v5/navgym-v5-smoke/job-<id>/:
#           sbatch --job-name=navgym-v5-smoke --array=0 --time=01:00:00 --export=ALL,NAVGYM_MODE=smoke cpu_navgym_v5.sbatch
#
# Wall time: v4 took ~9.2 h for 30 M steps on 16 CPUs; v5's env (visitation + coarse map + brake) and its two-CNN extractor
# are heavier, so 36 h (share allows 14 days; a run cut short fails its reached-last-step clause).
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
ROOT=${NAVGYM_ROOT:-$REPO/results/navgym-v5-20261002}
MODE=${NAVGYM_MODE:-train}
if [ "$MODE" = verdict ]; then
  mkdir -p "$ROOT"
  echo "=== NavGym v5 final verdict | HEAD $(git rev-parse --short HEAD) | $(date) ==="
  "$PY" scripts/bench/navgym_train.py verdict-v5 "$ROOT" --seeds 8 9 10 --final --out "$ROOT/verdict_v5.json"
  exit 0
fi
if [ "$MODE" = smoke ]; then
  U=$REPO/external/Berkeley-Humanoid-Lite
  SROOT=${NAVGYM_SMOKE_ROOT:-/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/campaign-navgym-v5/navgym-v5-smoke}/job-${SLURM_JOB_ID:-local}
  case "$SROOT" in */*-smoke/job-*) ;; *) echo "SMOKE FAIL: outputs must live under a *-smoke directory, got $SROOT"; exit 1 ;; esac
  [ -e "$SROOT" ] && { echo "SMOKE FAIL: $SROOT exists (never overwrite)"; exit 1; }
  mkdir -p "$SROOT"
  SOUT=$SROOT/armV5-s99
  CACHE=${TMPDIR:-/tmp}/bhl-navgym-v5-smoke-${SLURM_JOB_ID:-local}; mkdir -p "$CACHE"
  echo "=== SMOKE node: $(hostname) | HEAD $(git rev-parse --short HEAD) | NavGym v5 smoke (not scored) | $SROOT | $(date) ==="
  fail() { echo "SMOKE FAIL: $*"; exit 1; }
  T0=$(date +%s)
  # 1. v5 training: the train mode's flags; smoke overrides: seed 99, steps, evaluation sizes and the throw-away fresh set
  set +e
  "$PY" scripts/bench/navgym_train.py --out "$SOUT" --seed 99 --total-steps 50000 --env-version 3 --ppo v2 --idle-cost 0.008 \
      --n-envs 16 --eval-every 25000 --n-eval 2 --n-final-eval 2 --checkpoint-every 25000 --final-eval-fresh-base 90000 \
      > "$SROOT/train.log" 2>&1
  RC=$?
  set -e
  grep -vE "Warning|warnings.warn" "$SROOT/train.log" | tail -n 5 | cut -c1-400 || true
  [ "$RC" -eq 0 ] || fail "trainer exit code $RC (see $SROOT/train.log)"
  echo "smoke training + final evaluations + export: $(( $(date +%s) - T0 )) s wall"
  # 2. the summary, config and ONNX are the declared v5 configuration; the gate's per-seed judge runs on them
  "$PY" - "$SOUT" <<'PY' || fail "summary / config / ONNX checks"
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts/bench")
import navgym_train as nt
from bhl_robust.navgym.env import OBS_KEYS_V3, MazeNavEnv, build_obs
out = Path(sys.argv[1])
s = json.loads((out / "summary.json").read_text())
c = json.loads((out / "config.json").read_text())
bad = []
def need(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        bad.append(what)
need(s.get("env_version") == 3 and s.get("ppo") == "v2" and s.get("std_ceiling") is None, "env_version 3, ppo v2, no std ceiling")
need(s.get("v5_config") == nt.V5_CONFIG and c.get("v5_config") == nt.V5_CONFIG, "summary and config v5_config == V5_CONFIG")
need(s.get("stall_price") == nt.V4_STALL_PRICE and c.get("idle_cost") == 0.008, "the frozen v4 stall price")
need(c.get("env_shaping_gamma") == 1.0 and c.get("n_envs") == 16 and c.get("curriculum") == [[list(z) for z in st] for st in nt.CURRICULUM],
     "v4 shaping gamma, 16 envs, v4 curriculum")
for tag, o in (("final", s.get("onnx") or {}), ("best", (s.get("best") or {}).get("onnx") or {})):
    need(o.get("inputs") == list(OBS_KEYS_V3) and float(o.get("max_abs_diff_vs_torch", 1.0)) < 1e-4,
         f"{tag} ONNX inputs {o.get('inputs')} diff {o.get('max_abs_diff_vs_torch')}")
ys = s.get("yaw_stats_not_gated") or {}
need({"note", "final_actor_heldout_10000", "final_actor_fresh", "best_actor_heldout_10000", "best_actor_fresh"} <= set(ys),
     "yaw_stats_not_gated has the four evaluation tags")
need(all(ys[t][sz]["episodes"] == 2 and 0.0 <= ys[t][sz]["wz_saturation_share"] <= 1.0 for t in ys if t != "note" for sz in ys[t]),
     "yaw statistics per size cover the 2 episodes")
need(bool(s.get("reached_last_step")) and int(s.get("num_timesteps", 0)) >= 50000, "reached its (smoke) last step")
need(str((s.get("fresh_set") or {}).get("maze_seeds", "")).startswith("90000 + k"), "throw-away fresh set 90000 + k (never 61000)")
j = nt.judge_v5(s)
failing = {k for k, v in j["checks"].items() if not v}
need(j["checks"]["config_v5"] is True and j["meets"] is False, "judge_v5: the declared v5 configuration, and (a smoke) not meeting")
need({"reached_last_step", "never_used_set_is_61000", "n48"} <= failing <= {"reached_last_step", "never_used_set_is_61000", "n48",
     "5x5_success", "6x6_success", "6x6_collision"}, f"judge_v5 fails exactly the smoke's clauses: {sorted(failing)}")
# the exported actor runs on the gym's own v3 observation, built by build_obs keyed by its ONNX input names
import onnxruntime as ort
sess = ort.InferenceSession(str(out / "actor.onnx"))
keys = tuple(i.name for i in sess.get_inputs())
env = MazeNavEnv(sizes=((6, 6),), version=3, gamma=1.0)
obs, _ = env.reset(seed=4)
for _ in range(30):
    feeds = {k: np.asarray(v, np.float32)[None] for k, v in obs.items()}
    a = sess.run(None, feeds)[0][0]
    obs, r, te, tr, info = env.step(np.clip(a, -1, 1))
    if te or tr:
        break
ref = build_obs(keys, env.ranges, env.emap, env.x, env.y, env.yaw, env.goal_xy, env.prev_action, visit=env.visit)
need(keys == OBS_KEYS_V3 and all(np.array_equal(ref[k], obs[k]) for k in keys) and np.isfinite(a).all(),
     "ONNX actor runs on build_obs(keys from its input names) == the env's observation")
print("SMOKE CHECKS (training):", "PASS" if not bad else f"FAIL {bad}")
sys.exit(1 if bad else 0)
PY
  # 3. the verdict code path: a status check (no --final) only prints, and a non-final verdict is never written to a file
  #    named verdict_v5.json; a final verdict is written once, then never overwritten
  "$PY" scripts/bench/navgym_train.py verdict-v5 "$SROOT" --seeds 99 > "$SROOT/verdict_status.log"
  grep -q "status only" "$SROOT/verdict_status.log" && [ ! -e "$SROOT/verdict_v5.json" ] || fail "a status check (no --final) wrote a file"
  set +e
  "$PY" scripts/bench/navgym_train.py verdict-v5 "$SROOT" --seeds 99 --out "$SROOT/verdict_v5.json" > "$SROOT/verdict_nonfinal.log" 2>&1
  RC=$?
  set -e
  [ "$RC" -ne 0 ] && [ ! -e "$SROOT/verdict_v5.json" ] || fail "a non-final verdict was written to verdict_v5.json (exit $RC)"
  "$PY" scripts/bench/navgym_train.py verdict-v5 "$SROOT" --seeds 99 --final --out "$SROOT/verdict_v5.json" | cut -c1-600
  "$PY" scripts/bench/navgym_train.py verdict-v5 "$SROOT" --seeds 99 --final --out "$SROOT/verdict_v5.json" > "$SROOT/verdict_rerun.log"
  grep -q "REFUSING to overwrite" "$SROOT/verdict_rerun.log" || fail "verdict-v5 did not refuse to overwrite"
  grep -q '"verdict": "NEGATIVE"' "$SROOT/verdict_v5.json" || fail "the smoke's final verdict is not NEGATIVE"
  # 4. one 30 s physics episode of the exported actor on a training-range maze (< 10000), capture-pose integration
  T1=$(date +%s)
  set +e
  "$PY" scripts/bench/maze_explore.py --upstream "$U" --seeds 1 --seed-start 9150 --n 6 --m 6 --extra-openings 1 --time-limit 30 \
      --no-overwrite --cache-dir "$CACHE/phys" --policy "$SOUT/actor.onnx" --policy-capture-pose --out-dir "$SROOT/physics" \
      > "$SROOT/physics.log" 2>&1
  RC=$?
  set -e
  grep -E "^\{|MAZE-EXPLORE SUMMARY" "$SROOT/physics.log" | cut -c1-400 || true
  [ "$RC" -eq 0 ] || fail "maze_explore exit code $RC (see $SROOT/physics.log)"
  echo "smoke physics episode: $(( $(date +%s) - T1 )) s wall"
  "$PY" - "$SROOT/physics" "$SOUT/actor.onnx" <<'PY' || fail "physics episode checks"
import json, sys
from pathlib import Path
d, onnx = Path(sys.argv[1]), sys.argv[2]
e = json.loads((d / "seed9150.json").read_text())
s = json.loads((d / "summary.json").read_text())
pv, mi = e.get("policy_visitation") or {}, e.get("policy_map_integration") or {}
steps = round(e["elapsed_s"] / 0.04) + 1
checks = {
    "outcome is goal, time_out or fall": e["outcome"] in ("goal", "time_out", "fall"),
    "the exported v5 actor drove it": e.get("policy") == onnx and "LEARNED_navgym_policy" in e.get("control", ""),
    "v5 visitation memory stepped once per control step": pv.get("updates") == steps and pv.get("obs_keys") == ["lidar", "near", "map_visit", "map_coarse", "goal"]
                                                         and pv.get("dt_s") == 0.04 and pv.get("tau_s") == 60.0,
    "capture-pose integration recorded": mi.get("mode") == "capture_pose" and int(mi.get("updates_at_capture_pose", 0)) > 0,
    "hard 6x6, 1 extra opening": (e["maze"]["n"], e["maze"]["m"], e["maze"]["extra_openings"]) == (6, 6, 1),
    "summary: 1 episode, 30 s, capture pose": s["n"] == 1 and s["settings"]["time_limit"] == 30.0 and s["settings"].get("policy_capture_pose") is True,
}
# the transfer judge's validity checks accept a real maze_explore run (only the smoke's seed set and length substituted)
sys.path.insert(0, "scripts/bench")
import navgym_train as nt
rule = {**nt.V5_TRANSFER_RULE, "seeds": [9150], "n": 1, "time_limit_s": 30.0}
v = nt.judge_v5_transfer(["armV5-s99"], {"armV5-s99": {9150: e}}, {"armV5-s99": s}, rule=rule)
checks["judge_v5_transfer judges it without a validity problem"] = v["verdict"] in ("PASS", "NEGATIVE") and not v.get("problems")
print("judge_v5_transfer on the smoke episode:", v["verdict"], v.get("problems"))
for k, v in checks.items():
    print(("ok   " if v else "FAIL ") + k)
print(f"physics episode: outcome {e['outcome']}, elapsed {e['elapsed_s']} s, visitation updates {pv.get('updates')}, "
      f"capture-pose updates {mi.get('updates_at_capture_pose')}, loop-top fallbacks {mi.get('updates_at_loop_top_pose')}, "
      f"wall contacts {e['wall_contact_steps']}, braked steps {e['sensor_stats'].get('braked_robot_steps')}")
sys.exit(0 if all(checks.values()) else 1)
PY
  # 5. the transfer launcher's gate: with this NEGATIVE gym verdict it must not run any episode (exit 0, not_run.json),
  #    and a re-run refuses (exit 3); with no gym verdict, or one that is not final, it stops (exit 4) and writes nothing;
  #    on the leftovers of a transfer run that did not finish it refuses (exit 3) before running anything
  set +e
  NAVGYM_V5_ROOT="$SROOT" NAVGYM_V5_TRANSFER_OUT="$SROOT/transfer-smoke" bash slurm/repo20260923/cpu_navgym_v5_transfer.sbatch \
      > "$SROOT/transfer-gate.log" 2>&1
  RC1=$?
  NAVGYM_V5_ROOT="$SROOT/no-gym-verdict-smoke" NAVGYM_V5_TRANSFER_OUT="$SROOT/transfer-nogv-smoke" \
      bash slurm/repo20260923/cpu_navgym_v5_transfer.sbatch > "$SROOT/transfer-nogv.log" 2>&1
  RC2=$?
  NAVGYM_V5_ROOT="$SROOT" NAVGYM_V5_TRANSFER_OUT="$SROOT/transfer-smoke" bash slurm/repo20260923/cpu_navgym_v5_transfer.sbatch \
      > "$SROOT/transfer-rerun.log" 2>&1
  RC3=$?
  mkdir -p "$SROOT/nonfinal-gv-smoke" "$SROOT/transfer-partial-smoke/armV5-s99"
  echo '{"verdict": "PENDING", "final": false, "per_seed": {}}' > "$SROOT/nonfinal-gv-smoke/verdict_v5.json"   # hand-written
  NAVGYM_V5_ROOT="$SROOT/nonfinal-gv-smoke" NAVGYM_V5_TRANSFER_OUT="$SROOT/transfer-nonfinal-smoke" \
      bash slurm/repo20260923/cpu_navgym_v5_transfer.sbatch > "$SROOT/transfer-nonfinal.log" 2>&1
  RC4=$?
  echo '{}' > "$SROOT/transfer-partial-smoke/armV5-s99/seed9150.json"   # stands in for an unfinished run's leftovers
  NAVGYM_V5_ROOT="$SROOT" NAVGYM_V5_TRANSFER_OUT="$SROOT/transfer-partial-smoke" bash slurm/repo20260923/cpu_navgym_v5_transfer.sbatch \
      > "$SROOT/transfer-partial.log" 2>&1
  RC5=$?
  set -e
  cat "$SROOT/transfer-gate.log" | cut -c1-300
  echo "transfer launcher exits: gate $RC1, no gym verdict $RC2, re-run $RC3, non-final gym verdict $RC4, unfinished run $RC5"
  [ "$RC1" -eq 0 ] && [ -f "$SROOT/transfer-smoke/not_run.json" ] && [ ! -e "$SROOT/transfer-smoke/verdict.json" ] \
      && [ ! -e "$SROOT/transfer-smoke/astar" ] || fail "transfer launcher gate (exit $RC1; see $SROOT/transfer-gate.log)"
  [ "$RC2" -eq 4 ] && [ ! -e "$SROOT/transfer-nogv-smoke" ] || fail "transfer launcher without a gym verdict (exit $RC2, expected 4 and nothing written; see $SROOT/transfer-nogv.log)"
  [ "$RC3" -eq 3 ] && grep -q "REFUSING" "$SROOT/transfer-rerun.log" || fail "transfer launcher re-run over not_run.json (exit $RC3, expected 3)"
  [ "$RC4" -eq 4 ] && [ ! -e "$SROOT/transfer-nonfinal-smoke" ] && grep -q "NOT A FINAL GYM VERDICT" "$SROOT/transfer-nonfinal.log" \
      || fail "transfer launcher on a non-final gym verdict (exit $RC4, expected 4 and nothing written; see $SROOT/transfer-nonfinal.log)"
  [ "$RC5" -eq 3 ] && grep -q "did not finish" "$SROOT/transfer-partial.log" && [ ! -e "$SROOT/transfer-partial-smoke/not_run.json" ] \
      && [ ! -e "$SROOT/transfer-partial-smoke/verdict.json" ] \
      || fail "transfer launcher on an unfinished run's files (exit $RC5, expected 3 and nothing written; see $SROOT/transfer-partial.log)"
  rm -rf "$CACHE"
  echo "SMOKE PASS: v5 training, export, verdict, a 30 s physics episode and the transfer gate all ran ($(( $(date +%s) - T0 )) s wall) | $(date)"
  exit 0
fi
[ "$MODE" = train ] || { echo "NAVGYM_MODE must be train, verdict or smoke, got '$MODE'"; exit 2; }
SEED=${NAVGYM_SEED:-${SLURM_ARRAY_TASK_ID:?run as an array task (8-10) or set NAVGYM_SEED}}
case "$SEED" in 8|9|10) ;; *) echo "v5 seeds are 8, 9, 10 (0-7 were v2 / v3 / v2ctl / v4), got '$SEED'"; exit 2 ;; esac
STEPS=30000000
OUT=$ROOT/armV5-s${SEED}
if [ -e "$OUT" ]; then
  echo "REFUSING: $OUT exists (never overwrite a run directory or its result)"; exit 3
fi
mkdir -p "$OUT"
echo "=== node: $(hostname) | cpu: $(lscpu | sed -n 's/^Model name: *//p') | HEAD $(git rev-parse --short HEAD) | NavGym v5 (env v3, ppo v2, idle cost 0.008) | seed $SEED | steps $STEPS | $(date) ==="
set +e
"$PY" scripts/bench/navgym_train.py --out "$OUT" --seed "$SEED" --total-steps "$STEPS" --env-version 3 --ppo v2 --idle-cost 0.008 \
    --n-envs 16 --eval-every 1000000 --final-eval-fresh-base 61000 2>&1 | grep --line-buffered -vE "Warning|warnings.warn"
RC=${PIPESTATUS[0]}
set -e
echo "trainer exit code $RC (informational; the verdict is computed from summary.json)"
# this seed's checks plus the arm status so far (PENDING while other tasks run); the final verdict is the verdict mode
"$PY" scripts/bench/navgym_train.py verdict-v5 "$ROOT" --seeds 8 9 10 --out "$OUT/verdict_v5_at_end_of_s${SEED}.json"
echo "=== done: $(date) ==="
