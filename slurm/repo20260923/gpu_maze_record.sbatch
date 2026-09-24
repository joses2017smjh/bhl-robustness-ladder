#!/bin/bash
#SBATCH --job-name=maze-record
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --exclude=dgxh-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Repo-wide campaign (2026-09-23): Isaac recording of the maze-recovery
# success for a GIF, plus one failure example for the pair.
#
# The published Full-stage Both s0 policy (model_5997.pt, 32/32 first-episode
# success at seed 100, noise-on rescore 380/384 pooled) has numbers and no
# video. scripts/bench/maze_record.py replays it on v60 with the training
# observation noise ON, num_envs 1, until the first termination, through two
# recorders at once: gym RecordVideo on the Kit viewport (train.py --video; on
# this stack it draws the stale USD pose, docs/ISAAC_RENDER.md s10) and the
# camera sensor that follows the robot (every published Isaac clip). The GIF
# is built from the camera-sensor clip; the viewport clip is kept beside it.
#
# Boot 1  Both s0:  success at seed 100 (up to MAZE_REC_SUCCESS_SEEDS seeds if
#                   seed 100 itself fails), then a failure search over seeds
#                   101.. (cap MAZE_REC_FAIL_SEEDS = 8), then one controlled
#                   failure: the same seed 100 with a 2-step (40 ms) IMU delay
#                   (SF-01: 0.02 success), the tier-3 fallback for the pair.
# Boot 2  Blind s1 (published 29/32): only if boot 1 found no natural failure;
#                   failure search over seeds 101.. (cap 8).
# Then, outside the container: docs/gifs/isaac/maze_recovery_pair.gif with
# scripts/make_gifs.py pair_gif (success left / green, failure right / red)
# and its JSON sidecar.
#
# Verdict, computed from the JSON files (never from exit codes; Kit's close
# exits 0):
#   PASS        success clip + a failure clip of the published policy (Both s0
#               natural failure, or Blind s1) + GIF + sidecar
#   INCOMPLETE  success clip, but the only failure is the IMU-delay control
#               (GIF built and labelled as that) or no failure at all (no GIF)
#   FAIL        no usable success clip
#
# RTX renders: dgxh-1 has no Vulkan for Isaac Sim 6.0 (97b/97c), excluded.
# The viewport (RecordVideo) clip is taken only for the first episode of each
# boot (MAZE_REC_VIEWPORT_EPISODES): it is a diagnostic of the stale-pose
# finding, and its Kit app update per frame should not ride along on every
# GIF-bound episode. Each episode JSON carries sim_time_per_step when the
# stack exposes sim time, so a double-stepped viewport episode would show.
#
# Time: 2 Kit boots (~3-4 min each incl. v60_boot_gate) + <= 15 episodes of
# <= 600 steps with 640x360 renders per step (~1 min each) + the GIF.
# Budget checks stop new boots after MAZE_REC_BUDGET_S so the verdict prints.
set -uo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MAZE_REC_OUT=${MAZE_REC_OUT:-$REPO/results/repo-gpu-20260923/maze-record}
export MAZE_REC_STEPS=${MAZE_REC_STEPS:-600}
export MAZE_REC_TIMEOUT=${MAZE_REC_TIMEOUT:-1500}
export MAZE_REC_VIEWPORT_EPISODES=${MAZE_REC_VIEWPORT_EPISODES:-1}
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS MAZE_REC_OUT MAZE_REC_STEPS MAZE_REC_TIMEOUT MAZE_REC_VIEWPORT_EPISODES MAZE_REC_TASK MAZE_REC_CKPT MAZE_REC_LABEL MAZE_REC_SEARCHES"
SUCCESS_SEEDS=${MAZE_REC_SUCCESS_SEEDS:-4}
FAIL_SEEDS=${MAZE_REC_FAIL_SEEDS:-8}
BUDGET_S=${MAZE_REC_BUDGET_S:-3300}
GIF=$REPO/docs/gifs/isaac/maze_recovery_pair.gif
PUB_DIR=$REPO/results/weekend-20260919
mkdir -p "$MAZE_REC_OUT" "$(dirname "$GIF")"
source "$REPO/slurm/inner/v60_boot_gate.sh"
BOOT_LOCK="${V60_BOOT_LOCK:-/nfs/hpc/share/$USER/Humanoid_Lite/.v60-boot.lock}"
PYH=$WORKSPACE/venv/bin/python; [ -x "$PYH" ] || PYH=$(command -v python3)

echo "=== node: $(hostname) | maze recovery recording | stack $BHL_STACK | job ${SLURM_JOB_ID:-?} | $(date) ==="
cd "$REPO" && echo "repo HEAD $(git rev-parse --short HEAD 2>/dev/null) branch $(git branch --show-current 2>/dev/null)"

# --- preflight: recorder helpers, checkpoints as published, node tooling ---
"$PYH" "$REPO/scripts/bench/maze_record.py" --selftest || { echo "MAZE-RECORD RESULT: FAIL (maze_record.py selftest)"; exit 1; }
declare -A CKPTS
for spec in Both:0:both-s0 Blind:1:blind-s1; do
    IFS=: read -r arm seed key <<<"$spec"
    pub="$PUB_DIR/maze-eval-Full-$arm-s$seed.json"
    ck=$("$PYH" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$pub" 2>/dev/null) || ck=""
    lower=${arm,,}
    case "$ck" in
        */logs/rsl_rl/biped/2026-09-19_*_wknd-full-$lower-s$seed/model_*.pt) ;;
        *) echo "MAZE-RECORD RESULT: FAIL (bad/unreadable published checkpoint for $arm s$seed: '$ck')"; exit 1;;
    esac
    [ -f "$ck" ] || { echo "MAZE-RECORD RESULT: FAIL (checkpoint missing: $ck)"; exit 1; }
    CKPTS[$key]=$ck
    echo "ckpt $key -> $ck ($(sha256sum "$ck" | cut -c1-16)...)"
done
for tool in ffmpeg ffprobe; do
    command -v "$tool" >/dev/null || { echo "MAZE-RECORD RESULT: FAIL ($tool missing on $(hostname))"; exit 1; }
done
# Stale outputs of the searches this job repeats must not be able to score it.
for n in both-s0-success both-s0-failure both-s0-delay40ms blind-s1-failure; do
    rm -f "$MAZE_REC_OUT/$n.json" "$MAZE_REC_OUT/$n.mp4" "$MAZE_REC_OUT/$n.viewport.mp4"
done
rm -f "$MAZE_REC_OUT"/both-s0-s*.json "$MAZE_REC_OUT"/both-s0-s*.mp4 "$MAZE_REC_OUT"/blind-s1-s*.json "$MAZE_REC_OUT"/blind-s1-s*.mp4 \
      "$MAZE_REC_OUT"/*.error.txt "$MAZE_REC_OUT"/*.episodes.json

# --- inner script (runs inside the container; parameters arrive as forwarded env) ---
INNER=$MAZE_REC_OUT/.inner-${SLURM_JOB_ID:-local}.sh
cat >"$INNER" <<'INNER_EOF'
#!/bin/bash
set -uo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
: "${MAZE_REC_TASK:?}" "${MAZE_REC_CKPT:?}" "${MAZE_REC_LABEL:?}" "${MAZE_REC_SEARCHES:?}"
log="$MAZE_REC_OUT/$MAZE_REC_LABEL.log"
echo "=== inner: $MAZE_REC_TASK | $MAZE_REC_LABEL | searches: $MAZE_REC_SEARCHES | $(date) ==="
args=()
for s in $MAZE_REC_SEARCHES; do args+=(--search "$s"); done
timeout "$MAZE_REC_TIMEOUT" "$PY" "$REPO/scripts/bench/maze_record.py" --headless --enable_cameras \
    --task "$MAZE_REC_TASK" --checkpoint "$MAZE_REC_CKPT" --label "$MAZE_REC_LABEL" \
    --out-dir "$MAZE_REC_OUT" --max-steps "$MAZE_REC_STEPS" --viewport-episodes "$MAZE_REC_VIEWPORT_EPISODES" \
    "${args[@]}" > "$log" 2>&1
rc=$?
grep -E '^(SEARCH |\{"episode"|\[viewport\]|\[sensor\]|\[clip\]|RECORD-ERROR)' "$log" | cut -c1-300 || true
echo "=== inner exit $rc for $MAZE_REC_LABEL ==="
INNER_EOF
chmod +x "$INNER"

run_boot() {   # label task ckpt searches...
    local label=$1 task=$2 ckpt=$3; shift 3
    if [ "$SECONDS" -gt "$BUDGET_S" ]; then
        echo "=== budget ${BUDGET_S}s exceeded at ${SECONDS}s: skipping boot $label ==="; return 1
    fi
    v60_boot_gate
    local gate_pid=${!:-}
    echo "=== boot $label | t=${SECONDS}s | $(date) ==="
    MAZE_REC_LABEL="$label" MAZE_REC_TASK="$task" MAZE_REC_CKPT="$ckpt" MAZE_REC_SEARCHES="$*" bhl_exec "$INNER"
    echo "=== boot $label exit $? | t=${SECONDS}s ==="
    kill "$gate_pid" 2>/dev/null || true
    _v60_release "$BOOT_LOCK"
    return 0
}

outcome_of() {   # search name -> outcome string or "" (missing/None)
    "$PYH" -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("outcome") or "")
except Exception: print("")' "$MAZE_REC_OUT/$1.json" 2>/dev/null
}

# Boot 1: Both s0 -- success, natural-failure search, IMU-delay control.
run_boot both-s0 Velocity-BHL-MazeRecovery-Full-Both-v0 "${CKPTS[both-s0]}" \
    "both-s0-success:success:100:$SUCCESS_SEEDS" "both-s0-failure:failure:101:$FAIL_SEEDS" "both-s0-delay40ms:failure:100:2:2"
echo "both-s0-success -> '$(outcome_of both-s0-success)' | both-s0-failure -> '$(outcome_of both-s0-failure)' | both-s0-delay40ms -> '$(outcome_of both-s0-delay40ms)'"

# Boot 2: Blind s1 only when the published Both policy did not fail on its own.
if [ -z "$(outcome_of both-s0-failure)" ]; then
    run_boot blind-s1 Velocity-BHL-MazeRecovery-Full-Blind-v0 "${CKPTS[blind-s1]}" "blind-s1-failure:failure:101:$FAIL_SEEDS"
    echo "blind-s1-failure -> '$(outcome_of blind-s1-failure)'"
else
    echo "=== Both s0 failed on its own; Blind s1 boot not needed ==="
fi
rm -f "$INNER"

# --- GIF + sidecar + verdict, from the JSONs, outside the container ---
cd "$REPO"
"$PYH" - "$MAZE_REC_OUT" "$GIF" "$SUCCESS_SEEDS" "$FAIL_SEEDS" <<'PY'
import hashlib, json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, "scripts")
import make_gifs
from make_gifs import pair_gif

out, gif = Path(sys.argv[1]), Path(sys.argv[2])
success_seeds, fail_seeds = int(sys.argv[3]), int(sys.argv[4])
repo = Path.cwd()
problems, notes = [], []

def sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

def load(name):
    p = out / f"{name}.json"
    if not p.is_file():
        return None
    try:
        return json.load(open(p))
    except Exception as exc:
        problems.append(f"{name}.json unreadable: {exc}"); return None

def need_frames(rec):
    # Every control step is captured, so completeness is frames vs steps; a fall
    # can end an episode in < 50 steps (SF-04 fell at ~45), so no fixed floor.
    return max(10, int(0.9 * int(rec.get("steps") or 0)))

def clip(rec, name):
    """The camera-sensor mp4 for a search result; assembled from the node-local PNGs if the writer failed."""
    if rec is None or not rec.get("outcome"):
        return None
    mp4 = rec.get("mp4")
    if mp4 and Path(mp4).is_file():
        return Path(mp4)
    cs = rec.get("camera_sensor") or {}
    png = Path(cs.get("png_dir") or "")
    n = len(list(png.glob("frame_*.png"))) if png.is_dir() else 0
    if n >= need_frames(rec):
        dst = out / f"{name}.mp4"
        # This cluster's ffmpeg has no libx264 (bhl_robust.eval.video); mpeg4 is the fallback.
        for enc, extra in (("libx264", ["-preset", "veryfast", "-crf", "23"]), ("mpeg4", ["-q:v", "3"])):
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(round(rec.get("video_fps", 50))),
                                "-i", str(png / "frame_%04d.png"), "-c:v", enc, *extra, "-pix_fmt", "yuv420p", str(dst)])
            if r.returncode == 0 and dst.is_file() and dst.stat().st_size > 0:
                notes.append(f"{name}: mp4 assembled from {n} PNG frames with ffmpeg/{enc} (writer_error={cs.get('writer_error')})")
                rec["mp4"] = str(dst); (out / f"{name}.json").write_text(json.dumps(rec, indent=2) + "\n")
                return dst
    problems.append(f"{name}: no camera-sensor clip (writer_error={cs.get('writer_error')}, png frames={n})")
    return None

def frames_ok(rec, name):
    n = int((rec.get("camera_sensor") or {}).get("frames") or 0)
    if n < need_frames(rec):
        problems.append(f"{name}: only {n} camera-sensor frames for {rec.get('steps')} steps")
    return n >= need_frames(rec)

def sim_time_check(rec, name):
    """The viewport diagnostic runs a Kit app update per frame; sim_time_per_step > step_dt would mean it stepped physics."""
    spt, dt = rec.get("sim_time_per_step"), rec.get("step_dt")
    if spt is not None and dt and abs(float(spt) - float(dt)) / float(dt) > 0.05:
        notes.append(f"{name}: sim_time_per_step {float(spt):.4f} s differs from step_dt {float(dt):.4f} s "
                     f"(viewport recorded={bool((rec.get('viewport') or {}).get('recorded'))}); check the clip's real-time rate")

def policy_label(rec):
    t = rec.get("task", "")
    arm = t.replace("Velocity-BHL-MazeRecovery-Full-", "").replace("-v0", "")
    seed = os.path.basename(os.path.dirname(rec.get("checkpoint", "")))[-2:]
    return f"{arm} {seed}"

succ = load("both-s0-success")
ok_success = bool(succ and succ.get("outcome") == "success" and clip(succ, "both-s0-success") and frames_ok(succ, "both-s0-success"))
if succ and succ.get("outcome"):
    sim_time_check(succ, "both-s0-success")
if not ok_success:
    problems.append(f"no success clip (both-s0-success outcome={succ.get('outcome') if succ else 'missing'}, "
                    f"tried={succ.get('tried') if succ else None})")

tiers = (("both-s0-failure", "natural"), ("blind-s1-failure", "blind"), ("both-s0-delay40ms", "delay"))
fail, fail_name, tier = None, None, None
for name, t in tiers:
    rec = load(name)
    if rec and rec.get("outcome") and rec["outcome"] not in ("success", "step_cap") and clip(rec, name) and frames_ok(rec, name):
        fail, fail_name, tier = rec, name, t; sim_time_check(rec, name); break
    if rec and rec.get("outcome") is None:
        notes.append(f"{name}: no failure in {len(rec.get('tried') or [])} seed(s)")

verdict = "FAIL"
sidecar = None
if ok_success and fail is not None:
    # Older make_gifs hard-codes FONT at a Debian path; newer ones resolve it with font_file().
    if getattr(make_gifs, "FONT", None) and not Path(make_gifs.FONT).is_file():
        for cand in ("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            if Path(cand).is_file():
                make_gifs.FONT = cand; break
    term = fail["outcome"].replace("base_orientation", "FALL").replace("dead_end", "DEAD END").replace("time_out", "TIMEOUT").replace("_", " ")
    # pair_gif draws at a fixed 19 px on a 430 px panel: about 38 characters fit.
    left_label = f"{policy_label(succ)}  seed {succ['seed']}  REACHED  {succ['sim_seconds']:.1f}s  1x"
    if tier == "delay":
        right_label = f"{policy_label(fail)} s{fail['seed']} +40ms IMU lag  {term}  1x"
    else:
        right_label = f"{policy_label(fail)}  seed {fail['seed']}  {term}  {fail['sim_seconds']:.1f}s  1x"
    for bad in (",", ":", "'", "%"):
        left_label, right_label = left_label.replace(bad, " "), right_label.replace(bad, " ")
    if len(right_label) > 38:
        right_label = f"{policy_label(fail)} s{fail['seed']} {term}  1x"[:38]
    try:
        mb = pair_gif(Path(succ["mp4"]), Path(fail["mp4"]), gif, left_label, right_label, max_s=12.0, outline_right=True)
    except Exception as exc:
        problems.append(f"pair_gif failed: {exc!r}"); mb = None
    if gif.is_file() and mb:
        rel = lambda p: os.path.relpath(p, repo)
        ev = [out / "both-s0-success.json", out / f"{fail_name}.json"]
        sidecar = {
            "output": rel(gif), "output_sha256": sha(gif), "output_mb": round(mb, 2),
            "source_clip": [rel(succ["mp4"]), rel(fail["mp4"])],
            "source_sha256": [sha(succ["mp4"]), sha(fail["mp4"])],
            "viewport_clip": [rel(succ["viewport_mp4"]) if succ.get("viewport_mp4") else None,
                              rel(fail["viewport_mp4"]) if fail.get("viewport_mp4") else None],
            "evidence": [rel(p) for p in ev], "evidence_sha256": [sha(p) for p in ev],
            "labels_drawn": [left_label, right_label],
            "left": {"policy": policy_label(succ), "task": succ["task"], "checkpoint": succ["checkpoint"],
                     "checkpoint_sha256": succ["checkpoint_sha256"], "seed": succ["seed"], "outcome": succ["outcome"],
                     "terminations_fired": succ["terminations_fired"], "steps": succ["steps"], "sim_seconds": succ["sim_seconds"],
                     "imu_delay_steps": succ.get("imu_delay_steps", 0)},
            "right": {"policy": policy_label(fail), "task": fail["task"], "checkpoint": fail["checkpoint"],
                      "checkpoint_sha256": fail["checkpoint_sha256"], "seed": fail["seed"], "outcome": fail["outcome"],
                      "terminations_fired": fail["terminations_fired"], "steps": fail["steps"], "sim_seconds": fail["sim_seconds"],
                      "imu_delay_steps": fail.get("imu_delay_steps", 0), "tier": tier},
            "expected_success": {"left": True, "right": False},
            "capture_path": "camera_sensor (CameraCfg following the robot root; the gym RecordVideo viewport clip is kept beside each mp4 as *.viewport.mp4 and draws the stale USD pose on this stack, docs/ISAAC_RENDER.md s10)",
            "observation_corruption": succ.get("observation_corruption"), "num_envs": 1,
            "sim_dt": succ.get("sim_dt"), "step_dt": succ.get("step_dt"), "video_fps": succ.get("video_fps"),
            "playback_speed": 1.0, "stack": succ.get("stack"), "git_commit": succ.get("git_commit"),
            "search_caps": {"success_seeds": success_seeds, "failure_seeds": fail_seeds},
            "scope": "Isaac Sim 6.0 / Lab 3.0.0b2 | learned PPO gait + oracle waypoint teacher | training observation noise on | 1x",
            "notes": notes,
        }
        gif.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
        verdict = "PASS" if tier in ("natural", "blind") else "INCOMPLETE"
elif ok_success:
    verdict = "INCOMPLETE"
    problems.append("no failure clip: nothing to put under the red outline, GIF not built")

print()
print(f"{'search':22} {'outcome':20} {'seed':>4} {'steps':>5} {'sensor':>6} {'viewport':>8}  clip")
for name in ("both-s0-success", "both-s0-failure", "blind-s1-failure", "both-s0-delay40ms"):
    r = load(name)
    if r is None:
        print(f"{name:22} {'(not run)':20}"); continue
    if not r.get("outcome"):
        print(f"{name:22} {'none in ' + str(len(r.get('tried') or [])) + ' seeds':20}"); continue
    print(f"{name:22} {r['outcome']:20} {r['seed']:>4} {r['steps']:>5} {(r.get('camera_sensor') or {}).get('frames', 0):>6} "
          f"{(r.get('viewport') or {}).get('frames', 0):>8}  {r.get('mp4')}")
for n in notes:
    print("NOTE:", n)
for p in problems:
    print("PROBLEM:", p)
if sidecar:
    print(f"GIF: {sidecar['output']} ({sidecar['output_mb']} MB) sidecar {os.path.relpath(gif.with_suffix('.json'), repo)}")
    print(f"     left  {sidecar['labels_drawn'][0]}\n     right {sidecar['labels_drawn'][1]}")
reason = {"PASS": f"success + {tier} failure clip + GIF + sidecar",
          "INCOMPLETE": ("pair built from the IMU-delay control, no natural failure" if sidecar else "success clip only, no failure clip"),
          "FAIL": "no usable success clip"}[verdict]
print(f"MAZE-RECORD RESULT: {verdict} ({reason})")
sys.exit(0 if verdict == "PASS" else 1)
PY
rc=$?
rm -rf "${TMPDIR:-/tmp}/maze-record-frames"
echo "=== done: t=${SECONDS}s | $(date) ==="
exit $rc
