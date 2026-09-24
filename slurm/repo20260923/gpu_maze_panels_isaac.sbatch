#!/bin/bash
#SBATCH --job-name=maze-panels-isaac
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --exclude=dgxh-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-09-24, user request: the Isaac B5 maze (Full stage, the published Both
# s0 policy: lidar + stereo ARE its inputs) as a three-panel clip -- a fixed
# overhead view, the stereo pair's ray depth (raw 64x64 and the pooled 4x4 the
# policy reads) and the 36-sector lidar with its 500 raw hits.
#
# Also answers "why is the Isaac clip so grainy": performance.kit leaves the
# stochastic direct lighting on with the DL denoiser off (NGX is unavailable
# here). This job renders with those terms off (--render-quality clean) and
# measures the grain before recording. Predeclared: spatial grain (mean
# |f - box3(f)|) <= 12 on 0-255 (a third of the stock clip's 36.1) means the
# renderer fix is enough; otherwise the frames get an nlmeans pass before
# composition and the sidecar says so.
set -uo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}" OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MAZE_REC_OUT=${MAZE_REC_OUT:-$REPO/results/repo-gpu-20260923/maze-panels}
export MAZE_REC_TIMEOUT=${MAZE_REC_TIMEOUT:-2400}
export MAZE_REC_QUALITY=${MAZE_REC_QUALITY:-clean}
export MAZE_REC_FRAMES=${TMPDIR:-/tmp}/maze-panels-frames-$SLURM_JOB_ID
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS MAZE_REC_OUT MAZE_REC_TIMEOUT MAZE_REC_QUALITY MAZE_REC_FRAMES MAZE_REC_TASK MAZE_REC_CKPT MAZE_REC_LABEL MAZE_REC_SEARCHES"
LABEL=both-s0-panels-$MAZE_REC_QUALITY
GIF=${MAZE_PANELS_GIF:-$REPO/docs/gifs/isaac/maze_both_panels-$MAZE_REC_QUALITY.gif}
PUB=$REPO/results/weekend-20260919/maze-eval-Full-Both-s0.json
mkdir -p "$MAZE_REC_OUT" "$MAZE_REC_FRAMES" "$(dirname "$GIF")"
source "$REPO/slurm/inner/v60_boot_gate.sh"
PYH=$WORKSPACE/venv/bin/python; [ -x "$PYH" ] || PYH=$(command -v python3)
echo "=== node: $(hostname) | Isaac maze panels | quality $MAZE_REC_QUALITY | job $SLURM_JOB_ID | $(date) ==="
cd "$REPO" && echo "repo HEAD $(git rev-parse --short HEAD)"
"$PYH" "$REPO/scripts/bench/maze_record.py" --selftest || { echo "MAZE-PANELS RESULT: FAIL (selftest)"; exit 1; }
CK=$("$PYH" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$PUB")
[ -f "$CK" ] || { echo "MAZE-PANELS RESULT: FAIL (checkpoint missing: $CK)"; exit 1; }
echo "checkpoint $CK"
rm -f "$MAZE_REC_OUT"/"$LABEL"*.json "$MAZE_REC_OUT"/"$LABEL"*.mp4 "$MAZE_REC_OUT"/"$LABEL"*.npz "$MAZE_REC_OUT"/"$LABEL"*.error.txt

INNER=$MAZE_REC_OUT/.inner-$SLURM_JOB_ID.sh
cat >"$INNER" <<'INNER_EOF'
#!/bin/bash
set -uo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
log="$MAZE_REC_OUT/$MAZE_REC_LABEL.log"
args=(); for s in $MAZE_REC_SEARCHES; do args+=(--search "$s"); done
timeout "$MAZE_REC_TIMEOUT" "$PY" "$REPO/scripts/bench/maze_record.py" --headless --enable_cameras \
    --task "$MAZE_REC_TASK" --checkpoint "$MAZE_REC_CKPT" --label "$MAZE_REC_LABEL" --out-dir "$MAZE_REC_OUT" \
    --frames-root "$MAZE_REC_FRAMES" --max-steps 600 --no-viewport --panels --render-quality "$MAZE_REC_QUALITY" \
    --top-width 1280 --top-height 720 "${args[@]}" > "$log" 2>&1
rc=$?
grep -E '^(SEARCH |\{"episode"|\[panels\]|\[sensor\]|\[clip\]|RECORD-ERROR)' "$log" | cut -c1-400 || true
echo "=== inner exit $rc ==="
INNER_EOF
chmod +x "$INNER"
v60_boot_gate; gate_pid=${!:-}
MAZE_REC_LABEL="$LABEL" MAZE_REC_TASK=Velocity-BHL-MazeRecovery-Full-Both-v0 MAZE_REC_CKPT="$CK" \
    MAZE_REC_SEARCHES="$LABEL:success:100:3" bhl_exec "$INNER"
kill "$gate_pid" 2>/dev/null || true; _v60_release "${V60_BOOT_LOCK:-/nfs/hpc/share/$USER/Humanoid_Lite/.v60-boot.lock}"
rm -f "$INNER"

# --- compose + GIF outside the container ---
cd "$REPO"
"$PYH" - "$MAZE_REC_OUT" "$GIF" "$MAZE_REC_QUALITY" "$LABEL" <<'PY'
import json, os, subprocess, sys, hashlib
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/bench")
from bhl_robust.eval import panels
import compose_panels
out, gif, quality, label = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
rec_p = out / f"{label}.json"
if not rec_p.is_file():
    print("MAZE-PANELS RESULT: FAIL (no search result json)"); sys.exit(1)
rec = json.load(open(rec_p))
if rec.get("outcome") != "success":
    print(f"MAZE-PANELS RESULT: FAIL (outcome {rec.get('outcome')}, tried {rec.get('tried')})"); sys.exit(1)
pan = rec.get("panels") or {}
if not pan.get("npz") or not pan.get("top_frames"):
    print(f"MAZE-PANELS RESULT: FAIL (panels data missing: {pan})"); sys.exit(1)
top = Path(pan["top_png_dir"]); note = f"quality={quality}, grain {pan.get('grain_spatial')}"
denoised = False
if pan.get("needs_denoise"):
    dn = top.parent / (top.name + "-dn"); dn.mkdir(exist_ok=True)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "25", "-i", str(top / "frame_%04d.png"),
                        "-vf", "nlmeans=s=6:p=5:r=11", "-start_number", "0", str(dn / "frame_%04d.png")])
    if r.returncode == 0 and any(dn.glob("frame_*.png")):
        top, denoised, note = dn, True, note + ", nlmeans denoise applied"
mp4 = out / f"{label}.panels.mp4"
res = compose_panels.compose(rec, top, mp4, out / f".{label}-frames", note)
if not res.get("mp4"):
    png = Path(res["png_dir"])
    for enc, extra in (("libx264", ["-preset", "veryfast", "-crf", "20"]), ("mpeg4", ["-q:v", "3"])):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(round(rec["video_fps"])), "-i",
                            str(png / "frame_%04d.png"), "-c:v", enc, *extra, "-pix_fmt", "yuv420p", str(mp4)])
        if r.returncode == 0 and mp4.is_file():
            res["mp4"] = str(mp4); break
if not res.get("mp4"):
    print(f"MAZE-PANELS RESULT: FAIL (compose produced no mp4: {res})"); sys.exit(1)
g = panels.write_gif(mp4, gif, fps=8, speed=1.0, width=860)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
side = {"output": os.path.relpath(gif, Path.cwd()), "output_sha256": sha(gif), "output_mb": g["mb"], "gif": g,
        "source_clip": os.path.relpath(mp4, Path.cwd()), "source_sha256": sha(mp4),
        "evidence": os.path.relpath(rec_p, Path.cwd()), "evidence_sha256": sha(rec_p),
        "episode": {k: rec.get(k) for k in ("task", "checkpoint", "checkpoint_sha256", "seed", "outcome", "terminations_fired",
                                            "steps", "sim_seconds", "observation_corruption", "stack", "git_commit")},
        "render": {"quality": quality, "settings": rec.get("render_settings"), "effective": rec.get("rtx_effective"), "grain_spatial": pan.get("grain_spatial"),
                   "grain_temporal_mad": pan.get("grain_temporal_mad"), "grain_baseline": pan.get("grain_baseline"),
                   "denoised": denoised, "top_eye_offset_m": rec.get("top_eye_offset_m"), "top_frames": pan.get("top_frames"),
                   "sensor_rows": pan.get("sensor_rows"), "sensor_read_errors": pan.get("n_errors")},
        "panels": "top: fixed overhead RTX camera over env 0; stereo: RayCasterCamera ray depth 64x64 per eye + the pooled 4x4 the policy reads; lidar: 36 sector minima of 500 rays as the policy reads them, raw hits behind",
        "scope": "Isaac Sim 6.0 / Lab 3.0.0b2 | learned PPO policy consuming lidar + stereo + IMU | oracle waypoint heading | training observation noise on | 1x",
        "playback_speed": 1.0}
gif.with_suffix(".json").write_text(json.dumps(side, indent=2) + "\n")
print(f"GIF {side['output']} {g['mb']} MB (fps {g['fps']}, colours {g['max_colors']}, width {g['width']})")
print(f"grain spatial {pan.get('grain_spatial')} temporal {pan.get('grain_temporal_mad')} denoised={denoised}")
print("MAZE-PANELS RESULT: PASS" + (" (renderer fix sufficient)" if not denoised else " (nlmeans fallback used)"))
PY
rc=$?
rm -rf "$MAZE_REC_FRAMES"
echo "=== done: $(date) ==="; exit $rc
