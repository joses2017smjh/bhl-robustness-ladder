#!/bin/bash
#SBATCH --job-name=m7-rerender
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=results/mission7-campaign-20260923/media/m7-rerender-%j.out
# Mission 7 watchable re-renders (2026-09-23). The two Mission 7 GIF pairs were
# rendered with EpisodeRecorder's tracking camera, which sits behind the 1.1 m
# maze walls, and their launchers passed outline_right=True with the success on
# the right, so the successful panel was outlined red. This job re-renders the
# same four episodes (same flags as slurm/mission7_render.sbatch and
# slurm/mission7_render_approach.sbatch, plus --camera overhead: 4.5 m at
# -80 deg, steep enough to clear both the 1.1 m walls and the 1.45-2.55 m door
# lintels the robot stands under during the crossing) into media/overhead/,
# checks that the reruns reproduce the recorded outcomes and that the robot is
# actually in frame, then rebuilds the two GIFs at their documented paths with
# the FAILING panel red and the success green.
# M7_CAMERA=chase is NOT usable here: the ray probe shows the chase preset
# behind a wall for 100 % of the approach clips and ~70 % of the doors clips,
# so it fails the visibility gate (exit 3) by design.
# Runtime: the original renders took 1:52 (doors pair) and 0:44 (approach
# pair) on cn-gpu5/6 (sacct 21401734, 21401987); this job adds one
# segmentation probe per simulated second, two re-encodes and two GIF builds.
# The previous GIFs stay in git history and are copied to
# media/overhead/superseded-cache/ (ignored by .gitignore's *cache* rule).
# MuJoCo EGL needs a GPU node; MuJoCo-only, no container, no Isaac.
# Usage (from the repo root): sbatch slurm/repo20260923/gpu_m7_rerender.sbatch [repo]
set -euo pipefail
repo=${1:-/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder}
cd "$repo"
PY="${PYTHON:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python}"
export PYTHONPATH="$repo/src:$repo/scripts" MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
for t in ffmpeg ffprobe; do command -v "$t" >/dev/null || { echo "$t missing on $(hostname)"; exit 2; }; done   # ffprobe: shrink() reads clip durations
"$PY" -c "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1 1 .1\" type=\"box\"/></worldbody></mujoco>'); mujoco.Renderer(m, 64, 64); print('EGL OK on', __import__('socket').gethostname(), 'mujoco', mujoco.__version__)"

M=results/mission7-campaign-20260923/media
C=$M/overhead
CAM=${M7_CAMERA:-overhead}
mkdir -p "$C" "$C/superseded-cache"
for g in mission7-doors1-stall-vs-crossing-fix mission7-approach-negx-centering; do
  for ext in gif json; do [ -f "docs/gifs/$g.$ext" ] && cp -n "docs/gifs/$g.$ext" "$C/superseded-cache/" || true; done
done
echo "=== camera preset: $CAM ==="

# --- 1. four renders, flags identical to the original launchers ---------------
"$PY" scripts/render_mission7.py --repo "$repo" --out "$C/doors1-baseline-stall.mp4" --stage doors --index 1 --max-seconds 70 \
  --camera "$CAM" --visibility-every 25 \
  --caption "Doors/1 baseline: guarded stage, then a 134 s stall (timeout at 180 s)" \
  --evidence results/mission7-campaign-20260923/campaign-a-doors-00-07/result.json
"$PY" scripts/render_mission7.py --repo "$repo" --out "$C/doors1-crossing-fix-success.mp4" --stage doors --index 1 \
  --stage-activate --stage-wait-open 0 --pre-point 0.45 --cross-clear 0.35 --cross-kick --rejoin-advance --exit-ramp 1.2 --exit-ramp-center \
  --camera "$CAM" --visibility-every 25 \
  --caption "Doors/1 with the crossing fix (kick, cross-until-clear, centre-aimed ramp): success at 77.7 s -- single layout, replay gate not passed" \
  --evidence results/mission7-campaign-20260923/local-stage-v2/doors-1-kick-center/result.json
"$PY" scripts/render_mission7_approach.py --repo "$repo" --out "$C/approach-negx14-baseline.mp4" --controller recovery --index 14 --replay-order=-1,+0 \
  --camera "$CAM" --visibility-every 25 \
  --caption "Approach, world -x, layout 14: 0.50 m/s recovery translation -- hits goal_post_-1 on the first stride" \
  --evidence results/mission7-campaign-20260923/approach-negx-recovery/result.json
"$PY" scripts/render_mission7_approach.py --repo "$repo" --out "$C/approach-negx14-centered.mp4" --controller center_bias --index 14 --replay-order=-1,+0 \
  --camera "$CAM" --visibility-every 25 \
  --caption "Approach, world -x, layout 14: gap-midline centring with +0.015 m pre-bias -- through the posts to the goal (gate 62/64)" \
  --evidence results/mission7-campaign-20260923/approach-matrix-center-bias/result.json

# --- 2. outcome guards: the reruns must reproduce the recorded verdicts --------
"$PY" - "$C" <<'PYEOF'
import json, sys
from pathlib import Path
C = Path(sys.argv[1])
o = {n: json.load(open(C / f"{n}.json"))["outcome_of_this_run"] for n in
     ("doors1-baseline-stall", "doors1-crossing-fix-success", "approach-negx14-baseline", "approach-negx14-centered")}
ok = (not o["doors1-baseline-stall"]["success"] and o["doors1-baseline-stall"]["failure"] is None
      and o["doors1-baseline-stall"]["recording_stopped_early"]
      and o["doors1-crossing-fix-success"]["success"]
      and not o["approach-negx14-baseline"]["success"] and o["approach-negx14-baseline"]["failure"] == "excessive_collision"
      and o["approach-negx14-centered"]["success"])
print(json.dumps(o, indent=1))
if not ok:
    sys.exit("OUTCOME GUARD FAILED: a rerun does not reproduce its recorded verdict; GIFs not rebuilt")
print("outcomes reproduce the recorded verdicts")
PYEOF

# --- 3. robot visibility: exact segmentation probe (sidecar) + ffmpeg statistics
"$PY" - "$C" "${M7_FORCE_GIF:-0}" <<'PYEOF'
import json, sys
from pathlib import Path
from make_gifs import ffmpeg_visibility
C, force = Path(sys.argv[1]), sys.argv[2] == "1"
bad = []
for n in ("doors1-baseline-stall", "doors1-crossing-fix-success", "approach-negx14-baseline", "approach-negx14-centered"):
    meta = json.load(open(C / f"{n}.json"))
    vis = meta.get("robot_visibility") or {}
    stats = ffmpeg_visibility(C / f"{n}.mp4")
    print(f"--- {n}: segmentation probe {json.dumps(vis)}")
    print(f"    ffmpeg central-crop median std {stats['median_central_std']:.2f}, median frame diff {stats['median_central_diff']:.3f} "
          f"(old tracking-camera clips: std 3.5-9, diff 0-0.3)")
    for r in stats["per_second"]:
        print(f"    t={r['t_s']:3d}s  luminance {r['mean_luminance']:6.2f}  central_std {r['central_std']:5.2f}  central_diff {r['central_diff']:6.3f}")
    meta["ffmpeg_visibility"] = {k: stats[k] for k in ("median_central_std", "median_central_diff", "old_tracking_reference")}
    (C / f"{n}.json").write_text(json.dumps(meta, indent=2) + "\n")
    share, mean = vis.get("share_of_probes_with_robot_visible", 0.0), vis.get("robot_pixel_fraction_mean", 0.0)
    if share < 0.9 or mean < 1e-3:
        bad.append((n, share, mean))
if bad:
    print("ROBOT NOT VISIBLE (segmentation probe):", bad)
    if not force:
        print("VISIBILITY GATE FAILED; GIFs not rebuilt (set M7_FORCE_GIF=1 to override)", file=sys.stderr)
        sys.exit(3)
    print("M7_FORCE_GIF=1: continuing despite the visibility gate")
else:
    print("ROBOT VISIBLE in all four clips (>= 90 % of per-second probes, >= 0.1 % of pixels)")
PYEOF

# --- 4. keep every committable MP4 under the 5 MiB artifact guard -------------
shrink() {   # the recipe the original clips record (800 px wide, H.264), bitrate capped by duration
  local f=$1 enc encoders dur kbps
  encoders=$(ffmpeg -hide_banner -encoders 2>/dev/null || true)   # no grep -q in a pipe: pipefail + SIGPIPE
  case " $encoders " in
    *" libopenh264 "*) enc=libopenh264 ;;
    *" libx264 "*)     enc=libx264 ;;
    *)                 enc=mpeg4 ;;
  esac
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
  # <= 900 kb/s, and never more than fits 4.5 MB for this duration (0.9 safety on the 5 MiB guard)
  kbps=$("$PY" -c "import sys; d=float(sys.argv[1]); print(int(min(900, 0.9*4.5e6*8/max(d,1e-3)/1000)))" "$dur")
  ffmpeg -y -loglevel error -i "$f" -vf scale=800:-2 -c:v "$enc" -b:v "${kbps}k" -pix_fmt yuv420p "$f.tmp.mp4"
  mv "$f.tmp.mp4" "$f"
  "$PY" - "$f" "$enc" "$kbps" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
f, enc, kbps = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
j = f.with_suffix(".json"); meta = json.loads(j.read_text())
meta["output_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
meta["re_encoded"] = f"{enc} {kbps} kb/s, 800 px wide, after rendering, to stay under the 5 MB artifact guard; frames unchanged"
j.write_text(json.dumps(meta, indent=2) + "\n")
print(f"re-encoded {f.name} at {kbps} kb/s: {f.stat().st_size/1e6:.2f} MB")
PYEOF
}
for f in "$C"/*.mp4; do
  if [ "$(stat -c %s "$f")" -gt 4500000 ]; then shrink "$f"; else echo "$(basename "$f"): $(stat -c %s "$f") bytes, no re-encode"; fi
done
for f in "$C"/*.mp4; do   # hard stop BEFORE any GIF is touched
  sz=$(stat -c %s "$f"); [ "$sz" -le 5242880 ] || { echo "SIZE GUARD FAILED: $f is $sz bytes (> 5 MiB); GIFs not rebuilt"; exit 4; }
done

# --- 5. pair GIFs, failing panel red, success panel green; sidecars ----------
"$PY" - "$repo" "$C" "$CAM" <<'PYEOF'
import sys
from pathlib import Path
from make_gifs import pair_gif, pair_sidecar
repo, C, cam = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
G = repo / "docs/gifs"
jobs = (
    ("mission7-doors1-stall-vs-crossing-fix", "doors1-baseline-stall", "doors1-crossing-fix-success",
     "baseline: stalls after the plate", "crossing fix: reaches the goal", 40.0, 1.94),
    ("mission7-approach-negx-centering", "approach-negx14-baseline", "approach-negx14-centered",
     "baseline: post collision", "centred: through the gap", 12.0, 1.0),
)
for name, left, right, ll, rl, budget, speed in jobs:
    out = G / f"{name}.gif"
    staged = C / "superseded-cache" / f"{name}.new.gif"     # git-ignored staging; the documented path changes only on success
    prev = C / "superseded-cache" / f"{name}.json"
    kw = dict(max_s=budget, speed=speed, outline_left=True)
    mb = pair_gif(C / f"{left}.mp4", C / f"{right}.mp4", staged, ll, rl, **kw)
    if mb > 4.9:
        print(f"  {staged.name} over budget at {mb:.2f} MB; rebuilding at 8 fps / 96 colours")
        mb = pair_gif(C / f"{left}.mp4", C / f"{right}.mp4", staged, ll, rl, fps=8, max_colors=96, **kw)
    if staged.stat().st_size > 5 * 1024 * 1024:
        print(f"SIZE GUARD FAILED: {staged} still {mb:.2f} MB after the fallback; {out} left unchanged", file=sys.stderr)
        sys.exit(4)
    staged.replace(out)
    pair_sidecar(out, C / f"{left}.mp4", C / f"{right}.mp4", repo, left_label=ll, right_label=rl, red_panel="left",
                 playback_speed=speed, gif_budget_s=budget, camera=cam,
                 carry_from=prev if prev.is_file() else G / f"{name}.json",
                 extra={"red_outline_from": "0.4 s before the end of the failing clip (source time), the evaluator's hold on the verdict",
                        "rebuilt_by": "slurm/repo20260923/gpu_m7_rerender.sbatch"})
    print(f"  sidecar {out.with_suffix('.json').relative_to(repo)} written")
PYEOF

# --- 6. size guard on this job's own outputs (5 MiB committable limit) --------
for f in "$C"/*.mp4 docs/gifs/mission7-doors1-stall-vs-crossing-fix.gif docs/gifs/mission7-approach-negx-centering.gif; do
  sz=$(stat -c %s "$f"); echo "$sz bytes  $f"
  [ "$sz" -le 5242880 ] || { echo "SIZE GUARD FAILED: $f exceeds 5 MiB"; exit 4; }
done
# The repo-wide guard is informational here: it already fails on pre-existing
# untracked trace .json.gz files under results/mission7-*, which are not this job's.
"$PY" -m unittest tests.test_result_artifacts 2>&1 | tail -3 || echo "(repo-wide artifact test failed: see the oversized list above; this job's files passed)"
# .gitignore's `results/mission7-*/**/*doors*.json` rule (raw door traces) also
# matches the two doors sidecars written here; the clips themselves are not
# ignored. Stage the sidecars explicitly so no clip is committed without its record.
echo "NOTE: git ignores the doors sidecars by pattern; stage them with:"
echo "  git add -f $C/doors1-baseline-stall.json $C/doors1-crossing-fix-success.json"
echo MISSION7_RERENDER_COMPLETE
