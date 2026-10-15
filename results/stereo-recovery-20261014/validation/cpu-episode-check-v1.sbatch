#!/bin/bash
#SBATCH --signal=B:TERM@30
# CPU-only closed-loop plumbing check of the candidate campaign code (seed 4242).
set -euo pipefail
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1
R=/nfs/hpc/share/sanchej7/humanoid-research-20261011
REPO=$R/followups-20261014-stereo/repo
COMMIT=${1:?commit}
SECONDS_EP=${2:-16}
SIF=/nfs/hpc/share/sanchej7/Humanoid_Lite/container/bhl.sif
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
ORB=$R/followups-20261014-stereo/native-atlas-build-v1/runtime.tar.gz
DEST=$R/followups-20261014-stereo/cpu-episode-check-$SLURM_JOB_ID
S=$(mktemp -d /tmp/bhl-h34-recovery-check-${SLURM_JOB_ID}-XXXXXX)
trap 'rm -rf "$S"' EXIT
echo "host=$(hostname) job=$SLURM_JOB_ID commit=$COMMIT"
tar -xzf $R/native-sensor-controls-v2/frozen-campaign.tar.gz -C "$S" --wildcards 'source/external/*' 'inputs/actors/*' 'inputs/runtime/lio-runtime.tar.gz'
git -C "$REPO" archive "$COMMIT" src scripts slurm | tar -x -C "$S/source"
mkdir -p "$S/home" "$S/out" "$S/tmp"
U=$S/source/external/Berkeley-Humanoid-Lite
PP=$S/source/src:$U/source/berkeley_humanoid_lite:$U/source/berkeley_humanoid_lite_assets:$U/source/berkeley_humanoid_lite_lowlevel
run() { apptainer exec --containall --cleanenv --home "$S/home" --bind /nfs/hpc/share/sanchej7:/nfs/hpc/share/sanchej7:ro \
  --bind "$S:$S" --pwd "$S" --env PYTHONPATH="$PP" --env MUJOCO_GL="$1" --env PYOPENGL_PLATFORM="$2" --env OMP_NUM_THREADS=1 \
  --env TMPDIR="$S/tmp" --env PYTHONDONTWRITEBYTECODE=1 "$SIF" "${@:3}"; }
GL=""
for pair in "egl egl" "osmesa osmesa"; do
  set -- $pair
  if run "$1" "$2" "$PY" -c "import mujoco;m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 3\"/><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>');r=mujoco.Renderer(m,48,64);d=mujoco.MjData(m);mujoco.mj_forward(m,d);r.update_scene(d);print('render ok',r.render().shape)"; then GL="$1 $2"; break; fi
done
[ -n "$GL" ] || { echo "no software GL backend"; exit 3; }
echo "using GL: $GL"
set -- $GL
run "$1" "$2" "$PY" $R/followups-20261014-stereo/jobs/cpu_episode_check.py "$S/source" "$S/inputs" "$ORB" "$S/out" "$SECONDS_EP"
mkdir -p "$DEST"
cp "$S/out/check_result.json" "$DEST/"
for arm in permanent_stop resume_verified_relocalization; do
  mkdir -p "$DEST/$arm/native"
  cp "$S/out/$arm/episode.json" "$DEST/$arm/"
  cp "$S/out/$arm/native/native_frames.jsonl" "$S/out/$arm/native/recovery_receipt.json" "$S/out/$arm/native/stream_receipt.json" "$DEST/$arm/native/"
  cp "$S/out/$arm/native/orb/native-orb.log" "$DEST/$arm/native/" 2>/dev/null || true
done
du -sb "$S/out"/* | tee "$DEST/raw-sizes.txt"
