#!/bin/bash
#SBATCH --signal=B:TERM@30
# CPU-only relink of the ORB wrapper with read-only Atlas telemetry.
# Sources come from the exact commit below; libORB_SLAM3.so is reused unchanged.
set -euo pipefail
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
R=/nfs/hpc/share/sanchej7/humanoid-research-20261011
REPO=$R/followups-20261014-stereo/repo
COMMIT=8f7ddb0
SIF=/nfs/hpc/share/sanchej7/Humanoid_Lite/container/bhl.sif
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
BASE_SHA=91ca4b6e4ea8f318998f400d4f7dfcd6fe8ae88b375399c79d5d633c39d71516
LIB_SHA=c4f09d2dc42571310f4324b8d5cc460158d6bc34d9fe63affbc009cdabc80ffa
OUT=$R/followups-20261014-stereo/native-atlas-build-v1
WORK=$(mktemp -d /tmp/bhl-stereo-atlas-build-${SLURM_JOB_ID}-XXXXXX)
trap 'rm -rf "$WORK"' EXIT
echo "host=$(hostname) job=$SLURM_JOB_ID work=$WORK"
mkdir -p "$WORK/source" "$WORK/input" "$WORK/home"
git -C "$REPO" archive "$COMMIT" scripts/native src/bhl_robust | tar -x -C "$WORK/source"
tar -xzf "$R/native-sensor-controls-v2/frozen-campaign.tar.gz" -C "$WORK/input" inputs/runtime/orb-runtime.tar.gz
echo "$BASE_SHA  $WORK/input/inputs/runtime/orb-runtime.tar.gz" | sha256sum -c -
apptainer exec --containall --cleanenv --home "$WORK/home" \
  --bind /nfs/hpc/share/sanchej7:/nfs/hpc/share/sanchej7 --bind "$WORK:$WORK" --pwd "$WORK" "$SIF" \
  "$PY" "$WORK/source/scripts/native/orb_diagnostics_build.py" \
    --original-runtime "$WORK/input/inputs/runtime/orb-runtime.tar.gz" --original-runtime-sha256 "$BASE_SHA" \
    --work "$WORK/build" --output "$OUT" --atlas-telemetry --expected-native-library-sha256 "$LIB_SHA"
# Independent post-build checks inside the same image.
apptainer exec --containall --cleanenv --home "$WORK/home" --bind /nfs/hpc/share/sanchej7:/nfs/hpc/share/sanchej7 "$SIF" \
  env LD_LIBRARY_PATH="$OUT/runtime/lib" "$OUT/runtime/bin/orb_native" --version
"$PY" - "$OUT" "$COMMIT" "$REPO" "$SIF" <<'PYEOF'
import hashlib, json, os, pathlib, subprocess, sys
from datetime import datetime, timezone
out, commit, repo, sif = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
def d(p):
    h = hashlib.sha256()
    with open(p, 'rb') as s:
        for b in iter(lambda: s.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
result = json.loads((out / 'build-result.json').read_text())
assert result['status'] == 'PASS' and d(out / 'runtime.tar.gz') == result['runtime_archive_sha256']
prov = result['provenance']
assert prov['native_library_unchanged_sha256'] == 'c4f09d2dc42571310f4324b8d5cc460158d6bc34d9fe63affbc009cdabc80ffa'
assert d(out / 'runtime/lib/libORB_SLAM3.so') == prov['native_library_unchanged_sha256']
assert prov['atlas_telemetry'] == 'bhl-orb-atlas-telemetry-v1'
full = subprocess.check_output(['git', '-C', repo, 'rev-parse', commit], text=True).strip()
wrapper = subprocess.check_output(['git', '-C', repo, 'show', f'{commit}:scripts/native/orb_runner.cc'])
assert hashlib.sha256(wrapper).hexdigest() == prov['wrapper_sha256'] == d(out / 'runtime/orb_runner.cc')
receipt = dict(schema='bhl-stereo-recovery-atlas-wrapper-build-receipt-v1', status='PASS',
    job_id=os.environ['SLURM_JOB_ID'], host=os.uname().nodename, source_commit=full,
    input_runtime_archive_sha256=prov['original_runtime_archive_sha256'],
    input_runtime_chain=[prov['input_runtime_diagnostic_wrapper']['original_runtime_archive_sha256'], prov['original_runtime_archive_sha256']],
    runtime_archive=str(out / 'runtime.tar.gz'), runtime_archive_sha256=result['runtime_archive_sha256'],
    runtime_archive_bytes=(out / 'runtime.tar.gz').stat().st_size, binary_sha256=result['binary_sha256'],
    native_library_unchanged_sha256=prov['native_library_unchanged_sha256'], wrapper_sha256=prov['wrapper_sha256'],
    wrapper_version=prov['wrapper_version'], sif_sha256=d(sif), finished_utc=datetime.now(timezone.utc).isoformat())
(out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
PYEOF
