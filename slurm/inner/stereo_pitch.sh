#!/bin/bash
# Gate for the corrected maze stereo re-runs: the verdict line is the evidence,
# because Isaac can exit 0 after a crash.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUT="$REPO/results/stereo_pitch_probe.txt"
"$PY" scripts/bench/stereo_pitch_probe.py --num_envs "${NUM_ENVS:-8}" --enable_cameras 2>&1 | tee "$OUT.log"
grep -E "^quat order probed|^  stereo_|^STEREO-PITCH" "$OUT.log" > "$OUT" || true
cat "$OUT"
grep -q "^STEREO-PITCH PASS" "$OUT" || { echo "stereo_pitch.sh: no PASS verdict" >&2; exit 1; }
