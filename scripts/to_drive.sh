#!/bin/bash
# Copy the irreplaceable parts of this cluster's work to Google Drive.
#
# Resumable: rclone copy skips what already matches, so rerunning after a
# dropped connection costs a listing pass and nothing else. It is `copy`, never
# `sync` -- sync deletes on the destination to make it match the source, and a
# migration script should not be able to delete anything.
#
# Setup is in docs/DRIVE_SETUP.md; this assumes a configured remote named
# `gdrive`.
set -uo pipefail

RCLONE="${RCLONE:-$HOME/bin/rclone}"
REMOTE="${REMOTE:-gdrive}"
DEST="${DEST:-hpc-archive/Humanoid_Lite}"
BASE="/nfs/hpc/share/$USER/Humanoid_Lite"

command -v "$RCLONE" >/dev/null || { echo "no rclone at $RCLONE" >&2; exit 1; }
"$RCLONE" lsd "$REMOTE:" >/dev/null 2>&1 || {
    echo "remote '$REMOTE' is not configured or not reachable." >&2
    echo "See docs/DRIVE_SETUP.md -- step 1 needs a browser and is not automatable." >&2
    exit 1
}

# Drive rate-limits on requests, not bytes, so the tuning that matters is API
# calls per second rather than parallelism. --tpslimit keeps it under the 403
# "userRateLimitExceeded" threshold on a tree with thousands of small files.
FLAGS=(
    --transfers 4 --checkers 8 --tpslimit 10
    --drive-chunk-size 128M --fast-list
    --retries 5 --low-level-retries 20
    --stats 60s --stats-one-line --progress
)

# name : source : whether to send at all
copy_one() {   # label source
    local label=$1 src=$2
    if [ ! -e "$src" ]; then echo "  skip $label: $src not found"; return; fi
    echo
    echo "=== $label -> $REMOTE:$DEST/$label ==="
    du -sh "$src" 2>/dev/null | sed 's/^/    source: /'
    "$RCLONE" copy "$src" "$REMOTE:$DEST/$label" "${FLAGS[@]}" \
        --log-file "/tmp/rclone-$label.log" --log-level INFO
    echo "    rclone exit: $?"
}

# The venvs and the container are deliberately absent. They are reproducible
# from slurm/00 and slurm/01, and they are pinned to this cluster's driver --
# Isaac Sim 5.1's RTX plugins against driver 595/610 is the subject of a finding
# in this repo. Restoring them elsewhere gives you something that does not run.
copy_one training-logs "$BASE/bhl-robustness-ladder/external/Berkeley-Humanoid-Lite/logs"
copy_one slurm-logs    "$BASE/logs"
copy_one lehome-data   "$BASE/lehome-data"

echo
echo "############ verifying ############"
rc=0
for label in training-logs slurm-logs lehome-data; do
    case $label in
      training-logs) src="$BASE/bhl-robustness-ladder/external/Berkeley-Humanoid-Lite/logs" ;;
      slurm-logs)    src="$BASE/logs" ;;
      lehome-data)   src="$BASE/lehome-data" ;;
    esac
    [ -e "$src" ] || continue
    echo "--- $label ---"
    # `check` compares sizes and hashes both ways and is the only thing that
    # makes a "done" message mean anything.
    "$RCLONE" check "$src" "$REMOTE:$DEST/$label" --one-way --fast-list \
        --tpslimit 10 2>&1 | tail -4
    [ ${PIPESTATUS[0]} -ne 0 ] && rc=1
done

echo
if [ $rc -eq 0 ]; then
    echo "ALL VERIFIED -- every source file has a matching copy on $REMOTE:$DEST"
else
    echo "MISMATCHES FOUND. Rerun this script; it resumes. Do not delete anything"
    echo "from the cluster until this line reads ALL VERIFIED." >&2
fi
exit $rc
