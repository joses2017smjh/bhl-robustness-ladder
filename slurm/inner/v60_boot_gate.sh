#!/bin/bash
# Sourced by v60 batch scripts, outside the container: v60_boot_gate before bhl_exec.
#
# Two Isaac Sim 6.0 processes that boot on one node in the same moment crash in
# SimulationApp._start_app (21300300/21300301, 2026-09-13): Kit's data directory
# lives in the shared venv and the startups contend for it. Chaining every v60
# job serially avoids that but turns nine six-hour runs into a 54-hour queue.
#
# So serialise the *boots* instead of the runs. mkdir is atomic on NFS; the
# holder keeps the lock for BOOT_WINDOW seconds (long enough for Kit to be past
# startup) and a background subshell releases it. A lock older than 15 minutes
# belongs to a job that died inside its window and is cleared.
v60_boot_gate() {
    local lock="/nfs/hpc/share/$USER/Humanoid_Lite/.v60-boot.lock"
    local window=${BOOT_WINDOW:-150} waited=0
    while ! mkdir "$lock" 2>/dev/null; do
        if [ -n "$(find "$lock" -maxdepth 0 -mmin +15 2>/dev/null)" ]; then
            echo "v60_boot_gate: clearing stale lock ($(cat "$lock/holder" 2>/dev/null))"
            rm -rf "$lock"
            continue
        fi
        sleep $(( 10 + RANDOM % 10 )); waited=$(( waited + 15 ))
        [ $waited -gt 3600 ] && { echo "v60_boot_gate: gave up after an hour, booting anyway"; return 0; }
    done
    echo "${SLURM_JOB_ID:-?} $(hostname) $(date +%s)" > "$lock/holder"
    echo "v60_boot_gate: acquired after ~${waited}s, holding ${window}s"
    ( sleep "$window"; rm -rf "$lock" ) &
}
