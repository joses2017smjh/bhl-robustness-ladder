#!/bin/bash
# Sourced by v60 batch scripts, outside the container: v60_boot_gate before bhl_exec.
#
# Two Isaac Sim 6.0 processes that boot on one node in the same moment crash in
# SimulationApp._start_app (21300300/21300301, 2026-09-13): Kit's data directory
# lives in the shared venv and the startups contend for it. Chaining every v60
# job serially avoids that but turns nine six-hour runs into a 54-hour queue.
#
# So serialise the *boots* instead of the runs. mkdir is atomic on NFS. The
# holder keeps the lock for BOOT_WINDOW seconds, long enough for Kit to be past
# startup, then releases it -- from a background subshell for a long job, and
# from an EXIT trap for a job that ends inside the window, because Slurm kills
# the background subshell with the job. The first version had no trap: a 54 s
# probe (21317022) left its lock behind and stalled every v60 job for the
# 15-minute stale rule.
#
# Release only a lock this job still holds. A delayed unconditional rm would
# delete a lock another job took after this one's was freed. A lock older than
# five minutes is past any window and belongs to a job that died without its
# trap running (SIGKILL); it is cleared.
_v60_release() {
    [ "$(cut -d' ' -f1 "$1/holder" 2>/dev/null)" = "${SLURM_JOB_ID:-?}" ] && rm -rf "$1"
    return 0
}

v60_boot_gate() {
    local lock="${V60_BOOT_LOCK:-/nfs/hpc/share/$USER/Humanoid_Lite/.v60-boot.lock}"
    local window=${BOOT_WINDOW:-150} waited=0
    while ! mkdir "$lock" 2>/dev/null; do
        if [ -n "$(find "$lock" -maxdepth 0 -mmin +5 2>/dev/null)" ]; then
            echo "v60_boot_gate: clearing stale lock ($(cat "$lock/holder" 2>/dev/null))"
            rm -rf "$lock"
            continue
        fi
        sleep $(( 10 + RANDOM % 10 )); waited=$(( waited + 15 ))
        [ $waited -gt 3600 ] && { echo "v60_boot_gate: gave up after an hour, booting anyway"; return 0; }
    done
    echo "${SLURM_JOB_ID:-?} $(hostname) $(date +%s)" > "$lock/holder"
    echo "v60_boot_gate: acquired after ~${waited}s, holding ${window}s"
    ( sleep "$window"; _v60_release "$lock" ) &
    trap "_v60_release '$lock'" EXIT
}
