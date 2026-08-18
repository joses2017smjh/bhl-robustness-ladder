#!/bin/bash
# Serve the training curves for every run.
#
# Runs INSIDE the Apptainer image. TensorBoard's Rust data server is linked
# against GLIBC_2.29 and the login node ships 2.28, so running it natively there
# starts a web server that cannot actually read the event files:
#     tensorboard_data_server/bin/server: version `GLIBC_2.29' not found
# The container provides glibc 2.35, which makes this work on any host.
#
# Access, once it is serving on <host>:6006 --
#   from your laptop:   ssh -N -L 6006:localhost:6006 <user>@submit-b.hpc.engr.oregonstate.edu
#   in VS Code Remote:  Ports panel -> Forward a Port -> 6006
#   in an OnDemand desktop session: just open http://localhost:6006 in its browser
#
# Usage: ./scripts/tensorboard.sh [port]

set -euo pipefail
PORT=${1:-6006}

REPO=/nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder
source "$REPO/slurm/_env.sh"

LOGDIR=$UPSTREAM/logs/rsl_rl/biped
[ -d "$LOGDIR" ] || { echo "no runs yet at $LOGDIR" >&2; exit 1; }

echo "runs:"; ls -1 "$LOGDIR" | grep -v isaaclab | sed 's/^/  /'
echo
echo "serving on http://$(hostname -s):$PORT  (forward port $PORT to view)"

cat > "$WORKSPACE/.tb_inner.sh" <<INNER
#!/bin/bash
exec "\$UV_PROJECT_ENVIRONMENT/bin/tensorboard" \\
    --logdir "$LOGDIR" --port $PORT --bind_all --reload_multifile true
INNER
chmod +x "$WORKSPACE/.tb_inner.sh"

bhl_exec "$WORKSPACE/.tb_inner.sh"
