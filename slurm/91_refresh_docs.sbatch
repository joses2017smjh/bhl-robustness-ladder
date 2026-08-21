#!/bin/bash
#SBATCH --job-name=refresh
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Re-derive every published artifact from the event files: tidy curves ->
# curves.json -> README charts -> interactive explorer payload.
#
# On `share`, not `gpu`: this is pure CPU and the gpu partition's 8-GPU cap is
# the binding constraint on training throughput.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
cd "$REPO"

cat > "$WORKSPACE/.refresh_inner.sh" <<'INNER'
#!/bin/bash
set -euo pipefail
cd "$REPO"
"$PY" scripts/extract_curves.py "$UPSTREAM/logs/rsl_rl/biped" "$UPSTREAM/logs/rsl_rl/humanoid" "$UPSTREAM/logs/rsl_rl/coop_lift" results/curves/training_curves.csv
INNER
chmod +x "$WORKSPACE/.refresh_inner.sh"
bhl_exec "$WORKSPACE/.refresh_inner.sh"

# Downsample + summarise, then redraw. Both are plain-stdlib Python and run
# outside the container.
python3 scripts/summarise_curves.py
python3 scripts/make_charts.py
python3 scripts/build_dashboard.py

echo "=== refreshed $(date) ==="
ls -1 results/charts/*.svg | wc -l | sed 's/^/charts: /'
