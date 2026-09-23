#!/bin/bash
# SF-04 (docs/SENSOR_FUSION.md): train the BothRobust maze-recovery arm, seed 0,
# through the weekend pipeline (smoke -> train -> gate) with a recurrent actor.
# Usage (inside bhl_exec): inner_sf04_maze.sh <Stage> <iterations> [<PreviousStage>]
set -euo pipefail
stage=${1:?Approach Corridor Full}; iterations=${2:?iterations}; previous=${3:-}
export BHL_POLICY=recurrent          # train.py overlay: ActorCriticRecurrent (lstm, 256)
exec bash "$REPO/slurm/inner/weekend_maze.sh" train "$stage" BothRobust 0 1024 "$iterations" "$previous"
