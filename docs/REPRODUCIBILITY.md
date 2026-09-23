# Reproduce the evidence

This is an HPC experiment repository, not a self-contained pretrained robot
package. An example command is not a clean-machine installation test.

## Public artifacts and prerequisites

The repository includes task overlays, reward repairs, sensor adapters, CPU
tests, Slurm entrypoints, per-evaluation JSON and selected recordings. Policy
weights, shared Python environments, Isaac installations, full garment assets
and raw frame caches are not vendored. Absolute checkpoint paths in results
identify local artifacts, not public download URLs.

## Validation performed on September 20

```bash
PYTHONPATH=src /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python -m pytest -q tests
python scripts/summarize_weekend.py --slurm
python scripts/build_portfolio_media.py
```

Initialize nested upstream assets with `git submodule update --init --recursive`
before running the full suite: two tests read the pinned upstream URDF and
actuator configuration directly. Without those files, the other 143 tests pass
but those two fail with missing-file errors.

The existing Python 3.11 environment passed 145 tests. Reporting requires
Slurm access. Media conversion requires ffmpeg with drawtext and fontconfig;
it refuses to overwrite its three GIFs and SHA-256 sidecars. Inspect the
committed artifacts without rebuilding. The wrong-route video is an
intentional negative control, not a failed training job or renderer crash.

| Layer | Requirements | Evidence |
|---|---|---|
| Logic/tests | Python, pytest, NumPy, PyTorch, MuJoCo and upstream imports used by tests | `tests/` |
| New maze PPO | Isaac Sim 6 / Isaac Lab 3 beta, compatible CUDA GPU, upstream robot | `slurm/inner/weekend_maze.sh`; 36 evaluated stages |
| Shared-world MuJoCo | MuJoCo, ONNX Runtime, OmegaConf, upstream controller and exported 22-DoF gait | `scripts/bench/inspection_maze.py`, `team_airlock.py` |
| Folding | Adjacent LeHome checkout/assets, SmolVLA environment, GPU PhysX, working Storm cameras | `docs/CLOTH_FOLDING_WEEKEND.md` |
| Hardware perception | Rectified calibrated cameras, original timestamps, detector weights, measured IMU mounting/units | `docs/STEREO_INPUT.md`, `docs/IMU_INPUT.md` |

For another cluster, adapt account/partitions and paths in `slurm/_env.sh`
and the batch scripts, provide compatible dependencies/assets, then pass an
integration gate before a sweep. Existing `00`, `01`, `02` entrypoints
describe the original v51 setup, not installation of the parallel v60 stack.

The new corridor score is an Isaac first-episode result with oracle waypoints,
fixed geometry and observation noise disabled. MuJoCo inspection/cooperation
clips use a different, older gait; they do not demonstrate transfer of the
new maze checkpoints. Fold rates require completed class evaluations and
valid cameras/particles. Partial files are progress, not zero-filled scores.
