# bhl-robustness-ladder

Robustness experiments on the [Berkeley Humanoid Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite)
(BHL) — a small, 3D-printed, torque-limited 22-DoF humanoid.

Upstream BHL ships flat-ground locomotion with a fixed domain-randomization
preset and no curriculum. This repo adds three things it does not have, and
measures each one:

| # | Experiment | Question |
|---|---|---|
| 1 | **Push recovery** | Can a disturbance curriculum buy shove-rejection the baseline doesn't have, and how much? |
| 2 | **DR fidelity ladder** | How does sim2sim performance retention trade off against randomization strength? |
| 3 | **Rough terrain + curriculum** | Does Isaac Lab's terrain-level curriculum transfer to a robot with this little torque? |

Experiment 2 is the headline result: it produces a curve, not a demo.

## Repo layout

```
external/Berkeley-Humanoid-Lite   upstream, pinned at 984741a (submodule)
src/bhl_robust/
  tasks/        env configs registering the new gym tasks
  curricula/    push-magnitude ramp, terrain-level promotion
  terrains/     rough / slope / obstacle generators (no stairs)
  eval/         headless MuJoCo sim2sim evaluation harness
slurm/          sbatch scripts (OSU COE HPC, `gpu` partition)
container/      Apptainer definition for the Isaac Sim runtime
results/        aggregated metrics + plots
docs/           setup notes and findings
```

Nothing in `external/` is modified. All additions are overlays registered as
new Isaac Lab tasks, so upstream stays cleanly re-pinnable.

## Why a container

The cluster login node is Rocky 8 (**glibc 2.28**); Isaac Sim 5.1 requires
**glibc ≥ 2.34**. Conda cannot fix this — it ships its own `libstdc++`, not
glibc. Compute nodes are Rocky 9.8 (glibc 2.34), which is *exactly* at the
boundary, so `container/bhl.def` builds an Ubuntu 22.04 (glibc 2.35) base and
sidesteps the question entirely. The NVIDIA driver and Vulkan ICD are injected
from the host at runtime via `apptainer --nv`.

The image contains no Python packages. `uv` resolves the full stack from
upstream's `uv.lock` into a venv outside the git tree.

## Verified environment

| | |
|---|---|
| Cluster | OSU COE HPC, `gpu` partition (`eecs` account) |
| Node | `cn-gpu6` / `cn-gpu7` — 8× Quadro RTX 8000 48GB, compute 7.5 |
| Driver | 595.71.05 |
| Isaac Sim / Isaac Lab | 5.1.0 / 2.3.2.post1 |
| Python / torch | 3.11 / 2.7.0 (cu128) |

## Setup

```bash
git clone --recurse-submodules https://github.com/joses2017smjh/bhl-robustness-ladder.git
cd bhl-robustness-ladder
sbatch slurm/00_build_container.sbatch   # ~4 min
sbatch slurm/01_uv_sync.sbatch           # ~30-60 min, ~30GB
```

## Notes on upstream

Two upstream issues this project works around, documented in `docs/`:

- `scripts/sim2sim/play_mujoco.py` is an interactive demo, not an evaluator:
  it requires a gamepad, opens a GUI viewer, throttles to wall-clock realtime,
  and loops forever without emitting metrics. Experiment 2 needs a headless
  scripted harness, which lives in `src/bhl_robust/eval/`.
- The MJCF path hardcoded in `environments/mujoco.py` does not match the
  asset submodule's actual layout.

## License

Experiment code: MIT. Upstream BHL retains its own license under `external/`.
