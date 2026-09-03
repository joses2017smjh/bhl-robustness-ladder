# bhl-robustness-ladder

Robustness benchmark for a 3D-printed humanoid: train in Isaac Lab, score in MuJoCo.

Upstream ships flat-ground locomotion with no curriculum and no way to score a
policy. This adds both, then pushes the robot until it stops learning and
reports where. 156 policies, 6,348 scored sim2sim episodes, 13 findings — four
of which retract earlier findings in this same repo.

<p align="center">
  <img src="docs/gifs/multi_race.gif" width="860" alt="Four policies in one MuJoCo world taking identical shoves; three stay up, the un-randomized one falls."><br>
  <sub>Four policies, one world, identical 0.45 m/s shoves. Same solver, same
  clock — not a composite. The un-randomized robot is the one on the ground.</sub>
</p>

<p align="center">
  <img src="docs/gifs/carry_cube_pov.gif" width="500" alt="Two robots holding a cube with the robot's own colour and depth views alongside."><br>
  <sub>Two robots, one cube, with the acting robot's head camera on the right:
  colour, raw 64×64 depth, and the 8×8 the network actually receives.</sub>
</p>

---

## Quickstart

Isaac Sim needs a GPU and ~30 GB. The build runs under Slurm because that is
where the GPUs are; each `sbatch` is a single self-contained step.

```bash
git clone --recurse-submodules https://github.com/joses2017smjh/bhl-robustness-ladder.git
cd bhl-robustness-ladder
sbatch slurm/00_build_container.sbatch    # apptainer image, ~4 min
sbatch slurm/01_uv_sync.sbatch            # Isaac Lab 2.3.2 + deps, ~45 min, ~30 GB
sbatch slurm/02_smoke_train.sbatch        # 3 iterations; fails loudly if the stack is broken
```

`02` is the gate. It asserts a training iteration was logged **and** that mean
episode length exceeds 2 — a task where every episode ends on its first step
reports healthy iteration counts and learns nothing, which cost nine GPU-days
here before the check existed.

Scoring a trained policy needs no GPU:

```bash
python scripts/bench/coop_sim2sim.py --run-dir <run> --upstream external/Berkeley-Humanoid-Lite \
    --cache-dir /tmp/mjcf --seeds 8 --crews 2 3 4
```

## Architecture

<p align="center">
  <img src="docs/img/pipeline.svg" width="100%" alt="Isaac Lab trains at 4096 envs; the checkpoint exports to ONNX; a headless MuJoCo harness scores it into CSV and MP4.">
</p>

Training and evaluation use **different simulators on purpose**. A policy that
only works where it trained has learned PhysX, not locomotion.

| component | role |
|---|---|
| `src/bhl_robust/tasks/` | Isaac Lab env configs — terrain, push, depth, RGB, cooperative lift |
| `src/bhl_robust/eval/` | MuJoCo replay: MJCF patching, crew assembly, scoring, video |
| `src/bhl_robust/curricula/` | push and terrain-level curricula upstream lacks |
| `scripts/bench/` | gates. Each answers one question and refuses a verdict without a control |
| `slurm/` | 75 job scripts; `inner/` holds what runs inside the container |

The MuJoCo harness drives the policy through upstream's own `RlController`, so
observation construction is bit-identical to the sim2real deployment path.

## Results

Full table and evidence in [docs/FINDINGS.md](docs/FINDINGS.md).

**Where it wins**

| | |
|---|---|
| Domain randomization | blind policy holds to terrain difficulty d≈0.4; randomized arms reach d≈0.6 |
| Depth on low friction | final curriculum level **0.65 vs 0.26** blind — 3 seeds, no overlap |
| Depth on invisible hazards | **+10.6%** on friction patches flush with the floor, ray-cast-verified |
| Ray-cast depth cost | **1.6%** of throughput at 4,096 envs, 2.9% mean error vs closed form |
| Restoring the grippers | mean episode length **6.3 → 427.7** steps, 6 cells a side |
| Limb factorisation | arms/legs split finishes **47% above** a no-split control |

**Where it loses**

| | |
|---|---|
| Cooperative lift | best rollout is 7.8 cm, and the pair drops 41 cm before touching the cube |
| Plank task | 0.0 cm across 12 seeds — contact points exceed the shoulder span |
| Vision on the lift | depth-conditioned policies fall in almost every episode; blind ones do not |
| Task completion | **zero** success on all three redesigned tasks, gripper included |
| Cloth throughput | eight probe attempts, still no number |
| Seed counts | findings 10 and 12 rest on 1–2 seeds. §1 is a single-seed result that died on its third |

Four findings are retractions of earlier claims here. They stay in: a repo whose
argument is that the measurement was wrong cannot quietly fix its own
measurements.

## Stack

- Isaac Lab 2.3.2 / Isaac Sim 5.1 (PPO, 4,096 envs) — and a parallel Isaac Lab 3.0 / Isaac Sim 6.0 stack for RGB, because 5.1's RTX renderer segfaults on this cluster
- MuJoCo 3.x — scoring, replay, video
- rsl-rl 3.0.1 (v51) / 5.0.1 (v60); skrl 1.4.3 for MAPPO and IPPO
- PyTorch 2.7, Warp, ONNX Runtime
- Apptainer, Slurm, `uv`
