# bhl-robustness-ladder

**Train a 3D-printed humanoid in Isaac Lab. Score it in a different simulator.**

An 11.3 kg Berkeley Humanoid Lite, 6 Nm joints. Upstream ships flat-ground
locomotion with no curriculum and no way to score a policy. This repo adds both,
pushes the robot until it stops learning, and reports where — including four
findings that retract earlier claims in this same repository.

Jose Sanchez · MS Artificial Intelligence, Oregon State ·
[portfolio](https://jose-sanchez-portfolio-com.vercel.app) ·
[findings](docs/FINDINGS.md) ·
[gallery](docs/GALLERY.md) ·
[sanchej7@oregonstate.edu](mailto:sanchej7@oregonstate.edu)

156 policies · 6,348 scored sim2sim episodes · 13 findings, 4 of them retractions

<p align="center">
  <img src="docs/gifs/multi_lab.gif" width="860" alt="Four differently-coloured policies walking one obstacle course in a single MuJoCo world, with the hero robot's egocentric depth image along the bottom."><br>
  <sub>Four policies, one world, one command — green randomized, red
  un-randomized, blue push-trained, orange terrain-trained. Same solver, same
  clock, not a composite. Along the bottom is the orange robot's own egocentric
  depth. When a policy goes down it darkens and the frame takes a red border.</sub>
</p>

<p align="center">
  <img src="docs/gifs/isaac/cloth_sort_free_base_shirt.gif" width="440" alt="A free-standing humanoid sweeping a red shirt proxy into a basket.">
  <img src="docs/gifs/carry_cube_pov.gif" width="440" alt="Two robots holding a cube with the robot's own colour and depth views alongside."><br>
  <sub>Left: the free-standing robot sorts a shirt proxy in two sweeps (6/8
  shirts, 22/24 rigid proxies, no falls). Right: the best cooperative cube
  rollout — 7.8 cm of lift, then the pair drops 41 cm before touching the cube.</sub>
</p>

---

## Problem, solution, result

| | |
|---|---|
| **Problem** | A policy that only works where it trained has learned PhysX, not locomotion. Upstream cannot tell you which. |
| **Solution** | Train in Isaac Lab (PhysX, 4,096 envs). Export ONNX. Score with upstream's own `RlController` in headless MuJoCo — bit-identical observations to the sim2real path. |
| **Contribution** | The measurement stack, not just the policies: gates that refuse a verdict without a control, and a ledger that keeps retractions in public. |
| **Result** | Transfer **inverts** the training-reward ranking. Highest training reward falls **23%** in MuJoCo; the repo-default randomization falls **0%**. |

Training and evaluation use different simulators **on purpose**.

---

## Case studies

Each row is a short visual argument. The numbers live in
[docs/FINDINGS.md](docs/FINDINGS.md); every clip is in
[docs/GALLERY.md](docs/GALLERY.md).

### 1 · Domain randomization vs transfer

<p align="center">
  <img src="docs/gifs/multi_race.gif" width="720" alt="Four policies taking identical shoves. Three stay up; the un-randomized robot is on the ground."><br>
  <sub>Identical 0.45 m/s shoves. The un-randomized robot is the one on the ground.</sub>
</p>

A single scale `s` rewrites every range in upstream's `EventsCfg`. `s = 0` wins
training by 49% and loses transfer outright. Fall rate has a knee; upstream's
shipped default sits on the last rung before it.

### 2 · Depth without a renderer

Isaac Sim 5.1's RTX renderer segfaults on this cluster. Geometric depth never
needed it: `RayCasterCamera` costs **1.6%** of throughput at 4,096 envs, 2.9%
mean error vs closed-form floor geometry. On rough terrain, looking ahead raises
the curriculum; a privileged height map of the ground underfoot does not.

A later "depth helps on invisible ice" result is **retracted**: the friction
patches spawned a median 72 m from the robots. 4.4% could reach one in an
episode. The +10.6% was §6's rough-terrain depth gain again.

### 3 · Arms on a real floor

No 12-DoF biped finishes the lab course upright; three of four fall on a
**2.5 cm cable** (9% of leg length). Two of four 22-DoF policies clear it. At
terrain `d = 1.0` the humanoid falls **11.7%** where the biped falls **37.8%**.
Arms buy recoverable perturbation, not a higher step — and one recipe is *worse*
with arms.

### 4 · Cloth sorting, after the first scene was unreachable

The garment sat 0.74 m away; fingertips reach 0.28 m forward; in Isaac the robot
faced the other way. Redesigned inside a measured IK table. The shipped hands
had no colliders. Bare 4 Nm PD lagged 65–72 mm; inverse-dynamics feedforward
through the same drive tracks 1.4–1.6 mm.

**The free-standing robot sorts 22 of 24 rigid proxies, no falls** (shirt 6/8,
sock 8/8, jacket 8/8). One 8×8 Newton shirt on a pinned base sorts 4/4 under
one-way coupling. Two-way coupling goes NaN when the hand pushes the cloth.

---

## Results at a glance

**Where it wins**

| | |
|---|---|
| Domain randomization | blind policy holds to terrain difficulty d≈0.4; randomized arms reach d≈0.6 |
| Low-res cloth | **467 env-steps/s** at 81 vertices vs G-C1's 182 at 961 — 2.6× while carrying 3× the DoF |
| Depth on low friction | final curriculum level **0.65 vs 0.26** blind — 3 seeds, no overlap |
| Ray-cast depth cost | **1.6%** of throughput at 4,096 envs, 2.9% mean error vs closed form |
| Restoring the grippers | mean episode length **6.3 → 427.7** steps, 6 cells a side |
| Free-standing sort | **22 of 24** rigid proxies, no falls |

**Where it loses**

| | |
|---|---|
| Cooperative lift | best rollout is 7.8 cm; the pair drops 41 cm before touching the cube |
| Plank task | 0.0 cm across 18 seeds — contact points exceed the shoulder span |
| Vision on the lift | depth-conditioned policies fall in almost every episode; blind ones do not |
| Task completion | **zero** success on all three redesigned two-robot tasks, gripper included |
| Stereo on terrain | **retracted** — cameras were written `(w, x, y, z)`, Isaac Lab 3.0 reads `(x, y, z, w)`. Pointed down, 4×4 stereo **1.29** vs blind 0.74 |
| Depth on invisible hazards | **retracted** — ice spawned 72 m away |
| Limb factorisation | **did not replicate** — +11% on the mean at n=3, not +47% |
| Isaac spawn | coop/TaskV2 robots still spawn under the floor. **MuJoCo-scored numbers stand** |

Four findings are retractions of earlier claims here. They stay in: a repo whose
argument is that the measurement was wrong cannot quietly fix its own
measurements.

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

What is running, and which Slurm id produced which number:
[SLURM_JOBS.md](SLURM_JOBS.md).

## Architecture

<p align="center">
  <img src="docs/img/pipeline.svg" width="100%" alt="Isaac Lab trains at 4096 envs; the checkpoint exports to ONNX; a headless MuJoCo harness scores it into CSV and MP4.">
</p>

| component | role |
|---|---|
| `src/bhl_robust/tasks/` | Isaac Lab env configs — terrain, push, depth, RGB, cooperative lift, cloth-sort |
| `src/bhl_robust/cloth/` | Isaac-free cloth-sort core: garments, sweep primitive, kinematic C0–C5, cost gate |
| `src/bhl_robust/eval/` | MuJoCo replay: MJCF patching, crew assembly, scoring, video |
| `src/bhl_robust/curricula/` | push and terrain-level curricula upstream lacks |
| `scripts/bench/` | gates. Each answers one question and refuses a verdict without a control |
| `slurm/` | job scripts; `inner/` holds what runs inside the container |

The MuJoCo harness drives the policy through upstream's own `RlController`, so
observation construction is bit-identical to the sim2real deployment path.

## Stack

- Isaac Lab 2.3.2 / Isaac Sim 5.1 (PPO, 4,096 envs) — and a parallel Isaac Lab 3.0 / Isaac Sim 6.0 stack for RGB, because 5.1's RTX renderer segfaults on this cluster
- MuJoCo 3.x — scoring, replay, video
- rsl-rl 3.0.1 (v51) / 5.0.1 (v60); skrl 1.4.3 for MAPPO and IPPO
- PyTorch 2.7, Warp, ONNX Runtime
- Apptainer, Slurm, `uv`

## License

Experiment code MIT. Upstream BHL retains its own license under `external/`.
