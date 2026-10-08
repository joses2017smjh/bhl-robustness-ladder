# bhl-robustness-ladder

**Robot-learning experiments with checks that distinguish task success from a job that merely ran.**

Train Berkeley Humanoid Lite policies in Isaac Lab, evaluate selected gaits in
MuJoCo, and record checkpoints, task scores, negative controls and failures.
My contribution is the infrastructure around the upstream robot: curricula,
sensor adapters, shared-world missions and Slurm evaluation gates.

Jose Sanchez · MS Artificial Intelligence, Oregon State ·
[portfolio](https://jose-sanchez-portfolio-com.vercel.app) ·
[findings](docs/FINDINGS.md) ·
[gallery](docs/GALLERY.md) ·
[Email](mailto:josejsanchez20172@gmail.com)

Seeking ML / AI and robotics engineering roles. Python · PyTorch · Isaac
Lab · MuJoCo · ONNX · Slurm/HPC.

[Results, September 20](docs/WEEKEND_RESULTS_2026-09-20.md) ·
[Reproduce and inspect](docs/REPRODUCIBILITY.md) ·
[Architecture](#architecture) · [Public evidence](docs/PUBLIC_EVIDENCE.md)

<p align="center">
  <a href="results/weekend-20260919/inspection-maze.mp4"><img src="docs/gifs/weekend-inspection.gif" width="720" alt="MuJoCo humanoid completes two ordered inspection stops and exits a two-turn maze. Caption identifies learned gait, oracle waypoints and sensor braking."></a><br>
  <sub>Actual MuJoCo rollout, 17.36 s. Frozen Isaac-trained gait, known map/pose,
  and lidar/depth braking. Stations are proximity dwells, not SSD recognition.
  <a href="docs/gifs/weekend-inspection-failure.gif">Wrong-branch control</a> ·
  <a href="docs/gifs/weekend-team3.gif">Three-robot mission</a>.</sub>
</p>

<p align="center">
  <sub>The new maze PPO checkpoints below are evaluated in Isaac, not the
  checkpoints driving this MuJoCo video. The video uses the older August 18
  full-body gait. <a href="docs/GALLERY.md">Success and failure gallery</a>.</sub>
</p>

## Latest measured results

| Experiment | Evidence | Boundary |
|---|---|---|
| Repaired goal-reaching PPO | **379/384** final-stage first episodes; 4 sensor conditions × 3 seeds × 32 envs; all 36 curriculum jobs completed | 12-DoF biped, fixed corridor, oracle waypoints, observation noise disabled; not a sensor-benefit claim |
| Two-turn inspection mission | **3/3** nominal; **0/3** wrong-branch and **0/3** complete-outage controls | 22-DoF MuJoCo humanoid; known map/pose and frozen gait |
| Two-/three-robot airlock | **5/5** each; both negative controls **0/5** each | One physical world, explicit synchronization; no carrying or newly trained MARL |
| Dual-arm cloth folding | Baseline short pants **8/24**, adapted seed 1 **3/24**; only **5/12** evaluation cells completed | No demonstrated adaptation improvement; remaining cells failed/timed out; not humanoid folding |

[Public result report](results/weekend-20260919/SUMMARY.md) ·
[Protocols and limits](docs/WEEKEND_CAMPAIGN.md) ·
[Cloth diagnosis](docs/CLOTH_FOLDING_WEEKEND.md) ·
[Folding success, failure and arm-camera GIFs](docs/FOLDING_MEDIA.md).

The folding gallery pairs an **earlier learned-policy success and failure**
with the **new adapted policy’s failure**, including both actual wrist cameras.
The historical success is not evidence that the new adaptation improved.

---

## Problem, solution, result

| | |
|---|---|
| **Problem** | High training reward can hide task failure or poor transfer to another physics engine. |
| **Solution** | Train in Isaac Lab (PhysX, 4,096 envs). Export ONNX. Score with upstream's own `RlController` in headless MuJoCo — bit-identical observations to the sim2real path. |
| **Contribution** | Versioned tasks, checkpoint-gated curricula, sensor validity checks, physics-step contact scoring and an auditable job ledger. |
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
| Earlier manipulation tasks | **zero** success on the three tested two-robot object tasks, gripper included; separate from successful airlock navigation |
| Stereo on terrain | **retracted** — cameras were written `(w, x, y, z)`, Isaac Lab 3.0 reads `(x, y, z, w)`. Pointed down, 4×4 stereo **1.29** vs blind 0.74 |
| Depth on invisible hazards | **retracted** — ice spawned 72 m away |
| Limb factorisation | **did not replicate** — +11% on the mean at n=3, not +47% |
| Historical Isaac spawn | Earlier coop/TaskV2 runs spawned under the floor; quaternion conversion is now corrected. This does not retroactively validate those runs. |

Four findings are retractions of earlier claims here. They stay in: a repo whose
argument is that the measurement was wrong cannot quietly fix its own
measurements.

---

## Quickstart

Isaac Sim needs a GPU and ~30 GB. These batch scripts contain **Oregon
State-specific paths, account and partition settings**; adapt them before
submission elsewhere. They describe the original v51 setup, not the parallel
v60 campaign environment. See [requirements](docs/REPRODUCIBILITY.md).

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

The [dated results](docs/WEEKEND_RESULTS_2026-09-20.md) and
[public evidence guide](docs/PUBLIC_EVIDENCE.md) explain the recorded outcomes
and external artifacts. The historical job ledger remains available for older runs.

## Testing and engineering decisions

Run the CPU suite in an environment with the documented dependencies and
pinned upstream asset fixtures. The September 20 publication passed **146 tests**:

```bash
PYTHONPATH=src python -m pytest -q tests
```

Unit tests do not certify Isaac rendering or learned task performance. Those
require separate GPU/physics gates and their saved JSON results.

- Keep legacy task IDs unchanged; corrected maze tasks are versioned.
- Promote only the exact checkpoint that passed a separate task evaluation.
- Use ray depth for cheap geometric baselines; real stereo matching remains a
  separate calibrated component, not a claim of deployed perception.
- Record negative controls and invalid observations. Training loss, surviving
  to timeout and Slurm `COMPLETED` are not task-success metrics.
- Use RTX/A40 hardware for the tested rendering path, H100/V100 for compatible
  learning workloads, and CPUs for MuJoCo scoring.

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

No repository-level license file is currently provided. Upstream BHL retains
its own license under `external/`; external assets and models retain theirs.
