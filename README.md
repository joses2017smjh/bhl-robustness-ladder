# bhl-robustness-ladder

Robustness experiments on the [Berkeley Humanoid Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite)
(BHL) — an open-source, 3D-printed, **11.3 kg** humanoid whose joints are limited to **6 Nm**.
Those two numbers drive almost every result here: this machine has very little
authority to arrest a disturbance, so "how much can it take before it stops
learning?" is a real question rather than a formality.

Upstream ships flat-ground locomotion with a fixed domain-randomization preset,
no curriculum, and no way to score a policy. This repo adds three experiments
and — necessarily — the instrument to measure them.

**[▶ Explore every run interactively](https://claude.ai/code/artifact/de955af8-2236-4912-84fb-577e0a43ccbe)**
— isolate a run, switch metrics, watch the axis rescale.

| | Experiment | Question | Status |
|---|---|---|---|
| 1 | **Push recovery** | Can a disturbance curriculum buy shove-rejection, and how much? | **done** — 0.2 m/s is free; adaptive reaches 0.87 m/s |
| 2 | **DR fidelity ladder** | How does performance trade off against randomization strength? | **done** — 5-rung curve, 3 seeds each |
| 3 | **Rough terrain** | Does a terrain curriculum transfer to a robot this weak? | trained; curriculum reaches level 1.4 / 9 |

---

## Results so far

### The fidelity ladder

Randomization strength is expressed as a single scale `s`, where every range in
upstream's config is `center ± s·half_width`. `s = 1.0` is exactly what BHL
ships; `s = 0` pins every physics parameter to nominal.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/charts/dr_training_curves-dark.svg">
  <img alt="Training reward by DR rung. s=0 plateaus near 49, s=1 near 33, s=2 flat near 4." src="results/charts/dr_training_curves-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/charts/dr_ladder_summary-dark.svg">
  <img alt="Final reward falls and fall rate rises with randomization scale; the cliff is between s=1.0 and s=1.5." src="results/charts/dr_ladder_summary-light.svg">
</picture>

| rung | final reward (3 seeds) | fall rate | learns? |
|---|---|---|---|
| `s = 0.0` no randomization | 49.4 / 49.3 / 48.9 | 0.015 | yes |
| `s = 0.5` | 44.2 / 45.0 / 44.5 | 0.015 | yes |
| `s = 1.0` repo default | 32.9 / 34.8 / 33.2 | 0.041 | yes |
| `s = 1.5` | 16.0 / 15.7 / 16.5 | 0.211 | yes, degraded |
| `s = 2.0` double-width | 4.6 / 4.3 / 4.5 | 0.686 | **no** |

**The two panels tell different stories, and that is the result.** Reward
declines smoothly and almost linearly across the whole range — there is no
cliff in performance. *Stability* is the one with a knee: the fall rate is
essentially flat out to the repo default (0.015 → 0.015 → 0.041) and then turns
sharply upward (0.21 at `s = 1.5`, 0.69 at `s = 2.0`).

So the honest reading is that randomization buys robustness at a steady,
predictable cost in reward right up until `s ≈ 1.5`, past which the policy stops
reliably standing up at all. Upstream's shipped default sits comfortably inside
the stable region.

> **Correction:** an earlier version of this README claimed the cliff sat
> between `s = 1.0` and `s = 1.5`. That was read off partially-trained runs
> (~1,000 of 6,000 iterations), where `s = 1.5` was showing reward 8.3. Fully
> trained it reaches 16.0, and the decline is smooth. The conclusion changed
> once the runs finished.

> **Reading these numbers honestly:** training reward is *not* a performance
> ranking across rungs. A policy trained under heavier randomization is solving
> a strictly harder problem, so lower reward is expected and means nothing on
> its own. The real deliverable is **retention through sim2sim**, which the
> MuJoCo harness below measures.

### Push recovery — the sweep found the answer

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/charts/push_sweep-dark.svg">
  <img alt="Final reward falls with push ceiling: 33 at 0.2 m/s, 29 at 0.4, 22 at 0.6, 3 at 1.5." src="results/charts/push_sweep-light.svg">
</picture>

| ceiling | final reward | fall rate | cost vs no push |
|---|---|---|---|
| none (baseline) | 33.0 | 0.041 | — |
| 0.2 m/s | 33.0 / 34.4 | 0.037 | **free** |
| 0.4 m/s | 29.1 | 0.050 | −12% |
| 0.6 m/s | 22.1 / 23.0 | 0.100 | −32% |
| 1.5 m/s (pass 1) | 2.9 / 3.1 / 3.2 | 0.928 | destroyed |

Pass 1 asserted a ceiling of 1.5 m/s and destroyed the policy. The sweep shows
why that was such a bad guess: **0.2 m/s of shove rejection is essentially free**,
and the cost stays modest to 0.6 m/s. 1.5 m/s is not a hard setting, it is off
the end of the map.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/charts/push_collapse-dark.svg">
  <img alt="Push curriculum reward rises to 24.7 by iteration 1599 then collapses to about 3." src="results/charts/push_collapse-light.svg">
</picture>

The pass-1 trace above is kept because it localises the failure. The curriculum
arm learned to walk (reward 24.7 by iteration 1,599) and then the ramp destroyed
it, while the fixed-magnitude control never learned at all (~2.9 throughout).
The fixed arm failing confirms the curriculum was the right *idea*; the
curriculum arm peaking then collapsing shows the **ceiling**, not the schedule,
was what was wrong.

**The adaptive arm is the strongest result here.** Instead of a fixed schedule,
it raises the push only while the measured fall rate stays under 20% — the same
feedback principle Isaac Lab uses for terrain levels. It converged at
**0.85–0.88 m/s**, roughly *twice* the magnitude at which the naive time-based
ramp collapsed, and it reports that number as an outcome rather than requiring
it to be guessed. The caveat is that it ran into its own 1.0 m/s safety cap
(peak 1.000), so 0.87 is a lower bound on what this robot could tolerate, not a
measured ceiling.

### Rough terrain — it walks, but the curriculum stalls low

| arm | final reward | fall rate | terrain level reached (of 9) |
|---|---|---|---|
| bumpy (noise + slopes + obstacles) | 18.6 / 18.6 / 18.2 | 0.10 | **1.44 / 1.45 / 1.42** |
| smooth (no obstacles) | 17.1 / 17.7 | 0.12 | 1.21 / 1.28 |

The curriculum machinery works — environments are promoted and demoted, and the
level metric moves — but it plateaus at roughly **level 1.4 of 9**. The robot
learns to walk on mildly rough ground and then stops progressing. Given 6 Nm
joints and no height sensing, that is a believable ceiling rather than a bug,
and the level it stalls at is the measurement.

Two honest caveats:

- The obstacles did **not** turn out to be the limiter. The arm *with* obstacles
  reaches a slightly higher level than the arm without, which is the opposite of
  what I expected when I flagged obstacles as the risky choice.
- **That ablation is confounded, and I would not report it as clean.** Removing
  the obstacles redistributed their 20% share into more rough ground and slope
  (rough 0.40 → 0.50, each slope 0.20 → 0.25), so "smooth" is not "bumpy minus
  obstacles" — it is also *rougher on average*. A correct ablation holds the
  other proportions fixed and replaces the obstacle share with flat ground. The
  two smooth seeds also stopped slightly short of 6,000 iterations.

### Terrain transfer — the baseline to beat

Evaluating a terrain policy on flat ground would prove nothing, so the MuJoCo
side got its own height fields. Training terrain (Isaac Lab) and evaluation
terrain (MuJoCo `hfield`) are generated by **completely different code** — that
is deliberate. A policy that only copes with the exact surface it trained on has
memorised, not learned. What the two share is the *physical envelope*: amplitude,
gradient and obstacle height are matched to the training sub-terrain menu, so
`d = 1.0` means roughly the same thing on both sides (0.33 m peak-to-trough,
p95 gradient 12.2° against a 15° training target).

The height sampler was checked against MuJoCo's own geometry by raycast and
agrees to **under 1 mm**, so the spawn height and fall detection are registered
to the surface the solver actually uses.

Here is the flat-trained, blind `dr-off` policy scored across that axis — the
baseline any terrain policy has to beat:

| difficulty | fall rate | survival | distance |
|---|---|---|---|
| 0.00 (flat) | 8% | 7.5 s | 1.28 m |
| 0.05 | 58% | 4.6 s | 0.54 m |
| 0.10 | 92% | 3.4 s | 0.43 m |
| 0.20 | 100% | 2.4 s | 0.31 m |
| 0.60 | 100% | 1.1 s | 0.16 m |

It collapses between `d = 0.05` and `d = 0.10` — roughly 2 cm of unexpected
relief. That is not surprising for a **blind** policy (BHL has no height scanner,
only proprioception), and it is exactly why the difficulty grid is sampled
densely at the low end rather than uniformly to 1.0.

---

## Methodology

### Why this needed its own instrument

The obvious plan — "evaluate through the sim2sim path the repo already gives you"
— does not survive contact with the code. Upstream's `scripts/sim2sim/play_mujoco.py`:

- constructs a `Se2Gamepad()` and blocks on joystick input,
- calls `mj_viewer.sync()`, requiring a GUI,
- `sleep()`s to hold wall-clock realtime,
- loops forever and emits **no metric of any kind**.

It is a thing to watch, not a thing to measure. So `src/bhl_robust/eval/` keeps
the parts that determine transfer fidelity — the MJCF, the PD controller, the
sensor-derived observation layout — and drops the interactive scaffolding.

**The single most important design decision here:** the harness drives the policy
through upstream's *own* `RlController`. Observation construction is therefore
bit-identical to the sim2real deployment path. An evaluator that rebuilt the
45-dim observation vector independently would be measuring my reimplementation,
not the transfer.

### What gets measured

Every policy is scored on the identical protocol — 6 commanded velocities ×
5 seeds × 10 s episodes:

| metric | why |
|---|---|
| fall / survival time | fall threshold mirrors training's `bad_orientation` (0.78 rad), so a fall means the same thing it meant during training. On rough ground, sink is measured against the **local** surface — an absolute reference would score walking downhill as falling |
| linear velocity error | computed in the **yaw frame**, because that is the frame upstream rewards (`track_lin_vel_xy_yaw_frame_exp`); any other frame is not comparable |
| yaw-rate error, distance, height, tilt | gait quality beyond mere survival |
| push survival | fraction of shoves the policy stays upright through |

Tracking errors average over **surviving steps only** — otherwise a policy that
falls instantly posts a flatteringly small error.

### Pipeline

```mermaid
flowchart LR
  A["Isaac Lab<br/>4096 envs · PPO"] -->|"6,000 iters"| B["checkpoint"]
  B -->|"vendored play.py"| C["policy.onnx<br/>+ deploy.yaml"]
  C --> D["MuJoCo harness<br/>headless · scripted"]
  D --> E["per-episode CSV"]
  D --> F["MP4 clips"]
  E --> G["curve + ladder charts"]
  F --> H["success / failure reels"]
```

Training and evaluation deliberately use **different simulators** — Isaac Lab
(PhysX) to train, MuJoCo to score. That gap *is* the measurement: a policy that
only works in the simulator it was trained in has not learned locomotion, it has
learned PhysX.

### Experiment design notes

**DR ladder.** Rungs are a continuous scale rather than three named presets,
which is what turns three points into a curve and lets the cliff be *located*
rather than merely hit. `s = 0/1/2` reproduce the originally-trained rungs
bit-for-bit. One subtlety: mass is an *additive offset*, so it scales from zero,
while friction and gain multipliers scale around their centre — scaling mass
around its centre would silently inject a constant +0.5 kg at `s = 0` and break
agreement with the policies already trained.

**Push curriculum.** Implemented as a real Isaac Lab curriculum term that mutates
the live event config each iteration, logging the current magnitude so the ramp
is visible in TensorBoard. It binds to `curriculum` — **not** upstream's
`curriculums`. Isaac Lab builds its manager from `cfg.curriculum` (singular), so
upstream's `CurriculumsCfg` is never read; binding to the plural name would have
silently no-opped the entire experiment while still training and reporting a
plausible-looking number.

**Initial-state randomization is held constant** across every rung, so the ladder
isolates one variable.

### Reproducibility

The whole stack is pinned. Upstream is a submodule at `984741a`; the Python
environment resolves from upstream's `uv.lock` with `--locked`, so all runs share
a bit-identical dependency set. Every run also dumps its own fully-resolved
`params/env.yaml`, which is how the DR overrides were *verified* to land rather
than assumed.

<details>
<summary><b>Why a container, and why the login node can't run any of this</b></summary>

The cluster login node is Rocky 8 (**glibc 2.28**); Isaac Sim 5.1 requires
**≥ 2.34**. Conda cannot fix that — it ships its own `libstdc++`, not glibc.
Compute nodes are Rocky 9.8 (glibc 2.34), *exactly* at the boundary, so
`container/bhl.def` builds an Ubuntu 22.04 (glibc 2.35) base and sidesteps the
question. The NVIDIA driver and Vulkan ICD are injected at runtime via
`apptainer --nv`.

The image contains **no Python packages** — `uv` resolves the stack into a venv
outside the git tree, so the repo stays movable without a 30-minute resync.

Cluster-specific gotchas that cost real debugging time, documented in commits:

- `apptainer --cleanenv` drops every unforwarded variable — this killed a 9-job
  array in one second on `TASK: unbound variable`.
- `apptainer --env` splits values on **commas**, mangling every `[lo,hi]` range,
  so Hydra overrides are passed by file path.
- `/scratch` is **node-local, not shared**.
- TensorBoard's data server needs `GLIBC_2.29`, so on the login node it serves a
  page while silently reading nothing. It runs inside the container.
- numpy's C extension will not import on the login node at all.
</details>

---

## What was found in upstream

Working through this surfaced five genuine breakages, all worked around without
modifying `external/`:

| | issue | consequence |
|---|---|---|
| 1 | `CurriculumsCfg` bound to `curriculums`, Isaac Lab reads `curriculum` | every curriculum term is dead code, incl. their unused `terrain_levels_vel` |
| 2 | MJCF declares `meshdir="assets"` + `merged/*.stl`; meshes live flat in `meshes/` | **the MuJoCo model does not load at all** |
| 3 | `play.py` uses `OnPolicyRunner.obs_normalizer`, removed in rsl-rl 3.x | ONNX export raises before writing anything |
| 4 | `play.py` playback loop assumes old `env.get_observations()` arity | upstream's render path is unusable |
| 5 | `--experiment_name` is parsed but never applied | all runs land in one directory |

(2) is the interesting one: the meshes are **visual-only** (`contype="0"`; every
collision geom is a primitive), so physics is unaffected — but the model still
won't load, so it has to be repaired regardless.

---

## Repo layout

```
external/Berkeley-Humanoid-Lite   upstream, pinned at 984741a (submodule, unmodified)
src/bhl_robust/
  tasks/        overlay env configs registering new gym task ids
  curricula/    push-magnitude ramp
  eval/         headless MuJoCo harness, MJCF repair, video renderer
scripts/        vendored train/play entrypoints, chart + curve generation
slurm/          sbatch scripts (OSU COE HPC, `gpu` partition)
docs/           the interactive run explorer
results/        curves, charts, per-episode CSVs
```

## Running it

```bash
git clone --recurse-submodules https://github.com/joses2017smjh/bhl-robustness-ladder.git
cd bhl-robustness-ladder
sbatch slurm/00_build_container.sbatch   # ~4 min
sbatch slurm/01_uv_sync.sbatch           # ~30-60 min, ~30 GB
sbatch slurm/02_smoke_train.sbatch       # gate check

sbatch slurm/10_dr_ladder.sbatch         # DR rungs s=0,1,2
sbatch slurm/12_dr_ladder_fill.sbatch    # s=0.5, 1.5
sbatch slurm/13_push_sweep.sbatch        # push ceilings 0.2/0.4/0.6
sbatch slurm/20_evaluate_all.sbatch      # export -> score -> render

sbatch slurm/90_tensorboard.sbatch       # live curves
```

## Hardware

OSU COE HPC, `gpu` partition: Quadro RTX 8000 (48 GB, driver 595.71.05),
Rocky 9.8. One GPU per run; 6,000 PPO iterations at 4,096 envs takes ≈3 h.

## License

Experiment code MIT. Upstream BHL retains its own license under `external/`.
