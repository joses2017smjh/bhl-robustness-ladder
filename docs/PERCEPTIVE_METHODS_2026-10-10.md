# Perceptive terrain learning: implemented experiment

This adds actual Isaac Lab PPO training for a privileged terrain teacher and four measured-input students. It does not replace the previously published negative terrain qualification result. The new training cohort is development work in the 12-DoF Berkeley Humanoid Lite simulation.

All three development seeds completed. Strict collection verified **768 first-episode teacher trials**; **none of the three teachers met the 80% qualification gate**. Full student training stopped at that predeclared gate, leaving all 12 seed/arm combinations explicitly unrun.

## Research translated into code

| Reference | Implemented mechanism | Deliberate scope |
|---|---|---|
| [Learning Perceptive Humanoid Locomotion over Challenging Terrain, 2025](https://arxiv.org/abs/2503.00692) | Privileged ground-height teacher; measured-terrain student; separate training critic | No variational world-model reproduction or borrowed hardware result |
| [SOLO, August 2026](https://arxiv.org/abs/2608.26583) | Fourier-encoded terrain-cell queries retrieve sparse point evidence; next-state teacher/student action disagreement enters the reward before GAE | Project ablation using simulated LiDAR packets, not the paper's depth-image architecture or benchmark |
| [Echo in the Steps, September 2026](https://arxiv.org/abs/2609.28960) | Learned gating of causal sensor history using packet age and proprioception | No claim to reproduce its saliency prior, alternation loss or physical parkour |

The four implemented student arms are `dense`, `query`, `query_memory`, and `query_memory_trajectory`. The protocol requires all student arms within a seed to train against the same qualified teacher checkpoint. The query-only model sees the newest available packet; the memory models see four causal packets. Missing returns and unavailable pre-reset history remain explicitly invalid. Packets never cross episode resets.

The student actor receives 45 proprioceptive values and point packets containing body-relative, gravity-aligned XYZ, a validity flag and age. It never receives absolute position, true linear velocity, evaluator terrain or teacher actions at deployment. A 77-cell overhead ground scan is the teacher's privileged label. The student sensor consists of 128 actual Isaac ray-casts from a body-mounted eight-ring sensor, with 1cm XYZ noise, 10% missing returns and 40ms delay during training. Ray geometry has ideal simulator orientation and intersects the terrain mesh; there is no RGB inference, measured intensity or self-occlusion model in this learning experiment.

The original flat-ground gait initializes the teacher with zero weights on added height inputs. A parity check establishes that expansion alone preserves its action function. PPO then trains those terrain weights. Student training combines PPO, action imitation and edge-sensitive terrain reconstruction. The trajectory arm adds the *actual next simulated state's* action disagreement before advantage estimation; terminal transitions are masked. Training-only critics are removed from exported TorchScript actors, whose parity is checked on independent batches and entirely missing returns.

## Frozen development protocol

Three seeds, 256 environments per seed, 200 teacher iterations and 100 iterations per student, with 24 rollout steps per iteration. Generated terrain uses five difficulty rows and five columns so the 40/20/20/20 rough/up-slope/down-slope/obstacle proportions are actually represented. Levels are fixed during the matched ablation; there is no cross-arm curriculum carry-over. Roughness is capped at2cm, slope parameter at0.10 and obstacle height at2.5cm.

The teacher must pass an engineering development gate before full student training: at least 80% of first-episode 10 s trials must survive, integrate more than 0.1 m of commanded travel, and keep integrated velocity-tracking error at or below half that commanded distance. Both command and measured velocity use the gravity-aligned yaw frame. This rejects motion opposite the command while allowing correctly followed curved paths; net displacement is retained only as a diagnostic. The exact rule is checked against the frozen protocol. Auto-reset episodes do not count as survivors. Students are evaluated under paired 0/50% dropout and 0/80 ms delay conditions. Report survival, qualification count, tracking error, terrain reconstruction error and measured actor p95 inference latency. Evaluation reuses the development terrain geometry, so it cannot establish held-out terrain or hardware generalization.

Implementation: [`perceptive_policy.py`](../src/bhl_robust/research/perceptive_policy.py), [`perceptive_env_cfg.py`](../src/bhl_robust/tasks/perceptive_env_cfg.py), [`perceptive_train.py`](../scripts/bench/perceptive_train.py). Exact source freezes, job IDs, failures and run status are in the [ledger](../results/methods-campaign-20261010/perceptive/SLURM_JOBS.md). The separate execution smoke exercised every student and export even though its one-iteration teacher failed the scientific development gate; those smoke trials are excluded from the development results below.

## Completed development result

Each seed finished its 200 teacher PPO iterations and 256 first-episode, 10 s qualification trials. The controller and all three jobs completed with scheduler exit `0:0`; their scientific outcome is **NEGATIVE**.

| Training seed | Job | Qualified trials | Survived trials | Teacher inference p95, 256-environment batch |
|---|---|---:|---:|---:|
| 0 | 21757635 | 113/256 (44.14%) | 236/256 | 0.440 ms |
| 1 | 21757710 | 102/256 (39.84%) | 224/256 | 0.433 ms |
| 2 | 21757754 | 107/256 (41.80%) | 216/256 | 0.416 ms |

Across the 768 measured episodes, 322 qualified and 676 survived. These pooled counts are descriptive; the independent experimental units are the three training seeds. The four full student arms ran for **zero seeds**: all **12 seed/arm combinations** remain `UNRUN_AFTER_FAILED_TEACHER_GATE`. Student reconstruction, deployment performance and paired degradation comparisons are therefore unavailable. Teacher inference timings cover synchronized model forward passes for the evaluation batch; they do not measure a complete robot control loop.

The [strict collection](../results/methods-campaign-20261010/perceptive-review/completed-three-seed-audit/collection.json) verifies source, protocol, warm-start, checkpoint and raw-result hashes and reproduces qualification from the original episode rows. [Per-seed metrics](../results/methods-campaign-20261010/perceptive-review/completed-three-seed-audit/teacher-summary.json), original teacher evaluations and [verification receipts](../results/methods-campaign-20261010/perceptive-review/completed-three-seed-audit/verification.json) are retained beside it. The dispatcher's generic `total_episodes: 0` results from absent top-level episode fields in the seed receipts; the strict collector establishes the authoritative count of 768 from the raw teacher evaluations.
