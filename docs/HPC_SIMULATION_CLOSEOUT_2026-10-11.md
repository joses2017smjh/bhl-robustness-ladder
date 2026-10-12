# October 11 simulation campaigns: scientific closeout

The two remaining October 11 campaigns completed **680/680 scored episodes**:
200 native camera/fault controls and 480 independent terrain-confirmation trials.
Both collectors verified their complete declared cohorts and published final
archives. There are no outstanding scientific jobs in these two campaigns.
Execution `PASS` does not mean every method improved.

## Terrain: all five prespecified combinations confirmed

| Frozen actor | Heading clean goals | Regulated clean goals | Shared matched baseline | Baseline contact episodes / falls |
|---|---:|---:|---:|---:|
| `dr-default-s0` | 60/60 | 60/60 | 17/60 | 32 / 0 |
| `dr-default-s1` | 60/60 | Not qualified for confirmation | 1/60 | 57 / 2 |
| `dr-default-s2` | 60/60 | 60/60 | 1/60 | 56 / 3 |

All **300 candidate episodes** achieved clean goals with zero falls, contacts or
nonfinite states. The **180 unique baseline episodes** produced 19 clean goals,
145 contact episodes and five falls. Heading alone therefore achieved
**180/180 versus 19/180 clean goals**. Regulated comparisons reuse their actor's
baseline; those baseline trials are not counted again in the 480-episode total.

Each combination passed the unchanged gate: at least 54/60 clean goals, no falls,
contacts or nonfinite states, and a positive paired gain. The clean-goal-rate gain
was **71.7 percentage points** for the s0 comparisons, with descriptive 95% seed-cluster
interval **[63.3, 80.0]**, and **98.3 points [95.0, 100.0]** for s1/s2 comparisons.
The collector retained **44 rejected original LiDAR scans**, verified their timing
rejections and independently recomputed episode outcomes from raw traces.

Controllers use **native FAST-LIO2 estimated pose**, with evaluator ground truth
kept separate. Confirmation uses new procedural geometry and 20 reserved reset
seeds, but the same three frozen actors and flat, 1 cm step and 3° ramp families.
Intervals are conditional on those actors/families; five prespecified engineering
gates do not establish a family-wise statistical discovery, general robot safety
or hardware performance. This is separate from the oracle-pose hum5 program.

Source: [verified terrain archive](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/research-campaign-evidence-20261011/terrain-confirmation-final-20261011-5f09ee6f77d7.tar.gz),
[design and preserved execution history](TERRAIN_CONFIRMATION_2026-10-11.md).

## Native controls: completed comparison with negative navigation results

Five arms each completed 40 episodes across ten conditions, two routes and two
seeds. FAST-LIO2 achieved **19/40 clean goals**; native stereo, stereo–IMU,
permanent-stop stereo and verified LIO-assisted stereo recovery each achieved
**0/40**. All 200 episodes had zero falls and contacts. A stopped robot is not a
successful navigation trial.

| Condition | FAST-LIO2 clean goals | Mean episode full-pipeline p95 |
|---|---:|---:|
| Nominal | 2/4 | 159.6 ms |
| Camera 2.5 Hz | 4/4 | 153.1 ms |
| Camera 10 Hz | 0/4 | 5,780.3 ms |
| Dark exposure gain | 3/4 | 136.9 ms |
| Causal 40 ms exposure | 0/4 | 14,842.7 ms |
| Visual blackout | 2/4 | 194.0 ms |
| Quarter LiDAR returns | 2/4 | 145.0 ms |
| LiDAR extrinsic error | 2/4 | 148.6 ms |
| IMU clock +80 ms | 2/4 | 153.7 ms |
| Gyroscope bias | 2/4 | 152.9 ms |

Latency includes rendering, acquisition, encoding/hashing, native transport and
inference, and recovery; measured delay advances physics before delivery. These
are means of episode p95 values, not pooled p95 or isolated estimator timings.
The unchanged 350 ms pose-age gate remains part of every comparison. Four paired
episodes per condition support descriptive development findings, not a
general robustness ranking. All arms share the independent raw-LiDAR obstacle
brake, so an estimator-alone arm is not camera-only navigation. Zero observed
false-clear brake packets concern only the measured brake box.

The outcome-associated screen froze its threshold using seed 730000 before any
seed-730001 trial. On held-out LIO episodes, the selected 0.95 threshold and the
0.9 engineering baseline both accepted **11 clean and five adverse episodes**
and rejected four adverse episodes: **no gain over the baseline**. The other
four arms lacked outcome variation for fitting. This episode-end screen is
neither an online controller nor a calibrated failure probability.

Source: [verified native-control and health archive](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/research-campaign-evidence-20261011/native-controls-final-20261011-b48294ac61fb.tar.gz),
[frozen interventions and inference limits](NATIVE_SENSOR_CONTROLS_2026-10-11.md).

## Completed work and remaining scope

| Work | Status and next boundary |
|---|---|
| Original six research follow-ups | Initial implementations and measured studies complete: startup diagnosis, stereo–IMU navigation, native sensor faults, terrain control, dense fusion and map recovery. See [development results](METHODS_CAMPAIGN_RESULTS_2026-10-11.md). Negative findings remain valid. |
| Broader camera/fault controls, offline health screen, terrain confirmation | Complete: 200 + 480 episodes, as reported above. No additional cells remain in these declared cohorts. |
| Controlled stereo/LiDAR occlusion | [Complete: 30 cells / 300 pairs](FUSION_OCCLUSION_CONTROLS_2026-10-11.md); confidence filtering trades fewer false-free pixels for lower coverage and recall. |
| External camera/IMU replay | [TUM VI complete](TUMVI_EXTERNAL_VALIDATION_2026-10-11.md). The [EuRoC preparation and replay software](EUROC_FOLLOWUP_2026-10-12.md) is implemented and passed 61 focused tests; official dataset access still returned HTTP 429 and the legacy source timed out. Actual native smoke, replay and ATE remain unrun/unmeasured. |
| Final repository report observer | Complete. Reporter job **21788902** verified all original result bytes, unchanged scientific values and both freshly downloaded final archives. [Final verified report](research-completion-20261011/20261012T073850Z-1d71a69295d2.md) supersedes the retained incomplete report; no scientific episode was rerun. |
| Native navigation improvement | Open research objective. A [60-episode trace diagnosis](NATIVE_PIPELINE_DIAGNOSIS_2026-10-12.md) identified stale-pose stops and combined acquisition/recording overhead. It proposes an explicitly unrun 48-episode comparison of processing schedules; implementation, a fresh smoke and preregistration are needed before new outcomes. Do not tune or rerun this completed cohort to replace its negative results. |
| Online health-triggered control | Unrun separate experiment; the offline screen supplies no demonstrated advantage over its fixed baseline. |
| Earlier gated learning/navigation arms | Twelve perceptive student arms and 18 original stereo–IMU development episodes remain unrun after their original qualification failures. They are not automatically queued; a new method requires its own protocol and readiness gate. |
| C1 / moving-camera hardware validation | Blocked on robot or original recordings, timing/calibration evidence and measured mounting transforms. Planar intake and acquisition-time head kinematics remain necessary. The simulated 3D LiDAR and camera-fault study does not validate a moving camera with body-fixed IMU/C1. See [rig plan](HARDWARE_RIG_C1_STEREO_YAHBOOM_2026-10-11.md) and [Quest audit](QUEST_RIG_INTEGRATION_AUDIT_2026-10-11.md). |

The broader repository contains older independent backlog items. This closeout
covers the declared October 11 sensor/terrain program, not every historical
research idea. Bulk evidence belongs on shared storage or node-local scratch;
the authorized **2 TiB shared hard quota** and separate **25 GiB home limit**
remain in force, with the shared soft-quota grace tracked separately.

## Supported resume wording

- Implemented native estimated-pose humanoid terrain control and evaluated it in
  **480 held-out MuJoCo episodes**; heading control achieved **180/180 clean goals
  versus 19/180 baseline**, with observed contact episodes reduced **145 → 0**
  across three frozen actors and three terrain families.
- Built a reproducible native stereo/LiDAR/IMU evaluation pipeline with **200
  matched camera/fault trials**, causal latency accounting, independent raw-trace
  verification and a frozen held-out health screen; retained failed navigation
  and no-improvement results alongside successful execution evidence.

These are simulation and research-engineering claims. Neither is a physical
robot deployment or a demonstrated improvement from visual–inertial fusion.

## Closeout verification

The result-serialization repair passed **51 focused tests** and independent review against all 20 original native result files. [GitHub CI](https://github.com/joses2017smjh/bhl-robustness-ladder/actions/runs/38218284616) passed **2,997 tests**, with 132 skips and 10 subtests passed, plus actual trained-asset integration and seeded physics/fault replay. Reporter **21788902** completed with exit `0:0`, verified both freshly downloaded final artifacts and published the [complete immutable report](research-completion-20261011/20261012T073850Z-1d71a69295d2.md). The earlier failed report and all original scientific receipts remain intact. [Recovery and CI evidence](../results/simulation-completion-20261011/report-recovery/verification.json).

The subsequently added EuRoC software handoff also passed [full GitHub validation](https://github.com/joses2017smjh/bhl-robustness-ladder/actions/runs/38224054482): **3,012 tests**, 132 skips and 10 subtests passed, plus trained-asset and physics checks. Its actual external-recording replay remains blocked. The [storage check](../results/simulation-completion-20261011/report-recovery/storage-status.json) measured home at **21.02/25 GiB** and shared hard-quota headroom at about **537 GiB**; new bulk data stays on share or node-local storage.
