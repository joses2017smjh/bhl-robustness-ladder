# Completed Humanoid methods campaigns — October 11, 2026

The queued terrain, stereo-recovery, fresh-sensor, and perceptive-learning campaigns have finished and their declared cohorts were independently collected. The later sensor-fault and cost-consistency-fusion continuation also finished. Terrain heading control and pretrained stereo depth produced the clearest measured gains. Perceptive teacher qualification and stereo–IMU initialization remained negative; their dependent stages were not run.

These are **simulation development results**. Execution or collection `PASS` means the declared experiment and evidence validated; it does not mean every method improved. Independent confirmation, physical hardware validation, and the remaining controls in [the implementation report](RESEARCH_METHODS_2026-10-10.md#remaining-original-design-scope) remain unfinished.

The four-track finalizer finished October 10 at 19:00 PDT. The fault/fusion continuation finished at 20:15 PDT. This report supersedes the partial campaign counts in the October 10 status report while preserving the original diagnostic and smoke outcomes.

## Completion and evidence

| Campaign | Collected cohort | Scientific outcome |
|---|---|---|
| Native-pose terrain navigation | 9/9 jobs; 270/270 episodes | Heading: 90/90 clean goals; baseline: 9/90 |
| Stereo map recovery | 9/9 jobs; 18/18 episodes | Verified recovery: 2/9 clean goals; permanent stop: 0/9 |
| Fresh stereo/LiDAR/IMU comparison | 18/18 cells; 2,700/2,700 camera frames; 54/54 native estimator cells | Better pretrained depth accuracy and coverage, with higher latency; all native estimator cells qualified |
| Perceptive learning | 3/3 teacher seeds; 768 evaluations | 0/3 teachers qualified; 12 full student arms unrun |
| Native sensor faults | 48/48 replay shards; 144 estimator cells | 130 qualified, 14 negative |
| Local-cost/consistency fusion | 24/24 replay shards; 120 arm cells | Lower conditional error/false-free pixel counts with lower coverage and recall |

The collectors report no missing cells or provenance problems. The fault/fusion terminal observer records parent job `21757767` as `COMPLETED`, exit `0:0`, and verifies all **73/73 expected raw publications**. [Exact terminal JSON and a byte-verified copy manifest](../results/methods-campaign-20261010/final-collected/) are checked in. The manifest records source paths, sizes and SHA256 hashes; the copy operation did not rerun scientific inference.

## Terrain: complete matched development cohort

Three frozen locomotion actors were evaluated across flat terrain, 1 cm steps and a 3° ramp, ten matched reset seeds per actor/course, and three controllers. Each arm therefore has 90 episodes. All controllers use native estimated pose; evaluator ground truth is not supplied to the controller. A clean goal requires reaching the goal, completing the 40-second horizon, and having no fall, contact or nonfinite state.

| Controller | Clean goals | Contact episodes | Falls | Nonfinite episodes | Mean lateral path RMSE |
|---|---:|---:|---:|---:|---:|
| Forward-command baseline | 9/90 (10.0%) | 77/90 | 0 | 0 | 0.2332 m |
| Heading/cross-track pursuit | 90/90 (100.0%) | 0/90 | 0 | 0 | 0.0579 m |
| Regulated-speed pursuit | 86/90 (95.6%) | 0/90 | 0 | 0 | 0.0576 m |

Heading pursuit improved clean-goal rate by **90 percentage points**, removed all 77 observed baseline contact episodes, and reduced mean lateral RMSE by **75.2%**. Regulated pursuit improved clean-goal rate by **85.6 percentage points**. Zero falls were observed in every arm, so these results do not demonstrate a fall reduction.

The predeclared seed-cluster bootstrap intervals for clean-goal-rate difference are **[86.7, 93.3] percentage points** for heading and **[80.0, 91.1]** for regulated control. They are conditional on three fixed actors and three fixed terrain families with ten reset-seed clusters, not physical or generalized safety intervals.

All three heading/actor combinations and two regulated/actor combinations (`dr-default-s0` and `dr-default-s2`) qualify for **separately preregistered confirmation**. The third regulated combination did not qualify. Confirmation has **not** been executed; the navigation changes do not establish a newly terrain-trained gait.

Source: [terrain scientific summary](../results/methods-campaign-20261010/final-collected/terrain/scientific-summary.json), [verified release archive](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/methods-campaign-evidence-20261010/methods-final-terrain-20261010-99b46058189c.tar.gz).

## Stereo recovery: a limited improvement

The complete comparison contains three routes × three reset seeds × two arms: **18 episodes and nine matched pairs**.

| Arm | Clean goals | Falls | Contacts | Verified recovery events |
|---|---:|---:|---:|---:|
| Permanent stop after map changes | 0/9 | 0 | 0 | 0 |
| LiDAR-assisted, independently validated stereo map recovery | 2/9 | 0 | 0 | 60 |

The clean-goal-rate difference is **22.2 percentage points**, with a descriptive three-seed-cluster interval of **[7.2, 33.3] percentage points**. Sixty recovery events are not sixty successful trials: seven of nine recovery episodes still did not produce clean goals. The uncertainty calculation enumerates the 27 resamples of only three seed groups and one trained actor; it supports no population or hardware inference. Independent confirmation is unrun.

Source: [stereo scientific summary](../results/methods-campaign-20261010/final-collected/stereo/scientific-summary.json), [verified release archive](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/methods-campaign-evidence-20261010/methods-final-stereo-20261010-7123df0e9f50.tar.gz).

## Stereo depth and native estimation

The fresh cohort contains six geometry variants from three previously used scene families, each under nominal, low-texture and half-LiDAR-return conditions: **18 cells, 150 frames per cell, 2,700 frames total**. The values below are equal means across the six geometry groups, with all three conditions equally weighted within each group. Errors, coverage and latency must be read together.

| Metric | SGBM | C-Fast-FoundationStereo |
|---|---:|---:|
| MAE on common valid support | 0.1896 m | 0.0457 m |
| RMSE on common valid support | 0.5530 m | 0.1811 m |
| MAE on each method's native valid support | 0.1895 m | 0.0373 m |
| RMSE on each method's native valid support | 0.5528 m | 0.1116 m |
| Valid-depth coverage | 16.92% | 96.00% |
| Near-obstacle pixel recall | 23.56% | 89.72% |
| Mean of per-cell p95 latency | 136.8 ms | 300.1 ms |

Pretrained depth reduced **common-support MAE by 75.9%**. The paired difference is −0.1440 m, with a descriptive geometry-group bootstrap interval of [−0.1794, −0.1074] m. Coverage increased by **79.1 percentage points**, and near-obstacle pixel recall by **66.2 percentage points**. Latency increased by **163.3 ms** in the mean of per-cell p95 measurements; this is not a pooled p95 or a real-time closed-loop guarantee. Native-support errors use different coverage denominators and are not substituted for the common-support comparison. Pixel recall is not obstacle-instance detection.

All **54/54 native estimator cells** qualified:

| Native estimator | Qualified cells | Mean conditional ATE RMSE | Mean tracking fraction | Mean of per-cell p95 latency |
|---|---:|---:|---:|---:|
| FAST-LIO2 | 18/18 | 0.02091 m | 98.00% | 8.63 ms |
| FAST-LIVO2 | 18/18 | 0.02019 m | 98.67% | 17.81 ms |
| FAST-LIVO2, visual measurements disabled | 18/18 | 0.01982 m | 98.67% | 3.37 ms |

There is **no supported overall accuracy gain from adding vision** here. LIVO-minus-LIO mean ATE difference is −0.00072 m with interval [−0.00703, 0.00607] m; LIVO-minus-no-visual difference is +0.00038 m with interval [−0.00248, 0.00526] m. Visual fusion adds latency. These fresh-cohort results do not replace the earlier consumed-data diagnostic, where full LIVO qualified only two of three scenes.

The original confidence-gated versus simple ground-map fusion also shows no material overall benefit. With SGBM, gating changes height RMSE by only −0.000058 m while slightly reducing coverage/recall and increasing mean false-traversable ground cells by 0.0285. With pretrained stereo, the changes are −0.0000135 m height RMSE and −0.00333 mean false-traversable cells; the latter's interval includes zero. These small map-level effects do not establish navigation safety improvements.

Source: [sensor scientific summary and all paired statistics](../results/methods-campaign-20261010/final-collected/sensors/scientific-summary.json), [verified release archive](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/methods-campaign-evidence-20261010/methods-final-sensors-20261010-fdcf800ed070.tar.gz).

## Native sensor faults: severe return loss remains a failure mode

The continuation replays the six nominal geometry captures under eight predeclared conditions with three actual native estimators, using common 148-frame interior windows. There are **48 replay shards and 144 estimator cells**. Conditional error is always accompanied by tracking and the qualification gate; untracked frames are not zero-error observations.

| Condition | FAST-LIO2 qualified | FAST-LIVO2 qualified | LIVO without vision qualified |
|---|---:|---:|---:|
| Nominal | 6/6 | 6/6 | 6/6 |
| IMU clock +20 ms | 6/6 | 6/6 | 6/6 |
| IMU clock +80 ms | 6/6 | 6/6 | 6/6 |
| LiDAR extrinsic error: 2° yaw and 2 cm x | 6/6 | 6/6 | 6/6 |
| Gyroscope z bias after 5 s | 6/6 | 6/6 | 6/6 |
| Accelerometer x bias after 5 s | 6/6 | 6/6 | 6/6 |
| Retain half of LiDAR returns | 6/6 | 6/6 | 5/6 |
| Retain one-quarter of LiDAR returns | 1/6 | 2/6 | 2/6 |
| **Total** | **43/48** | **44/48** | **43/48** |

Retaining one-quarter of returns caused **13 of the 14 negative cells**. Relative to matched nominal replay, mean conditional ATE increased by **0.5244 m** for FAST-LIO2, **2.7220 m** for FAST-LIVO2, and **0.1476 m** for the visual ablation. LIVO also lost **16.26 percentage points of tracking** on average in this condition; the other two had zero mean tracking-fraction change despite worse pose error. Passing the engineering gate under the other tested perturbations is not proof of general robustness or an outcome-calibrated health estimate. No closed-loop recovery, collision or goal outcomes were measured in this replay.

Source: [native fault collection](../results/methods-campaign-20261010/final-collected/sensor-followup/native-collection.json).

## Local-cost/consistency fusion: an explicit coverage tradeoff

The fusion continuation completed **24 shards and 120 arm cells**: six geometry groups × four conditions × five arms. The method uses local photometric SAD cost and consistency filtering; it does not reproduce an aggregated SGM cost volume or the cited three-view fusion paper. Confidence scores are engineering weights, not calibrated probabilities.

Nominal-condition results, averaging errors/coverage/recall across six groups and summing pixel counts:

| Arm | Mean conditional MAE | Coverage | Near-obstacle pixel recall | False-free obstacle pixels | Unknown obstacle pixels |
|---|---:|---:|---:|---:|---:|
| SGBM | 0.09308 m | 23.21% | 30.78% | 48,772 | 46,953,284 |
| SGBM with local-cost filtering | 0.02942 m | 12.43% | 24.14% | 9,423 | 52,369,598 |
| Sparse projected LiDAR | 0.03477 m | 0.27% | 0.24% | 6,483 | 71,545,366 |
| Simple projected fusion | 0.09231 m | 23.40% | 30.92% | 55,044 | 46,851,827 |
| Cost-consistency fusion | 0.02904 m | 12.65% | 24.30% | 11,379 | 52,254,471 |

Cost-consistency fusion reduced nominal false-free obstacle pixels by **79.3%** compared with simple fusion, but increased unresolved obstacle pixels, reduced coverage from **23.40% to 12.65%**, and reduced pixel recall from **30.92% to 24.30%**. Its lower conditional MAE is partly a different-support comparison. Unknown space must not be described as safe free space.

The same tradeoff persists under the tested LiDAR perturbations:

| Condition | Simple / cost-consistency MAE | Simple / cost-consistency coverage | Simple / cost-consistency pixel recall | Simple / cost-consistency false-free pixels |
|---|---:|---:|---:|---:|
| 2° yaw + 2 cm extrinsic error | 0.09424 / 0.03274 m | 23.40% / 12.65% | 30.88% / 24.26% | 79,922 / 27,181 |
| Half returns | 0.09271 / 0.02924 m | 23.30% / 12.54% | 30.85% / 24.22% | 52,013 / 10,471 |
| Quarter returns | 0.09291 / 0.02934 m | 23.25% / 12.48% | 30.82% / 24.18% | 50,499 / 9,966 |

Source: [all 120 fusion-arm measurements](../results/methods-campaign-20261010/final-collected/sensor-followup/fusion-collection.json), [terminal observer](../results/methods-campaign-20261010/final-collected/sensor-followup/terminal-summary.json), [verified combined metadata archive](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/methods-campaign-evidence-20261010/sensor-followup-21757767-terminal-observer-21759391.tar.gz). The terminal summary lists all 73 raw assets in the [evidence release](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/tag/methods-campaign-evidence-20261010).

## Negative learning and inertial initialization results

Each perceptive teacher completed its fixed 200-iteration training budget and 256 evaluation episodes. Qualification requires the frozen survival and commanded-motion/tracking criteria in at least 80% of episodes.

| Training seed | Qualified episodes | Qualification rate | Surviving episodes | Survival rate | Inference p95 |
|---|---:|---:|---:|---:|---:|
| 0 | 113/256 | 44.14% | 236/256 | 92.19% | 0.440 ms |
| 1 | 102/256 | 39.84% | 224/256 | 87.50% | 0.433 ms |
| 2 | 107/256 | 41.80% | 216/256 | 84.38% | 0.416 ms |

**No teacher qualified.** All **12 full student arms remain unrun**, and there is no valid paired student comparison. Survival alone does not satisfy the gate. The earlier one-iteration execution smoke exercised all student variants but does not supply full-study performance. Source: [perceptive scientific summary](../results/methods-campaign-20261010/final-collected/perceptive/scientific-summary.json), [verified release archive](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/methods-campaign-evidence-20261010/methods-final-perceptive-20261010-e0d4ca3303a6.tar.gz).

The separately reported stereo–IMU experiment remains negative: the 900-frame paired diagnostic produced zero inertially initialized frames, and neither inertial arm in the four-episode navigation smoke initialized. All smoke episodes had zero goals, falls and contacts. The **18 development episodes remain unrun** after that readiness gate. These short smoke outcomes are not development goal trials. See [the native inertial report](STEREO_INERTIAL_METHODS_2026-10-10.md).

## Supported resume bullets

- Implemented estimated-pose heading control for a simulated humanoid; across 270 matched development episodes, achieved **90/90 clean goals versus 9/90 baseline**, eliminated observed contacts (**77 → 0**), and reduced mean lateral path RMSE **75.2%** across three frozen actors and three terrain families.
- Benchmarked classical and pretrained stereo on **2,700 simulated frames across 18 conditions/geometry cells**; reduced common-support depth MAE **75.9%** and increased valid-depth coverage **16.9% → 96.0%**, while quantifying the mean per-cell p95 latency tradeoff (**136.8 → 300.1 ms**).

The claims above describe measured simulation development work, not unseen-family confirmation, physical robot performance, or deployment readiness. Code validation remains the [passing 2,658-test CI run for commit `26423fa`](https://github.com/joses2017smjh/bhl-robustness-ladder/actions/runs/38089561191), followed by trained-model checks and physics replay; this terminal-evidence/report update adds no scientific code or experiments.
