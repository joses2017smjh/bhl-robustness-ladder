# Native processing-schedule results — October 12, 2026

**48/48 declared development episodes measured; 0 explicitly UNRUN after nominal smoke gates.** The separate 12-episode smoke and identical-input transport profile passed their execution prerequisites.

This comparison uses actual ORB-SLAM3 and FAST-LIO2 native calls in closed-loop simulation. The treatment models separate simulation acquisition, a bounded latest joined camera/LiDAR queue, selected PNG transport, native service and evidence recording. It is not a measured concurrent robot implementation.

| Pipeline | Measured | UNRUN | Clean goals | Falls | Contacts |
| --- | --- | --- | --- | --- | --- |
| Legacy | 24 | 0 | 0 | 0 | 0 |
| Bounded latest | 24 | 0 | 11 | 0 | 0 |

The primary paired stale-stop change (candidate minus legacy) is **-80.39 percentage points**, across 24 comparable pairs. The paired clean-goal count change is **11**, across 24 complete pairs. Negative stale-stop differences favor the candidate; positive goal differences favor the candidate.

Two independent reset seeds, each repeated across declared routes/conditions, support descriptive development results only. No population confidence interval, hardware validation or independent confirmation is claimed.

## Condition and estimator results

| Estimator / condition | Pipeline | Goals / measured | UNRUN | Mean stale % | Stale decisions / navigation decisions | Falls / contacts | Superseded events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stereo / Nominal | Legacy | 0/4 | 0 | 97.33 | 2920/3000 | 0/0 | 238 |
| Stereo / Nominal | Bounded latest | 0/4 | 0 | 4.20 | 126/3000 | 0/0 | 0 |
| Stereo / 10 Hz camera | Legacy | 0/4 | 0 | 100.00 | 3000/3000 | 0/0 | 1451 |
| Stereo / 10 Hz camera | Bounded latest | 0/4 | 0 | 7.97 | 239/3000 | 0/0 | 637 |
| Stereo / 40 ms exposure | Legacy | 0/4 | 0 | 100.00 | 3000/3000 | 0/0 | 724 |
| Stereo / 40 ms exposure | Bounded latest | 0/4 | 0 | 20.33 | 610/3000 | 0/0 | 1 |
| LiDAR–IMU / Nominal | Legacy | 0/4 | 0 | 60.00 | 1800/3000 | 0/0 | 2 |
| LiDAR–IMU / Nominal | Bounded latest | 4/4 | 0 | 3.03 | 91/3000 | 0/0 | 0 |
| LiDAR–IMU / 10 Hz camera | Legacy | 0/4 | 0 | 100.00 | 3000/3000 | 0/0 | 1445 |
| LiDAR–IMU / 10 Hz camera | Bounded latest | 4/4 | 0 | 18.80 | 564/3000 | 0/0 | 406 |
| LiDAR–IMU / 40 ms exposure | Legacy | 0/4 | 0 | 100.00 | 3000/3000 | 0/0 | 724 |
| LiDAR–IMU / 40 ms exposure | Bounded latest | 3/4 | 0 | 20.63 | 619/3000 | 0/0 | 38 |

Stale-stop fractions use pre-step navigation decisions at or after 10 seconds. Initialization is excluded. Short horizons remain failures for clean-goal scoring. A missing navigation denominator is N/A; it is never scored as zero stale stops. Superseded events count the joined queue, not unique discarded camera frames.

## Timing

| Estimator / condition | Pipeline | Native client ms | Modeled delivery service ms | Available packet capture age ms |
| --- | --- | --- | --- | --- |
| Stereo / Nominal | Legacy | 47.25 | 308.92 | 695.50 |
| Stereo / Nominal | Bounded latest | 44.32 | 129.16 | 330.00 |
| Stereo / 10 Hz camera | Legacy | 47.33 | 5987.19 | 11434.25 |
| Stereo / 10 Hz camera | Bounded latest | 53.88 | 136.46 | 357.75 |
| Stereo / 40 ms exposure | Legacy | 48.25 | 13675.54 | 24212.00 |
| Stereo / 40 ms exposure | Bounded latest | 58.91 | 149.63 | 380.00 |
| LiDAR–IMU / Nominal | Legacy | 10.03 | 151.44 | 440.00 |
| LiDAR–IMU / Nominal | Bounded latest | 10.32 | 125.16 | 330.00 |
| LiDAR–IMU / 10 Hz camera | Legacy | 41.74 | 5845.06 | 11752.00 |
| LiDAR–IMU / 10 Hz camera | Bounded latest | 10.65 | 112.55 | 400.00 |
| LiDAR–IMU / 40 ms exposure | Legacy | 53.84 | 11201.51 | 24362.00 |
| LiDAR–IMU / 40 ms exposure | Bounded latest | 14.98 | 183.87 | 440.00 |

Each timing column is the **mean of per-episode p95 values**, not a pooled population p95. Native client measures real wall time; modeled delivery service adds the pipeline’s declared charge. Capture age includes waiting and delivery at the control decision, including available untracked response timestamps. The 350 ms capture-age gate is unchanged. Simulator rendering, raycasting and evidence work are separately recorded; their removal from candidate modeled inference service is an explicit scheduling assumption, not an observed hardware speedup.

## Every paired route and seed

| Estimator / condition | Seed / route | Legacy goal / stale % | Candidate goal / stale % | Goal difference | Stale difference pp |
| --- | --- | --- | --- | --- | --- |
| LiDAR–IMU / 10 Hz camera | 750000 / occluders | 0 / 100.00 | 1 / 17.07 | 1 | -82.93 |
| Stereo / 10 Hz camera | 750000 / occluders | 0 / 100.00 | 0 / 4.80 | 0 | -95.20 |
| LiDAR–IMU / 40 ms exposure | 750000 / occluders | 0 / 100.00 | 1 / 5.87 | 1 | -94.13 |
| Stereo / 40 ms exposure | 750000 / occluders | 0 / 100.00 | 0 / 8.27 | 0 | -91.73 |
| LiDAR–IMU / Nominal | 750000 / occluders | 0 / 60.00 | 1 / 0.00 | 1 | -60.00 |
| Stereo / Nominal | 750000 / occluders | 0 / 94.93 | 0 / 0.13 | 0 | -94.80 |
| LiDAR–IMU / 10 Hz camera | 750001 / occluders | 0 / 100.00 | 1 / 15.33 | 1 | -84.67 |
| Stereo / 10 Hz camera | 750001 / occluders | 0 / 100.00 | 0 / 5.20 | 0 | -94.80 |
| LiDAR–IMU / 40 ms exposure | 750001 / occluders | 0 / 100.00 | 1 / 2.27 | 1 | -97.73 |
| Stereo / 40 ms exposure | 750001 / occluders | 0 / 100.00 | 0 / 16.00 | 0 | -84.00 |
| LiDAR–IMU / Nominal | 750001 / occluders | 0 / 40.00 | 1 / 8.00 | 1 | -32.00 |
| Stereo / Nominal | 750001 / occluders | 0 / 95.87 | 0 / 3.47 | 0 | -92.40 |
| LiDAR–IMU / 10 Hz camera | 750000 / straight | 0 / 100.00 | 1 / 22.53 | 1 | -77.47 |
| Stereo / 10 Hz camera | 750000 / straight | 0 / 100.00 | 0 / 5.07 | 0 | -94.93 |
| LiDAR–IMU / 40 ms exposure | 750000 / straight | 0 / 100.00 | 1 / 26.27 | 1 | -73.73 |
| Stereo / 40 ms exposure | 750000 / straight | 0 / 100.00 | 0 / 15.87 | 0 | -84.13 |
| LiDAR–IMU / Nominal | 750000 / straight | 0 / 60.00 | 1 / 4.13 | 1 | -55.87 |
| Stereo / Nominal | 750000 / straight | 0 / 98.53 | 0 / 4.80 | 0 | -93.73 |
| LiDAR–IMU / 10 Hz camera | 750001 / straight | 0 / 100.00 | 1 / 20.27 | 1 | -79.73 |
| Stereo / 10 Hz camera | 750001 / straight | 0 / 100.00 | 0 / 16.80 | 0 | -83.20 |
| LiDAR–IMU / 40 ms exposure | 750001 / straight | 0 / 100.00 | 0 / 48.13 | 0 | -51.87 |
| Stereo / 40 ms exposure | 750001 / straight | 0 / 100.00 | 0 / 41.20 | 0 | -58.80 |
| LiDAR–IMU / Nominal | 750001 / straight | 0 / 80.00 | 1 / 0.00 | 1 | -80.00 |
| Stereo / Nominal | 750001 / straight | 0 / 100.00 | 0 / 8.40 | 0 | -91.60 |

## Prerequisites and reproducibility

| Nominal pipeline / estimator gate | Readiness |
| --- | --- |
| bounded_latest/lidar_inertial | PASS |
| bounded_latest/stereo | PASS |
| legacy_cumulative/lidar_inertial | PASS |
| legacy_cumulative/stereo | PASS |

The profile reprocessed the same 40 retained stereo pairs, checked decoded identities and the original LiDAR/IMU array hashes, and replayed measured transport costs through the queue. Its native service is explicitly zero. Sequential file-cache effects prevent treating this microbenchmark as an inferential speedup estimate.

Allocation `21801949`; frozen source `152ccf4d8a616c5725bfb0c44ac838818e028ee3`. Both pipelines use the same allocation, resource limits, frozen policy, route, reset seed, camera specification, native thresholds and response quantization. Pair order alternates in the frozen protocol. Closed-loop images can diverge after actions diverge.

Rehashed frozen input archive, result receipts and compact numerical evidence; independently rescored goals/safety/stale counts and timing accounting. Full archived images/native streams were verified by the frozen dispatcher; this reporter does not rerun native estimators or reverify entire output archives.

Files: [numerical evidence](report-evidence.json), [group CSV](groups.csv), [paired CSV](pairs.csv), [group figure](groups.png), [paired figure](pairs.png). Figures are also supplied as SVG.

Verified 2026-10-12T13:27:39.213380-07:00 (America/Los_Angeles).
