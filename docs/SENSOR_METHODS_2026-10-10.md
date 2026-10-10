# Fresh stereo, uncertainty mapping and native fusion comparison

One executable pipeline now renders original calibrated stereo, timed3D LiDAR and200Hz IMU; runs SGBM and the pinned pretrained C-Fast-FoundationStereo model; constructs ground maps; compares sensor fusion; and replays three actual native estimators. Independent simulator depth, obstacle segmentation and downward ground queries are stored separately for evaluation.

## Cohort and comparisons

The declared full cohort is three existing scene families × two fresh geometry seeds × three conditions: nominal, uniform texture, and50% LiDAR-return dropout. Each of the18 cells contains150 stereo pairs over30s. Geometry and texture seeds are fixed before rendering; all corruptions of a geometry stay in one development group. These are new variants of previously used families, not an unseen-family or physical confirmation set. The full planned totals are2,700 stereo pairs and54 native estimator cells.

Exposure occurs at each LiDAR scan's tail. Every return retains its acquisition time, and the native transport only consumes measurements available by each camera time. A stationary initialization interval and analytic IMU remain explicit simulation assumptions. Original PNGs, timed points, IMU, scene XML, input hashes, depth predictions, map arrays and estimator traces are retained.

| Comparison | Measured outputs |
|---|---|
| SGBM vs [C-Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) | Native and common-support depth MAE/RMSE, coverage, independently labeled near-obstacle pixel recall, warm decode+inference p95 and cold session initialization |
| LiDAR map vs each stereo map | Ground-height error, covered ground cells, hazard recall and unknown hazards |
| Equal-mean fusion vs uncertainty-gated fusion, for each stereo model | False traversable cells, disagreements, hazard recall including unresolved hazards, uncertainty interval coverage |
| FAST-LIO2 vs actual [FAST-LIVO2](https://arxiv.org/abs/2408.14035) vs the same FAST-LIVO2 binary with vision disabled | Tracking coverage, fixed-scale pose error, native compute latency, actual photometric measurement participation |

The same-binary visual ablation matters: FAST-LIVO2 also changes the LiDAR mapping backend, so a difference from FAST-LIO2 alone cannot isolate the benefit of camera measurements. Only the left camera enters native FAST-LIVO2; the right image is used by the separate stereo-depth experiment.

## Mapping assumptions and boundaries

The scalar elevation-map experiment is informed by [probabilistic terrain mapping](https://doi.org/10.1109/LRA.2018.2849506). It uses per-cell ground support, vertical-column rejection and explicit unknown cells. One-pixel disparity standard deviation is projected along the optical ray into map-height variance. LiDAR range, attitude, translation and scan-motion uncertainty are declared engineering assumptions. Empirical95% interval coverage is measured; calibrated confidence is not assumed.

Fusion compares an equal mean with an inverse-variance estimate gated by three-standard-deviation disagreement. Conflicting cells become unknown. A shared pose-variance floor does not average away. Maps are instantaneous and rig-local, using calibrated extrinsics; they do not accumulate ground-truth-registered frames or claim estimated-pose global mapping. Obstacle recall is a pixel metric, not object-instance recall. Terrain hazard recall includes an explicit unresolved count so low coverage cannot masquerade as perfect detection.

The [collector](../scripts/bench/sensor_methods_collect.py) requires the exact18 identities,150 frames each, all54 native cells, source/input hashes and real CUDA execution evidence. It keeps the three conditions together and uses geometry groups for paired statistics. Missing or corrupt evidence produces an incomplete result, never an inferred zero-failure result.

Implementation: [`methods_sensor_capture.py`](../scripts/bench/methods_sensor_capture.py), [`sensor_map_methods.py`](../src/bhl_robust/research/sensor_map_methods.py), [`sensor_methods_campaign.py`](../scripts/bench/sensor_methods_campaign.py). Run provenance and current status are in the [ledger](../results/methods-campaign-20261010/sensors/SLURM_JOBS.md). A fresh60-frame smoke must exercise actual photometric updates before full cells are released.
