# Verified sensor replay pilot — 2026-10-08

Slurm job **21714203 completed the frozen pilot successfully**. An independent review rechecked the raw replay and recomputed its perception metrics without importing the production metric functions. The results establish a working simulated sensor benchmark; the four full research objectives remain open.

The replay contains **72 stereo pairs, 144 RGB images, 73,695 timed 3D LiDAR returns, and 960 IMU samples** across three scene groups. Each scene has 24 frames at 15 Hz, spanning 1.533 seconds from first to last timestamp. One scene is development, one validation, and one held out. This is a kinematic rig with ideal simulated cameras/IMU and ray LiDAR, rather than a humanoid walking or a physical sensor recording.

The independent review verified all **364 recorded replay file hashes**, all **144 stereo method/frame records** and their split aggregates, all **2,160 held-out fusion records**, and **72 unique held-out terrain maps**. Stereo and terrain reconstruction matched exactly; the largest fusion metric difference was 8.3e-17. Machine-readable evidence, hashes, counts, and timing samples are in [pilot-review.json](../results/sensor-campaign-20261008/v2/pilot-review.json). See the [campaign record](SENSOR_CAMPAIGN_2026-10-08.md) for frozen inputs and execution history, and the [demo](../results/sensor-campaign-20261008/demo/) for visual examples.

## Stereo depth: useful accuracy, coverage, and latency tradeoff

These measurements use the **single held-out ramp/step scene**. The model is the pinned official pretrained **C-Fast-FoundationStereo ONNX** member of the Fast-FoundationStereo family.

| Method | Native coverage on valid ground truth | Common-mask RMSE | Native-mask RMSE | Near-obstacle pixel recall | Warm p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| SGBM | 14.38% | 0.3104 m | 0.3104 m | 28.74% | 28.99 ms, CPU |
| C-Fast-FoundationStereo | 96.14% | 0.1467 m | 0.0926 m | 86.10% | 92.57 ms, Quadro RTX 8000 |

Both common-mask errors use exactly **241,024 shared supported pixels out of 1,675,679 valid ground-truth pixels**. Common-mask MAE is 0.1440 m for SGBM and 0.0467 m for the pretrained model. Native-mask MAE is 0.1440 m and 0.0443 m respectively. Coverage is reported alongside error because the native supports differ substantially: SGBM applies left/right matching consistency, while this network export provides only left disparity and its visibility/depth validity mask.

Near-obstacle recall uses independently segmented non-ground scene geometry within 3 m, with an absolute depth-error tolerance of 0.15 m. SGBM recalls **97,110/337,892** labelled near-obstacle pixels; the model recalls **290,940/337,892**. These are correlated pixel counts, not independent obstacle instances or robot safety outcomes.

The timing quantiles were recomputed from **200 measurements after 30 warmups per method** on the same host. They include warm-cache PNG decoding, preprocessing, inference, and depth/validity calculation; capture, rectification, fusion, output saving, and control are excluded. SGBM used one OpenCV CPU thread. The model's first prediction recorded **6,973 CUDA kernel events, including 218 convolution events**, with CPU shape/control operators allowed. Session initialization took 12.54 seconds and the first replay call 0.729 seconds, separately from warm timing.

The model's 92.57 ms p95 exceeds a 15 Hz camera's 66.67 ms frame interval under this sequential benchmark. A navigation deployment therefore needs a measured scheduling or acceleration solution. There is **one held-out scene**, so the report correctly withholds a bootstrap confidence interval and makes no general superiority claim.

## Fusion: the confidence gate needs revision

Both stereo frontends were evaluated with five arms: stereo alone, sparse LiDAR alone, arithmetic fusion, confidence-gated fusion, and a conservative minimum endpoint. Three return-retention rates and three extrinsic settings give **nine conditions per frame**. The full replay produced 6,480 arm/condition/frame records; 2,160 belong to the held-out scene. These are repeated evaluations of the same recorded frames, not 6,480 independent experiments.

The table shows simple versus gated fusion at **100% LiDAR retention**. Each row uses the same **337,892** labelled near-obstacle pixels. Unlike the stereo depth-tolerance recall above, fusion classifies a labelled near obstacle as detected when its predicted depth is at most 3 m. A farther valid prediction is false free space; an unknown prediction is also a miss.

| Frontend and calibration | Simple → gated false-free pixels | Simple → gated missed pixels | Simple → gated unknown pixels |
| --- | ---: | ---: | ---: |
| SGBM, nominal | 2,897 → 2,891 | 239,349 → 239,343 | 236,452 → 236,452 |
| SGBM, 3° + 3 cm stress | 3,091 → 3,091 | 239,406 → 239,406 | 236,315 → 236,315 |
| C-Fast-FoundationStereo, nominal | 18,068 → 18,036 | 61,729 → 61,697 | 43,661 → 43,661 |
| C-Fast-FoundationStereo, 3° + 3 cm stress | 18,219 → 18,289 | 61,880 → 62,040 | 43,661 → 43,751 |

The gate's nominal change is small and its stressed pretrained-model result is worse. It **does not meet the proposed 20% false-free-space reduction target**. This is a useful negative result: binary validity scores and this heuristic gate do not resolve calibration disagreement reliably in the pilot. These are descriptive results, without independent-pixel significance tests.

Sparse LiDAR alone has zero nominal false-free pixels but detects only **1,484/337,892** near-obstacle pixels and covers **8,766/1,675,679** valid-ground-truth pixels. Its missing coverage prevents interpreting zero false-free events as adequate obstacle perception. All scans retain their original 50 ms point timing; none of these methods compensates scan motion.

## Terrain: low observed height error, limited hazard coverage

The baseline maps raw 3D endpoints into rig-local elevation, slope, and roughness cells. Height labels come from independent dense vertical rays, mapped to the declared grid by nearest sample; geometric hazard labels come from that independent surface. Unknown cells remain unknown.

| LiDAR retention | Observed / labelled height cells | Height RMSE on observed cells | Detected / labelled hazard cells | Unknown hazard cells |
| --- | ---: | ---: | ---: | ---: |
| 100% | 5,257 / 21,888 | 0.0535 m | 356 / 3,781 | 17,520 / 21,888 |
| 50% | 3,493 / 21,888 | 0.0662 m | 208 / 3,781 | 19,384 / 21,888 |
| 20% | 1,650 / 21,888 | 0.0727 m | 64 / 3,781 | 21,183 / 21,888 |

At full retention, height coverage is **24.02%** and hazard recall is **9.42%**, with 128 false-positive hazard cell/frame counts. Height RMSE applies only to observed cells; it does not describe the 80.04% of cells lacking sufficient support for hazard classification. These are pooled cell/frame counts over 24 correlated frames. Terrain output is identical for the two stereo frontend runs because terrain uses LiDAR alone; the duplicate outputs are **not independent repetitions**. No terrain traversal was performed, and these geometric labels do not measure surface friction or safe footholds.

## Pose readiness and a defensible resume statement

All three scenes exported both ORB-SLAM3 stereo inputs and FAST-LIO2 timed LiDAR/IMU packets, and all six adapter export receipts were checked. Their status is **READY_INPUTS_WAITING_RUNTIME**. The scientific route remains **BLOCKED_RUNTIME**, with **zero estimated-pose outputs and zero closed-loop navigation episodes**. Drift, tracking failures, goals, collisions, falls, and traversal outcomes remain unmeasured.

A conservative project bullet supported by this completed run is:

> Built and validated a calibrated sensor replay benchmark on 72 simulated stereo pairs and 73,695 timed 3D LiDAR returns, comparing SGBM and pretrained stereo across five fusion methods and nine sensor conditions with verified CUDA execution and independent geometry labels.

The stronger next evidence comes from additional independent scenes, continuous calibrated physical capture, motion compensation and calibration-aware gating, native estimator replay, and then fresh closed-loop trials. The [resume evidence guide](SENSOR_RESUME_EVIDENCE_2026-10-08.md) connects these capabilities to official robotics job requirements.
