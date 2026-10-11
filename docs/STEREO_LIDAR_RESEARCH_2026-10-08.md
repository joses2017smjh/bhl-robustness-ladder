# Stereo and LiDAR research for humanoid navigation

Research plan prepared October 8, 2026. **The experiments and numerical targets below are proposed; they are not measured results.** Current run outcomes belong in the [task closeout](TASK_CLOSURE_2026-10-08.md) and [status ledger](STATUS.md).

**Implementation update, October 8:** the separately frozen [sensor pilot](SENSOR_PILOT_RESULTS_2026-10-08.md) completed on 72 rendered stereo pairs with timed 3D scans and IMU. SGBM/C-Fast-FoundationStereo, five fusion arms, geometric terrain evaluation and estimator-input exports now have executable code and retained outputs. Confidence gating did not meet the proposed improvement target. Native SLAM, estimated-pose closed-loop navigation and terrain traversal remain follow-up work. [Actual implementation and dependency queue](SENSOR_CAMPAIGN_2026-10-08.md).

## Research question and current foundation

Can **motion-aware, confidence-gated stereo–LiDAR fusion** preserve obstacle detection and navigation during gait, sensor delays, and, where supported, head movement better than either sensor alone or naïve fusion? A useful contribution would connect perception failures to robot behavior through controlled, reproducible experiments. Novelty has not been established; integrating published components does not itself establish a new algorithm.

There is already a calibrated CPU [StereoSGBM input path](STEREO_INPUT.md). It computes disparity from actual rectified left/right images, performs consistency/validity checks, and preserves invalid depth as NaN. Its synthetic numeric fixtures establish conversion correctness, not physical-camera accuracy. Existing maze PPO observations use simulated ray-cast depth. Calling those observations “stereo” does not demonstrate image-based matching, motion blur, camera exposure effects, or physical sensor performance. The H1 confirmation is now **DONE/PASS** with frozen 12-DoF gait/navigation components and oracle pose; its exact outcomes are linked in the [task closeout](TASK_CLOSURE_2026-10-08.md). It is background evidence, not a stereo–LiDAR or estimated-pose result. Preserve its completed protocol and evidence.

The read-only [sensor dataset auditor](../scripts/bench/sensor_dataset_audit.py) and its [published audit](../results/task-closure-20261008/sensor-data-audit/audit.json) found **nine integrity-valid legacy artifacts**, seven containing sparse logged sensor snapshots. Those snapshots include simulated paired ray-depth and pooled LiDAR sectors, generally at 1-second logged spacing. They do not supply the original RGB stereo pairs, raw timed LiDAR scans and calibration/timing evidence required for these research replays: **zero artifacts are replay-ready**. These files include reused probes/video episodes and must not be counted as nine independent experiments. Logged spacing is not the sensors' publication frequency.

The highest-value first deliverable is a synchronized dataset and benchmark connecting depth, uncertainty, localization, and downstream outcomes. Start with simulation and existing recordings; hardware claims require a separately identified physical capture cohort. The exact stereo camera, LiDAR dimensionality, IMU, mount, and synchronization capabilities still need inventory.

## Scope and the S0 → S4 ladder

| Stage | Question and baseline | Deliverable | Priority |
|---|---|---|---|
| **S0: data, calibration and time** | Do metric units, projection, image pairing and motion compensation agree? Compare calibrated/time-aware processing against intentionally perturbed calibration/timing. | Audited recordings, calibration, timestamp/frame contract, failure masks and motion sensitivity curves. | First; mostly CPU. |
| **S1: stereo depth** | Does a pretrained stereo model improve usable obstacle depth over the existing calibrated SGBM path? | Identical-image comparison with valid coverage, near-obstacle errors and latency. | First offline experiment; inference before training. |
| **S2: stereo–LiDAR fusion** | Does consistency/confidence gating outperform single sensors and simple fusion during sensor degradation? | Paired fusion ablations, risk–coverage curves and false-free-space measurements. | Strongest near-term research direction. |
| **S3: estimated-pose navigation** | Does improved sensing/localization help the robot when pose is estimated instead of supplied by the simulator? | Odometry replay and then matched closed-loop navigation. | After S0–S2 pass. |
| **S4: terrain and self-supervision** | Can geometric maps plus visual/proprioceptive information improve traversal decisions? | Elevation/traversability benchmark and, later, learned terrain cost. | Later extension; needs 3D coverage and appropriate locomotion. |

Limit the first campaign to the three experiments below. S4 and training large perception models are not prerequisites.

## Experiment 1 — motion, calibration and usable stereo depth

**Design.** Propose 24 independent scene/route sequences of at least 20 seconds each, with 6 development, 6 validation and 12 held-out test sequences. Separate scene assets/routes across splits where possible; neighboring frames or repeated corruptions of one capture are not independent samples. Target at least 15 captured stereo pairs per second if the actual rig supports it; 24 × 20 × 15 would be 7,200 original pairs, not an achieved collection count. Keep native timestamps/rate and record dropped frames rather than manufacturing the target rate.

Capture static viewing, straight walking and turning segments. Include head articulation only if the robot/model actually has that mechanism; otherwise evaluate body-induced sensor motion. Use a fixed gait/checkpoint, camera calibration, resolution, exposure configuration and path. Save oracle poses/depth for evaluation in a separate namespace that inference cannot read.

**Methods.** Run the existing SGBM path first, then one pretrained RAFT-Stereo configuration and one Fast-FoundationStereo configuration. FoundationStereo is an optional offline accuracy reference; WAFT-Stereo is an optional current challenger after the primary comparison. Freeze checkpoint hashes, resolution, iteration counts and preprocessing before opening the test split. All methods receive the same original images; neither a pretrained model nor SGBM receives simulator depth. [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), [FoundationStereo](https://nvlabs.github.io/FoundationStereo/), [Fast-FoundationStereo, CVPR 2026](https://github.com/NVlabs/Fast-FoundationStereo), and [WAFT-Stereo, 2026](https://arxiv.org/abs/2603.24836) provide primary reference implementations. Their upstream speed/accuracy claims are not BHL results.

For rectified images, the existing contract uses:

```text
d = u_left - u_right
d_eff = d - (cx_left - cx_right)
Z = fx_px * baseline_m / d_eff
```

`Z` is left-camera optical-axis depth, not slant range. Image resizing must also scale focal length/principal points/disparity consistently. Reject nonpositive effective disparity and invalid rectification borders. Store both validity and uncertainty; a matching residual is not automatically a calibrated probability. [OpenCV's calibration documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) supports the geometry and fixed-point disparity convention.

**Ablations.** On copies of the same recording, separately test stereo capture skew of 0/5/15/35 ms; camera–LiDAR time offsets of 0/10/25/50 ms; and extrinsic perturbations of 0/1/3 degrees or 0/1/3 cm. These are deliberately injected faults, not measured hardware errors. Evaluate realistic motion at those offsets rather than merely changing timestamp labels. The existing 35 ms stereo-pair rejection default is an input guard, not an accuracy guarantee. Never change its guard silently to retain difficult frames.

**Metrics.** Report depth MAE/RMSE/AbsRel on common evaluator-valid pixels, disparity EPE where disparity ground truth exists, valid coverage, near-obstacle recall, edge/thin-object error, and performance by range and motion. Report accepted-pixel error *and* all-pixel coverage to expose methods that improve accuracy by rejecting almost everything. Define obstacle labels and a proposed 0.2–3 m region before test evaluation; adjust that region on development data if the actual robot/calibration requires it. Preserve both common-mask and native-mask results.

**Proposed gate.** Data/calibration acceptance must pass before model comparison. A model advances only if it improves test-sequence obstacle recall or depth error over SGBM without increasing dangerous free-space errors, and meets the selected sensor/planner deadline on the named machine. The development target is at least 10% lower sequence-averaged near-range RMSE and no more than a 2 percentage-point coverage loss; this is a target, not a scientific finding. Confidence intervals and failure cases determine whether the observed effect supports an improvement claim.

## Experiment 2 — confidence-gated stereo–LiDAR fusion

Use the same frozen sequences and evaluation masks. LiDAR must be an actual recorded sensor stream or a clearly labeled simulated scan model. Subsampling dense renderer depth makes a useful geometry test, but does not reproduce a physical LiDAR's returns, timing or material response.

Compare five arms: **stereo only; LiDAR only; simple projected fusion; motion-compensated consistency-gated fusion; and a published nonlearning stereo–LiDAR method**. For simple fusion, state the exact projection, occlusion/z-buffer rule and weighting; do not imply sparse LiDAR is dense. The RA-L 2025 method by Yao, Ishikawa and Oishi combines SGM, sparse-disparity semidensification and a stereo/LiDAR consistency check, and exposes useful component ablations in its [paper](https://arxiv.org/abs/2504.05148) and [author code](https://github.com/yshry/libSGM_lidar). Its CUDA implementation requires a separate compatible environment.

Start gating with transparent rules: left/right stereo consistency, disparity validity, range-dependent error estimates, LiDAR projection visibility, and stereo/LiDAR disagreement. Calibrate thresholds/uncertainty on development/validation sequences only. Compare gating with and without motion compensation so their contributions can be separated. Keep unknown space unknown; sparse returns or completed depth must not automatically certify free space.

Before fusion, audit the legacy [Isaac sensor/sector path](../src/bhl_robust/sensors_rig.py), whose angular configuration is described as 500 rays per revolution and whose pooling truncates a remainder when the runtime ray count is not divisible by 36. Inspect the actual runtime ray count, angle ordering, full angular coverage and every raw ray's sector assignment. Add a boundary-obstacle fixture to show that pooling preserves obstacles around the full scan. The configuration/comment alone does not establish a measured coverage gap; a raw runtime scan and angle audit are required.

For sensors on different articulated links, use a timestamped transform chain:

```text
T_C_L(t) = inverse(T_B_C(q(t))) * T_B_L(q(t))
p_C(t_cam) = inverse(T_W_C(t_cam)) * T_W_L(t_point) * p_L
```

`T_A_B` maps coordinates in frame B into A. `q(t)` contains joint positions where relevant. The second equation accounts for sensor motion between a LiDAR point's capture time and the image time; the implementation must estimate/interpolate motion without reading evaluator ground truth. A static camera–LiDAR transform is valid only for a rigid relative mount. Spinning scans require per-point timing/deskew; a single scan timestamp can hide gait-induced distortion. [Kalibr](https://github.com/ethz-asl/kalibr) handles camera and camera–IMU calibration, while [FAST-Calib](https://github.com/hku-mars/FAST-Calib) provides spatial LiDAR–camera calibration; neither claim should be expanded into an unsupported universal timing solution.

**Fault matrix.** Start with nominal input and independent LiDAR return retention of 100/50/20%, then whole-image/whole-scan dropout bursts, stereo-only contrast/blur changes and the timing sweep from Experiment 1. Use fixed fault seeds/masks across methods. Confirm one selected stress condition on untouched test recordings after validation. RGB contrast/blur and synthetic return dropout are corruption studies; realistic fog/rain performance requires validated rendering or physical data.

**Metrics and proposed gate.** Retain Experiment 1's error/coverage metrics and add false-free-space rate in the near-obstacle region, disagreement rejection rate, risk–coverage curves, and tail latency. Use independent ground truth/withheld measurement locations for depth completion accuracy; scoring a method only at its own input LiDAR points is insufficient. The development target is a 20% relative reduction in false-free-space events over simple fusion in a preselected degraded condition, with nominal obstacle recall within 2 percentage points and latency below the declared deadline. Advance only if the paired test evidence supports the reduction; zero baseline events cannot support a percentage-reduction claim.

Optional [KBNet](https://github.com/alexklwong/calibrated-backprojection-network) is an RGB + sparse-depth + intrinsics completion baseline, not a stereo-fusion implementation. Introduce it after the nonlearning comparison; its older runtime dependencies should be isolated. A learning extension may train a small reliability predictor using recorded geometric residuals, but must beat the transparent gate on held-out scenes before replacing it.

## Experiment 3 — estimated pose and closed-loop behavior

First replay recorded sequences through [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) in stereo mode. Stereo-inertial is a separate arm only if usable synchronized IMU measurements exist. With 3D LiDAR and IMU, add [FAST-LIO2](https://github.com/hku-mars/FAST_LIO); then evaluate [FAST-LIVO2](https://arxiv.org/abs/2408.14035), which fuses image, LiDAR and IMU rather than being a ready-made binocular stereo frontend. Check point timing, calibration and supported sensor formats before selecting either. [FAST-LIVO2's official implementation](https://github.com/hku-mars/FAST-LIVO2) documents its ROS/environment requirements and synchronized rig. Do not label the comparison “LiDAR only” if IMU contributes.

Measure ATE, translational/rotational RPE, drift per traveled distance, lost-track time, recovery/relocalization failures, latency and peak memory. Use position + yaw (**PosYaw**) alignment for gravity-aligned inertial methods, and a declared **SE(3)** alignment for stereo-only methods. Preserve metric scale in both cases: fitting a free scale would conceal baseline/calibration errors. Declare the alignment, reference frame and treatment of missing trajectories before test evaluation; do not switch alignment to improve a result. Stereo or LiDAR pose estimation can fail while local depth remains usable; retain that distinction.

Then adapt observations to the existing navigation contract with explicit ray/image-depth conversion and masks. Freeze the gait, goal specification, planner/actor and command filter across sensing arms. Known map goals are allowed; ground-truth robot pose/depth may be read only by the evaluator. Log the exact source of every policy input, and reject adapters that leak oracle state. Maintain simulation time during inference: pausing physics until an answer arrives does not measure delayed control.

**Proposed cohort.** Run a development smoke of 6 layouts × 1 frozen navigation actor × 3 sensing/localization arms × 2 conditions = **36 episodes**, excluded from confirmation. If it passes, freeze a new 24-layout cohort with the same qualified `dr-default-s0` gait controller and the three frozen H1 navigation actors (`armV5-s8`, `armV5-s9`, `armV5-s10`): 24 layouts × 3 navigation actors × 3 arms × 2 conditions = **432 confirmation episodes**. The three actors are not three different gait controllers. Suggested arms are stereo-based estimation, LiDAR-inertial estimation, and gated fusion with its explicitly identified pose estimator. Choose nominal plus one validated timing/dropout stress; hash the schedules. Use fresh seeds distinct from H1, one matched layout/actor/fault schedule per comparison, and no tuning on confirmation failures.

Report goals, falls, collisions, timeouts and distance for every actor/condition, plus paired outcomes by layout. Proposed minimum usability targets are 20/24 nominal goals and 18/24 degraded goals per actor, with zero observed falls and no collision-rate increase relative to the named baseline. These targets do not imply statistical superiority or guaranteed safety. If baselines already have zero collisions/falls, report preservation rather than reduction. Expand the cohort or report inconclusive results when intervals are too wide; do not pool repeated frames/three actors as hundreds of independent environments.

## Dataset, splits and reproducibility

The acquisition manifest should record the following information; this is a proposed recording contract, to be reconciled with the repository's dataset audit tooling before capture.

| Record | Required information |
|---|---|
| Sequence | Unique ID, physical/simulated/derived origin, scene/route/group/split, duration, actual rates, capture software/commit, machine/simulator version, frozen gait/checkpoint hash. |
| Cameras | Original left/right files and hashes; exposure capture timestamps and clock domain; calibration/rectification hashes, dimensions, distortion, intrinsic matrices, baseline, masks, exposure/gain and mount frames. |
| LiDAR | Original scan files and hashes; 2D/3D and scan model; units/frame, scan time, per-point time and available ring/intensity/return flags, calibration, missing-return representation. |
| Motion | IMU gyro/acceleration with units/frames/timestamps, if present; joint states and timestamps; pose estimates with covariance/confidence; all articulation/frame conventions. |
| Evaluation | Ground-truth source and uncertainty, evaluator-only files/masks, independent obstacle labels, layout/fault seeds and hashes, commands/actions/outcomes. |
| Method | Code/checkpoint/config hashes, dependency lock/container, preprocessing, uncertainty thresholds, device, resolution, cold/warm timing and failures. |

Split by entire sequence/scene before extracting frames. Fix data and model selection using development/validation only; retain immutable test manifests. Bootstrap paired differences by independent sequence or layout, preserving repeated actors/fault variants within their group. Publish count denominators, evaluator masks, omitted/failed sequence reasons, environment receipt, and runnable replay commands. Save raw inputs/results outside Git when large, and commit manifest hashes plus compact summaries and failure examples.

For each runtime, measure cold startup separately, then at least 30 warmups and 200 timed iterations on fixed inputs. Include decoding, rectification, transfer, inference, fusion and observation preparation when reporting an end-to-end deadline. Record p50/p95/p99 and the actual hardware; GPU-forward timing alone is not sensor-to-action latency.

## External checks and hardware conditions

| Data/tool | Appropriate use and limitation |
|---|---|
| [Hilti 2023 robot sequences](https://github.com/Hilti-Research/hilti-slam-challenge-2023/blob/main/documentation/hardware/Robot.md) | Real stereo cameras, 3D LiDAR and IMU, including construction/night routes with synchronization/calibration documentation. Useful replay check; a construction robot does not establish humanoid gait performance. |
| [TartanAir](https://theairlab.org/tartanair-dataset/) | Synthetic stereo/depth/pose and diverse visual conditions. Verify the modalities actually downloaded; derived/simulated LiDAR is not a physical scan. |
| [ETH3D SLAM](https://eth3d.ethz.ch/slam_documentation) | Stereo/IMU and documented training ground truth for visual localization. It is not a LiDAR-fusion benchmark. |
| [KITTI depth completion](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) | External depth-completion metrics/splits; driving scenes differ from humanoid indoor geometry. A student implementation should use a held-out training-set split rather than assume leaderboard submission is appropriate. |

Do not substitute [Hilti 2026](https://hilti-trimble-challenge.com/dataset-2026) for a raw stereo–LiDAR fusion dataset: its released camera/IMU data excludes the LiDAR used for reference trajectory generation.

With **2D LiDAR**, start planar obstacle/navigation fusion and explicitly report the scan plane; overhangs, steps outside that plane and 3D foot-placement geometry remain unobserved. With **3D LiDAR**, S4 can build local elevation, surface normal, roughness and visibility/unknown layers. Compare geometric sensor maps with ideal evaluator maps before learning terrain costs. [Elevation Mapping](https://github.com/leggedrobotics/elevation_mapping_cupy) provides a legged-robot reference; use its [current branch/platform documentation](https://leggedrobotics.github.io/elevation_mapping_cupy/getting_started/installation.html) rather than assuming legacy ROS1 and ROS2 instructions are interchangeable.

Later, [Wild Visual Navigation, 2025](https://link.springer.com/article/10.1007/s10514-025-10202-x) motivates visual traversability supervised by proprioceptive experience. Test geometry-only versus geometry + frozen visual features before online learning. Contacts, slip, tracking error or interventions can supply labels, but a geometrically flat surface is not automatically high friction. A controller without stair/step capability cannot validate a perception-only claim of stair traversal.

## Compute order and resume evidence

1. **CPU first:** inventory/capture audit, calibrated SGBM, projection and timing replay, manifest integrity, geometry tests and small motion clips. Schedule no new PPO training for this phase.
2. **One isolated GPU inference environment:** process the frozen image corpus with one current pretrained stereo model; benchmark before adding further models or TensorRT export. Estimate full cost from a 100-pair smoke and actual capture counts. Do not overwrite the existing replay/Isaac environments.
3. **Offline estimator replay, then simulation confirmation:** prove pose/data compatibility before allocating the 432-episode proposal. Pilot timings determine the requested wall time; no duration or GPU budget is promised here.
4. **Later learning/hardware:** a small confidence/traversability model, broader real captures, or full terrain RL only after the simple baselines reveal a repeatable failure mechanism and an independent test set is available.

Resume-ready outcomes must describe work actually completed: synchronized recording/frame/scan counts and audit pass rate; depth or obstacle metrics versus a named baseline on held-out scenes; drift on identified sequences; exact paired goal/fall/collision counts; p95 latency on named hardware; and reproducible artifacts. Use placeholders until measurement, for example: “Evaluated [N] held-out stereo–LiDAR sequences; reduced [defined error] from [baseline] to [candidate], with [coverage] and [p95 latency] on [device].” A completed negative ablation is valid research evidence. Do not turn targets, upstream paper numbers, simulator oracle-depth results or future hardware access into claimed achievements.
