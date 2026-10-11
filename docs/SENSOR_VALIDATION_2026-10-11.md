# Stereo calibration diagnosis and physical validation intake

The new diagnostic measured 30 original stereo pairs: ten deterministically spaced frames in each of the three archived native-replay scenes. Source images and calibration were SHA256-verified against the original manifest. This is diagnosis on previously consumed simulation data, not independent confirmation.

| Scene | Mutual descriptor matches | Median absolute vertical residual | P95 absolute vertical residual | Frames within the diagnostic limit |
|---|---:|---:|---:|---:|
| Textured boxes | 232 | 0.075 px | 5.790 px | 0/10 |
| Thin posts | 299 | 0.100 px | 6.428 px | 3/10 |
| Ramp/step | 508 | 0.112 px | 3.020 px | 3/10 |

The diagnostic requires at least 20 matches per frame and vertical p95 at most 2 px. Four box frames had insufficient matches. These are engineering limits, not probabilities of estimator success. SIFT matches use a mutual 0.75 descriptor-ratio test without row, disparity or fitted-geometry filtering. The small medians and larger tails indicate descriptor-correspondence outliers may dominate the p95; this measurement alone does not prove a camera calibration defect. The separate original ORB feature-starvation finding remains valid. SIFT features are not presented as native ORB telemetry.

The residual measures consistency with the declared rectified camera geometry. It cannot identify every baseline, focal-length, distortion or camera–LiDAR extrinsic error; full calibration requires target/landmark observations and independent geometric references. The method follows the pixel-correspondence formulation in [OpenCV's epipolar geometry documentation](https://docs.opencv.org/4.x/da/de9/tutorial_py_epipolar_geometry.html), retaining unfiltered vertical residuals so an actual row offset remains measurable.

Run on a declared replay manifest:

```bash
PYTHONPATH=src python scripts/bench/sensor_validation.py \
  --manifest /recording/manifest.json --frames-per-sequence 10 \
  --output /new/diagnostic.json
```

[Measured evidence](../results/sensor-validation-20261011/diagnostic.json) contains all frame counts, residuals and original input checksums. A positive-control test injects a five-pixel vertical image shift and verifies it remains detectable; textureless or insufficient matches cannot pass.

## Physical sensor validation

No accessible physical stereo/LiDAR/IMU rig has been identified. The current compute node has no local video devices or ROS command-line interface. This is a bounded [local access inventory](../results/sensor-validation-20261011/hardware-access.json), not a claim that every remote machine has been searched. The user has been asked for the robot/sensor models, machine address or recording path. Physical validation remains **blocked on access**, and no simulation result is relabeled as hardware evidence.

The executable intake path is ready:

```bash
PYTHONPATH=src python scripts/bench/sensor_validation.py \
  --manifest /physical-recording/manifest.json --physical \
  --output /new/physical-intake.json
```

The manifest uses the existing `bhl-sensor-replay-v1` contract: calibrated rectified PNG pairs; explicit device-clock timestamps; hashes for every original inference input; timed 3D LiDAR XYZ, original intensity, rings and per-point times; finite SI-unit IMU samples spanning every scan. Original LiDAR point times or intensity cannot be synthesized for physical inputs. Each sequence declares `origin: physical`, grouping and data split.

Additional `physical_rig` fields are `robot_or_rig`, `left_camera`, `right_camera`, `lidar`, `imu`, `capture_date_utc`, `clock_sync_evidence` and `calibration_evidence`. The two evidence fields name existing files under `inference/`, included in the manifest's SHA256 map. Every stereo frame also records `left_timestamp_s` and `right_timestamp_s`; their measured difference must be at most 1 ms for this intake route. Both camera streams and canonical frame times must increase strictly, and the stereo midpoint must agree with canonical time within 1 ms. Original device timestamps therefore need a documented mapping to the declared common clock before intake. These limits must be revisited through a new protocol if the actual rig has a different timing contract.

Intake can return `READY_FOR_NATIVE_REPLAY`; it always leaves `hardware_validation_complete: false`. Device declarations and hashed calibration records still need independent inspection. Physical completion requires actual native replay, external pose/depth reference evaluation, and separately supervised physical navigation/traversal trials with recorded goals, collisions, falls and latency. Public [EuRoC recordings](https://projects.asl.ethz.ch/datasets/euroc-mav/) could supply real stereo/IMU replay, but lack the timed mobile 3D LiDAR stream for the complete rig and do not establish this robot's physical performance.

## Outcome-associated health screen

`scripts/bench/outcome_health.py` implements an additional preregistered offline comparison for the new camera/fault campaign. It fits each eligible arm on reset seed 730000 only; seed 730001 is evaluated against a separately written, checksum-locked model. It selects from fixed tracking-fraction thresholds by balanced outcome classification, retaining a 0.9 engineering threshold and accept-all comparators. A negative outcome means failure to complete a clean full-horizon goal; it is not synonymous with collision or unsafe pose.

Each included arm requires both routes and every condition. Fewer than two outcomes in either class produces `INSUFFICIENT_OUTCOME_VARIATION`, not an invented calibrated model. The evaluation reports false acceptance, clean acceptance and observed adverse-outcome fractions, including adverse episodes with high tracking. No result is claimed until the full eligible split is collected.

This screen uses an episode's final tracking fraction. It tests association with outcomes offline, not a causal online intervention or calibrated probability of safety. There is only one held-out reset seed and two reused route families; dependent episodes do not support population confidence intervals. Online health-triggered recovery remains a separate experiment after this diagnostic has evidence.
