# Stereo and lidar implementation and experiment queue

The four research routes have executable pilot stages and explicit follow-up
dependencies. This campaign captures real **rendered stereo RGB**, computes
disparity with SGBM and an official pretrained stereo model, and evaluates
projected fusion and 3D maps. Earlier paired ray-depth logs are not used as RGB.
The captured rig follows a known kinematic path; it does not execute humanoid
gait, estimated-pose navigation or physical hardware trials.

**Pilot execution completed:** smoke `21714137` and full pilot `21714203` both
PASS on a Quadro RTX 8000. The [independently checked report](SENSOR_PILOT_RESULTS_2026-10-08.md)
and [depth/terrain demos](../results/sensor-campaign-20261008/README.md) contain
the actual measurements. Confidence gating does not meet the proposed fusion
advancement target; native estimator execution and control trials remain open.

| Task | Implemented pilot | Full-route dependency |
|---|---|---|
| S0 capture and intake | Calibrated 320×240 RGB pairs, timed 3D ray scans, 200 Hz ideal IMU, independent depth/obstacle/terrain truth; file hashes and whole-scene splits | Physical sensor models, calibration, frame mounts and synchronization; longer independent routes |
| S1 stereo depth | SGBM versus pinned official **C-Fast-FoundationStereo ONNX**, native/common-mask MAE/RMSE/AbsRel, labeled obstacle pixel recall, coverage, cold/warm p50/p95/p99 | Larger held-out sequence cohort and a selected robot planner deadline before improvement qualification |
| S2 sensor fusion | Each sensor alone, arithmetic projection, consistency/confidence gate, conservative minimum; both stereo frontends, matched retention and extrinsic faults | Estimated-motion deskew, camera/lidar time-offset studies and separate confirmation; optional author-code comparison |
| S3 estimated pose | ORB-SLAM3 stereo input export, FAST-LIO2 timed-cloud/IMU replay, strict pose scoring and estimated-input navigation interface | **BLOCKED_RUNTIME:** native ORB-SLAM3 and ROS/FAST-LIO2; longer validation sequences; real estimator outputs and closed-loop integration |
| S4 3D terrain | Elevation, slope, roughness, supported/unknown hazard maps, independent height/hazard measurements | Paired traversal with a controller qualified for the chosen terrain; no traversal result yet |

The completed full pilot contains **72 stereo pairs**, 24 per scene, at 15 Hz. These are
three scene groups, with one development, one validation and one test group.
Repeated frames, fault variants and stereo methods are not independent scenes.
The short sequences establish data and software readiness; they cannot support
a generalization or SLAM drift claim. A separate smoke uses two frames per scene
and is excluded from scored pilot evidence.

The [canonical protocol and queue receipts](../results/sensor-campaign-20261008/README.md)
record immutable inputs and actual Slurm status. The pilot only starts after a
fresh successful smoke of the same archive. Source, model, wheel, runtime SIF,
calibration and output hashes are retained. One additional inference GPU is
allowed at a time; existing H3/H4 training sources and jobs are untouched.

## Method and data details

The official [Fast-FoundationStereo implementation](https://github.com/NVlabs/Fast-FoundationStereo)
links the [C-Fast-FoundationStereo model release](https://huggingface.co/nvidia/c-fast-foundationstereo).
This pilot uses its official dynamic-size ONNX checkpoint, not the serialized
research checkpoint. Model revision `53a84e28a368f37e699776250215fcb450d6a109`,
SHA256 `2c79bbb274dc0aef687a8732b987077e9e3aa26ff9dd653141cda48c57dad5d3`.
The model is 103,228,473 bytes; input padding preserves pixel disparity and camera
intrinsics. GPU execution requires actual profiled CUDA kernels. CPU shape/control
operators are allowed and recorded; silent whole-model CPU fallback is rejected.

The isolated runtime is `onnxruntime-gpu==1.22.0`, a pinned wheel unpacked into
private node scratch. Its CUDA/cuDNN dependencies come from the existing
read-only NVIDIA packages. This follows the runtime's
[documented CUDA library preloading](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls).
No training environment package is installed or changed.

RGB images and raw lidar/IMU inputs live under `inference/`. Independent depth,
non-floor obstacle labels and dense vertical terrain queries live under
`evaluator/`. Camera depth is optical-axis Z in meters, not slant range.
`T_A_B` maps B coordinates into A; lidar projection uses a nearest-pixel Z-buffer.
Capture ground labels come from simulator geometry segmentation. Obstacle pixel
recall is not an object detector or a traversal safety guarantee.

The fusion gate uses fixed engineering thresholds, not a calibrated probability.
It preserves unsupported pixels as NaN and reports unknown/missed obstacles with
false free space. Each frontend uses the same nested 100/50/20% point-retention
masks and nominal, 1-degree/1-cm and 3-degree/3-cm calibration perturbations.
All individual frames and denominators are retained. One pilot pass measures
fusion and terrain compute time; it is not a warm real-time control benchmark.

The [Yao, Ishikawa and Oishi RA-L paper](https://arxiv.org/abs/2504.05148)
combines discrete disparity-matching costs, lidar disparity semidensification
and consistency checks. Its [author code](https://github.com/yshry/libSGM_lidar)
is a separately isolated optional reference. Our transparent projection/gating
baselines are informed by consistency checking; they do not reproduce that
algorithm. The fifth arm is a conservative minimum baseline, not author code.

The lidar scan has 16 vertical rings and 64 columns spanning a frontal 160-degree
field, with actual per-point timestamps across 50 ms. It is an ideal ray sensor;
there are no measured intensity/material/weather effects. The current mapping
uses the rigid mount without estimated-motion deskew. Its distortion/error is
retained, rather than corrected with evaluator poses. Spatial uncertainty,
moving-scan deskew and hardware return physics need separate experiments.

Terrain maps preserve unknown cells and evaluate geometric hazards against
independent height labels. Obstacles can contaminate cell medians; geometric
flatness does not establish friction or gait capability. The legged-robot
[ElevationMappingCuPy implementation](https://github.com/leggedrobotics/elevation_mapping_cupy)
is a reference for later probabilistic mapping, not a dependency claimed to be
running here.

## Follow-up task queue

1. **S0-HARDWARE:** inventory stereo/lidar/IMU models, 2D versus 3D coverage,
   calibration, clock domains and mount frames. Audit existing recordings before
   selecting physical claims. Requested hardware details can extend the same
   replay schema.
2. **S1-CONFIRM:** qualify data on development/validation; freeze longer independent
   scenes and then compare the same two models. Proposed depth/coverage gates
   remain targets. This pilot has only one held-out scene, so sequence bootstrap
   correctly reports insufficient evidence for a confidence interval.
3. **S2-DESKEW:** consume timestamped estimated motion, add actual temporal offsets
   and matched dropout bursts, then compare gate with/without deskew. Do not use
   evaluator poses as the motion estimator. Confirm the selected stress after
   threshold selection; baseline zero events forbids a percentage reduction claim.
4. **S3-RUNTIME:** provision pinned native
   [ORB-SLAM3 stereo](https://github.com/UZ-SLAMLab/ORB_SLAM3) and
   [FAST-LIO2](https://github.com/hku-mars/FAST_LIO). Export commands and sensor-format
   requirements are generated by the intake stage. FAST-LIO2 uses lidar **and IMU**.
   This host has no ROS/catkin or installed native stereo executable.
5. **S3-REPLAY:** run real estimators on longer sequences and independently score
   metric-scale-preserving SE(3) alignment for stereo and yaw/translation alignment
   for inertial estimates. Report ATE, translational/rotational RPE, drift per
   distance and tracking failures. Missing poses are not filled with truth.
6. **S3-CLOSED-LOOP:** wire real pose packets into the navigation runner, preserve
   simulation time through compute delays, freeze gait/actor/command filter and
   log input provenance. Run 36 development episodes first; only after readiness
   passes queue the proposed 432 fresh confirmation episodes. Goals, collisions
   and falls are currently **unmeasured** for this route.
7. **S4-TRAVERSAL:** first identify terrain the frozen controller can traverse;
   compare sensing/maps under matched routes and record falls, slips and goals.
   Preserve map coverage and hazard miss rates rather than treating unknown as safe.

These tasks are dependencies in the experiment backlog. The full SLAM and
traversal campaigns are not submitted as jobs while their required runtimes and
controller integration are missing. The Slurm pilot queues their **readiness**
checks and exports, and reports that distinction explicitly.

## Replay commands

```bash
# Run inside an allocated GPU environment; output must be new/private.
python scripts/bench/sensor_capture.py --output replay --frames-per-scene 24
python scripts/bench/stereo_research.py --dataset replay --output stereo \
  --model /path/to/verified/model.onnx --provider CUDAExecutionProvider \
  --warmup 30 --timed 200
python scripts/bench/sensor_geometry_research.py --protocol protocol.json \
  --dataset replay --stereo-predictions stereo/predictions.json \
  --stereo-method sgbm --output geometry
python scripts/bench/pose_research.py intake --dataset replay --output pose
```

The frozen Slurm runner executes these stages and repeats fusion with
`--stereo-method c_fast_foundationstereo`. See the input protocol for wheel/model
download locations and checksums. Model/wheel inputs and frozen launch packs are
kept in durable home storage. Public Git contains the complete 39 MiB pilot output
archive, failed-smoke evidence, data manifests and replay instructions. Resume
metrics require completed, independently checked
outputs; targets, upstream paper numbers and queued jobs are not achievements.
