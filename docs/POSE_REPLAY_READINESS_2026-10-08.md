# Estimated-pose route: executable inputs and measurement boundary

The route now has native input exporters, a metric-scale trajectory scorer and
an estimated-pose policy boundary. The queued kinematic sensor pilot can run
the exporters. It cannot close the ORB-SLAM3 versus FAST-LIO2 comparison or the
closed-loop navigation task. Its short sequences and ideal analytic IMU are
declared limitations; no estimator outputs or navigation episodes have been
manufactured.

## Primary research and pinned implementation

The stereo arm is pure stereo ORB-SLAM3; adding IMU creates a separate arm.
The [ORB-SLAM3 paper](https://arxiv.org/abs/2007.11898) describes its visual,
inertial and map-management methods. This project uses the [official code at
4452a3c](https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4)
and requires its compiled C++ library, OpenCV, Eigen and Pangolin. ROS is
optional for the stereo executable. The vocabulary and compiled runtime must
be hashed separately. Those assets are not included in the current Python
environment.

The lidar–IMU arm is FAST-LIO2. Its [paper](https://arxiv.org/abs/2107.06829)
describes direct raw-point registration and incremental map maintenance.
The [official implementation at 7cc4175](https://github.com/hku-mars/FAST_LIO/tree/7cc4175de6f8ba2edf34bab02a42195b141027e9)
requires ROS, PCL, Eigen and the Livox driver build dependency. The pinned
`ikd-Tree` submodule must be preserved when building an isolated runtime.
Point timestamps and calibrated lidar–IMU synchronization are required by
the declared replay route. It never estimates missing point times from yaw or
substitutes zero-time scans.

Native-input details were checked against the pinned
[ORB stereo reader](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/Examples/Stereo/stereo_euroc.cc),
[ORB settings parser](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/src/Settings.cc),
[FAST-LIO point structure](https://github.com/hku-mars/FAST_LIO/blob/7cc4175de6f8ba2edf34bab02a42195b141027e9/src/preprocess.h)
and [FAST-LIO configuration](https://github.com/hku-mars/FAST_LIO/blob/7cc4175de6f8ba2edf34bab02a42195b141027e9/config/velodyne.yaml).
The ORB export preserves nanosecond image names and original image capture
times. It selects `Rectified` cameras and an integer nominal frame rate.
The FAST export preserves raw 3D returns, seconds from each scan header, ring
indices, gyro in rad/s and specific force in m/s². Its extrinsic is
`T_I_L = inverse(T_B_I) T_B_L`, with the IMU as the native base frame.
The ideal simulated ray scans have no intensity measurement. Their ROS
packaging alone inserts the required zero field, explicitly marked
`intensity_kind: unmeasured_zero_placeholder_for_ROS_field` in packets and the
adapter receipt. Original scan files remain unchanged. Missing physical or
external intensity is rejected rather than silently assigned a value.

## Run the intake on a captured replay

```bash
PYTHONPATH=src python scripts/bench/pose_research.py intake \
  --dataset /path/to/replay --output /path/to/new-pose-intake \
  --protocol /path/to/frozen-protocol.json
```

All declared inference-image, raw-scan and IMU hashes are mandatory. The
reader rejects path traversal, symbolic links, evaluator paths, unexpected
pose arrays inside sensor files, incorrect resolution, nonmonotonic capture
times, invalid rigid transforms, untimed scans, IMU gaps above 20 ms, and
cross-split scene reuse. It never reads the frame's `truth` file during
estimator preparation.

Each sequence produces:

| Artifact | Purpose |
|---|---|
| `stereo_slam/mav0/cam{0,1}/data/*.png` | Original stereo images in the native EuRoC directory shape |
| `stereo_slam/timestamps.txt` and `stereo.yaml` | Native timestamps and calibrated rectified camera settings |
| `lidar_imu/packets.jsonl` and `fastlio.yaml` | Original timed clouds and high-rate IMU with native topic/field settings |
| Both `adapter.json` files | Upstream pins, input/output hashes, origin and exact runtime prerequisites |
| Intake `campaign_result.json` | Execution readiness and separate scientific status |

Valid exports report execution `PASS`, scientific `BLOCKED_RUNTIME` and zero
estimated-pose outputs/closed-loop episodes. Invalid sensor inputs report
`BLOCKED_CALIBRATED_DATA`. Short pilot sequences remain
`PILOT_ONLY_SHORT_SEQUENCE`. A directory of valid exports is not a SLAM result.

The optional `publish-fastlio` command requires an existing ROS1 environment
and subscribers. It publishes `sensor_msgs/PointCloud2` and `sensor_msgs/Imu`
with their original capture stamps, paced in wall time. Missing ROS is recorded
as `BLOCKED_RUNTIME`; no stand-in odometry is produced. Stock ORB's stereo
example opens a viewer, so a validated headless frontend or a display is still
needed before a cluster replay is submitted.

## Score actual estimator outputs

The evaluator accepts `bhl-pose-trajectory-v1` JSON containing `sequence`,
`clock_domain`, `sensor_frame`, and one frame record per original capture:
`timestamp_s`, rigid `T_W_S` and explicit boolean `tracked`. Lost frames must
remain in the tracking record, using a finite unused pose placeholder. Stock
ORB trajectory files omit lost frames, so they alone cannot support tracking
failure metrics. A frontend or separate tracking-state log is required.

The companion `bhl-pose-estimator-run-v1` receipt records `method` as
`orb_slam3_stereo` or `fast_lio2_lidar_imu`, the pinned `upstream_commit`,
`runtime_sha256`, `input_manifest_sha256`, `calibration_sha256`, `config_sha256`,
`trajectory_sha256`, original `input_files_sha256`, and
`ground_truth_inputs: []`. It is an execution receipt, not an input-adapter
receipt. All file hashes are SHA-256 hex strings.

```bash
PYTHONPATH=src python scripts/bench/pose_research.py evaluate \
  --estimate native-estimator-trajectory.json \
  --truth independent-reference-trajectory.json \
  --receipt native-estimator-run.json --out pose-metrics.json
```

Estimate and reference must use the same clock and represented sensor frame.
ORB visual poses represent the left camera. FAST-LIO state poses represent
the IMU body; use the calibrated mount to represent the same frame as the
reference. Clock association is nearest, monotonic and one-to-one. The scorer
reports ATE, rotation error, consecutive-pair RPE with actual time intervals,
endpoint drift per reference distance, tracking failure segments and lost
time. Unassociated tracked samples remain visible rather than disappearing
from the denominators.

Stereo uses rigid SE(3) alignment; lidar–IMU uses yaw and translation only,
preserving gravity and exposing roll/pitch errors. Scale is fixed at one.
Collinear rigid-alignment cases use first-pose orientation and a fitted
translation, with that fallback disclosed. Stationary sequences have an
undefined drift-per-distance ratio. A single scored replay remains
`MEASURED_SINGLE_REPLAY_NOT_COMPARISON`.

## Closed-loop interface and remaining experiments

`PosePacket`, `EstimatedPoseGate` and `estimated_navigation_observation` in
`src/bhl_robust/research/pose_metrics.py` substitute estimated x/y/yaw in the
existing frozen NavGym observation builder. Capture and arrival times remain
distinct. Lost tracking, low confidence, stale captures, future/unarrived
packets and out-of-order outputs produce a zero body command before policy
inference. A new lost packet invalidates older confident outputs. Missing
scan returns remain unknown and cannot be replaced by maximum-range free
space. The adapter has no simulator-state or evaluator argument.

The caller still must integrate its occupancy/visitation maps using estimated
poses and place known goals in the estimator's map frame through independent
calibration/localization. The scorer's ground-truth alignment transform must
never be reused to anchor policy goals. `calibrated_body_pose` only applies a
known rigid sensor mount; it does not perform world registration. Physics must
continue while estimator inference is late, so measured delays affect command
age rather than pausing the simulation.

After runtime acceptance, capture the previously declared longer independent
scene/route sequences, freeze estimator configs before validation/test, and
run stereo versus lidar–IMU replay. Only successful replay acceptance enables
the 36-episode navigation development cohort; the separate 432-episode H1
actor confirmation cohort follows the development gate. Keep the frozen
12-DoF gait, three H1 NavGym actors, command filter, goal/collision/fall
instruments, layouts and fault seeds matched across arms. These counts remain
planned workload, not completed outcomes.

## Read-only local audit

On October 8, existing `navgym-v4-capture-pose-20261001` outputs contained
control traces with x/y/yaw and sensor summaries, rather than original
calibrated stereo images or raw timed 3D scans. Existing `results/terrain`
contained CSV locomotion results. The separately running `hum-terrain` jobs
invoke `locomotion_experiment.py`; `hum-nav-predict` explicitly retains oracle
pose/goal and simulated truth IMU. Those jobs do not supply this route's raw
sensor corpus. The current host has no `roscore`, `roslaunch` or `catkin_make`.
This audit did not alter any of their files or running jobs.
