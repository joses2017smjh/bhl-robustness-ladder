# Physical rig plan: RPLIDAR C1, stereo pan/tilt and Yahboom IMU

The user identified this hardware on October 11, 2026 and confirmed that only the camera moves; the IMU and C1 are fixed at the front/top/middle of the body. The [Quest repository audit](QUEST_RIG_INTEGRATION_AUDIT_2026-10-11.md) identifies an N150 Linux computer, stereo USB layout and command-only servo firmware. Current remote access, raw recordings, exact device revisions and measured mounting geometry remain unknown.
This is an implementation plan and experiment backlog, not measured hardware
performance or a frozen scoring protocol. [Machine-readable intake plan](../protocols/hardware-c1-stereo-yahboom-20261011.json).

## Device fit and verified limits

| Device | Evidence | Role in this project |
|---|---|---|
| WayPonDEV RPLIDAR C1 | SLAMTEC specifies planar scanning, 5,000 range samples/s, typical 10 Hz rotations, UART 460800 and nominal ±30 mm range accuracy. Target reflectivity affects specified range. These are vendor specifications, not our measurements. | Planar obstacle sensing and 2D mapping/localization; sparse stereo comparison within its observed plane. |
| MMlove USB stereo camera on two pan/tilt servos | User-provided listing advertises 1200P/60 FPS, 600P/120 FPS, global shutter and 94° view. Exact manufacturer documentation, per-eye dimensions, baseline and exposure synchronization have not been established. | Metric stereo after calibration; camera-only SLAM; stereo terrain perception; later active pointing. |
| Yahboom 10-axis IMU in case | Yahboom documents accelerometer, gyro, magnetometer, pressure, attitude and on-chip time; configurable output up to 200 Hz. Its older tutorial lists 10 Hz by default and USB for the cased model. Exact product revision and firmware must be identified. | Measured gyro/acceleration for inertial estimation after unit, noise, clock and mounting checks. AHRS attitude and pressure are auxiliary channels. |

Sources: [SLAMTEC C1 specification](https://www.slamtec.com/en/c1/spec),
[Yahboom module documentation](https://www.yahboom.net/public/upload/upload-html/1656403520/Introduction%20to%20the%20IMU%20module.html)
and [current Yahboom product family](https://category.yahboom.net/products/imu).
The camera's advertised frame rate does not establish processing throughput,
per-eye resolution or left/right exposure synchronization. Global shutter also
needs separate confirmation of simultaneous stereo exposure.

The existing FAST-LIO2 campaigns use timed **3D** scans. C1 observations do not
qualify that hardware route by placing a planar scan into a PointCloud2 message.
Start a separate planar track; retain the existing 3D intake and original
simulation labels. Space above and below the C1 scan plane remains unobserved:
a free planar ray must not clear an entire body-height column in a 3D obstacle map. [FAST-LIO input requirements](https://github.com/hku-mars/FAST_LIO).

## Confirmed mounting and moving-head path

**Selected layout: camera moves; IMU and LiDAR stay on the body.** Start with both servos locked at a documented pose for calibration and the fixed-head baseline. Active camera pointing remains the intended follow-up. The alternative layouts below explain estimator constraints; they are not claims about this robot. See the [source-backed integration design and stationary-body control](QUEST_RIG_INTEGRATION_AUDIT_2026-10-11.md#selected-moving-head-architecture).

| Physical layout | Consequence |
|---|---|
| Camera moves; IMU stays on body | Camera–IMU transform changes with head motion. Stock fixed-extrinsic stereo–IMU inference is not a valid moving-head baseline. Keep the head locked for that baseline, or implement and validate articulated visual–inertial estimation. |
| Camera and IMU move rigidly together | Camera–IMU calibration can remain fixed. The estimated head pose still needs a measured, time-aligned head-to-body transform before commanding body navigation. |
| Camera, IMU and C1 all move together | Relative sensor mounting can stay fixed, but a tilted C1 changes the scan plane. A horizontal 2D mapper's assumptions need separate validation. |

This follows from [ORB-SLAM3's fixed camera–IMU calibration](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/Examples/Stereo-Inertial/EuRoC.yaml).
The inspected Arduino firmware exposes **commanded pulses, not measured angles**. `CameraAim`
history is an approximate command estimate, not measured kinematics. Record feedback when
available; otherwise characterize repeatability, backlash and settling at each
fixed pose and retain that uncertainty. Do not silently equate command time with
physical exposure-time angle. Moving only the camera does not turn the C1 into a
3D scanner. A later mechanically swept-LiDAR study would require the LiDAR itself
to move, measured angles/timing and a separately qualified motion model.

For body-frame navigation, define transforms explicitly: if `T_W_C` maps camera
to world and `T_B_C(q)` maps camera to body, then
`T_W_B = T_W_C @ inverse(T_B_C(q))`. Use the joint state at the camera acquisition
time and include mounting translations, not just pan/yaw rotation.

Body-mounted sensors still tilt with a walking humanoid. Fixed mounting does not make the C1 scan plane permanently horizontal; test body roll/pitch and gait motion separately from the level planar baseline. Exact body-to-IMU/LiDAR offsets remain to be measured.

## Acquisition and calibration work

1. Identify the connected machine, operating system, device IDs/firmware, actual
   USB image layout and available modes. On Linux, `lsusb`, `v4l2-ctl --list-devices`,
   `v4l2-ctl -d /dev/videoN --list-formats-ext` and `/dev/serial/by-id/` are useful
   read-only inventory inputs. Replace `videoN` with the discovered device. No
   serial configuration, servo motion or robot actuation has been performed.
2. Measure delivered camera/IMU/scan rates, dropped frames, timestamp ordering and
   host latency. Record acquisition/device time and host receipt time separately,
   with clock source, offset/drift model and uncertainty. Unknown exposure sync
   stays unknown until measured or supported by device documentation.
3. Calibrate both camera intrinsics, lens distortion and stereo baseline in the
   chosen image mode. Use separate calibration and held-out target captures.
   Calibrate stereo-to-C1 geometry from common visible targets; C1 ranges are a
   noisy reference restricted to its scan plane, not complete depth ground truth.
4. Calibrate camera–IMU spatial/time offset with the head locked or the camera and
   IMU rigidly co-mounted. Estimate gyro/accelerometer noise and bias from static
   recordings, then verify axes, signs and SI units with controlled motion.
   Preserve measured acceleration/gyro and the device's AHRS outputs separately.
   The fused quaternion is not an independent reference trajectory.
5. Characterize servo/motor interference with magnetometer/heading while the rig
   is stationary. Compare gyro/accelerometer-only estimation with magnetic aiding
   separately. Pressure is auxiliary; it is not centimetre terrain-height ground
   truth.

[Kalibr's procedure](https://github.com/ethz-asl/kalibr/wiki/camera-imu-calibration)
provides a reproducible calibration starting point. Its model and the selected
rig's timing contract must match; no EuRoC calibration/noise constants are copied
into the physical rig as if measured.

## Planar recording contract to implement after device inspection

A separate adapter must preserve stereo images/calibration, original LaserScan
ranges/angles/quality fields, raw IMU packets and decoded SI channels, camera and
host clock metadata, mounting configuration, and measured joint states where
available. Keep original data hashes, driver commit/configuration, USB layout,
exposure settings and calibration/reference evidence. Store dropped/missing data
explicitly. A bounded initial recording should use HPC-share storage after upload,
with measured byte rate and free-space checks; do not assume compressed video size.

The inspected [official C1 launch](https://github.com/Slamtec/sllidar_ros2/blob/main/launch/sllidar_c1_launch.py)
uses 460800 baud and enables angular compensation. The inspected [ROS2 driver](https://github.com/Slamtec/sllidar_ros2/blob/main/src/sllidar_node.cpp)
uses host acquisition-call timing, derives `time_increment` from scan duration,
sorts/resamples ranges, and fills `intensities` from quality values. Preserve
those semantics: these fields do not by themselves prove hardware per-point
exposure timestamps or calibrated reflectivity. Pin the actual installed driver
before building a timing-sensitive adapter. If individual sample timing is
estimated, retain the model/uncertainty and label it estimated.

The previous `sensor_validation.py --physical` route requires genuine timed 3D
inputs and **does not currently implement this planar contract**. Its guards stay
intact. Sensor identification alone cannot return hardware-validation complete.

## Ordered research backlog and numbers to measure

All rows are **UNRUN on this rig**. Counts, thresholds, scene splits and execution
budgets will be frozen after acquisition/calibration checks, before scored trials.

| Task | Matched experiment | Results to report |
|---|---|---|
| RIG-01: calibration and data integrity | Locked head; repeated stationary and controlled-motion captures | Delivered Hz, stereo skew/uncertainty, loss %, timing jitter, reprojection/epipolar residuals, bias/noise, calibration repeatability |
| RIG-02: metric stereo | SGBM vs Fast-FoundationStereo at the same measured image mode; surveyed targets and C1 scan-plane overlap | Depth MAE/RMSE, common-support error, coverage, obstacle recall, false-free rate and full-pipeline p95 latency |
| RIG-03: planar localization | Planar LiDAR SLAM vs stereo; separately validated planar+IMU and rigid stereo–IMU arms | Independent-reference ATE/RPE where available, tracking availability, initialization/reset counts, recovery time, latency; loop closure error labelled separately from ATE |
| RIG-04: complementary obstacle sensing | C1 alone vs stereo alone vs simple vs confidence-gated fusion | Recall by obstacle height/range/material, false-free and unknown rates, missed low/overhanging obstacles, time to detection, coverage and compute cost |
| RIG-05: active camera pointing | Fixed forward vs predetermined sweep vs causal uncertainty/information-gain pointing | Hazard recall, time to detection, tracking interruptions/resets, head travel, body-pose error and p95 latency including settling; match route, speed and compute budgets |
| RIG-06: terrain perception | Measured blocks/ramps with stereo elevation/slope maps; varied head poses only after transform validation | Height/slope error, hazard precision/recall, unknown coverage and latency; physical traversal is a subsequent separately supervised experiment |
| RIG-07: calibration/fault sensitivity | Fixed recorded streams with declared time/extrinsic/exposure/dropout perturbations; separate physical fault captures where feasible | Degradation curves, false accepts/rejects, reset frequency and recovery time; distinguish offline perturbation from actual acquisition faults |

RIG-02 follows the existing [Fast-FoundationStereo implementation](https://github.com/NVlabs/Fast-FoundationStereo).
For RIG-03, [SLAM Toolbox](https://joss.theoj.org/papers/10.21105/joss.02783)
is a planar baseline candidate; its implementation and any IMU augmentation must
be pinned and benchmarked separately. Its [odometry transform input](https://github.com/SteveMacenski/slam_toolbox/tree/ros2) must have a declared real source. No wheel odometry has been identified for this rig; choose a scan-matching front end or another appropriate planar pipeline before calling an arm LiDAR-only. Do not supply undeclared stereo odometry or a fictitious static body transform. Do not assume adding IMU improves results.
RIG-05 is motivated by [active perception for navigation](https://arxiv.org/abs/2602.01266)
and [active-camera multi-object navigation](https://arxiv.org/abs/2210.07505).
These motivate a new comparison; this project does not claim to reproduce those
papers' methods or results.

Use an independent reference for metric accuracy: surveyed targets/known geometry
for depth and a qualified external trajectory reference for pose. Compare paired
runs on held-out routes, preserve failed initialization and unknown regions, and
report denominators and repetition counts. Do not pool correlated video frames
as independent trials. All numerical improvements remain unmeasured until these
experiments run. A resume can currently describe designing this validation plan;
it cannot claim physical accuracy, successful navigation or traversal from it.
