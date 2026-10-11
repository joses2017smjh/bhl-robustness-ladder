# Quest teleoperation repository: hardware integration evidence

Read-only review of GitHub commit
[`18f658bd6cf19857535f9be593067635501c4b66`](https://github.com/joses2017smjh/quest-vr-teleop/tree/18f658bd6cf19857535f9be593067635501c4b66),
October 11, 2026. No service, serial device or actuator was started. These are
code/configuration findings, not measurements of currently connected hardware.

The user separately confirmed that **only the stereo camera moves on pan/tilt**.
The IMU and C1 remain fixed at the front/top/middle of the robot body. Their exact
translations/orientations and the pan/tilt pivot geometry still require measurement.

## Questions the repository answers

| Question | Evidence and implication |
|---|---|
| Onboard computer | [VR documentation](https://github.com/joses2017smjh/quest-vr-teleop/blob/18f658bd6cf19857535f9be593067635501c4b66/docs/VR_TELEOP.md) identifies an Intel N150 Linux PC. [As-built notes](https://github.com/joses2017smjh/quest-vr-teleop/blob/18f658bd6cf19857535f9be593067635501c4b66/arm_validation/TELEOP_AND_ARM_BRINGUP.md) describe a Beelink EQ, Ubuntu 22.04 context and a later 6.12 kernel. Current OS/runtime was not remotely inspected. |
| Camera transport | [camera_share.py](https://github.com/joses2017smjh/quest-vr-teleop/blob/18f658bd6cf19857535f9be593067635501c4b66/scripts/teleop/camera_share.py) documents USB ID `32e4:2b10`, side-by-side MJPEG, and a default request of 1280×480 at 30 FPS: two 640×480 eye images. These are configured values, not verified delivered rates or exposure synchronization. |
| Head control | [servos.py](https://github.com/joses2017smjh/quest-vr-teleop/blob/18f658bd6cf19857535f9be593067635501c4b66/scripts/teleop/servos.py) sends pulse commands through an Arduino Nano at 115200 baud; D9 is pan and D5 tilt. It already supports operator tracking, Quest-head following and holding still. |
| Angle feedback | [Arduino firmware](https://github.com/joses2017smjh/quest-vr-teleop/blob/18f658bd6cf19857535f9be593067635501c4b66/scripts/arduino/claw_controller/claw_controller.ino) explicitly reports commanded pulses, not encoder feedback. Software `currentUs` is a slew-limited output command, not a physical position reading. |
| LiDAR/IMU transport | [sensor_share.py](https://github.com/joses2017smjh/quest-vr-teleop/blob/18f658bd6cf19857535f9be593067635501c4b66/scripts/teleop/sensor_share.py) has a C1 standard-scan reader at 460800 baud and a WitMotion-format IMU parser at 9600. The Yahboom product/revision must be matched to the actual device; the parser's label alone does not establish identity. |
| Existing depth display | [depth_flow.py](https://github.com/joses2017smjh/quest-vr-teleop/blob/18f658bd6cf19857535f9be593067635501c4b66/scripts/teleop/depth_flow.py) runs Depth Anything V2 separately on the eye images and displays relative depth. It does not compute calibrated metric stereo disparity. |
| Existing outputs | Latest display files are `/dev/shm/bhl_camera.jpg`, `/dev/shm/bhl_lidar.json`, and `/dev/shm/bhl_imu.json`. [Headset recording](https://github.com/joses2017smjh/quest-vr-teleop/blob/18f658bd6cf19857535f9be593067635501c4b66/scripts/teleop/record_headset.py) saves rendered Quest video under `~/Videos/bhl/`; it is not a raw synchronized sensor recording. |
| Remote access | The inspected public documentation uses `<pc-ip>` for the web interface. A current reachable SSH endpoint and qualified raw recording directory were not identified in the inspected source. |

## Bounded local recording check

A separate local checkout at `/nfs/hpc/share/sanchej7/quest-vr-teleop`, commit
`768d28eefe0172756e71b153506955f50d09c9a8`, contains `recorder.py` and
`LFD_RECORDING_FORMAT.md` that are absent from the reviewed GitHub commit. That
recorder captures arm state/actions and stereo JPEGs; it does not retain raw
LiDAR/IMU streams. Its image timing uses file-write modification times and its
camera angles are commands. It does not yet supply the synchronized inputs
required here. These local additions were neither changed nor published by this
audit.

The documented HPC directories `/nfs/hpc/share/sanchej7/bhl-data` and
`/nfs/hpc/share/sanchej7/bhl_recordings` were absent. Bounded checks of
`tools/lfd`, `arm_validation` and `configs` found 27 arm teleoperation logs but no
qualified stereo/LiDAR/IMU recording or stereo calibration file. This does not
rule out recordings on the robot or elsewhere; their location remains needed.

## What must change for quantitative sensor experiments

1. **Record original inputs before display reduction.** The camera path overwrites
   one JPEG. The C1 path caps display writes at about 5 Hz, rounds and subsamples
   range/angle pairs, and does not retain original quality/timing packets. IMU
   display writes are capped around 10 Hz and combine the latest decoded channel
   values, which need not be simultaneous. The existing code is useful for
   visualization but cannot reconstruct discarded measurements afterwards.
2. **Keep unit/time semantics explicit.** This IMU parser publishes acceleration
   in g and angular velocity in degrees/s, with host wall-clock update time.
   Preserve original packets, channel-specific times, sequence/drop diagnostics
   and clock provenance; then produce SI channels with a validated time mapping.
   Its 9600-baud display reader differs from the documented walking stack's
   expected 460800/quaternion configuration. Do not silently reconfigure the IMU
   or open a competing reader while that stack owns it.
3. **Do not promote head commands to measured kinematics.** `CameraAim` stores
   software angle estimates based on an approximate 0.09°/µs mapping. Output
   rate limiting and the Arduino's later slew mean even the saved command
   history is not measured acquisition-time pan/tilt. `angle_at()` selects a
   nearby stored command; it is not an encoder or exposure synchronization
   measurement. Calibrate fixed held poses first; continuous geometric fusion
   needs measured or independently validated joint state with uncertainty.
4. **Calibrate the original stereo stream.** The display code rotates a frame
   180° before splitting its halves because the camera is upside down. That can
   change eye order and image-coordinate conventions. Save original orientation
   and eye identity, then apply consistently calibrated rectification/transforms.
   A combined USB frame alone does not prove simultaneous left/right exposures.
5. **Keep compute results tied to the device.** The existing CPU preview documents
   low-rate relative-depth/flow processing. Those observations do not establish
   Fast-FoundationStereo throughput on the N150. Use calibrated SGBM as an onboard
   candidate and benchmark heavier models offline on the same recorded pairs;
   measure any onboard deployment's full pipeline separately.

## Selected moving-head architecture

- Body-mounted C1/IMU: qualify a planar body-localization path with its own timing,
  mounting calibration and declared odometry inputs.
- Moving camera: retain stereo-only depth/visual estimation independently of the
  body IMU initially. For body/world mapping, transform observations through the
  calibrated pan/tilt chain at acquisition time; retain unknown angles as unknown.
- Fixed-head reference: a separate locked pose enables a rigid camera–IMU
  calibration and an ordinary stereo–IMU baseline.
- Articulated visual–inertial fusion: a later explicitly implemented estimator,
  not the stock fixed-transform ORB-SLAM3 path with a time-varying camera mount.

A fixed sensor on a humanoid still moves with the body: body roll/pitch changes
its LiDAR scan plane. Begin planar qualification on a controlled, approximately
level route; evaluate tilt and gait motion separately. Coarse front/top/middle
placement does not establish a zero sensor offset or exact common origin.

First geometric control: keep the robot body stationary and sweep the camera.
A correct camera-to-body conversion should not report equivalent body rotation
or move a stationary obstacle in the body map. Then keep the head fixed and move
the body, followed by combined motion. Measure residual body-pose/map error,
pointing repeatability, settling time, and time alignment before scored active
perception trials. These tests are planned and have not run on this rig.
