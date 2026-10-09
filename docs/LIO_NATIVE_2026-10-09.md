# Pinned native FAST-LIO2 replay

This route executes the actual FAST-LIO2 iterated Kalman filter, native IMU
deskewing, scan-to-map residual/Jacobian and incremental ikd-Tree map. It does
not initialize motion from simulator poses or manufacture estimator output.

Upstream source is pinned to [`hku-mars/FAST_LIO` revision
7cc4175de6f8ba2edf34bab02a42195b141027e9](https://github.com/hku-mars/FAST_LIO/tree/7cc4175de6f8ba2edf34bab02a42195b141027e9),
including its ikd-Tree submodule at
`e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4`. The published algorithm and sensor
requirements are documented in the [pinned upstream
README](https://github.com/hku-mars/FAST_LIO/blob/7cc4175de6f8ba2edf34bab02a42195b141027e9/README.md)
and [FAST-LIO2 paper](https://arxiv.org/abs/2107.06829).

## Isolated build

On an allocated CPU node, with a C++14 compiler and Python 3.11 or newer:

```sh
python scripts/native/lio_build.py --output /tmp/fastlio-runtime-new
```

The builder fetches the exact source revision and checksum-pinned Ubuntu
Jammy PCL 1.12.1, Eigen 3.4, Boost 1.74, FLANN and LZ4 packages from
`scripts/native/lio_dependencies.json`. It extracts them into a private
directory; it does not install packages or change the shared environment.
The selected native libraries require glibc 2.34 or older, matching Rocky 9.
The build uses a compiler resolved from `--compiler` or `g++` and records the
exact compiler, command, binary hash, dependencies and dynamic-library check.
Runtime libraries are collected next to the binary. Dependencies and source
are retained in the build receipt or source snapshot as applicable; the root
campaign archive defines the final retention scope.

The ROS subscribers, publishers, visualization and debug recording are
replaced with a typed offline transport. The actual estimator callbacks and
global state are extracted byte-for-byte from upstream `laserMapping.cpp`;
the original license is retained. `IMU_Processing.hpp`, IKFoM and ikd-Tree are
used unchanged. Generated message shims contain only original SI values and
timestamps; they do not implement estimation. Callback section hashes and the
original source hashes are saved in `runtime.json`.

The transport performs the equivalent feature-disabled, given-offset
Velodyne conversion to PCL XYZ/intensity and curvature in milliseconds.
Original ring indices and raw per-point seconds are preserved in its binary
input. No inferred scan offsets or synthesized motion are permitted.

The scoped build uses `-O1`, whereas upstream normally uses `-O3`. Native
timing therefore describes this build; it cannot establish a comparison with
published upstream performance. This is an instrumented headless native
core, not an unmodified ROS node integration test.

## Replay and provenance

```sh
PYTHONPATH=src python scripts/native/lio_replay.py \
  --manifest /path/to/replay/manifest.json --sequence SEQUENCE_ID \
  --runtime /tmp/fastlio-runtime-new --output /tmp/fastlio-output-new
```

The audited adapter verifies original inference file hashes, calibration,
strictly increasing original capture times, timed scans and high-rate IMU.
The raw binary input contains only calibrated lidar/IMU inference data and
the declared estimator configuration. `T_I_L = inverse(T_B_I) @ T_B_L` maps
original lidar coordinates into the IMU body. It is preserved exactly and
online extrinsic estimation is disabled.

A scan receives only IMU samples through its original last return. FAST's
native IMU code supplies its own previous sample across scans. Output time
is the original **scan tail**, rather than the scan header or wall clock.
Recorded simulation intensity can use an explicitly declared zero
placeholder because the ray sensor lacks intensity physics; physical or
external data missing intensity are rejected by the original intake.

The native stream `native_frames.jsonl` records every scan, including:

- `timestamp_s`, `scan_begin_s` and original frame index;
- `tracked`, native initialization/failure state and `map_reset_id`;
- `T_W_I`, or **null** when no tracked estimate exists;
- effective native map matches and compute seconds.

Tracking is a transparent adapter criterion: the last native update must
have at least one effective map residual and a finite rigid pose. FAST-LIO2
does not expose a complete ORB-style tracking-state classifier. This
criterion does not claim observability, covariance qualification, successful
relocalization or collision-free navigation. No automatic reset is added.

`trajectory.json` is compatible with the existing strict pose scorer. Its
untracked rows use an explicitly marked identity storage placeholder only
because that schema requires a matrix for every row; those matrices must
never be scored or supplied to navigation. The authoritative native stream
retains null. `estimator_receipt.json` binds source/runtime, input manifest,
all original lidar/IMU hashes, calibration, configuration and trajectory.
No simulator truth is accepted by this entry point.

## Closed-loop native client

`bhl_robust.research.native_lio.NativeLioClient` runs the same verified binary
persistently through private FIFO transport. Each request supplies one
original timed scan and only new causal IMU samples through its tail. The
client returns the actual native `T_W_I` or null and native state before the
next controller command is chosen. Original input bytes and every native
response are retained and hashed in `stream_receipt.json`.

The scan dictionary requires `points_xyz_m`, absolute `point_time_s`,
`ring_index`, and `frame_timestamp_s` equal to the last return time. It also
requires original `intensity`, or explicit `origin: simulation` for the
declared unmeasured zero placeholder. IMU dictionaries require `timestamp_s`,
`gyro_rad_s`, and `specific_force_m_s2`. Previously consumed samples cannot
be repeated, future samples cannot be supplied, and gaps above 20 ms are
rejected. The first sensor request must begin after recorded IMU becomes
available; a scan spanning negative startup time with no IMU is invalid.

The native entry point flushes each response and allows clean EOF between
scans so an episode can stop at a goal before its maximum frame budget.
Partial scan payloads fail. Native tracking loss must cause the controller
to stop or apply its separately declared failure rule; it must never use an
identity placeholder as an estimated pose.

Native process success is an execution result. Tracking loss, initialization,
drift and task outcomes require separate reporting; a completed executable
must not be described as a scientific qualification.

## Predeclared development settings

The replay defaults use 0.15 m surface and map voxels, 0.1 m blind range,
all raw returns, four EKF iterations, known fixed extrinsics, no software
time correction, and upstream mapping noise values (gyro/acceleration 0.1;
bias values 0.0001). These settings are fixed before examining held-out
metrics. Any subsequent change requires a distinct recorded run.

Known scope limits: ideal simulated lidar/IMU are not hardware validation;
the replay is odometry/local mapping rather than loop closure; native
per-scan timing excludes acquisition, export and process startup. Actual
results, scheduler receipts and independent audit belong to the root
campaign result directory.
