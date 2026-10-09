# Native estimated-pose navigation development campaign

This new route uses **actual persistent FAST-LIO2 output** to command a
frozen learned 12-DoF gait in MuJoCo. It uses the existing ContactRunner,
upstream gait observations, ONNX actor and PD/physics path. It is a LIO-only
development route; ORB-SLAM3 closed-loop navigation remains a separate task.
The NavGym actor is not used and no physical experiment is represented.

The planner accepts only native tracked poses, fixed calibration, an external
prescribed route, simulation time and raw lidar returns. It never accepts a
simulator pose, evaluator goal crossing, geometry label, ideal trajectory or
injected localization noise. Initialization registers the first native pose
once to the **declared fixed starting station**, using the model's reset
configuration and calibrated IMU mount. Actual drift while settling is not
corrected by simulator truth. No subsequent registration or map reset is
silently repaired. No loop-closure or exploration claim is made.

## Predeclared development design

- Smoke: two independent straight-corridor episodes, seeds370000–370001,
  eight simulation seconds each. This checks actual native tracking and
  the physics/transport path; a goal is not expected within eight seconds.
- Development: straight5m corridor, dogleg and occluder corridor, each at
  seeds380000–380002; nine 40-second episodes for one frozen gait/native
  runtime. Known external waypoints are fixed in source before submission.
- Each episode starts with five seconds of zero gait commands. Raw native
  sensor acquisition begins at2s after initial landing transients; it does
  not use ideal steady IMU or oracle body pose for estimation.
- Raw lidar: sixteen elevations from−35°to25°,128azimuth columns from−80°to80°,
  8m maximum range,50ms sweep,5Hz. Each substep casts its own columns from
  the actual moving rig. Robot collision geometry is excluded from rays.
  MuJoCo `mj_step` retains sensor/kinematic fields from before its state
  integration; those samples are stamped `data.time−physics_dt`. Evaluator
  pose is computed from the post-integration free-joint state at `data.time`.
- IMU: native MuJoCo gyro and accelerometer at200Hz. The physics timestep
  must divide5ms. Original SI samples and absolute simulation timestamps
  are retained; no orientation/translation truth enters FAST-LIO2.
- IMU extrinsic comes from the actual base-mounted MuJoCo site. Lidar uses
  body-aligned axes and known mount translation `[.10,0,.38]`m. Native
  `T_W_I` is converted to body pose using `inverse(T_B_I)`.
- Fixed route law: maximum0.30m/s forward,0.45rad/s turn; turn in place for
  heading error≥0.75rad. A raw observed return within the fixed forward
  collision envelope stops motion. Empty/insufficient returns stay unknown.
- An untracked/null, reset, unregistered, future or stale (>350ms) pose
  commands zero. Estimated final-goal distance<0.25m commands zero. The
  evaluator never sends a successful goal stop back to the planner.

Measured `client.track` wall latency, including serialization/IPC and output
validation, is charged to real physics/gait updates while holding the
previous command and older available pose. Latency is rounded upward to
the gait period (normally40ms); this quantization is recorded. The computed
response, raw scan, registration and loss/reset state become visible only
after the charged physics interval. The old pose can independently become
stale and stop motion during computation. Each trace records sensor capture,
response arrival and controller consumption times. Acquired frames during a long
compute interval are retained as raw data; obsolete queued scans are dropped
explicitly while every intervening IMU sample is supplied to native LIO.

Success requires reaching the externally specified goal within0.30m,
remaining upright, avoiding all wall/fixture contacts and completing the
entire matched simulation horizon. Stopping short is a timeout. Simulator
truth is used only for sensor generation, goal/collision/fall outcomes and
separately saved drift evaluation. The scientific gate to any confirmation
is **9/9clean development goals**; otherwise the negative development
result is retained and no confirmatory experiment is released.

## Running the frozen route

The root campaign must freeze the following new source plus its existing
dependencies and the exact native runtime/deploy/ONNX inputs before smoke:

```sh
PYTHONPATH=src python scripts/bench/native_navigation_campaign.py \
  --upstream external/Berkeley-Humanoid-Lite \
  --deploy /frozen/deploy.yaml --checkpoint /frozen/policy.onnx \
  --runtime-archive /frozen/lio-build-outputs.tar.gz \
  --runtime-archive-sha256 EXACT_ARCHIVE_SHA256 --phase smoke
```

Use `--phase development` only after a fresh successful smoke of the same
frozen source/runtime/model. Runtime archives are checked and safely unpacked
in private work storage; only the verified private binary receives execute
permission. Shared Python/system environments remain unchanged.

The waypoint law is a fixed scripted planner; only locomotion uses the
frozen learned policy. Known external route waypoints are not a learned
exploration or map-building policy. ORB-SLAM3 closed-loop navigation has not
been implemented by this LIO-only entry point.

Each episode retains original sensor arrays and native binary requests,
all native tracked/lost states and null poses, client/runtime/config hashes,
every command and reason, actual latency holds, contact/fall outcomes and a
separate evaluator-only pose archive. Runtime assets stay in node-local
scratch to avoid duplicating the immutable robot mesh inventory in outputs.

Execution PASS means the episode ran and retained its records. Native drift,
goal rates, falls and scientific qualification require examining those
records. Ideal simulated sensors and one fixed actor cannot establish
hardware reliability or population safety.


## First actual closed-loop execution

Fresh smoke job `21740837` completed both declared 8-second episodes using the
actual native FAST-LIO2 binary and learned gait: 52 tracked frames of 58 total,
zero falls and contacts, no dropped scans. Each episode included three explicit
native initialization states; client wall p95 was 15.85/15.90 ms. Smoke goal
counts are excluded from qualification. Development job `21740945` uses the
same immutable source/runtime/model pack and the declared nine 40-second
episodes, after the successful fresh smoke. Its scientific verdict is pending.

The compiled model keeps the IMU site under a fixed child body. Calibration
therefore composes all static parent-body transforms plus the site transform,
and rejects any articulated, unrelated or mocap mount. This uses model
calibration metadata, without simulator pose registration.

`native_navigation_observe.py` separately verifies completion/raw/native-stream
hashes, keeps original null/lost states, and scores native IMU trajectories
against evaluator-only bracketed truth. Alignment has fixed metric scale 1
and permits gravity-preserving yaw/translation only. It also produces an
actual-data navigation-path figure; no estimator or controller is rerun.
The optional additional sensor capture/encoding cost hook is charged together
with client wall latency; it is zero for this LIO-only route.


The queued development v1 job 21740945 was cancelled before execution because
of the CPU queue. Resource-only v2 retains all 846 original source/input
manifest records byte-for-byte, changes only job identifiers and scheduling
to preempt with requeue disabled, and requires a new fresh smoke 21741213
before development 21741509. The scientific cohort and gate are unchanged.

A tiny durable collector bundle verifies its own ten source files, job/source
identity, complete raw archive and per-stream/evaluator hashes. After a
completed campaign, the command in
`results/native-campaign-20261009/navigation-lio-collector-v1/COLLECT_COMMAND.txt`
produces a Markdown report, JSON metrics, actual/native path figure and
collection receipt. It has been tested on the completed original smoke,
which remains explicitly SMOKE_ONLY. No inference is rerun and no push is
performed by this collector.
