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


## Actual development result — PASS, not confirmation

Development job **21741509** completed the frozen nine-episode cohort on
`cn-b09.hpc.engr.oregonstate.edu`: **9/9 clean goals**, zero falls, fixture
contacts or nonfinite states. Every episode completed all 40 simulation seconds
and 1,000 policy updates, including time after reaching the goal. This is a
**DEVELOPMENT_ONLY_NO_CONFIRMATION** result for one frozen gait, three prescribed
simulation fixtures and three reset groups. The predeclared development gate
passes; no holdout confirmation or hardware experiment has been run.

| Measurement | Actual development result |
|---|---:|
| Clean goals, full horizon, no fall/contact/nonfinite | 9/9 |
| Native tracked frames / all native frames | 1,674 / 1,701 |
| Tracking after the declared 5-second initialization | 100% in every episode |
| Per-episode aligned IMU ATE RMSE | 1.46–2.89 cm |
| Aggregate aligned IMU ATE RMSE, all 1,674 associated poses | 2.04 cm |
| Per-episode native client wall p95 | 13.55–14.16 ms |
| Final true goal distance | 0.128–0.269 m |
| Runner wall time / total simulated time | 272.70 s / 360 s |

ATE uses an **evaluator-only gravity-preserving yaw/translation alignment with
metric scale fixed at 1**. It is not unaligned global position error and the
alignment never enters the controller. Goal distances are measured in the
declared simulator world and satisfy the unchanged 0.30 m criterion. Client
latency includes serialization, native processing, IPC and output validation;
ideal simulation ray generation is excluded. The controller charges this wall
time to physics, rounded upward to the 40 ms policy period, before making the
response visible. These timings do not establish physical sensor throughput or
hardware real-time performance. The binary was compiled with GCC 12.5 at `-O1`.

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


The queued v1 development job `21740945` was cancelled before execution because
of the CPU queue. Resource-only v2 retains all 846 original source/input
manifest records byte-for-byte and changes only job identifiers and scheduling
to `preempt` with requeue disabled. Fresh v2 smoke **21741213 PASS** preceded
development via `afterok`: both 8-second episodes completed, with 52 tracked
frames of 58 total and no falls, contacts or nonfinite states. All smoke
outcomes remain **SMOKE_ONLY** and are excluded from the 9/9 development count.
The original v1 smoke `21740837` and cancelled pending submission are retained.

A tiny durable collector bundle verifies its own ten source files, job/source
identity, complete raw archive and per-stream/evaluator hashes. After a
completed campaign, the command in
`results/native-campaign-20261009/navigation-lio-collector-v1/COLLECT_COMMAND.txt`
produces a Markdown report, JSON metrics, actual/native path figure and
collection receipt. Its actual development collection verifies the complete
raw archive, original native input/output streams and separate evaluator
arrays without rerunning inference or the controller. A second audit
reconstructs all nine goal distances, body tilt/height and complete horizons
from the saved evaluator transforms at policy ticks. Contact flags are checked
against retained physics traces; this does not independently reconstruct
between-tick contacts or joint state.

Final independent provenance validation rehashed **all 846 frozen source/input
payloads**, all ten collector source files and all seven collection-receipt
files, and compared the public observer files byte-for-byte with their durable
copies. Artifacts:

- [Development report](../results/native-campaign-20261009/navigation-lio-v2/development-observer/report.md)
  and [actual path figure](../results/native-campaign-20261009/navigation-lio-v2/development-observer/actual-navigation-paths.png).
- [Final provenance and metric audit](../results/native-campaign-20261009/navigation-lio-v2/development-final-audit.json).
- [Independent outcome reconstruction](../results/native-campaign-20261009/navigation-lio-v2/development-outcome-audit.json),
  produced by `scripts/bench/native_navigation_outcome_audit.py`.
- Frozen navigation archive SHA256:
  `df3d91ae4ad8c4c8688d7ed3ff8639baff7a5860043f39414871c559301cfa26`.
- Actual raw output archive: 91,609,334 bytes, SHA256
  `98c8e35a146fa98046a9b1973ae388a4939431db2a2ae7fb45b5815a3c279633`.
  The full original remains durable; the public raw-artifact receipt records
  release publication separately from compact files in Git.

An appropriate project claim is: **Integrated pinned native FAST-LIO2 with a
frozen learned 12-DoF humanoid gait; achieved 9/9 full-horizon navigation goals
across three simulated corridor fixtures and three reset groups, with no
falls or fixture contacts, 2.04 cm aligned IMU ATE RMSE and 13.55–14.16 ms
per-episode client p95.** Include the simulation/development scope wherever
these numbers are presented. This campaign does not establish generalization,
a learned navigation planner or physical-robot reliability.
