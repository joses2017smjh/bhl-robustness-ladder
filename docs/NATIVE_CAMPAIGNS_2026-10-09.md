# Native estimation, navigation and terrain traversal campaigns

**Execution complete:** [measured results, demos, raw evidence and remaining research tasks](NATIVE_CAMPAIGN_RESULTS_2026-10-09.md). The protocol below retains its original criteria; completed negative experiments remain negative.

The October 8 stereo–LiDAR pilot and 276-cell terrain/navigation development
sweep did not run native SLAM or estimated-pose navigation, and did not establish
terrain traversal. This campaign implements and measures those missing stages.
Completed H3/H4 results are separate experiments and remain unchanged.

## Scientific inputs and measurements

| Stage | Actual implementation | Measurements and interpretation |
|---|---|---|
| Stereo SLAM | Pinned upstream ORB-SLAM3 C++ `TrackStereo`, headless display replacement | Every native tracking state, map changes, metric ATE/RPE, endpoint drift, lost time, compute and client latency; score each map separately |
| LiDAR–IMU odometry | Pinned FAST-LIO2 native IMU deskew, IKFoM update and ikd-Tree mapping behind a transport-only shim | Original timed 3D returns and 200 Hz SI IMU; no truth initialization; explicit null poses until tracking; disclose O1 native build and minimum effective-match tracking criterion |
| Estimated-pose navigation | Physical MuJoCo gait with a native estimator in the control loop | Goals, collisions, falls, stale/lost-pose stops, drift and capture-to-consumption age; measured native client latency advances physics |
| Terrain traversal | Frozen learned gait, 3D ray scans, IMU gravity alignment, local elevation/slope/roughness governor | Independent height error, known coverage and hazard recall together with full-horizon traversal outcomes; stationary survival is not success |

The estimator input boundary accepts images/calibration or raw LiDAR/IMU only.
Evaluator poses remain in a separate namespace. Offline rigid alignment has
scale fixed to one and is never returned to navigation. Navigation may use one
declared initial-frame registration and an externally specified goal, without
later evaluator corrections.

## Research incorporated

[ORB-SLAM3](https://arxiv.org/abs/2007.11898) supports stereo estimation and
multiple native maps. This motivates retaining lost states and map identifiers
rather than fitting a single trajectory across resets. This campaign uses
stereo-only mode; paper stereo-inertial benchmark figures are not our results.

[FAST-LIO2](https://arxiv.org/abs/2107.06829) directly registers LiDAR points
and maintains its map with ikd-Tree. Accordingly, replay preserves raw point
times and causal IMU measurements for the native deskewer instead of supplying
precomputed motion or transformed truth clouds. FAST-LIO2 is LiDAR–inertial
odometry and mapping; this adaptation does not add loop closure.

[Fankhauser, Bloesch and Hutter's terrain mapping work](https://doi.org/10.1109/LRA.2018.2849506),
described by the authors in
[elevation_mapping](https://github.com/ANYbotics/elevation_mapping), treats
local mapping and localization uncertainty together. The present local
geometric baseline tests a simpler question first: whether measured local
support is sufficient to help this frozen gait finish mild terrain. It does
not implement their probabilistic uncertainty propagation or claim equivalent
performance. Unknown support remains unknown; reporting coverage and missed
hazards prevents a low error on a few observed cells from hiding failure.

[Yao et al.'s stereo–LiDAR fusion paper](https://arxiv.org/abs/2504.05148)
uses matching-cost fusion, LiDAR semidensification and cross-sensor consistency.
The existing simple/confidence-gated fusion pilot is an engineering baseline
informed by this research, not a reproduction of that method. Native estimation
and physical traversal add the downstream measurements the pilot lacked.

## Frozen capture and replay

Three scene groups were assigned before capture: textured boxes/development,
thin posts/validation and ramp/step/test. Each new sequence covers 30 seconds,
with five stationary initialization seconds followed by a smooth closed path.
Original rectified stereo is 640 × 480 with a 0.12 m baseline. Replay v2 contains
450 stereo pairs at 5 Hz, 16 × 128 timed LiDAR rays over 50 ms and 200 Hz IMU.
The kinematic rig, ideal pinhole model, ideal IMU, ray returns and unmeasured
intensity placeholder are explicit simulation limitations.

Capture v1's full run exceeded its frozen storage cap and is **INCOMPLETE**.
Its receipts remain recorded. V2 reduces cadence while retaining the complete
path, increases the declared storage bound and requires a fresh smoke test.
No estimator score was used to choose that revision.

Native replay has a fresh executable/transport smoke followed by three scenes
per method. The smoke checks actual runtime operation, not tracking accuracy.
Before replay, freeze runtime/source/configuration/vocabulary and original data
hashes. The limited replay readiness gate requires at least 90% tracked frames
after the first five seconds, one continuous native map, translation ATE RMSE
at most 0.10 m, rotation error p95 at most 5 degrees and native compute p95 at
most 100 ms. These are engineering readiness thresholds, not a claim of
generalization, hardware readiness or successful navigation. A valid failed
initialization produces a negative result with no invented ATE.

For LiDAR scan-tail timestamps, the independent scorer interpolates bracketed
reference translations and rotations. It does not extrapolate the final
uncovered tail. Native inference never receives those reference values.

## Frozen terrain cohort

Screen all three existing DR-default gait actors on flat ground, 1 cm steps
and a 3 degree ramp, using two new physical variation groups: 18 episodes.
Each episode has a 5 m course and the same 40 second horizon. Qualification
requires all six goals and no falls, wall collisions or nonfinite states per
actor. Keep every qualifier; do not choose the best seed.

Only qualifying actors enter independent confirmation: baseline, LiDAR
governor and 50% LiDAR dropout, crossed with three terrains and eight new
groups. This is 72 episodes per qualifying actor, at most 216. Each actor/arm
requires at least 22 of 24 goals and no falls for its confirmation gate.
If no actor qualifies, report a negative screen with zero confirmation
episodes. The two-episode implementation smoke is excluded from both cohorts.

All native builds and scored runs use private frozen source/input bundles,
bounded Slurm allocations and durable completion receipts. Submission,
executable completion, scientific qualification and publication are separate
states. See `results/native-campaign-20261009/` and `SLURM_JOBS.md` for the exact
protocols, hashes, failures and job receipts.

## Execution and public evidence

The [raw capture release](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/tag/native-sensor-replay-20261009)
contains the actual319,616,699-byte dataset archive, SHA256
`ec15004ac7be4b58d33f78d6a4b5924d1d4a72b00c9332473d5617818ba08ca1`.
It is a sensor dataset; release publication is not an estimator or navigation verdict.

The [LIO navigation smoke audit](../results/native-campaign-20261009/navigation-lio-v1/smoke-observer/report.md)
and path figure come from two actual8-second episodes, excluded from qualification.
Scored jobs remain pending until their runner completion and raw measurements exist.
The failed allocation-step build21739893.1 is retained. A clean v6 native build runs as step21739893.2 with the same upstream algorithm and a corrected private OpenSSL multiarch header path; its queued backup is21743261 with dependent controller21743355. Actual allocation steps are distinct from backup batch submissions.
Resource revisions preserve source/input inventories and scientific thresholds;
cancelled pending submissions and failed build attempts remain visible.

The hosted [CPU regression run37992736230](https://github.com/joses2017smjh/bhl-robustness-ladder/actions/runs/37992736230)
passed, with zero JUnit errors/failures and124 optional skips. This verifies software
regression and trained-asset replay, not native campaign scientific acceptance.

Native per-frame compute, process/client wall time and sensor capture cost are
separate measurements. LIO navigation charges measured client/IPC time while
ideal ray generation represents the simulated sensor. Stereo navigation also
charges rendering and PNG encoding. This conservative stereo capture treatment
must be disclosed in method comparisons; it is not equivalent to a physical
camera's independently clocked acquisition pipeline.
