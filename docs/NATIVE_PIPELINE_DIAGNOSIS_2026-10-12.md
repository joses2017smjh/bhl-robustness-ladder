# Native navigation pipeline diagnosis — October 12, 2026

The completed sensor campaign contains 200 measured development episodes. A read-only audit of **60 original episodes** found that stale-pose stops, including accumulated acquisition/recording overhead, are a concrete obstacle to navigation. This is a diagnosis of the existing pipeline. An asynchronous alternative has **not been implemented or evaluated** by this audit, and no improved result is claimed.

## Evidence and coverage

[Compact per-episode evidence](../results/native-pipeline-diagnosis-20261012/profile.json) and the [read-only profiling script](../results/native-pipeline-diagnosis-20261012/profile_existing.py) retain original job IDs, episode paths and SHA256 hashes, launch hashes, raw result hashes, truth hashes, native trace hashes, stop counts and measured timing.

The subset includes all six shards whose original outputs remained on `cn-gpu7`: seed 730000 with nominal, 2.5 Hz camera, 10 Hz camera and dark exposure conditions; seed 730001 with reduced LiDAR returns and LiDAR extrinsic error. Each shard contains two routes and five estimator/controller arms. This **60/200 accessibility subset is post hoc, not a random or representative sample**. The 40 ms exposure condition was not trace-profiled here.

Every episode's bytes were checked against the hash in its original scientific result. Each result was checked against the recovered raw-result binding manifest. The profiler also verified evaluator truth and stream receipt hashes. Native frame snapshots were checked against the episode's frame count, client p95 and sensor-time total, and their hashes were retained; this diagnostic did not independently download their raw release archives again. Registration-error calculations are exploratory evaluator-only diagnostics at the nearest recorded acquisition-time truth sample, conditional on accepted tracking.

Scientific source archive SHA256: `3628b95ced285660e27c0b3b0ca92a097415beaeedfa912da751832acc3b8282`. The inspected campaign source SHA256 is `12d3d63143bb65158ec12bfd2f5cca0ae78d331a0ed7bed6f1fd994f6a1e678f`, matching the retained worker source and [this repository version](https://github.com/joses2017smjh/bhl-robustness-ladder/blob/b27ad9c54082878ad993ca9df3ed9aa0fd51d0b1/scripts/bench/native_sensor_controls_campaign.py).

## What the traces show

These nominal examples all use the straight route and seed 730000. Navigation occupies 750 policy steps after the fixed initialization period, at 40 ms per step.

| Method | Native client p95 | Full pipeline p95 | Stale-stop steps | Steps with nonzero commands | Clean goal |
|---|---:|---:|---:|---:|---|
| Stereo | 33.8 ms | 257.7 ms | 625/750 | 125/750 | No |
| Stereo–IMU | 40.2 ms | 268.2 ms | 712/750 | 34/750 | No |
| LiDAR–IMU | 9.2 ms | 144.7 ms | 85/750 | 419/750 | Yes |
| Verified LiDAR recovery of stereo coordinates | 48.1 ms | 273.1 ms | 712/750 | 36/750 | No |

Exact example IDs are `straight-s730000-nominal-stereo`, `straight-s730000-nominal-stereo_inertial`, `straight-s730000-nominal-lidar_inertial`, and `straight-s730000-nominal-verified_lio_recovery`. All belong to original job **21769166**.

The stereo example accepted **175/175 native updates** but spent **83.3%** of navigation steps in `native_stale_stop`. Its final distance to the goal was 5.187 m. This failure is compatible with tracking remaining available while the controller repeatedly rejects its age. The same example charged 23.08 seconds to the combined sensor-work bucket across the episode, versus 5.66 seconds in the native client calls. These are measured wall-time totals, not added simulation durations or component p95 differences.

Stereo–IMU first produced an accepted pose at acquisition time **12.6 s**, after the navigation window began at 10 s. Its first accepted registration's exploratory XY error was about **0.039 m**, and the nominal stereo example's was about **0.040 m**. Initialization delay is therefore another observed limitation, while these selected examples do not show a large initial position-registration offset. They do not establish correct registration in every condition, orientation accuracy, or population pose accuracy.

The ten seed-730000 episodes at **10 Hz**, original job **21769347**, all had **750/750 stale-stop steps and zero nonzero navigation commands**. Across those ten episodes, native client p95 ranged from **31.8–58.8 ms**, while full pipeline p95 ranged from **3.842–5.634 s**. For example, `straight-s730000-camera_10hz-stereo` measured 38.4 ms client p95 and 5.165 s full pipeline p95. It would be incorrect to attribute that 5.165 s to ORB computation alone or conclude that a physical 10 Hz camera intrinsically prevents navigation.

## Mechanism and limits of the diagnosis

In the frozen source, `ControlledSensors.on_substep` measures the whole acquisition callback and accumulates its cost in `unbilled`. That callback includes simulated sensor generation, rendering, PNG encoding, evidence image writes, hashing and bookkeeping. `newest()` charges accumulated work to the next request. `run_episode()` then delays response delivery by the measured native call plus that accumulated sensor cost. Further sensor callbacks execute during the resulting simulated hold interval and contribute to the next request's cost.

This code path permits accumulated acquisition work to amplify subsequent response delay. The recorded multi-second pipeline times and stale-stop traces are consistent with that mechanism. The measurements **do not separate rendering, raycasting, encoding, filesystem writes and hashing into individual timings**, so they cannot identify which one dominates the sensor bucket. The measurements also do not establish the runtime cost of the user's physical camera or robot computer.

The 350 ms gate checks capture age, including time spent waiting for processing and response delivery. A pipeline p95 below 350 ms therefore does not guarantee a fresh pose at every policy step. In these examples, complete tracking and low conditional position error coexist with poor command availability.

## Proposed next comparison — not launched

Evaluate a **bounded latest-frame pipeline** with separate acquisition, estimator service and evidence-recording stages. Keep at most one pending frame per stream; explicitly count superseded frames. Move evidence-only archival writes and hashing off the inference path, while retaining all computation that the deployed estimator actually needs. Record capture, enqueue, estimator start/end, delivery and archive timestamps separately. No simulation cost may simply disappear: the protocol must distinguish inference service cost from simulation rendering and instrumentation costs and report all three.

Use a new, frozen development protocol before examining new outcomes:

1. Replay an identical retained sensor sequence to profile the current pipeline and candidate pipeline. Verify decoded sensor values, timestamp ordering and input identities. This stage measures timing and queue behavior only and makes no closed-loop success claim.
2. Freeze **48 paired closed-loop episodes**: two pipeline implementations × stereo and LiDAR–IMU × nominal, 10 Hz and 40 ms exposure conditions × two routes × two previously unused seeds. Reserve the seeds before launch, record matched hardware/resource limits, and fix the execution order or its randomization in advance. Closed-loop frames may differ after controller behavior diverges; pairing is by route, reset seed and acquisition specification, not by falsely claiming identical resulting images.
3. Require a fresh same-source smoke covering the two pipelines, two estimator arms and three conditions before full trials. Check finite execution, correct timestamp/queue behavior, unchanged native initialization thresholds, and preservation of the existing nominal readiness and no-fall/contact gates. A negative prerequisite remains explicitly unrun; do not modify gates after observing outcomes.
4. Keep the **350 ms capture-age gate**, actors, route/goal criteria and native estimator thresholds unchanged. Commands must stop on stale, untracked or unregistered poses. Ground-truth position, evaluator geometry and future sensor samples remain unavailable to estimation and control. Asynchronous recording must preserve the evidence needed to reproduce scoring.

Primary outcomes are the paired change in stale-stop fraction and clean-goal completion. Secondary outcomes are capture-age p95 at control, estimator and full pipeline p95, queue residence, superseded/dropped frame counts, initialization time, tracking losses, and falls/contacts. Report negative outcomes and the control-availability/safety tradeoff. Two new seeds would still be bounded development evidence; independent confirmation and physical validation remain separate tasks.
