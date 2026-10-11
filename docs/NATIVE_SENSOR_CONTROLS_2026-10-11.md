# Native navigation acquisition and fault controls

This separate campaign adds matched camera, LiDAR and IMU controls to the completed development studies. It preserves their original sources and negative results. The new experiment declares **14 execution-smoke episodes**, followed by **up to 200 development episodes**, with explicit unrun cells for arms that fail nominal readiness. Nothing in this document is an achieved performance result.

## Matched arms and conditions

The five pose-estimator/controller arms are native ORB stereo, native ORB stereo–IMU, native FAST-LIO2, permanent-stop stereo with FAST-LIO2 running in shadow, and LIO-assisted verified stereo map recovery. The last two charge both native processes. Verified recovery retains three registration observations and three disjoint later validation observations; it never substitutes a LiDAR pose for stereo. All five share the same independent raw LiDAR obstacle brake, frozen locomotion policy, prescribed route, sensor acquisition and compute accounting. Thus “estimator alone” does not mean a camera-only obstacle planner.

| Condition | Fixed intervention |
|---|---|
| Nominal | 5 Hz stereo; 5 Hz timed 3D LiDAR; 200 Hz IMU |
| Camera 2.5 Hz | Camera schedule and native FPS changed together; LiDAR remains 5 Hz |
| Camera 10 Hz | Camera schedule and native FPS changed together; LiDAR remains 5 Hz |
| Dark exposure gain | Acquired camera intensities multiplied by 0.25 |
| Causal 40 ms exposure | Equal-weight integration of three actual rendered poses over the preceding 40 ms |
| Visual blackout | Both acquired images replaced with zero pixels from 12 through 16 seconds |
| Quarter LiDAR returns | Stable return subsampling, preserving both temporal endpoints |
| LiDAR calibration error | Reported extrinsic receives 2° yaw and 2 cm x error; physical sensor mounting remains fixed |
| IMU clock +80 ms | Timestamp and availability both delayed; no future samples are forwarded |
| Gyroscope bias | +0.05 rad/s z bias begins at 5 seconds |

Exposure samples use the robot's actual simulated pose at each sub-exposure time and retain their timestamps, hashes and PNGs. The averaging and intensity-gain models approximate acquisition effects; they are not calibrated hardware shutter, shot-noise or read-noise models. The planner's original 0.35-second freshness gate is fixed across camera rates, so low-rate results include the effect of that unchanged gate.

## Initialization and gates

Every arm and condition receives the same declared simulation-only initialization: stationary until 3 seconds, bounded sinusoidal forward and yaw commands during `[3, 9)` seconds, stationary again until navigation starts at 10 seconds. Command bounds are 0.10 m/s and 0.30 rad/s. This produces actual inertial excitation without changing ORB initialization thresholds. It is an explicit open-loop simulation maneuver, not an estimator fallback or physical actuation command.

The original nominal known-start registration assumption is retained: the first accepted native pose is anchored to the nominal start station. Initialization movement can introduce registration error; evaluator truth is retained separately and never used to correct that registration. Results must disclose this limitation.

The smoke cohort uses fresh seed `720000`: five nominal arms and the nine degraded conditions through the complete verified-recovery sensor path, each for 18 seconds. Every cell must execute with finite retained sensor/output evidence. Tracking degradation, failure to initialize, contact and falls under degraded conditions remain scientific outcomes and do not silently turn into runtime errors.

Each arm's full development is separately gated by its **nominal** smoke: an accepted native pose, full horizon and no fall, contact or nonfinite state. The stereo–IMU arm additionally requires actual native inertial initialization. A negative nominal gate leaves that arm's 40 development cells explicitly **unrun**. The cohort is not enlarged, retuned or retried after a scientific negative.

Development uses straight and occluder routes, seeds `730000` and `730001`, all ten conditions, and all five eligible arms for 40 seconds: **200 declared cells**. Twenty serial shards each contain two routes and five arms for one condition/seed. A strict collector requires every declared cell to be measured or explicitly accounted for by a negative nominal gate, verifies retained file hashes, and recomputes clean outcomes from evaluator traces.

## Measurements and inference boundary

- Clean goals, goal time, contacts, falls, full horizon and nonfinite states.
- Unique native updates, accepted tracking fraction, native map changes, tracking losses, completed recovery durations and right-censored unrecovered losses. Repeated delivery of a held packet does not count as a new native update.
- Full pipeline wall time: timed-ray generation, rendering, exposure processing, PNG encoding/hashing, native transport and inference, and recovery computation. Its measured delay is advanced through physics before a response becomes visible. Diagnostic sensor bookkeeping is conservatively included. Native initialization/process setup remains visible in total job wall time, not attributed to per-frame latency.
- Inference-visible ORB diagnostics, LIO effective-point counts, native states and poses, and measured latency per native event. Simulator poses remain in a separate evaluator artifact.
- Evaluator-only false-clear raw-brake packets: compare the perturbed input with original same-time simulated surface returns inside the actual brake box. Fewer than 20 perturbed returns cause an unknown stop and never count as safe free space. This does **not** measure complete 3D occupancy, unseen hazards, or pixel false-free space.

These are ideal simulated sensors and two reused route families. The cohort supplies development comparisons, not physical safety guarantees or independent terrain confirmation.

## Outcome-calibrated offline screen

The scheduler completes all seed-`730000` conditions first. It extracts only that seed's episode-level features/outcomes, fits the separately declared [offline health screen](../src/bhl_robust/research/outcome_health.py), and publishes the frozen model with independently verified hashes **before submitting any seed-`730001` trial**. The second seed is evaluated with that frozen model. Insufficient positive/negative outcomes remain explicitly unsupported; if every arm is gated, the health analysis is explicitly unavailable.

This screen uses episode-end tracking fraction and is neither an online controller nor a calibrated failure probability. Any use for real-time stopping requires a later closed-loop validation experiment.

## Execution and storage

The scientific runner is [native_sensor_controls_campaign.py](../scripts/bench/native_sensor_controls_campaign.py); its predeclared settings are in [the protocol](../protocols/native-sensor-controls-20261011.json). The source/input snapshot is separate from the [durable dispatcher](../scripts/bench/native_sensor_controls_dispatch.py).

New durable outputs are restricted to `/nfs/hpc/share/sanchej7/humanoid-research-20261011/native-sensor-controls-v1` and its explicitly scoped controller directory. The dispatcher reserves 2 GiB for one active output and preserves an 8 GiB shared-storage floor. Scientific work and rendering caches run in private node-local temporary storage. Completed raw archives are uploaded to the [October 11 evidence release](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/tag/research-campaign-evidence-20261011), freshly downloaded and SHA256-verified before duplicate durable raw packaging is retired. Unique measurements, failed attempts and compact provenance receipts are preserved. There are no automatic retries or hidden source changes.

The frozen same-source smoke must finish before any development submission. Actual job IDs and execution receipts are recorded separately; scheduled counts are not results.

## First execution receipt

GPU smoke **`21768425` completed 14/14 episodes**, exit `0:0`, on `cn-gpu7` in 6 minutes 26 seconds. All five nominal arm readiness gates passed. The stereo–IMU arm produced **38 actually initialized/accepted native frames** with the declared excitation and unchanged native initialization thresholds. All smoke episodes had zero contacts and falls; none reached a clean goal in the short horizon. This is readiness evidence, not a navigation success result or a causal comparison against the earlier failed initialization study.

The first utility controller (`21768472`) failed before plan validation or any development submission because that node's `/usr/bin/python3` lacked Python 3.11's `hashlib.file_digest`. Its log is preserved. A separately recorded utility-only correction (`21768520`) uses the existing shared Python 3.11 runtime; scientific source, inputs and thresholds remain unchanged. The corrected controller must verify and publish the completed smoke before starting the twenty declared development shards.

[Frozen source metadata, completed smoke, and controller receipts](../results/native-sensor-controls-20261011/) identify source archive `cad8dda245725ccd87d9f37a38e783b881e110ef4074c826533199e90bb31c41`, scientific runtime image and all input hashes. The [published raw smoke archive](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/research-campaign-evidence-20261011/native-sensor-controls-v1--native-controls-smoke-20261011-21768425.tar.gz) is 562,273,268 bytes, SHA256 `c42646a99467cbe7aca69e6445e059358212d05aa1f10e83f5acce2eac7816fd`; a fresh independent download verified it before duplicate shared packaging was retired. The corrected controller is waiting for its reserved output space while preserving the 8 GiB shared-storage floor. Full development results remain pending.
