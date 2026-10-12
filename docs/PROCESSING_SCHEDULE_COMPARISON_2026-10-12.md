# Native processing-schedule comparison

The 48-episode comparison requested on 2026-10-12 is implemented as a new,
robot-independent simulation campaign. It does not overwrite the completed
200-episode native sensor-control campaign or tune its negative outcomes.

## Fixed comparison

| Factor | Predeclared values |
| --- | --- |
| Processing | Instrumented cumulative callback reference; bounded latest joined input |
| Native pose | ORB-SLAM3 stereo; FAST-LIO2 LiDAR–IMU |
| Camera conditions | Nominal 5 Hz; 10 Hz; causal 40 ms exposure |
| Routes | Straight; occluders |
| Development seeds | 750000, 750001 |
| Horizon | 40 s, including the unchanged 10 s initialization |
| Same-source smoke | 12 cells, 18 s, separate seed 749000 |

The two schedules run in alternating order within matched pairs on the same
GPU allocation. The native algorithms, settings, learned gait, route, known-start
registration, raw LiDAR brake and 350 ms capture-age cutoff remain fixed.
Untracked, unregistered, reset and stale poses retain the same stop behavior.

## What the intervention measures

The reference charges all cumulative sensor callback work to the next native
request. The candidate separates simulated acquisition, measured required PNG
transport/native service and evidence archival. It holds one pending joined
camera/LiDAR event and one active estimator request. Replacing an unstarted input
retains the latest identities and every intervening causal IMU sample.

A real bounded single-worker recorder writes immutable image buffers, records
backpressure and reports failures. Its files must finish and selected decoded
pixels must match retained acquisition before an episode passes. MuJoCo rendering
and raycasting remain on the simulator thread. Required native client I/O remains
inside measured service, including the native wrappers' synchronous evidence.

This is a **model of separated processing stages**, with actual measured native
calls and archival work. It is not proof of a concurrently executing robot
pipeline or measured physical sensor latency. Both stages' wall-clock costs and
their distinct simulated availability times are retained; no discarded frame is
silently counted as processed. Pre-join captures are listed separately.

Before smoke, the same first 40 stereo pairs from the retained nominal sequence
are decoded and transported through both evidence/PNG paths. Original PNG and
sensor-array hashes, decoded values and timestamp order are checked. This is a
preprocessing/input-integrity profile, not another native trajectory or closed-
loop navigation result. File-cache effects prevent inferential speedup claims
from this small profile. A transport-only queue replay uses those measured
costs with the unchanged 25 Hz delivery quantization and arrival-step hold;
native service is explicitly zero in this input-profile model.

## Gates and outcomes

The full smoke must execute without software, sensor-integrity or nonfinite-state
failures. Each pipeline/estimator's nominal smoke must produce an accepted native
pose, finish its horizon and avoid falls/contacts. A negative nominal gate keeps
its declared development cells explicitly unrun. Degraded tracking outcomes and
closed-loop failures remain measured scientific results. Code corrections require
a new source freeze and fresh smoke; outcomes do not authorize threshold tuning.

Primary results are paired navigation stale-stop fraction and clean goals.
Decision timestamps are recorded before the physics update, avoiding the older
trace's 40 ms post-step offset. Shortened horizons are retained; no navigation
steps means the stale fraction is unavailable, rather than zero. Secondary
results include pose-age and processing p95, queue residence, superseded frames,
initialization/tracking, contacts and falls. The strict collector independently
recomputes goals, stale fractions, queue conservation and modeled delivery times.
Only complete matched pairs enter paired differences; the report exposes gaps.

All bulk evidence uses HPC share or allocation-local scratch. The campaign has a
100 GiB cap within the user's authorized 2 TiB shared-storage hard limit. Two
reserved seeds support bounded simulation development, not independent
confirmation, hardware validation or population performance claims.

GPU job **21801949** started on `cn-gpu6` (Quadro RTX 8000) on October 12.
The retained 40-frame preprocessing/input profile passed. The 12-cell native
smoke and subsequent 48-cell cohort are pending completion; no improvement
result is claimed yet. The source is frozen at `152ccf4`, archive SHA256
`46eb5d49361cb64f4f18dc8290a7bcb002c21b01aa035e3fe8c8e049949a0dda`.
Allocated CPU validation **21801920** passed 306 tests and 11 subtests.
[Protocols, frozen intake, submission, GPU receipt and passing profile](../results/processing-schedule-20261012/).
The one-allocation dispatcher automatically runs the strict collector after
all declared pairs; runtime errors preserve evidence and stop dependent work.
