# Locomotion, sensing, and hardware transfer

This is a proposed organization, not a claim of completed hardware transfer.

| Repository | Owns | Current evidence |
|---|---|---|
| `bhl-robustness-ladder` | Locomotion, navigation, sensor simulation, controlled policy evaluation. | Isaac Lab / MuJoCo studies and separately labelled external recording replays. |
| [`quest-vr-teleop`](https://github.com/joses2017smjh/quest-vr-teleop) | VR arm control, low-level interfaces, arm calibration, hardware demos. | Physical arm operation; not validation of learned leg policies. |
| Proposed `humanoid-hardware-transfer` | Rig calibration, real sensor recordings, deployment adapters, staged hardware trials. | Not created in this documentation refresh. Hardware policy transfer remains unvalidated. |

Keep the existing locomotion URL and history. Its clearer display title is **Humanoid Locomotion & Navigation**. Separate a hardware repository when it has a runnable deployment path, a distinct dependency set, and an actual first recording; do not copy a large research tree into an empty showcase.

Suggested hardware repository structure:

```text
calibration/       # Intrinsics, extrinsics, timing, joint mapping and units
recording/         # Sensor capture and timestamp checks
deployment/       # Policy adapter, action limits, watchdog and stop handling
configs/          # Named robot/sensor configurations, no credentials
tests/            # Replay and interface contract checks
results/          # Versioned trial manifests and small evaluation summaries
docs/             # Setup, rig diagram, trial procedure, known limits
```

Milestones for an honest sim-to-real claim:

1. Record and calibrate the physical sensor rig; publish timestamps, units, frames, and uncertainty.
2. Replay real recordings through the estimator; report resets, tracking loss, and reference limitations.
3. Verify observation/action ordering, timing, joint signs, limits, and stop behavior against the simulation adapter.
4. Run staged supported hardware trials under a documented procedure, preserving failures and interventions.
5. Compare a fixed policy and baseline on a declared physical task with episode counts and success criteria.

External real-camera datasets test a component on real observations. They do not prove physical closed-loop locomotion. Planned episodes, queued jobs, and selected GIFs are not aggregate validation results.
