# Implemented Humanoid research follow-ups

The six original research follow-ups now have executable methods, with a separate perceptive-learning experiment added. Implementation, successful execution and measured improvement are recorded separately. Full development campaigns are queued or running, and some initialization/learning results are already negative. Earlier results in [the native campaign report](NATIVE_CAMPAIGN_RESULTS_2026-10-09.md) remain unchanged.

## Coverage of the original six follow-ups

| Follow-up | Concrete implementation | Execution boundary |
|---|---|---|
| Explain stereo startup failures | Read-only native features, valid stereo-depth matches, state and map diagnostics | All 450 consumed frames replayed; both failed scenes remain below the unchanged feature threshold |
| Test stereo–IMU navigation | [Actual native IMU_STEREO transport, calibrated original IMU batches and initialization-gated navigation](STEREO_INERTIAL_METHODS_2026-10-10.md) | Full 900-frame paired replay complete and inertial initialization negative; fresh navigation smoke `21757768` submitted; 18 development episodes require qualification |
| Stress LiDAR–IMU estimation | [Clock offsets, extrinsic error, gyro/accelerometer bias and return loss](SENSOR_FAULT_METHODS_2026-10-10.md) | Native smoke complete; 48 full replay shards / 144 estimator cells queued in continuation `21757767` after fresh capture |
| Qualify estimated-heading terrain control | [Native-pose pursuit and regulated-speed controls](TERRAIN_METHODS_2026-10-10.md) | One 30-episode shard complete; remaining eight queued, 270 total declared |
| Measure dense stereo–LiDAR fusion | [Pretrained stereo and uncertainty maps](SENSOR_METHODS_2026-10-10.md), plus [local photometric-cost/consistency fusion](SENSOR_FAULT_METHODS_2026-10-10.md) | Actual GPU/native smoke complete; 18 fresh capture cells, followed by 24 five-arm cost/consistency shards |
| Test sensor-assisted recovery | [Disjoint estimated-pose registration and validation across native stereo map changes](STEREO_METHODS_2026-10-10.md) | Actual recovery exercised in smoke; 18 matched navigation episodes queued |

The implemented scope retains ideal simulated cameras at 5 Hz. Camera-rate/exposure/motion-blur ablations, fault-injected closed-loop recovery, independent terrain confirmation and hardware validation remain further experiments. Local image costs do not reproduce an aggregated SGM cost volume or the cited three-view fusion paper. These limits are explicit in the individual methods.

## Additional implemented comparisons

| Track | Implementation and evidence |
|---|---|
| Native-pose terrain control | [Heading/cross-track controller and regulated speed](TERRAIN_METHODS_2026-10-10.md), matched against the forward-command baseline across 270 declared development episodes |
| Stereo initialization and recovery | [Native feature diagnostics and independently validated map-frame recovery](STEREO_METHODS_2026-10-10.md), with 18 declared paired navigation episodes |
| Tightly coupled camera–LiDAR–IMU | [Built native FAST-LIVO2 and same-binary visual ablation](NATIVE_LIVO_2026-10-10.md), including retained-data negative results |
| Stereo depth and uncertainty maps | [Fresh 18-cell sensor cohort](SENSOR_METHODS_2026-10-10.md), two real stereo methods, seven map/fusion arms and three native estimators |
| Perceptive terrain policy | [Actual teacher/student PPO, query reconstruction, causal memory and trajectory-aware distillation](PERCEPTIVE_METHODS_2026-10-10.md), three planned training seeds |

The native visual/ablation smoke tracked 33/35 frames in each arm; the visual arm used photometric measurements in 32 frames and the disabled arm used zero. On the already-consumed three-scene diagnostic, FAST-LIO2 qualified 3/3 scenes, FAST-LIVO2 qualified 2/3, and FAST-LIVO2 with vision disabled qualified 3/3. Full visual fusion had higher ATE in every diagnostic scene. That is a measured negative result; native execution success does not establish improvement.

The first terrain shard completed 30/30 episodes for one gait on the new flat course: baseline 8/10 goals, heading 10/10 and regulated 10/10. It is one of nine development shards and does not qualify the complete campaign or physical robot. The stereo smoke completed four short episodes and exercised verified recovery; smoke outcomes are excluded from development scores.

The fresh sensor smoke processed 60 stereo pairs, executed both depth methods and all seven mapping arms, and ran three actual native estimators. CUDA profiling verified the pretrained model's GPU execution. FAST-LIVO2 used visual measurements in 57 frames. The learning smoke trained all four student variants for one iteration, evaluated all four dropout/delay conditions and checked deployment exports. Its teacher did not meet the scientific qualification gate; a smoke execution pass does not imply a qualified policy.

Full campaign controllers were submitted as terrain `21757620`, stereo `21757621`, perceptive learning `21757622`, and sensor mapping/native fusion `21757623`. These are orchestration jobs, not episode counts. Their immutable plans, source hashes and job receipts are retained in [the dispatch evidence directory](../results/methods-campaign-20261010/dispatch/). Each starts distinct predeclared jobs serially after checking its completed same-source smoke. The new learning protocol uses three distinct seed jobs, not an array.

The first full perceptive seed finished **NEGATIVE**: its 200-iteration teacher failed the declared tracking/survival qualification, so all four students for that seed remain unrun. The other seeds are evaluated independently with the same fixed design. Skipped student arms are not recorded as failed or zero-valued trials.

The native stereo–IMU diagnostic processed 450 original pairs in each mode. Stereo tracked 0/150, 0/150 and 122/150; stereo–IMU tracked 0/150 in all three and reported zero initialized frames. The ramp scene's 11 feature-eligible frames each produced the native insufficient-acceleration initialization message. This is a diagnosis on consumed data, not a new test-set result.

Main CPU regression: **2,417 passed, 255 skipped**. The initial eight missing-history failures passed after restoring the historical Git fixtures; the original failure log is retained. Additional native-inertial, fault, fusion, collector and publication contract tests passed in allocated compute. [Verification receipts](../results/methods-campaign-20261010/perceptive-review/verification.json) retain the exact commands, source hashes and limits. Repository CI on the final pushed commit remains a separate check.

Large raw archives are published to the [methods evidence release](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/tag/methods-campaign-evidence-20261010). Each is checked against its completion receipt, uploaded, freshly downloaded and SHA256-verified before duplicate home-directory packaging is retired. Source freezes and compact run receipts remain available. Serial dispatchers share a storage-reservation lock and stop on runtime failure or incomplete evidence; they never silently rerun or tune a failed scientific cohort.

For resume use, the implemented native integrations, measured-input deployment boundary, controlled ablations and reproducible experiment infrastructure are currently supportable contributions. Use performance gains only after the complete matched cohort is collected, and label all reported outcomes as simulation development results. Planned counts, smoke qualification and paper-reported performance are not achieved project results.
