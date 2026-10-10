# Humanoid research methods: implementation and campaign status

First-stage executable methods now address all six research themes, with a separate perceptive-learning experiment added. These cover narrower initial experiments than the complete original designs; several matched controls, acquisition/fault conditions and calibrated health gates remain unimplemented. Implementation, successful execution and measured improvement are recorded separately. Remaining development campaigns are queued or running, and some initialization/learning results are already negative. Earlier results in [the native campaign report](NATIVE_CAMPAIGN_RESULTS_2026-10-09.md) remain unchanged.

## Coverage of the original six follow-ups

| Follow-up | Concrete implementation | Execution boundary |
|---|---|---|
| Explain stereo startup failures | Read-only native features, valid stereo-depth matches, state and map diagnostics | All 450 consumed frames replayed; both failed scenes remain below the unchanged feature threshold |
| Test stereo–IMU navigation | [Actual native IMU_STEREO transport, calibrated original IMU batches and initialization-gated navigation](STEREO_INERTIAL_METHODS_2026-10-10.md) | Full 900-frame paired replay and four-episode navigation smoke completed; inertial initialization failed in both; 18 development episodes remain unrun |
| Stress LiDAR–IMU estimation | [Clock offsets, extrinsic error, gyro/accelerometer bias and return loss](SENSOR_FAULT_METHODS_2026-10-10.md) | Native smoke complete; 48 full replay shards / 144 estimator cells queued in continuation `21757767` after fresh capture |
| Qualify estimated-heading terrain control | [Native-pose pursuit and regulated-speed controls](TERRAIN_METHODS_2026-10-10.md) | Two 30-episode shards complete; remaining seven queued, 270 total declared |
| Measure dense stereo–LiDAR fusion | [Pretrained stereo and uncertainty maps](SENSOR_METHODS_2026-10-10.md), plus [local photometric-cost/consistency fusion](SENSOR_FAULT_METHODS_2026-10-10.md) | Actual GPU/native smoke complete; 18 fresh capture cells, followed by 24 five-arm cost/consistency shards |
| Test sensor-assisted recovery | [Disjoint estimated-pose registration and validation across native stereo map changes](STEREO_METHODS_2026-10-10.md) | Actual recovery exercised in smoke; 18 matched navigation episodes queued |

## Remaining original design scope

These items are not part of the queued campaigns and must not be described as implemented or completed:

- Initialization diagnostics still need quantitative epipolar and calibration-residual analysis.
- The new stereo–IMU navigation comparison has two arms. A matched LiDAR–IMU arm, camera-rate/exposure/motion-blur conditions and navigation IMU perturbations remain unimplemented. Current cameras are ideal simulated sensors at 5 Hz.
- Native sensor-fault stress is replay-only. Closed-loop goals/recovery and outcome-calibrated health gates remain unimplemented; existing thresholds are engineering readiness gates.
- Map recovery compares permanent stop with LiDAR-assisted stereo recovery. Each-estimator-alone controls, matched visual/LiDAR/IMU fault injection, and complete recovery-time/false-free-space evaluation remain unfinished.
- Dense fusion still needs a controlled occlusion condition. The disclosed local photometric-cost method does not reproduce an aggregated SGM cost volume or the cited three-view fusion paper.
- Independent terrain confirmation and hardware validation remain unrun. Confirmation requires the complete development cohort and its qualification gate.

The negative initialization and teacher outcomes remain valid results. Their dependent stages cannot be counted as completed trials or promoted by weakening the frozen gates.

## Additional implemented comparisons

| Track | Implementation and evidence |
|---|---|
| Native-pose terrain control | [Heading/cross-track controller and regulated speed](TERRAIN_METHODS_2026-10-10.md), matched against the forward-command baseline across 270 declared development episodes |
| Stereo initialization and recovery | [Native feature diagnostics and independently validated map-frame recovery](STEREO_METHODS_2026-10-10.md), with 18 declared paired navigation episodes |
| Tightly coupled camera–LiDAR–IMU | [Built native FAST-LIVO2 and same-binary visual ablation](NATIVE_LIVO_2026-10-10.md), including retained-data negative results |
| Stereo depth and uncertainty maps | [Fresh 18-cell sensor cohort](SENSOR_METHODS_2026-10-10.md), two real stereo methods, seven map/fusion arms and three native estimators |
| Perceptive terrain policy | [Actual teacher/student PPO, query reconstruction, causal memory and trajectory-aware distillation](PERCEPTIVE_METHODS_2026-10-10.md), three completed teacher seeds; all failed qualification, so full student arms remain unrun |

The native visual/ablation smoke tracked 33/35 frames in each arm; the visual arm used photometric measurements in 32 frames and the disabled arm used zero. On the already-consumed three-scene diagnostic, FAST-LIO2 qualified 3/3 scenes, FAST-LIVO2 qualified 2/3, and FAST-LIVO2 with vision disabled qualified 3/3. Full visual fusion had higher ATE in every diagnostic scene. That is a measured negative result; native execution success does not establish improvement.

The first two terrain shards completed 60/60 episodes for one gait. On the new flat course, clean goals were baseline 8/10, heading 10/10 and regulated 10/10. On the 1 cm steps, clean goals were baseline 0/10, heading 10/10 and regulated 10/10; contact episodes were 9, 0 and 0 respectively, with no falls. These are two of nine development shards and do not qualify the complete campaign or physical robot. The stereo recovery smoke completed four short episodes and exercised verified recovery; smoke outcomes are excluded from development scores.

The fresh sensor smoke processed 60 stereo pairs, executed both depth methods and all seven mapping arms, and ran three actual native estimators. CUDA profiling verified the pretrained model's GPU execution. FAST-LIVO2 used visual measurements in 57 frames. The learning smoke trained all four student variants for one iteration, evaluated all four dropout/delay conditions and checked deployment exports. Its teacher did not meet the scientific qualification gate; a smoke execution pass does not imply a qualified policy.

Full campaign controllers were submitted as terrain `21757620`, stereo `21757621`, perceptive learning `21757622`, and sensor mapping/native fusion `21757623`. These are orchestration jobs, not episode counts. Their immutable plans, source hashes and job receipts are retained in [the dispatch evidence directory](../results/methods-campaign-20261010/dispatch/). Each starts distinct predeclared jobs serially after checking its completed same-source smoke. The new learning protocol uses three distinct seed jobs, not an array.

All three full perceptive seeds finished **NEGATIVE**: each 200-iteration teacher failed the declared tracking/survival qualification, so all twelve full student arms remain unrun. The [independent three-seed collection](../results/methods-campaign-20261010/perceptive-review/completed-three-seed-audit/) verified every source, input and freshly downloaded raw archive with no provenance problems. Qualified episodes were 113/256 (44.14%), 102/256 (39.84%) and 107/256 (41.80%), below the preset 80% gate; survival alone was 236/256, 224/256 and 216/256. The fixed design was evaluated independently across the three seeds without adding iterations after failure. Skipped student arms are not recorded as failed or zero-valued trials.

The native stereo–IMU diagnostic processed 450 original pairs in each mode. Stereo tracked 0/150, 0/150 and 122/150; stereo–IMU tracked 0/150 in all three and reported zero initialized frames. The ramp scene's 11 feature-eligible frames each produced the native insufficient-acceleration initialization message. This is a diagnosis on consumed data, not a new test-set result.

Fresh navigation smoke `21757768` completed all four declared episodes with runtime exit 0. Neither of its two stereo–IMU episodes initialized, so the scientific readiness gate failed. The wrapper's nonzero exit records that negative gate, not an execution failure. All four smoke episodes had zero goals, falls and contacts; their short horizons are not development goal trials. The eighteen separately gated development episodes remain unrun.

Local CPU regression: **2,417 passed, 255 skipped**. The initial eight missing-history failures passed after restoring the historical Git fixtures; the original failure log is retained. Additional native-inertial, fault, fusion, collector and publication contract tests passed in allocated compute. [Verification receipts](../results/methods-campaign-20261010/perceptive-review/verification.json) retain the exact commands, source hashes and limits. The [GitHub CI run for implementation commit `c1d9df1`](https://github.com/joses2017smjh/bhl-robustness-ladder/actions/runs/38088753138) passed **2,648 tests, with 124 skipped and 10 additional passing subtests**, followed by actual trained-asset checks and seeded physics/fault replay. [Downloaded CI receipts](../results/methods-campaign-20261010/ci/c1d9df1/) preserve the exact commit and results.

Large raw archives are published to the [methods evidence release](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/tag/methods-campaign-evidence-20261010). Each is checked against its completion receipt, uploaded, freshly downloaded and SHA256-verified before duplicate home-directory packaging is retired. After all three perceptive jobs and their controller completed, their exact 40 MB source freeze was also published and independently verified before retiring its duplicate home copy. Compact receipts remain durable; finalizer `21757776` restores this exact source into node-local temporary storage and validates its complete manifest. Serial dispatchers share a storage-reservation lock and stop on runtime failure or incomplete evidence; they never silently rerun or tune a failed scientific cohort.

Sensor-fault/fusion continuation `21757767` independently collects its 144 native estimator cells and 120 fusion-arm cells. Terminal observer `21759391`, queued after that continuation, verifies its final metadata and all 73 expected raw publications, then publishes the summaries to the same release with a fresh-download checksum. Missing or failed evidence remains explicitly incomplete. This observer adds no scientific trials and does not change the frozen continuation.

For resume use, the implemented native integrations, measured-input deployment boundary, controlled ablations and reproducible experiment infrastructure are currently supportable contributions. Use performance gains only after the complete matched cohort is collected, and label all reported outcomes as simulation development results. Planned counts, smoke qualification and paper-reported performance are not achieved project results.
