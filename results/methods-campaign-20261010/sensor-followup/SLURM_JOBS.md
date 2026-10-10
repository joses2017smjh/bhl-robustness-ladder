# Native sensor-fault and cost-consistency continuation — predeclared October 10

This continuation implements the calibration/timing/IMU stress and dense-cost/consistency follow-ups from the October 9 result report. It depends on the exact fresh sensor-v2 source archive `207a5129cd3df243f48f6fddccd5f74030ba6a163c9aeb151a119b9d014bf5fe` and protocol `2d648f2ebd1632accfca7438347562dead448d0c759b073d1fa6a80ef62be9ac`.

The required full input shards are sensor jobs 00, 03, 06, 09, 12 and 15: the nominal captures for textured_boxes, thin_posts and ramp_step, each with geometry seeds 610000 and 610001. All are development groups and each must have 150 original frames. No result is counted until source, completion and fresh-download publication hashes match.

- Native matrix: six geometry groups × nominal, IMU clock +20 ms, IMU clock +80 ms, LiDAR yaw +2°/x +2 cm, gyro z bias +0.05 rad/s after5s, accelerometer x bias +0.3m/s² after5s, every-second-return and every-fourth-return conditions; three actual native estimator arms per shard. **48 shards / 144 native cells**.
- Dense fusion matrix: the same six groups × nominal, the declared LiDAR calibration error, every-second-return and every-fourth-return conditions; SGBM, SGBM/local-cost support, sparse LiDAR, equal projected fusion, and cost/consistency fusion. **24 shards / 120 depth/fusion cells**.
- Every shard uses the same148 interior frames, with original IMU retained. Each native arm completes before evaluator truth is opened. Metrics, native transports, output arrays, actual tracking failures and unresolved hazards remain explicit.
- Before full execution, a fresh same-source smoke runs all eight native fault conditions and all four fusion conditions on the retained60-frame capture. Smoke is excluded from full totals and must execute actual native visual measurements. Failure stops the continuation; there is no automatic rerun or parameter selection.
- CPU-only execution uses the unchanged Ubuntu container; host RHEL cannot load FAST-LIVO2's GLIBC_2.35 dependency. All scientific subprocesses are isolated from GitHub credentials. Raw output shards are published, independently downloaded and verified before a full collector verdict.

A preliminary21-test suite passed. Preliminary actual smoke in the Ubuntu container passed six native cells and58-frame five-arm fusion; its nominal and +80ms conditions produced55/58 tracked FAST-LIO2 and57/58 tracked FAST-LIVO2 outputs per arm. These are execution diagnostics, not robustness or performance claims. The first host-environment attempt failed before LIVO inference; its evidence is retained.

Deferred CPU job **21757767** is submitted with `afterany:21757623`, two CPUs,8GB RAM, a12-hour wall limit and no requeue. It independently checks source completion and every published input before running, so scheduler completion alone cannot release scientific work. The plan fixes an11-hour internal budget. Full results remain unrun while the six nominal captures are pending.

- [Frozen deferred plan](plan.json), SHA256 `c63ae34d02ce2c178cfd9ca38c7dab146e74db4164c8233e6a671798fb2706c9`.
- [Submission receipt](submission.json).
- [Frozen source archive](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/methods-campaign-evidence-20261010/sensor-followup-source-20261010-v1.tar.gz),76,683 bytes, SHA256 `703714105c2abda953f303fefb2b026f31a4a43769b0546d400fe2e77a573a43`.
- [Native runtime archive](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/methods-campaign-evidence-20261010/sensor-followup-runtimes-20261010-v1.tar.gz),107,606,864 bytes, SHA256 `dbe3e8cae808f54b8eb88ec07c50e3c5cc34c3220ba2b6ee03b55437b97c7112`.

Both published artifacts passed independent fresh-download SHA256 checks. Final targeted QA passed **31/31 tests**. Preliminary unit step21756762.34 passed21 tests; host smoke step21756762.35 failed on unavailable GLIBC_2.35 before LIVO inference; Ubuntu-container smoke step21756762.36 passed. The queued source adds complete dependency pinning after that preliminary smoke and must pass its own all-condition smoke before full trials. [Preliminary execution record](preliminary-smoke/container-pass/execution.json) remains explicitly smoke-only.

Runtime progress, publications and both strict collection reports will be written under `/nfs/stak/users/sanchej7/humanoid-methods-20261010/sensor-followup-v1`. Original and derived sensor data are processed on node-local scratch. Compact durable records reference independently verified raw release archives. On failure the active shard is published when possible and node-local working evidence is retained; no implicit rerun is performed.
