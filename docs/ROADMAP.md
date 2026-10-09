# Roadmap: pending work and stretch goals

Updated 2026-10-08. The current [task closure ledger](TASK_CLOSURE_2026-10-08.md)
reconciles completed deliverables with open robotics objectives. Historical
plans remain in [SOLUTIONS_2026-10-01.md](SOLUTIONS_2026-10-01.md); evidence is
linked from [STATUS.md](STATUS.md), [REPO_TASKS.md](REPO_TASKS.md) and the
[job ledger](../SLURM_JOBS.md). Acceptance rules remain unchanged.

## In flight

H1 completed and passed its unchanged gate; no closeout campaign remains in
flight. The new [stereo–lidar research ladder](STEREO_LIDAR_RESEARCH_2026-10-08.md)
starts with a dataset/calibration audit before new perception or navigation runs.

## Closed since the previous roadmap

| Item | Final result | Scope |
|---|---|---|
| H1 fresh matched yaw-filter confirmation | DONE, PASS: 576 independently validated episodes; all three actors clear the frozen goal/zero-fall gates | Candidate 259/288 goals and 0/288 falls versus baseline 247/288 and 9/288; mean yaw-command sign-flip rate 51.5% lower. Existing scripted filter, 12-DoF biped, oracle pose/goal, 48 shared layouts; [full evidence](H1_NAVIGATION_CONFIRMATION_2026-10-08.md). |
| H2 portable replay validation | DONE: fresh pinned CPU installation; local full suite 1,998 passed, 123 existing optional checks skipped, six subtests passed; [hosted CI](https://github.com/joses2017smjh/bhl-robustness-ladder/actions/runs/37817975052) succeeds | [Closure receipt](../results/task-closure-20261008/h2-replay/closure.json). Replay retains five exact repeats and 40/40 injected regressions detected; second HPC host agrees on 5/5 frozen two-second cases with maximum numeric difference 0.0. These cases do not establish general cross-platform determinism or a flake rate; original timing scope stays intact. |
| R1 / R2 / R1H turning experiments | DONE NEGATIVE: respectively 1/3, 0/3 and 0/3 seeds meet the unchanged joint rule; R1H seed 2 is complete | H4's repeatable 22-DoF recipe remains open; [verdicts](TASK_CLOSURE_2026-10-08.md#completed-negative-experiments). |
| Original NavGym v5 gym/physics transfer | DONE: three actors pass; physics 11/12, 12/12 and 12/12, zero falls | [Original transfer gate](../results/navgym-v5-transfer-20261002/verdict.json). The later 384-episode confirmation remains NEGATIVE because of six learned-actor falls. |
| NAV-01/02/03 and LOC-01/04/06/07/10 reporting/provenance tasks | DONE within their stated scopes | [Closure evidence](TASK_CLOSURE_2026-10-08.md#deliverables-closed-in-this-audit); media reassembly and caption correction add no simulation episodes. |
| Cloth spawn quaternion | DONE as a code defect; 13 targeted CPU checks and original-literal negative control | [Receipt](../results/task-closure-20261008/cloth-quaternion.json); no new Isaac runtime or task-performance result. |
| PlateCross / PlateCross2 | DONE NEGATIVE: no qualified gait, so no bench released | Mission 7 remains blocked; [selection receipts](TASK_CLOSURE_2026-10-08.md#completed-negative-experiments). |

## Next executable work

| Item | Prerequisite | Acceptance boundary |
|---|---|---|
| **H3: history/latency policy training** | Predeclare matched history-versus-feedforward training and held-out latency evaluation | No training result exists for H3. Sensor-tolerance studies cannot substitute for the proposed matched policy campaign. |
| **H4: reliable turning recipe** | A new observability/control hypothesis after R1H's failed qualification; freeze three seeds and unchanged gates before training | ≥2/3 seeds both pass turn-test v2 and qualify: ≥9/10 turns, ≤15° walk drift on ≥2/3 seeds, push falls ≤9/60. |
| **NAV-04: corrected terrain-sensor media** | Original stereo panels still show pre-fix policies | Rerender the corrected checkpoints and retain source/frame/output hashes. |
| **LOC-03: flatfill terrain retention** | Checkpoint export plus matched difficulty sweep | Complete the existing protocol, preserving all episodes and denominators. |
| **LOC-09: depth departure probe** | Diagnose the repeated rough-terrain timeout | Validated probe output; the unreplicated 2.9% error claim stays withdrawn. |
| **INS-03: larger airlock/stress extension** | Existing extension has ten fresh seeds, not the proposed twenty; delay/interruption requires script options | Report the achieved ten-seed scope separately; predeclare any additional cohort or stress gate. |
| **SF-05: physical IM10A** | User-accessible device and ROS 2 recordings: ≥2 h still at 200 Hz, rotations and tap test | Measured N/B/K, with K measured rather than an upper bound, plus calibration/extrinsics; [recording procedure](IMU_RECORDING.md). |

## Open robotics objectives

| Objective | Current evidence | Next meaningful step | Success rule |
|---|---|---|---|
| **Estimated-pose navigation** | Current navigation uses oracle pose and goal; bounded-error and drift tests describe estimator requirements | Offline visual/stereo-inertial estimator on recorded episodes, compared with truth | ≥10/12 fresh hard mazes with estimated pose and zero falls. |
| **Reliable learned navigation under dropout** | Fresh H1 passes all three actors' gates with the frozen scripted yaw filter; older unfiltered confirmation remains negative | Test calibrated sensor inputs and estimated pose on a separately declared cohort | Preserve H1's complete paired result; new sensors/estimators require their own unchanged goal/fall gates and all failures. |
| **Cube placement / cooperative carry** | Stand4 NEGATIVE, last-200 success 0.006/0.001 versus 0.10; flush-pad probe fixes grip but not wrist-induced rolling | Wrist-orientation/retention/release mechanism and an independently checked success predicate | Placement ≥0.10 on ≥1 of two seeds, with cube tilt and release checked; carry retains its own unchanged rule. |
| **Mission 7 routes** | Scripted benches and learned PlateCross variants are negative; old 10/10 upright replay did not establish genuine staged crossing | Reconcile inconsistent historical episode accounting; establish a new crossing mechanism before any new Mission 7 episodes | Genuine-crossing bench PASS, exact replay 10/10, then Doors and Transport each 16/16 on fresh layouts. Sensor-only/four-sensor studies stay gated. |
| **Sim-to-real** | Simulated sensors and policies only; latency budget approximately 30 ms in tested conditions | SF-05 recordings and a sensor-only hardware bench | A hardware claim backed by recorded device/runtime evidence. |

The separate Waiter work uses a modified 24-DoF gripper robot and different
arm limits. The immutable WBC/WBC1b backup selections are negative. Later
unpublished local WBC1c receipts select seed 7; seeds 7 and 8 pass v2,
qualification and Q2, with 0/12 Q2 falls and four walks each ≥1.5 m. This audit
inspected saved evidence without reproducing that simulation. An immutable
source/evidence publication remains a follow-up; neither branch closes H4 or
the later VLA phases. See the [separate evidence review](TASK_CLOSURE_2026-10-08.md#separate-waiter-backup-branch).
