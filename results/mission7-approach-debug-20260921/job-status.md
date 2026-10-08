## Submitted diagnostic jobs

Snapshot: 2026-09-21T22:34:26.052020+00:00. All tasks: 2 CPUs, 12 GB, zero GPUs, two-hour cap.
13 submissions /19 tasks; 11.96 allocated CPU-hours so far.

| Job | Study | Scheduler state | Scheduler dependency | Output under campaign |
|---|---|---|---|---|
| 21383807_0 | gait | COMPLETED | none | `gait-0/` |
| 21383807_1 | gait | COMPLETED | none | `gait-1/` |
| 21383807_2 | gait | COMPLETED | none | `gait-2/` |
| 21383808 | c4 | COMPLETED | none | `c4/` |
| 21384040 | gait_extra | COMPLETED | none | `gait_extra/` |
| 21384041 | approach | COMPLETED | afterok:21383807 | `approach/` |
| 21384042 | rewards | COMPLETED | afterok:21383807 | `rewards/` |
| 21384077 | fullroute | COMPLETED | afterok:21384040 | `fullroute/` |
| 21384200 | approach_recovery | COMPLETED | none | `approach_recovery/` |
| 21384285_0 | train | COMPLETED | none | `P1/` |
| 21384285_1 | train | COMPLETED | none | `P2/` |
| 21384285_2 | train | COMPLETED | none | `P3/` |
| 21384285_3 | train | COMPLETED | none | `P4/` |
| 21384354 | fullroute_recovery | COMPLETED | none | `fullroute_recovery/` |
| 21384572_0 | train_exploration | COMPLETED | none | `P5/` |
| 21384572_1 | train_exploration | COMPLETED | none | `P6/` |
| 21384641 | gait_endurance | COMPLETED | none | `gait_endurance/` |
| 21384691 | fall_replay | CANCELLED by 19646 | none | `fall_replay/` |
| 21384742 | fall_replay_retry | COMPLETED | none | `replay-on-original-node/fall_replay/` |

PPO has completed logical prerequisites in `feasibility-gate.json` and a runtime guard; no expired scheduler dependency was attached. No sensor jobs were submitted.
