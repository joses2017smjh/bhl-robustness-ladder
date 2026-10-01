# Roadmap: pending work and stretch goals

Updated 2026-10-01. Done since the last update: README hero is now the 22-DoF humanoid with sensor panels; NavGym v4 passed its gate; the solutions plan for the five open problems is [`SOLUTIONS_2026-10-01.md`](SOLUTIONS_2026-10-01.md). One line per item; evidence and rules live in
[`STATUS.md`](STATUS.md), [`REPO_TASKS.md`](REPO_TASKS.md) and the
[job ledger](../SLURM_JOBS.md). Every run gets a pass rule written down before it starts.

## In flight

| Item | State | Done when |
|---|---|---|

## Next (small, unblocked)

| Item | Why | Done when |
|---|---|---|
| **SF-05: record the real IM10A** over ROS 2 (≥ 2 h still at 200 Hz, then slow rotations per axis and a tap test; procedure [`IMU_RECORDING.md`](IMU_RECORDING.md), analysis `scripts/sensors/imu_allan.py`) | replace the datasheet noise in the sim IMU with measured noise, bias and rate; a 30 min record reads B about 10 % low (min method) and usually loses K | Allan-variance sigmas (N, B and K, with K measured rather than an upper bound, as `imu_measured.json`) committed; sim IMU and panel read "measured on your IM10A" |
| Fix the cloth task's spawn quaternion | same bug as the fixed cube tasks; guard test allowlists it | cloth driver removes the allowlist entry |

## Stretch goals

| Goal | Current evidence | First step | Success looks like |
|---|---|---|---|
| **Drop the oracle pose**: visual-inertial odometry (optical flow / stereo + IMU) drives the randomized maze | maze works 24/24 with oracle pose; pose error > 0.05 m already breaks goal judging | offline VO from the rig renders + sim IMU on recorded episodes; compare to truth | ≥ 10/12 hard mazes with estimated pose, 0 falls |
| **A turning humanoid recipe**, not one lucky checkpoint | 1/12 seeds qualifies (TurnBoth-s0, via v2x; noise-free v2 0/12); in MuJoCo four non-turners turn in 16/16 runs with their own training action noise, which points to PPO learning to step only in the noisy loop while the noise-free gates see the standing mean; fine-tunes trade turning for push robustness (v5) | gait clock + contact-schedule reward paid at every command, pushes from the start; clock in the actor (R1) and in the critic only (R2), 3 seeds each, gate unchanged ([plan](SOLUTIONS_2026-10-01.md)) | ≥ 2/3 seeds pass the turn test and push falls ≤ 0.15 |
| **Learned navigation that passes its own gate**, then runs the physics robot | gym gate **PASSED** (v4, 2/3 seeds); physics transfer **8/12 and 9/12** (need ≥ 10), 0 falls, misses are time-outs; on the same layouts in the gym s6 fails the same 3 mazes (its misses are its own) and s5 loses 2–3 more in physics; maze 50009's dead-end pocket traps both actors in the gym and s6 in physics, while s5 in physics turns up the x = 2 column and dithers (longest stall 87 s) | integrating each lidar packet at its capture pose was screened NEGATIVE in physics (`21501509`: s5 net +2/24, needed +4; s6 23 → 23; 0 falls), so next: route-level navigation (an A* sub-goal hybrid, labelled learned + scripted, or a visitation channel / recurrent policy), judged on fresh maze seeds | ≥ 10/12 never-seen mazes on the physics biped, each passing actor (a hybrid does not satisfy this row; it is reported separately as learned + scripted) |
| **Cube placement** | Stand3 stands but never places; replay `21470828`: both checkpoints earn the lift reward by re-orienting a cube that rests on its support (`object_is_lifted` tests centre height only); scripted lift-place fails because the cube rolls in the hands | scripted: a flush-pad wrist mechanism probe on exploration seeds; learned: a roll-proof lift reward and a hand-collider check, then a Stand4 with its rule written first | success ≥ 0.10 on ≥ 1 of 2 seeds, with cube tilt and release checked |
| **Sim-to-real** on the Berkeley Humanoid Lite with lidar, stereo and IM10A | sim only; IMU latency budget ≈ 30 ms; the hardware runtime writes the deploy limits (6 Nm legs, 4 Nm arms) into motor firmware, so fixes needing more torque are sim-only | SF-05 recordings with the kit ([`IMU_RECORDING.md`](IMU_RECORDING.md)), then a sensor-only bench check | a hardware claim backed by recorded data |
| Mission 7 route controller | blocked at 9/10 in replay (gate 10/10), the best of the cross-clear variants, which keep crossing until the plate is cleared; four fixed-1.2 s-crossing variants pass 10/10, but in G-ref `21397732` and wait-open `21400863` (identical crossing for crossing) the staged crossing never cleared a plate and door 0 was already open, and the two lateral-0.35 passes are unverified, so the replay cannot certify a crossing; the route gate's code counts episodes, not successes; the episode budget is inconsistent (101 left vs ≈ 53 over) | route-gate success criterion and real crossing clears now coded (M1, 2026-10-01); reconcile the budget (0 episodes); then, with a new budget line, a plate bench with TurnBoth-s0 as the stage gait (turn to the door, cross forward) | bench PASS, then 10/10 in replay, then Doors and Transport 16/16 on fresh layouts |
