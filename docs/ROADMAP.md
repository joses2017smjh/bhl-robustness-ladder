# Roadmap: pending work and stretch goals

Updated 2026-09-30. Done since the last update: README hero with sensor panels; NavGym v4 passed its gate. One line per item; evidence and rules live in
[`STATUS.md`](STATUS.md), [`REPO_TASKS.md`](REPO_TASKS.md) and the
[job ledger](../SLURM_JOBS.md). Every run gets a pass rule written down before it starts.

## In flight

| Item | State | Done when |
|---|---|---|

## Next (small, unblocked)

| Item | Why | Done when |
|---|---|---|
| **SF-05: record the real IM10A** over ROS 2 (10 min still, then slow rotations per axis) | replace the datasheet noise in the sim IMU with measured noise, bias and rate | Allan-variance sigmas committed; sim IMU and panel read "measured on your IM10A" |
| Humanoid maze clip (TurnBoth-s0, 12/12) | the result has no GIF yet | GIF reproduces a scored episode |
| Fix the cloth task's spawn quaternion | same bug as the fixed cube tasks; guard test allowlists it | cloth driver removes the allowlist entry |

## Stretch goals

| Goal | Current evidence | First step | Success looks like |
|---|---|---|---|
| **Drop the oracle pose**: visual-inertial odometry (optical flow / stereo + IMU) drives the randomized maze | maze works 24/24 with oracle pose; pose error > 0.05 m already breaks goal judging | offline VO from the rig renders + sim IMU on recorded episodes; compare to truth | ≥ 10/12 hard mazes with estimated pose, 0 falls |
| **A turning humanoid recipe**, not one lucky checkpoint | 1/12 seeds qualifies; fine-tunes trade turning for push robustness (v5) | train turn + push together from the start, 3 seeds | ≥ 2/3 seeds pass the turn test and push falls ≤ 0.15 |
| **Learned navigation that passes its own gate**, then runs the physics robot | gym gate **PASSED** (v4, 2/3 seeds); physics transfer **8/12 and 9/12** (need ≥ 10), 0 falls, misses are time-outs | close the gym-to-physics gap: train with the gait's measured response lag and the brake in the gym | ≥ 10/12 never-seen mazes on the physics biped, each passing actor |
| **Cube placement** | Stand3 stands but never places; replay: checkpoints fail differently | pick lowering-vs-reach only when evidence agrees | success ≥ 0.10 on ≥ 1 of 2 seeds |
| **Sim-to-real** on the Berkeley Humanoid Lite with lidar, stereo and IM10A | sim only; IMU latency budget ≈ 30 ms | SF-05 recordings, then a sensor-only bench check | a hardware claim backed by recorded data |
| Mission 7 route controller | blocked at 9/10 in replay (gate 10/10) | a new crossing idea | 10/10 in replay, then fresh layouts |
