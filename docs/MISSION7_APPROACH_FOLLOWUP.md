# Mission 7 follow-up: contact-safe control and privileged Approach

This follow-up continues the completed diagnosis in
[`MISSION7_APPROACH_DEBUG.md`](MISSION7_APPROACH_DEBUG.md). The six PPO studies,
benchmark geometry, and sensor comparison remain closed.

## Current status

The detailed contact probe ran on the same Haswell node (`cn-c22`) used for
the exact replay. Its first completed instrumentation pass was rejected:
extra per-contact force calls changed some replay trajectories. The probe was
then changed to reuse the existing single `mj_contactForce` call and to require
an explicit pose-match check before interpretation. The corrected filtered
probe then matched all ten retained episodes with maximum pose error `0.0`.

The first balanced privileged controller matrix reached 55/64. A replacement
using the measured 0.50 m/s translation command and the bounded 0.30 m/s stall
pulse reached 56/64 in the completed Approach portion, with 8/16 in world −x
and 16/16 in each other direction. Those failures are repeated goal-post
contacts before entering the goal region. The matched standstill control was
0/64. The run is retained as below-gate evidence; no sensor learning was
started.

The corrected contact probe, friction intervention, Approach matrix, and both
16-episode route batches are complete. The results below are diagnostic
evidence, not a passed qualification gate.

## Contact dynamics hypothesis

The valid ten-episode trace shows that every retained failure contacted one or
more raised plates before falling. The first touches have negative contact
distance (penetration), tilted normals during edge contact, and large impulses.
The filtered probe reused the existing force call and reproduced every original
trajectory exactly (`maximum_pose_difference = 0.0` for all 10 episodes):

| Quantity | Observed range |
|---|---:|
| penetration | 0.00003–0.036 m |
| normal force | 0–2,015 N |
| tangential force | 0–420 N |
| friction coefficients | `(1.0, 0.005, 0.0001)` |
| solver reference | `(0.02, 1.0)` |
| solver impedance | `(0.9, 0.95, 0.001, 0.5, 2.0)` |

Every retained episode contacted one or more plates before its recorded fall;
the valid unchanged replay reproduced all 10/10 falls. Plate geometry stays
enabled and unchanged. The evidence supports plate penetration/impulse as a
common interaction state, while not by itself proving that force alone causes
the later fall. A replay with plate sliding friction changed from `1.0` to
`0.20` also produced 10/10 falls, so that parameter-only intervention is
rejected.

## Candidate mitigation

`PlateSafeRouteController` keeps the measured route controller and official
button semantics, then holds zero translation for 1.0 s after physical plate
contact or gate activation. The exact ten retained failures are replayed first;
the same controller is then evaluated on 16 Doors and 16 Transport episodes.
The mitigation is accepted only if the ten replay traces remain upright and the
broader route results are reported separately. The completed mitigation replay
was 8/10 upright; its 16-episode Doors route was 1/16 success (4 falls,
11 timeouts). It therefore does not pass the pressure-plate gate. The Transport
array uses the same controller and produced 1/16 success, 13 falls, and 2
timeouts; it cannot retroactively make the exact-replay gate pass.

## Approach controller

The evaluated command interface uses the measured gait response: a nonzero
translation command at or above the calibrated onset range, explicit zero
braking and dwell, heading correction from privileged pose, and one bounded
0.30 m/s forward restart pulse when lateral motion stalls. The evaluation uses
16 layouts per target direction from held-out validation/test geometry, with
matched standstill episodes on identical reset streams.

The completed 64-episode matrix remains below the required 60/64 and
14/16-per-direction gate. Its failure breakdown is concentrated in world −x
goal-post contact; the other directions passed in that run. This is a
controller and clearance issue, not evidence that a sensor policy is ready.

## Gates

| Gate | Status | Evidence |
|---|---|---|
| Pressure-plate exact replay, 10/10 upright | **FAIL for current mitigation (8/10)**; valid unchanged probe 10/10 reproduced; friction-only replay 0/10 upright |
| Doors ≥16 and Transport ≥16 | Doors **1/16** (4 falls, 11 timeouts); Transport **1/16** (13 falls, 2 timeouts) |
| Privileged Approach ≥60/64, zero falls, ≥14/16/direction | **FAIL: 56/64**, 0 falls; world −x **8/16**, other directions 16/16 |
| Sensor-only Both readiness | **CLOSED** until both preceding gates pass |
| Four-sensor comparison | **CLOSED** |

## Next steps

1. Keep the sensor-only and four-sensor studies closed. Neither prerequisite
   control gate passed, and no new sensor jobs are queued.
2. If Mission 7 continues, investigate a controller-side clearance strategy
   for the world −x goal-post approach and a contact-safe plate maneuver. Any
   candidate must first reproduce the exact ten retained layouts at 10/10
   upright with plates enabled and unchanged geometry.
3. Re-run the 16 Doors and 16 Transport route batches only after that exact
   replay gate passes. Release sensor comparisons only after the privileged
   Approach gate reaches at least 60/64, zero falls, and at least 14/16 in each
   direction.

The complete route aggregate is
[`transport_summary.json`](../results/mission7-approach-followup-20260921/transport_array/transport_summary.json);
the valid contact evidence is in
[`contact_probe-rerun5/result.json`](../results/mission7-approach-followup-20260921/contact_probe-rerun5/result.json),
and the Approach matrix is in
[`approach_controller_speed05/result.json`](../results/mission7-approach-followup-20260921/approach_controller_speed05/result.json).

## Slurm receipts

All follow-up tasks request 2 CPUs, 12 GB, zero GPUs, and run on `cn-c22` for
paired physics. Invalid launches are preserved as infrastructure failures and
were canceled before consuming a scientific result.

| Job | Purpose | State/result |
|---:|---|---|
| 21385772 | contact probe, bad snapshot path | canceled infrastructure attempt |
| 21386044 | Approach, incomplete snapshot | canceled infrastructure attempt |
| 21386072 | contact probe, wrong input campaign | failed infrastructure attempt |
| 21386074 | Approach 0.40 m/s matrix | canceled after 55/64 below gate |
| 21386084 | first detailed contact probe | complete, instrumentation replay rejected |
| 21386145 | Approach 0.50 m/s + standstill controls | complete, 56/64; 0 falls; gate failed |
| 21386200 | plate-safe replay and routes | failed after replay; route controller initialization bug |
| 21386394 | corrected pre-contact replay and routes | superseded by corrected `21386488`; retained as intermediate |
| 21386215 | pose-matched contact probe, floor-filtered | canceled after first exact episode to bound trace size |
| 21386409 | floor-filtered pose-matched contact probe | superseded by corrected full-reset probe |
| 21386441 | plate sliding-friction intervention replay | canceled after reset-stream correction |
| 21386540 | corrected plate sliding-friction replay | complete, 10/10 falls; intervention rejected |
| 21386539 | corrected full-reset pose-matched contact probe | complete; 10/10 exact replay matches, 10/10 falls |
| 21386488 | corrected PlateSafe replay and Doors route | complete; replay 8/10 upright, Doors 1/16; canceled during partial Transport |
| 21386803 | parallel 16-episode Transport route array | complete; 1/16 success, 13 falls, 2 timeouts |

No GPU resources were used by these diagnostics.
