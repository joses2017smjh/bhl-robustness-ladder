# Stereo recovery campaign: results (2026-10-14)

**Verdict: NEGATIVE.** Resuming after a verified same-map relocalization did not complete any more goals than
stopping permanently.

**Scope.** Simulated sensors, one frozen gait policy, four seed clusters. This is development evidence, not
confirmation, and the result does not validate physical navigation.

- **Campaign:** `followups-20261014-stereo/stereo-recovery-v2-20261014`, job 21825868, COMPLETED 0:0 in 1h14m on
  cn-gpu5.
- **Frozen source:** `7638a64`, archive `19337226…`.
- **Completeness:** all 24 pairs measured, smoke PASS, collector status PASS, scientific status
  `COMPLETE_BOUNDED_DEVELOPMENT_NO_CONFIRMATION`.
- **Compact result:** `results/stereo-recovery-20261014/campaign-v2-result-21825868.json`.

## Primary outcome (predeclared)

| Arm | Clean goals | Wilson 95 % |
|---|---|---|
| `permanent_stop` (v1 logic) | 0 / 24 | 0-0.138 |
| `resume_verified_relocalization` | 0 / 24 | 0-0.138 |

- Exact two-sided McNemar: 0 discordant pairs, p = 1.0.
- Seed-cluster bootstrap of the goal difference (256 resamples): 0, interval [0, 0].
- The predeclared rule needed at least 6 resume-only goals.

## Secondary outcomes

| | `permanent_stop` | `resume_verified_relocalization` |
|---|---|---|
| Tracking losses | 33 | 36 |
| Losses by mechanism | same-map relocalization 25, new map 4, map reset 4 | same-map relocalization 24, new map 8, map reset 4 |
| Verified resumes | 0 | 21 |
| Completed navigation recoveries | 17 | 21 |
| Right-censored losses (never recovered) | 16 | 15 |
| Median recovery time | 0.20 s | 0.40 s |
| Mean stopped navigation time (of 40 s) | 24.8 s | 24.7 s |
| Falls / collisions | 0 / 0 | 0 / 0 |

- Stopped-time difference, resume minus control: −0.12 s. Seed-bootstrap interval [−0.90, +0.73]; exact sign test
  p = 0.82 over 20 nonzero pairs.
- Mean stopped time by condition, control / resume:
  - nominal: 21.5 / 22.2 s;
  - camera 10 Hz: 26.2 / 24.7 s;
  - 40 ms exposure: 26.8 / 27.2 s.
- Safety: no regression.
- The largest accepted resumed-pose error was 0.41 m, within the motion-scaled bound, which grows with commanded path
  length.

## What this shows

- **Resuming works but is not enough.** The resume arm resumed 21 times with no falls or collisions, yet stopped just
  as long. Losses recur, and a later new-map loss (8 in this arm) or map reset still ends the episode, before the robot
  reaches the goal in every pair.
- **The stop rule is not the bottleneck.** v1's 0/12 is reproduced at 0/24 under both rules.
- **The next work is loss frequency, not loss handling.** That means keeping ORB-SLAM3 tracking through the gait,
  through occluders and at 10 Hz or 40 ms exposure, or adding a fallback such as stereo-inertial or LiDAR odometry
  while vision is lost.

## Not shown

- Behaviour with a different policy, real sensors, or the deferred `stereo_inertial` arm.
- Whether a looser resume bound would help. It would trade safety margin for resumes, and resumes did not move goals
  here.
