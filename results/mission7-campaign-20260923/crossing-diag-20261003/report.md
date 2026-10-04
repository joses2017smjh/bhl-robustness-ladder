# Mission 7 crossing diagnosis

Design: SLURM_JOBS.md, 'User approval recorded 2026-10-03 09:55', item (D) Mission 7 crossing diagnosis; 'Correction to design (D)' (layouts 170-249); 'Coordinator re-freeze of (D)'s measurement and classifier, 2026-10-03 11:40'. Investigation, no gate.

Labels: gaits: LEARNED (shipped arms-dr1.0-s0 outside the stage; during the stage the controller is swapped to arms-turngait-clock-s2 = clock-s2, Turning R1's qualified seed, a 77-observation gait-clock policy); stage: SCRIPTED (PlateStage, stage_gait=clocks2, M2's turnboth law; bench v2's run_crossing unchanged); layout_and_plate_pose: ORACLE (as the existing stage uses them); purpose: DIAGNOSIS ONLY (investigation, no gate; nothing here is a bench verdict).

## Result

- Crossings run: 80 (clears 37, falls 2, no cross phase 6).
- Stalls (edge dwell >= 2.0 s or never past -0.20 m): 21; non-fall 21, fell 0 (excluded); 2 never past; 4 still cleared.
- Criteria overlaps among the non-fall stalls: {"per_criterion": {"wall_blocked": 0, "blocked_step_up": 1, "slow_progress": 4}, "combinations": {"blocked_step_up": 1, "none": 16, "slow_progress": 4}}.
- `M7_CROSSING_DIAG STALLS non-fall 21 (fall stalls 0, excluded; crossings that fell 2) class_counts_non_fall {"wall_blocked": 0, "blocked_step_up": 1, "slow_progress": 4, "other": 16} class_counts_fall_stalls {"wall_blocked": 0, "blocked_step_up": 0, "slow_progress": 0, "other": 0}`
- `M7_CROSSING_DIAG BLOCKED_STEP_UP_FRACTION non-fall 1/21 = 0.047619; including fall stalls 1/21 = 0.047619`
- `M7_CROSSING_DIAG SPECIFICITY clears meeting the blocked-step-up criterion on their own edge dwell 0/37 = 0.000000`
- `M7_CROSSING_DIAG DECISION F1 NOT TRAINED (non-fall blocked fraction 1/21 < 1/3); non-fall blocked-step-up fraction 1/21 = 0.047619; clear share meeting the criterion 0/37 = 0.000000`

## Layouts

- Declared 170-249 (80 layouts); absent from the train split (size 256): 0.
- Grid crossings 160, dropped by bench v2's drop rule 61, capped out 19 (the kept crossings in grid order (layout ascending, door 0 before door 1), after bench v2's drop rule; the first 80 are run, the rest are listed as capped out and never run), run 80.

## Per crossing

| crossing | heading | plate | entry | end | fell | clear (s after takeover) | edge dwell s | stall | class | criteria (wall/blocked/slow) | blocked swings W / first 2 s | swings W (leading) | wall steps / W steps | mean along speed / cmd vx |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L170d1 | -90 | round | standstill | complete | no | yes (8.84) | 2.16 | yes | slow_progress | n/n/y | 0 / 0 | 7 (5) | 3 / 55 | 0.106 / 0.294 |
| L171d1 | 180 | round | standstill | complete | no | no (12.12) | 1.64 | no | (control: other) | n/n/n | 0 / 0 | 4 (4) | 0 / 42 | 0.178 / 0.292 |
| L172d1 | 0 | square | standstill | complete | no | yes (4.76) | 1.72 | no | (control: other) | n/n/n | 2 / 2 | 6 (5) | 0 / 44 | 0.064 / 0.292 |
| L173d1 | 90 | square | standstill | complete | no | yes (8.08) | 0.64 | no | (control: other) | n/n/n | 2 / 2 | 2 (2) | 0 / 17 | 0.411 / 0.282 |
| L174d1 | -90 | square | standstill | complete | no | no (-) | never | yes | other | n/n/n | 0 / 0 | 13 (8) | 10 / 101 | 0.053 / 0.293 |
| L175d1 | 180 | square | standstill | complete | no | no (12.04) | 1.12 | no | (control: other) | n/n/n | 1 / 1 | 3 (2) | 0 / 29 | 0.265 / 0.288 |
| L176d0 | 0 | round | standstill | complete | no | yes (3.08) | 0.84 | no | (control: other) | n/n/n | 0 / 1 | 1 (1) | 1 / 22 | 0.289 / 0.284 |
| L177d0 | 90 | round | standstill | complete | no | no (-) | 1.48 | no | (control: other) | n/n/n | 0 / 0 | 6 (6) | 0 / 38 | 0.257 / 0.291 |
| L178d0 | -90 | round | standstill | complete | no | no (13.24) | 1.40 | no | (control: other) | n/n/n | 0 / 0 | 5 (4) | 0 / 36 | 0.187 / 0.291 |
| L179d0 | 180 | round | standstill | complete | no | no (13.92) | 2.12 | yes | slow_progress | n/n/y | 0 / 0 | 6 (6) | 0 / 54 | 0.062 / 0.294 |
| L180d0 | 0 | square | standstill | complete | no | yes (3.52) | 1.32 | no | (control: other) | n/n/n | 0 / 0 | 3 (3) | 3 / 34 | 0.182 / 0.289 |
| L181d0 | 90 | square | standstill | complete | no | yes (8.72) | 1.00 | no | (control: other) | n/n/n | 1 / 1 | 3 (2) | 0 / 26 | 0.213 / 0.286 |
| L182d0 | -90 | square | standstill | complete | no | no (-) | 4.00 | yes | other | n/n/n | 0 / 0 | 12 (10) | 11 / 101 | 0.079 / 0.296 |
| L183d0 | 180 | square | standstill | complete | no | no (13.60) | 2.72 | yes | other | n/n/n | 0 / 0 | 9 (7) | 0 / 69 | 0.078 / 0.294 |
| L184d1 | 0 | round | standstill | complete | no | yes (5.24) | 1.08 | no | (control: other) | n/n/n | 1 / 1 | 4 (4) | 1 / 28 | 0.214 / 0.289 |
| L185d0 | 90 | round | walking | complete | no | no (-) | 3.00 | yes | other | n/n/n | 0 / 0 | 8 (7) | 7 / 76 | 0.115 / 0.293 |
| L185d1 | 90 | round | standstill | complete | no | yes (7.20) | 0.64 | no | (control: other) | n/n/n | 0 / 0 | 2 (1) | 0 / 17 | 0.335 / 0.280 |
| L186d0 | -90 | round | walking | complete | no | no (10.40) | 1.16 | no | (control: other) | n/n/n | 0 / 0 | 3 (3) | 0 / 30 | 0.184 / 0.289 |
| L186d1 | -90 | round | standstill | complete | no | yes (8.96) | 0.60 | no | (control: other) | n/n/n | 0 / 0 | 1 (1) | 0 / 16 | 0.199 / 0.280 |
| L187d1 | 180 | round | standstill | complete | no | no (-) | 1.56 | no | (control: other) | n/n/n | 1 / 1 | 5 (4) | 0 / 40 | 0.318 / 0.290 |
| L188d1 | 0 | square | standstill | complete | no | no (-) | 3.04 | yes | other | n/n/n | 0 / 0 | 10 (8) | 0 / 77 | 0.064 / 0.296 |
| L189d1 | 90 | square | standstill | complete | no | yes (9.28) | 2.12 | yes | other | n/n/n | 1 / 1 | 7 (6) | 5 / 54 | 0.236 / 0.293 |
| L190d1 | -90 | square | standstill | complete | no | yes (8.00) | 1.08 | no | (control: other) | n/n/n | 1 / 1 | 3 (3) | 0 / 28 | 0.225 / 0.289 |
| L191d1 | 180 | square | standstill | complete | no | no (14.20) | 2.12 | yes | blocked_step_up | n/y/n | 3 / 3 | 6 (6) | 0 / 54 | 0.241 / 0.294 |
| L192d0 | 0 | round | standstill | complete | no | yes (3.92) | 1.12 | no | (control: other) | n/n/n | 1 / 1 | 3 (3) | 0 / 29 | 0.207 / 0.289 |
| L192d1 | 0 | round | walking | complete | no | yes (4.40) | 2.16 | yes | slow_progress | n/n/y | 0 / 0 | 7 (5) | 0 / 55 | 0.065 / 0.294 |
| L193d0 | 90 | round | standstill | complete | no | yes (6.52) | 1.00 | no | (control: other) | n/n/n | 0 / 0 | 3 (3) | 0 / 26 | 0.267 / 0.286 |
| L193d1 | 90 | round | walking | complete | no | yes (8.04) | 0.84 | no | (control: other) | n/n/n | 1 / 1 | 4 (3) | 0 / 22 | 0.319 / 0.282 |
| L194d0 | -90 | round | standstill | complete | no | yes (9.12) | 0.64 | no | (control: other) | n/n/n | 0 / 0 | 1 (1) | 0 / 17 | 0.119 / 0.280 |
| L195d0 | 180 | round | standstill | complete | no | no (-) | never | yes | other | n/n/n | 0 / 0 | 11 (6) | 40 / 106 | 0.004 / 0.003 |
| L196d0 | 0 | square | standstill | complete | no | yes (3.68) | 1.12 | no | (control: other) | n/n/n | 1 / 1 | 4 (4) | 2 / 29 | 0.200 / 0.289 |
| L197d0 | 90 | square | standstill | complete | no | yes (9.12) | 2.32 | yes | other | n/n/n | 0 / 0 | 7 (5) | 0 / 59 | 0.095 / 0.292 |
| L198d0 | -90 | square | standstill | complete | no | yes (8.00) | 0.72 | no | (control: other) | n/n/n | 1 / 1 | 3 (2) | 0 / 19 | 0.372 / 0.283 |
| L199d0 | 180 | square | standstill | complete | no | no (17.08) | 0.00 | no | (control: other) | n/n/n | 0 / 0 | 0 (0) | 0 / 1 | 0.000 / 0.000 |
| L200d0 | 0 | round | walking | complete | no | no (-) | 3.04 | yes | other | n/n/n | 0 / 0 | 10 (6) | 0 / 77 | 0.044 / 0.296 |
| L200d1 | 0 | round | standstill | complete | no | yes (4.92) | 0.72 | no | (control: other) | n/n/n | 0 / 0 | 3 (2) | 0 / 19 | 0.324 / 0.283 |
| L201d0 | 90 | round | walking | walk_timeout | no | no (-) | - | no | - | - | - / - | - (-) | - / - | - |
| L201d1 | 90 | round | standstill | complete | no | yes (6.88) | 1.28 | no | (control: other) | n/n/n | 0 / 0 | 3 (2) | 0 / 33 | 0.230 / 0.287 |
| L202d0 | -90 | round | walking | complete | no | no (11.04) | 1.32 | no | (control: other) | n/n/n | 0 / 0 | 4 (2) | 0 / 34 | 0.171 / 0.289 |
| L202d1 | -90 | round | standstill | complete | no | no (-) | 0.68 | no | (control: other) | n/n/n | 0 / 0 | 1 (0) | 0 / 18 | 0.329 / 0.282 |
| L203d1 | 180 | round | standstill | complete | no | no (16.00) | 1.40 | no | (control: other) | n/n/n | 0 / 0 | 4 (4) | 0 / 36 | 0.182 / 0.291 |
| L204d0 | 0 | square | walking | walk_timeout | no | no (-) | - | no | - | - | - / - | - (-) | - / - | - |
| L204d1 | 0 | square | standstill | complete | no | no (-) | 4.20 | yes | other | n/n/n | 1 / 1 | 13 (9) | 1 / 106 | 0.028 / 0.296 |
| L205d1 | 90 | square | standstill | complete | no | yes (6.88) | 1.00 | no | (control: other) | n/n/n | 1 / 1 | 3 (3) | 0 / 26 | 0.331 / 0.287 |
| L206d1 | -90 | square | standstill | complete | no | yes (8.64) | 0.84 | no | (control: other) | n/n/n | 1 / 1 | 2 (2) | 0 / 22 | 0.345 / 0.286 |
| L207d1 | 180 | square | standstill | complete | no | no (-) | 4.00 | yes | other | n/n/n | 0 / 0 | 8 (7) | 0 / 101 | 0.098 / 0.295 |
| L208d0 | 0 | round | standstill | complete | no | no (-) | 0.72 | no | (control: other) | n/n/n | 0 / 0 | 2 (2) | 0 / 19 | 0.349 / 0.284 |
| L208d1 | 0 | round | walking | walk_timeout | no | no (-) | - | no | - | - | - / - | - (-) | - / - | - |
| L209d0 | 90 | round | standstill | fall | yes | no (-) | - | no | - | - | - / - | - (-) | - / - | - |
| L209d1 | 90 | round | walking | complete | no | yes (7.64) | 0.60 | no | (control: other) | n/n/n | 2 / 2 | 3 (3) | 0 / 16 | 0.355 / 0.280 |
| L210d0 | -90 | round | standstill | complete | no | no (-) | 2.80 | yes | slow_progress | n/n/y | 0 / 0 | 8 (5) | 1 / 71 | 0.083 / 0.289 |
| L211d0 | 180 | round | standstill | fall | yes | no (-) | 1.36 | no | (control: other) | n/n/n | 0 / 0 | 5 (5) | 4 / 35 | 0.291 / 0.290 |
| L212d0 | 0 | square | standstill | complete | no | yes (4.04) | 1.92 | no | (control: slow_progress) | n/n/y | 1 / 1 | 5 (4) | 0 / 49 | 0.127 / 0.293 |
| L212d1 | 0 | square | walking | complete | no | yes (3.12) | 0.84 | no | (control: other) | n/n/n | 2 / 2 | 4 (4) | 0 / 22 | 0.156 / 0.286 |
| L213d0 | 90 | square | standstill | complete | no | yes (8.40) | 1.60 | no | (control: other) | n/n/n | 0 / 0 | 5 (4) | 2 / 41 | 0.329 / 0.290 |
| L214d0 | -90 | square | standstill | complete | no | yes (7.68) | 1.40 | no | (control: other) | n/n/n | 0 / 0 | 4 (4) | 1 / 36 | 0.183 / 0.291 |
| L215d0 | 180 | square | standstill | complete | no | no (15.48) | 0.68 | no | (control: other) | n/n/n | 1 / 1 | 3 (3) | 0 / 18 | 0.285 / 0.278 |
| L216d1 | 0 | round | standstill | complete | no | yes (3.92) | 0.72 | no | (control: other) | n/n/n | 0 / 0 | 1 (1) | 0 / 19 | 0.336 / 0.284 |
| L217d0 | 90 | round | walking | walk_timeout | no | no (-) | - | no | - | - | - / - | - (-) | - / - | - |
| L217d1 | 90 | round | standstill | complete | no | no (12.72) | 2.12 | yes | other | n/n/n | 0 / 0 | 5 (4) | 0 / 54 | 0.225 / 0.292 |
| L218d1 | -90 | round | standstill | complete | no | yes (8.64) | 1.00 | no | (control: other) | n/n/n | 0 / 0 | 4 (4) | 0 / 26 | 0.344 / 0.284 |
| L219d1 | 180 | round | standstill | complete | no | no (11.72) | 0.64 | no | (control: other) | n/n/n | 0 / 0 | 3 (3) | 0 / 17 | 0.188 / 0.282 |
| L220d0 | 0 | square | walking | complete | no | no (-) | 2.92 | yes | other | n/n/n | 1 / 1 | 10 (7) | 0 / 74 | 0.040 / 0.292 |
| L220d1 | 0 | square | standstill | complete | no | yes (4.64) | 1.04 | no | (control: other) | n/n/n | 0 / 0 | 4 (2) | 0 / 27 | 0.095 / 0.288 |
| L221d1 | 90 | square | standstill | complete | no | no (11.84) | 1.96 | no | (control: other) | n/n/n | 0 / 0 | 6 (6) | 0 / 50 | 0.308 / 0.293 |
| L222d1 | -90 | square | standstill | complete | no | yes (7.08) | 0.72 | no | (control: other) | n/n/n | 1 / 1 | 2 (1) | 0 / 19 | 0.383 / 0.283 |
| L223d1 | 180 | square | standstill | complete | no | no (16.32) | 3.04 | yes | other | n/n/n | 1 / 1 | 9 (8) | 0 / 77 | 0.056 / 0.296 |
| L224d0 | 0 | round | standstill | complete | no | yes (3.48) | 1.44 | no | (control: other) | n/n/n | 1 / 1 | 4 (4) | 0 / 37 | 0.180 / 0.290 |
| L225d0 | 90 | round | standstill | complete | no | yes (8.56) | 1.20 | no | (control: other) | n/n/n | 0 / 0 | 4 (2) | 0 / 31 | 0.223 / 0.288 |
| L226d0 | -90 | round | standstill | complete | no | no (11.28) | 0.64 | no | (control: other) | n/n/n | 0 / 0 | 2 (1) | 0 / 17 | 0.268 / 0.280 |
| L227d0 | 180 | round | standstill | complete | no | no (13.16) | 0.76 | no | (control: other) | n/n/n | 0 / 0 | 3 (2) | 0 / 20 | 0.157 / 0.284 |
| L228d0 | 0 | square | standstill | complete | no | yes (4.24) | 1.28 | no | (control: other) | n/n/n | 0 / 0 | 4 (2) | 0 / 33 | 0.094 / 0.290 |
| L228d1 | 0 | square | walking | walk_timeout | no | no (-) | - | no | - | - | - / - | - (-) | - / - | - |
| L229d0 | 90 | square | standstill | complete | no | yes (8.76) | 1.08 | no | (control: other) | n/n/n | 0 / 0 | 2 (2) | 0 / 28 | 0.263 / 0.288 |
| L230d0 | -90 | square | standstill | complete | no | no (-) | 3.84 | yes | other | n/n/n | 1 / 1 | 8 (6) | 13 / 97 | 0.066 / 0.291 |
| L231d0 | 180 | square | standstill | complete | no | no (13.40) | 0.80 | no | (control: other) | n/n/n | 0 / 0 | 2 (2) | 0 / 21 | 0.223 / 0.283 |
| L232d0 | 0 | round | walking | complete | no | no (-) | 1.84 | no | (control: slow_progress) | n/n/y | 0 / 0 | 5 (5) | 10 / 47 | 0.073 / 0.293 |
| L232d1 | 0 | round | standstill | complete | no | no (-) | 2.28 | yes | other | n/n/n | 0 / 0 | 7 (5) | 7 / 58 | 0.030 / 0.290 |
| L233d1 | 90 | round | standstill | complete | no | no (-) | 1.08 | no | (control: other) | n/n/n | 0 / 0 | 3 (3) | 0 / 28 | 0.309 / 0.286 |
| L234d0 | -90 | round | walking | complete | no | yes (7.60) | 1.12 | no | (control: other) | n/n/n | 1 / 2 | 3 (3) | 0 / 29 | 0.249 / 0.288 |

## Frozen thresholds

```
FROZEN THRESHOLDS (re-frozen 2026-10-03 per the coordinator re-freeze; corrected 2026-10-04 10:45; before any episode on 170-249)
  STALL_DWELL_S = 2.0                 design D: edge dwell >= 2.0 s is a stall (1e-6 s tolerance)
  ONPLATE_ALONG_M = -0.2              design D / edge_dwell.py: past the edge = base along > -0.20 m
  TOP_NORMAL_MIN_Z = 0.7071           a plate contact is TOP (support) iff its normal, from the plate into the foot, has
                                      world z >= 0.7071 (within 45 deg of vertical); otherwise it is the EDGE FACE
  SUPPORT_MIN_N = 8.0                 a foot is loaded iff floor + plate-top + other-plate-top normal force >= 8.0 N (5% of
                                      the robot's 160.2 N weight: 16.33 kg in the compiled model)
  SWING_MIN_STEPS = 2                 a swing = >= 2 consecutive unloaded policy steps (>= 0.08 s); its touchdown = the next
                                      loaded step
  LEADING_TOL_M = 0.01                a swing is the LEADING foot's iff at touchdown its toe along >= the other foot's toe
                                      along - 0.01 m (at or ahead of it)
  EDGE_FORCE_MIN_N = 1.0              obstruction (a): target-plate edge-face normal force >= 1.0 N at any step from lift-off
                                      to touchdown (1 N = the bench's own plate-press force threshold)
  EDGE_STOP_BEFORE_M = 0.02           obstruction (b): touchdown toe_to_edge >= -0.02 m ...
  EDGE_STOP_AFTER_M = 0.01            ... and <= +0.01 m, on the floor side: floor force >= 8.0 N and target-plate-top force
                                      < 8.0 N at touchdown
  BLOCKED_MIN_SWINGS = 3              blocked step-up iff >= 3 blocked swings (leading AND obstructed) touch down inside W
  GAIT_PERIOD_STEPS = 20              one clock-s2 gait period (0.8 s) in policy steps (0.04 s)
  SLOW_MIN_PERIODS = 2                slow progress needs >= 2 full gait periods in W
  ADVANCE_MIN_PER_PERIOD_M = 0.004    a period advances iff base along gains >= 0.004 m over it (differences one gait
                                      period apart cancel the gait's fore-aft sway)
  ADVANCING_PERIOD_FRACTION = 0.75    keeps advancing iff >= 75% of W's full periods advance
  SLOW_MAX_FRACTION = 0.5             too slowly iff mean along speed over W < 0.5 x mean commanded vx over W
  WALL_DOMINANT_FRACTION = 0.5        wall-blocked iff >= 50% of W's policy steps have a wall/door/goal-post contact (the
                                      bench sample's own "contacts" field)
  PRECEDENCE = wall_blocked > blocked_step_up > slow_progress > other   (coordinator re-freeze; conservative for F1)
  RULES (coordinator re-freeze 2026-10-03 11:40; the pure functions upward_normal, contact_key, is_loaded, is_leading,
  is_blocked_swing, stall_criteria, classify, decide and decision_lines encode them; the tests check each):
  - Plate contacts are classified by the contact NORMAL only: MuJoCo's frame[0:3] points from geom1 to geom2, so it is
    negated when the plate is geom2; n_up_z = the world z of the normal from the plate into the foot; TOP iff n_up_z
    >= 0.7071, else EDGE FACE; all TOP force (target and other plates) is support; contact heights are never used.
  - A BLOCKED swing is a swing of the LEADING foot (toe along = its sole box's most advanced corner along the door
    direction; at touchdown >= the other foot's toe along - 0.01 m) with obstruction evidence (a) or (b) above AND a
    FLOOR-SIDE touchdown: floor force >= 8.0 N and target-plate-top force < 8.0 N at touchdown, i.e. the foot did not
    get over the edge (coordinator correction 2026-10-04 10:45: a swing that brushes the edge face and lands on the
    plate top got over; the smoke audit found 4 of 11 such swings); no swing-height clause (clock-s2's flat-ground
    swings are themselves low; heights are reported only).
  - Each stall gets one class on its window W by the precedence wall_blocked > blocked_step_up > slow_progress > other;
    every criterion and the overlaps are reported.
  - Falls: a crossing that fell (bench v2: tilt >= 0.78 in any phase) is excluded from the stall count and reported
    separately; the blocked-step-up fraction is printed both ways (non-fall; including fall stalls); the decision uses
    the non-fall stalls.
  - SPECIFICITY: among the crossings that CLEAR (bench v2's clear), the share whose own window W (their own edge dwell)
    meets the blocked-step-up criterion (>= 3 blocked swings, regardless of precedence); printed on the decision line.
  - F1 is TRAINED iff non-fall stalls > 0 AND 3 x blocked >= non-fall stalls AND clears > 0 AND meeting clears x non-fall
    stalls < blocked x clears (exact integers: the non-fall fraction >= 1/3 and the clear share strictly lower than
    it).  0 non-fall stalls -> F1 NOT trained; 0 clears -> the specificity check cannot hold -> F1 NOT trained.  The
    decision line always prints the non-fall fraction and the clear share, never "undefined" or a True/False flag.
  - A partial or aborted run (a declared crossing missing or failed, an imported source changed during the run, the
    module failing, the job terminated) prints INCOMPLETE and decides nothing; so does a log without a DECISION line.
  - Smoke check (machinery only): a foot standing inside a ROUND target plate (all 8 sole corners inside the footprint
    shrunk by 0.01 m, >= 8.0 N of target-plate force) reads as loaded in >= 95% of >= 10 such foot-steps.
END FROZEN THRESHOLDS
```

## Disclosures

- layouts: 250-289 as first frozen does not exist beyond 255; corrected to 170-249 before any diagnosis episode (coordinator decision 2026-10-03, recorded in the ledger as a correction; SPLITS['train'] = (0, 256), so 250-289 would have yielded only 6 standstill crossings, L250-L255 d1)
- layouts: no Mission 7 crossing record exists for any layout in 170-249, so no run crossing has a public outcome; bench v2's construction yields 160 grid crossings there, its drop rule drops 61 walking ones, and of the 99 kept the first 80 in grid order (layout ascending, door 0 before door 1) are run: L170 d1 through L234 d0; 19 are capped out
- smoke: the instructed smoke layouts 290-299 do not exist; the smoke runs exploration train layouts 32, 33 and 38 (outside 170-249), which run mode refuses
- measurement and classifier re-frozen by the coordinator (2026-10-03 11:40) after an independent review of the smokes and synthetic checks only (no stall outcome): plate contacts classified by the contact normal (box-on-cylinder contacts sink, so contact heights misclassified round-plate support), and a blocked swing must be the leading foot's and show obstruction evidence (the earlier both-feet, low-swing test flagged false positives)
- classifier corrected by the coordinator (2026-10-04 10:45) after the implementer's audit of its own smoke records (layouts 32/33/38 only; no stall outcome): 4 of the 11 swings the re-frozen rule called blocked had touched down on the target plate top after brushing the edge face, so both obstruction evidences now also require a floor-side touchdown (the design's 'fails to get over the 3 cm edge')
- decision: precedence wall-blocked > blocked step-up > slow progress > other; falls excluded from the stall count and reported separately; specificity check on the clears; 0 non-fall stalls -> F1 NOT trained; INCOMPLETE decides nothing
- physics: the bench ran on haswell&el8 (cn-c21); this launcher runs on el9 nodes as instructed, so trajectories are not bit-comparable across CPU types (the layouts differ anyway)
- labels: the stage gait is LEARNED clock-s2, the stage SCRIPTED, layout and plate pose ORACLE

Sources identical to the clocks2 bench v2 (21517668, 9de55d39ad3c962bc8b3f563bd962a149e82bd6f): True (differing: []; drift allowed: False).
Imported sources changed during the run: [] (relevant drift changed: False; first imported during the run: []).

