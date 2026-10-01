# Ledger and roadmap corrections, 2026-10-01 (for the coordinator)

Nothing in this file has been applied. `SLURM_JOBS.md`, `docs/ROADMAP.md` and every other repo file are untouched,
apart from the new `docs/SOLUTIONS_2026-10-01.md`.
- Evidence paths are relative to `/nfs/hpc/share/sanchej7/Humanoid_Lite/`. They sit outside the repo and are not in
  git. If the bullets should cite repo paths instead, copy `solutions-20260930/verify-*` into
  `results/verification-20261001/` first; that is the coordinator's decision.
- Section A holds only corrections that the independent verification confirmed or partly confirmed.
- Section B holds two diagnostics without independent verification, labelled as such; append them only if wanted.
- Section C gives exact old → new text for `docs/ROADMAP.md`. Every "Success looks like" cell is kept or made
  stricter, never weaker.
- Section D lists other lines that carry the overturned claims (not edited).

## A. Append to SLURM_JOBS.md (after the last line, the `21487485` entry)

<!-- BEGIN APPEND A -->

**Corrections recorded 2026-10-01.** These come from an independent read-only verification of the 2026-09-30 solutions investigation (reports A–D, `solutions-20260930/REPORTS_completed.md`).
- No Slurm job ran. The only new simulation was coop exploration seed 100; turning re-runs on reset seeds 0–2, the seeds of the noise-free turn_test v2 screen, on checkpoints already scored there, with no verdict changed; and NavGym maze construction without a policy. No other scored seed set was run.
- Evidence is in `/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/verify-{coop,turning,mission7,navgym}/`, outside the repo.
- The entries above stay as written; each bullet names the statements it supersedes.

- **Posture/torque diagnostic `21402699` — corrected.** This supersedes the 28–30 Nm reading at :2799 and :2907, docs/STATUS.md:15 and docs/REPO_TASKS.md:94–97.
  - The "28–30 Nm" figure is the step-0 snapshot taken after `env.reset` and before any `env.step`, when every foot carries 0 N. `scripts/bench/task_v2_spawn_diag.py` installs the PD-hold override only inside `env.step`.
  - The figure equals 20 Nm/rad × the reset knee angle (1.39–1.52 rad). The standing pose's own step-0 snapshot reads 7.07–9.43 Nm and is likewise clipped at 6 Nm on 8/8 robot-envs.
  - With the feet loaded (pinch steps 2–5: 136–208 N per robot, tilt ≤ 0.105 rad), the largest joint torque is 4.15–6.35 Nm, at the knees. That matches the 4.8 Nm static knee estimate (≈ 80 % of the cap; `solutions-20260930/coop/static_limits.json`): near the cap, not "infeasible by a factor of five".
  - "Applied torque pinned at 6 Nm … from step 0" is wrong: at step 1, 0/8 pinch robot-envs are saturated.
  - "The standing pose starts within limits (0.2–0.7 Nm at step 2)" is a free-fall reading: 0 N at the feet on 7/8 robot-envs, and landing at step 3 gives 7.7–12.3 Nm.
  - Both poses topple under pure PD hold (pinch at step 16, standing at step 22/24), so this record cannot separate a torque-limited posture from a missing balance controller. COOP-17..20 should read "the pinch crouch needs ≈ 4.8 Nm static at the knee (≈ 80 % of the 6 Nm cap); infeasibility not shown", not "exceeds the 6 Nm actuator limit".
  - Withdrawing the figure does not make the crouch feasible: loaded knees exceed 6 Nm on 1/8 robot-envs by step 3.
  - Evidence: `verify-coop/spawn_torque_table.json`, `verify-coop/spawn_ranges.json`.
- **Stand3 replay `21470828` — mechanism recorded.** Report-only; the NEGATIVE verdicts stand.
  - The lift income comes from re-orienting a supported cube, not from lifting it. On steps with the cube centre above 0.61 m, its lowest corner is within 1 cm of the plinth top (0.41 m) or deck top (0.43 m) on 97.4 % (s0) / 79.7 % (s1) of steps. It is ≥ 2 cm clear of every support on only 0.8 % / 2.1 %.
  - A cube within 8° of flat is above 0.61 m on 2 of 44,990 (s0) and 0 of 40,510 (s1) valid steps.
  - `lifting_object` is `stand_mdp.gated_object_is_lifted`, which calls `coop_lift_mdp.object_is_lifted` (`coop_lift_mdp.py:249-264`): 1[centre z > 0.55 + minimal_height] × pinch kernel × upright gate. It has no orientation, support or contact test.
  - The height clause is met on 66.5 % / 74.0 % of valid steps at the 0.06 m level, and on 85.7 % / 81.5 % at 0.04 m. The replay-time level and the kernel and gate factors are not logged. Median tilt is 40° / 43°.
  - The lift curriculum sat at its configured ceiling, `STAND3_LIFT_MAX` = 0.06 m (`stand_mdp.py:501`; its only levels are 0.04 and 0.06), on 77.7 % / 91.6 % of last-200-iteration samples. It did not stall at a roll height.
  - At peak tilt, the tilt axis is within 30° of the robot a–b line in 93 % of s0 episodes but only 43 % of s1 episodes.
  - s0's held cube has its lowest corner over the plinth on 47.7 % of held steps, over the plinth–deck gap on 42.9 % and over the deck on 9.4 %.
  - `hand_force` is 0 on every valid step; s1 has 127 steps with a hand-link origin inside the cube. The hand asset was not inspected.
  - `s0/s1.trace.npz` are on disk but untracked (`.gitignore:27`).
  - Evidence: `verify-coop/stand3_recompute.json`, `verify-coop/stand3_lifted_share.json`, `verify-coop/tb_lift_levels.json`, `solutions-20260930/coop/stand3_tilt_support.json`.
- **Scripted grasp mechanism in MuJoCo (exploration seeds 100–104; a mechanism probe, not a result).**
  - Lift-hold went from 0/3 to 3/3 only with pad torsional friction (condim 4, 0.04 m) and elliptic cone + impratio 10 together. Separately they score tors02 0/3, tors04 0/3, elliptic alone 1/3 and noslip 0/2.
  - The base configuration fails only the floor-contact clause, after holds of 5.84–10.40 s.
  - In all three passes the cube is still held rolled 0.63–0.93 rad at 19 s; `LIFT_HOLD_RULE` has no orientation clause.
  - Lift-place 5/5 also needed reverse-keyframe lowering, chosen after three lowering targets were tried on the same seeds. With the published `PlaceParams.lower_to` (`scripted_carry.py:1015`), base scores 1/5, with the cube placed on its side (`LIFT_PLACE_RULE` has no orientation clause), and the contact variant 0/5.
  - The elliptic cone is a global MuJoCo option that also changes foot-floor contact. The pads meet the cube ≥ 26° off-face, so the 0.04 m torsional value assumes a flush pad that does not exist.
  - Seed-100 re-runs reproduced the saved rows exactly.
  - Evidence: `solutions-20260930/coop/mechanism_summary.json`, `verify-coop/grip_mechanism_base_tors04_ell_100.json`, `verify-coop/place_mechanism_*_100.json`.
- **Actuator limits — recorded.**
  - 6 Nm (legs) and 4 Nm (arms) are upstream Isaac training caps (`berkeley_humanoid_lite.py`) copied into `deploy.yaml` `effort_limits`.
  - Upstream's hardware runtime enforces them: `external/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_lowlevel/csrc/real_humanoid.cpp:53` loads them, and `:515` writes them to each motor as its firmware torque limit. A fix that needs more torque is sim-only until those limits are raised; gearbox and thermal safety at higher torque is not established here.
  - Firmware config: 15:1 gearing, 20 A and Kt 0.0919 / 0.1176 Nm/A, giving 27.6 / 35.3 Nm ideal current-limited output. Upstream MJCF `forcerange` is ±20 Nm.
  - "Paper tested 20 Nm", a cycloidal gearbox and the 6512/5010 motor mapping are unverified: no file in the repo or `external/` states them.
- **NavGym v4 physics transfer `21484212` — reading corrected.** The NEGATIVE verdict stands. This supersedes the clause at :3096 "the gym-to-physics gap (gait response lag, the speed brake, 10 Hz lidar) costs 3–4 goals in 12, much as v2's exploratory transfer (10/12 fresh) did".
  - A diagnostic replay of both actors in the 2-D gym on the same 12 layouts shows the misses are mostly the policies' own. The layout hashes are identical to the physics JSONs (`solutions-20260930/navgym/gym_replay.json`).
  - At nominal dynamics, heading 0 and 180 s, armV4-s6 fails exactly the three mazes it fails in physics (50001, 50002, 50009): its gym-to-physics gap is 0 goals.
  - armV4-s5 fails only 50009 in that run, against 50000, 50002, 50009 and 50010 in physics. The gap therefore costs s5 at most 3 goals, and about 2 against its five randomized-dynamics draws (50010 3/5, 50002 4/5, 50003 4/5, 50005 4/5, 50009 0/5).
  - Only the nominal comparison is time-matched (180 s, n = 1 per maze). The randomized draws ran route-scaled limits of 121–180 s (121 s on 50009) against physics' 180 s, so the "about 2" is not time-matched; a shorter limit can only add time-outs.
  - Two of those three (50000, 50002) already needed 111 s and 153 s of 180 s in the gym.
  - Maze 50009 (shortest route, 12.2 m) defeats both actors in gym and physics, while A* solves it in 41.0 s.
    - s6 shuttles three round trips between the start and the dead-end pocket (5,0)–(5,1), beside the route's branch at (4,0).
    - s5 in physics instead turns up the x = 2 column and dithers; its longest stall is 87.2 s.
  - The ego crop does show the route's first leg (4,1) on every pass. What the policy lacks is visitation history: the route's decisive turn (4,2)→(5,2) is 2.8 m from the branch, beyond the 2.4 m crop edge.
  - Yaw is bang-bang in both worlds: |wz| is at the 1 rad/s limit on 95–98 % of gym steps and 90–99 % of sampled physics steps, and final policy std is 9.4 and 13.7. It is a policy property, not a transfer artefact.
  - The v2 comparison was not re-checked: unverified.
  - Evidence: `verify-navgym/layout_check.json`, `verify-navgym/traces_check.json`, `verify-navgym/wz_sat_physics.json`, `verify-navgym/crop_recon.json`.
- **Isaac replay probe `21442353` — scope corrected.** This supersedes "< 8/32 = not learned (the failure is in the policy, not a sim2sim gap)" at :3027, docs/STATUS.md:13 and docs/REPO_TASKS.md:72.
  - The probe ran with observation corruption disabled (`scripts/bench/turn_diagnose.py:359`) and the deterministic exported TorchScript actor, so there was no action noise either.
  - It ran on the stock task `Velocity-Berkeley-Humanoid-Lite-v0`, not the arms' training tasks `Velocity-BHL-Arms-*-v0`. Whether domain randomization matched is unverified.
  - Its ≤ 2/32 shows only that the noise-free policies also stand in Isaac. "Not a sim2sim gap" holds for the noise-free loop, but "< 8/32 = not learned" does not follow.
  - In MuJoCo four non-turning checkpoints (TurnBoth-s1, TurnCmd-s0, TurnCmd-s1 and TurnHip-s0; the first three were in this probe) turn ≥ 150° in 16/16 runs with their own training action noise. The std comes from each `model_5999.pt` and is added before the 0.25 scale and fed back as the last action. Those 16 runs share two RNG streams; a third, independent stream gave 238°.
  - Without noise they reach 0/16 (10–18°), and at zero command with noise they turn −7.5 to +4.2°.
  - 10 of the 11 non-turning arm seeds pass a v2-style replay with the training observation noise alone. This is a diagnostic, not `turn_test.py`, and 10/11 is a floor.
  - Isaac with noise on is unverified.
  - Evidence: `solutions-20260930/turning/exp1_noise.log`, `solutions-20260930/turning/v2_obsnoise_x1.0.json`, `verify-turning/rerun_results.jsonl`, `verify-turning/obsnoise_cfg_check.txt`, `verify-turning/origin_check.jsonl`.
- **Turning — records added.**
  - TurnBoth-s0 is the 1/12 by v2x qualification (`21442352`: 10/10 turns on reset seeds 10–14, walk 3/3, push 7/60).
  - Under the noise-free `turn_test` v2 on reset seeds 0–2, 0/12 seeds pass; TurnBoth-s0 scores 5/6 (−11.4° on reset seed 1 at −0.6 rad/s).
  - The investigation's push runs with training noise (TurnBoth-s0 13/60 and TurnBoth-s1 18/60, against 7/60 and 9/60 noise-free) used observation noise only, on push seeds 0–9. Those are the qualify gate's scored push seeds, so TurnBoth-s1 has no unspent push seeds left for a future qualification.
  - "Every checkpoint that marches at zero command passes v2" is a post-hoc association inside the TurnBoth-s0 lineage, not a criterion:
    - All 7 marchers turn in ≥ 5/6 v2 runs, but only 4/7 pass v2 (CONT-s0/-s1 fail walk drift).
    - There are counterexamples: TurnTrack-s2 marches on reset seed 2 (9 lift-offs) but turns only 4.7 / −17.1°.
  - Evidence: `verify-turning/f6_march_vs_v2.txt`, `solutions-20260930/turning/batch6.log`.
- **Mission 7 exact replay — reading corrected.** This supersedes reading `21397732`'s PASS, recorded at :2462 and in docs/MISSION7_TASKS.md:47, as evidence of a crossing. :2691 already noted that the replay's fixed 1.20 s crossing "barely moved (0.13 m) and never reached the edge"; the bullets below extend that to all five G-ref crossings and add the door timing.
  - Four fixed-1.2 s-crossing variants pass the exact ten-fall replay 10/10 (`results/mission7-campaign-20260923/replay-gate-matrix-summary.json`; G-ref's own `results/mission7-approach-followup-20260922/plate-stage-cn-c22-guarded/result.json`):
    - `21397732` (G-ref)
    - `21400863` (wait-open 2.0)
    - `21400897` (stage-lateral 0.35)
    - `21400953` (stage-lateral 0.35 + wait-open 2.0)
  - Lateral 0.25 (`21400805`) scored 8/10.
  - In `21397732` the five staged 1.24 s crossings (layouts 0, 1, 4, 7, 14) moved the base +0.015 to +0.215 m along the door direction and ended 0.187–0.328 m before the plate centre. The furthest reach was −0.142 m, in layout 1.
  - Door 0's recorded activation preceded every crossing: layout 0 at 24.20 vs 24.48 s, 1 at 19.24 vs 20.20, 4 at 15.88 vs 16.20, 7 at 15.96 vs 16.24, and 14 at 18.24 vs 19.68 s. The gate never required the stage to press a plate, and `21400863` is identical to `21397732` crossing for crossing.
  - After hand-back, the replayed recorded commands did carry the robot across the correct round plate in layout 7: along −0.219 m at 17.48 s to +0.376 m at 21.12 s, both feet on it, max tilt 0.16.
  - In layouts 0, 1 and 4 the base stood inside the disc for 6.84, 3.44 and 2.28 s without falling.
  - Layouts 5 and 15 spent 27.4 / 27.2 s in approach toward the wrong square plate. They moved 0.06 m in their last 20 s under a ≈ 0.3 m/s command.
  - A replay 10/10 therefore cannot certify a crossing mechanism.
  - The lateral-0.35 gates (`21400897`, `21400953`) started 7 crossings each, against 5 for `21400863` (`replay-gate-matrix-summary.json`, `crossings_started`) and 5 for `21397732` (its five staged crossings, `verify-mission7/verify_crossings.out`), so they did not pass the same way. Whether they cleared a plate is unverified (only `result.json` was kept).
  - Evidence: `verify-mission7/verify_crossings.out`, `verify-mission7/poststage_check.out`, `verify-mission7/gref_L7_dump.txt`, `verify-mission7/gref_L0_dump.txt`.
- **Mission 7 cross-clear gates `21405537`–`21405545` — additions** (to :2883).
  - Of their 35 crossings, 15 were on the wrong square plate (layouts 5, 14 and 15 in every variant).
  - All 7 in-crossing falls were sideways entries onto the round plate in layouts 0, 4 and 7.
  - The 14 forward entries with 0 in-crossing falls are 5 layout-1 round-plate crossings and 9 square-plate crossings. They are not 14 independent trials behind a 0.19 upper bound.
  - 7 of the 14 forward entries cleared. The other 7 timed out at 4.04 s: six ended 0.30–0.38 m before the plate centre, one at +0.105 m.
  - `crossings_completed` (`slurm/repo20260923/m7_replay_gate_followup.sbatch:90`) counts every episode with 'recorded' after 'cross'. Its 7/7 therefore includes those time-outs and the two crossings in which the robot fell. Real clears were 4/7 (`21405537`) and 1/7 (`21405541`).
  - The v2-align layout-7 fall (`21405537`) entered 1.02 rad off the door direction, but was only 0.37–0.40 rad off at the stall. Its right foot was on the plate edge (≤ 90 N) under a mostly forward command [0.28, 0.11, +0.35 rad/s], and it fell 2.40 s after the kick.
  - The v4-settle080 layout-0 fall (`21405541`) was a pure sideways command [0.00, 0.30, 0] at 1.46–1.82 rad off. Its left foot was on the edge (≤ 344 N), the uncommanded yaw rate reached −0.99 rad/s, and it fell 1.92 s after the kick (tilt 0.12 → 0.78 in 0.64 s).
  - Only one of the five post-hand-back falls fits a "recorded commands applied 82–90° off" explanation: v3-settle190-align layout 5, at 86°. The others are 28–30°, 7–8°, 64–69° and 51–57°.
  - There is no align stage. `--align-yaw` adds a moving yaw term of at most 0.35 rad/s during approach and crossing only, so the align variants still entered layouts 0, 4 and 7 at 1.02–1.12 rad off.
  - Evidence: `verify-mission7/verify_crossings.json`, `verify-mission7/v2align_L7_dump.txt`, `verify-mission7/v4s080_L0_dump.txt`, `verify-mission7/poststage_check.out`.
- **Mission 7 route gate — code bug recorded** (`scripts/mission7_plate_safe.py:158-159`).
  - `doors_16` and `transport_16` are coded as `summaries[...]['episodes'] >= 16`.
  - So `21398514`'s `results/mission7-approach-followup-20260922/route-eval-cn-c22-v2/result.json` reports both true beside doors 1/16 and transport 0/16 successes.
  - docs/MISSION7_TASKS.md:32 and :48 read the gate as 16/16 successes and record FAIL.
  - The route-handoff probe that `m7_replay_gate_followup.sbatch` would release encodes no pass criterion.
  - Both must count successes in code before any route gate is released.
- **Mission 7 episode budget — inconsistent, unresolved.**
  - The docs say 101 of 512 remain: docs/STATUS.md:23, docs/REPO_TASKS.md:22 and docs/MISSION7_TASKS.md:37. MISSION7_TASKS.md:75 still says "Episodes used: 0 / 512".
  - This ledger says otherwise:
    - 3 remained at :2733.
    - 1 remained after `21401987` (:2737, :2742).
    - :2842 says "411/512 spent + 4 re-renders = 415, i.e. 3 over the single episode it had left", which contradicts itself.
    - 50 replay-gate episodes were charged on 2026-09-24 (:2883).
  - None of the 101 statements was reduced for the 54 episodes charged on 09-24, and `m7_replay_gate_followup.sbatch:32` also planned against 101.
  - Under this ledger's reading the line is ≈ 53 episodes over. A 32-episode route gate fits only under the 101 reading (101 − 54 = 47).
  - No Mission 7 episode is to be submitted until the user reconciles the line.
- **Small records.**
  - The login-node re-run of v2-align layout 7 reproduced the cn-c22 physics exactly: 1,112 samples, identical base xy and contact lists, first fall at 18.28 s. The saved JSON is not bitwise identical: 834 samples differ by ≤ 3.3e-16 in derived fields and by ≤ 1.4e-14 in contact-event floats (`verify-mission7/compare_rerun_L7.out`).
  - TurnBoth-s0's `deploy.yaml` differs from arms-dr1.0-s0's in two places: `policy_checkpoint_path`, and the export-time `command_velocity` ([0.937, −0.437, 0.466] vs [−0.128, 0.198, 0.407]), which `RlController.update()` ignores. `MissionEnv` builds its config from the shipped file and sets only `controller.policy`, so a policy swap alone is functionally equivalent (`verify-mission7/deploy_yaml.diff`).
  - The investigation report's "p(10/10) per variant ≈ 0.09–0.30" has no derivation in its files and is unverified.

<!-- END APPEND A -->

## B. Optional: diagnostics recorded without independent verification

Append after A only if wanted. Bullet 2 belongs with a commit of the three kit files, and should cite that commit.

<!-- BEGIN APPEND B -->

- **Biped gait sysid and NavGym deployment-pipeline check (2026-10-01; interactive, no Slurm job): investigator report, not independently verified.**
  - Seeds: gym maze seeds 9000–9071 only, all in the training range. Block 9000 was exploratory (13 configurations), 9024 a replicate, and 9048 fresh, with predictions written first.
  - Physics: five episodes on training mazes 9000–9002. Seeds 50000–50011 were only read from saved traces.
  - Identified dynamics: physics yaw responds within one 40 ms step. Fitted in the gym's own form, tau is 0.06–0.08 s with latency 0, against the gym's 0.15–0.45 s and 0–2 steps.
    - Gain, max rate, drift, dead time and start-up lag are inside the gym's range.
    - The gym lacks the turn-in-place and forward dead zones, the stride wobble, the speed brake and the 10 Hz lidar pipeline.
  - In the gym (armV4-s5, 72 mazes):
    - Identified dynamics alone: 47, against 50 with the default randomization (p = 0.51).
    - Full pipeline (dynamics + brake + 10 Hz packets integrated at the next step's pose + heading 0 + 1 s settle): 33 (p = 0.0005). On the non-exploratory blocks alone: 24/48 vs 34/48, p = 0.013. Failures are time-outs, with the brake active on 0.36–0.41 of steps.
    - Full pipeline with each packet integrated at its capture pose: 48 (vs the full pipeline p = 0.0015; vs default p = 0.77).
    - armV4-s6 is insensitive: 44/48 default, 44 with the full pipeline, 42 with the fix.
  - Physics, training maze 9001: s5 is stuck for 180 s without the fix (brake on 81 % of steps) and reaches the goal in 50.1 s with it (brake on 34 %).
  - Confirmed in the code:
    - The biped `--policy` branch of `scripts/bench/maze_explore.py` updates the learned ego map at the loop-top pose (`:798-799`).
    - The humanoid branch integrates its map from the capture pose (`:789-795`, pose recorded at `:837-840`).
  - The pooled gym counts were re-read from `data/gym_reeval_final.json`.
  - Predeclared checks P1–P7: 4 hold; P1 fails on s5 block 9048, and P3 and P4 fail.
  - Evidence: `solutions-20260930/sysid/data/gym_reeval_final.json`, `identified_model_table_final.json`, `predeclared_gymcheck_20261001.json`, `physloop/armV4-s5_capfix/seed9001.json`.
- **IM10A recording kit (2026-10-01; no job): investigator report.** Three new files:
  - `scripts/sensors/imu_allan.py`: Allan deviation and N / B / K identification. It reads ROS 2 bags or CSV and writes `imu_measured.json`, including a `kalibr_imu` block.
  - `tests/test_imu_allan.py`: 19 synthetic-recovery tests, 18 passed and 1 skipped. The skipped one is the bag reader, because `rosbags` is not in the shared venv. Re-run 2026-10-01 in 21 s.
  - `docs/IMU_RECORDING.md`: the recording procedure for the probable Hiwonder IM10A.
  - Conventions: N = σ(τ = 1 s); per-sample std = N·√f; B = σ_min / 0.6643; K = σ(3 s) on the +½ line, which equals `ImuNoise.gyro_bias_walk`.
  - Synthetic recovery (`solutions-20260930/imu-kit/results.md`):
    - N is recovered within 0.6 %.
    - At 200 Hz over 16 seeds, B (minimum method) reads 0.96 / 0.99 / 1.04 × true at 2 h, against 0.80 / 0.91 / 1.01 at 30 min.
    - In the same runs, rate random walk is visible in 13/16 records at 2 h but only 3/16 at 30 min, so a stationary record must be ≥ 2 h.
    - N reads ≈ 2 % low behind the module's default 20 Hz bandwidth.
  - No physical recording exists. Absolute latency needs a reference sensor of known latency.
  - Disk: project 30762 holds 1.529 TB against its 1.5 TB soft quota (2 TB hard), with 3 weeks 5 days of grace on 2026-10-01 (`lfs quota -p 30762 /nfs/hpc/share`). A 2 h bag is ≈ 0.7 GB.

<!-- END APPEND B -->

## C. docs/ROADMAP.md — exact old → new

Each OLD line occurs once in `docs/ROADMAP.md` at the line given; the file has no uncommitted changes as of 2026-10-01.
Items 1 and 3 link to `SOLUTIONS_2026-10-01.md`, which is itself untracked, so apply them with the plan's commit.

1. Line 3 (header):

OLD:
```text
Updated 2026-09-30. Done since the last update: README hero is now the 22-DoF humanoid with sensor panels; NavGym v4 passed its gate. One line per item; evidence and rules live in
```
NEW:
```text
Updated 2026-10-01. Done since the last update: README hero is now the 22-DoF humanoid with sensor panels; NavGym v4 passed its gate; the solutions plan for the five open problems is [`SOLUTIONS_2026-10-01.md`](SOLUTIONS_2026-10-01.md). One line per item; evidence and rules live in
```

2. Line 16 (Next table, SF-05). Apply only together with committing the three IMU kit files:

OLD:
```text
| **SF-05: record the real IM10A** over ROS 2 (10 min still, then slow rotations per axis) | replace the datasheet noise in the sim IMU with measured noise, bias and rate | Allan-variance sigmas committed; sim IMU and panel read "measured on your IM10A" |
```
NEW:
```text
| **SF-05: record the real IM10A** over ROS 2 (≥ 2 h still at 200 Hz, then slow rotations per axis and a tap test; procedure [`IMU_RECORDING.md`](IMU_RECORDING.md), analysis `scripts/sensors/imu_allan.py`) | replace the datasheet noise in the sim IMU with measured noise, bias and rate; a 30 min record reads B about 10 % low (min method) and usually loses K | Allan-variance sigmas (N, B and K, with K measured rather than an upper bound, as `imu_measured.json`) committed; sim IMU and panel read "measured on your IM10A" |
```

3. Line 24 (turning):

OLD:
```text
| **A turning humanoid recipe**, not one lucky checkpoint | 1/12 seeds qualifies; fine-tunes trade turning for push robustness (v5) | train turn + push together from the start, 3 seeds | ≥ 2/3 seeds pass the turn test and push falls ≤ 0.15 |
```
NEW:
```text
| **A turning humanoid recipe**, not one lucky checkpoint | 1/12 seeds qualifies (TurnBoth-s0, via v2x; noise-free v2 0/12); in MuJoCo four non-turners turn in 16/16 runs with their own training action noise, which points to PPO learning to step only in the noisy loop while the noise-free gates see the standing mean; fine-tunes trade turning for push robustness (v5) | gait clock + contact-schedule reward paid at every command, pushes from the start; clock in the actor (R1) and in the critic only (R2), 3 seeds each, gate unchanged ([plan](SOLUTIONS_2026-10-01.md)) | ≥ 2/3 seeds pass the turn test and push falls ≤ 0.15 |
```

4. Line 25 (learned navigation):

OLD:
```text
| **Learned navigation that passes its own gate**, then runs the physics robot | gym gate **PASSED** (v4, 2/3 seeds); physics transfer **8/12 and 9/12** (need ≥ 10), 0 falls, misses are time-outs | close the gym-to-physics gap: train with the gait's measured response lag and the brake in the gym | ≥ 10/12 never-seen mazes on the physics biped, each passing actor |
```
NEW:
```text
| **Learned navigation that passes its own gate**, then runs the physics robot | gym gate **PASSED** (v4, 2/3 seeds); physics transfer **8/12 and 9/12** (need ≥ 10), 0 falls, misses are time-outs; on the same layouts in the gym s6 fails the same 3 mazes (its misses are its own) and s5 loses 2–3 more in physics; maze 50009's dead-end pocket traps both actors in the gym and s6 in physics, while s5 in physics turns up the x = 2 column and dithers (longest stall 87 s) | integrate each lidar packet at its capture pose in the biped `--policy` path (deploy-side; removes s5's gap in a gym re-creation, not independently verified), screened on unused training-range mazes; then route-level navigation (an A* sub-goal hybrid, labelled learned + scripted, or a visitation channel / recurrent policy), judged on fresh maze seeds | ≥ 10/12 never-seen mazes on the physics biped, each passing actor (a hybrid does not satisfy this row; it is reported separately as learned + scripted) |
```

5. Line 26 (cube placement):

OLD:
```text
| **Cube placement** | Stand3 stands but never places; replay: checkpoints fail differently | pick lowering-vs-reach only when evidence agrees | success ≥ 0.10 on ≥ 1 of 2 seeds |
```
NEW:
```text
| **Cube placement** | Stand3 stands but never places; replay `21470828`: both checkpoints earn the lift reward by re-orienting a cube that rests on its support (`object_is_lifted` tests centre height only); scripted lift-place fails because the cube rolls in the hands | scripted: a flush-pad wrist mechanism probe on exploration seeds; learned: a roll-proof lift reward and a hand-collider check, then a Stand4 with its rule written first | success ≥ 0.10 on ≥ 1 of 2 seeds, with cube tilt and release checked |
```

6. Line 27 (sim-to-real). The kit reference needs the kit commit:

OLD:
```text
| **Sim-to-real** on the Berkeley Humanoid Lite with lidar, stereo and IM10A | sim only; IMU latency budget ≈ 30 ms | SF-05 recordings, then a sensor-only bench check | a hardware claim backed by recorded data |
```
NEW:
```text
| **Sim-to-real** on the Berkeley Humanoid Lite with lidar, stereo and IM10A | sim only; IMU latency budget ≈ 30 ms; the hardware runtime writes the deploy limits (6 Nm legs, 4 Nm arms) into motor firmware, so fixes needing more torque are sim-only | SF-05 recordings with the kit ([`IMU_RECORDING.md`](IMU_RECORDING.md)), then a sensor-only bench check | a hardware claim backed by recorded data |
```

7. Line 28 (Mission 7):

OLD:
```text
| Mission 7 route controller | blocked at 9/10 in replay (gate 10/10) | a new crossing idea | 10/10 in replay, then fresh layouts |
```
NEW:
```text
| Mission 7 route controller | blocked at 9/10 in replay (gate 10/10), the best of the cross-clear variants, which keep crossing until the plate is cleared; four fixed-1.2 s-crossing variants pass 10/10, but in G-ref `21397732` and wait-open `21400863` (identical crossing for crossing) the staged crossing never cleared a plate and door 0 was already open, and the two lateral-0.35 passes are unverified, so the replay cannot certify a crossing; the route gate's code counts episodes, not successes; the episode budget is inconsistent (101 left vs ≈ 53 over) | code the route-gate success criterion and reconcile the budget (0 episodes); then, with a new budget line, a plate bench with TurnBoth-s0 as the stage gait (turn to the door, cross forward) | bench PASS, then 10/10 in replay, then Doors and Transport 16/16 on fresh layouts |
```

## D. Other lines that carry the overturned claims (list only; not edited)

- **docs/STATUS.md:**
  - :13: "the other checkpoints do not turn in Isaac either (≤ 2/32), so not a sim2sim gap". Correct it to "in the noise-free loop".
  - :15: "the pinch crouch needs 28–30 Nm against a 6 Nm effort limit (`21402699`), so it cannot be held at any spawn height". Correct it to "≈ 4.8 Nm static at the knee (≈ 80 % of 6 Nm); infeasibility not shown". The Stand3 replay farming finding can also be added there.
  - :23–24: "Mission 7's remaining 101 CPU episodes". The budget is unresolved.
- **docs/REPO_TASKS.md:**
  - :22: "101 CPU episodes remaining of 512".
  - :72 (LOC-12): "so not a sim2sim gap".
  - :94–97 (COOP-17..20): the status "BLOCKED — spawn posture exceeds the 6 Nm actuator limit", and the text "demands 28–30 Nm at reset … pinned at 6 Nm from step 0".
- **docs/MISSION7_TASKS.md:**
  - :37 ("101 unspent") and :75 ("Episodes used: 0 / 512"): the budget.
  - :47: the replay PASS `21397732`. Add that its staged crossings never cleared a plate and door 0 was already open.
  - :32 and :48 are correct ("≥16/16" successes); it is the code at `scripts/mission7_plate_safe.py:158-159` that must change to match them.
