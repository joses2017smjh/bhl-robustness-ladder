# Humanoid task closure ledger — October 8, 2026

This is the current open/closed ledger for the Humanoid project. It reconciles
the resume-revamp tasks with the older [master backlog](REPO_TASKS.md),
[roadmap](ROADMAP.md), and [status page](STATUS.md). Historical reports and
frozen experiment outputs remain their original records. **DONE** means the
stated deliverable was completed; **DONE NEGATIVE** means an experiment finished
and failed its unchanged acceptance rule. Neither means an unsolved robotics
objective has been achieved.

## Resume-result tasks

| ID | Current state | Evidence and acceptance rule | Remaining work |
|---|---|---|---|
| H1 — matched navigation confirmation | **ACTIVE**, CPU job `21689706` | [Frozen protocol](../results/task-closure-20261008/h1-navigation/protocol.json), [submission](../results/task-closure-20261008/h1-navigation/submission.json), [status](../results/task-closure-20261008/h1-navigation/status.json). Planned 576 episodes: three frozen navigation actors × baseline/candidate × nominal/35% dropout × 48 fresh layouts. Candidate is the existing 0.2 s yaw filter. Each actor must reach ≥40/48 nominal and ≥36/48 dropout goals, with zero falls in both conditions. | Await complete episode contracts, paired results and final report. A submitted or completed scheduler job is not a passing experiment. No success or fall-reduction conclusion yet. |
| H2 — portable regression replay | **PENDING final validation** | [Original replay report](../results/resume-revamp-20261007/replay_report.json) establishes five exact same-host repeats and 40/40 detected injected regressions. Fresh installation, full CPU tests, hosted CI and second-host checks are being validated separately. | Close only after final fresh-environment/full-suite and hosted CI receipts are available, with any cross-host differences reported. Original timing numbers retain their original host and measurement scope. |
| H3 — history/latency policy training | **OPEN** | [Resume task record](resume-results-20261007/task-status.csv) identifies proposed training work; no new training result exists for this task. | Freeze a matched history-versus-feedforward training protocol and held-out latency evaluation before training. Existing sensor-tolerance measurements do not establish this result. |
| H4 — repeatable 22-DoF turning | **OPEN** | R1, R2 and R1H experiments are complete but negative. The unchanged joint rule requires ≥2/3 seeds to pass turn-test v2 and qualify; qualification requires ≥9/10 turns, straight-walk drift ≤15° on ≥2/3 seeds, and push falls ≤9/60. | A new justified recipe and repeated-seed training/evaluation. Qualified checkpoints from different recipes cannot be combined to satisfy one recipe's two-of-three rule. |

H1's development pilot used two layouts and eight episodes: both arms reached
2/4 goals with zero falls and the filter reduced command chatter by approximately
53%. Seeds 95000–95001 are consumed development data. The older 384-episode
navigation confirmation compares actors with A* and has no yaw-filter arm; it
cannot substitute for H1's fresh matched confirmation. Its **NEGATIVE** zero-fall
verdict is retained: six falls in 288 learned-actor episodes, versus zero in 96
A* episodes. [Older confirmation](../campaigns/20261006-confirmatory/results/verdict.json).

## Deliverables closed in this audit

| ID | Closure | Verified evidence and scope |
|---|---|---|
| NAV-01 | **DONE** — noise caveat and sensor-panel media | [Noise-on rescore](../results/repo-gpu-20260923/maze-noise/summary.json): 380/384 versus the published noise-off 379/384. [Panel sidecar](gifs/isaac/maze_both_panels.json) identifies the delivered recording. |
| NAV-02 | **DONE** — corrected stereo media provenance | [Sidecar](gifs/isaac/maze_stereo_fixed.json), [600-frame hash manifest](gifs/isaac/maze_stereo_fixed.inputs.json), and corrected GIF title. Nine complete event series were independently reread. Seed 0: 0.877 versus 0.001; corrected three-seed mean 0.7745489 rounds to **0.775**, versus pre-fix 0.021 and blind 0.738. The GIF was reassembled from original frames; no Isaac episode ran. These are training terrain levels, not success rates. |
| NAV-03 | **DONE** — legacy metric provenance, negative behavior retained | [Verdict](../results/repo-gpu-20260926/nav03-score/verdict.json): all four policies reach 0/32 buttons on first episodes, consistent with the retracted success interpretation. |
| NAV-06 | **DONE** — original v5 gym/physics transfer experiment | [Gym gate](../results/navgym-v5-20261002/verdict_v5.json) and [physics gate](../results/navgym-v5-transfer-20261002/verdict.json): 11/12, 12/12, 12/12 goals, zero falls, each actor above the frozen ≥10/12 rule. Later larger confirmation remains negative; this closes the original experiment, not general robust navigation. |
| LOC-01 | **DONE** — domain-randomization clip provenance and training qualifier | [Sidecar](gifs/dr_pair.json) hashes both original MP4s and records the matching strafe CSV rows. REPORT now states that default seed 1 stopped after 5,495 logged iterations, last zero-based index 5,494, rather than completing all 6,000. No evaluation rows were excluded. |
| LOC-04 | **DONE** — flat-versus-terrain caption correction | [Gallery](GALLERY.md#locomotion) states flat 0/60 at each randomization setting and two humanoid training policies; the terrain comparison is identified separately. |
| LOC-06 | **DONE** — explicit terrain denominators | CSV recount: humanoid 7/60 falls across two trained policies versus biped 34/90 across three, at difficulty 1.0. [Receipt](../results/task-closeout-20261008/media_provenance.json) records source hashes. The clip remains illustrative. |
| LOC-07 | **DONE** — shared-world race denominator | [Ten-seed result](../results/multi-race-20260923/multi_race_10seed.json) is complete: falls 1/10, 10/10, 0/10 and 0/10 for randomized, no-randomization, push-trained and terrain-trained policies. |
| LOC-10 | **DONE** — retraction and preserved original probe | [Immutable probe verdict](../results/ice_placement_probe_21328532.txt) records unreachable ice placement, reachable fraction 0.044. The gallery identifies the MuJoCo clip as flat ground. No ice-handling success is claimed. |
| CLO-SPAWN | **DONE** — cloth spawn-quaternion code defect | Fix published in `0004152`; [validation receipt](../results/task-closure-20261008/cloth-quaternion.json) records 13 targeted CPU checks and the original-literal negative control. The allowlist exception was removed. This closes a code defect; Isaac runtime and task-success claims require separate evidence. |

Media provenance was generated by
[`scripts/bench/media_provenance.py`](../scripts/bench/media_provenance.py)
against the original read-only assets. The compact
[receipt](../results/task-closeout-20261008/media_provenance.json) records the
executing source hash, runtime versions, CSV denominators, and output hashes.
The original GIF remains identifiable by its immutable Git object and hash.

## Completed negative experiments

| Experiment | Final outcome | Evidence |
|---|---|---|
| R1: actor gait clock | **DONE NEGATIVE**, 1/3 joint-qualified seeds; need 2/3 | [R1 verdict](../results/repo-gpu-20260923/turngait-r12-20261001/verdict/R1.json) |
| R2: critic-only clock | **DONE NEGATIVE**, 0/3; need 2/3 | [R2 verdict](../results/repo-gpu-20260923/turngait-r12-20261001/verdict/R2.json) |
| R1H: straight-walk mix and heading-hold reward | **DONE NEGATIVE**, 0/3; all outputs present | [R1H verdict](../results/repo-gpu-20260923/turngait-hold-20261002/verdict/R1H.json). Seed 2 is complete: v2 PASS, qualification turn 8/10 against ≥9/10, walk 3/3, push 3/60. |
| PlateCross / PlateCross2 | **DONE NEGATIVE**, no qualified seed and no bench released | [PlateCross](../results/repo-gpu-20260923/platecross-20261002/selection.json), [PlateCross2](../results/repo-gpu-20260923/platecross2-20261004/selection.json). PlateCross2 push falls are 20/60, 12/60 and 17/60 against ≤9/60. |

H4 and the Mission 7 route objective remain open despite these experiments
being complete. No threshold was relaxed and no new training ran in this audit.

## Partial or still-open repository work

| ID / objective | State and exact next prerequisite |
|---|---|
| INS-01 / INS-02 | Sensor-panel media is complete and 12 wrong-branch episodes are rejected; the originally proposed larger extension and explicit wrong-branch summary/gate predicate are not closed by those recordings. |
| INS-03 | **PARTIAL**: crews 2 and 3 each achieve 10/10 on fresh seeds 5–14, with both nominal controls 0/10; each also achieves 10/10 at 35% dropout. [Nominal crew 3](../results/inspection-airlock-20260923/team3-s5-14.json), [dropout crew 3](../results/inspection-airlock-20260923/team3-dropout035-s5-14.json). The proposed 20-seed extension is not complete. Delay/interruption stress needs script options. |
| NAV-04 | Corrected-checkpoint rerender of the terrain-sensor comparison is still open; the old stereo panels remain labeled as pre-fix recordings. |
| LOC-03 | Flatfill terrain retention sweep remains open; exported models and the matched difficulty sweep are required. |
| LOC-09 | The 2.9% depth-error claim is withdrawn. Rough-terrain departure validation still times out; a corrected probe is required. |
| LOC-05 / LOC-11 | LOC-05's expanded push comparison is measured; remaining historical media wording is not newly certified here. LOC-11 exposure/control/mechanism measurements are complete, but legacy placed-policy export/render reproducibility remains a provenance follow-up. |
| M7-01 / M7-03 | Route objective remains **BLOCKED**: require a genuine-crossing bench PASS, exact replay 10/10, then Doors and Transport each 16/16 on fresh layouts. Sensor-only/four-sensor release remains gated. Reconcile the inconsistent historical episode budget before further Mission 7 episodes. |
| SF-05 / hardware transfer | **OPEN**: physical IM10A stationary/rotation/tap recordings, measured Allan parameters and calibration/extrinsics; no hardware data was collected. |
| Estimated-pose navigation | **OPEN**: replace oracle pose with a measured estimator, then evaluate the unchanged fresh-maze gate. |
| Cooperative carry / placement | **OPEN**: learned and scripted approaches retain their negative outcomes; Stand4 last-200 success is 0.006/0.001 against 0.10. A physically supported grasp and release plus unchanged success checks are needed. |
| CLO-01 / CLO-04–09 and COOP-08 archival follow-up | **OPEN / dependency-bound or archival**: additional shirt seeds, Newton solver diagnostics, linked folding evaluation/media/documentation, and the superseded ladder clip's provenance remain outside this closeout. A quaternion code fix does not establish cloth-task success. Current linked-driver state and older draft corrections were not re-audited. |

## Separate Waiter backup branch

The immutable backup commit
[`c623f83790eb733db6af9897cfb6e97818149123`](https://github.com/joses2017smjh/bhl-robustness-ladder/tree/c623f83790eb733db6af9897cfb6e97818149123)
contains [Waiter WBC](https://github.com/joses2017smjh/bhl-robustness-ladder/blob/c623f83790eb733db6af9897cfb6e97818149123/results/waiter-20261005/wbc/selection.json)
and [WBC1b](https://github.com/joses2017smjh/bhl-robustness-ladder/blob/c623f83790eb733db6af9897cfb6e97818149123/results/waiter-20261005/wbc1b/selection.json)
selection receipts. Both are **NEGATIVE**, with no selected seed. Original WBC
seed 0 failed training; every remaining seed failed at least one joint gate.

This work uses a modified 24-DoF gripper asset, an 8 Nm arm cap and PD upper-body
control. It was inspected from committed receipts, not reproduced in this
publication checkout, and does not satisfy the 22-DoF H4 objective. Expert
demonstrations, VLA fine-tuning and RL VLA improvement remain open/unverified in
this audit; a four-phase plan is not evidence that all four phases ran.

### Later local WBC1c phase-1 evidence

The later saved **WBC1c** selection is **SELECTED, seed 7**. It is separate
from the negative WBC/WBC1b backup receipts above. Read-only inspection of the
local selection and its individual verdicts found:

| Seed | Turn-test v2 / qualification / Q2 | Q2 falls | Four straight walks | Unloaded push falls |
|---|---|---|---|---|
| 6 | PASS / NOT QUALIFIED / PASS | 0/12 | All ≥1.5 m; qualification drift fails 0/3 | 5/60 |
| 7 | PASS / QUALIFIED / PASS | 0/12 | All ≥1.5 m | 4/60 |
| 8 | PASS / QUALIFIED / PASS | 0/12 | All ≥1.5 m | 5/60 |

Jobs `21657607` and `21657608`, all three array tasks each, are recorded
COMPLETED with exit code 0. The actual local selection is
`/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/waiter-20261005/wbc1c/selection.json`,
SHA-256 `0b5fa5c65956a7c4d6bebd8100f49bf4dab0dc614e209122d949243e8a1ccbfc`.
It belongs to the dirty shared checkout at local committed head
`3f76dc1d575fc7d0e9c8b7864862ac046077b8ea`, not this published main checkout.
Publishing an immutable source/evidence bundle and reproducing its gate remain
follow-up work; this inspection did not rerun the simulation. The result uses
the modified 24-DoF robot with a 12-leg-joint policy and scripted upper-body
control. It does not close the 22-DoF H4 objective or phases 2–4. Loaded pushes
are reported separately, not selected on: seeds 7 and 8 each fell in 1/20.
