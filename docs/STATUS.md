# Repository status — one page

**Updated:** 2026-10-08. **Published branch:** `main`. The current
[task closure ledger](TASK_CLOSURE_2026-10-08.md) distinguishes completed
experiments from unmet robotics objectives. [Master backlog](REPO_TASKS.md) ·
[Roadmap](ROADMAP.md). Older findings and protocols remain their dated records.

| Workstream | Current state | Latest verified result | Next executable action |
|---|---|---|---|
| H1 matched navigation | DONE, PASS | All 576 records independently verified; candidate 259/288 goals and 0/288 falls versus baseline 247/288 and 9/288; 51.5% lower mean yaw-command sign-flip rate. All three actors pass unchanged gates. | [Report and limits](H1_NAVIGATION_CONFIRMATION_2026-10-08.md). Existing scripted filter, 12-DoF biped, oracle pose/goal; 48 shared layout blocks. |
| H2 portable replay | DONE | Fresh pinned CPU installation; local full suite 1,998 passed, 123 existing optional checks skipped, six subtests passed. [Hosted CI](https://github.com/joses2017smjh/bhl-robustness-ladder/actions/runs/37817975052) succeeds; five exact repeats and 40/40 regressions detected. Second HPC host agrees on 5/5 frozen two-second cases, maximum numeric difference 0.0. [Closure receipt](../results/task-closure-20261008/h2-replay/closure.json). | Maintain the pinned regression gate. Scope remains this frozen bundle and five comparison cases; original timings retain their warm CPU scope. |
| H3 history/latency training | ACTIVE; objective OPEN | Both GPU arms pass non-scored smoke `21711074`; six-cell array `21711075` running; first cell finished 500 iterations and 120 independently verified episodes | Complete all six cells and independently recompute the frozen 720-episode paired gate. [Protocol](H3_H4_CAMPAIGN_2026-10-08.md). |
| H4 turning humanoid | ACTIVE; objective OPEN; R1/R2/R1H DONE NEGATIVE | New 22-DoF R1HO recipe: fresh smoke `21711207` PASS, three-seed array `21711208` running. First smoke stopped before training on a configuration-capture check; receipt retained | Complete all three seeds and recompute ≥2/3 passing unchanged turn/walk/push rules. [New protocol](H3_H4_CAMPAIGN_2026-10-08.md); [prior verdicts](TASK_CLOSURE_2026-10-08.md#completed-negative-experiments). |
| Mission 7 routes | BLOCKED | Selective reset confirmation Transport 2/16 →4/16 is underpowered; scripted benches fail, PlateCross/PlateCross2 select no qualified seed, so no bench | Reconcile episode budget; genuine-crossing bench, exact replay 10/10, then both routes 16/16. Sensor release remains gated. |
| Privileged Approach | DONE | 62/64, zero falls; held-out −x 21/23 versus 12/23, p=0.012 | Sensor studies still wait on the route gate. |
| Sensor fusion | Simulated studies complete; calibrated capture/fusion OPEN | Frozen-gait latency budget approximately 30 ms; simulated dropout/delay studies and controls retained. Ray-depth navigation and calibrated image stereo are distinct pipelines. | [Stereo–lidar research ladder](STEREO_LIDAR_RESEARCH_2026-10-08.md): capture/calibration, motion-aware fusion, estimated pose. Physical IM10A recordings remain SF-05. |
| Navigation / B5 | Original v5 transfer DONE; older 384-episode confirmation NEGATIVE; fresh H1 PASS | Original v5 goals 11/12, 12/12, 12/12; older cohort six learned-actor falls versus zero A*. H1 independently verifies a separate matched filter comparison. | Estimated-pose navigation on a newly declared cohort; retain oracle pose/goal and clustered-layout limits of existing results. |
| Inspection media / controls | Media DONE; extension partly open | Sensor-panel recording delivered; 12 wrong-branch episodes rejected, 12 dropout episodes succeed | Complete remaining larger-extension and explicit wrong-branch gate work; do not infer control completeness from one media capture. |
| Multi-robot airlock | PARTIAL extension | Crews 2/3 each 10/10 fresh nominal, both controls 0/10, and 10/10 at 35% dropout | Original proposal asked twenty seeds; delay/interruption stress needs script options. |
| Locomotion / terrain / media | References and reporting corrections DONE; selected probes OPEN | LOC-01/04/06/07/10 provenance/captions closed; no-ice control and live-depth mechanism are already measured, not queued/unfunded | LOC-03 flatfill retention, LOC-09 departure timeout, LOC-11 legacy export/render provenance. |
| Cooperative carry / placement | OPEN, completed experiments negative | Stand4 last-200 success 0.006/0.001 versus 0.10; cube rolling/release and carry failures retained | Physically supported retention, wrist orientation and independently validated placement/release. |
| Cloth-sort code / extensions | Quaternion defect DONE; performance extensions OPEN | Correct quaternion initializer and 13 CPU checks; no new Isaac execution or task success | Additional shirt seeds and solver diagnostics remain separate. [Receipt](../results/task-closure-20261008/cloth-quaternion.json). |
| Linked folding | Outside this closeout | Historical v2/v3 outcomes remain in linked campaign reports; current external driver state was not re-audited | Read the linked repository's current STATUS/REPORT before any work there. |
| Waiter evidence | Separate; later WBC1c local, unpublished and not reproduced here | Backup WBC/WBC1b NEGATIVE. Later WBC1c selects seed 7: seeds 7/8 pass v2, qualification and Q2, 0/12 Q2 falls and four ≥1.5 m walks each; modified 24-DoF asset with scripted upper body | Publish immutable source/evidence and reproduce the gate separately; H4 and later VLA phases remain open. [Evidence review](TASK_CLOSURE_2026-10-08.md#separate-waiter-backup-branch). |

Historical 2026-09-23/24 campaign envelope: ≤8 concurrent jobs; CPU on `share` with
`--constraint=haswell&el8` for deterministic controls; GPU concurrency ≤8 jobs of ours, GPU hours uncapped from 2026-09-24 (user) on
`gpu`/`ampere`, ≤24 GPU-h for evaluation and rendering plus ≤24 GPU-h reserved
for one justified training/adaptation; ≥100 GB free; compact retained evidence
(raw traces trimmed, verdicts and receipts kept). Mission 7's CPU episodes are
its own budget line; the remaining count is inconsistent (101 in the docs, ≈ 53 over in the
ledger) and unresolved until the user reconciles it (ledger, 2026-10-01).

> **Disk, 2026-09-23 evening.** Free space on `/nfs/hpc/share` dipped to 96 GB (the share is 94 % used, 1.4 TB of it under this user's quota). Restored to 104 GB by gzip-compressing raw Mission 7 traces over 100 MB in place and deleting six unreferenced folding smoke-test checkpoints (6.7 GB). Largest remaining reclaim candidates, not touched: `results/weekend-20260919/fold-adapt-s{0,1}` (34 GB each, one 1.14 GB checkpoint per 100 steps; keeping only the last would free ≈ 60 GB) and `Humanoid_Lite/logs` (4.4 GB, 1,367 files).

> **Quota, 2026-10-01.** Project 30762 holds 1.529 TB against its 1.5 TB soft quota (2 TB hard), with 3 weeks 5 days of grace (`lfs quota -p 30762 /nfs/hpc/share`). The `fold-adapt-s{0,1}` reclaim estimate above is stale: those folders now hold 4.3 GB each. What to prune is the user's decision.

> **Quota, 2026-10-02.** The user approved deleting the intermediate training checkpoints: 49.5 GB in 360 finished runs (finals and every cited or parent checkpoint kept; manifest `solutions-20260930/ckpt_deleted_20261002.tsv`). The project went from 1.549 TB to 1.514 TB, still about 14 GB over the 1.5 TB soft quota.
