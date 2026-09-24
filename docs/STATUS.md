# Repository status — one page

**Updated:** 2026-09-23. **Branch:** `mission7-approach-followup` (all campaign
work; `main` untouched). Master backlog: [`REPO_TASKS.md`](REPO_TASKS.md).

| Workstream | State | Latest measured result | Next executable action |
|---|---|---|---|
| Mission 7 route controller | active | candidate 1 (selective reset) validated: Doors 1/16 → 1/16, Transport 2/16 → 4/16 on fresh layouts, +2/−0, underpowered | crossing composition vs exact replay; −x fine centring |; five stage variants replayed 2026-09-24, best 9/10, none pass the 10/10 gate
| Sensor fusion (new 2026-09-23) | **methodology + first measurement** | frozen gait tolerates L1 IMU noise; **IMU latency budget ≈ 30 ms** (200 Hz filter: 3/3 to 30 ms, 1/3 at 40, 0/3 at 60); route controller tolerates ≤ 0.15 m bounded position error and ≤ 20° heading but **any drift fails** | Isaac SF-01 (clean): noise tolerated; **one 20 ms IMU delay halves maze success (0.97 → 0.48), two end it**; **delay-only fine-tune restores 32/32 and 30/32** (control 27 and 0); **distilled recurrent student keeps ≥ 27/32 and the all-ingredients MLP fine-tune ≥ 29/32 under every tested outage and delay** (teacher 0/32 with a sensor zeroed; fixed route, Blind already 32/32); SF-04 negative: the recurrent dropout/delay arm falls in every Corridor episode (0/32); no hardware, no localization estimator, all sensors simulated |
| Privileged Approach | **gate passed, held-out replicated** | 62/64, 0 falls (test split); held-out −x **21/23 vs 12/23**, p = 0.012 | sensor studies still wait on the route gate |
| Inspection maze | extended, gate holds | **12/12 nominal, 0/12 complete outage, 12/12 at 35 % dropout, wrong-branch 12/12 rejected** on fresh seeds 3–14 | more layouts/approach orders need new script options (not yet exposed) |
| Multi-robot airlock | extended, gates hold | crews 2 and 3 **10/10 on fresh seeds 5–14 with both controls 0/10**; crew 2 10/10 at 35 % dropout | crew 3 dropout arm running; delay/interruption stress needs script options |
| Locomotion / terrain / sensors | reference + probes | pairs valid; ice_pair retracted; **placed-ice rung confirmed EXPOSED** (63/64 episodes touch ice); matched push: push-adaptive 0.100 vs control 0.211 (p = 0.056); lab traverse 5/20 vs 2/20 over 5 seeds | ice no-ice control unfunded; depth bench rerun pending |
| Navigation PPO / B5 | reference + failure | 379/384 oracle-route; maze button 0/12; **noise-on rescore 380/384** (caveat closed) | provenance reconciled, then multi-seed/route eval |
| Cooperative lift / carry / cube-to-shelf | **cube-to-shelf results invalidated by a diagnosed bug** | best lift 7.8 cm; occluded coop scores zero lift in MuJoCo (6 ckpts); TaskV2 episodes were killed at step 1 by an xyzw/wxyz misread in the fall check (fixed, rerun pending); crew 3/4 configs repaired and trained 4000 it: **no crew lift** (curriculum stuck at the first stage) | the pinch crouch needs 28–30 Nm against a 6 Nm effort limit (`21402699`), so it cannot be held at any spawn height; training withheld until the spawn posture and object placement are redesigned |
| Cloth-sort proxies | reference | free-base shirt 22/24, jacket 8/8; fixed-base 32/32 (pinned) | more layouts/objects, free base |
| Dual-arm folding (linked repo) | **external driver active** | v2 closed negative (baseline retained); v3 recovery-supervision running under its own driver | do not modify while `lh-v3-*` runs; consume its REPORT/STATUS |

Envelope for this campaign: ≤8 concurrent jobs; CPU on `share` with
`--constraint=haswell&el8` for deterministic controls; GPU concurrency ≤8 jobs of ours, GPU hours uncapped from 2026-09-24 (user) on
`gpu`/`ampere`, ≤24 GPU-h for evaluation and rendering plus ≤24 GPU-h reserved
for one justified training/adaptation; ≥100 GB free; compact retained evidence
(raw traces trimmed, verdicts and receipts kept). Mission 7's remaining 101
CPU episodes are its own budget line.

> **Disk, 2026-09-23 evening.** Free space on `/nfs/hpc/share` dipped to 96 GB (the share is 94 % used, 1.4 TB of it under this user's quota). Restored to 104 GB by gzip-compressing raw Mission 7 traces over 100 MB in place and deleting six unreferenced folding smoke-test checkpoints (6.7 GB). Largest remaining reclaim candidates, not touched: `results/weekend-20260919/fold-adapt-s{0,1}` (34 GB each, one 1.14 GB checkpoint per 100 steps; keeping only the last would free ≈ 60 GB) and `Humanoid_Lite/logs` (4.4 GB, 1,367 files).
