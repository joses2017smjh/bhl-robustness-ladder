# Repository status — one page

**Updated:** 2026-09-23. **Branch:** `mission7-approach-followup` (all campaign
work; `main` untouched). Master backlog: [`REPO_TASKS.md`](REPO_TASKS.md).

| Workstream | State | Latest measured result | Next executable action |
|---|---|---|---|
| Mission 7 route controller | active | candidate 1 (selective reset) validated: Doors 1/16 → 1/16, Transport 2/16 → 4/16 on fresh layouts, +2/−0, underpowered | crossing composition vs exact replay; −x fine centring |
| Sensor fusion (new 2026-09-23) | **methodology + first measurement** | attitude filter in the MuJoCo loop: frozen gait tolerates L1 IMU noise (RMSE ≤ 0.05) but **falls with ≥ 40 ms IMU delay** (0/3 every delayed arm) | filter runs at the 25 Hz policy rate; 5–40 ms boundary unmeasured; no hardware, no localization estimator, all sensors simulated |
| Privileged Approach | **gate passed, held-out replicated** | 62/64, 0 falls (test split); held-out −x **21/23 vs 12/23**, p = 0.012 | sensor studies still wait on the route gate |
| Inspection maze | extended, gate holds | **12/12 nominal, 0/12 complete outage, 12/12 at 35 % dropout, wrong-branch 12/12 rejected** on fresh seeds 3–14 | more layouts/approach orders need new script options (not yet exposed) |
| Multi-robot airlock | extended, gates hold | crews 2 and 3 **10/10 on fresh seeds 5–14 with both controls 0/10**; crew 2 10/10 at 35 % dropout | crew 3 dropout arm running; delay/interruption stress needs script options |
| Locomotion / terrain / sensors | reference | pairs valid; ice_pair retracted, ice_pair_placed n=2 | multi-seed comparisons per inventory |
| Navigation PPO / B5 | reference + failure | 379/384 oracle-route; maze button 0/12 | provenance reconciled, then multi-seed/route eval |
| Cooperative lift / carry / cube-to-shelf | unresolved failures | best lift 7.8 cm; 0/6, 0/18; underground-spawn artifacts retracted | feasibility diagnostics before any training |
| Cloth-sort proxies | reference | free-base shirt 22/24, jacket 8/8; fixed-base 32/32 (pinned) | more layouts/objects, free base |
| Dual-arm folding (linked repo) | **external driver active** | v2 closed negative (baseline retained); v3 recovery-supervision running under its own driver | do not modify while `lh-v3-*` runs; consume its REPORT/STATUS |

Envelope for this campaign: ≤8 concurrent jobs; CPU on `share` with
`--constraint=haswell&el8` for deterministic controls; GPU ≤2 concurrent on
`gpu`/`ampere`, ≤24 GPU-h for evaluation and rendering plus ≤24 GPU-h reserved
for one justified training/adaptation; ≥100 GB free; compact retained evidence
(raw traces trimmed, verdicts and receipts kept). Mission 7's remaining 101
CPU episodes are its own budget line.
