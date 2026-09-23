# Repository status — one page

**Updated:** 2026-09-23. **Branch:** `mission7-approach-followup` (all campaign
work; `main` untouched). Master backlog: [`REPO_TASKS.md`](REPO_TASKS.md).

| Workstream | State | Latest measured result | Next executable action |
|---|---|---|---|
| Mission 7 route controller | active | candidate 1 (selective reset) validated: Doors 1/16 → 1/16, Transport 2/16 → 4/16 on fresh layouts, +2/−0, underpowered | crossing composition vs exact replay; −x fine centring |
| Privileged Approach −x | reopened | 8/16 (coarse arms 7–9/16 or 0/16); walking envelope 0.546–0.605 m vs 0.61 m gap — marginal, not bound | `center` arm on the 16 −x layouts |
| Inspection maze | extending | 3/3 nominal, controls 0/3 (3 seeds) | 12 fresh seeds + dropout 0.35 + wrong-branch (`21401943–945`) |
| Multi-robot airlock | extending | 5/5 crew 2 and 3, controls 0/5 (5 seeds) | 10 fresh seeds, crews 2/3, dropout 0.35 (`21401946–948`) |
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
