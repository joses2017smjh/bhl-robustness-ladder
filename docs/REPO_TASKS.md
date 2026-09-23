# Repository master backlog

Every experiment and gallery entry mapped to its code, configuration,
checkpoint, evaluator and evidence, with a classification and the single next
executable action. Classifications: **completed reference**, **active
improvement**, **unresolved failure**, **dependency-blocked**, **retracted**,
**superseded**. A working renderer is not a successful experiment. Numbers in
the gallery and README are starting points; each row states what the evidence
files actually contain.

Status page: [`STATUS.md`](STATUS.md). Mission 7 detail: [`MISSION7_TASKS.md`](MISSION7_TASKS.md).
Folding detail: the linked repository's own `campaigns/*/STATUS.md`.

## Compute plan (declared 2026-09-23)

| Line | Limit | Accounting |
|---|---|---|
| Concurrency | ≤ 8 jobs | this session's `m7-`, `insp-`, `airlock`, `loco-`, `b5-`, `coop-` jobs |
| CPU (`share`, haswell&el8) | as needed within site limits | per-workstream episode counts below |
| GPU (`gpu`/`ampere`, ≤2 concurrent) | ≤ 24 GPU-h eval/render + ≤ 24 GPU-h one justified training | logged per job in `SLURM_JOBS.md` |
| Disk | ≥ 100 GB free; raw traces trimmed | `df` checked at each checkpoint |
| Mission 7 | 101 CPU episodes remaining of 512 | `MISSION7_TASKS.md` |
| Folding repo | none from this session while its driver runs | — |

## Coordination note

`lehome-fold-repro` (IsaacSimFolding, branch `main`) is being executed by its
own driver (`scripts/driver.py`, jobs `lh-v3-*`, campaign
`20260923-recovery-supervision-v3`). This session does not edit that
repository or submit folding jobs while the driver is live; its `REPORT.md`
and `STATUS.md` are consumed read-only and cross-referenced here.

## Backlog

(filled from the inventory workflow; one row per entry)

