# Processing-schedule evidence — October 12, 2026

Start with the [reviewed scientific report](../../docs/PROCESSING_SCHEDULE_RESULTS_2026-10-12.md).
All 48 declared development episodes, 12 separate smoke episodes and the
40-pair input profile completed. There were no scientific retries or unrun cells.

- [Frozen design](scientific-protocol.json), [launch protocol](launch-protocol.json),
  [intake](intake.json) and [GPU identity](execution-start.json).
- [Smoke gates](qualified-smoke.json), [dispatcher completion](dispatch-result.json),
  [strict collection](collection.json), [independent numerical report](report-evidence.json).
- [Group CSV](groups.csv), [paired CSV](pairs.csv), and PNG/SVG figures.
- [Verbatim automatically generated report](REPORT.md). Read this alongside the
  [order audit](reporting/order-audit.json): global order alternation did not
  counterbalance order within either estimator. The reviewed report states this
  limitation explicitly.
- [Reporting code and frozen plan](reporting/), including the final artifact review.
- [Second independent trace audit](independent-trace-audit/), with physics-derived
  goal checks, actual command counts, coordinate invalidations and per-seed results.
- [Publication index](publication/publication-index.json),
  [independent archive/remote-asset audit](publication/independent-audit.json),
  and the append-only publication manifest. Original archives remain GitHub
  release assets rather than Git blobs.
- [Slurm terminal states](slurm-accounting.json), [source-equivalent CI](ci.json),
  [allocated validation log](focused-validation-21801920.log) and [storage](storage.json).

Source archive SHA256:
`46eb5d49361cb64f4f18dc8290a7bcb002c21b01aa035e3fe8c8e049949a0dda`.
Native scientific source commit: `152ccf4d8a616c5725bfb0c44ac838818e028ee3`.
Files retain their original HPC paths as provenance; these paths are not promised
to exist on another machine. Reproduction requires the pinned source/runtime,
assets and input layout described by the frozen manifests.

LiDAR–IMU goals were 0/12 reference versus 11/12 candidate; stereo remained
0/12 in both schedules. Fewer stale stops are not synonymous with usable control.
These two-seed simulation results do not validate hardware, physical concurrency
or an order-independent speedup.
