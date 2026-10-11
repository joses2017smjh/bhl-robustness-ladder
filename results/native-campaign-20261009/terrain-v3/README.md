# Terrain traversal: completed negative qualification screen

The actual frozen-gait screen completed **18/18 episodes**: **3 clean goals,0 falls,15 side-wall contacts,0 numerical failures**. No actor achieved the predeclared6/6 clean goals. Confirmation stopped with **zero episodes**; no LiDAR-versus-baseline effect was measured. The scored screen uses the blind baseline only.

|Actor|Clean goals / six|Side-wall contacts|Falls|Qualified|
|---|---:|---:|---:|---|
|DR-default-s0|3/6|3|0|No|
|DR-default-s1|0/6|6|0|No|
|DR-default-s2|0/6|6|0|No|

Fresh smoke21742782 completed as SMOKE_ONLY: flat blind reference reached5.02m and survived40s; LiDAR ramp stopped at1.58m and timed out, with0falls/contacts. Smoke did not qualify actors, and no threshold was tuned after the negative ramp outcome.

The independently audited scored raw archive is19,583,862bytes, SHA256 `56f89af26505e2015185cfd9d635e618ad4616b982867929943b9f7465f87a99`. [All episode outcomes and raw-map audit](terrain-run-20261009-21742786/audit/report.md), [original raw data](terrain-run-20261009-21742786/outputs.tar.gz), [cohort summary](actual-results-summary.json), and [terrain map aggregates](screen-terrain-aggregates.json) are retained.

The raw3D returns, masks,IMU attitudes,inferred maps,and independent dense scene-height truth were checked against every episode/map frame. Height metrics include wall-top/side aliasing and repeated observations across time; they are not ground-only accuracy or independent sample counts.

V3 used exact original834source/model input hashes, CPU2/8GB/preempt/no-requeue, with fresh smoke21742782 gating scored21742786. Source SHA256 `1f04b186e62a31bfa27547375124b85f7bc8a46b688fc92577ea108d7d6293ed`. V1/v2 pending scored jobs were cancelled before any scored episodes began; their receipts and complete source-payload reconstruction maps are retained. Every scientific/failure outputs.tar.gz remains preserved.

All physics/sensors are simulated and the actors are12-DoF bipeds. The result identifies insufficient frozen-gait course qualification. It does not establish a fusion benefit, hardware safety, or terrain generalization.
