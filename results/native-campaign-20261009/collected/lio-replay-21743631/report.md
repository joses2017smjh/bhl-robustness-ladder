# Actual native estimator replay

Scientific status: **PASS**.

Smoke is excluded from scientific qualification. These are simulated replay measurements, with zero closed-loop navigation episodes.

|Scene|Method|Tracked / frames|Post-init tracking|ATE RMSE (m)|Native p95 (ms)|Readiness|
|---|---|---:|---:|---:|---:|---|
|textured_boxes_native|lio|147/150|1.000|0.0104|10.10|PASS|
|thin_posts_native|lio|147/150|1.000|0.0196|10.86|PASS|
|ramp_step_native|lio|147/150|1.000|0.0142|10.62|PASS|

ATE uses independent evaluator-only rigid alignment with scale fixed to one. Native compute excludes capture, validation/export, process startup and imposed replay waits; it is distinct from client wall latency.

Raw archive SHA256: `ba1c6d497f54989a798082f9eb9b3f9e2c4d31e415c98f53f8c7c3ddc1c888b2`.
