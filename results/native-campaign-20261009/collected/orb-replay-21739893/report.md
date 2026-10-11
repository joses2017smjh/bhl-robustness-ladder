# Actual native estimator replay

Scientific status: **NEGATIVE**.

Smoke is excluded from scientific qualification. These are simulated replay measurements, with zero closed-loop navigation episodes.

|Scene|Method|Tracked / frames|Post-init tracking|ATE RMSE (m)|Native p95 (ms)|Readiness|
|---|---|---:|---:|---:|---:|---|
|textured_boxes_native|orb|0/150|0.000|unscorable|45.11|NEGATIVE|
|thin_posts_native|orb|0/150|0.000|unscorable|27.02|NEGATIVE|
|ramp_step_native|orb|122/150|0.976|0.0044|48.60|PASS|

ATE uses independent evaluator-only rigid alignment with scale fixed to one. Native compute excludes capture, validation/export, process startup and imposed replay waits; it is distinct from client wall latency.

Raw archive SHA256: `f7e5659742e737445c949b255a5abb1a45bbe4b4fac5538fbe8d7c25e15b01e4`.
