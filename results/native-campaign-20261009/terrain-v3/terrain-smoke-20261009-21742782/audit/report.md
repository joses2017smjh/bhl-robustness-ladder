# Actual sensor-informed terrain traversal

Scientific status: **SMOKE_ONLY**.

Stopping short is a timeout even without falls. Qualification and confirmation are reported separately; shared actors/groups are paired clusters. All sensors and physics are simulated.

|Stage|Actor|Terrain|Group|Arm|Goal|Fall|Contact|Progress(m)|Height RMSE(m)|
|---|---|---|---:|---|---:|---:|---:|---:|---:|
|smoke|dr-default-s0|flat|250000|baseline|1|0|0|5.017|0.0025|
|smoke|dr-default-s0|ramp|250000|lidar|0|0|0|1.582|0.1102|

Height error compares raw inferred maps with independent dense scene geometry after commands were computed. Wall-top/side aliasing is included; these values are not ground-only accuracy. Smoke never qualifies actors.
