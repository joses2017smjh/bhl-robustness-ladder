# Actual native LIO navigation

Scientific status: **PASS**.

Smoke outcomes are excluded from qualification. Development uses one frozen learned gait, scripted external waypoints and ideal simulated sensors; these are not physical-robot results.

|Case|Seed|Clean goal|Fall|Contact|Post-init tracking|ATE RMSE(m)|Client p95(ms)|
|---|---:|---:|---:|---:|---:|---:|---:|
|straight|380000|1|0|0|1.000|0.0233|13.97|
|straight|380001|1|0|0|1.000|0.0148|14.16|
|straight|380002|1|0|0|1.000|0.0250|13.96|
|dogleg|380000|1|0|0|1.000|0.0204|13.96|
|dogleg|380001|1|0|0|1.000|0.0177|13.69|
|dogleg|380002|1|0|0|1.000|0.0166|13.91|
|occluders|380000|1|0|0|1.000|0.0289|14.12|
|occluders|380001|1|0|0|1.000|0.0178|13.95|
|occluders|380002|1|0|0|1.000|0.0146|13.55|

ATE aligns native IMU poses to independent truth with gravity-preserving yaw/translation and fixed scale1. This evaluator alignment never enters the controller.

Raw archive SHA256: `98c8e35a146fa98046a9b1973ae388a4939431db2a2ae7fb45b5815a3c279633`.
