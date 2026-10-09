# Actual native LIO navigation

Scientific status: **SMOKE_ONLY**.

Smoke outcomes are excluded from qualification. Development uses one frozen learned gait, scripted external waypoints and ideal simulated sensors; these are not physical-robot results.

|Case|Seed|Clean goal|Fall|Contact|Post-init tracking|ATE RMSE(m)|Client p95(ms)|
|---|---:|---:|---:|---:|---:|---:|---:|
|straight|370000|0|0|0|1.000|0.0341|15.56|
|straight|370001|0|0|0|1.000|0.0415|15.43|

ATE aligns native IMU poses to independent truth with gravity-preserving yaw/translation and fixed scale1. This evaluator alignment never enters the controller.

Raw archive SHA256: `6c8738e8258f8ee862128bdd48938917a3b25beb54d6fdc5b776cb06e65cc62a`.
