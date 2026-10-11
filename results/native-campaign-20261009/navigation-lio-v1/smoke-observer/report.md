# Actual native LIO navigation

Scientific status: **SMOKE_ONLY**.

Smoke outcomes are excluded from qualification. Development uses one frozen learned gait, scripted external waypoints and ideal simulated sensors; these are not physical-robot results.

|Case|Seed|Clean goal|Fall|Contact|Post-init tracking|ATE RMSE(m)|Client p95(ms)|
|---|---:|---:|---:|---:|---:|---:|---:|
|straight|370000|0|0|0|1.000|0.0341|15.85|
|straight|370001|0|0|0|1.000|0.0416|15.90|

ATE aligns native IMU poses to independent truth with gravity-preserving yaw/translation and fixed scale1. This evaluator alignment never enters the controller.

Raw archive SHA256: `a8befe1d7b4b8a55726102ab43acc89eaee053a854875e03a8a5b54b2060c63d`.
