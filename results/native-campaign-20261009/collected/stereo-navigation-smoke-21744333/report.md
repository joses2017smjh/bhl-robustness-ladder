# Actual native stereo estimated-pose navigation

Scientific status: **SMOKE_ONLY**.

actual native ORB stereo pose plus raw LiDAR obstacle brake and frozen 12-DoF learned gait; simulated body-attached images, known start/external routes; no hardware claim. Smoke outcomes are excluded from qualification.

|Case|Group|Goal|Fall|Contact|Original stereo pairs|Post-init tracking|Client p95(ms)|
|---|---:|---:|---:|---:|---:|---:|---:|
|straight|370000|0|0|0|29|1.000|38.26|
|straight|370001|0|0|0|29|0.929|38.87|

Original PNG hashes, camera-to-IMU transforms, native map IDs and causal response arrivals are independently checked. Drift scoring uses evaluator-only gravity/yaw/translation alignment with fixed metric scale1; no evaluator correction enters navigation.
