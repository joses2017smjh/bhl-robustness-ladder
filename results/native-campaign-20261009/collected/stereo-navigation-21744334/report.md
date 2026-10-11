# Actual native stereo estimated-pose navigation

Scientific status: **NEGATIVE**.

actual native ORB stereo pose plus raw LiDAR obstacle brake and frozen 12-DoF learned gait; simulated body-attached images, known start/external routes; no hardware claim. Smoke outcomes are excluded from qualification.

|Case|Group|Goal|Fall|Contact|Original stereo pairs|Post-init tracking|Client p95(ms)|
|---|---:|---:|---:|---:|---:|---:|---:|
|straight|380000|0|0|0|189|0.529|33.31|
|straight|380001|1|0|0|189|0.534|36.07|
|straight|380002|0|0|0|189|0.477|37.29|
|dogleg|380000|0|0|0|189|0.270|38.75|
|dogleg|380001|0|0|0|189|0.184|36.91|
|dogleg|380002|0|0|0|189|0.218|35.33|
|occluders|380000|0|0|0|189|0.241|36.96|
|occluders|380001|0|0|0|189|0.994|35.16|
|occluders|380002|0|0|0|189|0.276|35.83|

Original PNG hashes, camera-to-IMU transforms, native map IDs and causal response arrivals are independently checked. Drift scoring uses evaluator-only gravity/yaw/translation alignment with fixed metric scale1; no evaluator correction enters navigation.
