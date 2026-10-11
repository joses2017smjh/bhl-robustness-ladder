# Native tightly coupled fusion replay

Native FAST-LIVO2 uses the left camera, timed 3D LiDAR and IMU. Right stereo images are not fused.

| Sequence | Method | Tracked | Visual measurement frames | ATE RMSE (m) | Native p95 (ms) | Gate |
|---|---|---:|---:|---:|---:|---|
| textured_boxes_native | fast_lio2 | 147/150 | n/a | 0.01038 | 10.113 | PASS |
| textured_boxes_native | fast_livo2 | 148/150 | 147 | 0.02695 | 18.234 | PASS |
| textured_boxes_native | fast_livo2_no_visual | 148/150 | 0 | 0.02656 | 3.745 | PASS |
| thin_posts_native | fast_lio2 | 147/150 | n/a | 0.01959 | 10.921 | PASS |
| thin_posts_native | fast_livo2 | 148/150 | 147 | 0.24364 | 19.312 | NEGATIVE |
| thin_posts_native | fast_livo2_no_visual | 148/150 | 0 | 0.03361 | 3.821 | PASS |
| ramp_step_native | fast_lio2 | 147/150 | n/a | 0.01417 | 10.110 | PASS |
| ramp_step_native | fast_livo2 | 148/150 | 147 | 0.03372 | 18.647 | PASS |
| ramp_step_native | fast_livo2_no_visual | 148/150 | 0 | 0.02022 | 3.690 | PASS |

Same calibrated sensor bundle; FAST-LIO2 emits LiDAR-tail poses, FAST-LIVO2 emits camera-time poses. Fixed-scale gravity/yaw/translation alignment and bracketed evaluator truth. Tracking coverage is reported alongside conditional pose error.

FAST-LIVO2 camera+LiDAR+IMU versus its same-binary LiDAR+IMU arm; existing FAST-LIO2 has a different native map estimator.

Qualification is separate from successful execution. No closed-loop or physical-hardware improvement is inferred.
