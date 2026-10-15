# Processing-schedule order analysis

Design: `v1-20261012`. Primary verdict: **NOT_CONFIRMATORY_DESIGN**.

## lidar_inertial: 12 pairs, 2 seeds

| Pipeline | Clean goals | Wilson 95% |
| --- | ---: | --- |
| legacy_cumulative | 0/12 | [0.000, 0.242] |
| bounded_latest | 11/12 | [0.646, 0.985] |

Goal-rate difference 0.917; seed-cluster bootstrap 95% [0.833, 1.000] (exact enumeration of 3 cluster multisets with multinomial weights). Exact McNemar: 11 bounded-only vs 0 legacy-only, p = 0.000977. Seed sign test p = 0.5000.

| Paired difference (bounded - legacy) | Pairs | Mean | Median | Bootstrap 95% |
| --- | ---: | ---: | ---: | --- |
| modeled_pipeline_p95_ms | 12 | -5592.143 | -5858.060 | [-5854.459, -5329.827] |
| navigation_pose_age_p95_ms | 12 | -11794.667 | -11342.000 | [-11961.333, -11628.000] |
| native_client_p95_ms | 12 | -23.219 | -30.882 | [-25.162, -21.275] |
| stale_fraction | 12 | -0.725 | -0.786 | [-0.740, -0.710] |

Order test status: NOT_ESTIMABLE_SINGLE_SEQUENCE.

Across seeds, goal difference mean 0.917, SD 0.118, range 0.833 to 1.000.

## stereo: 12 pairs, 2 seeds

| Pipeline | Clean goals | Wilson 95% |
| --- | ---: | --- |
| legacy_cumulative | 0/12 | [0.000, 0.242] |
| bounded_latest | 0/12 | [0.000, 0.242] |

Goal-rate difference 0.000; seed-cluster bootstrap 95% [0.000, 0.000] (exact enumeration of 3 cluster multisets with multinomial weights). Exact McNemar: 0 bounded-only vs 0 legacy-only, p = 1.000000. Seed sign test p = 1.0000.

| Paired difference (bounded - legacy) | Pairs | Mean | Median | Bootstrap 95% |
| --- | ---: | ---: | ---: | --- |
| modeled_pipeline_p95_ms | 12 | -6518.799 | -5772.486 | [-7091.194, -5946.403] |
| navigation_pose_age_p95_ms | 12 | -11758.000 | -11106.500 | [-11991.333, -11524.667] |
| native_client_p95_ms | 12 | 4.762 | 6.836 | [3.197, 6.327] |
| stale_fraction | 12 | -0.883 | -0.921 | [-0.924, -0.841] |

Order test status: NOT_ESTIMABLE_SINGLE_SEQUENCE.

Across seeds, goal difference mean 0.000, SD 0.000, range 0.000 to 0.000.

Unscored warm-ups (reported separately): 0 episodes.

simulation schedule-model statistics over reset-seed clusters; no hardware, population or stereo claims
