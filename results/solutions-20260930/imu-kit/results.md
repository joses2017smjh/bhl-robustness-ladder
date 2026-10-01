# imu_allan.py synthetic recovery: true vs recovered at 100 and 200 Hz

Tool: `/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/imu-kit/repo_draft/scripts/sensors/imu_allan.py`  (function level: `overlapping_adev` + `fit_allan`, one seed per row; total 103 s, numpy 1.26.0).

Process per channel: white (per-sample std drawn as N*sqrt(fs)) + flicker (Allan floor 0.6643 B) + ImuNoise-style rate random walk K. Gyro: N 0.0035 deg/s/rtHz, B 10 deg/h, K 0.000261 deg/s/rt s. Accel: N 0.070 mg/rtHz, B 0.05 mg, K 0.00407 mg/rt s.

B truth for the min method = analytic minimum of the generated process / 0.6643 (white and RRW lift the floor above 0.6643 B); B_fit is compared with the flicker parameter B. K = +1/2 line at tau = 3 s (least-squares level; K_graph = floor-subtracted graphical read). n/v = not visible (only the bound K_upper is reported).

| fs | T | sensor | N true | N rec (ratio) | sigma_d true = N*sqrt(fs) | N_rec*sqrt(fs) | N_rec*sqrt(fs/2) | B analytic-min/0.664 | B_min rec (ratio) | tau_B rec / analytic (s) | B param | B_fit (ratio) | K true | K rec (ratio) | K_graph (ratio) | K_upper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 Hz | 2 h | gyro | 0.0035 deg/s/rtHz | 0.003499 (0.9996) | 0.035 deg/s | 0.03499 (0.9996) | 0.02474 (0.707) | 11.45 deg/h | 11.05 (0.966) | 20 / 24 | 10 | 10.04 (1.004) | 0.000261 deg/s/rt s | 0.0002368 (0.91) | 0.000301 (1.15) | 0.0004797 (1.84) |
| 100 Hz | 2 h | accel | 0.07 mg/rtHz | 0.06994 (0.9991) | 0.7 mg | 0.6994 (0.9991) | 0.4945 (0.706) | 0.05697 mg | 0.05155 (0.905) | 45 / 29 | 0.05 | 0.04955 (0.991) | 0.00407 mg/rt s | 0.003019 (0.74) | 0.004166 (1.02) | 0.006579 (1.62) |
| 100 Hz | 0.5 h | gyro | 0.0035 deg/s/rtHz | 0.003502 (1.0007) | 0.035 deg/s | 0.03502 (1.0007) | 0.02477 (0.708) | 11.44 deg/h | 9.326 (0.815) | 32 / 23 | 10 | 9.914 (0.991) | 0.000261 deg/s/rt s | n/v | n/v | 0.0005498 (2.11) |
| 100 Hz | 0.5 h | accel | 0.07 mg/rtHz | 0.06964 (0.9949) | 0.7 mg | 0.6964 (0.9949) | 0.4925 (0.704) | 0.05697 mg | 0.04607 (0.809) | 142 / 31 | 0.05 | 0.04957 (0.991) | 0.00407 mg/rt s | n/v | n/v | 0.007064 (1.74) |
| 200 Hz | 2 h | gyro | 0.0035 deg/s/rtHz | 0.003502 (1.0006) | 0.0495 deg/s | 0.04953 (1.0006) | 0.03502 (0.708) | 11.45 deg/h | 11.86 (1.036) | 13 / 24 | 10 | 9.818 (0.982) | 0.000261 deg/s/rt s | 0.0002837 (1.09) | 0.0002575 (0.99) | 0.0003858 (1.48) |
| 200 Hz | 2 h | accel | 0.07 mg/rtHz | 0.06994 (0.9992) | 0.9899 mg | 0.9891 (0.9992) | 0.6994 (0.707) | 0.05697 mg | 0.05877 (1.032) | 45 / 29 | 0.05 | 0.04926 (0.985) | 0.00407 mg/rt s | 0.004772 (1.17) | 0.004158 (1.02) | 0.007802 (1.92) |
| 200 Hz | 0.5 h | gyro | 0.0035 deg/s/rtHz | 0.003503 (1.0009) | 0.0495 deg/s | 0.04954 (1.0009) | 0.03503 (0.708) | 11.44 deg/h | 10.64 (0.930) | 160 / 23 | 10 | 10.01 (1.001) | 0.000261 deg/s/rt s | n/v | n/v | 0.000436 (1.67) |
| 200 Hz | 0.5 h | accel | 0.07 mg/rtHz | 0.06979 (0.9970) | 0.9899 mg | 0.9869 (0.9970) | 0.6979 (0.705) | 0.05697 mg | 0.05741 (1.008) | 18 / 30 | 0.05 | 0.04914 (0.983) | 0.00407 mg/rt s | 0.00634 (1.56) | 0.0063 (1.55) | 0.01294 (3.18) |

White-only control (same white draw, nothing else): N_rec*sqrt(fs) / sigma_d drawn:

- 100 Hz 2 h gyro: drawn std 0.9988 x sigma_d; N_rec*sqrt(fs) = 0.9992 x sigma_d (N*sqrt(fs/2) would be 0.707)
- 100 Hz 2 h accel: drawn std 0.9994 x sigma_d; N_rec*sqrt(fs) = 0.9989 x sigma_d (N*sqrt(fs/2) would be 0.706)
- 100 Hz 0.5 h gyro: drawn std 0.9989 x sigma_d; N_rec*sqrt(fs) = 1.0001 x sigma_d (N*sqrt(fs/2) would be 0.707)
- 100 Hz 0.5 h accel: drawn std 0.9947 x sigma_d; N_rec*sqrt(fs) = 0.9946 x sigma_d (N*sqrt(fs/2) would be 0.703)
- 200 Hz 2 h gyro: drawn std 1.0007 x sigma_d; N_rec*sqrt(fs) = 1.0002 x sigma_d (N*sqrt(fs/2) would be 0.707)
- 200 Hz 2 h accel: drawn std 0.9997 x sigma_d; N_rec*sqrt(fs) = 0.9987 x sigma_d (N*sqrt(fs/2) would be 0.706)
- 200 Hz 0.5 h gyro: drawn std 1.0007 x sigma_d; N_rec*sqrt(fs) = 1.0015 x sigma_d (N*sqrt(fs/2) would be 0.708)
- 200 Hz 0.5 h accel: drawn std 0.9971 x sigma_d; N_rec*sqrt(fs) = 0.9962 x sigma_d (N*sqrt(fs/2) would be 0.704)
