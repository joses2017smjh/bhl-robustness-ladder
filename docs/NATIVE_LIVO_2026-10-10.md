# Native FAST-LIVO2 methodology implementation — 2026-10-10

The new native integration combines the **left camera, timed 3D LiDAR and IMU** through FAST-LIVO2's original C++ estimator. This is the one-camera configuration described by [Zheng et al., FAST-LIVO2](https://arxiv.org/abs/2408.14035), rather than a stereo estimator. Its original IMU propagation, voxel-map LiDAR update and direct photometric visual update are compiled from the [official implementation](https://github.com/hku-mars/FAST-LIVO2/tree/0d2c0346107b75b59934975adec9a6eeeb913c64).

Implementation status and measured scientific results are separate. The actual native build passed in Slurm step `21756762.10`, and all 10 transport tests passed. Runtime/source/dependency hashes and the initial link-path failure are retained in [`results/methods-campaign-20261010/livo`](../results/methods-campaign-20261010/livo). Actual native smoke and the nine-cell retained-capture diagnostic have also completed. Full visual fusion qualified 2/3 scenes versus 3/3 for each LiDAR–IMU arm; its thin-posts result was negative. This reused development capture provides no fresh-cohort or navigation improvement claim. The previous October 9 results remain the baseline.

## Measured retained-capture diagnostic

The frozen 35-frame smoke exercised native photometric measurements in 32 frames; both visual and no-visual arms tracked 33/35. The subsequent predeclared diagnostic ran all three native arms on the original 450-image capture, producing **1,350 native frame outputs across nine cells**. These are consumed development inputs, not a held-out confirmation set. The protocol, receipts, raw native frames, trajectories, logs and evaluations are retained in [`measured/`](../results/methods-campaign-20261010/livo/measured).

| Scene | FAST-LIO2 ATE (m) | FAST-LIVO2 ATE (m) | Same-binary no-visual ATE (m) |
|---|---:|---:|---:|
| Textured boxes | 0.01038 | 0.02695 | 0.02656 |
| Thin posts | 0.01959 | **0.24364 — gate failed** | 0.03361 |
| Ramp and step | 0.01417 | 0.03372 | 0.02022 |

FAST-LIO2 tracked 147/150 frames in each scene. Both FAST-LIVO2 arms tracked 148/150; full fusion included real visual measurements in 147/150, while the ablation used none. Native p95 compute was 10.11–10.92 ms for FAST-LIO2, 18.23–19.31 ms for full FAST-LIVO2, and 3.69–3.82 ms for the same-binary ablation. Timings exclude image decode and sensor capture; receipts retain separate decode-plus-compute and process-wall measurements.

The full visual arm has higher conditional ATE in all three scenes. **There is no demonstrated improvement from visual fusion in this diagnostic.** Qualification is 3/3 for FAST-LIO2, 2/3 for full FAST-LIVO2 and 3/3 for its no-visual arm. Source, parameters and gate remain unchanged for the fresh development cohort; failure attribution requires that separate paired experiment. No closed-loop episode or physical sensor trial was run here.

## Native integration

- FAST-LIVO2 commit: `0d2c0346107b75b59934975adec9a6eeeb913c64`.
- Non-templated Sophus commit: `a621ff2e56c56c839a6c40418d42c3c254424b5c`.
- Officially recommended Vikit fork commit: `6c886c8e5d83997806e00294826d528cea3581dd`.
- The original `IMU_Processing.cpp`, `voxel_map.cpp`, `vio.cpp`, `frame.cpp`, `visual_point.cpp` and estimator headers are checked against their pinned Git bytes before compilation. The build receipt records every compiled input, package hash, private dependency header, binary and runtime library.
- ROS time, IMU and visualization message containers are replaced by typed headless shims. Estimation does not depend on ROS publishing. The transport follows the original mapper's sequential LIO and VIO updates, covariance propagation and voxel insertion. It omits ROS subscriptions, visualization publication and disk point-cloud export.
- The original non-inline IMU comparator is kept in one translation unit by including the original IMU implementation together with the transport. This avoids changing the estimator source. OpenMP is explicitly included and limited to one estimator thread. The build uses portable C++17 `-O1` and records that optimization level.
- The IMU process has static zero-initialized storage so its first propagation clock is defined before the constructor; every supplied trajectory uses the declared common simulation clock. Neither ground-truth positions nor initial ground-truth orientations enter the estimator.

The calibration explicitly computes `T_I_L = inverse(T_B_I) @ T_B_L` and `T_C_L = inverse(T_B_C) @ T_B_L`; a supplied `T_C_L` must agree. Each original image timestamp defines an update. Points are assigned to the first image strictly after their original capture time, matching upstream synchronization; IMU samples are consumed only through that camera timestamp. Initial frames without prior returns and unconsumed LiDAR tail returns are recorded. Images are decoded from their original PNG bytes at their original resolution. The right image is verified as part of the input bundle but is not consumed by the estimator.

Native outputs retain every original camera frame. Missing tracking is represented by `tracked=false` and `T_W_I=null`; no pose is fabricated. Tracking requires at least one native LiDAR residual or visual patch plus a finite pose. This status is **not calibrated confidence**. Native visual-update attempts and frames containing visual measurements are reported separately, so LiDAR-only behavior cannot be mistaken for successful visual fusion.

## Three-arm experiment

`scripts/bench/native_livo_methods_campaign.py` runs these predeclared arms on the same retained sensor bundle:

1. Existing native FAST-LIO2.
2. Native FAST-LIVO2 with camera, LiDAR and IMU.
3. The same FAST-LIVO2 binary with visual updates disabled.

The third arm isolates the contribution of visual updates. FAST-LIVO2 also changes the LiDAR map estimator, so a two-arm FAST-LIO2 comparison alone cannot attribute a difference to the camera.

Fresh sensor sequences should cover textured geometry, reduced texture, a geometrically ambiguous corridor, partial LiDAR return loss, and combined degradation. All corruptions or acquisition changes belong in retained, hashed inputs before inference. Run fresh same-source smoke first, then the predeclared development cohort. Reserve independent scene geometries and seeds for confirmation before parameter tuning.

Measure post-initialization tracking fraction, initialization time, pose error and rotation error, native visual participation, failure/recovery intervals, and latency. Report conditional pose error alongside coverage. Native FAST-LIO2 emits LiDAR-tail poses while FAST-LIVO2 emits camera-time poses; the evaluator uses bracketed reference interpolation without extrapolation and a fixed-scale gravity/yaw/translation alignment. Reports disclose differing update clocks. Replay itself produces no closed-loop goals, collisions or falls.

The native compute timer covers IMU deskew, voxel processing, LiDAR estimation, map insertion and direct visual estimation. PNG decode time is separate. The same-frame decode-plus-compute percentile is computed from summed per-frame values, and full native process wall time includes startup and input/output. Sensor capture/render cost is excluded and must be added for deployment claims.

## Reproduce

Run dependency preparation inside the existing Ubuntu 22.04 container. The sysroot is private; no system package or shared Python environment is changed. Compile only in an allocated Slurm step. Keep preparation receipts and the pinned native runtime as inputs to a frozen campaign.

```bash
python scripts/native/livo_build.py prepare --work /path/to/private-build
python scripts/native/livo_build.py build --work /path/to/private-build \
  --output /path/to/native-runtime --jobs 2

PYTHONPATH=src python scripts/native/livo_replay.py \
  --manifest /path/to/replay/manifest.json --sequence sequence-id \
  --runtime /path/to/native-runtime --output /path/to/new-replay-output
```

Add `--disable-visual` for the native camera ablation. `scripts/bench/native_livo_methods_campaign.py --help` exposes the three-arm wrapper. Its frozen protocol uses schema `bhl-native-livo-methods-v1`, method names `fast_lio2`, `fast_livo2`, `fast_livo2_no_visual`, explicit sequence IDs and the existing native replay gate fields.

Transport tests cover original clocks and SI values, future-data exclusion, unique IMU consumption, calibration consistency, malformed inputs, unavailable poses, omitted native frames and accidental visual updates in the ablation. Passing transport tests alone is not evidence that native fusion improves accuracy.
