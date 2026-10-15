# EuRoC V1_01_easy pose accuracy (follow-up 2, 2026-10-14)

**Recorded external data (EuRoC), not this project's robot.** One sequence, first 60 s (1,200 stereo pairs, about 19 m of
reference path), one run per arm, causal per-frame native ORB-SLAM3 outputs.

Evaluator: `scripts/bench/euroc_native_evaluate.py` (19 tests in `tests/test_euroc_native_evaluate.py`). The protocol
(segmentation rule, alignment, association) was frozen before any score was computed. Full outputs, outside git:
`/nfs/hpc/share/sanchej7/humanoid-research-20261011/euroc-accuracy-20261014/` (`REPORT.md`, `results.json`,
`per_frame_errors.csv`, `protocol.json`, evo cross-check).

## References

| Reference | Source | Pinned git blob |
|---|---|---|
| Original | EuRoC `state_groundtruth_estimate0` | `98c4a021…` |
| Recomputed | OpenVINS `ov_data/euroc_mav/V1_01_easy.csv` (vicon2gt, GPL-3.0; read in place, not copied) | `8063da27…` |

The two references differ by a constant body-frame offset: 5.5 deg and 4.3 cm. OpenVINS documents the original
orientation as inaccurate. That offset sets a ~5.5 deg floor on rotation error against the original reference and
explains the whole score difference between the two references.

## Results

Recomputed reference first; the original reference follows in brackets. Scale fixed at 1.

**Stereo**
- Coverage: 1200/1200 frames tracked, one continuous segment.
- SE3 alignment: ATE RMSE 0.020 m (0.036 m); mean rotation error 0.44 deg (5.55 deg).
- Drift over 10 s intervals: 0.78% (4.47%) of distance travelled.

**Stereo-IMU**
- Coverage: 1097/1200 frames tracked; 1041 IMU-initialized, from 7.95 s.
- Two causal pose jumps with the map id unchanged: 0.124 m at 8.05 s and 0.825 m at 20.60 s. The predeclared
  discontinuity rule splits the track there.
- Between the jumps (SE3 / 4-DoF alignment):
  - 8.05-20.55 s: ATE RMSE 0.011 m.
  - 20.60-59.95 s: ATE RMSE 0.019 m.
  - Mean rotation error 0.60-0.75 deg.
- Stitched across the jumps: ATE RMSE 0.28 m (SE3).

**Comparing the two arms**
- On identical timestamps between the jumps, the arms differ by at most 4.6 mm in ATE.
- The sign of that difference flips with the reference, so it does not rank them.
- Over the whole initialized window, the stereo-IMU jumps dominate.
- Stereo-IMU is smoother frame to frame: one-frame step error median 0.8 mm vs 2.7 mm.

## Checks

- **IMU frame.** The declared IMU-frame conversion T_W_I = T_W_Crect·inv(T_I_Crect) leaves 0.24 deg of residual against
  the recomputed reference. Wrong conversions leave 85-178 deg.
- **Vicon.** Vicon marker poses composed with inv(T_BS) land within 2.9-3.2 deg and 1.5-2.4 cm of both references, so
  both are IMU-frame poses.
- **evo.** evo 1.34 reproduces APE/RPE to about 1e-8.

## What this settles and what it does not

**Settled**
- Tracking coverage alone did hide an accuracy problem: the stereo-IMU arm's raw output jumps 0.8 m.
- Its between-jump accuracy matches stereo.

**Not settled**
- Behaviour on other sequences, or over the full 145 s sequence.
- Run-to-run spread.
- The cause of the jumps. They fit ORB-SLAM3's staged inertial refinement after IMU initialization, but the native log
  that would confirm it was not retained.
