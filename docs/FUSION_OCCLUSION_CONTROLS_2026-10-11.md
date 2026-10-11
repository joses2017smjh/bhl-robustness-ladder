# Controlled geometric occlusion and stereo–LiDAR fusion

Status: implementation and preregistration prepared; no results from this experiment are claimed until the same-source GPU smoke and independent collection complete.

The previous local-SAD consistency experiment reduced false-free predictions while losing coverage and obstacle recall. This follow-up tests whether that tradeoff changes when the sensors actually see different surfaces. It creates foreground geometry in MuJoCo, renders both physical camera viewpoints, and casts sparse LiDAR rays from the calibrated midpoint. It changes no image masks to manufacture occlusion.

## Frozen design

The authoritative design is [protocols/fusion_occlusion_20261011.json](../protocols/fusion_occlusion_20261011.json). `freeze` binds that design, all scientific Python sources, and the original pretrained model by SHA256 before execution.

| Cohort | Geometry groups | Conditions | Viewpoints | Frames |
|---|---:|---:|---:|---:|
| Smoke | 2 layouts × seed 740000 | Clear, partial, heavy | 2 | 12 |
| Development | 2 layouts × seeds 741000–741001 | Same three | 12 | 144 |
| Held out | 2 layouts × seeds 742000–742001 | Same three | 12 | 144 |

Layouts contain a background panel or staggered narrow posts. A real foreground box has half-width 0, 0.08 or 0.24 m. Each geometry seed fixes target displacement and texture; all three conditions share them. Viewpoints are predetermined linear offsets from x=0 to 0.2 m and y=−0.12 to 0.12 m. The rig is stationary for each acquisition. LiDAR has 16 rings × 128 azimuths and instantaneous acquisition. This removes scan-motion and deskew confounds; it does not test a walking robot.

Original images are 320×240. FoundationStereo pads to 320×256, and measured model latency includes this padding. Camera baseline is 0.12 m. The experiment uses the already pinned official C-Fast-FoundationStereo checkpoint (`2c79bbb274dc0aef687a8732b987077e9e3aa26ff9dd653141cda48c57dad5d3`); actual CUDA execution must be proven by its existing operator-profile check.

Nine arms receive the same RGB pair, raw LiDAR and calibration:

1. Sparse LiDAR alone.
2. SGBM alone, with its existing bidirectional consistency/uniqueness validity.
3. FoundationStereo alone, with its exported left-disparity validity.
4. Simple projected fusion for each stereo engine.
5. Agreement-gated fusion for each engine.
6. Local-SAD-cost plus agreement-gated fusion for each engine.

Agreement rejects dual-support differences exceeding max(0.15 m, 5% of nearer depth). Local cost reuses the existing 5×5 image SAD margin versus ±2 disparity pixels, minimum margin 0.10 and maximum mean SAD 20. It caps stereo weight at one half on agreeing support. These are engineering scores, not calibrated probabilities. Single-left-disparity FoundationStereo has no exported right-disparity consistency mask; that distinction remains visible in native-support coverage.

## Independent visibility and evaluation

All inference arms finish before evaluator depth or segmentation is rendered. The evaluator projects each left-view geometry point into independently rendered right-camera and midpoint-camera depth buffers. A nearer surface by more than 0.03 m marks the point occluded; out-of-view or invalid evaluator support remains separate. Midpoint-camera visibility describes the continuous LiDAR viewpoint; sparse ray sampling is an additional limitation. Nearest-pixel projection and depth-buffer discretization can affect boundary labels.

Retain every frame's inputs, predictions, support, cost scores, conflicts, evaluator depths and masks. Independent collection checks exact file inventories, hashes, frame identities, checkpoint/CUDA provenance and recomputes visibility plus all metrics. A smoke must exercise geometric occlusion in every nonclear cell. No development or held-out cell can start without a passing collection for all six same-source smoke cells.

Report native and all-arm-common support MAE with pixel denominators. All-arm common support can be tiny because of sparse LiDAR; report pairwise-common support for every fusion arm versus its stereo baseline, sparse LiDAR, simple fusion and agreement gate. Report visible/occluded strata separately. Keep coverage, missed obstacle pixels, unknown obstacle pixels, false-free obstacle pixels and obstacle-pixel recall alongside every accuracy comparison. “Missed” means no prediction within ±0.15 m of a true visible obstacle at ≤3 m. These are pixel proxies, not object detection or collision rates.

Timing measures real shared processing components and required PNG/NPZ decoding, summed according to each arm's dependencies. Report p95 per geometry cell plus cold model initialization. One same-input pipeline warm-up is excluded. Capture, output serialization and evaluator work are separately excluded. These are component-composed timings, not independent live-sensor deadline measurements. The 12 views within each geometry group are correlated; never count them as independent statistical trials. The two layouts share their RNG seed values and random draws. The held-out cohort therefore contains four geometry groups but only two independent seed clusters; results are descriptive and conditional on these fixed layout families. No significance or bootstrap interval is claimed.

## Execution and storage

The frozen serial entrypoint runs all six smoke cells on an allocated EGL GPU, independently collects them, and proceeds to the 24 scored cells only after that gate passes. A negative smoke writes a retained negative result with zero scored cells. The H34 launcher isolates the source, inputs, runtime and caches. A separate host publisher verifies completion/source/artifact pins and fresh-download SHA256 before publishing raw evidence. Output must be under `/nfs/hpc/share/sanchej7/humanoid-research-20261011` or allocated node-local `/tmp`, never home. Preserve ≥8 GiB of share free space. Each cell is capped at 96 MiB and the retained cohort at 2 GiB. Stop on a failed smoke or storage bound; preserve the negative receipt. No scientific parameter changes or threshold tuning are permitted after outcomes.

```bash
python scripts/bench/fusion_occlusion_campaign.py freeze \
  --output /nfs/hpc/share/sanchej7/humanoid-research-20261011/fusion-occlusion/protocol.json \
  --model /nfs/hpc/share/sanchej7/home-overflow/2026-10-11/humanoid-sensors-20261008/inputs/c_fast_foundationstereo_bp2_dynbatch_dynhw.onnx
```

`run --protocol ... --cell <exact frozen ID> --output <new directory>` produces one cell. `collect --protocol ... --phase smoke|dev|test --directories ... --output ...` validates the complete declared cohort. Scored cells also require `--smoke-receipt ...`. This is preparation, not evidence that GPU execution or scientific qualification has passed.

## Research basis and limits

[Yao, Ishikawa and Oishi (2025)](https://arxiv.org/abs/2504.05148) motivates testing cross-view consistency and sparse LiDAR support: their method combines discrete matching costs, LiDAR semidensification and consistency checks. This experiment uses a simpler local-SAD score and nearest-pixel sparse projection. It does not reproduce their aggregated SGM/discrete-cost machinery or semidensification, and their published KITTI numbers are not our results.

[Fast-FoundationStereo's official repository](https://github.com/NVlabs/Fast-FoundationStereo) provides the pretrained stereo family. Keeping its original checkpoint fixed tests a learned stereo baseline alongside SGBM under identical geometry and calibration. This remains a simulated sensor experiment with ideal pinhole images and ideal ray LiDAR. It provides no hardware, locomotion, SLAM or deployment guarantee.
