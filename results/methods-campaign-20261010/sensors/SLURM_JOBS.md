# Stereo, uncertainty mapping and native fusion — predeclared 2026-10-10

One 60-frame same-source GPU smoke (12s including native initialization and actual photometric participation). Only after execution PASS, 18 serial development jobs: three scene families × two new geometry seeds (610000/610001) × nominal, uniform texture, 50% LiDAR dropout. Every full cell is 150 calibrated stereo/timed-LiDAR/200Hz-IMU frames over 30s. Camera exposure is at the LiDAR tail; original timestamps retained. All methods share each exact raw capture.

Each job runs SGBM and pinned C-Fast-FoundationStereo, independent per-sensor ground maps and equal-mean versus uncertainty-gated fusion, then actual FAST-LIO2, FAST-LIVO2 and same-binary FAST-LIVO2 with vision disabled. Outcome: 2,700 paired frames, 54 native estimator cells if complete; these are planned counts. Independent evaluator depth, obstacle segmentation and downward ground queries are excluded from inference.

All variants remain development data. Group statistics by the six geometry variants; corruptions and frames are not independent sample units. No physical, independent-scene-family, navigation or traversal claim. One GPU/4CPUs/32GB, 1h smoke or2h/full cell, no requeue, one active full sensor job. Raw outputs archived and freshly downloaded/hash-verified before advancing.

Source tests/native runtime smoke pass; fresh complete pipeline not yet submitted.

Fresh pipeline smoke21757535 failed before capture: the guard required SLURM_JOB_ID, which the clean scientific container intentionally strips. Fixed the guard to also accept the frozen-launcher H34_PROTOCOL/H34_OUTPUT_DIR context while retaining EGL and real CUDA checks. Failed result and raw log published; no sensor frame or score was produced. A new source freeze and fresh smoke are required.

## Fresh source smoke and serial development queue

V2 archive `207a5129cd3df243f48f6fddccd5f74030ba6a163c9aeb151a119b9d014bf5fe`, protocol `2d648f2ebd1632accfca7438347562dead448d0c759b073d1fa6a80ef62be9ac`. Fresh GPU smoke **21757575 PASS**: 60 original stereo pairs, two depth methods, seven map/fusion arms and three native estimators. Real CUDA kernel execution is retained in the result. FAST-LIVO2 consumed camera measurements in 57 frames; all three native smoke readiness cells qualified. This is smoke evidence, not a full-cohort performance result.

The durable 18-cell controller is **21757623**. It reserves 352 MiB for each compressed output (60-frame smoke raw archive was 142,011,134 bytes; full cells have 150 frames) and preserves at least 128 MiB global home headroom before submission. It publishes and freshly downloads each raw archive before advancing. Full output is additionally bounded by the original frozen runner. The strict collector requires all 18 cells and 54 native results; 16 adversarial collector tests passed.
