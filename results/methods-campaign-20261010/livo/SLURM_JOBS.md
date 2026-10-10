# Native FAST-LIVO2 ledger — October 10, 2026

All stages were declared in this ledger before their launch. Build and smoke stages do not count as scored scientific replicates.

| Stage | Allocation / step | Status | Scope |
|---|---|---|---|
| Native build v1 | 21756762.6 | Compiled; link failed | 2 CPUs; C++17 O1. Native estimator compilation succeeded; private GDAL lookup was missing. `build-intake-v1.json` and `build-v1.log` retain the attempt. |
| Native relink v2 | 21756762.10 | PASS | Added existing private `usr/lib` to library lookup. Estimator source and parameters unchanged. Source and retained object hashes: `build-intake-v2.json`. |
| Transport QA | 21756762.11 | PASS 10/10 | Causal transport and native-output checks; 0.23 s pytest execution. |
| Native smoke v2 | 21756762.17 | PASS | First 35 original images from `textured_boxes_native`; actual visual and visual-disabled native arms. Require 35 outputs and at least one tracked frame in each; visual arm also requires an actual visual-measurement frame. |
| Retained-capture diagnostic v2 | 21756762.18 | COMPLETED; 8/9 qualified | Three original scenes × three native methods; 9 cells, 1,350 native frames. Reused development data, not confirmation. Original gates fixed. |

Build sources were copied into `/tmp/bhl-livo-source-v1-20261010` and the library-path-only revision `/tmp/bhl-livo-source-v2-20261010` before their steps. Dependencies were prepared privately inside the existing Ubuntu SIF and checked by file/package hashes. The final file-only runtime archive is described by `runtime-archive-v2.json`; native payload bytes match the first packaging attempt.

The replay snapshot `/tmp/bhl-livo-replay-source-v2-20261010` is separately frozen. `diagnostic-intake-v2.json`, `diagnostic-protocol-v1.json` and `diagnostic-driver-v2.py` retain source and original sensor/runtime hashes, smoke requirements and fixed replay gates. A fresh same-source smoke must pass before the diagnostic. No new-scene improvement or hardware outcome follows from reusing this older capture.

Replay-driver v1 was not submitted. Preflight identified that strict archive extraction restores regular files without execute permission; driver v2 explicitly restores the baseline executable permission before invocation. Native source bytes, inputs and gates are unchanged.

Measured evidence is retained under `measured/`. Smoke completed in 12 s wall time; the nine-cell diagnostic completed in 61 s. FAST-LIO2 qualified 3/3 scenes, full FAST-LIVO2 2/3, and its no-visual ablation 3/3. The negative thin-posts result is retained unchanged.
