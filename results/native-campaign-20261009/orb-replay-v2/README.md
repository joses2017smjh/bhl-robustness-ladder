# Actual native ORB-SLAM3 replay

**Completed; overall scientific verdict NEGATIVE.** Real upstream ORB-SLAM3 processed all **450/450** original stereo pairs across three preassigned simulation scene groups. One heldout scene qualified; the two other scenes never initialized. No parameters or gates were tuned after observing outputs.

| Scene / split | Actual tracking | Tracking after first 5 s | ATE RMSE | Native compute p95 | Gate |
|---|---:|---:|---:|---:|---|
| Textured boxes / development | 0/150 | 0% | Unscorable | 45.11 ms | NEGATIVE |
| Thin posts / validation | 0/150 | 0% | Unscorable | 27.02 ms | NEGATIVE |
| Ramp and step / heldout test | 122/150 | 97.6% | 4.39 mm | 48.60 ms | PASS |

The two uninitialized scenes returned native state1 (NOT_INITIALIZED) on every frame, with explicit null poses. The heldout scene returned 28 uninitialized frames followed by 122 actual OK states, in one native map with no reset. Its translation RPE RMSE was 2.84 mm over consecutive 0.2 s associated tracked intervals; rotation ATE p95 was 0.640 degrees. ATE uses evaluator-only rigid alignment with scale fixed to 1. Compute includes native frame decode/tracking and excludes capture, process startup, validation/export and replay waits. These measurements cover a bounded simulation replay, with zero closed-loop/hardware claims or population superiority claim.

Actual full execution: **Slurm step 21739893.8**; actual fresh smoke step 21739893.7. Pending batch 21744211 was cancelled before step execution, and no duplicate batch full was submitted. Source archive 499ede4237b94cba3008da2cb028243c35ac8019cb5fd284ba69c01c21cbf93a; unchanged genuine native runtime 0c9659c6214bafbb2e8c9e051358628b2eacef2cede23b0b37cedb6df1901704; original raw output archive f7e5659742e737445c949b255a5abb1a45bbe4b4fac5538fbe8d7c25e15b01e4.

See [verified collection](collection/report.md), [scientific summary](scientific-summary.json), original per-frame responses/native logs in `native-orb-replay-20261009-v2-21739893/`, and original step/environment/completion checksums. Startup-only schema repair and failed original attempts are retained in adjacent result folders.
