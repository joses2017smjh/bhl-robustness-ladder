# External EuRoC recorded-sensor replay: access blocked

**No native replay ran, and no real-sensor performance result is claimed.** On
2026-10-11 we attempted to obtain the official ETH EuRoC `V1_01_easy` sequence
as an external recorded-sensor supplement to the simulation experiments.
This does not complete validation of the user's physical hardware.

Before any native evaluation, the [protocol](../results/euroc-external-validation-20261011/protocol.json)
declared the first 60 seconds, at most 1,200 exactly paired camera frames,
native stereo versus native stereo–IMU, and the previously verified ORB runtime
SHA-256 `91ca4b6e4ea8f318998f400d4f7dfcd6fe8ae88b375399c79d5d633c39d71516`.
The proposed adapter must rectify the official stereo calibration, rotate the
camera-to-IMU transform into the rectified frame, preserve the original clock,
and pass only genuine causal IMU measurements. No adapter or successful
calibration validation is claimed here. Initialization, tracking, map changes,
and latency remain unmeasured; ATE also remains unmeasured.

The [official EuRoC page](https://projects.asl.ethz.ch/datasets/euroc-mav/)
describes stereo cameras, an IMU, calibration and ground truth, and directs
downloads to the [ETH Research Collection](https://www.research-collection.ethz.ch/items/bcaf173e-5dac-484b-bc37-faf97a594f1f)
under DOI `10.3929/ethz-b-000690084`. The collection lists the rights statement
**In Copyright – Non-Commercial Use Permitted**. Any later portfolio publication
must respect these terms; this repository contains only our protocol and access
receipts, with no dataset images or sensor CSV files.

| Access check | Observation |
| --- | --- |
| Official `vicon_room1.zip` size | 6,042,263,426 bytes (5.63 GiB) |
| Authorized local data budget | 4,294,967,296 bytes (4 GiB), download and extraction |
| Initial HTTP range probe | HTTP 206, bytes 0–127, ZIP signature present |
| Bounded ZIP-directory probes | HTTP 429 twice; no `Retry-After` header |
| Legacy official individual-sequence URL | Timed out after 25 seconds |
| Native camera frames evaluated | 0 |

The large archive was not downloaded. Requests stopped after the rate-limit
responses and the one legacy-location probe. A server ETag was observed but was
not treated as a verified digest of bytes we never downloaded. Zero raw dataset
bytes are retained. The original landing-page prefix stayed in local `/tmp`;
only its SHA-256 is recorded in the compact evidence.

The [access result](../results/euroc-external-validation-20261011/access-result.json),
[bounded access probe](../results/euroc-external-validation-20261011/access-probe.json),
and [checksum manifest](../results/euroc-external-validation-20261011/manifest.json)
preserve the exact blockers. A later run needs lawful access to the selected
sequence, a tested adapter and frozen calibration before executing this declared
window. Even a successful EuRoC replay would establish performance on an
external recording, not LiDAR validation, closed-loop locomotion, or a physical
robot deployment.
