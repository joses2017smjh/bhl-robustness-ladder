# EuRoC follow-up: software prepared, external recording still unavailable

The previously attempted EuRoC supplement now has an implemented input adapter
and native stereo/stereo–IMU runner. **61 focused software tests passed. No
EuRoC recording, actual native smoke, or scored replay ran.** The separately
completed TUM VI recording experiment remains the real-recording evidence.

The user's authorization to use the shared **2 TiB hard quota** removes the old
4 GiB storage restriction. A separate [operational amendment](../results/euroc-followup-20261012/operational-amendment.json)
sets a **12 GiB download-plus-extraction budget** on HPC share for this
supplement. The original first-60-seconds/at-most-1,200-pairs protocol, both
native arms and the pinned runtime stay unchanged. This storage budget is a
future acquisition bound; it does not claim downloaded or validated data.

## Access status

The [official ETH dataset page](https://projects.asl.ethz.ch/datasets/euroc-mav/)
continues to direct downloads to the ETH Research Collection. One fresh
request at **00:15 PDT on October 12** to the official bitstream returned
HTTP **429**, with no `Retry-After` header. One bounded check of the legacy
official individual-sequence URL at 00:23 PDT timed out after 25 seconds.
Requests stopped; no dataset bytes were downloaded and no alternate dataset
was substituted. [Current and legacy access receipts](../results/euroc-followup-20261012/).
The [October 11 failed attempt](EUROC_EXTERNAL_VALIDATION_2026-10-11.md) remains
preserved.

## Implemented and tested

[`euroc_native_validation.py`](../scripts/bench/euroc_native_validation.py)
reads the EuRoC pinhole/radial-tangential calibration and original 752×480
uint8 images. It rectifies both cameras and transforms the original
camera-to-body calibration into the rectified camera-to-IMU frame, including
a nonidentity body-to-IMU transform. It uses the recording's supplied IMU noise
parameters, original nanosecond clock and genuine causal gyro/acceleration
rows. It never reads a ground-truth pose for inference. The
[upstream ORB-SLAM3 EuRoC configuration](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/Examples_old/Stereo-Inertial/EuRoC.yaml)
documents the native camera-to-IMU convention; the adapter computes the
rectification adjustment explicitly rather than copying that example's numbers.

Preparation freezes every selected camera image, sensor calibration and original
camera/IMU CSV checksum. Native execution rejects missing manifest entries,
changed files, absent inputs and a changed selection. It runs both existing
native modes with the pinned runtime. Every frame is retained, including failed
initialization and tracking. Reports include camera coverage, IMU initialization,
coordinate-epoch changes (including reused map IDs), and offline computation and
pipeline latency. This is not paced real-time operation.

A full replay requires a completed **40-pair-per-arm native execution smoke**
from the same source, protocol, selected recording and runtime. Its frame traces
and settings must still match their checksums. The smoke is separate from the
scored cohort; initialization failure remains an observable outcome rather than
a reason to tune native parameters. Output directories cannot overwrite earlier
evidence.

The [test receipt](../results/euroc-followup-20261012/test-receipt.json) records
61 passes in 6.74 seconds on two allocated CPUs. Tests independently project
synthetic 3D points through distorted cameras to check equal rectified rows and
positive disparity; verify nonidentity IMU extrinsics and original-clock
causality; reject wrong camera modes, malformed manifests and changed smoke
traces; and exercise preparation → smoke → full-run orchestration with a
**mocked native client**. Existing native IPC and reset regressions also passed.
These synthetic tests establish software behavior, not EuRoC accuracy or actual
native execution readiness.

## Exact remaining handoff

Once official access is available, acquire the unchanged `V1_01_easy` recording
within the 12 GiB combined archive/extraction bound, inspect its actual archive
layout, safely extract the selected sequence, verify its origin and byte hashes,
and freeze that intake receipt. Current server responses prevent this step.
Respect the documented noncommercial rights; publish derived metrics and
checksums only, without original imagery or sensor CSV files.

Run the following inside the existing Ubuntu 22.04 container on allocated CPU
resources, with a private home/cache and a frozen, read-only source snapshot.
Use Python 3.11 and the already pinned `orb-stereo-inertial-runtime-v1-20261010.tar.gz`
archive (SHA256 `91ca4b6e4ea8f318998f400d4f7dfcd6fe8ae88b375399c79d5d633c39d71516`).
The dataset directory must contain `mav0/cam0`, `mav0/cam1`, and `mav0/imu0`.
The original scientific protocol is retained at
`results/euroc-external-validation-20261011/protocol.json`.

```bash
python scripts/bench/euroc_native_validation.py prepare \
  --dataset "$EUROC_DATASET" --protocol "$EUROC_PROTOCOL" \
  --output "$EUROC_WORK/selection.json"

python scripts/bench/euroc_native_validation.py run --stage smoke \
  --dataset "$EUROC_DATASET" --protocol "$EUROC_PROTOCOL" \
  --selection "$EUROC_WORK/selection.json" --runtime-archive "$EUROC_RUNTIME" \
  --output "$EUROC_WORK/native-smoke"

python scripts/bench/euroc_native_validation.py run --stage full \
  --dataset "$EUROC_DATASET" --protocol "$EUROC_PROTOCOL" \
  --selection "$EUROC_WORK/selection.json" --runtime-archive "$EUROC_RUNTIME" \
  --smoke-receipt "$EUROC_WORK/native-smoke/result.json" \
  --output "$EUROC_WORK/native-full"
```

Each command stops on missing input. Do not submit the full replay before a
fresh successful native execution smoke, freeze and record the scored launch
in the job ledger, and independently verify resulting traces before publishing.
The replay runner intentionally leaves **ATE unmeasured** until the EuRoC
reference frame, timestamp association and separate evaluator are independently
verified. No physical-rig calibration, LiDAR validation, locomotion, navigation
or hardware claim follows from this supplement.
