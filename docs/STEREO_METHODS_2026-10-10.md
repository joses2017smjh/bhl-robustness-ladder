# Native stereo diagnostics and verified map recovery

This implements the next experiment after the October 9 negative stereo results. It does not change their measurements or relax ORB-SLAM3's initialization gate. The methods below require a newly frozen runtime and new smoke before scored development runs.

## Native feature and initialization diagnostics

The diagnostic wrapper reports the actual current ORB-SLAM3 frame's extracted left/right features, positive stereo-depth matches, total feature count and initialization condition. The original pinned tracking/local-mapping/loop-closure library remains byte-identical. A const inline accessor added only to the wrapper's temporary `System.h` exposes already-public current-frame fields after synchronous `TrackStereo` returns. The initialization threshold remains the upstream `mCurrentFrame.N > 500`.

`FEATURE_COUNT_NOT_ABOVE_500` is emitted only when the native state is `NOT_INITIALIZED` and the actual native feature count is at most 500. The other possible labels are `OTHER_NATIVE_INITIALIZATION_CONDITION` and `NOT_IN_INITIALIZATION`. These describe observed conditions, not a fabricated causal explanation. The depth-match count includes finite positive native depths; it does not claim to count all attempted descriptor correspondences or epipolar rejections.

Build with `scripts/native/orb_diagnostics_build.py` inside the original Ubuntu 22.04 SIF. It verifies the original runtime archive and every member, downloads the exact original development-header package versions and verifies their SHA-256 values, and compiles only `orb_runner.cc`. It records source, header, compiler command, package and unchanged native-library hashes. The optional package cache reuses only verified original `.deb` bytes. The ordinary runtime build does not define `BHL_ORB_DIAGNOSTICS` and retains its previous interface.

After replay, summarize genuine `native_frames.jsonl` files using:

```bash
python scripts/bench/stereo_diagnostics.py \
  --frames /path/to/scene-a/native_frames.jsonl /path/to/scene-b/native_frames.jsonl \
  --output /new/path/native-feature-diagnostics.json
```

The summarizer rejects streams missing actual native telemetry and records every source checksum. Existing image datasets may be reused for diagnosis; they cannot become untouched confirmation sets after inspecting these diagnostics.

## LIO-verified stereo map recovery

`src/bhl_robust/research/stereo_recovery.py` implements a strict inference-only SE(3) continuity check. It takes only native stereo poses/map IDs and native FAST-LIO2 poses, match counts, reset states and original timestamps. There is no simulator pose, evaluator, terrain geometry or goal input. This is explicitly a hybrid method; it is not stereo-only recovery or a reproduction of a tightly coupled estimator.

1. Anchor the continuous native LIO coordinate frame to the initial valid stereo coordinate frame using a synchronized estimated-pose pair.
2. Stop on a new stereo map ID or a same-ID pose jump inconsistent with LIO. Preserve raw native stereo outputs.
3. Fit a rigid, unit-scale map transform from three distinct estimated-pose pairs, spaced at least 150 ms apart.
4. Freeze that candidate and validate it against three strictly later observations. Calibration and validation windows are temporally disjoint, not claimed statistically independent.
5. Resume stereo-pose control only if all calibration and validation residuals are at most 6 cm and 0.10 rad. Transform stereo poses into the original frame; never substitute the LIO pose as a fallback.
6. Continue a 15 cm / 0.20 rad continuity watchdog. Stop if the LIO reference loses tracking, has fewer than 30 effective matches, is older than 10 ms relative to the camera pair, comes from the future, or resets. A reference reset permanently latches a stop because the original coordinate anchor is no longer trusted.

These thresholds are predeclared engineering gates, not calibrated probabilities of correctness. LIO can itself drift without resetting; shared error can evade this consistency check. Fresh physical data and external outcome evaluation remain necessary.

## Executable matched campaign

`scripts/bench/stereo_methods_campaign.py` compares:

| Arm | Pose control | Native LIO use |
|---|---|---|
| `permanent_stop` | Original stereo pose; permanent stop after map change | Shadow only; computation charged |
| `verified_lio_recovery` | Stereo pose transformed after verified recovery | Registration and validation; no pose fallback |

Both arms retain the original raw LiDAR obstacle brake, body-attached 640×480 stereo, 0.12 m baseline, 5 Hz captures, full body roll/pitch, 1200-feature extraction recipe, frozen learned 12-DoF gait and prescribed external routes. Both run the native LIO and ORB clients sequentially on the same causal sensor packet and charge measured native/client/recovery computation plus capture/render/PNG/hash costs before making a response available to control. The control loop retains the previous pose until arrival and applies its existing staleness stop. This makes the comparator's cost scope explicit; it differs from the historical stereo-only-compute baseline.

```bash
python scripts/bench/stereo_methods_campaign.py \
  --phase smoke \
  --upstream /path/to/Berkeley-Humanoid-Lite \
  --deploy /path/to/deploy.yaml --checkpoint /path/to/policy.pt \
  --orb-runtime-archive /path/to/diagnostic-runtime.tar.gz \
  --orb-runtime-sha256 EXACT_ARCHIVE_SHA256 \
  --lio-runtime-archive /path/to/native-lio-runtime.tar.gz \
  --lio-runtime-sha256 EXACT_ARCHIVE_SHA256 \
  --output /new/output/directory
```

Use the repository's frozen Slurm intake for the actual invocation. Smoke runs two fresh straight-route reset seeds, 470000–470001, for both arms at 8 s each (4 episodes). Development runs straight/dogleg/occluders × fresh reset seeds 480000–480002 × both arms at 40 s (18 episodes). These are repeated resets of one frozen policy, not independently trained agents. Report all episodes, native telemetry, recovery decisions/transforms/timestamps, clean goals, falls, contacts and full charged delivery latency. A development improvement requires more clean goals with no increase in falls or contacts; it still requires held-out confirmation.

The development cohort is submitted as nine sequential paired jobs using `--case straight --seed 480000` (and the other declared pairs), with both arms in each job. Complete original PNGs remain in each job's raw archive. Root publication verifies the remote archive digest and a fresh download before retiring its local duplicate; no credential enters the physics job. Each pair reports `DEVELOPMENT_PAIR_NO_GLOBAL_VERDICT`, and `stereo_methods_collect.py` requires all 18 outcomes, rejects duplicate/missing pairs, and verifies matching actor/runtime hashes. It reports the paired goal-rate difference and a descriptive 95% percentile interval from all 27 resamples of the three whole reset-seed groups, keeping routes together. Three groups and one actor are insufficient for a population or hardware claim.

No stereo–IMU initialization improvement is claimed. Native ORB-SLAM3 stereo–IMU also requires the visual feature gate and has further inertial initialization conditions. This campaign specifically implements native stereo diagnostics and independently estimated LIO map continuity.

## Research basis and scope

- [ORB-SLAM3 paper](https://arxiv.org/abs/2007.11898) supplies the native visual/inertial multi-map framework; [pinned initialization implementation](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/src/Tracking.cc) supplies the actual feature gate.
- [FAST-LIO2 paper](https://arxiv.org/abs/2107.06829) supplies the separate native LiDAR–IMU reference estimator.
- The six-observation SE(3) registration/check and conservative resume policy are this project's experimental method. Neither paper is cited as evidence that this specific recovery gate already improves navigation.

Implementation validation covers map changes, same-ID pose jumps, disjoint calibration/validation, rejected inconsistent evidence, minimum observation spacing, future/stale/weak/missing reference evidence, latched reference resets, missing stereo pose, rigid unit scale and malformed telemetry. Synthetic unit fixtures are validation of software contracts; only actual simulator/native runs count as experiment results.

## Measured native startup diagnosis

The read-only diagnostic replay completed all450 original frames in allocation step21756762.20. In textured boxes and thin posts, the maximum extracted feature counts were463 and427, respectively: every frame remained below the unchanged native initialization requirement of more than500 features. Both scenes tracked0/150. Ramp/step reached541 features and tracked122/150, first tracking at5.6s. This directly supports feature starvation as the startup blocker in those two consumed scenes; it is not a new improvement experiment. Original native estimator-library bytes and the threshold were unchanged. [Raw diagnostic evidence](../results/methods-campaign-20261010/stereo/diagnostic-replay/measured/diagnostics.json).
