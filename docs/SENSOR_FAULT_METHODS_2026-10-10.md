# Paired sensor faults and cost-consistency fusion

Two additional executable experiments extend the [fresh sensor campaign](SENSOR_METHODS_2026-10-10.md). They use retained raw observations from its six nominal development geometry groups. They do not edit or rescore the original October 9 experiments. Implementation and a declared matrix are not measured improvements; the full cohorts remain unrun until their input-bound protocols and fresh smokes pass.

## Native estimation stress

[`sensor_fault_campaign.py`](../scripts/bench/sensor_fault_campaign.py) compares actual FAST-LIO2, FAST-LIVO2, and the same FAST-LIVO2 binary with visual updates disabled. The three estimators use identical derived sensor recordings within each condition.

| Condition | Exact perturbation |
|---|---|
| Nominal | Original measurements and calibration |
| Delayed IMU clock | Reported IMU timestamps shifted forward by 20 ms or 80 ms |
| LiDAR calibration error | Declared LiDAR extrinsic rotated 2° about body z and translated 2 cm along body x; measured points unchanged |
| Gyroscope bias | Add 0.05 rad/s about IMU z beginning at original acquisition time 5 s |
| Accelerometer bias | Add 0.3 m/s² along IMU x beginning at original acquisition time 5 s |
| Return loss | Keep every second or fourth original return and retain both scan endpoints |

The endpoint rule means the retained count is approximately one half or one quarter, not exactly that fraction. It preserves the scan beginning and tail, so all methods and faults retain matching scoring times. Point coordinates and retained per-point offsets are unchanged; actual native IMU deskew still runs. No estimated trajectory is synthesized in Python.

Every arm selects the same original camera frames `[1:-1]`, giving 148 scored outputs from a 150-frame recording. All original high-rate IMU samples remain available to the transport. Trimming the camera boundaries permits the positive IMU clock shifts without extrapolating or inventing edge samples. A shifted IMU sample's true acquisition time is never later than its reported availability time. Negative clock shifts and future measurements are not introduced. Constant biases begin after the stationary initialization interval to avoid a test that merely lets initialization absorb the bias.

Only inference sensor files are opened when deriving an arm. The erroneous calibration is supplied to estimation; independent scoring retains the original calibration and truth. Native inference for all three arms finishes before the evaluator opens truth. Raw native transports, frames, runtimes, source hashes, input identities, and derived-manifest receipts are retained.

The full matrix is **six geometry groups × eight conditions × three native methods = 144 estimator cells**. One executable shard is a geometry/condition pair with three methods. The collector requires all 48 shards, reconstructs native scores from raw outputs, and preserves absent or failed cells as incomplete. Comparisons report paired group differences in tracking and conditional ATE, plus qualification and native latency. They do not count individual frames as independent experiments. The old readiness thresholds are frozen controls; they are not a calibrated health probability, and this replay measures no goals, collisions, falls, or closed-loop recovery.

## Dense stereo and sparse LiDAR consistency

The existing SGBM baseline already performs independent left/right OpenCV matching, a one-pixel left–right check, local texture filtering, uniqueness filtering, and speckle rejection. Its output is binary match support. Adding another left–right check would not establish a new confidence method.

[`stereo_consistency.py`](../src/bhl_robust/research/stereo_consistency.py) instead adds a disclosed **local photometric-cost margin** to those accepted SGBM matches. It compares a 5 × 5 mean absolute grayscale residual at the SGBM correspondence against correspondences displaced by ±2 pixels. A margin of at least 0.10 and mean residual at most 20 are required. Complete valid, in-bounds patches are required; missing texture, occlusion holes and borders cannot acquire confidence from zero padding. The continuous margin/residual score is an engineering weight, not a probability or learned uncertainty.

The fusion experiment compares five arms:

1. Original SGBM with its existing binary support.
2. SGBM filtered by the local cost margin.
3. Sparse LiDAR projected into the left camera, retaining the nearest return per pixel.
4. Equal-mean fusion on common support, preserving each sensor's single support.
5. Cost-weighted fusion with agreement within `max(0.15 m, 5% of nearer depth)`; conflicting dual support becomes unknown.

The weighted arm gives stereo at most half the weight on agreeing dual support. It retains confident stereo-only pixels and LiDAR-only pixels, without hole filling. The four paired conditions are nominal, the declared LiDAR calibration error, and the two return-loss conditions. The full matrix is **six groups × four conditions × five arms**, executed as 24 shards. All arms share original input pixels, selected frames and independent evaluator labels.

[`stereo_consistency_campaign.py`](../scripts/bench/stereo_consistency_campaign.py) retains actual depth, support, confidence and conflict arrays. Its collector independently recomputes depth and obstacle metrics from those arrays and original truth. Reported metrics include depth MAE/RMSE, coverage, common-support error, near-obstacle pixel recall, false-free obstacle pixels and unresolved obstacle pixels. Latency includes input loading, both SGBM directions, local costs, projection and all fusion arms, excluding sensor capture and evaluator/output work. Frame measurements are aggregated within a geometry group.

This uses local image SAD costs, **not the aggregated SGM cost volume**. It is not a reproduction of [Yao et al.'s LiDAR semidensification and three-view method](https://arxiv.org/abs/2504.05148). Projected scans retain raw point times but this mapping arm does not motion-deskew the instantaneous projection; its limitation remains explicit. Acquired low-texture recordings are part of the separate fresh sensor comparison, not secretly added to this six-group nominal-input protocol.

## Frozen execution and collection

The freezer verifies and binds source files, original capture manifests, both native binaries and runtime receipts. Full freezing requires exactly the six predeclared nominal development groups with 150 original frames each. Smoke freezing accepts one group with 35–150 frames and explicitly records `phase: smoke`; smoke results cannot satisfy a full collection. Source or input changes require a new protocol and smoke.

Run the commands on allocated compute, using new output/work directories. Native replay needs no rendering GPU; original capture already occurred on allocated EGL GPUs.

```bash
PYTHONPATH=src python scripts/bench/sensor_fault_campaign.py freeze \
  --manifests /retained/fresh/replay/manifest.json \
  --lio-runtime /verified/fastlio/runtime \
  --livo-runtime /verified/fastlivo/runtime \
  --smoke --output /durable/fault-smoke-protocol.json

PYTHONPATH=src python scripts/bench/sensor_fault_campaign.py run \
  --protocol /durable/fault-smoke-protocol.json \
  --group textured_boxes-g610000 --fault imu_clock_plus80ms \
  --output /durable/fault-smoke-result --work /tmp/new-fault-smoke-input

PYTHONPATH=src python scripts/bench/stereo_consistency_campaign.py run \
  --protocol /durable/fault-smoke-protocol.json \
  --group textured_boxes-g610000 --fault nominal \
  --output /durable/consistency-smoke-result --work /tmp/new-consistency-smoke-input
```

For the full cohort, freeze all six fresh nominal manifests together without `--smoke`, then run the declared group/condition shards. Both scripts provide `collect --protocol ... --directories ... --output ...`. Missing evidence produces `INCOMPLETE`; successful execution does not imply robustness improvement. Full native fault stress and full cost-consistency fusion each require their own complete collector verdict before a resume comparison can be stated.

## Queued continuation

CPU job **21757767** is queued after the fresh sensor controller21757623. It downloads the six nominal raw recordings only after verifying their exact source, completion and publication receipts. It freezes their original manifest hashes using the already published method source and fixed conditions, runs all eight native fault smokes and four fusion smokes, then releases the48 native and24 fusion shards only after both smoke collectors pass. No runtime tuning or automatic retry is allowed.

The [ledger, pinned plan and source/runtime publication receipts](../results/methods-campaign-20261010/sensor-followup/SLURM_JOBS.md) record31 passing targeted tests and a preliminary actual native/fusion smoke. Full results remain pending. The driver publishes and independently downloads every raw output shard before reporting either full collector verdict.
