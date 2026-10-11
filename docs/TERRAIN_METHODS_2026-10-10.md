# Terrain heading control and uncertainty-aware ground mapping

Implementation date: October 10, 2026. This document declares a **new development campaign**. It does not change the negative October 9 qualification: three of eighteen clean crossings, fifteen wall contacts and no qualifying actor. Implementation, unit validation, actual smoke and full development results are separate states; only launch/completion receipts establish that experiments ran.

## Questions and methods

The earlier collisions with no falls motivate a controller ablation before retraining the gait. Three frozen `dr-default-s0/s1/s2` actors each receive identical matched physical courses and reset seeds. The path controller uses native FAST-LIO2 LiDAR–IMU estimates registered at a declared start station. Simulator position is available only to sensor generation and the evaluator.

| Arm | Steering and speed | Common behavior |
|---|---|---|
| `baseline` | Forward command 0.30 m/s, zero yaw correction | Five-second stationary initialization, estimated-pose goal stop, native loss/reset/staleness stop |
| `heading` | Straight-path pure pursuit, 0.60 m lookahead; yaw rate clipped to 0.50 rad/s | Same common behavior and learned gait |
| `regulated` | Same pursuit; reduce speed with curvature or observed lateral obstacle proximity; stop for frontal obstacles | Same common behavior and learned gait |

Speed is bounded between 0.08 and 0.30 m/s when moving. Regulation is inspired by [Macenski et al., *Regulated Pure Pursuit for Robot Path Tracking*](https://arxiv.org/abs/2305.20026). The code implements a small gait-limited controller; it does not claim to reproduce Nav2's complete controller or its results.

The baseline is a **fresh matched baseline with native goal/loss stopping**. It is not a repeat of the earlier blind screen, which stopped on an evaluator goal condition. All comparative claims use the new baseline running alongside the new methods.

## Frozen experiment

- Three frozen actors × flat/1 cm steps/3° ramp × ten fresh reset seeds `410000..410009` × three arms = **270 development episodes**.
- Five-metre prescribed straight route; forty-second matched horizon. A clean goal requires coming within 0.30 m of the true goal, completing the full horizon and having zero falls, wall contacts or nonfinite states.
- The controller stops within 0.20 m using its own estimated goal; this stop latches, so a later estimate does not restart the robot.
- Original onset/friction variation remains ±2.5 cm and 0.80±0.02. Four raised boxes outside the walking corridor add asymmetric scan structure. This explicitly changes the old geometry, identically in all three new arms.
- Actual native FAST-LIO2 runs in **all arms**. The sensor rig retains 5 Hz timed 3D scans and 200 Hz IMU acquisition; elevation angles span −75°..25° to include near ground. Input packets and native replies are retained.
- Native client wall time is charged to 40 ms gait updates. New poses and tracking flags are unavailable until arrival. Simulated ray generation and diagnostic map updates are excluded from this latency metric, so it is not an end-to-end real-time claim.
- The first smoke uses actor s0, flat ground, seed `400000`, eight seconds per arm. All three smoke episodes must finish their horizon with native tracked frames and no nonfinite state. The smoke does not establish traversal success.
- Development is nine shards: one actor and one terrain per shard, thirty episodes each. All ten seeds and all three arms remain together within each shard. Arm execution order is a declared deterministic permutation from `RNG(seed+500000)` per matched seed group. No adaptive replacement of failed cells or actors.
- Seeds `420000..420019` are reserved. Confirmation is **not included** in this protocol. Any actor/arm with at least 27/30 clean development goals and zero falls/contacts/nonfinite is eligible for a separately frozen confirmation design with new course geometry.

Primary comparisons are paired clean-goal rate differences, contact reductions, lateral path RMSE, minimum centre-to-wall clearance and falls. Clearance is a robot-centre geometric measure, not whole-body clearance. Summaries retain native tracking/latency and ground-map metrics. The complete collector rejects missing, duplicate or unexpected cells and independently checks goal/contact labels against retained traces. Its 95% intervals bootstrap ten reset-seed clusters, conditional on the three fixed actors and terrain families; those intervals do not estimate general robot safety.

## Probabilistic map implementation

`UncertainElevationMap` accepts raw point clouds, estimated sensor transforms, explicit point variance, pose variance and acquisition times. Per-cell medians are fused with scalar Bayesian updates. Range uncertainty, rotation lever-arm uncertainty, translation uncertainty and within-scan motion contribute to measurement variance. A shared pose-variance floor prevents the many correlated points in one scan from producing artificial certainty. Variance grows with age; support older than two seconds becomes unknown. Large innovations replace old heights instead of averaging a terrain edge away.

This approach is informed by the [ANYbotics elevation-mapping implementation](https://github.com/ANYbotics/elevation_mapping) and the uncertainty-aware elevation-mapping literature linked there. It is an independent compact implementation, not a reproduction of that full estimator. FAST-LIO2's runtime does not export calibrated pose covariance here: the declared 5 mm translation and 0.5° attitude uncertainties are **assumptions whose empirical interval coverage must be measured**. `point_variance_m2` can be a scalar, one vertical variance per point, or an N×3 local diagonal covariance, supporting either LiDAR or camera-derived point clouds.

Ground inference rejects vertical columns with more than 8 cm within-cell height spread and cells more than 18 cm above the scan's low envelope. This is a geometric ground heuristic, not semantic labeling. Unknown cells never count as traversable. Height, variance, observation count, age, slope, neighboring step and hazard support are available in the output grid. The map is diagnostic in this first controller ablation; it is not fed to the gait or used to explain a controller gain that it did not cause.

**Ground-only evaluation** uses separate downward rays restricted to physical `terrain_*` geometry and excludes the wall footprint and outside-corridor cells. Neither ground labels nor those rays enter inference. Report coverage alongside height MAE/RMSE/p95, empirical 95% interval coverage, hazard recall, false traversable ground cells and unknown hazard cells. Missing support has unavailable accuracy, never zero error. Final map grids and all input scans are retained; every delivered scan also produces a map-metric record.

## Commands

After the pinned upstream submodule is present, make a portable frozen protocol from the existing actor inventory:

```bash
python scripts/bench/terrain_methods_campaign.py --create-protocol \
  --template /nfs/stak/users/sanchej7/humanoid-native-20261009/terrain-v3/protocol.json \
  --runtime-archive /nfs/stak/users/sanchej7/humanoid-native-20261009/artifacts/lio-runtime-inference.tar.gz \
  --output /absolute/new/terrain-methods-protocol.json
```

Use the repository's `h34_campaign.py` freeze/intake/Slurm flow. The generated protocol contains a smoke job and nine bounded CPU development jobs; root records the intake, source/archive hashes and scheduler IDs in `SLURM_JOBS.md` before submission. A newly frozen source requires a new smoke. No rendering is required for these jobs.

```bash
# Examples executed inside the frozen launcher environment:
python scripts/bench/terrain_methods_campaign.py --protocol "$H34_PROTOCOL" --phase smoke
python scripts/bench/terrain_methods_campaign.py --protocol "$H34_PROTOCOL" \
  --phase development --actor dr-default-s0 --terrain flat

# Run only after all nine development output directories are collected:
python scripts/bench/terrain_methods_campaign.py --protocol /absolute/frozen/protocol.json \
  --collect /absolute/all-collected-shards --output /absolute/complete-summary.json
```

The input protocol includes every actor and runtime hash, package versions, source hashes, exact controller/map parameters and reserved cohorts. The runner validates inputs again after execution. Frozen source and raw scientific outputs are retained through the existing campaign machinery.

## Validation and result boundary

`tests/test_terrain_methods.py` verifies shared stale/reset/goal stopping, heading direction, speed regulation, obstacle stops, variance propagation, covariance rotation, age invalidation, vertical-wall rejection, ground-only denominators, unavailable errors and paired negative summaries. The allocated Slurm unit check passed **34/34 tests**; actual native smoke and development status must be established by their own retained receipts.

No contact reduction, traversal improvement, map accuracy or calibrated-confidence result is claimed from implementation or unit tests alone. Any supported resume number must come from the complete matched development campaign or its separately reserved confirmation.
