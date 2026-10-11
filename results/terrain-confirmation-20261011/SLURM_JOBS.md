# Independent terrain confirmation — predeclared October 11, 2026

No confirmation outcomes were inspected when this cohort was specified. The complete
270-episode development collector qualified heading pursuit for all three actors,
and regulated pursuit only for `dr-default-s0` and `dr-default-s2`.

The new protocol fixes **480 scored episodes**, nine actor/terrain shards, and a
separate **24-episode, 12-second smoke**. The scored cohort uses the previously
reserved reset seeds **420000–420019** and full **40-second horizons**. Smoke uses
430000, which is excluded from the scored cohort. Every candidate is paired to its
actor/terrain/seed baseline; 180 unique baseline episodes support 300 candidate
episodes, without counting reused baselines twice in the cohort total.

The terrain generator changes the physical fixtures: onset 0.95–1.15 m; six step
treads of 0.29–0.41 m at levels [1,2,2,3,2,1] cm; a 3° ramp with 1.35–1.65 m rise
length and a 0.25–0.45 m plateau; six newly positioned exterior LiDAR landmarks.
The corridor, goal, actor checkpoints, sensor/native runtime, controller gains,
and outcome criteria are unchanged. Every seeded geometry is serialized in the
protocol before smoke. Simulator geometry and truth never enter controller input.

Smoke requires all 24 identities, native tracking, finite state, no falls, and the
full 12-second horizon. The nine scored jobs remain unsubmitted until this exact
frozen source/input smoke passes. No gain, geometry, seed or gate changes are
allowed in response to outcomes. Negative outcomes are retained.

A candidate confirms the predeclared engineering gate with at least **54/60 clean
goals**, zero falls/contacts/nonfinite states, and a positive matched clean-goal
rate difference. The collector reports all five comparisons regardless of gate
outcome and recomputes goals, contacts and falls from retained evaluator traces.
The 10,000-resample percentile intervals use 20 shared geometry/reset-seed clusters
and bootstrap seed 900011. They are descriptive marginal intervals for five
prespecified comparisons, conditional on the three fixed actors and three terrain
families; they do not establish family-wise discovery, hardware validity or
population reliability.

Resources are CPU-only: one task, 4 CPUs, 8 GB, up to 1 hour for smoke and 4 hours
per scored shard. Each raw shard has a 1 GiB output cap. All durable writes go to
`/nfs/hpc/share/sanchej7/humanoid-research-20261011/terrain-confirmation-v2`;
node-local temporary extraction and caches avoid the 25 GB home quota. The cohort
will be serially dispatched or bounded by the available shared-storage reserve.
No physical actuation is involved.

Implementation: `scripts/bench/terrain_confirmation_campaign.py` and
`src/bhl_robust/research/terrain_confirmation.py`. Focused allocated-node QA:
**24 passed**, including cohort leakage, new geometry, source qualification,
negative gates, independent rescoring, and tamper rejection.

Submission identities and checksums are recorded below only after freezing.

- Frozen source archive: `404632c60f0403fd130f735c99440a461d23eff2bee8d12ac8f7421d2dae9400` (75,595,294 bytes; 957 verified files).
- Frozen protocol: `341b9e44af80d4e9da9da027168ed99b197c2209b408469507d06f0558dbcfdb`.
- Smoke job: **21768302**, submitted 2026-10-11 09:37:56 UTC; full jobs remain gated.
- Three dependencies changed on repository main after development (`gait_clock.py`, `harness.py`, `mjcf_assets.py`). The confirmation freeze restores all71 original scientific files from the byte-verified development archive; the shared checkout is not rolled back. Exact hashes are recorded in `development-source-restoration.json`.

## Preserved receipt-writing failure and v2 execution

Job **21768302** failed after its first episode because the augmentation wrapper attempted to create `episode.json` a second time. No scored confirmation episodes ran. V1 source, raw output and failure receipts remain intact. V2 preserves the original receipt and writes a separate `confirmation-episode.json`, including the original receipt hash. It changes no geometry, seeds, controller, estimator, horizon or scientific gate. The added regression test reproduces the exclusive-write behavior.

- Corrected source archive: `68ba91a3727ad08ce67cadec8e9b921c5ef6b39d6fc69682ac9fa6bd1a2bd36a` (75,595,349 bytes).
- Corrected frozen protocol: `d0ae5591b052d2fde8bbf7df70410778f53ad3e0b5a3350ff340297a6aa405c6`.
- Fresh same-source smoke: **21768354**, submitted 2026-10-11 09:43:03 UTC.
- Allocated QA: **33 passed** (25 scientific/cohort/receipt checks and 8 dispatcher guard checks), 0.49 seconds.
- Full dispatch: at most three simultaneous shards, 1 GiB reserved per shard, 8 GiB shared free-space floor, no retries. Each completed raw archive is freshly downloaded and SHA256 verified before its local durable duplicate is retired. A strict 480-episode collector and compact final publication run automatically after all nine shards finish.

Publication uses the October11 research release. The never-submitted first dispatcher plan used the older release setting; its replacement plan is `9d331dae00ab2e02ec22eb81cb98f7f9c2c15e1315b2793d46570e0dd1b39eda` under `terrain-confirmation-dispatch-v2`. Scientific source and smoke are identical. Source publication was independently downloaded and SHA256 verified at the new release before dispatch.

## Same-source qualification and scored dispatch

Smoke **21768354 completed PASS**, scheduler `COMPLETED 0:0`, all24 episodes, native tracking in every episode, no falls/nonfinite states; wall time291.9s. Gate to confirmation is true. Dispatcher **21768392** was submitted only after this result was independently checked against completion hashes and Slurm accounting (2026-10-11 09:49:26 UTC). It will launch all nine predeclared shards subject to the storage/concurrency guard, then independently collect all480 episodes and publish the summary.

## Preserved incomplete cohort and separately registered acquisition revision

The first scored cohort is **INCOMPLETE**. Job21768408 (s0 steps) completed seven episodes, then the original native client rejected scan timing during the next baseline episode. Its last accepted scan had only four effective returns and a 1ms original time span as the robot moved beyond the finite course. The dispatcher stopped without retrying or submitting the remaining six shards. Two already-running jobs are allowed to finish; observer21768577 publishes them and releases their storage reservations. None of these data enter the revised confirmation.

The revised protocol registers **new seeds440000–440019** and **smoke450000**, before either is executed. The 480-episode/24-smoke matched design, geometry-generation rules, frozen actors, controllers, native binary, horizon and engineering gates remain unchanged. The new acquisition adapter checks the original native timing contract before submitting a scan. Zero-span, overlapping or mismatched original-time scans are recorded verbatim as unavailable measurements. No synthetic points, retimed returns or calibration changes are permitted. The IMU cursor does not advance for rejected scans; the next valid scan receives every still-causal sample. The original native guard and 350ms pose-age stop remain unchanged, so an unavailable measurement does not create a pose estimate.

Every episode retains the exact rejected packets and reasons plus their count. Independent collection verifies those reasons against each retained packet. The previously consumed420000-series cohort is explicitly excluded. A new source freeze and complete same-source smoke are required before the new480 episodes. Focused allocated QA:40/40 tests passed, including zero-span retention, unchanged point times, IMU continuity after rejection, repeated-scan rejection and all-arm causal stopping.

Revised sourceSHA `8a2ec6a64f4cac0868cbf21fd018dacc2f7179daabc634724c5b38cd3bf8851f`, protocolSHA `18323a18a3b963e67869de46554513c178ac71037d0caff9a4f68400579fd65c`. Fresh smoke **21768612** submitted2026-10-11 10:06:51UTC after reserving512MiB with the shared8GiB floor. Inactive v1source was independently published/download-verified before its75,595,294-byte durable duplicate was retired; active v2/v3sources remain intact.

Revised smoke **21768612 PASS**, all24 episodes,259.1s, no rejected scans/falls/nonfinite states and native tracking throughout. New full-cohort controller **21768672** submitted10:12:36UTC only after same-source completion+Slurm checks; planSHA `9d4b641030da689431bf9aba255ebf7a0479319297fcb2fddb6198b8c5293bdd`. Storage bounds and the strict final480collector remain active.

## Orchestration-only storage fairness adjustment

At 10:20 UTC, utility **21768672** was gracefully terminated with its scientific workers **21768687** and **21768688** still running. Replacement utility **21768752** adopts those existing submissions and caps concurrent terrain shards at **two**, allowing the parallel native-camera campaign access to the shared storage reserve. Source, protocol, cohort, gates and jobs are unchanged; no scientific episode was cancelled or repeated. The prior plan, state and reason are preserved in `dispatch-v3-superseded-cap3/`. Current plan SHA256: `67771d8042087f5a5c5a18c944203f15bb3f5f115f77f6dc115458d286c4d1fd`. The 1 GiB per-shard reservation, 8 GiB floor, verified publication and strict final collection remain in force.

## Superseded cohort retention completed

Old jobs **21768407** and **21768409** completed their 60-episode shards; with the seven completed steps episodes before failure, the original cohort retains **127 completed scored episodes** and remains **INCOMPLETE**. Observer **21768577** completed at 10:32 UTC after publishing and freshly downloading/SHA-verifying both remaining raw archives (914,713,019 bytes), then retiring only their durable packaging duplicates and releasing both reservations. The inactive v2 source (75,595,349 bytes) was independently downloaded and SHA-verified again at 10:32:41 UTC before its duplicate was retired. Its restore receipt is in `freeze-v2/source-retirement.json`. These data remain separate from the new 480-episode confirmation; the active v3 source remains intact.

## Second storage fairness adjustment

After publication released the old cohort reservations, unrelated shared-storage activity still left only about 11.3 GiB free, below the 8 GiB floor plus two 1 GiB terrain reservations and the native-camera 2 GiB reservation. At 10:34 UTC, utility **21768752** was gracefully replaced by **21768854** with orchestration concurrency **one**; both active terrain workers **21768687/21768688** continue untouched. The new controller will submit no replacement until fewer than one terrain shard remains active, allowing native controls to start after the first completes. Current plan SHA256: `e2f92da10a0fc161e24e67dc59587365c40df40a8b7749fe3031eddd1fcc64f3`. Previous cap-two plan/state/reason are retained in `dispatch-v4-superseded-cap2/`. Scientific source, protocol, cohort, gates and reservation sizes are unchanged.
