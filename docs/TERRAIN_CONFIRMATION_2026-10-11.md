# Held-out terrain confirmation — October 11, 2026

The complete 270-episode development campaign qualified five actor/controller
combinations for a new experiment. This confirmation evaluates those five using
**480 fresh episodes** on newly generated physical courses and the
newly reserved seeds **440000–440019**. The three learned actors, their controller gains,
native FAST-LIO2 feedback, observation boundaries and success definition stay fixed.

The matched design includes 180 unique baseline episodes and 300 candidate episodes:
all three actors use heading pursuit, while only `dr-default-s0` and
`dr-default-s2` use regulated pursuit. The unqualified `dr-default-s1` regulated
controller is excluded. Baseline outcomes are shared between the two prespecified
comparisons where appropriate, without inflating the cohort count.

New courses change obstacle onset, the number and widths of 1 cm step treads, the
3° ramp's rise and plateau lengths, and six exterior LiDAR registration landmarks.
The protocol serializes every geometry before execution. A separate 24-episode,
12-second smoke at seed 450000 must exercise all actors, terrain families and
controllers before the nine scored shards can start. Scored episodes retain the
full 40-second horizon, including survival after reaching the goal.

Each of the five candidate combinations confirms the engineering gate with at
least 54/60 clean goals, no falls, contacts or nonfinite states, and a positive
matched goal-rate difference. The collector reports every result, including
negative gates, and independently recomputes outcomes from retained simulator
traces. Its intervals resample 20 shared geometry/reset-seed clusters and remain
conditional on these fixed actors and course families. This is held-out simulation
confirmation, not hardware validation or a population safety estimate.

## Evidence and execution

[The predeclared ledger](../results/terrain-confirmation-20261011/SLURM_JOBS.md)
contains the full design, source hashes, job identities and preserved failures.
The [frozen protocol](../results/terrain-confirmation-20261011/protocol.json)
records all 63 scored/smoke course geometries. Source and raw publications use the
[October 11 research release](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/tag/research-campaign-evidence-20261011).

The first smoke, job 21768302, failed when the augmentation wrapper attempted an
exclusive write to the already-created base receipt. The corrected wrapper keeps
that receipt and creates `confirmation-episode.json`; a regression test covers
the failure. No controller, geometry, seed, horizon or gate was tuned. Failed
source, raw output and execution receipts remain intact. Fresh smoke job 21768354
used the corrected source and passed all 24 episodes. However, its scored steps shard 21768408 later failed on a degenerate original scan timestamp. The original 420000-series confirmation remains **INCOMPLETE**; dispatcher 21768392 stopped, and the other two jobs completed separately. The incomplete cohort retains 127 completed episodes (60 flat, seven steps and 60 ramp); none enter the revised cohort.

Three dependencies on repository main changed after development. The confirmation
source archive restores all 71 development-pinned scientific files in an isolated
staging tree and adds the new confirmation code. The current checkout is not
rolled back. The [restoration receipt](../results/terrain-confirmation-20261011/development-source-restoration.json)
and input manifest preserve every hash. Running protocol creation against changed
scientific dependencies fails explicitly; reproduction should start from the
verified frozen source archive.

All durable output is on HPC share. The dispatcher now permits one active shard
for new submissions; two already-running shards were preserved during the cap
reduction. A locked registry shares the 1 GiB reservation per terrain shard and
8 GiB free-space floor with the other October 11 campaigns. Completed raw
archives are published and independently downloaded for checksum verification
before their local durable packaging duplicates are removed. Compact receipts
remain. A final collector requires all 480 declared identities and publishes its
summary automatically after the nine jobs finish. No physical actuation is used.

Focused allocated-node verification passed **40 tests**, including reserved-cohort
leakage, new physical geometry, qualification provenance, trace rescoring,
exclusive receipt writing, source tampering, storage bounds and dispatcher scope.

## New acquisition contract and current status

A new confirmation-only adapter retains each invalid-time raw packet verbatim,
with its rejection reason, and withholds it from the unchanged native client.
It never fabricates points or spreads timestamps. The IMU cursor remains unchanged
until a valid scan arrives, preserving every causal integration sample. The
existing 350 ms pose-age stop remains the control response to unavailable pose.
Independent collection checks every retained rejection against the original
native timing contract and reports invalid-scan counts.

The revised protocol was frozen before executing seeds 440000–440019 or smoke 450000.
The old consumed cohort is explicitly excluded. It retains the same 480-episode
matched design, actors, controllers, terrain-generation rules, native binary,
horizons and qualification thresholds. Its source archive is
`8a2ec6a64f4cac0868cbf21fd018dacc2f7179daabc634724c5b38cd3bf8851f`;
protocol SHA256 is
`18323a18a3b963e67869de46554513c178ac71037d0caff9a4f68400579fd65c`.
The new smoke **21768612 passed all 24 episodes** in 259.1 seconds, with native tracking, no rejected scans, and no falls or nonfinite states. Controller **21768854** now schedules the new 480-episode cohort as shared storage reservations allow. Orchestration-only replacements lowered the concurrency cap from three to two, then one as unrelated shared-storage use grew, preserving storage capacity for the parallel camera campaign. Existing scientific jobs 21768687 and 21768688 were adopted without cancellation or repetition. The scientific source, protocol, seeds and gates are unchanged; all orchestration plans and replacement reasons are retained.
The required fresh same-source smoke gate has passed; no confirmation result is claimed until all 480 episodes are independently collected.
The failed shard was published and independently downloaded for checksum
verification; its local durable duplicate and storage reservation were released.
Observer 21768577 completed the same verified publication and retirement for both original remaining shards. Their raw archives and the inactive v2 source remain downloadable from the release; compact receipts and restore information remain local.
