# Terrain traversal v2: unchanged science, smaller CPU reservation

This version changes only the smoke/run CPU request from four to two.
[The inherited-member proof](resource-change-proof.json) retains exactly the
same 834 frozen source/input members and identical scientific configuration
as v1. No fixture, actor, group, threshold or success rule was tuned after the
v1 ramp/LiDAR timeout. The [v1 raw smoke evidence](../terrain-v1/README.md)
remains retained, including its negative traversal outcome.

The [protocol](protocol.json), [input inventory](manifest.json) and
[intake](intake.json) bind the new archive. Requests are one CPU task, two CPUs,
8 GiB RAM, `share,eecs`, no GPU; caps remain 20 minutes for the two-episode
smoke and five hours for the conditional scored run.

- Fresh smoke: **21740943**, [submission](terrain-smoke-20261009-submission.json).
- Scored qualification/conditional confirmation: **21740946**,
  [submission](terrain-run-20261009-submission.json), `afterok:21740943`.

Submission is not a measured result. Qualification still requires six clean
full-course, full-horizon goals per actor. All passing actors proceed to the
untouched paired cohort; no passing actor means zero confirmation episodes
and a recorded negative research result. See the
[scientific implementation](../../../docs/TERRAIN_TRAVERSAL_2026-10-09.md).


V2 jobs21740943/21740946 were later cancelled while both remained PENDING; no episodes ran. See `pending-cancellation.json`. The exact-source replacement [v3](../terrain-v3/README.md) changes partition to preempt and explicitly disables requeue, with fresh smoke21742782 and scored21742786.
