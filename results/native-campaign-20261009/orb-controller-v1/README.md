# Immutable ORB post-build continuation

Backup job **21742797** depends on `afterok:21741510`. It is CPU-only
orchestration (one CPU, 4 GiB, 20 minutes, preempt, no requeue), with frozen
archive SHA256 `1562c43b5aa5ba6cc01c22cf25a249a96f3da88f182f5061029ea268f255504f`.
The durable directory is
`/nfs/stak/users/sanchej7/humanoid-native-20261009/orb-controller-v1`.

[plan.json](plan.json) pins the actual build attempt and both separately frozen
scientific templates. After a genuine PASS build it verifies all build/runtime
members, compacts the full runtime with original source/licenses/vocabulary,
repeats actual executable and dependency smoke checks inside the pinned SIF,
then promotes each unchanged template and submits its fresh smoke/dependent run.
No estimator output is consulted to choose settings or thresholds.

A separately predeclared [allocated-step promotion plan](../orb-build-step-v1/promotion-plan.json)
may use identical frozen controller code if actual build step 21739893.1
finishes first. In that case the pending backup jobs alone can be cancelled,
with receipts preserved, after manual promotion succeeds. Submission is not
build completion and is not a SLAM/navigation scientific result.
