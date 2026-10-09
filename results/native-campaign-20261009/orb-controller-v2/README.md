# Corrected immutable ORB continuation

Backup controller **21743355** depends strictly on `afterok:21743261`.
Frozen controller archive SHA256:
`8b10a4edb7c11c5bfbd4417bc16717598cd4f6fd831573144d12845265bad7cf`.
It requests one CPU, 4 GiB, 20 minutes, preempt, no requeue, and no GPU.

[plan.json](plan.json) pins the corrected clean v6 build and both original
scientific templates. It requires actual native build PASS/exit zero, verifies
all original completion/runtime member hashes, preserves full source/license
and vocabulary files, and repeats real executable-version/dependency smoke
checks inside the pinned Ubuntu SIF. It then fills only the deferred runtime
hash/input into each template and submits fresh smoke/dependent-run jobs.

The [alternate allocated-step plan](../orb-build-step-v2/promotion-plan.json)
changes only actual execution provenance/artifact destination if genuine step
21739893.2 succeeds first. Original failed v5/cancelled-backup attribution is
retained as an original request, not represented as actual runtime execution.
No scientific output is read or used to modify algorithms, gates, actor,
calibration, cases, or seeds. This is orchestration, not scientific completion.
