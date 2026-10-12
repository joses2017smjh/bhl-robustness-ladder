# Completing the October 11 HPC simulation campaigns

**Final status, October 12, 00:38 PDT:** both campaigns completed: **200/200 native-control episodes and 480/480 terrain-confirmation episodes**. All five terrain combinations confirmed; native stereo navigation and the offline health screen retained negative results. [Scientific closeout](HPC_SIMULATION_CLOSEOUT_2026-10-11.md) and [verified final report](research-completion-20261011/20261012T073850Z-1d71a69295d2.md). The chronology below preserves recovery-time observations.

At 14:27 PDT, physical rig access is still unavailable, but it is **not a
prerequisite** for the two remaining campaigns in the
[October 11 research scope](RESEARCH_SCOPE_2026-10-11.md). These are the current
campaigns being completed unattended; this does not close every historical
research idea elsewhere in the repository.

| Campaign | Retained evidence at recovery | Remaining execution |
|---|---|---|
| Native camera/fault controls | 50/200 declared cells across five shards, including ten recovered original episodes | Fifteen predeclared shards; calibrate/freeze the health screen before the held-out seed; strict full collection |
| Independent native-pose terrain confirmation | 260/480 episodes across five verified shards | Four predeclared shards; independent raw-trace rescoring and all five confirmation decisions |

H3/H4 and the earlier development, occlusion and external TUM VI replay campaigns
remain completed with their reported positive or negative outcomes. No additional
physical robot motion or recording is required to finish these simulation
cohorts. Hardware calibration, sensor synchronization and validation on the user's
rig remain separate, unrun tasks.

## Packaging recovery, without repeating a scientific episode

Native shard `21776269` completed all ten episodes with runtime exit zero, but
its 2,939,286,434 raw output bytes exceeded the original 2 GiB packaging limit.
The wrapper therefore reported `INCOMPLETE` and Slurm reported failure. That
failure and its original receipt remain intact.

Recovery job **21778472** found the original node-local output, verified the
971-file scientific source inventory and retained all **15,273 output files**.
It published two complete tar volumes, independently downloaded each and checked
its SHA256. It did not rerun, retune or discard episodes. Only after verified
publication was the failed job's stale 2 GiB shared-space reservation released.
[Recovery and original failure receipts](../results/simulation-completion-20261011/recovery-evidence-manifest.json).

The replacement native controller **21778567** uses the same scientific archive
`3628b95ced285660e27c0b3b0ca92a097415beaeedfa912da751832acc3b8282`, protocol
`08a71ea1dcaad9460ef6f3990af227323e48492442fd3acb675a334a5808caf0`, and successful
14-episode smoke **21769132**. Its
[operational plan](../results/simulation-completion-20261011/native-dispatch-plan.json)
adopts completed work and submits each remaining original job once. Calibration
seed 730000 must finish and its health model must be published before any seed
730001 trial can be submitted. A simulation failure still stops new submissions.

The operational change keeps raw outputs on compute-node storage, raises the
explicit raw packaging/extraction allowance to 8 GiB per shard, and publishes
binary archive parts of at most 1 GiB each. Each part is freshly downloaded and
hash-verified; reconstruction also verifies the complete archive hash. Shared
storage receives compact receipts with a 16 MiB reservation per native job.
Frozen-source extraction retains its original 2 GiB bound. Experiment arguments,
environments, policies, seeds and score gates are unchanged.

The terrain controller **21776307** remains responsible for its original frozen
cohort. After reservation release, its next worker **21778527** entered the queue.
Later workers are automatically submitted as capacity permits. This campaign is
distinct from the separately running `hum5-nav-*` program, whose navigation uses
oracle pose; that program and its existing gate/report dependencies were left
under their own controller.

## Storage and unattended evidence

Two redundant compressed installation-cache wheels were removed after verifying
their independently installed packages. Installed environments, unpacked caches,
checkpoints and experiment evidence were preserved. The deletion reclaimed
**4.067 GiB**; subsequent accounting showed **13.759 GiB free on HPC share**.
Home remained **16.514/25 GiB**. The shared dispatchers retain their **8 GiB
free-space floor**. [Checksums and before/after checks](../results/simulation-completion-20261011/storage-cleanup.json).

Native controller state is under
`/nfs/hpc/share/sanchej7/humanoid-research-20261011/native-sensor-controls-dispatch-v3/`;
terrain state is under the sibling `terrain-confirmation-dispatch-v6/`.
Each collector verifies the complete declared cohort and publishes its final
evidence to the [research release](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/tag/research-campaign-evidence-20261011).
Collection success does not imply method improvement or a passed qualification
gate. Failed prerequisite gates and unrun cells remain explicit.

The completion observer independently checks final compact identities and
downloaded final artifacts, then publishes an immutable report under
`docs/research-completion-20261011/`. It also reports terminal failures or its
deadline as incomplete. It does not repeat the collectors' raw-trace rescoring,
alter existing results, bypass failed gates or access hardware. Observer **21778690** is submitted with a 44-hour observation deadline and a
47-hour allocation. [Observer plan](../results/simulation-completion-20261011/observer-plan.json) and
[submission receipt](../results/simulation-completion-20261011/observer-submission.json) preserve its identity.
At 14:31 PDT, both controllers were running and native worker **21778625** and
terrain worker **21778527** had started. The native controller adopted all five
retained shards. [Timestamped live status](../results/simulation-completion-20261011/live-status.json).

**105 focused tests passed**, covering the existing dispatch helpers and new
recovery, transport and report paths. Independent review checked frozen-source
identity, recovery scope and storage handling. Controller and
observer launch receipts record their actual scheduler status; queued work is
not claimed as completed simulation.

## First complete cycle under the replacement controller

At 14:40 PDT, native worker **21778625** had completed all ten visual-blackout
episodes, finished packaging on node-local storage, and freshly download-verified
its publication. Slurm recorded `COMPLETED 0:0` in **9 minutes 42 seconds**. Its
971 source/input files matched the original frozen archive. Controller 21778567
then autonomously submitted **21778787**, which was running the next condition.
Native retained progress is now **60/200 cells**; terrain still has **260/480
archived episodes** with its next shard running. Neither full campaign is complete.

[Worker completion, raw-publication receipt and controller continuation](../results/simulation-completion-20261011/first-transport-completion/verification.json)
verify the new operational path through execution, publication and the next
submission. [Full GitHub regression and physics workflow](https://github.com/joses2017smjh/bhl-robustness-ladder/actions/runs/38176721952)
checks implementation commit `18a95b9`; its current status is available at that link.

GitHub implementation validation completed successfully: **2,969 tests passed**,
132 skipped and 10 subtests passed, followed by the trained-asset integration and
seeded physics/fault replay checks. [Retained CI receipt](../results/simulation-completion-20261011/github-ci.json).
Later progress commits change documentation/evidence only; the checked implementation remains unchanged.

## Shared-storage quota clarification

The user explicitly authorized use of the shared-storage 2 T allowance. A direct
Lustre check confirms that inherited project **30762** has a **2 TiB hard quota**
and a **1.5 TiB soft quota**, with a **30-day block grace period** after crossing
the soft threshold. The project remained below the soft threshold at inspection,
with about **524 GiB remaining to the hard quota**.

The earlier `df` free-space observations were headroom below the soft threshold;
they did not represent all space available under the hard quota. Future storage
decisions should use project usage, hard headroom, remaining grace and physical
filesystem capacity. The home **25 GiB** limit is separate. HPC share is authorized
for bulk simulation data; installed runtimes and retained experiment evidence
should not be removed merely because `df` approaches the soft threshold.

[User authorization and exact quota-command evidence](../results/simulation-completion-20261011/shared-storage-authorization.json)
are retained locally and here. This clarification does not modify the already
frozen scientific cohorts or restart the healthy running controllers. Their
current conservative admission checks and node-local output publication continue.
