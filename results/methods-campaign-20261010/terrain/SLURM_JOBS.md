# Native terrain controller campaign — preregistered October 10, 2026

Status: source frozen; actual native smoke is the next authorized submission. No development job submitted yet.

Scientific declaration: [Terrain methods](../../../docs/TERRAIN_METHODS_2026-10-10.md).
Portable protocol: `/tmp/terrain-methods-20261010-protocol.json` (planned; hash recorded after generation).
Durable intake: `/nfs/stak/users/sanchej7/humanoid-methods-20261010/terrain-v1` (planned).

All three frozen gait actors are compared on the same three course families, ten fresh seeds `410000..410009` and three controller arms. Total development denominator: **270 episodes**. Prior qualification results are unchanged. Smoke seed `400000` is disjoint. Reserved confirmation seeds `420000..420019` are not run in this campaign.

| Declared job | Purpose | Episodes | Resource cap | Dependency |
|---|---|---:|---|---|
| `terrain-methods-smoke-20261010` | Three-arm actual native integration smoke, 8 s each | 3 | 4 CPUs, 8 GB, CPU share/eecs, 30 min | Frozen inputs verified |
| `terrain-methods-dr-default-s0-flat-20261010` | Frozen actor s0, flat, all seeds/arms | 30 | 4 CPUs, 8 GB, CPU share/eecs, 4 h | Same-source smoke PASS; lane A |
| `terrain-methods-dr-default-s0-small_steps-20261010` | Frozen actor s0, steps, all seeds/arms | 30 | Same | Same-source smoke PASS; lane B |
| `terrain-methods-dr-default-s0-ramp-20261010` | Frozen actor s0, ramp, all seeds/arms | 30 | Same | Lane A predecessor completed |
| `terrain-methods-dr-default-s1-flat-20261010` | Frozen actor s1, flat, all seeds/arms | 30 | Same | Lane B predecessor completed |
| `terrain-methods-dr-default-s1-small_steps-20261010` | Frozen actor s1, steps, all seeds/arms | 30 | Same | Lane A predecessor completed |
| `terrain-methods-dr-default-s1-ramp-20261010` | Frozen actor s1, ramp, all seeds/arms | 30 | Same | Lane B predecessor completed |
| `terrain-methods-dr-default-s2-flat-20261010` | Frozen actor s2, flat, all seeds/arms | 30 | Same | Lane A predecessor completed |
| `terrain-methods-dr-default-s2-small_steps-20261010` | Frozen actor s2, steps, all seeds/arms | 30 | Same | Lane B predecessor completed |
| `terrain-methods-dr-default-s2-ramp-20261010` | Frozen actor s2, ramp, all seeds/arms | 30 | Same | Lane A predecessor completed |

At most two four-CPU development jobs run concurrently. Actual smoke completion is inspected before any development submission; pending dependencies alone are insufficient. Scientific source/input changes require a new frozen intake and fresh smoke. Later shard submission is subject to measured output storage availability; unfinished cells stay incomplete, never count as failures or improvements.

The 450 MB working home-budget target requires publishing and independently verifying completed scientific archives before removing any duplicate home packaging. Original campaign evidence is not touched. Source/runtime bundle is expected to be 60–90 MB compressed; raw-output size will be estimated from the actual smoke before development launch. Local node scratch is used for execution and collection.

## Submission and completion receipts

None yet. Scheduler IDs, archive SHA256, source manifest identity, output SHA256 and actual smoke decisions will be appended here as measured.

## Frozen intake v1

- Archive: `4a9d31af1de04e0fdd380e8c2dda01a6b0b65a9819b3d394b5829f57c0b90c77`, **75,295,483 bytes**.
- Protocol: `8060a32ceea75ca29a4eb08444385f71d755e6c7055760790d418b509d858e31`.
- 866 frozen source files; 186,399,881 uncompressed input bytes. Parent commit `e17e76631c9dcadd6f21a941bce00287617a3229`; current local implementation edits are explicitly recorded in intake.
- Native runtime and three actor/checkpoint identities are SHA-verified inputs.
- Allocated unit check: **34/34 PASS** in 0.88 s.
- First actual submission authorized: `terrain-methods-smoke-20261010`, 4 CPU/8 GB/30 min, no requeue. No scored development submission until actual smoke completion is inspected.

Actual smoke submitted **21757058**, October 10, 2026 at 13:22 PDT. Outcome pending; submission is not a pass.

## Actual smoke result and first development decision

Smoke **21757058 PASS**, 3/3 full eight-second horizons, native tracking in all arms, zero falls/contacts/nonfinite. Scheduler completed 52 s; measured runner wall time 36.13 s. Raw archive **4,270,699 bytes**, SHA256 `9a461bd0aee6ec27c4f09586e9da22ecbf026efc3ecd0f93c2467da909218615`. This is integration evidence only; no five-metre success claim.

The measured output rate projects ~214 MB per 30-episode development shard. Development concurrency is therefore reduced to **one** until publication/offload space is verified. Next authorized submission: `terrain-methods-dr-default-s0-flat-20261010`, afterok:21757058, same frozen archive and protocol, all ten fresh seeds × three arms. Remaining eight shards stay predeclared but unsubmitted.

First development shard submitted **21757070** (s0/flat, 30 episodes), October 10 at 13:23 PDT, afterok:21757058. Remaining eight shards have not been submitted.

## Durable serial dispatcher

A separate credentialed host utility is frozen at `/nfs/stak/users/sanchej7/humanoid-methods-20261010/terrain-dispatch-v1`. Plan SHA256 `ca1551302b518f7b352a94560b3e4ba5623a65efaaf51885ed330c01584479a4`; source/helper checksums are in its public intake. Resource cap: **1 CPU, 1 GB, CPU share/eecs, 24 h, no requeue**. It adopts completed first shard21757070 and submits the remaining eight frozen development jobs serially, each afterok:21757058. The utility survives the interactive allocation.

Each next job requires a completed expected30episode result, release upload with fresh-download checksum verification and local duplicate archive retirement. A shared fcntl reservation registry promises256MiB per terrain output; after outstanding promises, at least128MiB real home-quota headroom must remain. Missing space delays submission. Runtime/integrity/cohort failure stops dispatch without retuning or rerunning. Negative complete scientific outcomes are retained. No Git branch is pushed by this utility.

Dispatcher guard QA:14/14 PASS in0.07s; source freeze includes those exact guarded behaviors. New combined18test standalone CPU check is running. Dispatcher submission follows; no submission made in this intake record.

Durable dispatcher submitted **21757434**, after standalone allocated CPU QA21757387 passed **18/18 tests** (14 dispatcher guards +4 sensor-map fusion tests). It will publish/verify existing shard21757070 and then drive the remaining eight scientific shards under global storage reservations. The public submission receipt records the exact utility invocation.

## Durable serial continuation

The first 30-episode flat shard **21757070** completed: baseline 8/10 clean goals, heading 10/10, regulated 10/10; zero falls. It remains only one of nine shards. Controller V1 **21757434** stopped before its next scientific job because Slurm had expired the completed smoke's dependency record. The rejected submission created no scientific job and is preserved under `unsubmitted-attempts`.

Controller V2 **21757620** independently rechecks the completed same-source smoke through its original receipts and Slurm accounting before every submission. It does not depend on an expired controller record. The second shard was submitted as **21757630**. Only the exact previously rejected pre-submission request is retried; runtime failures never rerun automatically. The site sbatch wrapper also rejected the first V2 utility request because it split a wrapped command; no job was created. The preserved correction submits a script file.
