# Processing-schedule order confirmation (predeclared, frozen, not yet run)

**Status: FROZEN_NOT_SUBMITTED.** Source commit `f86d753b1eca73bc0dbbefdcac72411d0460a1d1`, frozen archive
SHA256 `3be0441de4f74182d87a0664b4d8430da72bbc42d93197f90e3ad83d3b2e573b`, CPU validation and freeze job **21825716**. No GPU job has
been submitted. The [v1 results](PROCESSING_SCHEDULE_RESULTS_2026-10-12.md) stand
unchanged; this campaign can confirm or fail to confirm them, never replace them.

## Question

The v1 campaign (job 21801949) reported native FAST-LIO2 (LiDAR–IMU) clean goals
of **0/12 with the cumulative reference versus 11/12 with the bounded latest-input
schedule**. Its [order audit](../results/processing-schedule-20261012/reporting/order-audit.json)
found the order was fixed within each estimator: every LiDAR–IMU pair ran the
bounded schedule first, and every stereo pair ran the reference first. Each
2-episode shard also started a fresh process, so the first episode always ran
cold. This campaign tests whether the LiDAR–IMU goal gain survives when each
schedule runs first equally often, every scored episode follows a warm-up, and
the seeds are new.

## Frozen design

| Factor | v1 (2026-10-12) | Order confirmation (this design) |
| --- | --- | --- |
| Native estimator | ORB-SLAM3 stereo and FAST-LIO2 LiDAR–IMU | FAST-LIO2 LiDAR–IMU only |
| Scored seeds | 750000, 750001 | 751000–751007 (eight new seeds) |
| Pairs per estimator | 12 | 48 (8 seeds × 3 conditions × 2 routes) |
| Order within a pair | global alternation, fixed within each estimator | each schedule first in 24/48 pairs |
| Warm-up | none | 2 unscored warm-ups per shard, one per schedule |
| Shard | one pair | one seed × condition: warm-ups, straight pair, occluders pair |
| Fresh smoke | 12 cells, seed 749000 | 6 cells plus 2 warm-ups, seed 749000 |
| Conditions, routes, 40 s horizon, native settings, gait, gates | — | unchanged |
| GPU | `a40\|rtx8000\|…` (ran on cn-gpu6, RTX 8000) | `rtx8000` only, same 2 CPU / 16 GB / 1 GPU |

**Order rule.** Pair (seed index *i*, condition index *c*, route index *r*) runs
`(legacy_cumulative, bounded_latest)[(i + c + r) % 2]` first. Each shard therefore
runs ABBA or BAAB. Each schedule runs first in exactly half of the pairs within
every condition, every route, every condition × route cell (4/4) and every seed
(3/3). It also leads half of the 24 shards. The protocol records `pair_index`,
`position`, `first_pipeline` and `shard_execution_index` for every cell. Every
episode writes its declared and actual process position and wall-clock bounds
into `episode.json`. The collector rejects any mismatch between declared and
recorded order.

**Warm-ups.** Every shard re-extracts its scratch and starts a new Python
process, and every episode starts its own native process. Each shard therefore
begins with one 18 s warm-up per schedule, on the shard's condition, on the
straight route, with a reserved seed (749900), in the order of the shard's first
pair. Warm-ups are unscored by construction:

- they use a reserved seed and the role `warmup`;
- their raw images are written to unarchived scratch, and only compact timing receipts are kept under `output/warmup/`;
- they never enter completed cells, readiness gates, groups, pairs or the primary analysis;
- the collector rejects any scored row that is a warm-up.

Warm-ups are reported separately, by schedule.

**Seeds.** 751000–751007 and 749900 occur nowhere else in the repository, the
research root's protocols and reservations, or the sibling follow-up clones (word-bounded search,
2026-10-14). A test re-checks this with `git grep` and pins the seeds to the
files of this design.

## Why LiDAR–IMU only, and why eight seeds

This design spends the same 48-pair budget as "both estimators × 4 seeds", but on
eight seed clusters of the one estimator with a goal gain to confirm.

Stereo was 0/12 under both schedules. Every bounded stereo episode stopped for
good after native map identity was lost, so there is no stereo goal gain to test
for order effects. The stereo wrapper is also being changed in a separate
follow-up. The cost of dropping stereo is that its stale-fraction and latency
differences keep their v1 order qualification.

Running both estimators with eight seeds would need 51 launcher jobs, more than
the launcher's 32-job cap.

The power numbers below are for the exact two-sided McNemar test at α = 0.05.
They are recomputed by a test from `scipy.stats.binomtest`.

| Scenario (bounded-only, legacy-only pair rates) | 12 pairs | 24 pairs | 48 pairs |
| --- | ---: | ---: | ---: |
| v1-like (0.90, 0.00) | 1.000 | 1.000 | 1.000 |
| Half the v1 effect (0.48, 0.02) | 0.478 | 0.951 | 1.000 |
| A third of the v1 effect (0.32, 0.02) | 0.130 | 0.680 | 0.978 |
| Pure order artifact (0.40, 0.40), rejection rate | 0.023 | 0.030 | 0.034 |

Pairs within a seed share a world, so seed clustering lowers the effective
sample size. With a design effect of 1.5–2:

- 24 pairs fall to 12–16 effective pairs, giving half-effect power of 0.48–0.75;
- 48 pairs fall to 24–32 effective pairs, giving half-effect power of 0.95–0.99.

Eight seeds also give eight bootstrap clusters (6,435 exact multisets) instead
of four (35) or v1's two (3).

## Predeclared hypotheses and decision rule

- **H1 (primary).** In LiDAR–IMU, `bounded_latest` yields more clean goals than
  `legacy_cumulative` in matched pairs. This must hold with each schedule run
  first in exactly half of the pairs and both pipelines warmed up in every shard.
- **H2 (secondary).** Position within a pair (first or second) does not change
  clean goals or latency.
- **H3 (secondary).** `bounded_latest` lowers per-episode p95 modeled pipeline
  service and p95 navigation pose age.

**Primary outcome.** The paired clean-goal difference (bounded minus legacy) over
all 48 LiDAR–IMU pairs. A clean goal is unchanged from v1: within 0.30 m in the
evaluator trace, the full 40 s horizon, and no fall, contact or nonfinite state.

**CONFIRMED** requires every one of the following:

- all 48 pairs measured;
- exact two-sided McNemar p < 0.05 (`binomtest(bounded_only, discordant, 0.5)`);
- bounded-only pairs outnumber legacy-only pairs;
- the seed-cluster bootstrap 95% interval of the goal-rate difference lies above 0;
- bounded clean goals exceed legacy clean goals separately in the legacy-first
  pairs and in the bounded-first pairs.

The last condition is deliberately stricter than McNemar alone. It requires the
gain in the situation v1 never observed: the bounded schedule running second.

**NOT_CONFIRMED** is the predeclared negative result: the cohort is complete but
any CONFIRMED condition fails. An order artifact would look like this:

- bounded wins concentrated in bounded-first pairs;
- no bounded advantage in legacy-first pairs;
- a significant order test.

A significant excess of legacy-only pairs is flagged **REVERSED**.

**INCOMPLETE_NO_VERDICT** applies when any execution error occurs or any primary
cell is missing, duplicated or left UNRUN by a smoke gate. No verdict is
computed in that case.

**Flags:**

- **ORDER_EFFECT_DETECTED**: the first-position test or the sequence-by-winner
  Fisher test has p < 0.05. This is reported whatever the verdict.
- **ATTENUATED**: the result is CONFIRMED, but the bootstrap interval lies
  entirely below the v1 difference of 11/12.

## Statistics (implemented in `scripts/bench/processing_schedule_order_analysis.py`)

- **Per-pair differences**, bounded minus legacy, for clean goal, stale
  fraction and three latency metrics.
- **Goal rates** per schedule with Wilson 95% intervals. Episodes are treated as
  independent here.
- **Discordant share** with an exact Clopper–Pearson interval.
- **Goal-rate difference** with a seed-cluster bootstrap 95% interval. All six
  pairs of a seed are resampled together. All multisets are enumerated exactly
  with multinomial weights, and percentiles use the inverted CDF. Monte Carlo
  (20,000 resamples, seed 20261014) is used only above 200,000 multisets.
- **Seed-level sign test** on the per-seed mean goal difference.
- **Latency.** Paired differences in p95 modeled pipeline service
  (`full_pipeline_p95_ms`), p95 navigation pose age
  (`navigation_pose_age_p95_ms`) and p95 native client wall time
  (`client_wall_p95_ms`). Each reports mean and median with bootstrap intervals.
  A pair missing a value is excluded from that metric only, and counted.
- **Variability across seeds**: per-seed goal rates, goal difference and
  latency differences, with SD, minimum and maximum.
- **Order (first versus second position within the arm):**
  - discordant pairs won by the first-run schedule (binomial test against 0.5);
  - a Fisher exact test of sequence by winner;
  - goal rate by position for each schedule;
  - the latency period effect, with a two-sided permutation test.
- **Multiplicity.** There is one primary test. All other p-values are secondary
  and unadjusted.

The analysis runs on the strict collector's `campaign_result.json`. Collections
from the v1 design are analysed descriptively and are always reported as
NOT_CONFIRMATORY_DESIGN.

## Exclusion and stopping rules

- No scored episode is excluded after execution. Falls, contacts, early
  termination and nonfinite physics all count as no clean goal.
- Warm-ups and smoke cells are never scored.
- Smoke-gated cells stay UNRUN and are never counted as zero. Any gated primary
  cell gives INCOMPLETE_NO_VERDICT.
- There is one allocation and one attempt. The dispatcher refuses a second
  submission in the same campaign directory.
- There is no interim look at outcomes, no optional stopping, and no seed added
  after outcomes are known.
- Any execution error stops the remaining shards without retry. A rerun needs a
  new campaign directory, a fresh smoke and the unchanged protocol.

## Compute and storage

These estimates use the v1 LiDAR–IMU stage timings from job 21801949.

- **Runner time per 40 s episode**: 46 s nominal, 60 s at 10 Hz, 87 s with exposure.
- **18 s warm-ups**: 25–45 s each.
- **Archive and dispatcher verification**: about 50 s per shard.

Total wall time is about 2.3–3.0 h for:

- 24 shards (about 2.4 h);
- the 6-cell smoke with its warm-ups (about 5 min);
- profile and collection (under 1 min).

The 12 h limit leaves wide margin.

Raw output is about 2.2 GiB per seed, so roughly 18–22 GiB for the campaign. The
largest stage, an exposure shard, is about 1.2 GiB against the 2 GiB per-stage
archive cap. The margin holds because warm-up images are not archived. The
campaign cap is 48 GiB. The project had about 508 GiB of hard-quota headroom at
freeze, and it is above the soft limit within the user-authorized grace period.

## Launch (for the lead; do not run before review)

Run the `submit` wrapper from this clone. Its `scripts/` are byte-identical to
the frozen commit `f86d753`; later commits add only documentation and receipts.

```bash
cd /nfs/hpc/share/sanchej7/humanoid-research-20261011/followups-20261014-order/repo
/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python scripts/bench/processing_schedule_dispatch.py \
    submit --campaign /nfs/hpc/share/sanchej7/humanoid-research-20261011/processing-schedule-order-v1-20261014
```

`submit` does four things:

1. re-verifies the frozen intake against the live dispatcher;
2. writes `submission-start.json`;
3. issues exactly the line below;
4. records `submission.json`.

Do not run that line by hand. `execute` refuses any allocation that has no
`submission.json` from the wrapper.

```text
/apps/slurm/25.11.8/bin/sbatch --parsable --ntasks=1 --account=eecs --job-name=schedule-order-96 --partition=gpu --cpus-per-task=2 --mem=16G --time=12:00:00 --gres=gpu:1 --constraint=rtx8000 --export=NONE --no-requeue --output=/nfs/hpc/share/sanchej7/humanoid-research-20261011/processing-schedule-order-v1-20261014/processing-schedule-%j.out --error=/nfs/hpc/share/sanchej7/humanoid-research-20261011/processing-schedule-order-v1-20261014/processing-schedule-%j.out /nfs/hpc/share/sanchej7/humanoid-research-20261011/processing-schedule-order-v1-20261014/processing_schedule.sbatch /nfs/hpc/share/sanchej7/humanoid-research-20261011/processing-schedule-order-v1-20261014 3be0441de4f74182d87a0664b4d8430da72bbc42d93197f90e3ad83d3b2e573b
```

The job runs 27 stages in order:

- the identical-input profile;
- the fresh 6-cell smoke (with its 2 warm-ups);
- the 24 counterbalanced LiDAR–IMU shards;
- the strict collector.

After `dispatch-result.json` reports `COMPLETE`, run the frozen analysis once.
It needs only CPU and takes seconds:

```bash
/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python scripts/bench/processing_schedule_order_analysis.py \
    --collection /nfs/hpc/share/sanchej7/humanoid-research-20261011/processing-schedule-order-v1-20261014/order-collection-20261014-<jobid>/campaign_result.json \
    --output /nfs/hpc/share/sanchej7/humanoid-research-20261011/processing-schedule-order-analysis-20261014
```

The verdict is in `analysis.json`; `analysis.md` summarizes it. If the campaign
is `INCOMPLETE`, do not retry in the same directory. Freeze a new attempt with
tag `v2` and run a fresh smoke.

## Validation

CPU job **21825716** ran on `share`, node cn-c24, with 2 CPUs and 8 GB. It
completed 0:0 in 1 min 05 s (MaxRSS 2.4 GiB), at commit `f86d753` with a clean
tree. The exact procedure is
[`slurm/repo20261014/processing_schedule_order_freeze.sbatch`](../slurm/repo20261014/processing_schedule_order_freeze.sbatch).

- **Tests.** 345 tests and 22 subtests passed (v1's freeze: 306 and 11). This
  covered the 13 v1 freeze test files plus
  `tests/test_processing_schedule_order_analysis.py`. The tests check:
  - counterbalance within every stratum and ABBA/BAAB in every shard;
  - that warm-ups are absent from cells, gates, groups and pairs;
  - that the seeds are new;
  - that the counts and dispatcher constants match the protocol, and that the
    power table recomputes;
  - an end-to-end synthetic campaign through the real runner, collector and
    analysis, including order and warm-up tamper cases;
  - the analysis statistics: exact McNemar, Wilson intervals, an exact
    bootstrap against brute force, the permutation test, and a planted order
    artifact.
- **v1 replay.** The new collector, run on v1's retained collector inputs,
  reproduced every field of the published
  [v1 collection](../results/processing-schedule-20261012/collection.json) and
  all 24 result hashes. Its order audit reproduces the v1 confound: stereo
  12 legacy-first and 0 bounded-first, LiDAR–IMU the reverse. The analysis of
  the replay gives LiDAR–IMU 0/12 versus 11/12, exact McNemar p = 0.000977, order
  not estimable, and the verdict NOT_CONFIRMATORY_DESIGN
  ([replay analysis](../results/processing-schedule-order-20261014/v1-replay-analysis.md)).
- **Freeze.** `prepare` produced the campaign with status FROZEN_NOT_SUBMITTED:
  - 27 jobs;
  - scientific protocol SHA256 `a3f4a9343a7bf68428656fbe6fed09dc3f867cd8a329ad4e66aadc713fbbae35`;
  - launch protocol SHA256 `650229ae5ece8e7dc5577fa58308193823bbf35c2bffe43886cea6e7f50d66db`;
  - archive of 262,963,004 bytes.
- **Offline pre-checks after the freeze:**
  - `verified_intake` passes against the live dispatcher;
  - the archive's own campaign code regenerates the frozen scientific protocol
    byte-for-byte;
  - the frozen sources equal commit `f86d753`.
- **Old design pinned by tests:**
  - `declared_protocol()` still reproduces v1's scientific protocol bytes
    (SHA256 `b41d516b…`);
  - `declared_jobs()` reproduces v1's launch inventory;
  - the v1 sbatch command is reproduced exactly.

Receipts:
[validation](../results/processing-schedule-order-20261014/validation.json) ·
[freeze](../results/processing-schedule-order-20261014/freeze.json) ·
[intake](../results/processing-schedule-order-20261014/intake.json) ·
[scientific protocol](../results/processing-schedule-order-20261014/scientific-protocol.json) ·
[launch protocol](../results/processing-schedule-order-20261014/launch-protocol.json) ·
[test log](../results/processing-schedule-order-20261014/focused-validation-21825716.log) ·
[accounting](../results/processing-schedule-order-20261014/slurm-accounting.json).

## Limits and open risks

- `--constraint=rtx8000` matches cn-gpu5, cn-gpu6 and cn-gpu7 (all cascadelake,
  el9). v1 ran on cn-gpu6. Relaxing the constraint would weaken comparability,
  because the legacy schedule charges measured wall-clock callback cost into
  simulated time.
- Other jobs sharing the node can change wall-clock costs. Within a shard, pairs
  run back to back, so this exposure is balanced across schedules, not removed.
- A shard now runs six episodes in one process instead of two. The v1 smoke ran
  twelve in one process without incident.
- The seeds are not reserved by any registry. The parallel stereo follow-up
  should avoid 751000–751007 and 749900.
- This remains a simulation schedule model with one learned gait and ideal
  simulated sensing. It makes no hardware timing, population or stereo claims.
