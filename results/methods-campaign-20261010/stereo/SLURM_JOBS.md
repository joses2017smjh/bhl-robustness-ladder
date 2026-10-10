# Stereo methods campaign predeclared ledger — October 10, 2026

The original October 9 results remain unchanged. New source/runtime is frozen before smoke; scored development uses that same archive only after actual smoke PASS. One GPU job is active at a time. Every development job retains both paired arms and all original inputs; root archives verified raw evidence to GitHub before the next pair is submitted, to respect the home quota.

## Utility build and tests

- Diagnostic wrapper attempts v1/v2: failed before inference (missing original OpenCV headers, then missing generated g2o header). Logs retained under `/tmp/bhl-orb-diag-build[-v2]-20261010.log`.
- Diagnostic wrapper v3: BUILD PASS; runtime SHA-256 `31f2d5b1ce9f5ae81f9ea18e4dc9769a82995950b1530a5a3cabdcc0080e5d08`. Original compiled native SLAM library is unchanged; original generated header was restored and byte-verified.
- Allocated CPU QA: 66 passed. First environment-only attempt had two existing strict input-path rejections under Slurm's scratch symlink; explicit `TMPDIR=/tmp` fixed the environment, no implementation change.

## GPU jobs

| Job | Episodes | Status |
|---|---:|---|
| `stereo-methods-smoke-20261010-v1` | 4 | INCOMPLETE 21757159: packaging rejected before inference |
| `stereo-methods-straight-s480000-20261010-v1` | 2 | SUPERSEDED, NOT SUBMITTED |
| `stereo-methods-straight-s480001-20261010-v1` | 2 | SUPERSEDED, NOT SUBMITTED |
| `stereo-methods-straight-s480002-20261010-v1` | 2 | SUPERSEDED, NOT SUBMITTED |
| `stereo-methods-dogleg-s480000-20261010-v1` | 2 | SUPERSEDED, NOT SUBMITTED |
| `stereo-methods-dogleg-s480001-20261010-v1` | 2 | SUPERSEDED, NOT SUBMITTED |
| `stereo-methods-dogleg-s480002-20261010-v1` | 2 | SUPERSEDED, NOT SUBMITTED |
| `stereo-methods-occluders-s480000-20261010-v1` | 2 | SUPERSEDED, NOT SUBMITTED |
| `stereo-methods-occluders-s480001-20261010-v1` | 2 | SUPERSEDED, NOT SUBMITTED |
| `stereo-methods-occluders-s480002-20261010-v1` | 2 | SUPERSEDED, NOT SUBMITTED |

No global result is reported from a partial pair set. Development outcomes must be collected by `stereo_methods_collect.py`, which rejects missing or duplicate cases, altered runtime/actor hashes, and unmeasured outcome values.

## Frozen scientific intake

Source/runtime archive SHA-256 `1407c1bb1f63bc15fd1bac500079852f874a21c785d3d1be82964e9104d2cdc9`, protocol SHA-256 `944ae6dbed4cf934dfabe5757392b5e8e42acca14f1445da1b050728a751c660`; 871 verified members; archive 186197192 bytes. Final targeted QA: 76 passed, including independent paired-outcome aggregation and the root perceptive-policy checks.

## Read-only original capture diagnostic step

Predeclared allocated 2-CPU utility step in 21756762: fresh 40-frame native smoke, then all 450 original pairs with genuine native telemetry. Source is the same SHA-verified frozen scientific archive. This consumed capture supports diagnosis only, not new confirmation. Exact runner and input hashes: `diagnostic-replay/plan.json`.

## Packaging-only revision v2

The strict runtime extractor rejected directory headers before native initialization. File-only repacking preserves every native runtime payload byte (proof: `native-diagnostic-build/file-only-repack.json`). Native runtime archive SHA-256 `8ef2f4a0813ff97f457a97b9971e41555241b83f66c68b10c34a4a3986a28c3f`; source/protocol is frozen again and a fresh actual smoke is required. No scientific setting or original result changed.

| Job | Episodes | Status |
|---|---:|---|
| `stereo-methods-smoke-20261010-v2` | 4 | PASS 21757496; 4 episodes, zero falls/contacts, one verified recovery |
| `stereo-methods-straight-s480000-20261010-v2` | 2 | NOT_SUBMITTED |
| `stereo-methods-straight-s480001-20261010-v2` | 2 | NOT_SUBMITTED |
| `stereo-methods-straight-s480002-20261010-v2` | 2 | NOT_SUBMITTED |
| `stereo-methods-dogleg-s480000-20261010-v2` | 2 | NOT_SUBMITTED |
| `stereo-methods-dogleg-s480001-20261010-v2` | 2 | NOT_SUBMITTED |
| `stereo-methods-dogleg-s480002-20261010-v2` | 2 | NOT_SUBMITTED |
| `stereo-methods-occluders-s480000-20261010-v2` | 2 | NOT_SUBMITTED |
| `stereo-methods-occluders-s480001-20261010-v2` | 2 | NOT_SUBMITTED |
| `stereo-methods-occluders-s480002-20261010-v2` | 2 | NOT_SUBMITTED |

## Durable serial dispatcher

Predeclared 1-CPU / 1-GiB / 24-hour utility outside the scientific container. Frozen plan `dispatcher/plan.json` SHA-256 `7e3a7db98f8f96eb882b7574e7ada8af530585ab9e71b5be3148e4f9b1c57080` verifies actual smoke 21757496 completion/PASS and the scientific archive before submitting each of nine paired jobs. It reserves 192 MiB per active shard, preserves 300 MiB global headroom, publishes original raw evidence with checksum and fresh-download verification, and retires only the local duplicate archive. No implicit failed-job retries or source changes.

## Durable serial continuation

Actual same-source V2 smoke **21757496 PASS** completed all four short episodes and exercised verified map-frame recovery. Its original raw archive was published with fresh-download SHA256 verification. Controller V1 **21757524** was cancelled while waiting for storage before any development submission. Controller V2 **21757621** now queues the same nine predeclared matched pairs after rechecking the completed smoke from durable receipts and Slurm accounting. It reserves 192 MiB per raw archive and preserves 128 MiB home headroom.

Read-only original-data diagnostic step **21756762.20** processed all 450 pairs. Textured boxes and thin posts stayed below the unchanged `N > 500` initialization feature threshold (maxima 463 and 427) and tracked 0/150 each. Ramp/step tracked 122/150. The original negative results are unchanged.
