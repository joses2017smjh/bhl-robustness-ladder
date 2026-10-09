# H3 / H4 campaign evidence

These campaigns are **COMPLETE: H3 NEGATIVE, H4 PASS**. Final evidence was
reviewed October 9, 2026. Parents, training seeds, final-checkpoint choices and
all acceptance gates were declared before scored work.
[Design, complete results and limits](../../docs/H3_H4_CAMPAIGN_2026-10-08.md).

| Task | Scored array | Complete independent result |
|---|---|---|
| H3: IMU history versus matched repeated-current control | `21711075`, all six cells complete | **NEGATIVE:** 720/720 episodes, 0/3 paired seeds pass. At primary 80 ms delay, each arm/seed qualifies 25/30; every paired gain is zero. [Summary](h3/finalization/collection/scientific-summary.json). |
| H4: 22-DoF R1HO command-latched heading | `21711208`, all three seeds complete | **PASS:** 3/3 seeds jointly pass; 30/30 qualification turns, 9/9 straight walks; push falls 8/60, 9/60, 8/60. [Summary](h4/finalization/collection/scientific-summary.json). |

H3 uses a 12-DoF simulated biped and three matched fine-tuning seeds from one
parent. H4 uses a 22-DoF simulated humanoid, simulator yaw and three fine-tunes
of one selected R1H-s2 parent. These counts reuse declared benchmark/reset
cases; they establish no independent-sample significance or physical-robot
outcome. H3 supplies a complete controlled negative ablation. H4 supports a
repeated-seed simulation qualification claim retaining its push-fall counts.

Compact final evidence:

- [H3 finalization](h3/finalization/finalization.json), [collection receipt](h3/finalization/collection/collection.json), [publication verification](h3/finalization/publication-verification.json). All 720 original episode summaries are included in six per-cell `latency-evaluation.json` files.
- [H4 finalization](h4/finalization/finalization.json), [collection receipt](h4/finalization/collection/collection.json), [publication verification](h4/finalization/publication-verification.json). Every original turn/walk/qualification JSON and all 180 push-case CSV rows are included under three per-cell `gates/` directories.

Publication verification rehashed **every** original completion-listed member:
H3 **112 files / 242,242,616 bytes**; H4 **76 files / 70,348,951 bytes**.
The compact publication copies retain **932,839** and **250,058 bytes** before
adding the publication receipts themselves. Large raw traces, output/source
archives, checkpoints, policies and training events remain in durable storage.
Original completion and collection receipts retain exact hashes and locations;
absolute deployment paths are provenance, not relocated runnable exports.
No new training or evaluation ran during this publication review.

The original [first H3 cell verification](h3/v1/h3-run-20261008-21711158/independent-cell-verification.json)
remains beside the complete six-cell report. Both arms passed fresh non-scored
smoke `21711074`. H4's fresh smoke `21711207` passed before its full array.
CPU report jobs `21711668` and `21711669` independently recomputed the final
verdicts using immutable [observer-v2](observer-v2/observer-intake.json).
The [preflight](observer-v2/pipeline-preflight/preflight-verification.json)
is integration evidence and is excluded from the scored cohort.

[Compact reconstruction bundles](RECONSTRUCTION.md) retain original teacher
inputs and source overlays; both reconstruct all **815** frozen file hashes.
`h3/v1/` and `h4/v2/` retain canonical protocols, source manifests, parent pins,
intake/submission receipts and passing GPU smoke artifacts. Publication does
not retroactively replace their frozen source bytes.

`h4/v1/` preserves the first INCOMPLETE smoke: configuration capture occurred
before scene construction, whereas the teacher YAML was captured afterward.
No scored training ran from that archive. The blocked array was cancelled;
v2 fixed capture timing while preserving the strict recipe check and protocol.
Local CPU preflights and all GPU smoke episodes are non-scored evidence.

Original durable roots:

```text
/nfs/stak/users/sanchej7/humanoid-h34-20261008/h3-v1
/nfs/stak/users/sanchej7/humanoid-h34-20261008/h4-v2
/nfs/stak/users/sanchej7/humanoid-h34-20261008/h3-finalization
/nfs/stak/users/sanchej7/humanoid-h34-20261008/h4-finalization
```

Complete PASS, complete NEGATIVE and INCOMPLETE remain distinct. Neither a
scheduler exit nor one seed's pass is the scientific verdict. These report
jobs did not publish Git changes; the reviewed compact copies above preserve
their original bytes and complete negative records.
