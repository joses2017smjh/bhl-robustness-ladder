# H3 / H4 campaign evidence

These campaigns are **ACTIVE**, with scientific outcomes pending. All gates,
parents, training seeds and final-checkpoint choices were declared before
scored work. See [the design](../../docs/H3_H4_CAMPAIGN_2026-10-08.md).

| Task | GPU smoke | Scored array | Expected scientific report |
|---|---|---|---|
| H3: IMU history versus matched repeated-current control | `21711074`, both arms PASS | `21711075`, six cells, one concurrent GPU | All 720 episodes; independently verified traces and paired 2/3-seed gate |
| H4: 22-DoF R1HO command-latched heading | `21711207`, PASS | `21711208`, three cells, one concurrent GPU | All three seeds; original v2 plus turn/walk/push qualification; joint 2/3-seed gate |

The first H3 history cell (training seed 0) completed all **500 iterations**
and **120 scored episodes**. Its [independent cell verification](h3/v1/h3-run-20261008-21711158/independent-cell-verification.json)
recomputed causal packet history, progress, RMSE and falls from raw traces.
This is **one of six cells**, with the paired scientific comparison pending.

[Compact reconstruction bundles](RECONSTRUCTION.md) include original teacher
inputs and exact source overlays; both reproduce all **815** frozen file hashes.

`h3/v1/` and `h4/v2/` contain the exact canonical protocols, runtime-source
manifests, parent pins, intake/submission receipts and passing GPU smoke
outputs. The full source/asset archives remain in durable HPC home storage.
Their intake files record the original public base commit plus the dirty/new
source inventory and every runtime byte hash. Publication commits do not
retroactively replace the frozen inputs.

`h4/v1/` preserves the first smoke's INCOMPLETE result: configuration capture
occurred before scene construction, whereas the teacher YAML was captured
after construction. No scored training ran from that archive. Its blocked
array was cancelled; v2 moved the comparison after construction without
relaxing the strict physical-recipe check or changing the scientific protocol.

`h3/local_cpu_preflight.json` and `h4/local_cpu_preflight.json` contain actual
parent-policy expansion/export and short CPU physics checks. These and all
GPU smoke episodes are non-scored integration evidence.

Durable roots:

```text
/nfs/stak/users/sanchej7/humanoid-h34-20261008/h3-v1
/nfs/stak/users/sanchej7/humanoid-h34-20261008/h4-v1
/nfs/stak/users/sanchej7/humanoid-h34-20261008/h4-v2
```

Final reports must distinguish complete PASS, complete NEGATIVE, and
INCOMPLETE. Neither scheduler completion nor a single seed's pass closes
the objective. No achieved H3 latency improvement or H4 repeated-seed
qualification is available for a resume yet. A complete report will supply
the measured latency-conditioned counts, turn success, walk drift and push
fall counts to support an accurate project bullet.

CPU report jobs `21711668` (H3) and `21711669` (H4) use the immutable
[observer-v2](observer-v2/observer-intake.json) after each full array finishes.
The [actual preflight](observer-v2/pipeline-preflight/preflight-verification.json)
checks execution and read-only mounts; its intentionally incomplete cohort
is excluded from scientific evidence. Outputs will be retained in the durable
`h3-finalization/` and `h4-finalization/` sibling directories. Publication of
final reports requires their subsequent review; these jobs do not push Git.
