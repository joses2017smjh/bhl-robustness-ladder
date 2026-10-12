# October 11 simulation completion evidence

Observed at 2026-10-12T00:39:19+00:00.

This report covers the native camera/fault controls and independent native-pose terrain confirmation. Hardware validation remains unrun; other research programs are outside this report.

Overall collection status: **INCOMPLETE**. Completed collection does not mean every method improved.

## Native camera and sensor-fault controls

Controller status: **INCOMPLETE**.

Completed shards recorded by the controller: 0. Complete scientific collection has not been verified.

Recorded failure: native summary does not bind every shard result

## Independent terrain confirmation

Controller status: **COMPLETE**.

Measured **480/480** declared cells; **0** unrun after a negative prerequisite gate.

**5/5** prespecified actor/controller combinations confirmed.

| Actor / controller | Candidate clean goals | Matched baseline clean goals | Falls / contacts | Gate |
|---|---:|---:|---:|---|
| dr-default-s0 / heading | 60/60 | 17/60 | 0 / 0 | PASS |
| dr-default-s0 / regulated | 60/60 | 17/60 | 0 / 0 | PASS |
| dr-default-s1 / heading | 60/60 | 1/60 | 0 / 0 | PASS |
| dr-default-s2 / heading | 60/60 | 1/60 | 0 / 0 | PASS |
| dr-default-s2 / regulated | 60/60 | 1/60 | 0 / 0 | PASS |

Matched baselines are reused across controllers: 180 unique baseline and 300 candidate episodes. The cohort uses 20 shared geometry/reset-seed clusters, fixed actors and terrain families.

[Verified final evidence](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/research-campaign-evidence-20261011/terrain-confirmation-final-20261011-5f09ee6f77d7.tar.gz). Scientific source SHA256: `8a2ec6a64f4cac0868cbf21fd018dacc2f7179daabc634724c5b38cd3bf8851f`.

[Machine-readable report and retained receipt inventory](./report.json). The observer verifies compact identities, final-artifact bytes and published summary consistency. Raw-trace rescoring was performed by the campaign collectors, not repeated by this observer.
