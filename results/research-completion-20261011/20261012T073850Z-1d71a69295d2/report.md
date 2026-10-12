# October 11 simulation completion evidence

Observed at 2026-10-12T07:38:50+00:00.

This report covers the native camera/fault controls and independent native-pose terrain confirmation. Hardware validation remains unrun; other research programs are outside this report.

Overall collection status: **COMPLETE**. Completed collection does not mean every method improved.

## Native camera and sensor-fault controls

Controller status: **COMPLETE**.

Measured **200/200** declared cells; **0** unrun after a negative prerequisite gate.

| Estimator/controller | Measured episodes | Clean goals | Falls | Contacts |
|---|---:|---:|---:|---:|
| lidar_inertial | 40 | 19 | 0 | 0 |
| permanent_stop | 40 | 0 | 0 | 0 |
| stereo | 40 | 0 | 0 | 0 |
| stereo_inertial | 40 | 0 | 0 | 0 |
| verified_lio_recovery | 40 | 0 | 0 | 0 |

Counts pool fixed conditions and two reset seeds for description; they do not establish population reliability. The final archive also retains the separately frozen offline health screen; this report does not claim an online health-control improvement.

[Verified final evidence](https://github.com/joses2017smjh/bhl-robustness-ladder/releases/download/research-campaign-evidence-20261011/native-controls-final-20261011-b48294ac61fb.tar.gz). Scientific source SHA256: `3628b95ced285660e27c0b3b0ca92a097415beaeedfa912da751832acc3b8282`.

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
