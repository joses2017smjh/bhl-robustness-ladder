# Native stereo–IMU execution ledger

The implementation and the recorded execution outcomes are separate from
scientific qualification. A complete runtime can preserve a negative result.

| Stage | Compute record | Execution | Scientific result |
|---|---|---|---|
| Native wrapper relink | Allocation step 21756762.31 | PASS; original native SLAM library retained | Interface verification only |
| Paired replay smoke | Allocation step 21756762.32 | 70/70 frame responses retained | Stereo 7/35; stereo–IMU 0/35, no inertial initialization |
| Complete paired original-data replay | Allocation step 21756762.39 | 900/900 responses retained across six scene/arm combinations | Stereo 0/150, 0/150, 122/150; stereo–IMU 0/150 in every scene; diagnostic reuse of prior simulation data |
| Fresh-seed native navigation smoke | GPU job 21757768 | Four 20-second episodes completed; native runtime exit 0 | NEGATIVE readiness gate; zero initialized/accepted stereo–IMU poses; no falls, contacts, or nonfinite episodes |
| Paired navigation development | Seeds 580000–580002, three routes, both arms | **0/18 run** | **UNRUN_AFTER_NEGATIVE_SMOKE**; not queued or qualified by these results |

The navigation smoke used seeds 570000 and 570001. Stereo accepted 89/89 and
88/88 native poses; stereo–IMU accepted 0/89 for both seeds and commanded zero
velocity throughout. Neither arm reached a goal within these short smoke
horizons. That 0/4 count is not the success rate of the unrun 18-episode,
40-second development cohort. The orchestration wrapper returned 1 specifically
for `FAILED_SCIENTIFIC_READINESS_GATE`; the native runtime returned 0 and the
campaign's problem list was empty.

No experiments or parameter tuning followed the negative navigation result.
The 640×480, 5 Hz ideal-camera scope remains unchanged. Camera-rate, exposure,
and motion-blur ablations remain unimplemented.

Evidence:

- [Implementation and measured interpretation](../../../docs/STEREO_INERTIAL_METHODS_2026-10-10.md).
- [Navigation summary and verified episode hashes](navigation-summary.json).
- [Navigation campaign result](stereo-inertial-nav-remote-v1/stereo-inertial-nav-smoke-20261010-21757768/campaign_result.json), [runtime completion](stereo-inertial-nav-remote-v1/stereo-inertial-nav-smoke-20261010-21757768/completion.json), and [orchestration gate outcome](stereo-inertial-nav-remote-v1/stereo-inertial-nav-smoke-20261010-21757768/remote-result.json).
- [Raw navigation publication receipt](stereo-inertial-nav-remote-v1/stereo-inertial-nav-smoke-20261010-21757768/publication.json). Archive bytes: 147,993,306; SHA-256: `b4e4f47be2826412d48063bd254c0425767522d64a7489a299aab6cae134887c`. Independently downloaded again during this review, and the hash matched.
- [Native replay execution](execution.json), [initialization analysis](initialization-analysis.json), and [40-test verification record](validation.json).
