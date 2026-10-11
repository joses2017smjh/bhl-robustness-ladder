# Native run collection — 2026-10-09

actual simulated campaigns; smoke/replay/navigation/traversal kept explicit; no hardware validation.

PASS means the experiment's declared scientific gate passed; NEGATIVE is a completed failed gate; INCOMPLETE means missing, interrupted or unverifiable evidence. Smoke is labeled SMOKE_ONLY and never qualifies an actor. Replay has zero closed-loop episodes.

|Campaign|Job|Scientific status|Audit|Measured episodes|Evidence|
|---|---:|---|---|---:|---|
|lio-navigation|21741509|PASS|PASS|9|[verified collection](collected/lio-navigation-21741509/publication-observation.json)|
|terrain-traversal|21742786|NEGATIVE|PASS|18|[verified collection](collected/terrain-traversal-21742786/publication-observation.json)|
|lio-replay|21743631|PASS|PASS|0|[verified collection](collected/lio-replay-21743631/publication-observation.json)|
|orb-replay|21739893|NEGATIVE|PASS|0|[verified collection](collected/orb-replay-21739893/publication-observation.json)|
|stereo-navigation|21744334|NEGATIVE|PASS|9|[verified collection](collected/stereo-navigation-21744334/publication-observation.json)|
|stereo-navigation-smoke|21744333|SMOKE_ONLY|PASS|2|[verified collection](collected/stereo-navigation-smoke-21744333/publication-observation.json)|

Each unique result directory contains source/job/completion hashes, derived audits and original outcome receipts. Raw archives above50MiB are linked from the verified prerelease; smaller raw archives are retained beside their collection. Native binaries and private frozen input packs are excluded.
