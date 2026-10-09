# Terrain traversal v3: exact-source preempt revision

Fresh smoke **21742782** and scored run **21742786** were submitted with explicit no-requeue on `preempt`, CPU2/8GB. Scored execution is held `afterok:21742782`. Submission is not a scientific result.

The archive is `1f04b186e62a31bfa27547375124b85f7bc8a46b688fc92577ea108d7d6293ed`. All **834** source/model input files were independently extracted and verified; the manifest is byte-identical to v1/v2. Only partition and explicit no-requeue change from v2. The scientific protocol, actor cohort, sensor settings, map/controller thresholds and independent outcome criteria remain fixed.

The combined scored job first runs18 blind qualification episodes. Only actors achieving6/6 clean full-horizon goals qualify. All qualified actors enter72 confirmation episodes each (maximum216); if none qualifies, the scientific result is negative and confirmation stops. Stopping short of the5m goal is a failure even without falls.

The earlier two-episode v1 smoke execution passed, but its LiDAR-ramp episode stopped at1.58m and timed out; this negative outcome is preserved in [v1](../terrain-v1/README.md). Its nested smoke-only `qualified_actors` field is not screen eligibility; the top-level smoke field is correctly empty. No threshold was adjusted after that smoke.

V2 jobs21740943/21740946 remained PENDING and were cancelled with a PENDING-state guard before this revision; see [cancellation receipt](../terrain-v2/pending-cancellation.json). Preemption will leave incomplete evidence rather than restarting into exclusive receipt directories.
