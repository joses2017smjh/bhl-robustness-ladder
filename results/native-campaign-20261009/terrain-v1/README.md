# Terrain traversal campaign, October 9, 2026

The frozen [protocol](protocol.json), [input inventory](manifest.json),
[intake](intake.json) and [full relocation verification](freeze-verification.json)
describe actual simulation inputs, rather than a proposed result.
[Sixteen unit checks passed](terrain-tests.xml).

The fresh smoke, Slurm **21740514**, completed two real frozen-gait physics
episodes. Its [raw audit](terrain-smoke-20261009-21740514/raw-smoke-audit.json)
recomputed all trace outcomes and all 400 map-frame height supports/errors.
The [complete raw archive](terrain-smoke-20261009-21740514/outputs.tar.gz) retains
every scan, IMU orientation, inferred/evaluator map, command and outcome.

| Smoke trial | Full-horizon progress | Falls / wall contacts | Outcome |
| --- | --- | --- | --- |
| DR-default-s0, flat, blind baseline | 5.0169 m | 0 / 0 | Goal and full 40-second survival |
| DR-default-s0, ramp, LiDAR governor | 1.5824 m | 0 / 0 | Timeout; observed geometry hazards caused stop commands |

These two episodes are **smoke only**, and do not qualify any actor. The
top-level [smoke result](terrain-smoke-20261009-21740514/campaign_result.json)
correctly records zero qualified actors. A known metadata error in the nested
`smoke/campaign_result.json` inside the raw archive lists its evaluated actor
under `qualified_actors`; that field is ignored scientifically. Qualification
requires the separate six-trial-per-actor screen, recomputed from episode
records, and cannot be granted by the smoke output.

The raw maps include endpoint returns from terrain and side walls. Later ramp
maps contain side-wall geometry as the robot heading changes. Their height
error therefore measures all sampled map surfaces; it is not a ground-only
accuracy claim or a successful ramp-traversal claim. No thresholds were tuned
after inspecting the smoke.

The scored job **21740636** was submitted after the successful smoke using
the same frozen archive. It will run the 18-episode controller qualification
screen first, then at most 216 untouched paired trials if actors qualify.
If no actor qualifies, the result must be negative with zero confirmation
episodes. Running or scheduled work supplies no measured confirmation result.

See [the scientific protocol and implementation limits](../../../docs/TERRAIN_TRAVERSAL_2026-10-09.md).
