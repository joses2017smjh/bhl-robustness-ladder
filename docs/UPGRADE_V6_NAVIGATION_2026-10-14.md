# V6 navigation: two extra goals, fewer clean completions

**Outcome audit published October 14, 2026.** The selected budgeted-recovery
controller reached **288/288 goals**, versus **286/288** for the existing
recovery reference, with **zero falls in both**. Clean goals decreased from
**282 to 276** and contact episodes increased from **5 to 12**. This is a
tradeoff in the measured cohort, not a general navigation improvement.

The frozen gate passes because it requires goal/fall nonregression and a strict
aggregate gain against the declared reference; it does not gate on clean goals.
The selected controller is `budget_adaptive`, selected on development data before
confirmation. It gives the learned navigation actor at most one third of the
180-second episode budget before handing off to scripted A* on the observed
lidar map. Recovery also responds to stalls and repeated positions. The baseline
recovery reference, `hybrid_adaptive`, uses the existing watchdog without that
budget deadline.

## Complete confirmation

The retained cohort contains **1,152 episodes**: four methods × three fixed
navigation actors × two sensing conditions × 48 shared maze seeds
**2110000–2110047**. The conditions are nominal sensing and 35% exteroceptive
packet dropout. The body uses a frozen 12-DoF gait; navigation actors are
`armV5-s8`, `armV5-s9`, and `armV5-s10`.

| Method | Goals / 288 | Clean goals / 288 | Falls | Timeouts | Contact episodes | Episodes using recovery |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 218 | 206 | 57 | 13 | 53 | 0 |
| Yaw-only control | 256 | 250 | 13 | 19 | 21 | 0 |
| Existing recovery reference | 286 | 282 | 0 | 2 | 5 | 26 |
| Selected budgeted recovery | 288 | 276 | 0 | 0 | 12 | 183 |

Each method/policy/sensor combination uses the same 48 generated maze layouts.
Repeated actors and sensor settings are clustered observations, not 288
independent environments. Zero observed falls does not establish general safety
or hardware reliability.

**Oracle pose and goal are used.** Recovery reads the observed lidar occupancy
map, not the simulator's true maze topology. The selected controller records
**zero world-model influence steps**: this study supports a scripted recovery
result around learned navigation and walking, not a learned-world-model gain.
It is separate from native FAST-LIO2 estimated-pose experiments and from the
22-joint maze GIF in the main README.

## What was independently checked for publication

The local audit read every original confirmation record and checked its SHA256
against the completion receipt. It verified all 24 completion receipts, all 48
confirmation binding files, all 84 development binding files, and the declared
source/model hashes. Published outcome extracts were compared against raw
records, including goal status, layout identity, falls, contacts, and recovery
exposure. No simulator jobs or training ran for this documentation update.

The [compact snapshot](../results/upgrade-v6-navigation-20261014/) contains:

- [1,152 outcome extracts](../results/upgrade-v6-navigation-20261014/confirmation-episodes.json), with the original raw-record hash for every episode;
- the [protocol](../results/upgrade-v6-navigation-20261014/protocol.json), [development gate](../results/upgrade-v6-navigation-20261014/development-gate.json), and [navigation verdict](../results/upgrade-v6-navigation-20261014/navigation-verdict.json);
- [frozen scientific source](../results/upgrade-v6-navigation-20261014/source/), [original source/model hashes](../results/upgrade-v6-navigation-20261014/source-hashes.json), and [publication audit](../results/upgrade-v6-navigation-20261014/audit.json);
- a standard-library recount command, runnable from the repository root:

```bash
python3 results/upgrade-v6-navigation-20261014/verify_snapshot.py
```

The verifier checks snapshot hashes, cohort completeness, common layout identity,
and all table counts. Absolute runtime paths in JSON were normalized to logical
campaign paths. The environment-specific `common.py` is represented by its
original hash. Full simulator traces, model binaries, and external environment
assets remain outside this compact snapshot: the command reproduces the count
audit, not the physics experiment. The original report and protocol hashes are
retained in `audit.json`.

## The overall campaign is still incomplete

The V6 report remains **INCOMPLETE** because its separate terrain confirmation
found unmatched physical terrain assets. Its harder-terrain stage was therefore
skipped. The later V8 terrain development gate also failed; confirmation and
challenge stages were skipped, and the optional learned checkpoint was
unavailable. These findings do not support a terrain-learning improvement.

The complete navigation sub-study above does not change those terrain verdicts.
The [October 11 native-pose terrain confirmation](TERRAIN_CONFIRMATION_2026-10-11.md)
is a different, previously published campaign with different controllers and
course families.
