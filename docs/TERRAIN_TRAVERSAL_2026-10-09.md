# Sensor-informed terrain traversal: frozen qualification and untouched trials

The executable campaign is [terrain_traversal_campaign.py](../scripts/bench/terrain_traversal_campaign.py).
It drives the actual frozen 12-DoF Berkeley Humanoid Lite learned actors through
the existing MuJoCo gait observation and PD-control path. This is simulation
evidence; it does not qualify physical terrain operation or estimated-pose SLAM.

## Why qualification comes first

The completed October 8 development study exposed a misleading outcome: the
DR-s1.5 ensemble governor had zero falls across 648 episodes and also **zero
successful traversals**, with mean progress about 0.159 m. Fewer falls alone
cannot establish useful locomotion. Existing rough-ground policies also fall
frequently. This campaign therefore tests basic physical-course competence
before spending compute on a sensor comparison.

Every trial retains the same five-metre course and 40-second matched horizon.
After one second of settling, the blind reference requests 0.30 m/s. A goal
requires crossing all five metres inside the corridor and remaining upright,
without side-wall contact or numerical failure, through the full 40 seconds.
Reaching the goal causes an evaluator stop command; survival continues until
the matched horizon. Stopping before the goal is a timeout, including when the
robot has zero falls.

## Frozen protocol

The three existing DR-default actors, training seeds 0, 1 and 2, are screened
separately on flat ground, one-centimetre steps and a three-degree ramp, each
with two fresh group seeds: 250000 and 250001. Each actor must succeed in all
six trials with zero falls, side-wall contacts and numerical failures. All
passing actors proceed; there is no best-seed selection. If none passes,
confirmation records `SKIPPED_NO_QUALIFIED_ACTOR` with zero episodes. The failed
screen remains a negative research result.

For each qualified actor, untouched groups 260000–260007 are paired across
three terrain fixtures and three arms, yielding 72 trials per actor and at
most 216 trials:

| Arm | Command inputs |
| --- | --- |
| `baseline` | Fixed 0.30 m/s forward reference; no terrain feedback |
| `lidar` | Fixed geometric governor using the current raw 3D scan and IMU attitude |
| `lidar50` | Same governor after retaining half the current scan's returns using a predeclared seed |

Each actor/arm target is at least 22/24 clean goals with zero observed falls,
side-wall contacts and numerical failures. The report retains every failure,
timeout and arm result. Actor and terrain groups are paired clusters; pooled
episodes do not certify safety or independent generalization. A scientific
target failure is separate from successful execution of the experiment.

Group seeds change physical terrain onset by at most 2.5 cm and sliding friction
by at most 0.02 around 0.8. They also define initial-state perturbations and
dropout randomness. Steps have five 36-cm treads at heights 1, 2, 3, 2 and 1 cm;
the ramp has three-degree ascent/descent over 1.8 m. The rigid geometric/contact
fixtures do not model loose substrates or deformable materials.

## Sensor and inference separation

Each instantaneous scan has 16 vertical rings and 64 azimuth rays, sampled at
5 Hz. Vertical angles range from -75 to -12 degrees, horizontal coverage is
120 degrees and maximum range is five metres. MuJoCo ray intersections produce
raw endpoints in the lidar frame; robot geometry is excluded. These are actual
scene intersections, rather than ideal height-field samples supplied to the
controller. There is no simulated rotating-scan motion distortion or deskew in
this bounded route.

IMU orientation removes roll/pitch to produce a gravity-aligned local map that
retains body heading. No robot translation, terrain identifier, terrain height
samples, evaluator collision labels or goal positions enter terrain inference.
The simulated IMU attitude is ideal; this campaign does not establish attitude
estimation robustness. The local map is rebuilt from each scan, without SLAM,
temporal accumulation or a world-frame pose input.

The 10-cm map uses endpoint-median height and supported local-plane slope,
residual roughness and neighboring height differences. Unknown cells remain
unknown. The governor examines a 20–70 cm lookahead corridor, ±18 cm wide. It
stops when fewer than 55% of its cells have supported terrain estimates. On
supported terrain it slows from 0.30 to 0.20 m/s above 2° slope, 3-mm roughness or
1-cm neighboring height differences, and stops at 8°, 15 mm or 25 mm respectively.
These fixed thresholds are engineering hypotheses, not fitted safety limits.

Only the evaluator accesses simulator translation and downward terrain rays
for course progress and local-surface sink. Falls retain the existing 0.78-rad
tilt or 0.25-m local-sink definitions. Side-wall contacts are checked at every
physics substep. The initial run does not include stereo fusion, estimated-pose
navigation, or gait retraining; those remain separately labeled routes.

## Commands and retained evidence

Create the protocol from the actual recursive upstream assets and three frozen
deploy configs before freezing/submitting the campaign:

```bash
python scripts/bench/terrain_traversal_campaign.py make-protocol \
  --upstream /path/to/Berkeley-Humanoid-Lite \
  --actor dr-default-s0:/path/to/dr-default-s0/deploy.yaml \
  --actor dr-default-s1:/path/to/dr-default-s1/deploy.yaml \
  --actor dr-default-s2:/path/to/dr-default-s2/deploy.yaml \
  --output /path/to/new/protocol.json
python scripts/bench/terrain_traversal_campaign.py smoke \
  --protocol /path/to/new/protocol.json --output /path/to/new/smoke
python scripts/bench/terrain_traversal_campaign.py screen \
  --protocol /path/to/new/protocol.json --output /path/to/new/screen
python scripts/bench/terrain_traversal_campaign.py run \
  --protocol /path/to/new/protocol.json --screen /path/to/new/screen \
  --output /path/to/new/confirmation
```

The two-episode smoke exercises the flat blind reference and ramp lidar governor.
It has `SMOKE_ONLY` scientific status and cannot qualify an actor. Scored runs
must be predeclared in `SLURM_JOBS.md` and executed on compute nodes. No GPU or
renderer is required. Plan one CPU task with four allocated CPUs and 8 GB RAM:
up to 20 minutes for smoke, one hour for the 18-episode screen, and four hours
for the conditional maximum 216 confirmation trials. These are resource caps,
not claimed measured runtimes.

Every episode saves JSON commands/outcomes at the gait update rate and an NPZ
with every raw scan, retained indices, timestamps, IMU attitude and inferred
height/slope/roughness/step maps. Separate evaluator-only dense vertical rays
measure height RMSE/coverage and geometric-hazard recall after each command
has been computed; their arrays are retained under `evaluator_truth_*` names
and never enter inference. Scan hashes bind the JSON to its raw sensor
evidence. Protocol receipts retain actor, source, asset and package hashes.
Confirmation recomputes screen eligibility from the raw episode records and
requires the same immutable protocol hash. Output directories must be new;
existing evidence is never silently overwritten.

For the immutable cluster launcher, add `--launcher` to `make-protocol`. This
emits one protocol containing the scientific settings, source-relative hashes,
six named deploy/checkpoint inputs and bounded smoke/run jobs. The entrypoint
accepts `--phase smoke|screen|run --protocol "$H34_PROTOCOL"`, uses
`H34_OUTPUT_DIR`, and resolves models from `H34_INPUTS_DIR` or the extracted
protocol's `inputs` sibling. The scored `run` performs the 18-episode screen and
conditional confirmation sequentially in the same frozen job, capped at five
hours combined; no unpinned result is imported between these stages. The
standalone `screen` phase remains available for a qualification-only run.

Unit checks exercise unknown-space stops, degenerate plane support, gravity
alignment, crossing-versus-survival scoring, stopped-robot timeouts, duplicate
screen groups and an inconsistent saved success flag. The real compute smoke
is still required before the scored screen.
