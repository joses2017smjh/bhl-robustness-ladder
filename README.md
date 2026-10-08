# Humanoid Robustness Ladder

**Train humanoid policies in Isaac Lab and test what they actually accomplish.**

I built the task curricula, sensor adapters, MuJoCo evaluation harnesses and
Slurm checkpoint gates around the upstream Berkeley Humanoid Lite. The project
studies locomotion, goal reaching and multi-robot coordination, with scored
recordings and negative controls.

[Watch the demo](results/weekend-20260919/inspection-maze.mp4) ·
[Results](docs/WEEKEND_RESULTS_2026-09-20.md) ·
[Success/failure gallery](docs/GALLERY.md) ·
[Portfolio case study](https://jose-sanchez-portfolio-com.vercel.app/projects/bhl-robustness-ladder/) ·
[Roadmap and stretch goals](docs/ROADMAP.md)

<p align="center">
  <a href="docs/gifs/random-maze-humanoid-sensors.gif"><img src="docs/gifs/random-maze-humanoid-sensors.gif" width="100%" alt="Top view of a randomized 6x6 maze: the 22-DoF humanoid (arms and legs) turns in place and walks forward along an orange breadcrumb path to a green goal. Under the view, simulated IMU gyro and accelerometer traces. Right: the brake's 8x8 ray depth, lidar sectors and the lidar-built map with the A* plan; then the stereo rig's RGB, 160x120 rendered depth, left-eye optical flow, and the IMU attitude filter against truth. Badge: GIF at 5x."></a><br>
  <sub>The 22-DoF humanoid in a maze it has never seen: a new maze every seed, so the route cannot be memorized. Learned gait
  (one qualified turning checkpoint); scripted A* on a map the robot builds from its own lidar; it turns in place, then walks forward.
  Oracle pose and goal. <b>12/12</b> never-seen hard mazes, no falls, no wall contact; the 12-DoF biped runs the same mission at <b>24/24</b>.
  Right-hand column and IMU strip are <b>display only</b>: stereo RGB, rendered depth, optical flow and a simulated
  IM10A IMU with datasheet noise. Click for full size &middot;
  <a href="docs/gifs/random-maze-explore-sensors.gif">biped version</a> &middot; <a href="docs/RANDOM_MAZE.md">tables and stress tests</a></sub>
</p>

## Highlights

<table>
  <tr>
    <td width="50%" align="center">
      <img src="docs/gifs/navgym-learned-maze.gif" width="420" alt="Top view of a randomized 6x6 maze. The biped walks to the green goal under forward-speed and turn commands from a learned navigation policy; side panels show ray depth, lidar sectors and the lidar map, with no planner. Badge: GOAL REACHED 51.3 s, GIF at 4x."><br>
      <sub><b>Exploratory.</b> The planner is replaced by a PPO navigation policy trained in a 2-D gym and deployed unchanged on the physics biped (no planner in the loop; oracle pose and goal).
      <b>10/12</b> never-seen 6×6 mazes, 0 falls (A*: 12/12); the policy missed its predeclared gym bar by one clause, and this clip's maze is inside its training-seed range.</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/gifs/isaac/maze_both_panels.gif" width="420" alt="Isaac Sim top view of a biped walking a corridor to a button; side panels show its stereo ray depth, the pooled 4x4 it reads, and 36 lidar sectors."><br>
      <sub>Isaac Sim: a learned PPO policy that reads its own 36 lidar sectors and pooled ray depth (no RGB) reaches the button; oracle waypoint heading.
      This checkpoint <b>32/32</b>, all 12 checkpoints <b>380/384</b> with training observation noise on. Blind policies score 95/96, so this is not a sensor-benefit claim.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/gifs/inspection-maze-panels.gif" width="420" alt="The hero inspection episode from above, with robot-eye view, 8x8 stereo ray depth and 36-sector lidar panels. Badge: COMPLETED 17.36 s."><br>
      <sub>Two ordered inspection stops and the exit of a two-turn maze, with the lidar and ray depth that drive its speed brake. Learned gait, oracle waypoints. <b>3/3</b>; both failure controls <b>0/3</b> (<a href="docs/gifs/weekend-inspection-failure.gif">wrong-branch control</a>).</sub>
    </td>
    <td width="50%" align="center">
      <img src="docs/gifs/weekend-team3.gif" width="420" alt="Three humanoids in one MuJoCo world wait at a red airlock door, cross, and meet on green rendezvous discs. Caption: COMPLETED, learned gait plus oracle team supervisor, 1.1x."><br>
      <sub>Three robots in one world: inspect, wait, cross the airlock, rendezvous. Learned gait, oracle team supervisor. <b>5/5</b>; both controls <b>0/5</b>.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="docs/gifs/dr_pair.gif" width="420" alt="Two bipeds given the same strafe command in MuJoCo. Left, randomized, walks. Right, un-randomized, falls."><br>
      <sub>Isaac-trained PPO, scored in MuJoCo. The highest-training-reward policy (right, no randomization) falls in <b>21/90</b> episodes; the default randomization <b>0/90</b>.</sub>
    </td>
    <td width="50%" align="center">
      <sub><a href="docs/GALLERY.md">Every clip, successes and failures</a> ·
      <a href="docs/STATUS.md">status of every workstream, negatives included</a> ·
      <a href="SLURM_JOBS.md">job ledger with predeclared rules</a></sub>
    </td>
  </tr>
</table>

<sub>Learned = PPO policy, frozen at evaluation. Scripted = hand-written planner or supervisor.
Oracle = simulator ground truth (pose, goal, waypoints) given to the controller.</sub>

## Results and limits

| Experiment | Measured result | What the result covers |
|---|---|---|
| Confirmed locomotion robustness | **3,600** episodes; flat falls **77/360 → 0/360**, push falls **341/360 → 13/360** (randomization off → default) | Five settings × three training seeds × two conditions; frozen 12-DoF biped in MuJoCo; shared command/reset seeds; [full verdict](campaigns/20261006-confirmatory/results/dr_verdict.json) |
| Confirmed learned navigation (NavGym v5) | Each actor reaches **44–46/48** nominal goals and **43–46/48** under 35% packet dropout; zero-fall gate **NEGATIVE** | Three actors + matched A* on 48 fresh layouts per condition, **384** episodes; learned gait + scripted brake, oracle pose/goal; [per-actor verdict](campaigns/20261006-confirmatory/results/verdict.json) |
| Goal-reaching PPO | **379/384** successful first episodes in the final stage; all **36** curriculum jobs completed | Four sensor conditions × three seeds × 32 environments; fixed route, oracle waypoints, observation noise disabled |
| Inspection mission | **3/3** nominal; wrong-branch and complete-outage controls **0/3** each | Frozen gait and supervised navigation in MuJoCo; no object recognition |
| Two-/three-robot airlock | **5/5** each; both negative controls **0/5** each | Shared physical world and explicit synchronization; no carrying or newly trained MARL |
| Separate dual-arm folding study | Baseline short pants **8/24**, adapted seed 1 **3/24**; **5/12** evaluation cells completed | Official ever-triggered checker; no demonstrated adaptation improvement |

**2026-10-07 confirmatory results:** the locomotion campaign completed its flat-ground replication gate (**PASS**). The navigation campaign completed all 384 episodes but missed its predeclared zero-fall gate (**NEGATIVE**): all three learned actors fell in nominal sensing, and one also fell under packet dropout. A* observed no falls. The smaller demonstration cohorts above remain historical results. [Protocols and saved verdicts](campaigns/20261006-confirmatory/) and the [job ledger](SLURM_JOBS.md) retain the evidence and negative findings.

The [dated report](docs/WEEKEND_RESULTS_2026-09-20.md) links per-seed evidence.
[Published records](docs/PUBLIC_EVIDENCE.md) retain scores, episode outcomes
and traces while omitting new private machine paths and scheduler receipts.
[Folding clips](docs/FOLDING_MEDIA.md) distinguish historical checker success
from the new adaptation failure. A completed rendering job is not a successful fold.

## Why this project

Training reward can hide task failure. The evaluation harness measures arrivals,
dwells, falls and physics-step contacts, then checks matched failure controls.
The [historical findings](docs/FINDINGS.md) also preserve corrected and retracted
claims, including an ice-patch placement error and a camera quaternion mismatch.

Key engineering decisions:

- **Gate checkpoint promotion:** each curriculum stage loads the exact checkpoint
  that passed the preceding task evaluation.
- **Separate training from evaluation:** Isaac supplies parallel PPO training;
  selected exported gaits run through the upstream controller in MuJoCo.
- **Validate sensor input:** timestamps, stale packets, invalid depth and IMU
  conventions have explicit contracts and tests. RGB stereo matching remains a
  separate component from the paired ray-depth policy observations.
- **Keep the denominator:** first-episode scores avoid inflating success by
  repeatedly completing easier episodes. Partial folding arrays stay incomplete.

## Portable simulation regression replay

[Measured replay results](results/resume-revamp-20261007/replay_report.json) ·
[Raw timing samples](results/resume-revamp-20261007/policy_timing.csv) ·
[Navigation pilot](results/resume-revamp-20261007/navigation_development.json)

The October 7 revamp packages an actual trained **12-DoF biped** checkpoint,
upstream controller, deployment configuration, MJCF and referenced meshes with
30 file hashes and pinned runtime versions. A relocated bundle ran from `/tmp`
with `PYTHONPATH` unset; it uses no Isaac installation or GPU inference.

Five seeded nominal trajectories reproduced exactly on the same host, and
**40/40 injected sensor, action and contact-scoring regressions** were detected.
A changed trajectory is a regression relative to the reference; these gates do
not establish safe behavior under faults or cross-host determinism.

Caching immutable validation limits preserves **208/208** tested decisions and
reduces median observation/inference/validation time **49–69%** in two
counterbalanced 5,000-call comparisons. A separate 10,000-call warm measurement
reports **p99 1.09 ms**, max 16.28 ms and **0 observed misses** against the
configured **40 ms policy period** on an Intel Xeon Platinum 8480CL host. Physics,
networking, startup and real-time scheduling are excluded; the lower-level
4 ms control deadline is outside this measurement.

```bash
# Supply an actual 12-DoF, no-history deployment and its upstream assets.
PYTHONPATH=src python scripts/bench/sim_regression.py \
  --deploy path/to/deploy.yaml --checkpoint path/to/policy.onnx \
  --upstream path/to/Berkeley-Humanoid-Lite --out /tmp/bhl-replay-bundle
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python /tmp/bhl-replay-bundle/replay.py --bundle /tmp/bhl-replay-bundle \
  --out /tmp/bhl-replay-results --calls 10000
```

The builder refuses missing models; replay verifies source, asset and runtime
hashes before physics. Output directories cannot be reused. This integration
suite complements the existing unit tests. A manual CI template is available at
[`ci/replay-workflow.yml.example`](ci/replay-workflow.yml.example); its artifact
intake passed local checks, and the workflow has not been executed on GitHub. Bundle generation and relocation
were tested with the existing environment; a fresh dependency installation and
GitHub-hosted execution remain unverified.

The separate fresh navigation pilot evaluated the existing yaw filter on one
actor, two layouts and two sensing conditions (**8 episodes**). Both baseline and
candidate reached **2/4** goals with no falls; command flip rates fell about
**53%**. This establishes reduced chatter in the pilot, without establishing a
goal-success or fall-rate improvement. The consumed layouts are development data.

## Quickstart: CPU checks

Use Python 3.11 on Linux. No Isaac installation or policy weights are needed for
the unit tests. The nested upstream assets supply two robot-configuration fixtures.

```bash
git clone --recurse-submodules https://github.com/joses2017smjh/bhl-robustness-ladder.git
cd bhl-robustness-ladder
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
OMP_NUM_THREADS=2 PYTHONPATH=src python -m pytest -q tests
```

The October 7 full CPU suite returned **2,105 passed, two failed, three skipped**.
Both failures were stale whole-file assertions that rejected existing extra task
registrations. Narrow assertion fixes preserved the frozen controller behavior
and the exact registration contract; both affected tests then passed. The full
suite was not rerun after those assertion fixes.
[Validation receipt](results/resume-revamp-20261007/final_validation.json).

The publication snapshot previously had **146 passing CPU tests** in Python 3.11. The fresh-environment installation above has not been independently
validated; [exact versions and prerequisites](docs/REPRODUCIBILITY.md) are recorded.
Unit tests cover sensor validity, stereo geometry, task gates, contact scoring,
quaternions, garment splits and robot invariants. They do not certify GPU rendering
or policy performance.

## Architecture and stack

![Isaac training, ONNX export, MuJoCo scoring and recorded results](docs/img/pipeline.svg)

| Component | Responsibility |
|---|---|
| `src/bhl_robust/tasks/` | Isaac task configurations, rewards and curriculum stages |
| `src/bhl_robust/eval/` | MuJoCo missions, contacts, sensor supervision and recording |
| `src/bhl_robust/cloth/` | Humanoid sorting primitives and folding-data validation |
| `scripts/bench/`, `tests/` | Task gates and CPU regression checks |
| `slurm/` | Cluster launchers, dependencies and checkpoint promotion |

Python · PyTorch · Isaac Lab / Isaac Sim · MuJoCo · OpenCV · ONNX Runtime ·
Apptainer · Slurm. The project maintains separate Isaac Lab 2.3.2 / Sim 5.1 and
Lab 3.0 beta / Sim 6.0 environments; their compatibility limits are documented.

## Training, deployment and remaining work

GPU training requires external Isaac installations, robot assets and cluster
configuration. MuJoCo policy evaluation also requires exported weights that are
not bundled. The Slurm scripts contain site-specific paths and account settings;
follow [reproducibility](docs/REPRODUCIBILITY.md) before adapting them.

This is a simulation research repository; no hosted interactive simulator or
hardware deployment is claimed. Open work includes held-out layouts, reducing
privileged navigation inputs, completing folding evaluations and validating
stable terminal folds. [Publication checks](docs/PUBLICATION_CHECK_2026-09-20.md) and the
[campaign protocol](docs/WEEKEND_CAMPAIGN.md) record the current boundaries.

## Author and license

Jose Sanchez Gonzalez — seeking **ML/AI engineering and robotics roles**.
[Portfolio](https://jose-sanchez-portfolio-com.vercel.app) ·
[Résumé](https://jose-sanchez-portfolio-com.vercel.app/resume.pdf) ·
[Email](mailto:josejsanchez20172@gmail.com)

No repository-level license has been selected. Upstream code, models and assets
retain their own licenses; see the nested upstream repositories.


<!-- bhl-confirmation-results-2026-10-06 source_sha256=d74574dbc07e93b48655168268d22b84feb481122b232bf220c31b7e227b7b5e -->
### Recorded confirmation outcomes — campaign 2026-10-06
Append-only evidence update at `2026-10-06T12:56:38.990567+00:00`. This dated record supersedes the earlier queued-status paragraph for this campaign; historical experiment results above remain their original records.
**Locomotion: PASS. Navigation: NEGATIVE.** Recorded episode files: locomotion **3600/3,600 planned**; navigation **384/384 planned**. File counts alone do not certify valid episodes.
| Randomization rung (3 training seeds) | Flat falls / reported episodes | Push04 falls / reported episodes | Flat mean displacement |
|---|---|---|---|
| dr-off | 77/360 | 341/360 | 1.408 m |
| dr-s0.5 | 4/360 | 268/360 | 1.849 m |
| dr-default | 0/360 | 13/360 | 1.984 m |
| dr-s1.5 | 0/360 | 0/360 | 1.767 m |
| dr-aggressive | 60/360 | 60/360 | 0.129 m |

Flat denominators are 360 per rung when complete (3 policies × 6 commands × 20 shared reset seeds), with another 360 disturbed episodes per rung. The flat replication rule requires zero falls for each default policy and fewer falls than its matched unrandomized policy. Push04 is descriptive. Full per-training-seed/command tracking, displacement and paired outcomes remain in `dr_verdict.json`; surviving-step errors alone cannot rank failed policies. When the verdict is INCOMPLETE, reported partial counts do not establish a validated complete experiment.

| Navigation actor | Condition | Goals / reported episodes | Clean | Falls | Complete evidence |
|---|---|---|---|---|---|
| armV5-s8 | nominal | 44/48 | 44/48 | 1/48 | True |
| armV5-s8 | drop35 | 43/48 | 43/48 | 0/48 | True |
| armV5-s9 | nominal | 46/48 | 44/48 | 1/48 | True |
| armV5-s9 | drop35 | 44/48 | 44/48 | 2/48 | True |
| armV5-s10 | nominal | 44/48 | 44/48 | 2/48 | True |
| armV5-s10 | drop35 | 46/48 | 45/48 | 0/48 | True |
| astar | nominal | 44/48 | 39/48 | 0/48 | True |
| astar | drop35 | 45/48 | 42/48 | 0/48 | True |

Each complete navigation cell contains 48 shared held-out 6×6 layouts. Each final learned actor must meet both ≥40/48 nominal and ≥36/48 drop35 goals with zero falls; all actors and the matched A* control are reported. Clean means no wall contact at any physics step. An incomplete cell cannot establish validated success, regardless of its reported partial goal count.

Scope: simulation only. Isaac-trained frozen 12-DoF PPO gait; navigation uses a learned PPO actor with oracle pose/goal and a scripted reactive speed brake; A* is the reference. Shared layouts, commands and reset seeds are clustered observations, not independent trained policies. Neither PASS nor a zero observed fall count certifies hardware deployment.

Immutable JSON sources: `/nfs/hpc/share/sanchej7/Computer_Vision/project_results_upgrade/humanoid_confirmatory/dr_verdict.json` and `/nfs/hpc/share/sanchej7/Computer_Vision/project_results_upgrade/humanoid_confirmatory/verdict.json`. Locomotion JSON SHA256: `7bd2629ae71b8bddd61e1b4c9d88c85d1abd5116f145d00f3783755bd82fcc04`; navigation JSON SHA256: `7b9a1901f1febd48e6ff4849ecbd26a2ca6c15f36326da5be378d2020b9b8393`. Combined source SHA256: `d74574dbc07e93b48655168268d22b84feb481122b232bf220c31b7e227b7b5e`. Executable protocol and publisher are archived in `campaigns/20261006-confirmatory`.
