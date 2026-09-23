# Robotics portfolio audit and publication plan — September 20, 2026

Target: robotics ML / simulation engineering. The Apptronik simulation-architect
application provides a useful direction, not evidence of staff-level industry
experience. Its IDE notes file is not present on this HPC checkout.

Website: https://jose-sanchez-portfolio-com.vercel.app

GitHub: https://github.com/joses2017smjh

## 1. Prioritized audit

1. **Correct the evidence first.** Maze jobs are complete, not queued. Folding
   adaptation did not outperform its valid short-pants baseline. Keep Isaac
   PPO results separate from the older gait shown in MuJoCo. A missing asset
   in a rendering job is an infrastructure failure, not a failed policy.
2. **Make the hiring direction explicit.** Lead with robotics simulation,
   perception and evaluation. Do not present the Apptronik job title as a
   title already held. Keep language/retrieval projects below the robotics work.
3. **Create a profile entry point.** GitHub API inspection found no profile
   README repository and no pinned repositories. Feature three strong projects,
   not five loosely related ones.
4. **Make the first demo inexpensive and clear.** Replace the 14 MB BHL hero
   with a captioned 0.63 MB actual simulator GIF; link successes, controls and
   raw scores. The website should use MP4 with posters and reduced-motion support.
5. **State reproducibility limits.** Cluster paths and external weights/assets
   are not a portable install. CPU tests, simulation integration and policy
   evaluation are different layers. The BHL root has no license file; remove
   the unsupported MIT statement rather than inventing a license.

The public portfolio already uses Astro, responsive components, accessible
navigation, local fonts and reduced-motion handling. Preserve its visual
system; content and evidence alignment have higher value than a redesign.

## 2. Recommended featured projects

| Order | Project | Why it earns the slot | Evidence boundary |
|---|---|---|---|
| 1 | [BHL robustness ladder](https://github.com/joses2017smjh/bhl-robustness-ladder) | Simulation debugging, staged RL, multi-robot worlds, HPC provenance | Known-route evaluation, not SSD autonomy or hardware transfer |
| 2 | [Isaac robotic pruning](https://github.com/joses2017smjh/isaac-sim-pruning-workflow) | Sensor-conditioned control, USD scene work, independent grading, success/failure media | One known target and rigid-piece release, not wood fracture |
| 3 | [SPUR depth service](https://github.com/joses2017smjh/spur-depth-service) | Perception-to-service engineering, ONNX parity, GPU-specific latency | Synthetic validation, not field RMSE |

Keep folding as a clearly marked research/debugging case, not a fourth
successful flagship. Keep coursework and other ML projects available below
the featured section. Do not rename repositories: existing deep links and
résumé references are useful.

GitHub profile pinning is not exposed by the authenticated GraphQL mutation
schema inspected here. After publication: profile → Customize your pins →
select these three in order. A featured README is not the same as pinned repos.

## 3. Revised GitHub profile README

Exact publishable content: [github-profile-README.md](github-profile-README.md).
It states the role, links résumé/contact, embeds a real low-bandwidth demo and
introduces the three verified projects. It avoids aggregate policy counts
whose scope has shifted and unsupported folding-reproduction claims.

## 4. Reusable technical project README structure

1. **Name + outcome:** one sentence describing what the system demonstrably does.
2. **Demo:** one short captioned GIF or poster linked to MP4. Label simulator,
   controller type, playback speed and any oracle inputs.
3. **Problem / contribution / measured result:** distinguish upstream work from
   your contribution; link exact result files with denominator and scope.
4. **Quickstart:** real commands, prerequisites, external assets and a smoke
   test. Mark cluster-specific instructions explicitly.
5. **Architecture:** one small data-flow diagram only if it explains the system.
6. **Engineering decisions:** why this solver/controller/data split; alternatives
   and failure modes.
7. **Validation:** unit tests, integration checks, task metrics, controls and
   known failures. Separate these rather than reporting one blended score.
8. **Deployment and limits:** supported runtimes, hardware, licensing and
   reproducibility gaps; links to deeper protocol and job ledger.

The BHL README now follows this structure without removing historical evidence.

## 5. Revised portfolio structure and copy

**Hero:** Robot perception and learned control, tested in simulation.

**Role:** Seeking robotics ML and simulation engineering roles.

**Supporting line:** Three projects, from sensor inputs to measured outcomes.
Watch the robots, inspect the code, and see where the systems still fail.

**Primary CTA:** Explore selected work.

**Secondary CTAs:** Robotics résumé · Email.

Page order: hero → three featured visual case studies → further work →
background/résumé → hardware build log → contact.

Each case study leads with a demo and uses problem, solution, contribution,
result, limits and technologies. Deep technical detail goes to the repository.
BHL shows its known-route score and the separate multi-robot/inspection mission;
pruning shows the graded release and the tracking-loss stop; SPUR shows metric
depth, service/export evidence and synthetic-data limits.

Contact: **Hiring for robotics simulation or perception?**

I’m looking for a robotics ML or simulation engineering role where I can
build, test and improve robot-learning systems with a team.

## 6. GIF/demo shot lists

These are edit plans, not claims that every proposed composite has been filmed.

| Featured project | 0–5 seconds: starting state | 5–15 seconds: flow | 15–22 seconds: differentiator | 22–30 seconds: result |
|---|---|---|---|---|
| BHL | Show two inspection stations, route and dead-end branch | Humanoid visits stations and exits | Show wrong-branch rejection or a matched policy fall from the existing gait controls | Freeze the linked score; label oracle waypoints and the older gait |
| Pruning | Show the known spur and wrist RGB-D | Approach, gate release at 7.8 s, begin retreat | Contrast tracking-loss stop with no release | Show independent 17/17 grade and one-target/rigid-piece boundary |
| SPUR | Show synthetic RGB and calibration context | Request metric depth and reconstruct the point cloud | Show Torch/ONNX parity and GT-vs-predicted-mask comparison | Show named-GPU latency with precision; label synthetic-only validation |

Available new BHL GIFs: [inspection](gifs/weekend-inspection.gif),
[wrong-branch control](gifs/weekend-inspection-failure.gif),
[three robots](gifs/weekend-team3.gif). Their adjacent JSON sidecars record
source hashes, episode scores and speed. The wrong-branch GIF is an intentional
supervisor error, not a learned-policy failure. Existing
[multi-policy push footage](gifs/multi_race.gif) shows actual learned gaits
succeeding and failing under a shared disturbance.

No new weekend folding-policy video completed. Historical single-garment
success/failure examples exist in the folding checkout, but lack the new
strict camera/physics audit; do not present them as the adaptation result.
Replay-control movies are not learned-policy demonstrations.

## 7. Exact implementation changes

- BHL: result-first README, current maze/folding documentation, setup limits,
  profile source and this audit, plus labelled GIFs with provenance.
- Rendering: inspection evaluator accepts an explicit video route/sensor mode
  and rejects mismatched selections, so failure captures are not silently skipped.
- GitHub: publish the scoped BHL implementation/evidence and create the exact
  username profile repository; update descriptive metadata without renaming repos.
- Website: update hero/contact, three featured projects, current BHL/folding
  copy, local MP4/posters and evidence links. Preserve résumé content.
- Tests: BHL CPU suite 145 passed; website build and public-page checks recorded
  in the final handoff. Do not claim visual browser checks unless they execute.
- Do not cancel held user jobs, restart expensive folding arrays, commit
  checkpoints/environments, or publish unrelated dirty files.

## 8. Final recruiter-style quality check

- Can a reviewer identify the sought role and strongest project in seconds?
- Does each headline have a visible source, denominator and limit?
- Are simulation, real hardware, scripted supervision and learned control distinct?
- Are failures, oracle inputs and partial evaluations visible rather than hidden?
- Do repository, profile, website and résumé links agree?
- Does the site retain keyboard access, reduced-motion posters and mobile layout?
- Are installation limits and missing weights/assets stated?
- Are pins/manual actions and deployment verification reported honestly?

The credible pitch is **simulation and evaluation engineering with inspectable
results**, not a claim of solved general humanoid autonomy. Staff-level scope
still needs evidence of production ownership and technical leadership; project
polish must not substitute for that experience.
