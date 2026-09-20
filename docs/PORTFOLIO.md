# GitHub and portfolio audit — September 20, 2026

Target roles: **ML/AI engineer and robotics engineer**.
[GitHub](https://github.com/joses2017smjh) ·
[Portfolio](https://jose-sanchez-portfolio-com.vercel.app).

## 1. Prioritized audit

1. **Publish the BHL proof behind the existing portfolio links.** The prepared
   BHL results were local. The public version now pairs a short simulator demo
   with exact results, controls and limitations. It excludes raw workspace
   prefixes, machine assignments and submission commands.
2. **Correct completion claims.** Maze: 36/36 jobs and gates passed. Folding:
   5/12 evaluation cells completed; adaptation improvement is unproven. The
   repaired media gate rendered a failed fold. Other training jobs remain
   failed/running. Queue absence alone is not success.
3. **Make the role and contribution obvious.** Align the profile and BHL README
   with ML/AI engineering and robotics; distinguish authored infrastructure
   from upstream robots/models. Lead with four complementary projects.
4. **Reduce setup friction.** Add a CPU test dependency file, clone/install/check
   commands, exact tested versions and external-asset requirements. A fresh
   installation remains unverified; the existing environment passed 145 tests.
5. **Finish the public entry points.** Keep the deployed visual portfolio and
   direct README/source links. Native GitHub pins still require a manual
   profile action. BHL's root license still needs the owner's selection.

## 2. Recommended featured projects and pins

| Order | Project | Engineering signal | Evidence boundary |
|---|---|---|---|
| 1 | [Vision-guided pruning](https://github.com/joses2017smjh/isaac-sim-pruning-workflow) | RGB-D control, dual-ToF release checks and independent recording grading | Controlled known-target episode; rigid-piece surrogate, not wood fracture |
| 2 | [SPUR depth service](https://github.com/joses2017smjh/spur-depth-service) | ML inference API, multi-view refinement, ONNX export and parity checks | Synthetic validation; field accuracy unverified |
| 3 | [Humanoid Robustness Ladder](https://github.com/joses2017smjh/bhl-robustness-ladder) | Staged RL, simulation debugging, CPU tests and HPC evaluation gates | Fixed routes and privileged inputs; no hardware walking claim |
| 4 | [MetaNaviT](https://github.com/joses2017smjh/MetaNavT) | Retrieval evaluation, provenance and reviewable file operations | Published fixture benchmark uses hash embeddings and overlap reranking |

For a robotics-specific application, move BHL above SPUR. Keep folding as a
supporting research/debugging case. Repository names stay stable to preserve
résumé and project links. Existing GitHub descriptions, topics and portfolio
homepages already describe the demonstrated work.

Profile → **Customize your pins** → select these four. The inspected API does
not provide a repository-pinning mutation; a README feature is not a native pin.

## 3. Revised profile README

[Exact Markdown](github-profile-README.md) includes a one-line introduction,
the confirmed target roles, actual technologies, four selected projects,
a genuine demo and direct portfolio/résumé/LinkedIn/email links. Each project
links to its source and states the boundary of its evidence.

## 4. Reusable technical README structure

[Copyable template](PROJECT_README_TEMPLATE.md): outcome → demo → problem and
contribution → measured results → quickstart → architecture/decisions →
validation/deployment/limits → author/license. Its uppercase placeholders are
intentional template fields and must be replaced before use.

The [BHL README](../README.md) implements this structure. The other featured
READMEs already received evidence and setup revisions; do not overwrite those
with a generic template. BHL's long historical case studies remain accessible
through Findings, the gallery and the detailed reports.

## 5. Portfolio structure and copy

The deployed site already leads with **“I build perception and simulation for
robots.”** Keep the visual case studies, four featured projects, brief outcomes,
README/source links and résumé/contact routes. GitHub supplies the deeper proof.

Suggested role line: **Seeking ML/AI engineering and robotics roles.**

BHL card copy: **Humanoid policies, tested beyond training reward.** Built Isaac
Lab curricula, checkpoint gates and MuJoCo evaluation. The fixed-route Isaac
benchmark reaches 379/384 first-episode successes; separate recorded missions
show learned gait plus supervised navigation and matched failure controls.

Keep the existing responsive layout, video posters, native playback controls,
reduced-motion behavior and pause control. No redesign or new website deployment
is required for this BHL publication. The prior site validation records its
production build and mobile/desktop browser checks in the portfolio repository.

## 6. Demo shot lists, 15–30 seconds each

These are recording/edit recommendations; they are not new benchmark runs.

| Project | 0–5 s: starting state | 5–15 s: flow | 15–22 s: differentiator | 22–30 s: result |
|---|---|---|---|---|
| Pruning | Known spur and wrist RGB-D | Approach, gated release, retreat | Contrast tracking-loss stop | Show the selected recording's 17/17 grade and surrogate-release limit |
| SPUR | Synthetic RGB and camera calibration | Request depth and display point cloud | Torch/ONNX parity and mask ablation | Show published synthetic accuracy with dataset/hardware labels |
| BHL | Stations, route and dead-end branch | Humanoid visits stations and exits | Matched wrong-branch control | Show 3/3 nominal vs 0/3 control; label oracle map and older gait |
| MetaNaviT | Research files and a concrete question | Retrieve sources and inspect provenance | Preview a file operation before approval | Show fixture benchmark: Recall@50 0.938; 136 queries, 61 files |

BHL's [gallery](GALLERY.md) links inspection, wrong-branch and three-robot media.
The [folding gallery](FOLDING_MEDIA.md) links five clips: historical checker
success/failure, new adaptation failure, and its two wrist-camera views.
Historical folding success is ever-triggered; a terminal fold was not recorded.
Do not edit clips from different checkpoints into an implied controlled comparison.

## 7. Exact implementation changes

- `README.md`: concise entry point with hero GIF, contribution, scored results,
  CPU setup, architecture, tradeoffs and current limits.
- `requirements-test.txt`, `docs/REPRODUCIBILITY.md`: tested dependency versions,
  pinned upstream fixtures and explicit fresh-install limits.
- `docs/FOLDING_MEDIA.md`, five GIF/PNG/JSON sets, gallery/campaign/cloth/job
  documentation: reconcile the replacement rendering gate with the preserved
  blocked array; label historical and new checkpoints separately.
- `results/weekend-20260919/`: portable evidence, original/public hashes and an
  independent publication audit; retained numeric fields match the raw records.
- Profile Markdown, this audit and the reusable README template: exact copy for
  the confirmed roles, recommended pins and all four demo sequences.
- Publish from an isolated checkout based on the public branch. Preserve the
  original research checkout, raw evidence and unrelated uncommitted work.

## 8. Recruiter-style quality check

- Role, contribution, actual demo and results are visible near the top.
- Headline scores have denominators, protocols and linked evidence.
- Learned policies, scripted supervision, oracle inputs and simulator boundaries
  are labeled. Infrastructure success is separate from task success.
- Contact and portfolio links use the same identity as the live profile.
- No invented deployment, hardware result, license, performance metric or badge.
- Private-path/credential-pattern and changed-document link checks precede push.
- Remaining owner inputs: select a BHL license and set native profile pins.
- Remaining engineering work: complete failed/stalled evaluations, test held-out
  layouts and run a fresh dependency installation. These are not marked done.
