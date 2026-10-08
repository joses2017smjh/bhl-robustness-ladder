# Session handoff — published September 20, 2026

## Start from the folding project's status

User clarification after publication: use
`~/hpc-share/Humanoid_Lite/lehome-fold-repro` for folding status and things to do.
Start with [lehome-fold-repro/SESSION_STATUS.md](../../lehome-fold-repro/SESSION_STATUS.md),
then its linked media-gate report, latest campaign status and original audit.
This publication handoff supplements those records; it is not a replacement
for the folding project's execution notes. Newer dated artifacts supersede
old pending-status snapshots. Strict training/evaluation results still live
in the linked BHL campaign.

The folding status index records the valid failed media gate, the remaining
15 illustrative episodes without a replacement array, evaluation blockers,
legacy trainer/scorer limitations and unfinished documentation corrections.
No scheduler query or job action was performed for this clarification.

## Outcome and user constraints

Target roles: **machine learning / AI engineer and robotics engineer**.
The user authorized completion and push. Publication is complete; the earlier
broad-push rejection was resolved with a separate curated public commit that
omits new raw scheduler receipts and sanitizes HPC paths/metadata. Do not
retry the older raw commit.

No HPC job was cancelled, released, altered, requeued or submitted in this
publication turn. The earlier two-seed training/evaluation work was already
queued by prior sessions. Do not modify queued/running jobs without a new
instruction. Do not start another full sweep just to improve the portfolio.

## Published commits

| Repository | Branch | Commit |
|---|---|---|
| bhl-robustness-ladder | main | 27c8454cad85ce86a5c5c8da8b5e622f38c3ea7c |
| joses2017smjh profile | main | 38aa9de5c9465af979e42c3b98ad32bd59d0f6b2 |
| spur-depth-service | master | 37fa2a38409a7e4ede6f34bbaa4793bf1eda01ef |
| isaac-sim-pruning-workflow | develop | 8edb3d799161734daa43caf6f81fd5d16868da37 |
| MetaNavT | main | 4670406cd2787f923294aeadc184e5e3b87b713f |
| IsaacSimFolding | main | a154097535286b75dc285a9357102b73be8d436a |
| Jose_Sanchez_Portfolio.com | main | 92e8dbea5a80e3b83af92f8a95da73b52046121a |

All were normal fast-forward pushes. GitHub production deployment
`6556728722` succeeded for the site commit. The public domain returned the new
ML/AI + robotics heading, four featured projects, current metrics and both
new wrist videos with HTTP 200:
https://jose-sanchez-portfolio-com.vercel.app

Recommended pins, in order: SPUR, BHL, Isaac pruning, MetaNaviT. Descriptions,
homepage links and topics were inspected and already accurately describe these
projects. No repository renames were needed. Actual pin selection remains a
manual GitHub profile action; the four projects are featured in the README/site.

## Important local Git state

The original working tree still has local commit `8201d7c` plus unrelated,
uncommitted research changes. **Do not push this original main or force-push.**
The public BHL commit was built directly on prior public main `1f1740f` in an
isolated clone; `8201d7c` is not its ancestor. Public and original local history
now diverge intentionally. Keep active research jobs and files intact. For new
public changes, start from current GitHub main in an isolated checkout and copy
only reviewed changes. Do not merge/reset the dirty original automatically.

Publication checkouts are in `/tmp/ml-robotics-publication-20260920/`
(`bhl`, `profile`, `spur`, `pruning`, `metanavt`, `folding`, `site`). They can be
recreated from the published repositories if temporary storage disappears.
The site is a worktree attached to `/tmp/robotics-portfolio-stvWjr/site` and has
an untracked local `node_modules` symlink; never publish that symlink.
The BHL clone has two fixture-only symlinks inside its uninitialized upstream
submodule; these were not staged or vendored.

Local durable publication records: `results/local-publication-20260920/`.
They include the commit map, deployment/browser reports and the original
folding-media build recipe. That recipe contains local input paths and should
remain a local execution record. The public media sidecars are sanitized.
A copy of the published BHL README is retained there; this workspace's root
README/code still describe the local research checkout. Portfolio/profile and
LinkedIn source documents were synchronized locally for the next session.

## What the folding GIFs actually establish

[Published gallery](https://github.com/joses2017smjh/bhl-robustness-ladder/blob/main/docs/FOLDING_MEDIA.md)
and [live folding page](https://jose-sanchez-portfolio-com.vercel.app/projects/isaac-folding/).

Five GIFs are in `docs/gifs/folding-*`, each under 2 MB, with PNG posters and
JSON provenance. Website equivalents are MP4 with reduced-motion posters.

- Earlier raster-adapted short-pants policy: success from pose episode 509,
  failure from 503. These are actual policy actions, not demonstration replay.
  The success label is latched first-hit; its record does not establish terminal
  success or the first successful timestep. Historical timing was missing, so
  playback is explicitly adjusted to 10 fps plus a one-second end hold.
- New adapted seed 0, short-sleeve top: completed 600 actions, 601 validated
  renders, 9,774 particles, maximum cloth displacement 0.215 m. Both ever-hit
  and terminal success are false. Overhead and both actual wrist cameras show
  that same failed episode at half simulation speed plus a one-second hold.
- No new adapted-policy successful GIF is verified. Do not relabel the older
  successful clip as an adaptation result or a terminally stable fold.

The later repair job `21367715` completed at 10:35:15 PDT. Original gate
`21360435` failed; array `21360436` and report `21360437` were dependent at the
prior audit. Their status was not refreshed or changed in this publication
turn. Do not treat old queue snapshots as current scheduler state.

Original new recording directory:
`/nfs/hpc/share/sanchej7/Humanoid_Lite/lehome-fold-repro/campaigns/20260920-media-root-fix/outputs/adapt_s0_Top_Short_Seen_0_pose2/`.
Historical source GIFs are in the same checkout's `results/`, named
`rollout_Pant_Short_Seen_0_repro_ep509_policy_success.gif` and
`rollout_Pant_Short_Seen_0_repro_ep503_policy_failure.gif`.

## Results and failure diagnosis to preserve

- Strict short-pants baseline 8/24 versus new adapted seed 1 3/24;
  both 0/4 on Unseen garments. No demonstrated improvement.
- Only 5/12 class/checkpoint cells completed; two failed and five timed out.
  Partial/infrastructure episodes must not enter a completed-fold denominator.
- Historical CPU cloth could freeze, and the old 6/24 count continued after
  23,250 swallowed render errors. It is not a valid fresh-camera baseline.
- Current incomplete runs expose garment-switch cleanup stalls and a scorer
  index mismatch (index 11029 with 10869 particles), not merely bad policies.
- Training improvements already implemented included successful-capture-only
  selection, class balance, garment-level splits and real future action chunks.
  Two 1500-update adaptations completed; best checkpoints were seed0 step1200
  and seed1 step1000. The clean split still has only two long-pants training
  successes. Data/schedule/checkpoint differences prevent isolating one cause.
- Isaac maze: 379/384 final-stage first episodes over four conditions × three
  seeds × 32 environments. Fixed route and oracle waypoints, noise off.
  Separate older frozen-gait MuJoCo inspection 3/3; team sizes each 5/5, with
  negative controls. Do not conflate the new Isaac checkpoints with those videos.

[Next experiments](FOLDING_NEXT_EXPERIMENTS.md) cites primary research checked
September 20: test 50/10/5 executed-action horizons with unchanged weights,
then compare uniform versus grasp-window sampling at a fixed dataset/budget.
These are proposals, not submitted jobs or new algorithms already implemented.
Fix scorer/garment lifetime gates before attempting a broader benchmark.

## Validation actually performed

- Curated BHL: **146 CPU tests passed**, 18.33 s, with
  `PYTHONPATH=src /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python -m pytest -q tests`.
  Two fixture files were byte-verified against nested upstream `fc90fedd`,
  pinned by parent `984741a`; without them, 144 pass / 2 missing-file failures.
- Scientific numeric/boolean values unchanged in 85 curated JSONs; evidence
  hashes updated where public JSON paths were sanitized.
- Five folding GIFs visually inspected; camera identity, frame timing, output
  hashes and provenance verified. No generated motion or image interpolation.
- Changed BHL text scan: no personal HPC paths or credential patterns; no
  missing relative Markdown targets or files over 95 MB. Not a full security audit.
- Site production build: 15 pages; check:portfolio passes 4 featured projects and
 193 local media references. Chromium: 24 page/viewport combinations, 320/390/768/1440 px:
  no overflow, broken loaded images, HTTP/JS errors; keyboard/reduced-motion pass.
- Live deployment: six pages, résumé and both wrist MP4s return 200 with expected
  new content. Featured README edits were source/link checked; no clean-machine
  installation or new inference benchmark was claimed.

## LinkedIn and manual follow-up

[Exact LinkedIn fields](LINKEDIN_2026-09-20.md) are prepared and published.
The live LinkedIn account was **not edited**. Paste/review the headline, About,
project/Featured entries and skills there. Existing public contact was retained:
`josejsanchez20172@gmail.com`. The optional email question received no reply;
current public profile/site already used this address.

Preserve career facts and team attribution. No fabricated experience, policy
success, aggregate metric or license badge was added. BHL still has no root
license file; selecting a license remains the owner's decision.


## Independent follow-up publication check

BHL public main is now `04dda644345fe660dca65bdb1d77559727c8a0a3`, a normal
fast-forward follow-up to `27c8454`. This preserves the concurrent publication
and adds a shorter README, CPU dependency file, reusable README template,
corrected Findings links and `docs/PUBLICATION_CHECK_2026-09-20.md`.
The exact final source passed **146 CPU tests in 17.84 seconds**. Remote
README, dependency file and verification note match the local publication.
The live ML profile is `38aa9de`; the portfolio returns HTTP 200 with its
updated ML hero. No HPC jobs were changed or submitted.

The publication checkout is `/tmp/bhl-public-final-20260920` on durable local
branch `publish/readme-checks-20260920`. The earlier isolated draft
`publish/weekend-20260920` (`bbf066d`) was never pushed and is superseded.
Do not push that draft or original raw `main`. The original working tree's
branch/history and unrelated research edits are preserved; its clean README
and reproducibility files plus the three new setup/template/check files were
updated to the published versions for local reading. This does not reconcile
the raw research branch with public history.
