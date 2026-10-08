"""Install executable confirmation artifacts and update only existing prose."""
from pathlib import Path
import json
import shutil


def main():
    campaign = Path(__file__).resolve().parent
    repo = Path('/nfs/stak/users/sanchej7/hpc-share/Humanoid_Lite/bhl-robustness-ladder')
    archive = repo / 'campaigns/20261006-confirmatory'
    if archive.exists():
        raise FileExistsError('refusing to overwrite archived frozen campaign')
    archive.mkdir(parents=True)
    copied = []
    for file in campaign.rglob('*'):
        if not file.is_file() or any(part in ('jobs', 'locomotion', 'runs', '__pycache__') for part in file.relative_to(campaign).parts):
            continue
        if file.suffix.lower() in ('.md', '.txt'):
            raise ValueError(f'user prohibited a new prose file: {file}')
        # Keep the archive compact: execution logs stay at the live campaign.
        if file.suffix == '.log':
            continue
        destination = archive / file.relative_to(campaign)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, destination)
        copied.append(str(destination.relative_to(repo)))
    marker = '2026-10-06 confirmatory campaign'
    updates = {}
    readme = repo / 'README.md'
    content = readme.read_text()
    needle = '| Goal-reaching PPO |'
    pos = content.index(needle)
    content = content[:pos] + ('| Learned maze navigation (NavGym v5) | **35/36** goals; **0 falls**; **33/36** clean | '
               'Three final PPO actors on the same 12 held-out 6×6 layouts: 11/12, 12/12, 12/12; learned gait + navigator, oracle pose and goal; A* reference 12/12 |\n') + content[pos:]
    needle = 'The [dated report](docs/WEEKEND_RESULTS_2026-09-20.md) links per-seed evidence.'
    content = content.replace(needle, ('**2026-10-06 confirmatory campaign:** fresh, frozen evaluations are queued: '
               '3,600 locomotion episodes over all five randomization settings and three trained policies per setting, '
               'plus 384 matched navigation/A* episodes across nominal sensing and 35% packet dropout. '
               'Results remain pending. [Protocol, inputs and executable launchers](campaigns/20261006-confirmatory/) '
               'and [job ledger](SLURM_JOBS.md) retain the scope and gates.\n\n') + needle, 1)
    updates[readme] = content
    random_maze = repo / 'docs/RANDOM_MAZE.md'
    updates[random_maze] = random_maze.read_text() + '\n\n## NavGym v5 and fresh confirmation (2026-10-06)\n\n' + (
        'The final NavGym v5 actors trained with visitation memory, a wider coarse map, a yaw-change cost and a '
        'deployment range brake passed their predeclared gym gate on all three training seeds. In the subsequent '
        'MuJoCo transfer, the final actors reached **11/12, 12/12 and 12/12** goals with **zero falls**; clean '
        '(no physics-step wall contact) counts were **11/12 for each actor**. This is **35/36 actor-layout rollouts '
        'on 12 shared held-out layouts**, not 36 independent layouts. The matched A* reference reached 12/12. '
        'The gait and navigator are learned; pose and goal coordinate are oracle inputs; the reactive speed brake '
        'is scripted. The older published NavGym GIF remains an exploratory v2 example. '
        'Evidence: [`verdict.json`](../results/navgym-v5-transfer-20261002/verdict.json) and '
        '[final gym gate](../results/navgym-v5-20261002/verdict_v5.json).\n\n'
        'The **2026-10-06 confirmatory campaign** freezes all three final actors and the same gait for 48 new '
        '6×6 layouts (72000–72047), each under nominal sensing and 35% lidar/depth packet dropout. A* runs the '
        'identical layouts and sensing conditions. The project targets, declared before scored execution, require '
        'each actor to reach at least 40/48 nominal goals and 36/48 dropout goals with zero falls; wall contact '
        'and completion times are reported separately. Failures and missing files remain in the denominator. '
        'Array **21601709**, after locomotion array **21601701**; final evidence verdict **21601710**. '
        '[Frozen artifacts](../campaigns/20261006-confirmatory/); live outputs remain under '
        '`Computer_Vision/project_results_upgrade/humanoid_confirmatory`. These jobs are **pending evidence**, '
        'not a measured improvement.\n')
    status = repo / 'docs/STATUS.md'
    content = status.read_text().replace('**Updated:** 2026-09-28.', '**Updated:** 2026-10-06.', 1)
    updates[status] = content + '\n\n### 2026-10-06 confirmatory campaign\n\n' + (
        'Queued CPU-only confirmation, maximum two new evaluations simultaneously: **21601701** (15 cells, '
        '3,600 locomotion episodes), **21601709** (8 cells, 384 navigation/A* episodes, after the complete '
        'locomotion array), **21601710** (final verdict after both arrays, including failed tasks). No training '
        'checkpoint or favorable episode is selected after scoring. All five randomization rungs retain their '
        'three final training seeds; the exact upstream observation/controller, 6 Nm PD saturation and fall '
        'definitions are preserved. Fresh reset seeds 71000–71019, six fixed commands and both flat/no-push '
        'and 0.4 m/s matched velocity kicks. Tracking and displacement accompany falls so standing still '
        'cannot imply successful locomotion. Deployment smoke: all 15 models have exact ONNX parity under '
        'one inference thread; two complete physical rollouts have identical episode metrics; 80 relevant '
        'regression tests pass. Frozen executable archive: [campaigns](../campaigns/20261006-confirmatory/). '
        'Evidence is **pending**, with an estimated 5–8 hours of CPU execution plus scheduling.\n')
    ledger = repo / 'SLURM_JOBS.md'
    updates[ledger] = ledger.read_text() + '\n\n## 2026-10-06 confirmatory campaign\n\n' + (
        '| Job | Purpose | Fixed scope | Dependency |\n|---|---|---|---|\n'
        '| 21601701 | Fresh locomotion confirmation | 15 final policies × 6 commands × 20 reset seeds × '
        '2 conditions = 3,600 episodes; array 0–14%2; CPU 1, 8 GB, 4 h per cell | none |\n'
        '| 21601709 | Frozen learned navigator vs A* | 3 final actors + A* × nominal/drop35 × 48 '
        '6×6 layouts = 384 episodes; array 0–7%2; CPU 1, 8 GB, 4 h per cell | afterany:21601701 |\n'
        '| 21601710 | Final evidence validation | JSON episode contracts, complete file sets, source/model '
        'hashes, paired counts, project-local rules | afterany:21601701:21601709 |\n\n'
        'Source commit and 263 source/asset/model SHA256 values, exact package versions, all command and '
        'seed declarations, and immutable submission receipt are in '
        '[the executable archive](campaigns/20261006-confirmatory/). Only code, YAML/ONNX inputs, JSON receipts '
        'and XML test evidence are added; no new `.md` or `.txt` file. The active campaign is '
        '`/nfs/stak/users/sanchej7/hpc-share/Computer_Vision/project_results_upgrade/humanoid_confirmatory`; '
        'launchers and deploy paths are site-specific as in the existing reproducibility guide. Archive scripts '
        'can inspect/run that frozen campaign with `--campaign` set to the active path. A new independent '
        'campaign requires a new seed declaration before execution, not a rerun selected for a favorable result.\n\n'
        'Flat replication rule: each of the three default-randomization policies must have zero falls and '
        'fewer falls than its matched unrandomized policy. Other rungs and push04 are reported without '
        'promising favorable outcomes; displacement and surviving-step tracking errors accompany rates. '
        'Navigation rule: each final actor ≥40/48 nominal and ≥36/48 drop35 goals, zero falls; clean contacts '
        'reported separately. Shared layouts/reset seeds are not treated as independent trained policies. '
        'Incomplete evidence cannot pass, and final verdict files cannot overwrite a prior verdict.\n\n'
        'Unscored smoke seeds 70999/73999 were kept separate. All 15 exported models produced exact '
        'outputs under single-thread inference; unrandomized/default complete 10 s rollout metrics matched '
        'the upstream backend exactly. Measured smoke wall times were 3.40 s and 6.20 s per episode on '
        'the development host. Relevant regression suite: **80 passed**, one upstream ONNX deprecation '
        'warning, 38.77 s, no cache/prose generation. Navigation capture-pose/v5 observation/dropout contract '
        'was exercised on a 30 s pilot, whose timeout is unscored. Final status is **QUEUED / pending evidence**.\n')
    for path, value in updates.items():
        if marker in path.read_text():
            raise ValueError(f'campaign already documented: {path}')
        path.write_text(value)
    print(json.dumps({'archived_files': len(copied), 'updated_existing_prose': [str(p) for p in updates]}))


if __name__ == '__main__':
    main()
