"""Append verified overnight outcomes to existing project documentation only.

No earlier prose, run inputs, evaluation evidence or resume claims are rewritten.
Missing, invalid and incomplete evidence remains explicitly INCOMPLETE.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path

from dr_confirmatory import validate as validate_dr
from navgym_confirmatory import digest, read, validate_inputs, write_new

ROOT = Path(__file__).resolve().parent
REPO = Path('/nfs/stak/users/sanchej7/hpc-share/Humanoid_Lite/bhl-robustness-ladder')
DOCS = ('README.md', 'docs/RANDOM_MAZE.md', 'docs/STATUS.md', 'SLURM_JOBS.md')
MARKER = '<!-- bhl-confirmation-results-2026-10-06 '


def integer(value, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f'invalid count: {value!r}, maximum {maximum}')
    return value


def load_final(path, protocol, kind):
    if not path.is_file():
        return {'status': 'INCOMPLETE', 'reason': 'immutable final verdict missing', 'data': None}
    value = read(path)
    if value.get('final') is not True or value.get('protocol_sha256') != digest(protocol):
        raise ValueError(f'{kind}: final flag or protocol hash mismatch')
    if value.get('status') not in ('PASS', 'NEGATIVE', 'INCOMPLETE'):
        raise ValueError(f'{kind}: unknown verdict status')
    declared = read(protocol)
    if kind == 'locomotion':
        expected = {a['name'] for a in declared['policies']}
        if set(value['per_actor']) != expected:
            raise ValueError('locomotion actor set changed')
        complete = all(r['complete'] for r in value['per_actor'].values())
        replicated = complete
        for row in value['per_actor'].values():
            for condition in declared['conditions']:
                counts = row['per_condition'][condition]
                n = integer(counts['episodes'], 120)
                integer(counts['falls'], n)
                if row['complete'] and n != 120:
                    raise ValueError('complete locomotion cell has wrong denominator')
        for seed in range(3):
            off = value['per_actor'][f'dr-off-s{seed}']['per_condition']['flat']
            default = value['per_actor'][f'dr-default-s{seed}']['per_condition']['flat']
            replicated = replicated and default['falls'] == 0 and off['falls'] > default['falls']
        for rung, conditions in value['descriptive_pooled_by_rung'].items():
            actors = [a['name'] for a in declared['policies'] if a['rung'] == rung]
            if len(actors) != 3:
                raise ValueError('pooled rung does not have three declared training seeds')
            for condition, counts in conditions.items():
                if counts['episodes'] != sum(value['per_actor'][a]['per_condition'][condition]['episodes'] for a in actors):
                    raise ValueError('pooled locomotion denominator mismatch')
                if counts['falls'] != sum(value['per_actor'][a]['per_condition'][condition]['falls'] for a in actors):
                    raise ValueError('pooled locomotion fall count mismatch')
        expected_status = 'INCOMPLETE' if not complete else 'PASS' if replicated else 'NEGATIVE'
    else:
        expected = {c['name'] for c in declared['cells']}
        if set(value['per_cell']) != expected:
            raise ValueError('navigation cell set changed')
        complete = all(r['complete'] for r in value['per_cell'].values())
        passed = complete
        for cell in declared['cells']:
            row = value['per_cell'][cell['name']]
            n = integer(row['n'], 48)
            goals = integer(row['goals'], n)
            integer(row['clean'], goals)
            falls = integer(row['falls'], n)
            if row['complete'] and n != 48:
                raise ValueError('complete navigation cell has wrong denominator')
            if cell['actor'] != 'astar':
                passed = passed and goals >= declared['project_targets'][cell['condition']]['min_goals'] and falls == 0
        expected_status = 'INCOMPLETE' if not complete else 'PASS' if passed else 'NEGATIVE'
    if value['status'] != expected_status:
        raise ValueError(f'{kind}: verdict disagrees with the declared count rule')
    return {'status': value['status'], 'reason': None, 'data': value}


def gather(campaign):
    paths = {'locomotion_protocol': campaign / 'dr_protocol.json', 'navigation_protocol': campaign / 'protocol.json',
             'locomotion_verdict': campaign / 'dr_verdict.json', 'navigation_verdict': campaign / 'verdict.json'}
    source_hashes = {name: digest(path) if path.is_file() else None for name, path in paths.items()}
    source_hashes['publisher_code'] = digest(Path(__file__))
    rows = {}
    try:
        validate_dr(campaign)
        validate_inputs(campaign)
        inputs_error = None
    except (ValueError, OSError, KeyError) as error:
        inputs_error = str(error)
    for kind, protocol, verdict in (
            ('locomotion', 'locomotion_protocol', 'locomotion_verdict'),
            ('navigation', 'navigation_protocol', 'navigation_verdict')):
        try:
            if inputs_error:
                raise ValueError(f'frozen input validation failed: {inputs_error}')
            rows[kind] = load_final(paths[verdict], paths[protocol], kind)
        except (ValueError, KeyError, OSError, TypeError) as error:
            rows[kind] = {'status': 'INCOMPLETE', 'reason': str(error), 'data': None}
    recorded = {'locomotion': len(list((campaign / 'locomotion').glob('*/*-cmd*-seed*.json'))),
                'navigation': len(list((campaign / 'runs').glob('*/seed*.json')))}
    stamp = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    return rows, recorded, paths, source_hashes, stamp


def render(campaign, rows, recorded, hashes, stamp):
    utc = datetime.now(timezone.utc).isoformat()
    text = [f'\n\n{MARKER}source_sha256={stamp} -->\n',
            '### Recorded confirmation outcomes — campaign 2026-10-06\n',
            f'Append-only evidence update at `{utc}`. This dated record supersedes the earlier queued-status '
            'paragraph for this campaign; historical experiment results above remain their original records.\n',
            f'**Locomotion: {rows["locomotion"]["status"]}. Navigation: {rows["navigation"]["status"]}.** '
            f'Recorded episode files: locomotion **{recorded["locomotion"]}/3,600 planned**; navigation '
            f'**{recorded["navigation"]}/384 planned**. File counts alone do not certify valid episodes.\n']
    dr = rows['locomotion']['data']
    if dr:
        text += ['| Randomization rung (3 training seeds) | Flat falls / reported episodes | '
                 'Push04 falls / reported episodes | Flat mean displacement |\n',
                 '|---|---|---|---|\n']
        for rung in ('dr-off', 'dr-s0.5', 'dr-default', 'dr-s1.5', 'dr-aggressive'):
            conditions = dr['descriptive_pooled_by_rung'][rung]
            flat, push = conditions['flat'], conditions['push04']
            distance = f'{flat["mean_distance_m"]:.3f} m' if flat['mean_distance_m'] is not None else 'unavailable'
            text += [f'| {rung} | {flat["falls"]}/{flat["episodes"]} | '
                     f'{push["falls"]}/{push["episodes"]} | {distance} |\n']
        text += ['\nFlat denominators are 360 per rung when complete (3 policies × 6 commands × 20 shared reset '
                 'seeds), with another 360 disturbed episodes per rung. The flat replication rule requires '
                 'zero falls for each default policy and fewer falls than its matched unrandomized policy. '
                 'Push04 is descriptive. Full per-training-seed/command tracking, displacement and paired '
                 'outcomes remain in `dr_verdict.json`; surviving-step errors alone cannot rank failed policies. '
                 'When the verdict is INCOMPLETE, reported partial counts do not establish a validated complete experiment.\n']
    else:
        text += [f'Locomotion evidence was not promoted: {rows["locomotion"]["reason"]}. '
                 'No missing or invalid denominator is substituted with zero falls.\n']
    nav = rows['navigation']['data']
    if nav:
        text += ['\n| Navigation actor | Condition | Goals / reported episodes | Clean | Falls | Complete evidence |\n',
                 '|---|---|---|---|---|---|\n']
        for name, row in nav['per_cell'].items():
            actor, condition = name.rsplit('-', 1)
            text += [f'| {actor} | {condition} | {row["goals"]}/{row["n"]} | '
                     f'{row["clean"]}/{row["n"]} | {row["falls"]}/{row["n"]} | {row["complete"]} |\n']
        text += ['\nEach complete navigation cell contains 48 shared held-out 6×6 layouts. Each final learned '
                 'actor must meet both ≥40/48 nominal and ≥36/48 drop35 goals with zero falls; all actors and '
                 'the matched A* control are reported. Clean means no wall contact at any physics step. '
                 'An incomplete cell cannot establish validated success, regardless of its reported partial goal count.\n']
    else:
        text += [f'Navigation evidence was not promoted: {rows["navigation"]["reason"]}. '
                 'Missing or invalid results remain INCOMPLETE.\n']
    text += ['\nScope: simulation only. Isaac-trained frozen 12-DoF PPO gait; navigation uses a learned PPO '
             'actor with oracle pose/goal and a scripted reactive speed brake; A* is the reference. Shared '
             'layouts, commands and reset seeds are clustered observations, not independent trained policies. '
             'Neither PASS nor a zero observed fall count certifies hardware deployment.\n',
             f'\nImmutable JSON sources: `{campaign}/dr_verdict.json` and `{campaign}/verdict.json`. '
             f'Locomotion JSON SHA256: `{hashes["locomotion_verdict"] or "MISSING"}`; navigation JSON SHA256: '
             f'`{hashes["navigation_verdict"] or "MISSING"}`. Combined source SHA256: `{stamp}`. '
             'Executable protocol and publisher are archived in `campaigns/20261006-confirmatory`.\n']
    return ''.join(text).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=ROOT)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    rows, recorded, paths, hashes, stamp = gather(campaign)
    addition = render(campaign, rows, recorded, hashes, stamp)
    if args.preview:
        # Pure read-only preflight; no docs, receipts or evaluation files created.
        for name in DOCS:
            path = REPO / name
            if not path.is_file() or MARKER in path.read_text():
                raise ValueError(f'cannot publish to missing/already published doc: {path}')
        print(json.dumps({'preview': True, 'statuses': {k: v['status'] for k, v in rows.items()},
                          'recorded_episode_files': recorded, 'source_sha256': stamp,
                          'existing_documents': DOCS, 'append_bytes_per_document': len(addition)}, indent=2))
        return 0
    receipt = campaign / 'publication_receipt.json'
    if receipt.exists():
        raise FileExistsError('publication receipt already exists; refusing duplicate publication')
    pre_sha = {}
    published = []
    with ExitStack() as stack:
        handles = []
        # O_APPEND means earlier/user prose is never overwritten. Lock all four existing files first.
        for name in DOCS:
            path = REPO / name
            fd = os.open(path, os.O_RDWR | os.O_APPEND)  # no O_CREAT: only existing documentation
            stack.callback(os.close, fd)
            fcntl.flock(fd, fcntl.LOCK_EX)
            content = path.read_bytes()
            if MARKER.encode() in content:
                raise ValueError(f'campaign result already published to {path}')
            if (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (path.stat().st_dev, path.stat().st_ino):
                raise RuntimeError(f'document was replaced while opening: {path}')
            pre_sha[name] = hashlib.sha256(content).hexdigest()
            handles.append((name, path, fd))
        # All evidence must still match the JSON files inspected above immediately before appending.
        for name, path in paths.items():
            actual = digest(path) if path.is_file() else None
            if actual != hashes[name]:
                raise RuntimeError(f'evidence changed during publication: {path}')
        for name, path, fd in handles:
            if digest(path) != pre_sha[name]:
                raise RuntimeError(f'user edited {path}; no stale content is rewritten')
            if (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (path.stat().st_dev, path.stat().st_ino):
                raise RuntimeError(f'user replaced {path}; no stale content is rewritten')
            if os.write(fd, addition) != len(addition):
                raise OSError(f'short append: {path}')
            os.fsync(fd)
            published.append(name)
        write_new(receipt, {'utc': datetime.now(timezone.utc).isoformat(), 'slurm_job_id': os.getenv('SLURM_JOB_ID'),
                  'source_sha256': stamp, 'source_hashes': hashes, 'prior_doc_sha256': pre_sha,
                  'appended_existing_docs': published, 'statuses': {k: v['status'] for k, v in rows.items()},
                  'recorded_episode_files': recorded, 'method': 'append-only; no prior prose rewritten'})
    print(json.dumps({'published': published, 'statuses': {k: v['status'] for k, v in rows.items()}, 'source_sha256': stamp}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
