"""Fresh reset seeds, matched disturbances, exact upstream locomotion controller."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone

from navgym_confirmatory import digest, read, write_new, wilson


def validate(campaign):
    p = read(campaign / 'dr_protocol.json')
    for name, expected in p['input_sha256'].items():
        path = Path(name) if name.startswith('/') else campaign / name
        if digest(path) != expected:
            raise ValueError(f'frozen input changed: {path}')
    for package, expected in p['package_versions'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f'package version changed: {package}')
    return p


def run(campaign, cell):
    p = validate(campaign)
    actor = p['policies'][cell]
    directory = campaign / 'locomotion' / actor['name']
    directory.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(campaign / 'snapshot/src'), str(campaign / 'snapshot/upstream_lowlevel'),
                    str(campaign / 'snapshot/scripts/bench')]
    import numpy as np
    from omegaconf import OmegaConf
    from bhl_robust.eval.harness import EvalConfig, HeadlessMujocoEnv, run_episode
    from bhl_robust.eval.mjcf_assets import prepare_mjcf
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController, OnnxPolicy
    from team_airlock import CpuPolicy
    cfg = OmegaConf.load(campaign / f"models/{actor['name']}/deploy.yaml")
    # One ONNX thread avoids host-wide affinity storms and needless per-job CPU use.
    # The upstream controller, observation history, action scaling and PD remain exact.
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    # Confirm single-thread inference equals the original backend before any scoring.
    original = OnnxPolicy(cfg.policy_checkpoint_path)
    probes = np.random.default_rng(99001).normal(size=(10, 1, 45)).astype(np.float32)
    difference = max(float(np.max(np.abs(policy.forward(obs) - original.forward(obs)))) for obs in probes)
    if difference > 1e-6:
        raise ValueError(f'ONNX backend changed predictions: {difference}')
    del original
    controller = RlController(cfg)
    controller.policy = policy
    write_new(directory / 'run_receipt.json', {'actor': actor, 'protocol_sha256': digest(campaign / 'dr_protocol.json'),
              'utc': datetime.now(timezone.utc).isoformat(), 'host': platform.node(), 'python': sys.version,
              'slurm_job_id': os.getenv('SLURM_JOB_ID'), 'array_task_id': os.getenv('SLURM_ARRAY_TASK_ID'),
              'onnx_single_thread_max_abs_diff': difference})
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='bhl-dr-confirm-', dir=os.getenv('TMPDIR', '/tmp')) as cache:
        scene = prepare_mjcf(Path(p['upstream']), Path(cache), 'biped')
        env = HeadlessMujocoEnv(cfg, scene)
        rows = []
        for condition, settings in p['conditions'].items():
            ec = EvalConfig(**p['settings'], seeds=tuple(p['reset_seeds']),
                            commands=tuple(tuple(c) for c in p['commands']), **settings)
            for command_index, command in enumerate(ec.commands):
                for seed in ec.seeds:
                    outcome = asdict(run_episode(env, controller, command, seed, ec))
                    # Undefined metrics after an immediate fall stay null, never favorable zeros.
                    outcome = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in outcome.items()}
                    row = {'actor': actor['name'], 'training_seed': actor['training_seed'], 'rung': actor['rung'],
                           'condition': condition, 'command_index': command_index, **outcome}
                    write_new(directory / f'{condition}-cmd{command_index}-seed{seed}.json', row)
                    rows.append(row)
            print(json.dumps({'actor': actor['name'], 'condition': condition, 'n': sum(r['condition'] == condition for r in rows),
                              'falls': sum(r['condition'] == condition and r['fell'] for r in rows)}), flush=True)
    validate(campaign)
    write_new(directory / 'process_result.json', {'complete': True, 'episodes': len(rows),
              'inputs_still_valid': True, 'wall_s': time.monotonic() - start})
    return 0


def row_stats(rows):
    n, k = len(rows), sum(r['fell'] for r in rows)
    def mean(key):
        values = [r[key] for r in rows if r[key] is not None]
        return statistics.mean(values) if values else None
    return {'episodes': n, 'falls': k, 'fall_rate': k / n if n else None,
            'fall_wilson95_descriptive': wilson(k, n) if n else None,
            'mean_survival_s': mean('survival_s'), 'mean_distance_m': mean('distance_m'),
            'mean_surviving_linear_tracking_error_m_s': mean('lin_vel_err'),
            'mean_surviving_yaw_tracking_error_rad_s': mean('yaw_rate_err'),
            'pushes_applied': sum(r['pushes_applied'] for r in rows),
            'pushes_survived': sum(r['pushes_survived'] for r in rows)}


def summarize(campaign, final):
    p = validate(campaign)
    results, evidence, problems = {}, {}, []
    expected = {(condition, ci, seed) for condition in p['conditions']
                for ci in range(len(p['commands'])) for seed in p['reset_seeds']}
    for actor in p['policies']:
        directory = campaign / 'locomotion' / actor['name']
        rows = []
        for path in directory.glob('*-cmd*-seed*.json'):
            rows.append(read(path))
        actual = {(r['condition'], r['command_index'], r['seed']) for r in rows}
        valid = actual == expected and len(rows) == len(expected)
        for r in rows:
            valid = valid and (r['actor'] == actor['name'] and r['rung'] == actor['rung'] and
                    r['training_seed'] == actor['training_seed'] and
                    [r[f'command_{axis}'] for axis in ('vx', 'vy', 'wz')] == p['commands'][r['command_index']] and
                    r['terrain_difficulty'] == 0.0 and isinstance(r['fell'], bool) and
                    0 <= r['survival_s'] <= p['settings']['episode_s'] and r['pushes_survived'] <= r['pushes_applied'])
        try:
            receipt = read(directory / 'run_receipt.json')
            process = read(directory / 'process_result.json')
            valid = valid and receipt['protocol_sha256'] == digest(campaign / 'dr_protocol.json') and process['complete'] is True and process['inputs_still_valid'] is True
        except (OSError, ValueError, KeyError):
            valid = False
        if not valid:
            problems.append(actor['name'])
        evidence[actor['name']] = {(r['condition'], r['command_index'], r['seed']): r for r in rows}
        results[actor['name']] = {'complete': valid, 'per_condition': {condition: row_stats([r for r in rows if r['condition'] == condition])
                    for condition in p['conditions']},
                    'per_command': {condition: {str(ci): row_stats([r for r in rows if r['condition'] == condition and r['command_index'] == ci])
                    for ci in range(len(p['commands']))} for condition in p['conditions']}}
    paired = {}
    replicated = not problems
    for seed in range(3):
        ref = evidence[f'dr-off-s{seed}']
        mine = evidence[f'dr-default-s{seed}']
        paired[str(seed)] = {}
        for condition in p['conditions']:
            shared = sorted(k for k in set(ref) & set(mine) if k[0] == condition)
            paired[str(seed)][condition] = {'paired_episodes': len(shared),
                'both_fall': sum(mine[k]['fell'] and ref[k]['fell'] for k in shared),
                'off_only_fall': sum(not mine[k]['fell'] and ref[k]['fell'] for k in shared),
                'default_only_fall': sum(mine[k]['fell'] and not ref[k]['fell'] for k in shared),
                'neither_fall': sum(not mine[k]['fell'] and not ref[k]['fell'] for k in shared)}
        a, b = results[f'dr-off-s{seed}']['per_condition']['flat'], results[f'dr-default-s{seed}']['per_condition']['flat']
        replicated = replicated and b['falls'] == 0 and a['falls'] > b['falls']
    pooled = {rung: {condition: row_stats([r for actor in p['policies'] if actor['rung'] == rung
                                        for r in evidence[actor['name']].values() if r['condition'] == condition])
                    for condition in p['conditions']} for rung in sorted({a['rung'] for a in p['policies']})}
    value = {'status': 'INCOMPLETE' if problems else 'PASS' if replicated else 'NEGATIVE', 'final': final,
             'incomplete_actors': problems, 'protocol_sha256': digest(campaign / 'dr_protocol.json'),
             'per_actor': results, 'paired_off_vs_default': paired, 'descriptive_pooled_by_rung': pooled,
             'rule': p['rule'], 'scope': 'Simulation only: frozen Isaac-trained 12-DoF biped policies in MuJoCo. '
             'Three training seeds per rung; shared reset seeds/commands are clustered, not 360 independent trained policies. '
             'Fall/survival paired with tracking and displacement; conditional surviving-step errors cannot alone rank failed policies.'}
    if final:
        write_new(campaign / 'dr_verdict.json', value)
    print(json.dumps(value, indent=2, allow_nan=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'run', 'summary'))
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--cell', type=int)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.mode == 'run':
        if args.cell is None or not 0 <= args.cell < 15:
            parser.error('run requires --cell 0..14')
        return run(campaign, args.cell)
    if args.mode == 'summary':
        return summarize(campaign, args.final)
    validate(campaign)
    print('FROZEN LOCOMOTION INPUT CHECK PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
