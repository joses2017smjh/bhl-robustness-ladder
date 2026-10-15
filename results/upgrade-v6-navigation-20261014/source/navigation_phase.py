"""Run and audit paired V6 navigation without changing earlier frozen evidence.

Learned influence is a command-level activity proxy. Goals, falls and restricted
completion times are measured separately; a statistical veto is not a gain.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from common import ROOT, PYTHON, UPSTREAM, digest, read, validate, write_new
from navigation_recovery import SETTINGS as RECOVERY_SETTINGS, BUDGET_FRACTION, MODES

SCHEMA = 'paired-navigation-phase-v6'
SENSORS = ('reactive', 'reactive_dropout')
OUTCOMES = ('goal', 'fall', 'time_out', 'nonfinite_action', 'nonfinite_state')
COUNTERS = ('steps', 'active_steps', 'model_evaluated_steps', 'model_authority_steps',
            'model_influence_steps', 'model_nonzero_influence_steps',
            'model_progress_preserving_influence_steps', 'model_zero_output_steps',
            'invalid_input_stops', 'observed_danger_stops', 'reference_fallback_steps')
ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def declarations(p, stage):
    require(stage in ('development', 'confirmation'), 'unknown navigation stage')
    return p['navigation_cells' if stage == 'development' else 'confirmation_cells']


def settings_for(p, stage):
    settings = p['navigation_settings' if stage == 'development' else 'confirmation_settings']
    seeds = settings['maze_seeds']
    require(bool(seeds) and seeds == list(range(seeds[0], seeds[0]+len(seeds))),
            'navigation requires a nonempty consecutive seed declaration')
    require(math.isfinite(settings['time_limit_s']) and settings['time_limit_s'] > 0,
            'invalid navigation time limit')
    return settings


def selected_cell(p, declared, gate=None):
    cell = dict(declared)
    if cell['method'] in ('selected', 'reference'):
        require(gate is not None and gate.get('passed') is True, 'unqualified confirmation resolution')
        chosen = gate['selected_method']
        require(chosen in p['navigation_gate']['candidates'], 'unknown selected global method')
        reference = p['navigation_gate']['reference_methods'][chosen]
        require(gate['reference_method'] == reference, 'selected/reference declaration mismatch')
        cell['method'] = chosen if cell['method'] == 'selected' else reference
    return cell


def variant_for(p, cell):
    variant=p['navigation_variants'][cell['method']]
    require(variant['recovery'] in MODES,'unknown recovery variant')
    return dict(variant,navigator='astar' if variant['recovery']=='planner' else cell['actor'])


def directory_for(root, stage, declared):
    return Path(root)/f'navigation_{stage}'/declared['name']


def phase_paths(root, stage, declared):
    base = Path(root)/'navigation_phase_receipts'/stage
    return base/(declared['name']+'.json'), base/(declared['name']+'_complete.json')


def identities(root, p, cell=None):
    root = Path(root)
    paths = [root/name for name in ('navigation_phase.py', 'navigation_experiment.py',
                                   'world_model.py', 'adaptive_command.py', 'common.py','navigation_recovery.py')]
    paths += [root/'snapshot/scripts/bench/maze_explore.py', Path(p['qualification'])]
    if cell is not None:
        paths += [root/'models'/f"{cell['gait']}-s{cell['training_seed']}"/name
                  for name in ('policy.onnx', 'deploy.yaml')]
        if variant_for(p,cell)['navigator']!='astar':paths.append(root/'models'/cell['actor']/'actor.onnx')
        if cell['method'] in p['navigation_gate']['model_methods']:
            paths.append(Path(p['world_model_checkpoint']))
    return {str(path): digest(path) for path in paths}


def command_for(root, p, stage, declared, gate=None):
    root = Path(root); cell = selected_cell(p, declared, gate); settings = settings_for(p, stage)
    variant=variant_for(p,cell)
    command = [str(PYTHON), str(root/'navigation_experiment.py'), 'evaluate',
        '--campaign', str(root), '--gait-name', cell['gait'],
        '--training-seed', str(cell['training_seed']), '--navigator', variant['navigator'],
        '--governor', variant['governor'], '--recovery-mode',variant['recovery'],'--sensor-mode', cell['sensor_mode'],
        '--seeds', str(len(settings['maze_seeds'])), '--seed-start', str(settings['maze_seeds'][0]),
        '--time-limit', str(float(settings['time_limit_s'])),
        '--qualification', str(Path(p['qualification'])),
        '--out-dir', str(directory_for(root, stage, declared))]
    if cell['method'] in p['navigation_gate']['model_methods']:
        command += ['--checkpoint', str(Path(p['world_model_checkpoint']))]
    return command


def inner_command(root, p, stage, declared, gate=None):
    cell = selected_cell(p, declared, gate); settings = settings_for(p, stage); root = Path(root)
    command=['maze_explore.py', '--upstream', str(UPSTREAM), '--gait',
        str(root/'models'/f"{cell['gait']}-s{cell['training_seed']}"/'deploy.yaml'),
        '--seeds', str(len(settings['maze_seeds'])), '--seed-start', str(settings['maze_seeds'][0]),
        '--n', '6', '--m', '6', '--extra-openings', '1', '--time-limit', str(float(settings['time_limit_s'])),
        '--sensor-mode', cell['sensor_mode'], '--dropout-probability', '.35', '--no-overwrite',
        '--out-dir', str(directory_for(root, stage, declared))]
    if variant_for(p,cell)['navigator']!='astar':
        command+=['--policy', str(root/'models'/cell['actor']/'actor.onnx'),
                  '--policy-capture-pose', '--policy-cmd-filter', 'none']
    return command


def checked_gate(root, p):
    path = Path(root)/'navigation_gate.json'; gate = read(path)
    require(gate['schema'] == SCHEMA and gate['protocol_sha256'] == digest(Path(root)/'protocol.json'),
            'navigation prerequisite gate identity mismatch')
    require(gate['source_sha256'] == identities(root, p), 'navigation prerequisite source mismatch')
    require(type(gate['passed']) is bool, 'navigation prerequisite passed flag invalid')
    # Gate binding files contain the full episode hash inventories validated at
    # selection. The final report re-audits the episodes themselves as well.
    for name, expected in gate['binding_files_sha256'].items():
        require(digest(Path(root)/name) == expected, 'navigation prerequisite receipt changed')
    if gate['passed']:
        expected_bindings = {str(path.relative_to(root)) for cell in p['navigation_cells']
                             for path in phase_paths(root, 'development', cell)}
        require(gate['complete'] is True and set(gate['binding_files_sha256']) == expected_bindings
                and not gate['problems'] and gate['selected_method'] in p['navigation_gate']['candidates'],
                'invalid selected navigation prerequisite')
        candidate = next(c for c in gate['candidates'] if c['method'] == gate['selected_method'])
        require(candidate['passed'] is True and gate['reference_method'] ==
                p['navigation_gate']['reference_methods'][gate['selected_method']],
                'selected navigation prerequisite not eligible')
    else:
        require(gate['selected_method'] is None and gate['reference_method'] is None,
                'failed navigation gate selects a controller')
    return gate


def counter_values(governor, modeled):
    statistics = governor['statistics']; counters = {}
    for field in COUNTERS:
        value = statistics[field]
        require(type(value) is int and value >= 0, 'invalid governor counter '+field)
        counters[field] = value
    order = [counters[k] for k in ('model_progress_preserving_influence_steps',
        'model_nonzero_influence_steps', 'model_influence_steps', 'model_authority_steps',
        'model_evaluated_steps', 'active_steps', 'steps')]
    require(order == sorted(order), 'inconsistent useful/model/active counter ordering')
    require(counters['invalid_input_stops'] <= counters['steps']
            and counters['observed_danger_stops'] <= counters['steps']
            and counters['model_zero_output_steps'] <= counters['model_authority_steps'],
            'governor event counter exceeds exposure')
    if not modeled:
        require(all(counters[k] == 0 for k in ('model_authority_steps', 'model_influence_steps',
            'model_nonzero_influence_steps', 'model_progress_preserving_influence_steps',
            'model_zero_output_steps', 'model_evaluated_steps')),
            'unmodeled controller claims learned influence')
    return counters


def validate_trace(row, counters):
    trace = row['trace']; require(isinstance(trace, list), 'invalid navigation trace')
    witnessed = 0
    for frame in trace:
        info = frame['governor']
        recovery=frame.get('navigation_recovery')
        if recovery and recovery.get('active'):
            require(info.get('navigation_phase')=='planner_recovery' and not any(info.get(k) for k in
                    ('model_authority','model_influence','model_nonzero_influence','model_progress_preserving_influence')),
                    'planner recovery falsely claims model authority or influence')
        for key in ('requested', 'executed'):
            values = info[key]
            require(len(values) == 3 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values),
                    'nonfinite sampled governor command')
        if info.get('model_progress_preserving_influence'):
            raw, output = info['requested'], info['executed']
            translating = math.hypot(*raw[:2]) > 1e-8
            active = math.hypot(*output[:2]) > 1e-8 if translating else raw[2]*output[2] > 1e-8
            require(info.get('model_authority') and info.get('model_influence')
                    and info.get('model_nonzero_influence') and active
                    and max(abs(v) for v in output) > 1e-8,
                    'zero/killed task command falsely labeled useful model influence')
            witnessed += 1
    require(witnessed <= counters['model_progress_preserving_influence_steps'],
            'sampled useful influence exceeds total counter')


def compact_row(row, cell, declared, settings, expected, root):
    require(row['seed'] in settings['maze_seeds'] and row['outcome'] in OUTCOMES,
            'unexpected seed or physical outcome')
    require(type(row['success']) is bool and row['success'] == (row['outcome'] == 'goal'),
            'navigation success/outcome mismatch')
    actor=expected['navigator']!='astar'
    require(row['sensor_mode'] == cell['sensor_mode'] and (row['cmd_filter']['mode'] == 'none' if actor else row.get('cmd_filter') is None),
            'row sensor/filter identity mismatch')
    require(row['gait']['checkpoint_sha256'] == expected['gait_sha256']
            and Path(row['gait']['deploy']).resolve() == (Path(root)/'models'/f"{cell['gait']}-s{cell['training_seed']}"/'deploy.yaml').resolve()
            and Path(row['gait']['checkpoint']).resolve() == (Path(root)/'models'/f"{cell['gait']}-s{cell['training_seed']}"/'policy.onnx').resolve()
            and (Path(row['policy']).resolve() == (Path(root)/'models'/cell['actor']/'actor.onnx').resolve() if actor else row['policy'] is None),
            'row gait/actor identity mismatch')
    governor = row['governor']
    require(governor['mode'] == expected['governor'] and governor['model_sha256'] == expected['model_sha256']
            and governor['maze_source_sha256'] == expected['maze_source_sha256']
            and governor['consumed_development'] is False,
            'row governor/model/source identity mismatch')
    require([row['maze'][k] for k in ('n', 'm', 'extra_openings')] == [6, 6, 1]
            and isinstance(row['maze']['layout_sha256'], str) and bool(row['maze']['layout_sha256']),
            'row layout identity mismatch')
    require(row['goal_check']['judge']['reached'] == row['success'], 'true goal judge mismatch')
    limit = settings['time_limit_s']; elapsed = row['elapsed_s']; completion = row['completion_s']
    require(isinstance(elapsed, (int, float)) and math.isfinite(elapsed) and 0 <= elapsed <= limit+.05,
            'invalid physical episode exposure')
    if row['success']:
        require(isinstance(completion, (int, float)) and math.isfinite(completion)
                and 0 < completion <= limit+.05 and abs(completion-elapsed) <= .05,
                'invalid raw completion time')
    else:
        require(completion is None, 'failed episode has successful completion time')
        if row['outcome'] == 'time_out':
            require(elapsed >= limit-.08, 'truncated timeout counted complete')
    require(math.isfinite(row['path_length_m']) and row['path_length_m'] >= 0, 'invalid traveled distance')
    counters = counter_values(governor, expected['model_sha256'] is not None)
    recovery=row['navigation_recovery']
    require(recovery['mode']==expected['recovery_mode'] and recovery['settings']==RECOVERY_SETTINGS,
            'recovery configuration mismatch')
    for key in ('steps','actor_steps','recovery_steps','plans','plan_failures'):
        require(type(recovery[key]) is int and recovery[key]>=0,'invalid recovery counter')
    require(recovery['steps']==counters['steps']==recovery['actor_steps']+recovery['recovery_steps']
            and recovery['plan_failures']<=recovery['plans']<=recovery['steps'],'recovery exposure mismatch')
    require(counters['model_evaluated_steps']<=recovery['actor_steps'],
            'model evaluation exceeds actor exposure; planner recovery cannot claim model activity')
    require(recovery['active']==bool(recovery['recovery_steps']),'recovery latch/counter mismatch')
    if recovery['mode']=='none':require(recovery['recovery_steps']==0 and recovery['trigger_s'] is None,'disabled recovery activated')
    if recovery['mode']=='planner':require(recovery['actor_steps']==0 and recovery['trigger_s']==0.,'planner did not own full episode')
    if recovery['trigger_s'] is not None:
        require(math.isfinite(recovery['trigger_s']) and 0<=recovery['trigger_s']<=elapsed+.05,'invalid recovery trigger time')
    if recovery['mode']=='budget_watchdog':
        budget=recovery['time_budget'];deadline=limit*BUDGET_FRACTION
        require(budget['episode_time_limit_s']==limit and budget['actor_fraction']==BUDGET_FRACTION
                and budget['actor_deadline_s']==deadline,'recovery time budget configuration mismatch')
        if elapsed >= deadline+.04:
            require(recovery['active'] and recovery['trigger_s']<=deadline+.04,
                    'budget recovery did not activate by declared deadline')
        if recovery.get('reason')=='actor_time_budget_exhausted':
            require(recovery['active'] and deadline<=recovery['trigger_s']<=deadline+.04,
                    'budget recovery trigger inconsistent with declared deadline')
    require(all('navigation_recovery' in frame for frame in row['trace']),'missing sampled recovery record')
    validate_trace(row, counters)
    return dict(seed=row['seed'], method=cell['method'], declared_method=declared['method'],
        cell=declared['name'], training_seed=cell['training_seed'], sensor_mode=cell['sensor_mode'],
        layout_sha256=row['maze']['layout_sha256'], outcome=row['outcome'], success=row['success'],
        elapsed_s=elapsed, completion_s=completion,
        restricted_completion_time_s=min(limit, completion) if row['success'] else limit,
        path_length_m=row['path_length_m'], counters=counters,
        invalid=bool(row['outcome'] in ('nonfinite_action', 'nonfinite_state') or counters['invalid_input_stops']),
        model_zero_output=bool(counters['model_zero_output_steps']),recovery=recovery)


def audit_cell(root, p, stage, declared, index, gate=None, require_complete=True):
    root = Path(root); directory = directory_for(root, stage, declared)
    receipt_path, complete_path = phase_paths(root, stage, declared)
    receipt = read(receipt_path); cell = selected_cell(p, declared, gate); settings = settings_for(p, stage)
    expected_sources = identities(root, p, cell)
    require(receipt['schema'] == SCHEMA and receipt['stage'] == stage and receipt['cell_index'] == index
            and receipt['declared_cell'] == declared and receipt['resolved_cell'] == cell
            and receipt['protocol_sha256'] == digest(root/'protocol.json')
            and receipt['source_sha256'] == expected_sources
            and receipt['command'] == command_for(root, p, stage, declared, gate)
            and receipt['gate_sha256'] == (digest(root/'navigation_gate.json') if stage == 'confirmation' else None),
            'phase master receipt mismatch')
    allowed = {f'seed{s}.json' for s in settings['maze_seeds']} | {'upgrade_receipt.json', 'upgrade_process.json', 'summary.json'}
    require({path.name for path in directory.iterdir()} == allowed
            and all(path.is_file() and not path.is_symlink() for path in directory.iterdir()),
            'navigation exact artifact inventory mismatch')
    artifacts = {name: digest(directory/name) for name in sorted(allowed)}
    if require_complete:
        complete = read(complete_path)
        require(complete['complete'] is True and complete['returncode'] == 0
                and complete['protocol_sha256'] == digest(root/'protocol.json')
                and complete['receipt_sha256'] == digest(receipt_path)
                and complete['artifact_sha256'] == artifacts, 'phase completion/inventory binding mismatch')
    inner = read(directory/'upgrade_receipt.json'); process = read(directory/'upgrade_process.json')
    variant=variant_for(p,cell)
    expected = dict(protocol_sha256=digest(root/'protocol.json'), gait_name=cell['gait'],
        training_seed=cell['training_seed'], navigator=variant['navigator'], governor=variant['governor'],
        recovery_mode=variant['recovery'],recovery_settings=RECOVERY_SETTINGS,
        recovery_source_sha256=digest(root/'navigation_recovery.py'),
        gait_sha256=digest(root/'models'/f"{cell['gait']}-s{cell['training_seed']}"/'policy.onnx'),
        consumed_development=False,
        model_sha256=digest(Path(p['world_model_checkpoint'])) if cell['method'] in p['navigation_gate']['model_methods'] else None,
        maze_source_sha256=digest(root/'snapshot/scripts/bench/maze_explore.py'),
        qualification_sha256=digest(Path(p['qualification'])), wrapper_sha256=digest(root/'navigation_experiment.py'),
        world_model_sha256=digest(root/'world_model.py'), adaptive_source_sha256=digest(root/'adaptive_command.py'),
        command=inner_command(root, p, stage, declared, gate), reset_seeds=settings['maze_seeds'])
    require(all(inner.get(k) == value for k, value in expected.items()), 'inner navigation provenance/command mismatch')
    require(process['completed'] is True and process['returncode'] == 0, 'navigation process did not complete')
    rows = []
    for seed in settings['maze_seeds']:
        raw = read(directory/f'seed{seed}.json')
        require(raw['seed'] == seed, 'episode filename/seed mismatch')
        rows.append(compact_row(raw, cell, declared, settings, expected, root))
    summary = read(directory/'summary.json'); report = statistics(rows)
    require(summary['n'] == len(rows) and summary['seeds'] == settings['maze_seeds']
            and summary['success'] == report['goals'] and summary['falls'] == report['falls']
            and summary['time_outs'] == report['timeouts']
            and summary['settings']['sensor_mode'] == cell['sensor_mode']
            and summary['settings']['time_limit'] == settings['time_limit_s'], 'summary/episodes mismatch')
    return rows, artifacts


def statistics(rows):
    totals = {name: sum(r['counters'][name] for r in rows) for name in COUNTERS}
    restricted = sum(r['restricted_completion_time_s'] for r in rows)
    return dict(episodes=len(rows), goals=sum(r['success'] for r in rows),
        falls=sum(r['outcome'] == 'fall' for r in rows), timeouts=sum(r['outcome'] == 'time_out' for r in rows),
        invalid=sum(bool(r['invalid']) for r in rows), model_zero_output_episodes=sum(bool(r['model_zero_output']) for r in rows),
        recovery_episodes=sum(r.get('recovery',{}).get('recovery_steps',0)>0 for r in rows),
        recovery_steps=sum(r.get('recovery',{}).get('recovery_steps',0) for r in rows),
        actor_steps=sum(r.get('recovery',{}).get('actor_steps',0) for r in rows),
        restricted_completion_time_sum_s=restricted,
        restricted_completion_time_mean_s=restricted/len(rows) if rows else None,
        total_distance_m=sum(r['path_length_m'] for r in rows), counters=totals,
        model_progress_influence_fraction=totals['model_progress_preserving_influence_steps']/max(1, totals['active_steps']))


def audit(root, stage):
    """Return compact audited rows; missing or inconsistent evidence is a problem."""
    root = Path(root); rows = []; cells = []; problems = []; skipped = []; bindings = {}; evidence = {}
    try:
        p = validate(root); declared_cells = declarations(p, stage); settings_for(p, stage)
        gate = checked_gate(root, p) if stage == 'confirmation' else None
    except ERRORS as exc:
        return dict(stage=stage, rows=[], cells=[], problems=[{'stage':stage, 'reason':str(exc)}],
                    skipped=[], complete=False, binding_files_sha256={}, evidence_sha256=None)
    names = [cell['name'] for cell in declared_cells]
    if len(set(names)) != len(names):
        problems.append({'stage':stage, 'reason':'duplicate cell declaration'})
    base = root/f'navigation_{stage}'
    if base.exists():
        unexpected = sorted(path.name for path in base.iterdir() if path.name not in names)
        if unexpected: problems.append({'stage':stage, 'reason':'unexpected navigation cells', 'names':unexpected})
    for index, declared in enumerate(declared_cells):
        directory = directory_for(root, stage, declared)
        receipt_path, complete_path = phase_paths(root, stage, declared)
        try:
            if stage == 'confirmation' and not gate['passed']:
                require({x.name for x in directory.iterdir()} == {'SKIPPED.json'}, 'invalid skipped cell inventory')
                skip = read(directory/'SKIPPED.json')
                require(skip == dict(schema=SCHEMA, skipped=True, stage=stage, cell_index=index,
                    declared_cell=declared, gate_sha256=digest(root/'navigation_gate.json'),
                    protocol_sha256=digest(root/'protocol.json'), source_sha256=identities(root, p),
                    reason='Global navigation development gate did not pass'), 'skip gate/source binding mismatch')
                skipped.append(declared['name']); evidence[str((directory/'SKIPPED.json').relative_to(root))] = digest(directory/'SKIPPED.json')
                continue
            cell_rows, artifacts = audit_cell(root, p, stage, declared, index, gate)
            rows.extend(cell_rows); cells.append(dict(cell=declared['name'], method=cell_rows[0]['method'],
                declared_method=declared['method'], training_seed=declared['training_seed'],
                sensor_mode=declared['sensor_mode'], **statistics(cell_rows)))
            for path in (receipt_path, complete_path):
                bindings[str(path.relative_to(root))] = digest(path)
            for name, sha in artifacts.items():evidence[str((directory/name).relative_to(root))] = sha
        except ERRORS as exc:
            problems.append({'cell':declared['name'], 'reason':str(exc)})
    layouts = defaultdict(set)
    for row in rows: layouts[row['seed']].add(row['layout_sha256'])
    mismatched = sorted(seed for seed, values in layouts.items() if len(values) != 1)
    if mismatched: problems.append({'stage':stage, 'reason':'matched raw maze layouts differ', 'seeds':mismatched})
    return dict(stage=stage, rows=rows, cells=cells, problems=problems, skipped=skipped,
        complete=not problems and len(cells)+len(skipped) == len(declared_cells),
        protocol_sha256=digest(root/'protocol.json'), binding_files_sha256=bindings,
        evidence_sha256=stable_hash(evidence), layout_clusters=len(layouts),
        pairing='Every method/policy/sensor uses the same declared raw maze layouts; layouts are clusters, not independent method trials.')


def comparisons(rows, candidate, reference, settings, declared=False):
    result = []; column = 'declared_method' if declared else 'method'
    seeds = settings['maze_seeds']
    for policy in range(3):
        for sensor in SENSORS:
            subset = lambda method: [r for r in rows if r[column] == method
                and r['training_seed'] == policy and r['sensor_mode'] == sensor]
            a, b = subset(reference), subset(candidate); sa, sb = statistics(a), statistics(b)
            keyed_a = {r['seed']:r for r in a}; keyed_b = {r['seed']:r for r in b}
            matched = (len(a) == len(b) == len(seeds) and set(keyed_a) == set(keyed_b) == set(seeds)
                and all(keyed_a[s]['layout_sha256'] == keyed_b[s]['layout_sha256'] for s in seeds))
            target = settings['min_nominal_goals' if sensor == 'reactive' else 'min_drop35_goals']
            passed = bool(matched and sa['invalid'] == sb['invalid'] == 0 and sb['falls'] == 0
                and sb['model_zero_output_episodes'] == 0 and sb['goals'] >= max(target, sa['goals']))
            won = sum(keyed_b[s]['success'] and not keyed_a[s]['success'] for s in seeds) if matched else 0
            lost = sum(keyed_a[s]['success'] and not keyed_b[s]['success'] for s in seeds) if matched else 0
            result.append(dict(training_seed=policy, sensor_mode=sensor, reference=reference,
                candidate=candidate, matched=matched, passed=passed, target_goals=target,
                strict_gain_observed=bool(matched and (sb['goals'] > sa['goals'] or sb['falls'] < sa['falls'])),
                paired_goal_wins=won, paired_goal_losses=lost, baseline=sa, selected=sb))
    return result


def selection_gate(root, audited, p):
    require(audited['stage'] == 'development', 'selection requires development evidence')
    settings = p['navigation_settings']; rules = p['navigation_gate']; candidates = []
    for method in rules['candidates']:
        original = comparisons(audited['rows'], method, 'baseline', settings)
        reference = rules['reference_methods'][method]
        same_reference = comparisons(audited['rows'], method, reference, settings)
        report = statistics([r for r in audited['rows'] if r['method'] == method])
        modeled = method in rules['model_methods']
        strict = any(c['strict_gain_observed'] for c in same_reference)
        useful = report['model_progress_influence_fraction']
        eligible = bool(audited['complete'] and not audited['problems'] and all(c['passed'] for c in original)
            and all(c['passed'] for c in same_reference) and strict
            and (not modeled or useful >= rules['minimum_model_progress_influence_fraction']))
        candidates.append(dict(method=method, reference_method=reference, modeled=modeled, passed=eligible,
            baseline_comparisons=original, same_reference_comparisons=same_reference,
            strict_reference_gain_observed=strict, statistics=report))
    qualified = [c for c in candidates if c['passed']]
    chosen = min(qualified, key=lambda c:(-c['statistics']['goals'],
        c['statistics']['restricted_completion_time_sum_s'], rules['candidates'].index(c['method']))) if qualified else None
    return dict(schema=SCHEMA, passed=chosen is not None,
        protocol_sha256=digest(Path(root)/'protocol.json'), source_sha256=identities(root,p),
        selected_method=chosen['method'] if chosen else None,
        reference_method=chosen['reference_method'] if chosen else None,
        candidates=candidates, problems=audited['problems'],
        binding_files_sha256=audited['binding_files_sha256'], evidence_sha256=audited['evidence_sha256'],
        complete=audited['complete'], selection_used_confirmation=False,
        ranking='One global candidate: total goals, then lower restricted completion time including failures at the full limit, then declaration order.',
        influence_scope='Progress-preserving influence is command activity, not causal task benefit. Learned gains additionally require goal/fall improvement against the same reference.',
        pairing=audited.get('pairing'))


def confirmation_verdict(root, audited, p, gate):
    require(audited['stage'] == 'confirmation', 'confirmation verdict requires confirmation evidence')
    if not gate.get('passed'):
        return dict(passed=False, status='SKIPPED_NOT_QUALIFIED' if audited['complete'] else 'INCOMPLETE',
                    problems=audited['problems'], selected_method=None, confirmation_changed_candidate=False)
    settings = p['confirmation_settings']; rows = audited['rows']; modeled = gate['selected_method'] in p['navigation_gate']['model_methods']
    checks = {reference: comparisons(rows, 'selected', reference, settings, declared=True)
              for reference in ('baseline', 'yaw_only', 'reference')}
    selected = [r for r in rows if r['declared_method'] == 'selected']; report = statistics(selected)
    strict = any(c['strict_gain_observed'] for c in checks['reference'])
    passed = bool(audited['complete'] and not audited['problems'] and all(c['passed'] for c in checks['baseline'])
        and all(c['passed'] for c in checks['reference']) and strict
        and (not modeled or report['model_progress_influence_fraction'] >=
            p['navigation_gate']['minimum_model_progress_influence_fraction']))
    clusters = {r['seed'] for r in selected}; fall_clusters = {r['seed'] for r in selected if r['outcome'] == 'fall'}
    full_inventory = bool(audited['complete'] and len(selected) == 6*len(settings['maze_seeds'])
        and clusters == set(settings['maze_seeds'])
        and all(c['matched'] for checks_for_reference in checks.values() for c in checks_for_reference))
    return dict(passed=passed, status='CONFIRMED_FINITE_SAMPLE' if passed else 'INCOMPLETE' if not audited['complete'] else 'NOT_CONFIRMED',
        selected_method=gate['selected_method'], reference_method=gate['reference_method'], comparisons=checks,
        statistics=report, strict_reference_gain_observed=strict, problems=audited['problems'],
        layout_clusters=len(clusters), layouts_with_any_selected_fall=len(fall_clusters),
        conditional_zero_fall_upper95_per_layout=(1-.05**(1/len(clusters))) if full_inventory and not fall_clusters else None,
        confidence_scope='The conditional one-sided zero-fall binomial bound treats raw maze layouts as independent clusters for these fixed policies and sensing modes. It is not a full reliability guarantee; policy/sensor repeats are not pooled as independent safety trials.',
        confirmation_changed_candidate=False)


def run(root, stage, index):
    root = Path(root).resolve(); p = validate(root); cells = declarations(p, stage)
    require(0 <= index < len(cells), 'navigation cell index out of range')
    declared = cells[index]; gate = checked_gate(root,p) if stage == 'confirmation' else None
    out = directory_for(root,stage,declared)
    if stage == 'confirmation' and not gate['passed']:
        out.mkdir(parents=True,exist_ok=False)
        write_new(out/'SKIPPED.json', dict(schema=SCHEMA,skipped=True,stage=stage,cell_index=index,
            declared_cell=declared,gate_sha256=digest(root/'navigation_gate.json'),
            protocol_sha256=digest(root/'protocol.json'),source_sha256=identities(root,p),
            reason='Global navigation development gate did not pass'))
        return 0
    cell = selected_cell(p,declared,gate); command = command_for(root,p,stage,declared,gate)
    receipt_path,complete_path = phase_paths(root,stage,declared)
    require(not out.exists(), 'navigation output already exists')
    receipt = dict(schema=SCHEMA,stage=stage,cell_index=index,declared_cell=declared,resolved_cell=cell,
        protocol_sha256=digest(root/'protocol.json'),source_sha256=identities(root,p,cell),
        command=command,gate_sha256=digest(root/'navigation_gate.json') if stage == 'confirmation' else None,
        job_id=os.getenv('SLURM_JOB_ID'))
    write_new(receipt_path,receipt)
    environment = dict(os.environ,TMPDIR='/tmp',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run(command,check=True,cwd=root,env=environment)
    validate(root)
    require(receipt['source_sha256'] == identities(root,p,cell), 'navigation sources changed during execution')
    _,artifacts = audit_cell(root,p,stage,declared,index,gate,require_complete=False)
    write_new(complete_path,dict(complete=True,returncode=0,protocol_sha256=digest(root/'protocol.json'),
        receipt_sha256=digest(receipt_path),artifact_sha256=artifacts))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode',choices=('run','gate'))
    parser.add_argument('--campaign',type=Path,default=ROOT); parser.add_argument('--stage',choices=('development','confirmation'))
    parser.add_argument('--cell',type=int); args=parser.parse_args()
    if args.mode == 'run':
        if args.stage is None or args.cell is None:parser.error('run requires --stage and --cell')
        return run(args.campaign,args.stage,args.cell)
    p=validate(args.campaign); result=selection_gate(args.campaign,audit(args.campaign,'development'),p)
    write_new(args.campaign/'navigation_gate.json',result)
    print(json.dumps(dict(passed=result['passed'],selected_method=result['selected_method'],problems=result['problems'])),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
