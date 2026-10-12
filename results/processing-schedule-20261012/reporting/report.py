#!/usr/bin/env python3
"""Independent compact-evidence report; never imports/modifies scientific source.

Run only after the unique campaign dispatch finishes. The reporter rehashes
compact evidence and independently scores its numerical traces. Full image/raw
archives remain bound by original completion receipts, not reprocessed here.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tarfile
from zoneinfo import ZoneInfo

import numpy as np

DEFAULT_CAMPAIGN = Path('/nfs/hpc/share/sanchej7/humanoid-research-20261011/processing-schedule-v1-20261012')
ARCHIVE = '46eb5d49361cb64f4f18dc8290a7bcb002c21b01aa035e3fe8c8e049949a0dda'
SOURCE = '152ccf4d8a616c5725bfb0c44ac838818e028ee3'
PIPELINES = ('legacy_cumulative', 'bounded_latest')
ARMS = ('stereo', 'lidar_inertial')
CONDITIONS = ('nominal', 'camera_10hz', 'causal_40ms_exposure')
CASE_NAMES = {'straight': 'straight', 'occluders': 'occluders'}
CONDITION_NAMES = {'nominal': 'Nominal', 'camera_10hz': '10 Hz camera', 'causal_40ms_exposure': '40 ms exposure'}
ARM_NAMES = {'stereo': 'Stereo', 'lidar_inertial': 'LiDAR–IMU'}
PIPELINE_NAMES = {'legacy_cumulative': 'Legacy', 'bounded_latest': 'Bounded latest'}
KEY = ('case', 'seed', 'condition', 'arm', 'pipeline')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def eq_number(actual, expected, label):
    if actual is None or expected is None:
        require(actual is expected, label + ': unavailable value mismatch')
    else:
        require(finite(actual) and finite(expected) and math.isclose(actual, expected, abs_tol=1e-9, rel_tol=1e-9), label + ': numeric mismatch')


def p95(values):
    return float(np.quantile(values, .95)) if values else None


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def secure_path(root, relative):
    root = Path(root).resolve()
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root), 'evidence path escapes its owning directory')
    require(path.is_file(), 'required evidence missing: ' + str(path))
    return path


def verify_file(path, expected, lineage):
    actual = sha(path)
    require(actual == expected, 'evidence hash mismatch: ' + str(path))
    lineage[str(path)] = actual
    return actual


def cell_key(cell):
    return tuple(cell[k] for k in KEY)


def row_key(row):
    return (row['case'], row['seed'], row['condition'], row['method'], row['pipeline'])


def rescore(row, horizon=40.):
    """Recompute primary outcomes from decision/evaluator traces, not headlines."""
    trace = row['trace']
    require(trace and all(finite(r[k]) for r in trace for k in ('decision_timestamp_s', 'timestamp_s',
        'evaluator_goal_distance_m', 'evaluator_height_m', 'evaluator_tilt_rad')), 'finite nonempty evaluator trace required')
    require(all(a['timestamp_s'] < b['timestamp_s'] and a['decision_timestamp_s'] < b['decision_timestamp_s']
        for a, b in zip(trace, trace[1:])), 'trace clock regressed')
    require(all(t['decision_timestamp_s'] < t['timestamp_s'] for t in trace), 'decision must precede physics observation')
    require(all(type(t['evaluator_contact']) is bool for t in trace), 'explicit contact booleans required')
    require(type(row['nonfinite']) is bool, 'explicit nonfinite status required')
    eq_number(trace[-1]['timestamp_s'], row['elapsed_simulation_s'], 'trace horizon')
    reached = any(t['evaluator_goal_distance_m'] <= .30 for t in trace)
    contact = any(t['evaluator_contact'] for t in trace)
    fell = any(t['evaluator_tilt_rad'] >= .78 or t['evaluator_height_m'] < row['evaluator_initial_height_m'] - .25 for t in trace)
    full = row['elapsed_simulation_s'] >= horizon - 1e-6
    goal = reached and full and not (contact or fell or row['nonfinite'])
    for name, actual in (('goal_reached_evaluator', reached), ('collision', contact), ('fell', fell), ('full_horizon', full), ('success', goal)):
        require(type(row[name]) is bool and row[name] == actual, 'trace outcome mismatch: ' + name)
    nav = [t for t in trace if t['decision_timestamp_s'] >= 10. - 1e-9]
    stale = sum(t['reason'] == 'native_stale_stop' for t in nav)
    fraction = stale / len(nav) if nav else None
    require(row['navigation_decisions'] == len(nav) and row['stale_navigation_decisions'] == stale, 'navigation denominator/count mismatch')
    eq_number(row['stale_navigation_fraction'], fraction, 'stale fraction')
    ages = [t['decision_timestamp_s'] - t['native_timestamp_s'] for t in nav if t['native_timestamp_s'] is not None]
    require(all(finite(age) and age >= -1e-8 for age in ages), 'future or invalid available capture')
    capture_p95 = p95(ages)
    eq_number(row['navigation_pose_age_p95_ms'], None if capture_p95 is None else 1000 * capture_p95, 'available capture age p95')
    services = row['schedule_timing']
    delivered = row['response_deliveries']
    native = [s['native_client_s'] for s in services]
    require(all(finite(x) and x >= 0 for x in native), 'finite native service required')
    eq_number(row['client_wall_p95_ms'], None if not native else 1000 * p95(native), 'native client p95')
    delivery_values = [d['wall_seconds'] for d in delivered]
    eq_number(row['full_pipeline_p95_ms'], None if not delivery_values else 1000 * p95(delivery_values), 'delivered service p95')
    return dict(clean_goal=goal, fell=fell, contact=contact, full_horizon=full,
        navigation_decisions=len(nav), stale_decisions=stale, stale_fraction=fraction,
        available_capture_age_p95_ms=row['navigation_pose_age_p95_ms'],
        native_client_p95_ms=row['client_wall_p95_ms'], modeled_delivery_service_p95_ms=row['full_pipeline_p95_ms'])


def audit_timing(row, evidence):
    require(row['acquisition_schedule'] == evidence and row['pipeline'] == evidence['pipeline'], 'timing receipt/episode differ')
    queue = evidence['queue']
    def count(value):
        require(type(value) is int and value >= 0, 'nonnegative integer count required')
        return value
    require(count(queue['produced']) == count(queue['selected']) + count(queue['superseded']) + count(queue['pending']), 'joined event conservation failed')
    require(queue['pending'] <= 1 and queue['maximum_waiting_events'] == 1, 'joined queue bound violated')
    requests = evidence['requests']
    services = row['schedule_timing']
    require(queue['selected'] == len(requests) == len(services) == row['native_frames'], 'request/frame count mismatch')
    require(set(queue['streams']) == {'camera', 'lidar'}, 'both stream counters required')
    for name, stream in queue['streams'].items():
        selected = {r[name + '_id'] for r in requests}
        require(count(stream['produced_unique']) == count(stream['selected_unique']) + count(stream['superseded_before_selection']) + count(stream['pending_unselected']), 'stream conservation failed')
        require(stream['selected_unique'] == len(selected) and stream['dispatches'] == len(requests), 'unique stream selections mismatch')
        require(stream['reused_dispatches'] == len(requests) - len(selected) and stream['pending_unselected'] <= 1, 'stream reuse/bound mismatch')
        require(evidence[name + '_captures_total'] == stream['produced_unique'] + len(evidence[name + '_never_joined']), 'unjoined capture ledger mismatch')
    recorder = evidence['recorder']
    require(recorder.get('joined') is True and recorder.get('pending', 0) == 0 and recorder.get('error') is None, 'recording did not finish cleanly')
    if row['pipeline'] == 'bounded_latest':
        require(0 < recorder['maximum_pending'] <= recorder['capacity'] == 8 and recorder['tasks'] == evidence['camera_captures_total'], 'recorder bounds/task count mismatch')
    dt = row['latency_quantization_s']
    require(finite(dt) and dt > 0, 'positive policy period required')
    for timing, request in zip(services, requests):
        pre, native, start = request['preprocessing_service_s'], timing['native_client_s'], request['request_s']
        require(all(finite(x) and x >= 0 for x in (pre, native, start)), 'finite measured service required')
        capture = request['camera_capture_s' if row['method'] == 'stereo' else 'lidar_capture_s']
        require(capture <= start + 1e-9, 'future capture reached native service')
        eq_number(timing['capture_s'], capture, 'preserved estimator capture')
        eq_number(timing['service_s'], pre + native, 'service counted once')
        ticks = max(1, math.ceil((pre + native) / dt))
        require(timing['delivery_ticks'] == ticks, 'delivery quantization changed')
        eq_number(timing['ready_s'], start + pre, 'selected preprocessing ready time')
        eq_number(timing['estimator_end_s'], start + pre + native, 'estimator service end')
        eq_number(timing['delivery_s'], start + ticks * dt, 'modeled delivery')
        eq_number(timing['capture_age_at_delivery_s'], timing['delivery_s'] - capture, 'delivery capture age')
        q = request['queue']
        require(q['enqueue_s'] <= start, 'future event selected')
        eq_number(q['queue_residence_s'], start - q['enqueue_s'], 'queue residence')
    require(len(row['response_deliveries']) + row['responses_still_pending_at_horizon'] == len(services), 'response conservation failed')
    require(row['responses_still_pending_at_horizon'] in (0, 1), 'more than one response in flight')
    for actual, modeled in zip(row['response_deliveries'], services):
        eq_number(actual['arrival_s'], modeled['delivery_s'], 'actual response arrival')
        eq_number(actual['wall_seconds'], modeled['service_s'], 'actual response service')


def grouping(rows, unrun):
    groups = []
    for arm in ARMS:
        for condition in CONDITIONS:
            for pipeline in PIPELINES:
                values = [r for r in rows if (r['method'], r['condition'], r['pipeline']) == (arm, condition, pipeline)]
                skipped = sum((c['arm'], c['condition'], c['pipeline']) == (arm, condition, pipeline) for c in unrun)
                groups.append(dict(arm=arm, condition=condition, pipeline=pipeline, measured=len(values), unrun=skipped,
                    clean_goals=sum(r['success'] for r in values), falls=sum(r['fell'] for r in values), contacts=sum(r['collision'] for r in values),
                    full_horizons=sum(r['full_horizon'] for r in values), navigation_decisions=sum(r['navigation_decisions'] for r in values),
                    stale_decisions=sum(r['stale_navigation_decisions'] for r in values),
                    mean_episode_stale_fraction=mean([r['stale_navigation_fraction'] for r in values]),
                    mean_episode_client_p95_ms=mean([r['client_wall_p95_ms'] for r in values]),
                    mean_episode_model_service_p95_ms=mean([r['full_pipeline_p95_ms'] for r in values]),
                    mean_episode_available_capture_age_p95_ms=mean([r['navigation_pose_age_p95_ms'] for r in values]),
                    superseded_joined_events=sum(r['acquisition_schedule']['queue']['superseded'] for r in values),
                    pending_joined_events=sum(r['acquisition_schedule']['queue']['pending'] for r in values),
                    incomplete_responses=sum(r['responses_still_pending_at_horizon'] for r in values)))
    return groups


def pairing(expected, rows, unrun):
    indexed = {row_key(row): row for row in rows}
    skipped = {cell_key(cell) for cell in unrun}
    result = []
    for key in sorted({cell_key(cell)[:-1] for cell in expected}):
        sides = [indexed.get(key + (p,)) for p in PIPELINES]
        pair = dict(zip(KEY[:-1], key))
        for pipeline, row in zip(PIPELINES, sides):
            pair[pipeline] = {'status': 'MEASURED', 'clean_goal': row['success'], 'stale_fraction': row['stale_navigation_fraction'],
                'fell': row['fell'], 'contact': row['collision'], 'full_horizon': row['full_horizon']} if row else {'status': 'UNRUN_AFTER_NEGATIVE_NOMINAL_ARM_SMOKE' if key + (pipeline,) in skipped else 'MISSING'}
        complete = all(sides)
        stale_comparable = complete and all(row['stale_navigation_fraction'] is not None for row in sides)
        pair.update(complete=complete, stale_comparable=stale_comparable,
            clean_goal_difference=int(sides[1]['success']) - int(sides[0]['success']) if complete else None,
            stale_fraction_difference=sides[1]['stale_navigation_fraction'] - sides[0]['stale_navigation_fraction'] if stale_comparable else None)
        result.append(pair)
    return result


def audit(campaign):
    campaign = Path(campaign).resolve()
    lineage = {}
    intake, dispatch, submission = (read(campaign / name) for name in ('intake.json', 'dispatch-result.json', 'submission.json'))
    require(intake['archive_sha256'] == dispatch['archive_sha256'] == submission['archive_sha256'] == ARCHIVE, 'wrong frozen source archive')
    require(intake['source_commit'] == SOURCE, 'wrong frozen scientific source commit')
    require(dispatch['status'] == 'COMPLETE' and not dispatch['execution_error'] and not dispatch['unrun'], 'dispatch is incomplete; final claims unavailable')
    require(dispatch['job_id'] == submission['job_id'], 'unique allocation identity differs')
    archive = campaign / 'frozen-campaign.tar.gz'
    verify_file(archive, ARCHIVE, lineage)
    with tarfile.open(archive, 'r:gz') as tar:
        protocol_bytes = tar.extractfile('inputs/scientific-protocol.json').read()
        frozen_protocol = json.loads(protocol_bytes)
        manifest = json.load(tar.extractfile('manifest.json'))
    require(hashlib.sha256(protocol_bytes).hexdigest() == intake['scientific_protocol_sha256'], 'scientific protocol hash mismatch')
    runner = read(campaign / 'protocol.json')
    verify_file(campaign / 'protocol.json', intake['protocol_sha256'], lineage)
    stages = {r['name']: r for r in dispatch['completed']}
    names = [r['name'] for r in runner['jobs']]
    require(len(names) == 27 and list(stages) == names and len(dispatch['completed']) == 27, 'all27 ordered stages required')
    stage_results = {}
    for name in names:
        directory = campaign / f"{name}-{dispatch['job_id']}"
        completion, launch = read(directory / 'completion.json'), read(directory / 'launch.json')
        require(completion['archive_sha256'] == launch['archive_sha256'] == ARCHIVE and completion['exit_status'] == 0
            and not completion.get('error') and completion['status'] in ('PASS', 'NEGATIVE'), 'stage execution/source did not pass')
        path = directory / 'campaign_result.json'
        verify_file(path, stages[name]['result_sha256'], lineage)
        recorded = completion['files']['campaign_result.json']
        require(path.stat().st_size == recorded['bytes'] and sha(path) == recorded['sha256'], 'stage result completion binding failed')
        require(completion['files']['outputs.tar.gz']['sha256'] == stages[name]['raw_archive_sha256'], 'raw archive receipt binding differs')
        stage_results[name] = read(path)
    profile, smoke, collection = (stage_results[names[i]] for i in (0, 1, -1))
    require(collection['status'] == 'PASS' and collection['schema'] == 'bhl-processing-schedule-v1-collection', 'final scientific collector did not pass')
    require(profile['status'] == 'PASS' and profile['frames'] == profile['input_equivalence']['frames'] == 40
        and profile['input_equivalence']['status'] == 'PASS', 'identical-input profile did not pass')
    require(profile['original_episode_sha256'] == intake['profile_episode_sha256'], 'wrong retained profiling episode')
    smoke_path = campaign / 'qualified-smoke.json'
    verify_file(smoke_path, collection['qualified_smoke_sha256'], lineage)
    require(read(smoke_path) == smoke and smoke['status'] == 'PASS' and smoke['execution_gate'] is True
        and smoke['episodes'] == 12 and not smoke['problems'] and not smoke['unrun_cells']
        and smoke['completed_cells'] == frozen_protocol['smoke']['cells'], 'same-source12-cell smoke incomplete')
    for name, digest in smoke['source_sha256'].items():
        require(manifest['source/' + name]['sha256'] == digest, 'smoke source hash differs from frozen file')
    require(profile['source_sha256'] == smoke['source_sha256'], 'profile and smoke source differ')
    readiness = smoke['arm_development_ready']
    require(set(readiness) == {p + '/' + a for p in PIPELINES for a in ARMS} and all(type(v) is bool for v in readiness.values()), 'four explicit nominal gates required')
    require(collection['arm_development_ready'] == readiness, 'collector gates differ')
    rows, unrun, seen = [], [], set()
    expected = frozen_protocol['development']['cells']
    expected_keys = {cell_key(c) for c in expected}
    require(len(expected) == len(expected_keys) == 48, '48 unique declared episodes required')
    linked = {str(Path(v['path']).resolve()): v['sha256'] for v in collection['source_results']}
    require(len(linked) == 24, '24 source pair results required')
    for name in names[2:-1]:
        directory = campaign / 'collector-inputs' / name
        path = directory / 'campaign_result.json'
        require(str(path.resolve()) in linked, 'pair result not bound to scientific collection')
        verify_file(path, linked[str(path.resolve())], lineage)
        result = read(path)
        require(result == stage_results[name], 'compact source differs from stage result')
        require(result['status'] == 'PASS' and result['phase'] == 'development' and not result['problems']
            and result['source_sha256'] == smoke['source_sha256'] and result['input_sha256'] == smoke['input_sha256']
            and result['arm_development_ready'] == readiness, 'mixed-source or incomplete pair')
        require(result['episodes'] == len(result['completed_cells']) == len(result['episode_files']), 'pair episode count mismatch')
        for cell, relative in zip(result['completed_cells'], result['episode_files']):
            key = cell_key(cell)
            require(key in expected_keys and key not in seen and readiness[cell['pipeline'] + '/' + cell['arm']], 'unexpected/duplicate/gated measured episode')
            episode = secure_path(directory, relative)
            verify_file(episode, result['episode_sha256'][relative], lineage)
            row = read(episode)
            require(row_key(row) == key, 'episode identity mismatch')
            for relative_file, field in (('inference_sensors.npz', 'inference_sensors_sha256'), ('evaluator_truth.npz', 'evaluator_truth_sha256'),
                    ('native/stream_receipt.json', 'stream_receipt_sha256'), ('sensors/processing_schedule.json', 'processing_schedule_sha256')):
                verify_file(secure_path(episode.parent, relative_file), row[field], lineage)
            rescore(row, frozen_protocol['development']['seconds'])
            audit_timing(row, read(episode.parent / 'sensors/processing_schedule.json'))
            rows.append(row)
            seen.add(key)
        for cell in result['unrun_cells']:
            key = cell_key(cell)
            require(key in expected_keys and key not in seen and not readiness[cell['pipeline'] + '/' + cell['arm']]
                and cell['reason'] == 'UNRUN_AFTER_NEGATIVE_NOMINAL_ARM_SMOKE', 'invalid/duplicate unrun cell')
            seen.add(key)
            unrun.append(cell)
    require(seen == expected_keys and collection['declared_cells'] == 48 and collection['measured_cells'] == len(rows)
        and collection['unrun_after_nominal_gate'] == len(unrun) and not collection['missing_cells'], 'final declared cohort incomplete')
    groups, pairs = grouping(rows, unrun), pairing(expected, rows, unrun)
    for pipeline in PIPELINES:
        for arm in ARMS:
            values = [r for r in rows if r['pipeline'] == pipeline and r['method'] == arm]
            aggregate = collection['by_pipeline_arm'][pipeline + '/' + arm]
            require(aggregate['episodes'] == len(values) and aggregate['clean_goals'] == sum(r['success'] for r in values)
                and aggregate['falls'] == sum(r['fell'] for r in values) and aggregate['contacts'] == sum(r['collision'] for r in values), 'published aggregate count differs from traces')
            eq_number(aggregate['mean_stale_fraction'], mean([r['stale_navigation_fraction'] for r in values]), 'aggregate stale fraction')
    complete = [p for p in pairs if p['complete']]
    comparable = [p for p in pairs if p['stale_comparable']]
    paired = collection['paired']
    require(paired['complete_pairs'] == len(complete) and paired['stale_comparable_pairs'] == len(comparable), 'paired denominator mismatch')
    eq_number(paired['mean_paired_stale_fraction_difference'], mean([p['stale_fraction_difference'] for p in comparable]), 'paired stale difference')
    eq_number(paired['clean_goal_count_difference'], sum(p['clean_goal_difference'] for p in complete) if complete else None, 'paired goal difference')
    return dict(status='VERIFIED_COMPLETE', verified_utc=datetime.now(timezone.utc).isoformat(),
        verified_local=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(), job_id=dispatch['job_id'],
        frozen_source_commit=SOURCE, source_archive_sha256=ARCHIVE, scientific_protocol_sha256=intake['scientific_protocol_sha256'],
        declared_episodes=48, measured_episodes=len(rows), unrun_episodes=len(unrun), smoke_episodes=12, smoke_readiness=readiness,
        groups=groups, pairs=pairs, paired_summary=paired, unrun_cells=unrun,
        input_profile=dict(frames=40, input_equivalence=profile['input_equivalence'], timings=profile['timings'], scope=profile['scope']),
        execution=read(campaign / 'execution-start.json'), submission_resources=submission['resources'],
        lineage_sha256=lineage, report_generator_sha256=sha(__file__),
        verification_scope='Rehashed frozen input archive, result receipts and compact numerical evidence; independently rescored goals/safety/stale counts and timing accounting. Full archived images/native streams were verified by the frozen dispatcher; this reporter does not rerun native estimators or reverify entire output archives.',
        scientific_scope=frozen_protocol['scope'])


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
        + ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def fmt(value, scale=1., suffix=''):
    return 'N/A' if value is None else f'{value * scale:.2f}{suffix}'


def write_markdown(report, output):
    groups = report['groups']
    totals = []
    for p in PIPELINES:
        selected = [g for g in groups if g['pipeline'] == p]
        totals.append([PIPELINE_NAMES[p], sum(g['measured'] for g in selected), sum(g['unrun'] for g in selected),
            sum(g['clean_goals'] for g in selected), sum(g['falls'] for g in selected), sum(g['contacts'] for g in selected)])
    paired = report['paired_summary']
    lines = ['# Native processing-schedule results — October 12, 2026', '',
        f"**{report['measured_episodes']}/48 declared development episodes measured; {report['unrun_episodes']} explicitly UNRUN after nominal smoke gates.** The separate 12-episode smoke and identical-input transport profile passed their execution prerequisites.", '',
        'This comparison uses actual ORB-SLAM3 and FAST-LIO2 native calls in closed-loop simulation. The treatment models separate simulation acquisition, a bounded latest joined camera/LiDAR queue, selected PNG transport, native service and evidence recording. It is not a measured concurrent robot implementation.', '',
        table(['Pipeline', 'Measured', 'UNRUN', 'Clean goals', 'Falls', 'Contacts'], totals), '',
        f"The primary paired stale-stop change (candidate minus legacy) is **{fmt(paired['mean_paired_stale_fraction_difference'], 100, ' percentage points')}**, across {paired['stale_comparable_pairs']} comparable pairs. The paired clean-goal count change is **{paired['clean_goal_count_difference'] if paired['clean_goal_count_difference'] is not None else 'N/A'}**, across {paired['complete_pairs']} complete pairs. Negative stale-stop differences favor the candidate; positive goal differences favor the candidate.", '',
        'Two independent reset seeds, each repeated across declared routes/conditions, support descriptive development results only. No population confidence interval, hardware validation or independent confirmation is claimed.', '',
        '## Condition and estimator results', '',
        table(['Estimator / condition', 'Pipeline', 'Goals / measured', 'UNRUN', 'Mean stale %', 'Stale decisions / navigation decisions', 'Falls / contacts', 'Superseded events'],
            [[ARM_NAMES[g['arm']] + ' / ' + CONDITION_NAMES[g['condition']], PIPELINE_NAMES[g['pipeline']],
                f"{g['clean_goals']}/{g['measured']}" if g['measured'] else 'UNRUN', g['unrun'],
                fmt(g['mean_episode_stale_fraction'], 100), f"{g['stale_decisions']}/{g['navigation_decisions']}",
                f"{g['falls']}/{g['contacts']}", g['superseded_joined_events']] for g in groups]), '',
        'Stale-stop fractions use pre-step navigation decisions at or after 10 seconds. Initialization is excluded. Short horizons remain failures for clean-goal scoring. A missing navigation denominator is N/A; it is never scored as zero stale stops. Superseded events count the joined queue, not unique discarded camera frames.', '',
        '## Timing', '',
        table(['Estimator / condition', 'Pipeline', 'Native client ms', 'Modeled delivery service ms', 'Available packet capture age ms'],
            [[ARM_NAMES[g['arm']] + ' / ' + CONDITION_NAMES[g['condition']], PIPELINE_NAMES[g['pipeline']],
                fmt(g['mean_episode_client_p95_ms']), fmt(g['mean_episode_model_service_p95_ms']), fmt(g['mean_episode_available_capture_age_p95_ms'])] for g in groups]), '',
        'Each timing column is the **mean of per-episode p95 values**, not a pooled population p95. Native client measures real wall time; modeled delivery service adds the pipeline’s declared charge. Capture age includes waiting and delivery at the control decision, including available untracked response timestamps. The 350 ms capture-age gate is unchanged. Simulator rendering, raycasting and evidence work are separately recorded; their removal from candidate modeled inference service is an explicit scheduling assumption, not an observed hardware speedup.', '',
        '## Every paired route and seed', '',
        table(['Estimator / condition', 'Seed / route', 'Legacy goal / stale %', 'Candidate goal / stale %', 'Goal difference', 'Stale difference pp'],
            [[ARM_NAMES[p['arm']] + ' / ' + CONDITION_NAMES[p['condition']], f"{p['seed']} / {p['case']}",
                ('UNRUN' if p[PIPELINES[0]]['status'] != 'MEASURED' else f"{int(p[PIPELINES[0]]['clean_goal'])} / {fmt(p[PIPELINES[0]]['stale_fraction'], 100)}"),
                ('UNRUN' if p[PIPELINES[1]]['status'] != 'MEASURED' else f"{int(p[PIPELINES[1]]['clean_goal'])} / {fmt(p[PIPELINES[1]]['stale_fraction'], 100)}"),
                p['clean_goal_difference'] if p['clean_goal_difference'] is not None else 'N/A', fmt(p['stale_fraction_difference'], 100)] for p in report['pairs']]), '',
        '## Prerequisites and reproducibility', '',
        table(['Nominal pipeline / estimator gate', 'Readiness'], [[k, 'PASS' if v else 'NEGATIVE — associated development cells UNRUN'] for k, v in report['smoke_readiness'].items()]), '',
        'The profile reprocessed the same 40 retained stereo pairs, checked decoded identities and the original LiDAR/IMU array hashes, and replayed measured transport costs through the queue. Its native service is explicitly zero. Sequential file-cache effects prevent treating this microbenchmark as an inferential speedup estimate.', '',
        f"Allocation `{report['job_id']}`; frozen source `{report['frozen_source_commit']}`. Both pipelines use the same allocation, resource limits, frozen policy, route, reset seed, camera specification, native thresholds and response quantization. Pair order alternates in the frozen protocol. Closed-loop images can diverge after actions diverge.", '',
        report['verification_scope'], '',
        'Files: [numerical evidence](report-evidence.json), [group CSV](groups.csv), [paired CSV](pairs.csv), [group figure](groups.png), [paired figure](pairs.png). Figures are also supplied as SVG.', '',
        f"Verified {report['verified_local']} (America/Los_Angeles).", '']
    (output / 'REPORT.md').write_text('\n'.join(lines))


def plot(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ('#3776ab', '#dd7b22')
    labels = [(a, c) for a in ARMS for c in CONDITIONS]
    indexed = {(g['arm'], g['condition'], g['pipeline']): g for g in report['groups']}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    positions = np.arange(len(labels))
    for n, pipeline in enumerate(PIPELINES):
        values = [indexed[a, c, pipeline] for a, c in labels]
        x = positions + (n - .5) * .36
        stale = [100 * g['mean_episode_stale_fraction'] if g['mean_episode_stale_fraction'] is not None else np.nan for g in values]
        goals = [g['clean_goals'] if g['measured'] else np.nan for g in values]
        axes[0].bar(x, stale, width=.34, label=PIPELINE_NAMES[pipeline], color=colors[n])
        axes[1].bar(x, goals, width=.34, label=PIPELINE_NAMES[pipeline], color=colors[n])
        for location, group, s, goal in zip(x, values, stale, goals):
            axes[0].text(location, (s + 2 if np.isfinite(s) else 3), f'{s:.1f}' if np.isfinite(s) else ('UNRUN' if not group['measured'] else 'N/A'), ha='center', fontsize=8, rotation=0 if np.isfinite(s) else 90)
            axes[1].text(location, (goal + .08 if np.isfinite(goal) else .1), f"{group['clean_goals']}/{group['measured']}" if group['measured'] else 'UNRUN', ha='center', fontsize=8, rotation=0 if group['measured'] else 90)
    axes[0].set_ylim(0, 112)
    axes[0].set_ylabel('Mean episode stale-stop fraction (%)')
    axes[1].set_ylim(0, 4.7)
    axes[1].set_ylabel('Clean goals (annotation: goals / measured)')
    axes[1].set_xticks(positions, [ARM_NAMES[a] + '\n' + CONDITION_NAMES[c] for a, c in labels])
    axes[0].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.grid(axis='y', alpha=.25)
        axis.set_axisbelow(True)
    fig.suptitle('Native scheduling simulation: 48 declared episodes, two reset seeds\nActual native estimators; modeled pipeline separation; no hardware timing claim', fontsize=13)
    fig.savefig(output / 'groups.png', dpi=180)
    fig.savefig(output / 'groups.svg', metadata={'Date': None})
    plt.close(fig)
    pairs = report['pairs']
    fig, axes = plt.subplots(1, 2, figsize=(13, 12), sharey=True, gridspec_kw={'width_ratios': [3, 1]}, constrained_layout=True)
    y = np.arange(len(pairs))
    for i, pair in enumerate(pairs):
        delta = pair['stale_fraction_difference']
        if delta is None:
            axes[0].text(0, i, 'UNRUN pair' if not pair['complete'] else 'N/A denominator', va='center', fontsize=8, color='#777777')
        else:
            value = delta * 100
            axes[0].barh(i, value, height=.65, color=colors[0] if value <= 0 else colors[1])
            axes[0].plot(value, i, '.', color='#222222')
        goal = pair['clean_goal_difference']
        if goal is None:
            axes[1].text(0, i, 'UNRUN', ha='center', va='center', fontsize=8, color='#777777')
        else:
            axes[1].plot(goal, i, 'o', color=colors[0] if goal >= 0 else colors[1])
            axes[1].annotate(f"{int(pair[PIPELINES[0]]['clean_goal'])}→{int(pair[PIPELINES[1]]['clean_goal'])}", (goal, i), xytext=(6, 0), textcoords='offset points', va='center', fontsize=8)
    axes[0].set_yticks(y, [f"{ARM_NAMES[p['arm']]} / {CONDITION_NAMES[p['condition']]} / {p['seed']} / {p['case']}" for p in pairs], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlim(-105, 105)
    axes[1].set_xlim(-1.45, 1.65)
    axes[1].set_xticks([-1, 0, 1])
    axes[0].set_xlabel('Candidate − legacy stale-stop percentage points\nNegative favors candidate')
    axes[1].set_xlabel('Clean-goal difference\nPositive favors candidate')
    for axis in axes:
        axis.axvline(0, color='#777777', lw=.8)
        axis.grid(axis='x', alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle('Every declared seed/route pair — descriptive development outcomes\nMissing outcomes remain UNRUN; missing navigation denominators remain N/A', fontsize=12)
    fig.savefig(output / 'pairs.png', dpi=180)
    fig.savefig(output / 'pairs.svg', metadata={'Date': None})
    plt.close(fig)


def write_outputs(report, output, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report-evidence.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    for name, rows in (('groups', report['groups']), ('pairs', report['pairs'])):
        flattened = [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows]
        with (output / (name + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
            writer.writeheader()
            writer.writerows(flattened)
    if plots:
        plot(report, output)
    write_markdown(report, output)
    inventory = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}
    (output / 'artifact-manifest.json').write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--campaign', type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--status-only', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.status_only:
        path = args.campaign / 'dispatch-result.json'
        if path.exists():
            result = read(path)
            print(json.dumps({k: result[k] for k in ('status', 'job_id', 'execution_error', 'unrun')}, indent=2))
        else:
            progress = sorted(args.campaign.glob('progress-*.json'))
            print(json.dumps(dict(status='PENDING_FINAL_COLLECTION', job_id=read(args.campaign / 'submission.json')['job_id'],
                completed_stages=len(read(progress[-1])['completed']) if progress else 0), indent=2))
        return 0
    if args.output is None:
        parser.error('--output is required for a verified report')
    report = audit(args.campaign)
    write_outputs(report, args.output, plots=not args.no_plots)
    print(json.dumps(dict(status=report['status'], measured=report['measured_episodes'], unrun=report['unrun_episodes'], output=str(args.output)), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
