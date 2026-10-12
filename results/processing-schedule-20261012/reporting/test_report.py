"""Reporter boundary tests; fixtures are synthetic and never research results."""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('standalone_schedule_report', Path(__file__).with_name('report.py'))
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def row():
    trace = [dict(decision_timestamp_s=step * .04, timestamp_s=(step + 1) * .04,
        evaluator_goal_distance_m=5., evaluator_height_m=1., evaluator_tilt_rad=0., evaluator_contact=False,
        native_timestamp_s=2. if step >= 52 else None,
        reason='native_stale_stop' if step >= 250 else 'initialization') for step in range(1000)]
    ages = [x['decision_timestamp_s'] - 2. for x in trace if x['decision_timestamp_s'] >= 10. - 1e-9]
    return dict(case='straight', seed=750000, condition='nominal', method='stereo', pipeline='legacy_cumulative',
        trace=trace, elapsed_simulation_s=40., evaluator_initial_height_m=1., nonfinite=False, goal_reached_evaluator=False,
        collision=False, fell=False, full_horizon=True, success=False,
        navigation_decisions=750, stale_navigation_decisions=750, stale_navigation_fraction=1.,
        navigation_pose_age_p95_ms=1000 * REPORT.p95(ages), client_wall_p95_ms=20., full_pipeline_p95_ms=30.,
        schedule_timing=[dict(native_client_s=.02)], response_deliveries=[dict(wall_seconds=.03)])


def test_independent_trace_scoring_preserves_actual_decision_denominator():
    measured = REPORT.rescore(row())
    assert measured['navigation_decisions'] == measured['stale_decisions'] == 750
    assert measured['stale_fraction'] == 1.
    assert measured['clean_goal'] is False


@pytest.mark.parametrize('mutation', ['goal', 'stale', 'contact', 'future', 'age', 'native', 'horizon'])
def test_report_rejects_tampered_result_claims(mutation):
    value = row()
    if mutation == 'goal':
        value['success'] = True
    elif mutation == 'stale':
        value['stale_navigation_decisions'] -= 1
    elif mutation == 'contact':
        value['trace'][-1]['evaluator_contact'] = True
    elif mutation == 'future':
        value['trace'][-1]['native_timestamp_s'] = 100.
    elif mutation == 'age':
        value['navigation_pose_age_p95_ms'] = 0.
    elif mutation == 'native':
        value['client_wall_p95_ms'] = 2.
    else:
        value['elapsed_simulation_s'] = 20.
    with pytest.raises(ValueError):
        REPORT.rescore(value)


def test_missing_navigation_is_unavailable_and_unrun_is_not_a_failure_count():
    expected = [dict(case='straight', seed=750000, condition='nominal', arm='stereo', pipeline=p) for p in REPORT.PIPELINES]
    one = row()
    one.update(stale_navigation_fraction=None, success=False)
    pairs = REPORT.pairing(expected, [one], [dict(expected[1], reason='UNRUN_AFTER_NEGATIVE_NOMINAL_ARM_SMOKE')])
    assert pairs[0]['complete'] is False
    assert pairs[0]['clean_goal_difference'] is None
    assert pairs[0]['legacy_cumulative']['status'] == 'MEASURED'
    assert pairs[0]['bounded_latest']['status'] == 'UNRUN_AFTER_NEGATIVE_NOMINAL_ARM_SMOKE'
    both = copy.deepcopy(one)
    both['pipeline'] = 'bounded_latest'
    paired = REPORT.pairing(expected, [one, both], [])
    assert paired[0]['complete'] is True
    assert paired[0]['clean_goal_difference'] == 0
    assert paired[0]['stale_fraction_difference'] is None


def test_evidence_hash_change_blocks_report(tmp_path):
    path = tmp_path / 'episode.json'
    path.write_text('{}')
    pin = REPORT.sha(path)
    path.write_text('{"success":true}')
    with pytest.raises(ValueError, match='hash mismatch'):
        REPORT.verify_file(path, pin, {})


def test_standalone_figures_handle_unrun_and_missing_denominators(tmp_path):
    groups = []
    for a in REPORT.ARMS:
        for c in REPORT.CONDITIONS:
            for p in REPORT.PIPELINES:
                groups.append(dict(arm=a, condition=c, pipeline=p, measured=4 if p == REPORT.PIPELINES[0] else 0,
                    clean_goals=0, mean_episode_stale_fraction=.8 if p == REPORT.PIPELINES[0] else None))
    expected = [dict(case=case, seed=seed, condition=c, arm=a, pipeline=p)
        for case in ('straight', 'occluders') for seed in (750000, 750001)
        for c in REPORT.CONDITIONS for a in REPORT.ARMS for p in REPORT.PIPELINES]
    pairs = REPORT.pairing(expected, [], expected)
    REPORT.plot(dict(groups=groups, pairs=pairs), tmp_path)
    for name in ('groups.png', 'groups.svg', 'pairs.png', 'pairs.svg'):
        assert (tmp_path / name).stat().st_size > 1000
