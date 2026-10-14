"""Predeclared order-confirmation statistics on synthetic and retained collections."""
import importlib.util
from itertools import combinations, product
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import binomtest, fisher_exact

REPO = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("schedule_order_analysis", REPO / "scripts/bench/processing_schedule_order_analysis.py")
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)

SEEDS = tuple(range(751000, 751008))
CONDITIONS = ("nominal", "camera_10hz", "causal_40ms_exposure")
CASES = ("straight", "occluders")
REF, CAND = A.REFERENCE, A.CANDIDATE


def pairs_for(outcome, *, seeds=SEEDS, latency=None):
    """48 counterbalanced pairs; outcome(i, c, r, first) -> (legacy_goal, bounded_goal)."""
    pairs = []
    for i, seed in enumerate(seeds):
        for c, condition in enumerate(CONDITIONS):
            for r, case in enumerate(CASES):
                first = (REF, CAND)[(i + c + r) % 2]
                legacy, bounded = outcome(i, c, r, first)
                pair = dict(case=case, seed=seed, condition=condition, method="lidar_inertial",
                    legacy_clean_goal=legacy, candidate_clean_goal=bounded, clean_goal_difference=int(bounded) - int(legacy),
                    legacy_stale_fraction=.85, candidate_stale_fraction=.12, stale_fraction_difference=-.73,
                    stale_comparison_available=True, both_full_horizon=True, legacy_fell=False, candidate_fell=False,
                    legacy_collision=False, candidate_collision=False,
                    execution_order=[first, CAND if first == REF else REF], first_pipeline=first,
                    sequence="legacy_first" if first == REF else "bounded_first",
                    legacy_full_pipeline_p95_ms=900. + i, candidate_full_pipeline_p95_ms=60. + i,
                    legacy_navigation_pose_age_p95_ms=1500., candidate_navigation_pose_age_p95_ms=120.,
                    legacy_client_wall_p95_ms=20., candidate_client_wall_p95_ms=25.)
                if latency:
                    latency(pair, i, c, r, first)
                pairs.append(pair)
    return pairs


def collection_for(pairs, design=A.ORDER_DESIGN, **changes):
    result = dict(schema=A.COLLECTION_SCHEMA, status="PASS", design=design, declared_cells=96, measured_cells=2 * len(pairs),
                  unrun_after_nominal_gate=0, missing_cells=[],
                  paired=dict(pairs=pairs, complete_pairs=len(pairs), incomplete_pairs=[]),
                  warmups=dict(episodes=48, scored=False, by_pipeline={}), order_audit=dict(within_arm_counterbalanced=True))
    result.update(changes)
    return result


def strong(i, c, r, first):
    return (i, c, r) == (5, 1, 0), (i, c, r) != (3, 0, 1)


def test_strong_counterbalanced_gain_is_confirmed_with_exact_statistics():
    result = A.analyse(collection_for(pairs_for(strong)))
    verdict, arm = result["verdict"], result["arms"]["lidar_inertial"]
    goals = arm["goals"]
    assert verdict["status"] == "CONFIRMED" and all(verdict["checks"].values())
    assert (goals["legacy_goals"], goals["bounded_goals"], goals["pairs"]) == (1, 47, 48)
    assert goals["mcnemar"]["bounded_only"] == 46 and goals["mcnemar"]["legacy_only"] == 0
    assert goals["mcnemar"]["p_value"] == pytest.approx(binomtest(46, 46, .5).pvalue)
    ci = binomtest(47, 48).proportion_ci(confidence_level=.95, method="wilson")
    assert goals["bounded_wilson95"] == pytest.approx([ci.low, ci.high])
    boot = goals["seed_cluster_bootstrap95"]
    assert "6435" in boot["method"] and boot["clusters"] == 8 and boot["interval"][0] > 0
    assert boot["estimate"] == pytest.approx(46 / 48) == goals["rate_difference"]
    assert goals["seed_sign_test"]["positive_seeds"] == 8 and goals["seed_sign_test"]["p_value"] == pytest.approx(2 * .5 ** 8)
    assert arm["order"]["status"] == "ESTIMATED" and "ORDER_EFFECT_DETECTED" not in verdict["flags"]
    assert arm["order"]["sequence_pairs"] == {"legacy_first": 24, "bounded_first": 24}
    assert len(arm["seed_variability"]["per_seed"]) == 8 and set(arm["per_condition"]) == set(CONDITIONS)
    latency = arm["continuous"]["modeled_pipeline_p95_ms"]
    assert latency["available_pairs"] == 48 and latency["difference"]["mean"] == pytest.approx(-840.)
    assert latency["seed_cluster_bootstrap95"]["interval"] == pytest.approx([-840., -840.])
    assert arm["continuous"]["stale_fraction"]["difference"]["mean"] == pytest.approx(-.73)
    json.dumps(result, allow_nan=False)


def test_pure_order_artifact_is_negative_and_flags_the_order_effect():
    artifact = lambda i, c, r, first: (first == REF, first == CAND)
    result = A.analyse(collection_for(pairs_for(artifact)))
    verdict, order = result["verdict"], result["arms"]["lidar_inertial"]["order"]
    assert verdict["status"] == "NOT_CONFIRMED" and "ORDER_EFFECT_DETECTED" in verdict["flags"]
    assert not verdict["checks"]["mcnemar_p_below_alpha"] and not verdict["checks"]["gain_in_legacy_first_pairs"]
    assert order["first_position_test"]["first_run_wins"] == 48
    assert order["first_position_test"]["p_value"] == pytest.approx(binomtest(48, 48, .5).pvalue)
    assert order["sequence_by_winner_fisher"]["table"] == [[0, 24], [24, 0]]
    assert order["sequence_by_winner_fisher"]["p_value"] == pytest.approx(fisher_exact([[0, 24], [24, 0]]).pvalue)
    assert order["goal_period_effect"] == pytest.approx(1.) and order["goal_schedule_effect_adjusted"] == pytest.approx(0.)


def test_null_and_reversed_outcomes_are_negative():
    null = A.analyse(collection_for(pairs_for(lambda i, c, r, first: (False, False))))
    assert null["verdict"]["status"] == "NOT_CONFIRMED"
    assert null["arms"]["lidar_inertial"]["goals"]["mcnemar"]["p_value"] == 1. and null["arms"]["lidar_inertial"]["goals"]["mcnemar"]["discordant"] == 0
    reversed_ = A.analyse(collection_for(pairs_for(lambda i, c, r, first: (True, False))))
    assert reversed_["verdict"]["status"] == "NOT_CONFIRMED" and "REVERSED" in reversed_["verdict"]["flags"]


def test_confirmed_but_smaller_gain_is_flagged_attenuated():
    half = lambda i, c, r, first: (False, c == 0 or (c == 1 and r == 0))  # 24/48 bounded, 0 legacy
    result = A.analyse(collection_for(pairs_for(half)))
    assert result["verdict"]["status"] == "CONFIRMED" and "ATTENUATED" in result["verdict"]["flags"]
    assert result["verdict"]["bootstrap_interval95"][1] < A.V1_LIDAR_DIFFERENCE
    assert result["verdict"]["v1_reference"]["inside_new_interval"] is False


@pytest.mark.parametrize("change", ["missing_pair", "unrun", "incomplete_pair", "measured", "status"])
def test_incomplete_or_gated_primary_cohort_has_no_verdict(change):
    pairs = pairs_for(strong)
    collection = collection_for(pairs)
    if change == "missing_pair":
        collection = collection_for(pairs[:-1], measured_cells=96)
    elif change == "unrun":
        collection["unrun_after_nominal_gate"] = 2
    elif change == "incomplete_pair":
        collection["paired"]["incomplete_pairs"] = [dict(case="straight", seed=751000)]
    elif change == "measured":
        collection["measured_cells"] = 94
    else:
        collection["status"] = "INCOMPLETE"
        with pytest.raises(ValueError):
            A.analyse(collection)
        return
    assert A.analyse(collection)["verdict"]["status"] == "INCOMPLETE_NO_VERDICT"


def test_v1_collection_is_descriptive_only_and_order_is_not_estimable():
    v1 = json.loads((REPO / "results/processing-schedule-20261012/collection.json").read_text())
    result = A.analyse(v1)
    assert result["design"] == A.V1_DESIGN and result["verdict"]["status"] == "NOT_CONFIRMATORY_DESIGN"
    lidar = result["arms"]["lidar_inertial"]
    assert (lidar["goals"]["legacy_goals"], lidar["goals"]["bounded_goals"], lidar["pairs"]) == (0, 11, 12)
    assert lidar["goals"]["mcnemar"]["p_value"] == pytest.approx(2 * .5 ** 11)
    assert lidar["order"]["status"] == "NOT_AVAILABLE_NO_RECORDED_ORDER"
    assert lidar["continuous"]["modeled_pipeline_p95_ms"]["available_pairs"] == 0
    assert result["arms"]["stereo"]["goals"]["mcnemar"]["discordant"] == 0
    single = pairs_for(strong)
    for pair in single:
        pair.update(first_pipeline=CAND, execution_order=[CAND, REF])
    confounded = A.analyse(collection_for(single, design=A.V1_DESIGN))
    assert confounded["arms"]["lidar_inertial"]["order"]["status"] == "NOT_ESTIMABLE_SINGLE_SEQUENCE"


@pytest.mark.parametrize("clusters", [[[1, 0, 1], [0], [1, 1], [0, 0, 1, 1]], [[.5, -.25], [2.], [-1., 0., 3.]],
                                      [[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [1] * 6, [0, 0, 1, 1, 1, 1], [1] * 6]])
def test_exact_multiset_bootstrap_equals_brute_force_ordered_resampling(clusters):
    result = A.cluster_bootstrap(clusters)
    g = len(clusters)
    stats = [sum(sum(clusters[j]) for j in choice) / sum(len(clusters[j]) for j in choice) for choice in product(range(g), repeat=g)]
    expected = np.quantile(stats, [.025, .975], method="inverted_cdf")
    assert result["interval"] == pytest.approx(list(expected))
    assert result["estimate"] == pytest.approx(sum(map(sum, clusters)) / sum(map(len, clusters)))


def test_bootstrap_skips_empty_clusters_and_uses_seeded_monte_carlo_above_limit(monkeypatch):
    assert A.cluster_bootstrap([]) is None and A.cluster_bootstrap([[], []]) is None
    result = A.cluster_bootstrap([[1., 1.], [], [0.]])
    assert result["clusters"] == 3 and 0 <= result["interval"][0] <= result["interval"][1] <= 1
    monkeypatch.setattr(A, "EXACT_LIMIT", 10)
    first, second = A.cluster_bootstrap([[1.], [0.], [1.], [1.]]), A.cluster_bootstrap([[1.], [0.], [1.], [1.]])
    assert "Monte Carlo" in first["method"] and first == second


def test_permutation_test_is_exact_for_small_groups_and_detects_latency_period_effect():
    a, b = [3., 1., 4.], [1., 5., 9., 2.]
    observed = np.mean(a) - np.mean(b)
    pooled = a + b
    brute = [np.mean([pooled[i] for i in idx]) - np.mean([pooled[i] for i in range(7) if i not in idx]) for idx in combinations(range(7), 3)]
    test = A.permutation_test(a, b)
    assert test["p_value"] == pytest.approx(np.mean(np.abs(brute) >= abs(observed) - 1e-9)) and "exact" in test["method"]
    def planted(pair, i, c, r, first):
        # bounded minus legacy = -800 ms schedule effect; +50 ms whoever runs first
        legacy, bounded = 900. + 7 * i + 3 * c, 100. + 7 * i + 3 * c
        if first == REF:
            legacy += 50.
        else:
            bounded += 50.
        pair.update(legacy_full_pipeline_p95_ms=legacy, candidate_full_pipeline_p95_ms=bounded)
    result = A.analyse(collection_for(pairs_for(strong, latency=planted)))
    effect = result["arms"]["lidar_inertial"]["order"]["continuous_period_effects"]["modeled_pipeline_p95_ms"]
    assert effect["period_effect"] == pytest.approx(50.) and effect["schedule_effect_adjusted"] == pytest.approx(-800.)
    assert effect["permutation"]["p_value"] < .001 and "Monte Carlo" in effect["permutation"]["method"]
    assert result["arms"]["lidar_inertial"]["order"]["continuous_period_effects"]["navigation_pose_age_p95_ms"]["permutation"]["p_value"] == 1.


def test_missing_latency_values_are_excluded_per_metric_and_counted():
    def missing(pair, i, c, r, first):
        if (i, c, r) == (0, 0, 0):
            pair["candidate_navigation_pose_age_p95_ms"] = None
    arm = A.analyse(collection_for(pairs_for(strong, latency=missing)))["arms"]["lidar_inertial"]
    assert arm["continuous"]["navigation_pose_age_p95_ms"]["available_pairs"] == 47
    assert arm["continuous"]["navigation_pose_age_p95_ms"]["excluded_pairs"] == 1
    assert arm["continuous"]["modeled_pipeline_p95_ms"]["available_pairs"] == 48 and arm["goals"]["pairs"] == 48


def test_zero_discordant_pairs_do_not_call_binomtest_with_zero_trials():
    assert A.exact_mcnemar(0, 0)["p_value"] == 1. and A.wilson(0, 0) is None
    assert A.seed_sign_test([0., 0.])["p_value"] == 1.


def test_inconsistent_pair_records_are_rejected():
    pairs = pairs_for(strong)
    pairs[3]["clean_goal_difference"] = 0
    with pytest.raises(ValueError, match="disagrees"):
        A.analyse(collection_for(pairs))
    pairs = pairs_for(strong)
    pairs[4]["execution_order"] = [CAND, CAND]
    with pytest.raises(ValueError, match="execution order"):
        A.analyse(collection_for(pairs))


def test_confirmatory_constants_match_the_frozen_protocol():
    spec = importlib.util.spec_from_file_location("schedule_campaign_for_analysis_test", REPO / "scripts/bench/processing_schedule_campaign.py")
    campaign = importlib.util.module_from_spec(spec)
    previous = sys.path.copy()
    try:
        spec.loader.exec_module(campaign)
    finally:
        sys.path[:] = previous
    protocol = campaign.declared_protocol(campaign.ORDER_DESIGN)
    expected = A.CONFIRMATORY[A.ORDER_DESIGN]
    assert expected == dict(arm=protocol["primary_outcome"]["arm"], pairs=protocol["primary_outcome"]["pairs"],
                            seeds=len(protocol["development"]["seeds"]))
    assert (A.ALPHA, A.RNG_SEED, A.RESAMPLES, A.EXACT_LIMIT) == (protocol["primary_outcome"]["alpha"], 20261014, 20000, 200000)
    assert A.V1_DESIGN == campaign.V1_DESIGN and A.ORDER_DESIGN == campaign.ORDER_DESIGN


def test_command_line_writes_reports_once(tmp_path):
    collection = tmp_path / "campaign_result.json"
    collection.write_text(json.dumps(collection_for(pairs_for(strong))))
    assert A.main(["--collection", str(collection), "--output", str(tmp_path / "analysis")]) == 0
    written = json.loads((tmp_path / "analysis/analysis.json").read_text())
    assert written["verdict"]["status"] == "CONFIRMED" and written["schema"] == A.SCHEMA
    assert "CONFIRMED" in (tmp_path / "analysis/analysis.md").read_text()
    with pytest.raises(FileExistsError):
        A.main(["--collection", str(collection), "--output", str(tmp_path / "analysis")])
