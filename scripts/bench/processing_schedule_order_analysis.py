#!/usr/bin/env python3
"""Predeclared analysis of the counterbalanced processing-schedule confirmation.

Reads the strict collector output (the collection stage campaign_result.json)
and writes analysis.json and analysis.md. The statistics are those frozen in
the order-confirmation-20261014 scientific protocol: per-pair differences,
exact McNemar, Wilson intervals, seed-cluster bootstrap intervals by exact
multiset enumeration, a seed-level sign test, latency differences, variability
across seeds and explicit first-versus-second-position (order) tests.

Collections of the order-confounded v1 design are analysed descriptively and
can never be CONFIRMED. Only the standard library, numpy and scipy are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations, combinations_with_replacement
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, fisher_exact

SCHEMA = "bhl-processing-schedule-order-analysis-v1"
COLLECTION_SCHEMA = "bhl-processing-schedule-v1-collection"
REFERENCE, CANDIDATE = "legacy_cumulative", "bounded_latest"
V1_DESIGN, ORDER_DESIGN = "v1-20261012", "order-confirmation-20261014"
# Primary cohort of the confirmatory design (cross-checked against the frozen
# protocol by tests/test_processing_schedule_order_analysis.py).
CONFIRMATORY = {ORDER_DESIGN: dict(arm="lidar_inertial", pairs=48, seeds=8)}
ALPHA, LEVEL = .05, .95
EXACT_LIMIT, RESAMPLES, RNG_SEED = 200_000, 20_000, 20261014
V1_LIDAR_DIFFERENCE = 11 / 12
LATENCY = {"modeled_pipeline_p95_ms": "full_pipeline_p95_ms",
           "navigation_pose_age_p95_ms": "navigation_pose_age_p95_ms",
           "native_client_p95_ms": "client_wall_p95_ms"}


def wilson(k, n):
    """Wilson score interval for k successes out of n episodes."""
    if n == 0:
        return None
    ci = binomtest(k, n).proportion_ci(confidence_level=LEVEL, method="wilson")
    return [float(ci.low), float(ci.high)]


def exact_mcnemar(bounded_only, legacy_only):
    """Exact two-sided McNemar test: binomial test on the discordant pairs."""
    discordant = bounded_only + legacy_only
    if discordant == 0:
        return dict(bounded_only=0, legacy_only=0, discordant=0, p_value=1.0, discordant_share_bounded=None,
                    discordant_share_interval=None, note="no discordant pairs; exact test undefined, reported as p = 1")
    test = binomtest(bounded_only, discordant, .5, alternative="two-sided")
    ci = test.proportion_ci(confidence_level=LEVEL, method="exact")
    return dict(bounded_only=bounded_only, legacy_only=legacy_only, discordant=discordant, p_value=float(test.pvalue),
                discordant_share_bounded=bounded_only / discordant, discordant_share_interval=[float(ci.low), float(ci.high)])


def weighted_percentiles(values, weights, levels):
    """Inverted-CDF percentiles: smallest value whose cumulative weight reaches each level."""
    order = np.argsort(np.asarray(values, float), kind="stable")
    values, weights = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    cdf = np.cumsum(weights) / weights.sum()
    return [float(values[min(int(np.searchsorted(cdf, q - 1e-12, side="left")), len(values) - 1)]) for q in levels]


def cluster_bootstrap(clusters, level=LEVEL):
    """Seed-cluster bootstrap of the pooled mean; clusters are lists of values.

    All multisets of resampled clusters are enumerated with multinomial weights
    (C(2G-1, G) of them, 6435 for eight seeds), which equals enumerating all
    G**G ordered resamples. Monte Carlo is used only above EXACT_LIMIT.
    """
    clusters = [[float(v) for v in cluster] for cluster in clusters]
    totals = np.array([sum(c) for c in clusters], float)
    counts = np.array([len(c) for c in clusters], float)
    groups = len(clusters)
    if groups == 0 or counts.sum() == 0:
        return None
    tails = ((1 - level) / 2, 1 - (1 - level) / 2)
    multisets = math.comb(2 * groups - 1, groups)
    if multisets <= EXACT_LIMIT:
        values, weights = [], []
        log_scale = math.lgamma(groups + 1) - groups * math.log(groups)
        for combo in combinations_with_replacement(range(groups), groups):
            repeats = Counter(combo)
            size = sum(counts[j] * m for j, m in repeats.items())
            if size == 0:
                continue
            values.append(sum(totals[j] * m for j, m in repeats.items()) / size)
            weights.append(math.exp(log_scale - sum(math.lgamma(m + 1) for m in repeats.values())))
        method = f"exact enumeration of {multisets} cluster multisets with multinomial weights"
    else:
        draws = np.random.default_rng(RNG_SEED).integers(0, groups, size=(RESAMPLES, groups))
        sizes = counts[draws].sum(axis=1)
        keep = sizes > 0
        values = totals[draws].sum(axis=1)[keep] / sizes[keep]
        weights = np.ones(len(values))
        method = f"Monte Carlo, {RESAMPLES} cluster resamples, numpy seed {RNG_SEED}"
    low, high = weighted_percentiles(values, weights, tails)
    return dict(estimate=float(totals.sum() / counts.sum()), interval=[low, high], level=level, clusters=groups,
                method=method, percentile="inverted CDF")


def seed_sign_test(per_seed):
    positive, negative = sum(v > 0 for v in per_seed), sum(v < 0 for v in per_seed)
    p_value = float(binomtest(positive, positive + negative, .5).pvalue) if positive + negative else 1.0
    return dict(positive_seeds=positive, negative_seeds=negative, zero_seeds=len(per_seed) - positive - negative, p_value=p_value)


def permutation_test(first, second):
    """Two-sided permutation p-value for mean(first) - mean(second)."""
    first, second = np.asarray(first, float), np.asarray(second, float)
    if len(first) == 0 or len(second) == 0:
        return None
    pooled, k = np.concatenate([first, second]), len(first)
    n, total = len(pooled), pooled.sum()
    observed = first.mean() - second.mean()
    assignments = math.comb(n, k)
    if assignments <= EXACT_LIMIT:
        sums = np.array([pooled[list(index)].sum() for index in combinations(range(n), k)])
        stats = sums / k - (total - sums) / (n - k)
        p_value = float(np.mean(np.abs(stats) >= abs(observed) - 1e-9))
        method = f"exact over {assignments} assignments"
    else:
        order = np.random.default_rng(RNG_SEED).random((RESAMPLES, n)).argsort(axis=1)
        sums = pooled[order[:, :k]].sum(axis=1)
        stats = sums / k - (total - sums) / (n - k)
        p_value = float((np.sum(np.abs(stats) >= abs(observed) - 1e-9) + 1) / (RESAMPLES + 1))
        method = f"Monte Carlo, {RESAMPLES} permutations, numpy seed {RNG_SEED}"
    return dict(difference=float(observed), p_value=p_value, method=method)


def mean(values):
    return float(np.mean(values)) if len(values) else None


def describe(values):
    values = [float(v) for v in values]
    return dict(n=len(values), mean=mean(values), median=float(np.median(values)) if values else None,
                sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def pair_difference(pair, metric):
    legacy, candidate = pair.get("legacy_" + metric), pair.get("candidate_" + metric)
    if legacy is None or candidate is None:
        return None
    return float(candidate) - float(legacy)


def by_seed(pairs, value):
    """Cluster lists keyed by seed in sorted order, skipping unavailable values."""
    seeds = sorted({p["seed"] for p in pairs})
    return seeds, [[v for v in (value(p) for p in pairs if p["seed"] == seed) if v is not None] for seed in seeds]


def goal_summary(pairs):
    n = len(pairs)
    legacy = sum(bool(p["legacy_clean_goal"]) for p in pairs)
    bounded = sum(bool(p["candidate_clean_goal"]) for p in pairs)
    bounded_only = sum(bool(p["candidate_clean_goal"]) and not p["legacy_clean_goal"] for p in pairs)
    legacy_only = sum(bool(p["legacy_clean_goal"]) and not p["candidate_clean_goal"] for p in pairs)
    return dict(pairs=n, legacy_goals=legacy, bounded_goals=bounded,
                legacy_rate=legacy / n if n else None, bounded_rate=bounded / n if n else None,
                legacy_wilson95=wilson(legacy, n), bounded_wilson95=wilson(bounded, n),
                rate_difference=(bounded - legacy) / n if n else None, mcnemar=exact_mcnemar(bounded_only, legacy_only))


def continuous_differences(pairs):
    metrics = dict(LATENCY, stale_fraction="stale_fraction")
    result = {}
    for name, metric in metrics.items():
        values = [pair_difference(p, metric) for p in pairs]
        present = [v for v in values if v is not None]
        _, clusters = by_seed(pairs, lambda p, metric=metric: pair_difference(p, metric))
        result[name] = dict(field=metric, available_pairs=len(present), excluded_pairs=len(values) - len(present),
            legacy_mean=mean([p["legacy_" + metric] for p in pairs if p.get("legacy_" + metric) is not None]),
            bounded_mean=mean([p["candidate_" + metric] for p in pairs if p.get("candidate_" + metric) is not None]),
            difference=describe(present), seed_cluster_bootstrap95=cluster_bootstrap(clusters) if present else None,
            direction="bounded_latest minus legacy_cumulative; negative is lower for the candidate")
    return result


def seed_variability(pairs):
    rows = []
    for seed in sorted({p["seed"] for p in pairs}):
        chosen = [p for p in pairs if p["seed"] == seed]
        row = dict(seed=seed, pairs=len(chosen), legacy_goals=sum(bool(p["legacy_clean_goal"]) for p in chosen),
                   bounded_goals=sum(bool(p["candidate_clean_goal"]) for p in chosen),
                   goal_difference=mean([int(bool(p["candidate_clean_goal"])) - int(bool(p["legacy_clean_goal"])) for p in chosen]))
        for name, metric in dict(LATENCY, stale_fraction="stale_fraction").items():
            row[name + "_difference"] = mean([v for v in (pair_difference(p, metric) for p in chosen) if v is not None])
        rows.append(row)
    summary = {key: describe([r[key] for r in rows if r[key] is not None]) for key in
               ["goal_difference"] + [name + "_difference" for name in list(LATENCY) + ["stale_fraction"]]}
    return dict(per_seed=rows, across_seeds=summary)


def order_effects(pairs):
    """First versus second execution position within one arm."""
    if not pairs or any(p.get("first_pipeline") not in (REFERENCE, CANDIDATE) for p in pairs):
        return dict(status="NOT_AVAILABLE_NO_RECORDED_ORDER")
    sequences = {REFERENCE: [p for p in pairs if p["first_pipeline"] == REFERENCE],
                 CANDIDATE: [p for p in pairs if p["first_pipeline"] == CANDIDATE]}
    counts = {name: len(rows) for name, rows in sequences.items()}
    def winner(p):
        return CANDIDATE if p["candidate_clean_goal"] else REFERENCE
    discordant = [p for p in pairs if bool(p["candidate_clean_goal"]) != bool(p["legacy_clean_goal"])]
    first_wins = sum(winner(p) == p["first_pipeline"] for p in discordant)
    table = [[sum(winner(p) == CANDIDATE for p in discordant if p["first_pipeline"] == first),
              sum(winner(p) == REFERENCE for p in discordant if p["first_pipeline"] == first)] for first in (REFERENCE, CANDIDATE)]
    position = {}
    for pipeline, prefix in ((REFERENCE, "legacy"), (CANDIDATE, "candidate")):
        first = [bool(p[prefix + "_clean_goal"]) for p in pairs if p["first_pipeline"] == pipeline]
        second = [bool(p[prefix + "_clean_goal"]) for p in pairs if p["first_pipeline"] != pipeline]
        position[pipeline] = dict(first_position=dict(episodes=len(first), goals=sum(first), wilson95=wilson(sum(first), len(first))),
                                  second_position=dict(episodes=len(second), goals=sum(second), wilson95=wilson(sum(second), len(second))))
    goal_by_sequence = {name: dict(pairs=len(rows), bounded_goals=sum(bool(p["candidate_clean_goal"]) for p in rows),
                                   legacy_goals=sum(bool(p["legacy_clean_goal"]) for p in rows),
                                   mcnemar=exact_mcnemar(sum(bool(p["candidate_clean_goal"]) and not p["legacy_clean_goal"] for p in rows),
                                                         sum(bool(p["legacy_clean_goal"]) and not p["candidate_clean_goal"] for p in rows)))
                        for name, rows in (("legacy_first", sequences[REFERENCE]), ("bounded_first", sequences[CANDIDATE]))}
    result = dict(sequence_pairs=dict(legacy_first=counts[REFERENCE], bounded_first=counts[CANDIDATE]),
                  goal_by_sequence=goal_by_sequence, goal_rate_by_position=position)
    if not counts[REFERENCE] or not counts[CANDIDATE]:
        result.update(status="NOT_ESTIMABLE_SINGLE_SEQUENCE",
                      note="every pair of this arm ran in one order; schedule and order effects are confounded")
        return result
    goal_difference = {name: mean([int(bool(p["candidate_clean_goal"])) - int(bool(p["legacy_clean_goal"])) for p in rows])
                       for name, rows in sequences.items()}
    result.update(status="ESTIMATED",
        first_position_test=dict(discordant=len(discordant), first_run_wins=first_wins,
            p_value=float(binomtest(first_wins, len(discordant), .5).pvalue) if discordant else 1.0,
            note="valid as a period test when discordant pairs are balanced across sequences"),
        sequence_by_winner_fisher=dict(table_rows=["legacy_first", "bounded_first"], table_columns=["bounded_won", "legacy_won"],
            table=table, p_value=float(fisher_exact(table).pvalue)),
        goal_period_effect=(goal_difference[CANDIDATE] - goal_difference[REFERENCE]) / 2,
        goal_schedule_effect_adjusted=(goal_difference[CANDIDATE] + goal_difference[REFERENCE]) / 2)
    latency = {}
    for name, metric in dict(LATENCY, stale_fraction="stale_fraction").items():
        groups = {first: [v for v in (pair_difference(p, metric) for p in rows) if v is not None] for first, rows in sequences.items()}
        test = permutation_test(groups[CANDIDATE], groups[REFERENCE])
        latency[name] = dict(legacy_first_pairs=len(groups[REFERENCE]), bounded_first_pairs=len(groups[CANDIDATE]),
            period_effect=test["difference"] / 2 if test else None,
            schedule_effect_adjusted=(mean(groups[CANDIDATE]) + mean(groups[REFERENCE])) / 2 if test else None,
            permutation=test,
            definition="period effect = (mean bounded-minus-legacy in bounded-first pairs - same in legacy-first pairs) / 2; positive means the first position yields the larger value")
    result["continuous_period_effects"] = latency
    return result


def per_condition(pairs):
    return {condition: goal_summary([p for p in pairs if p["condition"] == condition])
            for condition in sorted({p["condition"] for p in pairs})}


def safety(pairs):
    return {prefix: dict(falls=sum(bool(p.get(prefix + "_fell")) for p in pairs),
                         contacts=sum(bool(p.get(prefix + "_collision")) for p in pairs),
                         recorded=all(prefix + "_fell" in p for p in pairs))
            for prefix in ("legacy", "candidate")}


def analyse_arm(pairs):
    goals = goal_summary(pairs)
    seeds, clusters = by_seed(pairs, lambda p: int(bool(p["candidate_clean_goal"])) - int(bool(p["legacy_clean_goal"])))
    goals["seed_cluster_bootstrap95"] = cluster_bootstrap(clusters)
    goals["seed_sign_test"] = seed_sign_test([float(np.mean(c)) for c in clusters])
    goals["per_pair_differences"] = [dict(seed=p["seed"], condition=p["condition"], case=p["case"],
        first_pipeline=p.get("first_pipeline"), difference=int(bool(p["candidate_clean_goal"])) - int(bool(p["legacy_clean_goal"])))
        for p in pairs]
    return dict(pairs=len(pairs), seeds=seeds, goals=goals, continuous=continuous_differences(pairs),
                seed_variability=seed_variability(pairs), order=order_effects(pairs),
                per_condition=per_condition(pairs), safety=safety(pairs))


def check_pairs(pairs):
    for p in pairs:
        if type(p.get("legacy_clean_goal")) is not bool or type(p.get("candidate_clean_goal")) is not bool:
            raise ValueError("explicit Boolean clean-goal outcomes required")
        if p.get("clean_goal_difference") != int(p["candidate_clean_goal"]) - int(p["legacy_clean_goal"]):
            raise ValueError("pair clean-goal difference disagrees with its outcomes")
        if "execution_order" in p and (sorted(p["execution_order"]) != sorted((REFERENCE, CANDIDATE))
                                       or p["execution_order"][0] != p.get("first_pipeline")):
            raise ValueError("pair execution order is not one of the two declared sequences")


def verdict(collection, arms):
    design = collection.get("design", V1_DESIGN)
    spec = CONFIRMATORY.get(design)
    if spec is None:
        return dict(status="NOT_CONFIRMATORY_DESIGN", design=design,
                    reason="design without within-arm counterbalanced order; descriptive analysis only")
    arm = arms.get(spec["arm"])
    complete = (collection.get("status") == "PASS" and collection.get("missing_cells") == []
                and collection.get("unrun_after_nominal_gate") == 0
                and collection.get("measured_cells") == collection.get("declared_cells")
                and not collection["paired"].get("incomplete_pairs")
                and arm is not None and arm["pairs"] == spec["pairs"] and len(arm["seeds"]) == spec["seeds"])
    if not complete:
        return dict(status="INCOMPLETE_NO_VERDICT", design=design, arm=spec["arm"],
                    reason="primary cohort incomplete, gated or missing; no primary verdict is computed")
    goals, order = arm["goals"], arm["order"]
    mcnemar, interval = goals["mcnemar"], goals["seed_cluster_bootstrap95"]["interval"]
    by_sequence = order.get("goal_by_sequence", {})
    checks = dict(mcnemar_p_below_alpha=mcnemar["p_value"] < ALPHA,
                  bounded_only_exceeds_legacy_only=mcnemar["bounded_only"] > mcnemar["legacy_only"],
                  bootstrap_lower_bound_positive=interval[0] > 0,
                  gain_in_legacy_first_pairs=by_sequence.get("legacy_first", {}).get("bounded_goals", 0) > by_sequence.get("legacy_first", {}).get("legacy_goals", 0),
                  gain_in_bounded_first_pairs=by_sequence.get("bounded_first", {}).get("bounded_goals", 0) > by_sequence.get("bounded_first", {}).get("legacy_goals", 0))
    status = "CONFIRMED" if all(checks.values()) else "NOT_CONFIRMED"
    flags = []
    order_p = [order.get("first_position_test", {}).get("p_value", 1.), order.get("sequence_by_winner_fisher", {}).get("p_value", 1.)]
    if order.get("status") != "ESTIMATED":
        flags.append("ORDER_TEST_NOT_ESTIMABLE")
    elif min(order_p) < ALPHA:
        flags.append("ORDER_EFFECT_DETECTED")
    if status == "CONFIRMED" and interval[1] < V1_LIDAR_DIFFERENCE:
        flags.append("ATTENUATED")
    if mcnemar["legacy_only"] > mcnemar["bounded_only"] and mcnemar["p_value"] < ALPHA:
        flags.append("REVERSED")
    return dict(status=status, design=design, arm=spec["arm"], checks=checks, flags=flags,
                goal_rate_difference=goals["rate_difference"], bootstrap_interval95=interval, mcnemar_p_value=mcnemar["p_value"],
                v1_reference=dict(goal_rate_difference=V1_LIDAR_DIFFERENCE, legacy="0/12", bounded="11/12",
                                  inside_new_interval=interval[0] <= V1_LIDAR_DIFFERENCE <= interval[1]))


def analyse(collection):
    if collection.get("schema") != COLLECTION_SCHEMA or collection.get("status") != "PASS":
        raise ValueError("a passing strict processing-schedule collection is required")
    pairs = collection["paired"]["pairs"]
    check_pairs(pairs)
    arms = {arm: analyse_arm([p for p in pairs if p["method"] == arm]) for arm in sorted({p["method"] for p in pairs})}
    warmups = collection.get("warmups")
    return dict(schema=SCHEMA, design=collection.get("design", V1_DESIGN), verdict=verdict(collection, arms),
        collection=dict(declared_cells=collection.get("declared_cells"), measured_cells=collection.get("measured_cells"),
                        unrun_after_nominal_gate=collection.get("unrun_after_nominal_gate"),
                        complete_pairs=collection["paired"].get("complete_pairs"),
                        incomplete_pairs=len(collection["paired"].get("incomplete_pairs", [])),
                        order_audit=collection.get("order_audit")),
        arms=arms,
        warmups=None if warmups is None else dict(episodes=warmups.get("episodes"), scored=warmups.get("scored"),
                                                  by_pipeline=warmups.get("by_pipeline")),
        settings=dict(alpha=ALPHA, level=LEVEL, exact_limit=EXACT_LIMIT, monte_carlo_resamples=RESAMPLES, rng_seed=RNG_SEED,
                      wilson="scipy.stats.binomtest(k, n).proportion_ci(method='wilson')",
                      mcnemar="scipy.stats.binomtest(bounded_only, discordant, 0.5), two-sided"),
        scope="simulation schedule-model statistics over reset-seed clusters; no hardware, population or stereo claims")


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(fmt(v, digits) for v in value) + "]"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(result):
    v = result["verdict"]
    lines = ["# Processing-schedule order analysis", "",
             f"Design: `{result['design']}`. Primary verdict: **{v['status']}**"
             + (f" (flags: {', '.join(v['flags'])})" if v.get("flags") else "") + ".", ""]
    if "checks" in v:
        lines += ["| Predeclared condition | Met |", "| --- | --- |"] + [f"| {k} | {m} |" for k, m in v["checks"].items()] + [""]
    for arm, data in result["arms"].items():
        g = data["goals"]
        boot = g["seed_cluster_bootstrap95"]
        lines += [f"## {arm}: {data['pairs']} pairs, {len(data['seeds'])} seeds", "",
                  "| Pipeline | Clean goals | Wilson 95% |", "| --- | ---: | --- |",
                  f"| legacy_cumulative | {g['legacy_goals']}/{g['pairs']} | {fmt(g['legacy_wilson95'])} |",
                  f"| bounded_latest | {g['bounded_goals']}/{g['pairs']} | {fmt(g['bounded_wilson95'])} |", "",
                  f"Goal-rate difference {fmt(g['rate_difference'])}; seed-cluster bootstrap 95% "
                  f"{fmt(boot['interval'] if boot else None)} ({boot['method'] if boot else 'n/a'}). Exact McNemar: "
                  f"{g['mcnemar']['bounded_only']} bounded-only vs {g['mcnemar']['legacy_only']} legacy-only, p = {fmt(g['mcnemar']['p_value'], 6)}. "
                  f"Seed sign test p = {fmt(g['seed_sign_test']['p_value'], 4)}.", "",
                  "| Paired difference (bounded - legacy) | Pairs | Mean | Median | Bootstrap 95% |", "| --- | ---: | ---: | ---: | --- |"]
        for name, c in data["continuous"].items():
            b = c["seed_cluster_bootstrap95"]
            lines.append(f"| {name} | {c['available_pairs']} | {fmt(c['difference']['mean'])} | {fmt(c['difference']['median'])} | {fmt(b['interval'] if b else None)} |")
        o = data["order"]
        lines += ["", f"Order test status: {o['status']}."]
        if o["status"] == "ESTIMATED":
            lines.append(f"First-run wins among discordant pairs: {o['first_position_test']['first_run_wins']}/"
                         f"{o['first_position_test']['discordant']} (p = {fmt(o['first_position_test']['p_value'], 4)}); "
                         f"sequence-by-winner Fisher p = {fmt(o['sequence_by_winner_fisher']['p_value'], 4)}; "
                         f"goal period effect {fmt(o['goal_period_effect'])}.")
        across = data["seed_variability"]["across_seeds"]["goal_difference"]
        lines += ["", f"Across seeds, goal difference mean {fmt(across['mean'])}, SD {fmt(across['sd'])}, "
                      f"range {fmt(across['minimum'])} to {fmt(across['maximum'])}.", ""]
    if result.get("warmups"):
        lines += [f"Unscored warm-ups (reported separately): {result['warmups']['episodes']} episodes.", ""]
    lines.append(result["scope"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=Path, help="strict collector campaign_result.json")
    parser.add_argument("--output", required=True, type=Path, help="new directory for analysis.json and analysis.md")
    args = parser.parse_args(argv)
    result = analyse(json.loads(args.collection.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "analysis.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (args.output / "analysis.md").open("x") as stream:
        stream.write(markdown(result))
    print(json.dumps(dict(design=result["design"], verdict=result["verdict"]["status"], flags=result["verdict"].get("flags", [])), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
