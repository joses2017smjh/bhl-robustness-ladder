"""Complete-cohort paired outcomes; incomplete runs never become zero success."""
from itertools import product
import numpy as np

CASES = ("straight", "dogleg", "occluders")
SEEDS = (480000, 480001, 480002)
ARMS = ("permanent_stop", "verified_lio_recovery")


def aggregate(pairs):
    expected = set(product(CASES, SEEDS, ARMS))
    indexed = {}
    for row in pairs:
        key = row.get("case"), row.get("seed"), row.get("arm")
        if key not in expected or key in indexed:
            raise ValueError("unknown or duplicate predeclared stereo pair")
        for field in ("clean_goal", "falls", "contacts"):
            if type(row.get(field)) is not bool:
                raise ValueError("explicit measured Boolean episode outcome required")
        if type(row.get("verified_recoveries")) is not int or row["verified_recoveries"] < 0:
            raise ValueError("nonnegative measured recovery count required")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("all 18 declared outcomes required before aggregate verdict")
    arms = {arm: {"episodes": 9, "clean_goals": sum(row["clean_goal"] for key, row in indexed.items() if key[2] == arm),
                  "falls": sum(row["falls"] for key, row in indexed.items() if key[2] == arm),
                  "contacts": sum(row["contacts"] for key, row in indexed.items() if key[2] == arm),
                  "verified_recoveries": sum(row["verified_recoveries"] for key, row in indexed.items() if key[2] == arm)} for arm in ARMS}
    clusters = [float(np.mean([int(indexed[(case, seed, ARMS[1])]["clean_goal"])-int(indexed[(case, seed, ARMS[0])]["clean_goal"])
                              for case in CASES])) for seed in SEEDS]
    resamples = [float(np.mean([clusters[i] for i in choice])) for choice in product(range(3), repeat=3)]
    low, high = np.quantile(resamples, [.025, .975])
    improved = (arms[ARMS[1]]["clean_goals"] > arms[ARMS[0]]["clean_goals"]
                and arms[ARMS[1]]["falls"] <= arms[ARMS[0]]["falls"]
                and arms[ARMS[1]]["contacts"] <= arms[ARMS[0]]["contacts"])
    return {"schema": "bhl-stereo-recovery-paired-results-v1", "status": "PASS" if improved else "NEGATIVE",
            "scientific_status": "SIMULATION_DEVELOPMENT_ONLY_CONFIRMATION_REQUIRED", "episodes": 18,
            "pairs": 9, "seed_groups": 3, "arms": arms, "clean_goal_rate_difference": float(np.mean(clusters)),
            "seed_cluster_bootstrap_percentile_interval_95": [float(low), float(high)],
            "uncertainty_scope": "descriptive exact enumeration of 27 resamples of three reset-seed groups; routes remain grouped; only three clusters and one trained actor, no population or hardware inference",
            "measured_development_improvement": improved, "independent_confirmation": "NOT_RUN"}
