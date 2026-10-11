from itertools import product
import pytest
from bhl_robust.research.stereo_recovery_metrics import aggregate, CASES, SEEDS, ARMS


def cohort():
    return [{"case": case, "seed": seed, "arm": arm, "clean_goal": arm == ARMS[1],
             "falls": False, "contacts": False, "verified_recoveries": int(arm == ARMS[1])}
            for case, seed, arm in product(CASES, SEEDS, ARMS)]


def test_complete_paired_cohort_reports_measured_difference_and_scope():
    result = aggregate(cohort())
    assert result["status"] == "PASS"
    assert result["episodes"] == 18
    assert result["clean_goal_rate_difference"] == 1.
    assert result["seed_cluster_bootstrap_percentile_interval_95"] == [1., 1.]
    assert result["independent_confirmation"] == "NOT_RUN"


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "nonboolean"])
def test_incomplete_or_invalid_cohort_never_becomes_negative_result(change):
    rows = cohort()
    if change == "missing": rows.pop()
    if change == "duplicate": rows[-1] = rows[0]
    if change == "unknown": rows[-1]["seed"] = 1
    if change == "nonboolean": rows[-1]["clean_goal"] = 1
    with pytest.raises(ValueError): aggregate(rows)


def test_goal_gain_with_increased_contacts_fails_development_gate():
    rows = cohort()
    rows[-1]["contacts"] = True
    assert aggregate(rows)["status"] == "NEGATIVE"
