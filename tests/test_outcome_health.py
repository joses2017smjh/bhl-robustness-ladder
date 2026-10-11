import copy
import pytest
from bhl_robust.research.outcome_health import (
    CALIBRATION_SEED, EVALUATION_SEED, FEATURE, evaluate, fit,
)
from bhl_robust.research.navigation_sensor_controls import CONDITIONS


def cohort(seed):
    return [{"method": "lidar_inertial", "case": case, "condition": condition, "seed": seed,
             "success": i < 5, "full_horizon": True, "fell": False, "collision": False,
             "nonfinite": False, FEATURE: .95 if i < 5 else .25}
            for case in ("straight", "occluders") for i, condition in enumerate(CONDITIONS)]


def test_calibration_only_fit_and_distinct_holdout():
    model = fit(cohort(CALIBRATION_SEED))
    report = evaluate(model, cohort(EVALUATION_SEED))
    assert report["arms"]["lidar_inertial"]["frozen_gate"]["balanced_accuracy"] == 1
    with pytest.raises(ValueError, match="wrong-seed"):
        fit(cohort(EVALUATION_SEED))
    with pytest.raises(ValueError, match="wrong-seed"):
        evaluate(model, cohort(CALIBRATION_SEED))


def test_bad_tracking_is_not_silently_successful_pose():
    model = fit(cohort(CALIBRATION_SEED))
    heldout = cohort(EVALUATION_SEED)
    for r in heldout:
        r[FEATURE] = .99
    score = evaluate(model, heldout)["arms"]["lidar_inertial"]["frozen_gate"]
    assert score["adverse_acceptance_rate"] == 1
    assert score["balanced_accuracy"] == .5


def test_negative_no_variation_is_not_a_calibrated_model():
    rows = cohort(CALIBRATION_SEED)
    for r in rows:
        r["success"] = False
    model = fit(rows)
    assert model["models"]["lidar_inertial"]["threshold"] is None
    assert model["models"]["lidar_inertial"]["status"] == "INSUFFICIENT_OUTCOME_VARIATION"


def test_incomplete_duplicates_inconsistent_or_tampered_inputs_rejected():
    rows = cohort(CALIBRATION_SEED)
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError):
            fit(bad)
    bad = copy.deepcopy(rows)
    bad[0]["fell"] = True
    with pytest.raises(ValueError, match="inconsistent"):
        fit(bad)
    model = fit(rows)
    model["models"]["lidar_inertial"]["threshold"] = 0
    with pytest.raises(ValueError, match="checksum"):
        evaluate(model, cohort(EVALUATION_SEED))
