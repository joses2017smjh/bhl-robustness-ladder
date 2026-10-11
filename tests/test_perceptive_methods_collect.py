import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1]/"scripts/bench/perceptive_methods_collect.py"
SPEC = importlib.util.spec_from_file_location("perceptive_collect_tests", PATH)
COLLECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECT)


def evaluation(seed, condition, *, qualified=True, student=False):
    rows = [dict(environment=i, survived=True, qualified=qualified, completed_policy_steps=250,
                 commanded_distance_m=2., integrated_tracking_error_m=.2 if qualified else 4., displacement_m=2.) for i in range(2)]
    return dict(seed=seed, dropout=condition[0], delay_steps=condition[1], episodes=2, rows=rows,
                survivors=2, qualified=2 if qualified else 0, qualification_rule=COLLECT.QUALIFICATION.copy(),
                inference_p95_ms=1., reconstruction_mae_m=.01 if student else None)


def payload(seed=0, *, ready=True):
    settings = dict(seeds=[0, 1, 2], arms=list(COLLECT.ARMS), qualification=COLLECT.QUALIFICATION.copy(),
        teacher_qualification_fraction=.8, teacher_iterations=2, student_iterations=1, num_envs=2,
        student_evaluation_conditions=[dict(dropout=d, delay_policy_steps=t) for d, t in COLLECT.CONDITIONS])
    protocol = dict(perceptive=settings)
    result = dict(schema="bhl-perceptive-training-v1", seed=seed, phase="run", protocol_sha256="protocol",
        status="PASS" if ready else "NEGATIVE", scientific_status="DEVELOPMENT_ONLY" if ready else "TEACHER_QUALIFICATION_FAILED",
        teacher_qualified=ready, teacher_checkpoint_sha256="teacher", jobs_completed=["teacher"])
    records = {"teacher-training.json": [dict(iteration=i) for i in range(2)],
               "teacher-development.json":evaluation(510000+seed, (0., 0), qualified=ready)}
    hashes = {"teacher.pt":"teacher"}
    if ready:
        result["student_results"] = {}
        for arm in COLLECT.ARMS:
            result["jobs_completed"].append(arm)
            result["student_results"][arm] = dict(checkpoint_sha256=arm, episodes=8, qualified=8, survivors=8)
            hashes.update({arm+".pt":arm, arm+"-actor.jit":arm+"-export"})
            records[arm+"-training.json"] = [dict(iteration=0)]
            records[arm+"-deployment.json"] = dict(schema="bhl-perceptive-actor-v1", arm=arm,
                inputs=dict(proprio=["batch", 45], points=["batch", 4, 128, 5]), output=["batch", 12], maximum_parity_error=0.)
            records[arm+"-development.json"] = [evaluation(530000+seed, c, student=True) for c in COLLECT.CONDITIONS]
    else:
        result["student_training"] = "UNRUN_AFTER_FAILED_TEACHER_GATE"
    records["campaign_result.json"] = result
    return protocol, records, hashes, result


def checked(seed=0, ready=True):
    protocol, records, hashes, result = payload(seed, ready=ready)
    return protocol, COLLECT.validate_payload(records, hashes, result, protocol, dict(protocol_sha256="protocol"))


def test_three_distinct_seeds_and_all_arms_conditions_are_required_for_comparison():
    protocol, first = checked(0)
    rows = [first, checked(1)[1], checked(2)[1]]
    result = COLLECT.aggregate(rows, protocol)
    assert result["status"] == "PASS" and result["completed_student_arms"] == 12
    assert result["paired_student_comparison"] == "COMPLETE"
    assert all(v["paired_seeds"] == [0, 1, 2] and v["differences"] == [0., 0., 0.] for v in result["paired_differences"].values())


def test_negative_teacher_stop_is_complete_outcome_with_explicit_unrun_students():
    protocol, first = checked(0, ready=False)
    result = COLLECT.aggregate([first, checked(1)[1], checked(2)[1]], protocol)
    assert result["status"] == "NEGATIVE" and result["qualified_teachers"] == 2
    assert result["completed_student_arms"] == 8 and result["unrun_student_arms_after_negative_teacher"] == 4
    assert result["paired_differences"] == {}


def test_missing_and_duplicate_seeds_never_imply_zero_failures_or_complete_cohort():
    protocol, first = checked(0)
    result = COLLECT.aggregate([first, checked(1)[1]], protocol)
    assert result["status"] == "INCOMPLETE" and result["missing_seeds"] == [2]
    duplicate = COLLECT.aggregate([first, copy.deepcopy(first), checked(1)[1], checked(2)[1]], protocol)
    assert duplicate["status"] == "INCOMPLETE" and duplicate["missing_seeds"] == [0]
    assert duplicate["paired_differences"] == {}


@pytest.mark.parametrize("fault", ("wrong_direction", "duplicate_condition", "wrong_seed", "missing_arm", "changed_checkpoint", "critic_export"))
def test_invalid_student_evidence_is_rejected(fault):
    protocol, records, hashes, result = payload()
    if fault == "wrong_direction":
        records["query-development.json"][0]["rows"][0]["integrated_tracking_error_m"] = 4.
    elif fault == "duplicate_condition":
        records["query-development.json"][1] = copy.deepcopy(records["query-development.json"][0])
    elif fault == "wrong_seed":
        records["query-development.json"][0]["seed"] = 530001
    elif fault == "missing_arm":
        result["jobs_completed"].remove("query")
    elif fault == "changed_checkpoint":
        hashes["query.pt"] = "changed"
    else:
        records["query-deployment.json"]["inputs"]["privileged_critic"] = ["batch", 125]
    with pytest.raises(ValueError):
        COLLECT.validate_payload(records, hashes, result, protocol, dict(protocol_sha256="protocol"))


def test_negative_teacher_cannot_hide_student_checkpoint_or_claim_pass():
    protocol, records, hashes, result = payload(ready=False)
    hashes["dense.pt"] = "unexpected training"
    with pytest.raises(ValueError, match="explicit unrun"):
        COLLECT.validate_payload(records, hashes, result, protocol, dict(protocol_sha256="protocol"))
