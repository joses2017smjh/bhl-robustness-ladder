#!/usr/bin/env python3
"""Verify three frozen perceptive seeds, including declared negative teacher stops.

Read-only collection from completed jobs or explicitly provided raw-archive
mirrors. Missing student trials are never counted as failures or successes.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]/"src"))
import sensor_methods_collect as ARCHIVE
from bhl_robust.research.perceptive_policy import ARMS, QUALIFICATION
SAFE = ARCHIVE.SAFE
CONDITIONS = ((0., 0), (.5, 0), (0., 2), (.5, 2))


def validate_protocol(protocol):
    settings = protocol["perceptive"]
    if (settings["seeds"] != [0, 1, 2] or settings["arms"] != list(ARMS) or
            settings.get("qualification") != QUALIFICATION or settings["teacher_qualification_fraction"] != .8 or
            [(v["dropout"], v["delay_policy_steps"]) for v in settings["student_evaluation_conditions"]] != list(CONDITIONS)):
        raise ValueError("three-seed perceptive protocol or qualification differs")
    return settings


def checked_evaluation(value, *, seed, condition, episodes):
    if (value["seed"] != seed or (value["dropout"], value["delay_steps"]) != condition or
            value["episodes"] != episodes or value["qualification_rule"] != QUALIFICATION or
            len(value["rows"]) != episodes or [r["environment"] for r in value["rows"]] != list(range(episodes))):
        raise ValueError("evaluation seed, condition, qualification or first-episode cohort mismatch")
    for row in value["rows"]:
        numbers = [row[name] for name in ("commanded_distance_m", "integrated_tracking_error_m", "displacement_m")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in numbers):
            raise ValueError("invalid episode distance")
        if type(row["survived"]) is not bool or type(row["qualified"]) is not bool:
            raise ValueError("explicit boolean episode outcomes required")
        distance, error, _ = numbers
        qualified = row["survived"] and distance > .1 and error <= .5*distance
        if (row["qualified"] != qualified or type(row["completed_policy_steps"]) is not int or
                not 1 <= row["completed_policy_steps"] <= 250 or
                (row["survived"] and row["completed_policy_steps"] != 250)):
            raise ValueError("episode qualification or survival was not reproduced")
    if (value["survivors"] != sum(r["survived"] for r in value["rows"]) or
            value["qualified"] != sum(r["qualified"] for r in value["rows"])):
        raise ValueError("episode aggregate differs from raw outcomes")
    if (not math.isfinite(value["inference_p95_ms"]) or value["inference_p95_ms"] <= 0 or
            (value["reconstruction_mae_m"] is not None and
             (not math.isfinite(value["reconstruction_mae_m"]) or value["reconstruction_mae_m"] < 0))):
        raise ValueError("invalid inference latency or reconstruction error")
    return dict(dropout=condition[0], delay_steps=condition[1], episodes=episodes,
        survivors=value["survivors"], qualified=value["qualified"],
        qualification_fraction=value["qualified"]/episodes,
        survival_fraction=value["survivors"]/episodes,
        inference_p95_ms=value["inference_p95_ms"], reconstruction_mae_m=value["reconstruction_mae_m"])


def validate_payload(records, hashes, result, protocol, intake):
    settings = validate_protocol(protocol)
    seed = result["seed"]
    if (seed not in settings["seeds"] or isinstance(seed, bool) or result["phase"] != "run" or
            result["protocol_sha256"] != intake["protocol_sha256"] or result["schema"] != "bhl-perceptive-training-v1" or
            records["campaign_result.json"] != result):
        raise ValueError("perceptive seed, source or run identity mismatch")
    logs = records["teacher-training.json"]
    if (len(logs) != settings["teacher_iterations"] or
            [r["iteration"] for r in logs] != list(range(settings["teacher_iterations"])) or
            result["teacher_checkpoint_sha256"] != hashes["teacher.pt"]):
        raise ValueError("teacher training/checkpoint evidence incomplete")
    teacher = checked_evaluation(records["teacher-development.json"], seed=510000+seed,
        condition=(0., 0), episodes=settings["num_envs"])
    ready = teacher["qualification_fraction"] >= settings["teacher_qualification_fraction"]
    if result["teacher_qualified"] is not ready:
        raise ValueError("teacher gate differs from measured first episodes")
    students = {}
    if not ready:
        if (result["status"] != "NEGATIVE" or result["scientific_status"] != "TEACHER_QUALIFICATION_FAILED" or
                result["student_training"] != "UNRUN_AFTER_FAILED_TEACHER_GATE" or result["jobs_completed"] != ["teacher"] or
                result.get("student_results") or any(name == arm+".pt" or name.startswith(arm+"-") for name in hashes for arm in ARMS)):
            raise ValueError("negative teacher gate must retain explicit unrun students")
    else:
        if (result["status"] != "PASS" or result["scientific_status"] != "DEVELOPMENT_ONLY" or
                result["jobs_completed"] != ["teacher", *ARMS] or set(result["student_results"]) != set(ARMS)):
            raise ValueError("qualified teacher requires all four completed student arms")
        for arm in ARMS:
            outcome = result["student_results"][arm]
            logs = records[arm+"-training.json"]
            if (len(logs) != settings["student_iterations"] or
                    [r["iteration"] for r in logs] != list(range(settings["student_iterations"])) or
                    hashes[arm+".pt"] != outcome["checkpoint_sha256"]):
                raise ValueError("student training or checkpoint evidence incomplete")
            deployment = records[arm+"-deployment.json"]
            if (deployment["arm"] != arm or deployment["schema"] != "bhl-perceptive-actor-v1" or
                    set(deployment["inputs"]) != {"proprio", "points"} or
                    deployment["inputs"]["proprio"] != ["batch", 45] or deployment["inputs"]["points"] != ["batch", 4, 128, 5] or
                    deployment["output"] != ["batch", 12] or
                    not math.isfinite(deployment["maximum_parity_error"]) or not 0 <= deployment["maximum_parity_error"] <= 1e-5 or
                    arm+"-actor.jit" not in hashes):
                raise ValueError("measured-input student export or parity evidence invalid")
            checks = records[arm+"-development.json"]
            if len(checks) != 4:
                raise ValueError("every student requires four paired degradation conditions")
            evaluated = [checked_evaluation(value, seed=530000+seed, condition=condition, episodes=settings["num_envs"])
                         for value, condition in zip(checks, CONDITIONS)]
            if any(v["reconstruction_mae_m"] is None for v in evaluated):
                raise ValueError("student reconstruction evidence is unavailable")
            if any(outcome[k] != sum(v[k] for v in evaluated) for k in ("episodes", "qualified", "survivors")):
                raise ValueError("student summary differs from paired episode evidence")
            students[arm] = dict(conditions=evaluated, checkpoint_sha256=hashes[arm+".pt"], actor_sha256=hashes[arm+"-actor.jit"])
    return dict(seed=seed, status=result["status"], teacher_qualified=ready, teacher=teacher,
        teacher_checkpoint_sha256=hashes["teacher.pt"], students=students,
        student_training="COMPLETED" if ready else "UNRUN_AFTER_FAILED_TEACHER_GATE")


def collect_job(job, protocol, intake, mirrors):
    job = Path(job)
    completion, launch = SAFE.read_json(job/"completion.json"), SAFE.read_json(job/"launch.json")
    if (job.is_symlink() or completion.get("status") not in ("PASS", "NEGATIVE") or completion.get("exit_status") != 0 or
            completion.get("error") or not completion.get("finished_utc") or
            completion["archive_sha256"] != intake["archive_sha256"] or launch["archive_sha256"] != intake["archive_sha256"] or
            launch["source_files_verified"] != intake["source_files"]):
        raise ValueError("incomplete perceptive execution or source lineage mismatch")
    declared = [j for j in protocol["jobs"] if j["kind"] == "run" and j == launch["job"]]
    if len(declared) != 1 or job.name != declared[0]["name"]+"-"+str(launch["job_id"]):
        raise ValueError("undeclared or mismatched perceptive launch")
    if not {"outputs.tar.gz", "launch.json", "runtime.log", "campaign_result.json"} <= set(completion["files"]):
        raise ValueError("missing durable completion evidence")
    for name, specification in completion["files"].items():
        if PurePosixPath(name).name != name:
            raise ValueError("unsafe completion filename")
        if name != "outputs.tar.gz":
            ARCHIVE.checked_file(job/name, specification)
    result = SAFE.read_json(job/"campaign_result.json")
    if result["status"] != completion["status"]:
        raise ValueError("completion and scientific status differ")
    index = protocol["perceptive"]["seeds"].index(result["seed"])
    for command in (declared[0]["args"], launch["command"]):
        if (command.count("--cell") != 1 or command[command.index("--cell")+1] != str(index) or
                command.count("--phase") != 1 or command[command.index("--phase")+1] != "run"):
            raise ValueError("launch seed/cell mismatch")
    archive = ARCHIVE.find_archive(job, completion, intake["archive_sha256"], mirrors)
    records, hashes, total = ARCHIVE.archive_records(archive, declared[0].get("max_output_mb", 256)*1024**2)
    if total != completion["output_bytes"]:
        raise ValueError("perceptive archive byte count differs")
    value = validate_payload(records, hashes, result, protocol, intake)
    value["evidence"] = dict(job=str(job), archive=str(archive), archive_sha256=completion["files"]["outputs.tar.gz"]["sha256"],
                             completion_sha256=SAFE.digest(job/"completion.json"))
    return value


def aggregate(rows, protocol, problems=()):
    validate_protocol(protocol)
    problems = list(problems)
    seeds = [r["seed"] for r in rows]
    duplicates = {seed for seed in seeds if seeds.count(seed) > 1}
    if duplicates:
        problems.append(dict(error="duplicate seed evidence quarantined", seeds=sorted(duplicates)))
        rows = [r for r in rows if r["seed"] not in duplicates]
    if any(r["seed"] not in (0, 1, 2) for r in rows):
        raise ValueError("unexpected training seed")
    complete = len(rows) == 3 and not problems
    teacher_ready = sum(r["teacher_qualified"] for r in rows)
    all_students = complete and teacher_ready == 3
    comparisons = {}
    if all_students:
        rows = sorted(rows, key=lambda r:r["seed"])
        for arm in ARMS[1:]:
            for index, condition in enumerate(CONDITIONS):
                for metric in ("qualification_fraction", "survival_fraction", "inference_p95_ms", "reconstruction_mae_m"):
                    pairs = [(r["students"][arm]["conditions"][index][metric], r["students"]["dense"]["conditions"][index][metric]) for r in rows]
                    differences = [a-b for a, b in pairs] if all(a is not None and b is not None for a, b in pairs) else None
                    comparisons[f"{arm}_minus_dense/dropout{condition[0]}_delay{condition[1]}/{metric}"] = dict(
                        paired_seeds=[0, 1, 2], differences=differences,
                        mean_difference=sum(differences)/3 if differences is not None else None)
    return dict(schema="bhl-perceptive-methods-collection-v1",
        status=("PASS" if teacher_ready == 3 else "NEGATIVE") if complete else "INCOMPLETE",
        scientific_status="COMPLETE_DEVELOPMENT_NO_CONFIRMATION" if complete else "PARTIAL_EVIDENCE_NO_GLOBAL_VERDICT",
        observed_seeds=len(rows), expected_seeds=3, missing_seeds=sorted({0, 1, 2}-{r["seed"] for r in rows}),
        qualified_teachers=teacher_ready, evaluated_teachers=len(rows),
        completed_student_arms=sum(len(r["students"]) for r in rows), expected_student_arms_if_teachers_qualify=12,
        unrun_student_arms_after_negative_teacher=4*sum(not r["teacher_qualified"] for r in rows),
        paired_student_comparison="COMPLETE" if all_students else "UNAVAILABLE_REQUIRES_THREE_QUALIFIED_TEACHERS_AND_ALL_STUDENTS",
        paired_differences=comparisons, rows=rows, problems=problems,
        analysis_unit="three independent training seeds; fixed matched environment identities and four conditions within seed; no episode-level significance claim",
        scope="development Isaac velocity tracking; negative teacher gate is a measured outcome, unrun students are not zero-valued observations; no navigation or hardware claim")


def collect(campaign, job_dirs, mirror_dirs=()):
    campaign = Path(campaign)
    intake, protocol = SAFE.read_json(campaign/"intake.json"), SAFE.read_json(campaign/"protocol.json")
    validate_protocol(protocol)
    frozen, _ = SAFE.verify_frozen_archive(campaign, intake)
    for name, specification in protocol["input_files"].items():
        if frozen["inputs/"+name]["sha256"] != specification["sha256"]:
            raise ValueError("perceptive warm-start input pin changed")
    rows, problems = [], []
    for job in job_dirs:
        try:
            rows.append(collect_job(job, protocol, intake, mirror_dirs))
        except (ValueError, OSError, KeyError, IndexError, TypeError, tarfile.TarError) as error:
            problems.append(dict(job=str(job), error=f"{type(error).__name__}: {error}"))
    result = aggregate(rows, protocol, problems)
    result.update(source_archive_sha256=intake["archive_sha256"], protocol_sha256=intake["protocol_sha256"],
                  input_sha256={k:v["sha256"] for k,v in protocol["input_files"].items()}, collector_sha256=SAFE.digest(__file__))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--campaign-dir", required=True, type=Path)
    p.add_argument("--job-dirs", nargs="*", type=Path, default=[])
    p.add_argument("--mirror-dir", action="append", type=Path, default=[])
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    result = collect(args.campaign_dir, args.job_dirs, args.mirror_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:result[k] for k in ("status", "observed_seeds", "qualified_teachers", "completed_student_arms", "missing_seeds")}))
    return 1 if result["status"] == "INCOMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
