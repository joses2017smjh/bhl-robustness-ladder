"""Frozen, paired NavGym confirmation; task success is read from episode evidence.

The project-local thresholds are evidence targets, not hardware qualifications.
No episode or final verdict is overwritten, and incomplete evidence cannot pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def read(path):
    return json.loads(Path(path).read_text())


def validate_inputs(campaign):
    protocol = read(campaign / "protocol.json")
    for name, expected in protocol["input_sha256"].items():
        path = Path(name) if name.startswith("/") else campaign / name
        if digest(path) != expected:
            raise ValueError(f"frozen input changed: {path}")
    for package, expected in protocol["package_versions"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"package version changed: {package}")
    return protocol


def run_cell(campaign, index):
    p = validate_inputs(campaign)
    cell = p["cells"][index]
    out = campaign / "runs" / cell["name"]
    if out.exists():
        raise FileExistsError(f"refusing an existing cell directory: {out}")
    out.mkdir(parents=True)
    command = [sys.executable, str(campaign / "snapshot/scripts/bench/maze_explore.py"),
               "--upstream", p["upstream"], "--gait", str(campaign / "models/gait/deploy.yaml"),
               "--seeds", str(len(p["maze_seeds"])), "--seed-start", str(p["maze_seeds"][0]),
               "--n", "6", "--m", "6", "--extra-openings", "1", "--time-limit", "180",
               "--sensor-mode", cell["sensor_mode"], "--dropout-probability", "0.35",
               "--no-overwrite", "--out-dir", str(out)]
    if cell["actor"] != "astar":
        command += ["--policy", str(campaign / f"models/{cell['actor']}/actor.onnx"), "--policy-capture-pose"]
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(campaign / "snapshot/src"), str(campaign / "snapshot/upstream_lowlevel")]),
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    receipt = {"command": command, "protocol_sha256": digest(campaign / "protocol.json"),
               "utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
               "platform": platform.platform(), "python": sys.version,
               "slurm_job_id": os.getenv("SLURM_JOB_ID"), "array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
               "cell": cell, "package_versions": p["package_versions"]}
    write_new(out / "run_receipt.json", receipt)
    with tempfile.TemporaryDirectory(prefix="bhl-confirm-", dir=os.getenv("TMPDIR", "/tmp")) as cache:
        command += ["--cache-dir", cache]
        with (out / "run.log").open("x") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    # Detect a shared-asset change during execution as invalid evidence.
    valid = True
    try:
        validate_inputs(campaign)
    except (ValueError, OSError) as e:
        valid = False
        receipt["input_error_after_run"] = str(e)
    write_new(out / "process_result.json", {"returncode": result.returncode, "inputs_still_valid": valid,
                                            "input_error": receipt.get("input_error_after_run")})
    print(json.dumps({"cell": cell["name"], "returncode": result.returncode, "inputs_still_valid": valid}), flush=True)
    return 0 if result.returncode == 0 and valid else 1


def wilson(k, n):
    """Descriptive interval for one actor; pooled rows are never treated as IID."""
    z = 1.959963984540054
    center = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt((k / n) * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [center - radius, center + radius]


def inspect_cell(campaign, p, cell):
    out = campaign / "runs" / cell["name"]
    problems, episodes = [], []
    for seed in p["maze_seeds"]:
        path = out / f"seed{seed}.json"
        if not path.is_file():
            problems.append(f"missing seed {seed}")
            continue
        try:
            e = read(path)
            expected_actor = None if cell["actor"] == "astar" else str(campaign / f"models/{cell['actor']}/actor.onnx")
            checks = [e["seed"] == seed, e["maze"]["n"] == 6, e["maze"]["m"] == 6,
                      e["maze"]["extra_openings"] == 1, e["sensor_mode"] == cell["sensor_mode"],
                      e["gait"]["checkpoint_sha256"] == p["gait_sha256"], e["policy"] == expected_actor,
                      e["initial_heading_rad"] == 0.0, e["imu"]["source"] == "truth",
                      e["pose_error"]["pose_bias_m"] == 0.0, e["pose_error"]["pose_noise_m"] == 0.0,
                      e["pose_error"]["pose_yaw_deg"] == 0.0,
                      e["success"] == (e["outcome"] == "goal"),
                      e["clean_success"] == (e["success"] and e["wall_contact_steps"] == 0),
                      e["maze"]["layout_sha256"] == p["layout_sha256"][str(seed)]]
            if expected_actor:
                checks += [e["policy_map_integration"]["mode"] == "capture_pose",
                           e["policy_visitation"]["obs_keys"] == ["lidar", "near", "map_visit", "map_coarse", "goal"]]
            if not all(checks):
                problems.append(f"seed {seed}: evidence contract mismatch")
            episodes.append(e)
        except (ValueError, KeyError, TypeError) as error:
            problems.append(f"seed {seed}: {error}")
    if sorted(x.name for x in out.glob("seed*.json")) != sorted(f"seed{s}.json" for s in p["maze_seeds"]):
        problems.append("episode file set differs from declared seeds")
    counts = {"goals": sum(e["success"] for e in episodes), "clean": sum(e["clean_success"] for e in episodes),
              "falls": sum(e["outcome"] == "fall" for e in episodes), "n": len(episodes)}
    try:
        summary, receipt, process = [read(out / f) for f in ("summary.json", "run_receipt.json", "process_result.json")]
        if (summary["n"] != len(p["maze_seeds"]) or summary["seeds"] != p["maze_seeds"] or
                summary["settings"]["time_limit"] != 180.0 or summary["success"] != counts["goals"] or
                summary["falls"] != counts["falls"] or summary["clean_success"] != counts["clean"] or
                receipt["protocol_sha256"] != digest(campaign / "protocol.json") or
                process["returncode"] != 0 or process["inputs_still_valid"] is not True):
            problems.append("summary, receipt, or process invalid")
    except (OSError, ValueError, KeyError) as error:
        problems.append(f"incomplete final files: {error}")
    complete = not problems and counts["n"] == len(p["maze_seeds"])
    successes = [e["completion_s"] for e in episodes if e["success"]]
    cmds = [t["cmd"] for e in episodes for t in e["trace"]]
    sampled = sum(e["sensor_stats"]["sampled_robot_steps"] for e in episodes)
    stats = {key: sum(e["sensor_stats"][key] for e in episodes) for key in
             ("received_extero_packets", "dropped_extero_packets", "braked_robot_steps", "stale_stop_robot_steps")}
    stats["braked_step_fraction"] = stats["braked_robot_steps"] / sampled if sampled else None
    # Existing trace samples are 5 Hz, rounded to 0.001; no claimed precise tracking error.
    telemetry = {"mean_abs_commanded_vx_m_s": statistics.mean(abs(c[0]) for c in cmds) if cmds else None,
                 "mean_abs_commanded_wz_rad_s": statistics.mean(abs(c[2]) for c in cmds) if cmds else None,
                 "path_m": sum(e["path_length_m"] for e in episodes), "sensor_stats": stats}
    minimum = p["project_targets"][cell["condition"]]["min_goals"]
    passed = complete and counts["goals"] >= minimum and counts["falls"] == 0
    return {**counts, "complete": complete, "problems": problems,
            "project_target_met": passed if cell["actor"] != "astar" else None,
            "success_rate": counts["goals"] / counts["n"] if counts["n"] else None,
            "success_wilson95": wilson(counts["goals"], counts["n"]) if counts["n"] else None,
            "median_success_time_s": statistics.median(successes) if successes else None,
            "failed_seeds": [e["seed"] for e in episodes if not e["success"]],
            "contact_seeds": [e["seed"] for e in episodes if e["wall_contact_steps"] > 0],
            "telemetry": telemetry}, {e["seed"]: e for e in episodes}


def summarize(campaign, final=False):
    p = validate_inputs(campaign)
    rows, evidence = {}, {}
    for cell in p["cells"]:
        rows[cell["name"]], evidence[cell["name"]] = inspect_cell(campaign, p, cell)
    paired = {}
    for actor in p["actors"]:
        for condition in ("nominal", "drop35"):
            mine, ref = evidence[f"{actor}-{condition}"], evidence[f"astar-{condition}"]
            shared = sorted(set(mine) & set(ref))
            paired[f"{actor}-{condition}"] = {
                "n_paired": len(shared),
                "both_goal": sum(mine[s]["success"] and ref[s]["success"] for s in shared),
                "learned_only_goal": sum(mine[s]["success"] and not ref[s]["success"] for s in shared),
                "astar_only_goal": sum(not mine[s]["success"] and ref[s]["success"] for s in shared)}
    complete = all(row["complete"] for row in rows.values())
    target = all(row["project_target_met"] for name, row in rows.items() if not name.startswith("astar-"))
    result = {"status": "INCOMPLETE" if not complete else "PASS" if target else "NEGATIVE",
              "final": final, "protocol_sha256": digest(campaign / "protocol.json"), "per_cell": rows,
              "paired_vs_astar": paired, "labels": p["labels"], "project_targets": p["project_targets"],
              "statistical_scope": "48 distinct maze layouts shared across 3 trained navigation policies and 2 conditions. "
                                   "Report per-policy results; pooled episodes share layouts and are not IID. "
                                   "Wilson intervals are descriptive per-cell intervals, not robustness certificates."}
    if final:
        write_new(campaign / "verdict.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "run", "summary"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--cell", type=int)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.mode == "run":
        if args.cell is None or not 0 <= args.cell < 8:
            parser.error("run requires --cell 0..7")
        return run_cell(campaign, args.cell)
    if args.mode == "summary":
        return summarize(campaign, args.final)
    validate_inputs(campaign)
    print("FROZEN INPUT CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
