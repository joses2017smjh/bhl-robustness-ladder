"""Small, frozen paired development pilot of the existing NavGym yaw filter.

This does not change the default navigator or train a new policy. The existing
0.2-second pre-brake yaw filter is a hypothesis from the measured gait/gym lag.
One actor, two fresh layouts and two sensing conditions are development evidence,
not a replacement for the 48-layout / three-actor confirmation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def validate(campaign):
    p = json.loads((campaign / "protocol.json").read_text())
    for name, expected in p["input_sha256"].items():
        f = Path(name) if name.startswith("/") else campaign / name
        if not f.is_file() or digest(f) != expected:
            raise ValueError(f"frozen pilot input changed/missing: {name}")
    for name, expected in p["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"package mismatch: {name}")
    return p


def prepare(args):
    from omegaconf import OmegaConf
    campaign = args.campaign.resolve()
    campaign.mkdir(parents=True, exist_ok=False)
    snapshot = campaign / "snapshot"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(REPO / "src", snapshot / "src", ignore=ignore)
    shutil.copytree(REPO / "scripts/bench", snapshot / "scripts/bench", ignore=ignore)
    upstream_lowlevel = args.upstream / "source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel"
    shutil.copytree(upstream_lowlevel, snapshot / "upstream_lowlevel/berkeley_humanoid_lite_lowlevel", ignore=ignore)
    cfg = OmegaConf.load(args.gait)
    gait_checkpoint = args.gait_checkpoint or Path(cfg.policy_checkpoint_path)
    if not gait_checkpoint.is_file() or not args.actor.is_file():
        raise FileNotFoundError("actual gait and navigation actor files are required")
    (campaign / "models").mkdir()
    shutil.copyfile(gait_checkpoint, campaign / "models/gait.onnx")
    shutil.copyfile(args.actor, campaign / "models/actor.onnx")
    cfg.policy_checkpoint_path = str(campaign / "models/gait.onnx")
    OmegaConf.save(cfg, campaign / "models/deploy.yaml")
    files = {str(f.relative_to(campaign)): digest(f) for f in sorted(snapshot.rglob("*")) if f.is_file()}
    files.update({str(f.relative_to(campaign)): digest(f) for f in sorted((campaign / "models").glob("*"))})
    # The shared source MJCF/mesh assets are immutable inputs even though the
    # generated scene cache is local to each worker.
    robot = args.upstream / "source/berkeley_humanoid_lite_assets/data/robots/berkeley_humanoid/berkeley_humanoid_lite"
    for sub in ("mjcf", "meshes"):
        for f in sorted((robot / sub).rglob("*")):
            if f.is_file():
                files[str(f.resolve())] = digest(f)
    cells = [{"name": f"{mode}-{condition}", "mode": mode, "condition": condition,
              "sensor_mode": "reactive" if condition == "nominal" else "reactive_dropout"}
             for mode in ("none", "wz-lpf") for condition in ("nominal", "drop35")]
    p = {"scope": "DEVELOPMENT ONLY: one frozen actor, two fresh layouts, baseline versus existing yaw LPF; no training/hardware.",
         "created_utc": datetime.now(timezone.utc).isoformat(), "upstream": str(args.upstream.resolve()),
         "overlay_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
         "source_actor": str(args.actor.resolve()), "source_actor_sha256": digest(args.actor),
         "source_gait_sha256": digest(gait_checkpoint), "tau_s": .2, "time_limit_s": 180.,
         "seeds": [95000, 95001], "cells": cells, "input_sha256": files,
         "packages": {name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "onnxruntime", "omegaconf")},
         "decision": "Report falls, clean goals and measured yaw chatter for every cell. Do not promote from two layouts or tune tau on this pilot.",
         "labels": {"learned": "frozen 12-DoF gait plus NavGym v5 actor", "scripted": "reactive speed brake and optional pre-brake yaw LPF",
                    "oracle": "pose and goal coordinate"}}
    write_new(campaign / "protocol.json", p)
    print(json.dumps({"campaign": str(campaign), "cells": cells, "seeds": p["seeds"], "episodes": 8}))
    return 0


def run_cell(args):
    campaign = args.campaign.resolve()
    p = validate(campaign)
    if args.cell is None or not 0 <= args.cell < len(p["cells"]):
        raise ValueError("run requires cell 0..3")
    cell = p["cells"][args.cell]
    out = campaign / "runs" / cell["name"]
    out.mkdir(parents=True, exist_ok=False)
    cmd = [sys.executable, str(campaign / "snapshot/scripts/bench/maze_explore.py"),
           "--upstream", p["upstream"], "--gait", str(campaign / "models/deploy.yaml"),
           "--policy", str(campaign / "models/actor.onnx"), "--policy-capture-pose",
           "--policy-cmd-filter", cell["mode"], "--policy-cmd-tau", str(p["tau_s"]),
           "--sensor-mode", cell["sensor_mode"], "--dropout-probability", ".35",
           "--seeds", str(len(p["seeds"])), "--seed-start", str(p["seeds"][0]),
           "--n", "6", "--m", "6", "--extra-openings", "1", "--time-limit", str(p["time_limit_s"]),
           "--out-dir", str(out), "--no-overwrite"]
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(campaign / "snapshot/src"), str(campaign / "snapshot/upstream_lowlevel")]),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    with tempfile.TemporaryDirectory(prefix="bhl-nav-dev-") as cache:
        cmd += ["--cache-dir", cache]
        receipt = {"command": cmd, "protocol_sha256": digest(campaign / "protocol.json"), "cell": cell,
                   "host": platform.node(), "platform": platform.platform(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                   "created_utc": datetime.now(timezone.utc).isoformat()}
        write_new(out / "receipt.json", receipt)
        with (out / "run.log").open("x") as f:
            proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    valid, problem = True, None
    try:
        validate(campaign)
    except (ValueError, OSError) as exc:
        valid, problem = False, str(exc)
    write_new(out / "process.json", {"returncode": proc.returncode, "inputs_still_valid": valid, "input_problem": problem})
    print(json.dumps({"cell": cell["name"], "returncode": proc.returncode, "inputs_still_valid": valid}))
    return 0 if proc.returncode == 0 and valid else 1


def summarize(args):
    campaign = args.campaign.resolve()
    p = validate(campaign)
    rows, raw, problems = {}, {}, []
    for cell in p["cells"]:
        d = campaign / "runs" / cell["name"]
        es = {}
        for seed in p["seeds"]:
            try:
                e = json.loads((d / f"seed{seed}.json").read_text())
                checks = [e["seed"] == seed, e["gait"]["checkpoint_sha256"] == p["source_gait_sha256"],
                          e["sensor_mode"] == cell["sensor_mode"], e["cmd_filter"]["mode"] == cell["mode"],
                          e["policy"] == str(campaign / "models/actor.onnx"),
                          e["maze"]["n"] == 6, e["maze"]["m"] == 6, e["maze"]["extra_openings"] == 1]
                if not all(checks):
                    problems.append(f"{cell['name']} seed {seed}: contract mismatch")
                es[seed] = e
            except (OSError, ValueError, KeyError) as exc:
                problems.append(f"{cell['name']} seed {seed}: {exc}")
        try:
            proc = json.loads((d / "process.json").read_text())
            receipt = json.loads((d / "receipt.json").read_text())
            if proc["returncode"] != 0 or not proc["inputs_still_valid"] or receipt["protocol_sha256"] != digest(campaign / "protocol.json"):
                problems.append(f"{cell['name']}: process/integrity mismatch")
        except (OSError, KeyError, ValueError) as exc:
            problems.append(f"{cell['name']}: {exc}")
        values = list(es.values())
        raw[cell["name"]] = es
        rows[cell["name"]] = {"episodes": len(values), "goals": sum(e["success"] for e in values),
                              "clean_goals": sum(e["clean_success"] for e in values), "falls": sum(e["outcome"] == "fall" for e in values),
                              "wall_contact_steps": sum(e["wall_contact_steps"] for e in values),
                              "median_gait_flips_per_s": statistics.median(e["cmd_filter"]["gait_wz"]["flips_per_s"] for e in values) if values else None,
                              "median_gait_saturated_share": statistics.median(e["cmd_filter"]["gait_wz"]["saturated_share"] for e in values) if values else None}
    paired = {}
    for condition in ("nominal", "drop35"):
        baseline, candidate = raw[f"none-{condition}"], raw[f"wz-lpf-{condition}"]
        shared = sorted(set(baseline) & set(candidate))
        for seed in shared:
            if baseline[seed]["maze"]["layout_sha256"] != candidate[seed]["maze"]["layout_sha256"]:
                problems.append(f"{condition} seed {seed}: layout mismatch")
        both = [s for s in shared if baseline[s]["success"] and candidate[s]["success"]]
        paired[condition] = {"matched_layouts": len(shared), "both_goal": len(both),
                             "baseline_only_goal": sum(baseline[s]["success"] and not candidate[s]["success"] for s in shared),
                             "candidate_only_goal": sum(candidate[s]["success"] and not baseline[s]["success"] for s in shared),
                             "matched_success_time_delta_s": [candidate[s]["completion_s"] - baseline[s]["completion_s"] for s in both]}
    verdict = {"status": "INCOMPLETE" if problems else "COMPLETED_DEVELOPMENT_PILOT", "problems": problems,
               "per_cell": rows, "paired": paired, "protocol_sha256": digest(campaign / "protocol.json"),
               "limitations": [p["scope"], "Two shared layouts and one actor; results cannot establish navigation improvement/generalization.",
                               "Existing yaw LPF evaluated; no new training or governor implementation. Default path unchanged.",
                               "Fresh development seeds are consumed by this pilot, not available as a future untouched confirmatory test."]}
    write_new(campaign / "development_report.json", verdict)
    print(json.dumps(verdict, indent=2))
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("prepare", "run", "summary"))
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--gait", type=Path)
    ap.add_argument("--gait-checkpoint", type=Path)
    ap.add_argument("--actor", type=Path)
    ap.add_argument("--upstream", type=Path)
    ap.add_argument("--cell", type=int)
    args = ap.parse_args()
    if args.mode == "prepare":
        if any(x is None for x in (args.gait, args.actor, args.upstream)):
            ap.error("prepare requires gait, actor and upstream")
        return prepare(args)
    return run_cell(args) if args.mode == "run" else summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
