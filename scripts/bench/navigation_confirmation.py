"""Frozen matched NavGym confirmation; preparation precedes all scored runs.

Three final actors, two sensing conditions, 48 fresh layouts and two fixed
arms (576 episodes). Runtime smoke uses a separate seed and is never scored.
Only the existing pre-brake yaw filter is evaluated; no policy is retrained.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
import time

REPO = Path(__file__).resolve().parents[2]
ACTORS = ("armV5-s8", "armV5-s9", "armV5-s10")
SEEDS = tuple(range(120000, 120048))
SMOKE_SEED = 119999


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate(campaign):
    p = json.loads((campaign / "protocol.json").read_text())
    for name, expected in p["input_sha256"].items():
        f = Path(name) if name.startswith("/") else campaign / name
        if not f.is_file() or digest(f) != expected:
            raise ValueError(f"frozen confirmation input changed/missing: {name}")
    for name, expected in p["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"package mismatch: {name}")
    return p


def prepare(a):
    from omegaconf import OmegaConf
    campaign = a.campaign.resolve()
    campaign.mkdir(parents=True, exist_ok=False)
    snapshot = campaign / "snapshot"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(REPO / "src", snapshot / "src", ignore=ignore)
    shutil.copytree(REPO / "scripts/bench", snapshot / "scripts/bench", ignore=ignore)
    low = a.upstream / "source/berkeley_humanoid_lite_lowlevel/berkeley_humanoid_lite_lowlevel"
    shutil.copytree(low, snapshot / "upstream_lowlevel/berkeley_humanoid_lite_lowlevel", ignore=ignore)
    cfg = OmegaConf.load(a.gait)
    gait = Path(cfg.policy_checkpoint_path)
    (campaign / "models").mkdir()
    shutil.copyfile(gait, campaign / "models/gait.onnx")
    cfg.policy_checkpoint_path = "models/gait.onnx"  # run with cwd=campaign; relocation preserves bytes
    OmegaConf.save(cfg, campaign / "models/deploy.yaml")
    models = {}
    for actor in ACTORS:
        src = a.actors / actor / "actor.onnx"
        shutil.copyfile(src, campaign / "models" / f"{actor}.onnx")
        models[actor] = {"source": str(src.resolve()), "sha256": digest(src)}
    files = {str(f.relative_to(campaign)): digest(f) for f in sorted(snapshot.rglob("*")) if f.is_file()}
    files.update({str(f.relative_to(campaign)): digest(f) for f in sorted((campaign / "models").glob("*"))})
    robot = a.upstream / "source/berkeley_humanoid_lite_assets/data/robots/berkeley_humanoid/berkeley_humanoid_lite"
    for sub in ("mjcf", "meshes"):
        for f in sorted((robot / sub).rglob("*")):
            if f.is_file():
                files[str(f.resolve())] = digest(f)
    cells = [{"name": f"{actor}-{mode}-{condition}", "actor": actor, "mode": mode,
              "condition": condition, "sensor_mode": "reactive" if condition == "nominal" else "reactive_dropout"}
             for actor in ACTORS for mode in ("none", "wz-lpf") for condition in ("nominal", "drop35")]
    p = {"schema": "bhl-h1-confirmation-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
         "scope": "Fresh matched MuJoCo simulation confirmation; three frozen NavGym actors; no training or hardware.",
         "upstream": str(a.upstream.resolve()),
         "overlay_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
         "source_gait": str(gait), "source_gait_sha256": digest(gait), "source_actors": models,
         "candidate": "wz-lpf", "baseline": "none", "tau_s": .2,
         "candidate_lock": "Existing 0.2 s pre-brake yaw LPF; chosen from gait/gym lag before this cohort; no post-score selection or tuning.",
         "seeds": list(SEEDS), "cells": cells, "max_episodes": 576, "max_concurrent_evaluators": 2,
         "time_limit_s": 180., "settings": {"n": 6, "m": 6, "extra_openings": 1, "settle_s": 1.,
             "cruise": .30, "turn_rate": .6, "inflate": .30, "replan_s": .4, "map_res": .10,
             "dropout_probability": .35, "random_heading": False, "policy_capture_pose": True},
         "runtime_smoke": {"seed": SMOKE_SEED, "actor": "armV5-s9", "conditions": ["nominal"],
             "arms": ["none", "wz-lpf"], "time_limit_s": 30., "episodes": 2,
             "scope": "Runtime/intake only, excluded from all confirmation statistics; cannot select/tune candidate."},
         "candidate_rule": "EACH actor >=40/48 nominal goals and >=36/48 drop35 goals, with zero falls in both conditions. Missing/invalid/mispaired episodes -> INCOMPLETE; failed clause -> NEGATIVE.",
         "comparison_rule": "Report all paired goal/fall outcomes, contacts, command flips and matched successful completion times. Claim fall reduction only if the complete matched baseline has more falls; zero candidate falls alone cannot establish reduction.",
         "statistical_scope": "48 shared layouts/reset seeds across 3 trained actors and 2 conditions; pooled episodes are not independent trained policies. No hardware robustness claim.",
         "labels": {"learned": "frozen 12-DoF gait + NavGym v5 actor", "scripted": "speed brake + candidate yaw LPF", "oracle": "pose + goal"},
         "seed_audit": json.loads(a.seed_audit.read_text()), "input_sha256": files,
         "packages": {n: importlib.metadata.version(n) for n in ("mujoco", "numpy", "onnxruntime", "omegaconf", "torch")}}
    write_new(campaign / "protocol.json", p)
    print(json.dumps({"prepared": str(campaign), "protocol_sha256": digest(campaign / "protocol.json"), "episodes": 576}), flush=True)
    return 0


def run_cell(campaign, cell, smoke=False):
    p = validate(campaign)
    seed, count, limit = (SMOKE_SEED, 1, 30.) if smoke else (SEEDS[0], len(SEEDS), p["time_limit_s"])
    out = campaign / ("runtime-smoke" if smoke else "runs") / cell["name"]
    out.mkdir(parents=True, exist_ok=False)
    cmd = [sys.executable, str(campaign / "snapshot/scripts/bench/maze_explore.py"),
           "--upstream", p["upstream"], "--gait", str(campaign / "models/deploy.yaml"),
           "--policy", str(campaign / "models" / f"{cell['actor']}.onnx"), "--policy-capture-pose",
           "--policy-cmd-filter", cell["mode"], "--policy-cmd-tau", str(p["tau_s"]),
           "--sensor-mode", cell["sensor_mode"], "--dropout-probability", ".35",
           "--seeds", str(count), "--seed-start", str(seed), "--n", "6", "--m", "6",
           "--extra-openings", "1", "--time-limit", str(limit), "--out-dir", str(out), "--no-overwrite"]
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(campaign / "snapshot/src"), str(campaign / "snapshot/upstream_lowlevel")]),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="bhl-nav-confirm-") as cache:
        cmd += ["--cache-dir", cache]
        write_new(out / "receipt.json", {"command": cmd, "cwd": str(campaign), "cell": cell,
            "protocol_sha256": digest(campaign / "protocol.json"), "host": platform.node(),
            "platform": platform.platform(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "scope": "UNSCORED runtime smoke" if smoke else "confirmation", "created_utc": datetime.now(timezone.utc).isoformat()})
        with (out / "run.log").open("x") as stream:
            proc = subprocess.run(cmd, cwd=campaign, env=env, stdout=stream, stderr=subprocess.STDOUT)
    valid, problem = True, None
    try:
        validate(campaign)
    except (ValueError, OSError) as exc:
        valid, problem = False, str(exc)
    rec = {"returncode": proc.returncode, "inputs_still_valid": valid, "input_problem": problem,
           "wall_s": round(time.monotonic() - started, 3)}
    write_new(out / "process.json", rec)
    print(json.dumps({"cell": cell["name"], **rec}), flush=True)
    return rec


def run(a):
    campaign = a.campaign.resolve()
    p = validate(campaign)
    if a.mode == "smoke":
        results = [run_cell(campaign, c, smoke=True) for c in p["cells"]
                   if c["actor"] == "armV5-s9" and c["condition"] == "nominal"]
        write_new(campaign / "runtime-smoke.json", {"scope": p["runtime_smoke"], "processes": results,
            "protocol_sha256": digest(campaign / "protocol.json"), "host": platform.node()})
    else:
        # A write-once run marker also prevents a second coordinator doubling concurrency.
        write_new(campaign / "run-start.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": digest(campaign / "protocol.json"), "max_workers": 2, "host": platform.node()})
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda c: run_cell(campaign, c), p["cells"]))
    return 0 if all(r["returncode"] == 0 and r["inputs_still_valid"] for r in results) else 1


def summary(a):
    campaign = a.campaign.resolve()
    p = validate(campaign)
    rows, raw, problems = {}, {}, []
    for cell in p["cells"]:
        d, episodes = campaign / "runs" / cell["name"], {}
        for seed in p["seeds"]:
            try:
                e = json.loads((d / f"seed{seed}.json").read_text())
                checks = [e["seed"] == seed, e["gait"]["checkpoint_sha256"] == p["source_gait_sha256"],
                    e["sensor_mode"] == cell["sensor_mode"], e["cmd_filter"]["mode"] == cell["mode"],
                    e["policy"] == str(campaign / "models" / f"{cell['actor']}.onnx"),
                    e["maze"]["n"] == 6, e["maze"]["m"] == 6, e["maze"]["extra_openings"] == 1,
                    e["policy_map_integration"]["mode"] == "capture_pose",
                    e["success"] == (e["outcome"] == "goal"),
                    e["clean_success"] == (e["success"] and e["wall_contact_steps"] == 0),
                    cell["mode"] == "none" or e["cmd_filter"]["tau_s"] == .2]
                if not all(checks):
                    problems.append(f"{cell['name']} seed {seed}: contract mismatch")
                episodes[seed] = e
            except (OSError, ValueError, KeyError) as exc:
                problems.append(f"{cell['name']} seed {seed}: {exc}")
        try:
            proc = json.loads((d / "process.json").read_text())
            receipt = json.loads((d / "receipt.json").read_text())
            if proc["returncode"] or not proc["inputs_still_valid"] or receipt["protocol_sha256"] != digest(campaign / "protocol.json"):
                problems.append(f"{cell['name']}: process/integrity mismatch")
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f"{cell['name']}: {exc}")
        raw[cell["name"]] = episodes
        values = list(episodes.values())
        rows[cell["name"]] = {"episodes": len(values), "goals": sum(e["success"] for e in values),
            "clean_goals": sum(e["clean_success"] for e in values), "falls": sum(e["outcome"] == "fall" for e in values),
            "wall_contact_steps": sum(e["wall_contact_steps"] for e in values),
            "median_gait_flips_per_s": statistics.median(e["cmd_filter"]["gait_wz"]["flips_per_s"] for e in values) if values else None}
    paired = {}
    for actor in ACTORS:
        for condition in ("nominal", "drop35"):
            b, c = raw[f"{actor}-none-{condition}"], raw[f"{actor}-wz-lpf-{condition}"]
            shared = sorted(set(b) & set(c))
            for seed in shared:
                if b[seed]["maze"]["layout_sha256"] != c[seed]["maze"]["layout_sha256"]:
                    problems.append(f"{actor} {condition} seed {seed}: layout mismatch")
            both = [s for s in shared if b[s]["success"] and c[s]["success"]]
            paired[f"{actor}-{condition}"] = {"matched_layouts": len(shared), "both_goal": len(both),
                "baseline_only_goal": sum(b[s]["success"] and not c[s]["success"] for s in shared),
                "candidate_only_goal": sum(c[s]["success"] and not b[s]["success"] for s in shared),
                "baseline_only_fall": sum(b[s]["outcome"] == "fall" and c[s]["outcome"] != "fall" for s in shared),
                "candidate_only_fall": sum(c[s]["outcome"] == "fall" and b[s]["outcome"] != "fall" for s in shared),
                "matched_success_time_delta_s": [c[s]["completion_s"] - b[s]["completion_s"] for s in both]}
    per_actor = {actor: all(rows[f"{actor}-wz-lpf-{condition}"]["goals"] >= threshold and
        rows[f"{actor}-wz-lpf-{condition}"]["falls"] == 0 for condition, threshold in (("nominal", 40), ("drop35", 36))) for actor in ACTORS}
    bfalls = sum(r["falls"] for k, r in rows.items() if "-none-" in k)
    cfalls = sum(r["falls"] for k, r in rows.items() if "-wz-lpf-" in k)
    status = "INCOMPLETE" if problems else "PASS" if all(per_actor.values()) else "NEGATIVE"
    report = {"status": status, "candidate_rule": p["candidate_rule"], "problems": problems, "per_cell": rows,
        "per_actor_pass": per_actor, "paired": paired, "protocol_sha256": digest(campaign / "protocol.json"),
        "fall_comparison": {"baseline_falls": bfalls, "candidate_falls": cfalls,
            "fewer_observed_candidate_falls": not problems and cfalls < bfalls},
        "limitations": [p["scope"], p["statistical_scope"], "Runtime smoke is excluded; no post-score candidate tuning or cohort exclusion."]}
    write_new(campaign / "confirmation_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return 1 if status == "INCOMPLETE" else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("prepare", "smoke", "run", "summary"))
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--gait", type=Path)
    ap.add_argument("--actors", type=Path)
    ap.add_argument("--upstream", type=Path)
    ap.add_argument("--seed-audit", type=Path)
    a = ap.parse_args()
    if a.mode == "prepare":
        if any(x is None for x in (a.gait, a.actors, a.upstream, a.seed_audit)):
            ap.error("prepare needs gait, actors, upstream, seed-audit")
        return prepare(a)
    return summary(a) if a.mode == "summary" else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
