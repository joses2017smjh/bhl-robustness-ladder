"""Bounded native MuJoCo terrain qualification followed by untouched sensor trials."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bhl_robust.research.terrain_traversal import (ARMS, TERRAINS, TraversalSettings,
    evaluate_episode, make_environment, prepare_course, qualified_actor)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def make_protocol(args) -> int:
    from omegaconf import OmegaConf
    actors = []
    inputs = {}
    for actor in args.actor:
        name, deploy_name = actor.split(":", 1)
        if not name or "/" in name or "--" in name or name in (".", ".."):
            raise ValueError("actor name must be a simple nonempty identifier")
        deploy = Path(deploy_name).resolve()
        cfg = OmegaConf.load(deploy)
        checkpoint = Path(cfg.policy_checkpoint_path).resolve()
        if int(cfg.num_joints) != 12:
            raise ValueError("this protocol screens the frozen 12-DoF biped actors")
        actors.append({"name": name, "deploy": str(deploy), "checkpoint": str(checkpoint),
                       "checkpoint_sha256": digest(checkpoint), "variant": "biped"})
        inputs[str(deploy)] = digest(deploy)
        inputs[str(checkpoint)] = digest(checkpoint)
    if not actors or len(actors) > 3 or len({a["name"] for a in actors}) != len(actors):
        raise ValueError("one to three uniquely named frozen actors required")
    upstream = args.upstream.resolve()
    for path in sorted((upstream / "source/berkeley_humanoid_lite_assets/data/robots/berkeley_humanoid/berkeley_humanoid_lite").rglob("*")):
        if path.is_file() and path.suffix in (".xml", ".stl"):
            inputs[str(path)] = digest(path)
    if len(inputs) <= 2*len(actors):
        raise ValueError("actual recursive upstream assets are required")
    for path in (Path(__file__).resolve(), ROOT/"src/bhl_robust/research/terrain_traversal.py",
                 ROOT/"src/bhl_robust/research/sensor_geometry.py", ROOT/"src/bhl_robust/eval/harness.py",
                 ROOT/"src/bhl_robust/eval/gait_clock.py", ROOT/"src/bhl_robust/eval/mjcf_assets.py"):
        inputs[str(path)] = digest(path)
    lowlevel = upstream / "source/berkeley_humanoid_lite_lowlevel"
    for path in sorted(lowlevel.rglob("*.py")):
        if path.is_file():
            inputs[str(path)] = digest(path)
    protocol = {"schema": "bhl-terrain-traversal-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "upstream": str(upstream), "actors": actors, "settings": asdict(TraversalSettings()),
                "screen_groups": [250000, 250001], "confirmation_groups": list(range(260000, 260008)),
                "screen_terrains": list(TERRAINS), "confirmation_terrains": list(TERRAINS),
                "confirmation_arms": list(ARMS), "screen_rule": "each actor 6/6 clean goals, zero falls/collisions/nonfinite; retain every passing actor",
                "confirmation_rule": "per retained actor/arm >=22/24 clean goals, zero falls/collisions/nonfinite; paired descriptive comparisons, no independent safety claim",
                "scope": "MuJoCo learned-gait simulation; instantaneous 3D lidar/IMU terrain governor; no native SLAM, physical data or terrain-adapted gait training",
                "input_sha256": inputs,
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "mujoco", "onnxruntime", "omegaconf", "Pillow")}}
    if args.launcher:
        # The outer H3/H4 launcher freezes this single predeclared file. Only
        # source-relative names and named read-only input members survive into
        # the scientific block, so compute-node scratch relocation is safe.
        input_files = {}
        portable_actors = []
        actor_paths = set()
        for actor in actors:
            portable = {"name": actor["name"], "variant": actor["variant"],
                        "checkpoint_sha256": actor["checkpoint_sha256"]}
            for kind in ("deploy", "checkpoint"):
                name = f"actors/{actor['name']}/"+("deploy.yaml" if kind == "deploy" else "policy.onnx")
                input_files[name] = {"path": actor[kind], "sha256": inputs[actor[kind]]}
                portable[kind+"_input"] = name
                actor_paths.add(actor[kind])
            portable_actors.append(portable)
        source_sha = {str(Path(name).relative_to(ROOT)): sha for name, sha in inputs.items() if name not in actor_paths}
        scientific = {k: v for k, v in protocol.items() if k not in ("input_sha256", "upstream", "actors")}
        scientific.update(upstream_source=str(upstream.relative_to(ROOT)), actors=portable_actors,
                          source_sha256=source_sha)
        jobs = []
        for phase, kind, limit, cap in (("smoke", "smoke", "00:20:00", 64),
                                       ("run", "run", "05:00:00", 1024)):
            jobs.append({"name": f"terrain-{phase}-20261009", "kind": kind,
                         "entrypoint": "scripts/bench/terrain_traversal_campaign.py",
                         "args": ["--protocol", "{protocol}", "--phase", phase],
                         "resources": {"cpus": 4, "memory_gb": 8, "gpus": 0,
                                       "partition": "share,eecs", "time_limit": limit},
                         "env": {}, "max_output_mb": cap})
        protocol = {"schema_version": 1, "campaign_id": "terrain-traversal-20261009",
                    "runtime": {"python": "/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python",
                                "share_root": "/nfs/hpc/share/sanchej7",
                                "sif": "/nfs/hpc/share/sanchej7/Humanoid_Lite/container/bhl.sif", "stack": "cpu"},
                    "additional_source": ["scripts/bench/terrain_traversal_campaign.py",
                                          "src/bhl_robust/research/terrain_traversal.py"],
                    "input_files": input_files, "terrain_traversal": scientific, "jobs": jobs,
                    "scope": scientific["scope"],
                    "execution": "fresh two-episode smoke, then 18-episode controller screen followed immediately by conditional untouched confirmation; stop if no actor qualifies"}
    write_new(args.output, protocol)
    print(json.dumps({"protocol": str(args.output), "actors": len(actors), "screen_episodes": 6*len(actors),
                      "maximum_confirmation_episodes": 72*len(actors)}))
    return 0


def validate(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    if protocol.get("schema_version") == 1 and "terrain_traversal" in protocol:
        outer = protocol
        protocol = dict(outer["terrain_traversal"])
        source = Path(os.getenv("REPO", ROOT)).resolve()
        input_root = Path(os.getenv("H34_INPUTS_DIR", path.parent/"inputs")).resolve()
        def resolve_member(root, name):
            relative = Path(name)
            member = root/relative
            if (relative.is_absolute() or ".." in relative.parts or not str(relative) == name or
                    not member.resolve().is_relative_to(root)):
                raise ValueError("unsafe relocated terrain input member")
            return str(member)
        protocol["upstream"] = resolve_member(source, protocol["upstream_source"])
        protocol["input_sha256"] = {resolve_member(source, name): sha
                                    for name, sha in protocol["source_sha256"].items()}
        actors = []
        for actor in protocol["actors"]:
            relocated = dict(actor)
            for kind in ("deploy", "checkpoint"):
                name = actor[kind+"_input"]
                relocated[kind] = resolve_member(input_root, name)
                protocol["input_sha256"][relocated[kind]] = outer["input_files"][name]["sha256"]
            actors.append(relocated)
        protocol["actors"] = actors
    if protocol["schema"] != "bhl-terrain-traversal-v1":
        raise ValueError("wrong traversal protocol schema")
    settings = TraversalSettings(**protocol["settings"])
    settings.validate()
    if (settings.seconds != 40. or settings.goal_distance_m != 5. or
            protocol["screen_terrains"] != list(TERRAINS) or
            protocol["confirmation_terrains"] != list(TERRAINS) or
            protocol["confirmation_arms"] != list(ARMS) or
            len(protocol["screen_groups"]) != 2 or len(protocol["confirmation_groups"]) != 8 or
            set(protocol["screen_groups"]) & set(protocol["confirmation_groups"])):
        raise ValueError("changed predeclared five metre qualification/cohort contract")
    if not protocol["input_sha256"]:
        raise ValueError("immutable input inventory required")
    for name, expected in protocol["input_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"frozen terrain campaign input changed: {name}")
    for name, expected in protocol["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"package changed: {name}")
    for actor in protocol["actors"]:
        if protocol["input_sha256"].get(actor["checkpoint"]) != actor["checkpoint_sha256"]:
            raise ValueError("actor checkpoint is not in immutable inventory")
    return protocol


def load_controller(actor: dict, upstream: Path):
    from omegaconf import OmegaConf
    sys.path.insert(0, str(upstream/"source/berkeley_humanoid_lite_lowlevel"))
    from bhl_robust.eval.gait_clock import make_controller
    import onnxruntime as ort

    class CpuPolicy:
        def __init__(self, checkpoint):
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(checkpoint), options, providers=["CPUExecutionProvider"])
            self.key = self.session.get_inputs()[0].name

        def forward(self, observation):
            return self.session.run(None, {self.key: observation})[0]

    cfg = OmegaConf.load(actor["deploy"])
    cfg.policy_checkpoint_path = actor["checkpoint"]
    controller = make_controller(cfg)
    controller.policy = CpuPolicy(actor["checkpoint"])
    return cfg, controller


def run(args) -> int:
    p = validate(args.protocol)
    settings = TraversalSettings(**p["settings"])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol_sha = digest(args.protocol)
    started = time.monotonic()
    receipt = {"protocol_sha256": protocol_sha, "mode": args.mode, "host": platform.node(),
               "slurm_job_id": os.getenv("SLURM_JOB_ID"), "created_utc": datetime.now(timezone.utc).isoformat(),
               "input_sha256": p["input_sha256"], "packages": p["packages"]}
    write_new(out/"receipt.json", receipt)
    actors = p["actors"]
    if args.mode == "run":
        if args.screen is None:
            raise ValueError("confirmation requires the completed qualification screen")
        screen = json.loads((args.screen/"campaign_result.json").read_text())
        if (screen["protocol_sha256"] != protocol_sha or screen["mode"] != "screen" or
                screen["status"] not in ("PASS", "NEGATIVE_CONTROLLER_SCREEN")):
            raise ValueError("screen does not match the immutable campaign protocol")
        # Recompute eligibility from each raw episode; don't trust a saved selection list.
        selected = []
        for actor in actors:
            rows = [json.loads((args.screen/name).read_text()) for name in screen["episode_files"]
                    if name.startswith(actor["name"]+"--")]
            for row in rows:
                if (row["protocol_sha256"] != protocol_sha or
                        row["checkpoint_sha256"] != actor["checkpoint_sha256"] or row["arm"] != "baseline"):
                    raise ValueError("screen episode identity mismatch")
            if qualified_actor(rows):
                selected.append(actor)
        actors = selected
        if not actors:
            write_new(out/"campaign_result.json", {"status": "SKIPPED_NO_QUALIFIED_ACTOR", "mode": args.mode,
                "protocol_sha256": protocol_sha, "episodes": 0, "qualified_actors": [],
                "scientific_status": "NEGATIVE_CONTROLLER_SCREEN", "reason": "No frozen gait passed all six physical qualification traversals; confirmation was not run."})
            return 0
    from bhl_robust.eval.mjcf_assets import prepare_mjcf
    rows, episode_files = [], []
    groups = p["screen_groups"] if args.mode in ("smoke", "screen") else p["confirmation_groups"]
    arms = ["baseline"] if args.mode in ("smoke", "screen") else p["confirmation_arms"]
    if args.mode == "smoke":
        actors, groups, terrains = actors[:1], groups[:1], ("flat", "ramp")
    else:
        terrains = p["screen_terrains"] if args.mode == "screen" else p["confirmation_terrains"]
    with tempfile.TemporaryDirectory(prefix="bhl-sensed-terrain-") as directory:
        cache = Path(directory)
        flat = prepare_mjcf(Path(p["upstream"]), cache/"assets", "biped")
        for actor in actors:
            cfg, controller = load_controller(actor, Path(p["upstream"]))
            for terrain in terrains:
                for group in groups:
                    name = f"{actor['name']}--{terrain}--g{group}"
                    scene, geometry = prepare_course(flat, cache/name, terrain, group, settings)
                    trial_arms = (["lidar"] if args.mode == "smoke" and terrain == "ramp" else arms)
                    for arm in trial_arms:
                        stem = name+"--"+arm
                        environment = make_environment(cfg, flat, scene)
                        row = evaluate_episode(environment, controller, arm, group+1000000, settings,
                                               out/(stem+".npz"), sensor_seed=group+2000000)
                        row.update(actor=actor["name"], checkpoint_sha256=actor["checkpoint_sha256"],
                                   protocol_sha256=protocol_sha, terrain=terrain, group=group,
                                   geometry_evaluator=geometry, raw_scan_file=stem+".npz",
                                   raw_scan_sha256=digest(out/(stem+".npz")))
                        write_new(out/(stem+".json"), row)
                        episode_files.append(stem+".json")
                        rows.append(row)
                        print(json.dumps({k: row[k] for k in ("actor", "terrain", "group", "arm", "success", "fell", "collision", "forward_progress_m")}), flush=True)
    validate(args.protocol)
    qualified = [a["name"] for a in actors if qualified_actor([r for r in rows if r["actor"] == a["name"]])]
    if args.mode == "screen":
        status = "PASS" if qualified else "NEGATIVE_CONTROLLER_SCREEN"
    else:
        status = "PASS"  # execution status; scientific target has its own explicit field.
    arm_results = {}
    for actor in actors:
        for arm in sorted({row["arm"] for row in rows}):
            subset = [r for r in rows if r["actor"] == actor["name"] and r["arm"] == arm]
            arm_results[actor["name"]+"/"+arm] = {"episodes": len(subset),
                "successes": sum(r["success"] for r in subset), "falls": sum(r["fell"] for r in subset),
                "collisions": sum(r["collision"] for r in subset), "nonfinite": sum(r["nonfinite"] for r in subset),
                "timeouts": sum(r["failure"] == "timeout" for r in subset),
                "target_met": bool(args.mode == "run" and len(subset) == 24 and sum(r["success"] for r in subset) >= 22 and
                                   not any(r["fell"] or r["collision"] or r["nonfinite"] for r in subset))}
    scientific = ("SMOKE_ONLY" if args.mode == "smoke" else status if args.mode == "screen" else
                  "TARGET_MET" if all(a["target_met"] for a in arm_results.values()) else "TARGET_NOT_MET")
    result = {"status": status, "scientific_status": scientific, "mode": args.mode,
              "protocol_sha256": protocol_sha, "episodes": len(rows), "qualified_actors": qualified if args.mode == "screen" else [a["name"] for a in actors],
              "episode_files": episode_files, "arm_results": arm_results,
              "wall_seconds": time.monotonic()-started, "scope": p["scope"],
              "controller_inputs": "raw instantaneous 3D lidar returns and simulated IMU attitude; frozen gait proprioception; no terrain truth or translational pose",
              "evaluator_inputs": "simulator pose and independent terrain-height rays; evaluator goal stop after crossing; full matched horizon required",
              "stereo_fusion": "NOT_RUN: this bounded campaign isolates lidar mapping and point dropout; prior stereo comparison remains separate",
              "confidence": "Shared actor and terrain groups are paired clusters; episode counts do not establish independent robot safety."}
    write_new(out/"campaign_result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "scientific_status", "episodes", "qualified_actors", "wall_seconds")}))
    return 0


def frozen_entrypoint(args) -> int:
    """One bounded run performs qualification before spending on confirmation."""
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    top = {"status": "INCOMPLETE", "phase": args.phase, "route": "sensor_informed_terrain_traversal",
           "protocol_sha256": digest(args.protocol), "stages": [], "problems": []}
    try:
        validate(args.protocol)
        if args.phase == "smoke":
            run(SimpleNamespace(mode="smoke", protocol=args.protocol, output=output/"smoke"))
            child = json.loads((output/"smoke/campaign_result.json").read_text())
            top.update(status="PASS", scientific_status="SMOKE_ONLY", episodes=child["episodes"],
                       qualified_actors=[], stages=[{"name": "smoke", "result": "smoke/campaign_result.json"}])
        else:
            run(SimpleNamespace(mode="screen", protocol=args.protocol, output=output/"screen"))
            screen = json.loads((output/"screen/campaign_result.json").read_text())
            top["stages"].append({"name": "screen", "result": "screen/campaign_result.json"})
            top.update(screen_episodes=screen["episodes"], qualified_actors=screen["qualified_actors"])
            if args.phase == "screen":
                top.update(status="PASS" if screen["qualified_actors"] else "NEGATIVE",
                           scientific_status=screen["scientific_status"], confirmation_episodes=0)
            else:
                run(SimpleNamespace(mode="run", protocol=args.protocol, output=output/"confirmation", screen=output/"screen"))
                confirmation = json.loads((output/"confirmation/campaign_result.json").read_text())
                top["stages"].append({"name": "confirmation", "result": "confirmation/campaign_result.json"})
                target = confirmation.get("scientific_status") == "TARGET_MET"
                top.update(status="PASS" if target else "NEGATIVE", scientific_status=confirmation["scientific_status"],
                           confirmation_episodes=confirmation["episodes"], arm_results=confirmation.get("arm_results", {}))
        top["scope"] = "Actual learned biped physics and current-scan 3D lidar/IMU terrain inference; simulated sensors, no SLAM or physical trials"
        top["acceptance_meaning"] = "Negative controller qualification stops confirmation; no fall or goal is inferred from process completion"
    except Exception as error:
        top["problems"].append(f"{type(error).__name__}: {error}")
    top["wall_seconds"] = time.monotonic()-started
    write_new(output/"campaign_result.json", top)
    return 0 if top["status"] in ("PASS", "NEGATIVE") else 1


def main() -> int:
    if "--phase" in sys.argv[1:]:
        frozen = argparse.ArgumentParser(description=__doc__)
        frozen.add_argument("--phase", choices=("smoke", "screen", "run"), required=True)
        frozen.add_argument("--protocol", type=Path, default=os.getenv("H34_PROTOCOL"))
        frozen.add_argument("--output", type=Path, default=os.getenv("H34_OUTPUT_DIR"))
        args = frozen.parse_args()
        if args.protocol is None or args.output is None:
            frozen.error("protocol/output or the frozen H34 environment are required")
        return frozen_entrypoint(args)
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="mode", required=True)
    create = commands.add_parser("make-protocol")
    create.add_argument("--upstream", type=Path, required=True)
    create.add_argument("--actor", action="append", required=True, help="simple_name:/absolute/deploy.yaml")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--launcher", action="store_true", help="emit one portable frozen H34 launcher protocol")
    for mode in ("smoke", "screen", "run"):
        command = commands.add_parser(mode)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if mode == "run":
            command.add_argument("--screen", type=Path, required=True)
    args = ap.parse_args()
    return make_protocol(args) if args.mode == "make-protocol" else run(args)


if __name__ == "__main__":
    sys.exit(main())
