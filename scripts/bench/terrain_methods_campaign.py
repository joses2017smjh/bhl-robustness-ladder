#!/usr/bin/env python3
"""Frozen-gait matched terrain control campaign with native FAST-LIO2 feedback."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from bhl_robust.research.native_lio import NativeLioClient, sha256
from bhl_robust.research.pose_metrics import transform
from bhl_robust.research.terrain_methods import (ARMS, TERRAINS, DEVELOPMENT_SEEDS,
    CONFIRMATION_SEEDS, ControlSettings, MapSettings, UncertainElevationMap,
    ground_map_metrics, path_command)
from bhl_robust.research.terrain_traversal import TraversalSettings, prepare_course
from native_navigation_campaign import CausalSensors, DelayedNativeState, write_json

SCOPE = ("MuJoCo simulation development; three frozen12DoF actors; matched nativeFAST-LIO2 "
         "startup/loss/estimated-goal stops; new course/seed cohort; no physical validation, "
         "terrain-trained gait or independent confirmation claim")


def create_protocol(args):
    original = json.loads(args.template.read_text())
    scientific = original["terrain_traversal"]
    actors = scientific["actors"]
    if len(actors) != 3:
        raise ValueError("exactly three previously frozen actors required")
    inputs = dict(original["input_files"])
    runtime = args.runtime_archive.resolve()
    inputs["runtime/lio-runtime.tar.gz"] = {"path": str(runtime), "sha256": sha256(runtime)}
    source_files = list(scientific["source_sha256"])
    source_files.extend(["scripts/bench/terrain_methods_campaign.py", "src/bhl_robust/research/terrain_methods.py",
                         "scripts/bench/native_navigation_campaign.py", "scripts/bench/team_airlock.py",
                         "src/bhl_robust/research/native_lio.py", "src/bhl_robust/research/pose_metrics.py",
                         "src/bhl_robust/eval/multi_robot.py"])
    source = {name: sha256(ROOT/name) for name in sorted(set(source_files))}
    methods = {"schema": "bhl-terrain-methods-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "actors": actors, "upstream_source": scientific["upstream_source"], "source_sha256": source,
               "packages": scientific["packages"], "runtime_input": "runtime/lio-runtime.tar.gz",
               "arm_order": "deterministic permutation RNG(seed+500000) for each matched reset-seed group",
               "development_seeds": list(DEVELOPMENT_SEEDS), "confirmation_seeds_reserved": list(CONFIRMATION_SEEDS),
               "terrains": list(TERRAINS), "arms": list(ARMS), "seconds": 40., "initialization_seconds": 5.,
               "control_settings": asdict(ControlSettings()), "map_settings": asdict(MapSettings()),
               "scope": SCOPE, "ground_truth_inputs_to_controller": [],
               "primary_metric": "paired clean-goal success differences by actor/terrain/reset seed; full40s horizon, zero wall contacts/falls/nonfinite",
               "confirmation_gate": "For an actor/arm: >=27/30 clean goals and zero falls/contacts/nonfinite; reserve confirmation seeds and require separately frozen new geometry before use",
               "uncertainty": "seed-paired development summaries; three training seeds are not independent robot safety evidence",
               "old_campaign_unchanged": "18-episode20261009 qualification remains negative; no rescoring or replacing those episodes"}
    jobs = [{"name": "terrain-methods-smoke-20261010", "kind": "smoke",
             "entrypoint": "scripts/bench/terrain_methods_campaign.py", "args": ["--protocol", "{protocol}", "--phase", "smoke"],
             "resources": {"cpus": 4, "memory_gb": 8, "gpus": 0, "partition": "share,eecs", "time_limit": "00:30:00", "requeue": False},
             "env": {}, "max_output_mb": 128}]
    for actor in actors:
        for terrain in TERRAINS:
            jobs.append({"name": f"terrain-methods-{actor['name']}-{terrain}-20261010", "kind": "run",
                "entrypoint": "scripts/bench/terrain_methods_campaign.py",
                "args": ["--protocol", "{protocol}", "--phase", "development", "--actor", actor["name"], "--terrain", terrain],
                "resources": {"cpus": 4, "memory_gb": 8, "gpus": 0, "partition": "share,eecs", "time_limit": "04:00:00", "requeue": False},
                "env": {}, "max_output_mb": 1536})
    outer = {"schema_version": 1, "campaign_id": "terrain-methods-20261010", "runtime": original["runtime"],
             "additional_source": source_files, "input_files": inputs, "terrain_methods": methods, "jobs": jobs,
             "scope": SCOPE, "execution": "fresh three-arm8s native smoke then nine actor/terrain shards of30 episodes each; no confirmation in this protocol"}
    write_json(args.output, outer)
    print(json.dumps({"protocol": str(args.output), "development_episodes": 270, "smoke_episodes": 3, "jobs": len(jobs)}))


def validate_protocol(path):
    import importlib.metadata
    outer = json.loads(path.read_text())
    p = dict(outer["terrain_methods"])
    if (p.get("schema") != "bhl-terrain-methods-v1" or p["arms"] != list(ARMS) or
        p["terrains"] != list(TERRAINS) or p["development_seeds"] != list(DEVELOPMENT_SEEDS) or
        p["confirmation_seeds_reserved"] != list(CONFIRMATION_SEEDS) or p["seconds"] != 40. or
        p["initialization_seconds"] != 5. or len(p["actors"]) != 3 or
        p["control_settings"] != asdict(ControlSettings()) or p["map_settings"] != asdict(MapSettings())):
        raise ValueError("changed declared terrain methods experiment")
    source = Path(os.getenv("REPO", ROOT)).resolve()
    input_root = Path(os.getenv("H34_INPUTS_DIR", path.parent/"inputs")).resolve()
    relocated = bool(os.getenv("H34_WORK_DIR") or os.getenv("H34_INPUTS_DIR"))
    def member(root, name):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not (root/relative).resolve().is_relative_to(root):
            raise ValueError("unsafe frozen input member")
        return root/relative
    inventory = {str(member(source, name)): expected for name, expected in p["source_sha256"].items()}
    p["upstream"] = member(source, p["upstream_source"])
    actors = []
    for actor in p["actors"]:
        row = dict(actor)
        for kind in ("deploy", "checkpoint"):
            name = actor[kind+"_input"]
            value = member(input_root, name) if relocated else Path(outer["input_files"][name]["path"])
            row[kind] = str(value)
            inventory[str(value)] = outer["input_files"][name]["sha256"]
        if inventory[row["checkpoint"]] != actor["checkpoint_sha256"]:
            raise ValueError("frozen actor identity mismatch")
        actors.append(row)
    runtime_name = p["runtime_input"]
    runtime = member(input_root, runtime_name) if relocated else Path(outer["input_files"][runtime_name]["path"])
    inventory[str(runtime)] = outer["input_files"][runtime_name]["sha256"]
    for name, expected in inventory.items():
        if sha256(name) != expected:
            raise ValueError("frozen terrain methods source/input changed: "+name)
    for name, expected in p["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("frozen package changed: "+name)
    p.update(actors=actors, runtime_archive=runtime, input_inventory=inventory)
    return p


def make_world(terrain, seed, cache):
    from bhl_robust.eval.multi_robot import _FLAT_WORLD
    base = cache/"base.xml"
    base.write_text(_FLAT_WORLD)
    scene, geometry = prepare_course(base, cache/"course", terrain, seed, TraversalSettings())
    root = ET.parse(scene).getroot()
    world = root.find("worldbody")
    for geom in world.findall("geom"):
        if geom.get("name", "").startswith("course_wall_"):
            geom.set("name", "wall_"+geom.get("name"))
    # Raised exterior structures support native scan registration without
    # inserting obstacles into the1.5m walking corridor. They are new matched
    # course geometry, shared by all arms, not estimator coordinate inputs.
    for i, (x, y) in enumerate(((1., -.95), (2.7, .95), (4., -.95), (5.6, .95))):
        ET.SubElement(world, "geom", name=f"wall_external_feature_{i}", type="box", group="0",
                      pos=f"{x} {y} .95", size=".10 .10 .30", rgba=".2 .4 .7 1")
    geometry["external_registration_features"] = "four raised asymmetric boxes outside corridor, all arms identical"
    return ET.tostring(root, encoding="unicode"), geometry


class TerrainSensors(CausalSensors):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elevation = np.deg2rad(np.linspace(-75, 25, 16))


def evaluator_ground_grid(model, data, settings):
    """Ground-only independent rays; temporary geom groups restored immediately."""
    import mujoco
    x, y = settings.axes()
    truth = np.full((len(y), len(x)), np.nan)
    ground = np.zeros_like(truth, dtype=bool)
    old_groups = model.geom_group.copy()
    try:
        model.geom_group[:] = 0
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("terrain_"):
                model.geom_group[g] = 2
        hit = np.zeros(1, np.int32)
        for row, yy in enumerate(y):
            for col, xx in enumerate(x):
                # Exclude physical wall footprints and outside corridor; ray
                # mask additionally ensures even tall wall tops cannot be truth.
                if abs(yy) >= .75:
                    continue
                distance = mujoco.mj_ray(model, data, np.array([xx, yy, 3.]), np.array([0., 0., -1.]),
                                         np.array([0, 0, 1, 0, 0, 0], np.uint8), 1, -1, hit)
                if distance >= 0:
                    truth[row, col] = 3.-float(distance)
                    ground[row, col] = True
    finally:
        model.geom_group[:] = old_groups
    hazard = np.zeros_like(ground)
    for row, col in zip(*np.where(ground)):
        differences = []
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            r, c = row+dy, col+dx
            if 0 <= r < len(y) and 0 <= c < len(x) and ground[r, c]:
                differences.append(abs(truth[r, c]-truth[row, col]))
        if differences:
            maximum = max(differences)
            hazard[row, col] = maximum >= settings.hazard_step_m or math.degrees(math.atan(maximum/settings.resolution_m)) >= settings.hazard_slope_deg
    return truth, ground, hazard


def run_episode(p, actor, runtime, terrain, seed, arm, output, seconds):
    import mujoco
    from omegaconf import OmegaConf
    from bhl_robust.eval.gait_clock import make_controller
    from bhl_robust.eval.multi_robot import _WORLDS, build_multi
    sys.path.insert(0, str(p["upstream"]/"source/berkeley_humanoid_lite_lowlevel"))
    from team_airlock import ContactRunner, CpuPolicy
    output.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(actor["deploy"])
    cfg.policy_checkpoint_path = actor["checkpoint"]
    controller = make_controller(cfg)
    controller.policy = CpuPolicy(actor["checkpoint"])
    with tempfile.TemporaryDirectory(prefix="bhl-terrain-methods-") as directory:
        cache = Path(directory)
        world, geometry = make_world(terrain, seed, cache)
        _WORLDS["terrain_methods"] = world
        model, slots = build_multi(p["upstream"], cache/"assets", 1, [arm], variant="biped", world="terrain_methods")
        slot = slots[0]
        runner = ContactRunner(model, slots, [cfg], [controller])
        runner.reset(np.random.default_rng(seed+1000000))
        owners = np.array([0 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "").startswith(slot.prefix) else -1 for g in range(model.ngeom)])
        runner.configure_contacts(owners)
        sensors = TerrainSensors(runner, slot, owners)
        mapping = UncertainElevationMap(MapSettings(**p["map_settings"]))
        truth_height, ground_mask, truth_hazard = evaluator_ground_grid(model, runner.d, mapping.settings)
        np.savez_compressed(output/"evaluator_ground.npz", height_m=truth_height, ground_mask=ground_mask,
                            hazard=truth_hazard, x_m=mapping.x, y_m=mapping.y)
        retained_world = ET.fromstring(world)
        for element in retained_world.iter():
            if element.get("file") == str(cache/"course/ramp_height.png"):
                element.set("file", "ramp_height.png")
        (output/"world.xml").write_text(ET.tostring(retained_world, encoding="unicode"))
        if (cache/"course/ramp_height.png").exists():
            shutil.copyfile(cache/"course/ramp_height.png", output/"ramp_height.png")
        known_start = np.eye(4)
        known_start[:3, 3] = model.qpos0[slot.qpos_adr:slot.qpos_adr+3]
        registration = None
        delivery = DelayedNativeState()
        command = np.zeros(3)
        dt = float(cfg.policy_dt)
        traces, records, map_scores = [], [], []
        goals = fell = collision = nonfinite = False
        first_goal = None
        initial_z = float(runner.d.qpos[slot.qpos_adr+2])
        estimated_goal_latched = False
        with NativeLioClient(runtime, sensors.t_imu_lidar, output/"native", max_frames=int(seconds*5)+10) as client:
            for step in range(int(round(seconds/dt))):
                now = float(runner.d.time)
                arrived = delivery.advance(step)
                latest = delivery.available
                if arrived and latest["tracked"] and registration is None:
                    registration = known_start@sensors.t_body_imu@np.linalg.inv(transform(latest["T_W_I"]))
                if arrived and latest["tracked"] and registration is not None and latest.get("map_reset_id") == 0:
                    t_world_body = registration@transform(latest["T_W_I"])@np.linalg.inv(sensors.t_body_imu)
                    # Body-point packet shares the exact native response capture
                    # timestamp. Native pose changes become available causally.
                    points_l = delivery.raw-sensors.t_body_lidar[:3, 3]
                    mapping.update(points_l, t_world_body@sensors.t_body_lidar, latest["timestamp_s"])
                    snapshot = mapping.snapshot(now)
                    map_scores.append({"time_s": now, **ground_map_metrics(snapshot, truth_height, ground_mask, truth_hazard=truth_hazard)})
                if delivery.pending is None:
                    packet = None if arrived else sensors.newest()
                    if packet is not None:
                        scan, imu, dropped = packet
                        started = time.monotonic()
                        computed = client.track(scan, imu)
                        elapsed = time.monotonic()-started
                        records.append(computed)
                        delivery.stage(computed, scan["points_xyz_m"]+sensors.t_body_lidar[:3, 3], step, elapsed, dt)
                # Query only the already delivered response. Computing a newer
                # response must never leak its pose or tracking state early.
                command, diagnostic, pose = path_command(latest, registration, sensors.t_body_imu,
                                                          delivery.raw, now, arm, ControlSettings(**p["control_settings"]))
                if diagnostic["reason"] == "estimated_goal_stop":
                    estimated_goal_latched = True
                if estimated_goal_latched:
                    command = np.zeros(3)
                    diagnostic["reason"] = "estimated_goal_latched_stop"
                if now < p["initialization_seconds"]:
                    command = np.zeros(3)
                    diagnostic["reason"] = "declared_stationary_initialization"
                target = controller.update(runner.observe(0, command))
                if not np.isfinite(target).all():
                    nonfinite = True
                    break
                runner.step([target])
                nonfinite = not np.isfinite(runner.d.qpos).all() or not np.isfinite(runner.d.qvel).all()
                if nonfinite:
                    break
                position = runner.d.qpos[slot.qpos_adr:slot.qpos_adr+3].copy()
                tilt = runner.tilt(0)
                fell = tilt >= .78 or position[2] < initial_z-.25
                collision = collision or runner.hit_wall
                distance = float(np.linalg.norm(position[:2]-np.array([5., 0.])))
                if distance <= .30 and not goals:
                    goals = True
                    first_goal = float(runner.d.time)
                traces.append({"time_s": float(runner.d.time), "command": command.tolist(), "diagnostic": diagnostic,
                    "evaluator_xyz_m": position.tolist(), "evaluator_quaternion_wxyz": runner.d.qpos[slot.qpos_adr+3:slot.qpos_adr+7].tolist(),
                    "evaluator_goal_distance_m": distance, "evaluator_tilt_rad": tilt, "evaluator_wall_contact": bool(runner.hit_wall),
                    "evaluator_center_clearance_m": float(.75-abs(position[1])),
                    "native_available_at_s": delivery.available_at_s, "native_timestamp_s": latest["timestamp_s"] if latest else None,
                    "native_tracked": bool(latest and latest["tracked"]), "estimated_body_pose": pose.tolist() if pose is not None else None})
                if fell or collision:
                    break
        sensors.save(output/"inference_sensors.npz")
        snapshot = mapping.snapshot(float(runner.d.time))
        np.savez_compressed(output/"final_elevation_map.npz", **{k: v for k, v in snapshot.items() if isinstance(v, np.ndarray)})
        elapsed = float(runner.d.time)
        full = elapsed >= seconds-1e-6
        latencies = [r["wall_seconds"] for r in delivery.history]
        result = {"schema": "bhl-terrain-methods-episode-v1", "status": "PASS", "actor": actor["name"],
            "checkpoint_sha256": actor["checkpoint_sha256"], "terrain": terrain, "seed": seed, "arm": arm,
            "success": bool(goals and full and not (fell or collision or nonfinite)), "goal_reached_evaluator": goals,
            "first_goal_time_s": first_goal, "fell": bool(fell), "collision": bool(collision), "nonfinite": bool(nonfinite),
            "elapsed_simulation_s": elapsed, "full_horizon": full, "required_horizon_s": seconds,
            "lateral_path_rmse_m": float(np.sqrt(np.mean([r["evaluator_xyz_m"][1]**2 for r in traces]))) if traces else None,
            "minimum_center_wall_clearance_m": min((r["evaluator_center_clearance_m"] for r in traces), default=None),
            "native_frames": len(records), "native_tracked_frames": sum(r["tracked"] for r in records),
            "native_client_wall_p95_ms": float(np.quantile(latencies, .95)*1000) if latencies else None,
            "response_deliveries": delivery.history, "responses_still_pending_at_horizon": int(delivery.pending is not None),
            "native_latency_scope": "measured client wall charged to40ms gait updates; ray generation and diagnostic elevation mapping excluded; no end-to-end realtime claim",
            "initial_registration": registration.tolist() if registration is not None else None,
            "known_start_pose": known_start.tolist(), "T_body_imu": sensors.t_body_imu.tolist(), "T_body_lidar": sensors.t_body_lidar.tolist(),
            "geometry_evaluator": geometry, "trace": traces, "ground_map_metrics": map_scores,
            "final_ground_map_metrics": ground_map_metrics(snapshot, truth_height, ground_mask, truth_hazard=truth_hazard),
            "ground_truth_inputs_to_controller": [], "scope": SCOPE,
            "files_sha256": {f.name: sha256(f) for f in output.iterdir() if f.is_file()}}
        write_json(output/"episode.json", result)
        return result


def summarize(rows, phase):
    expected = {(r["actor"], r["terrain"], r["seed"], r["arm"]) for r in rows}
    if len(expected) != len(rows):
        raise ValueError("duplicate terrain experiment cell")
    arms = {}
    for arm in ARMS:
        subset = [r for r in rows if r["arm"] == arm]
        valid_lateral = [r["lateral_path_rmse_m"] for r in subset if r["lateral_path_rmse_m"] is not None]
        arms[arm] = {"episodes": len(subset), "goals": sum(r["success"] for r in subset),
                     "falls": sum(r["fell"] for r in subset), "contacts": sum(r["collision"] for r in subset),
                     "nonfinite": sum(r["nonfinite"] for r in subset),
                     "lateral_path_rmse_m_mean": float(np.mean(valid_lateral)) if valid_lateral else None}
    paired = {}
    baselines = {(r["actor"], r["terrain"], r["seed"]): r for r in rows if r["arm"] == "baseline"}
    for arm in ARMS[1:]:
        pairs = [(baselines[(r["actor"], r["terrain"], r["seed"])], r) for r in rows if r["arm"] == arm and (r["actor"], r["terrain"], r["seed"]) in baselines]
        paired[arm] = {"matched_pairs": len(pairs), "goal_gain_count": sum(int(b["success"])-int(a["success"]) for a, b in pairs),
                       "contact_reduction_count": sum(int(a["collision"])-int(b["collision"]) for a, b in pairs),
                       "uncertainty_scope": "descriptive paired development counts; train-seed and course clustering prevents independent episode safety interpretation"}
    return {"status": "PASS", "scientific_status": "SMOKE_ONLY" if phase == "smoke" else "MEASURED_DEVELOPMENT_NO_CONFIRMATION",
            "phase": phase, "episodes": len(rows), "arms": arms, "paired_vs_baseline": paired,
            "confirmation_executed": False, "scope": SCOPE, "ground_truth_inputs_to_controller": []}



def collect_campaign(protocol_path, roots, output):
    """Require all270 declared identities and independently rescore goal/contact traces."""
    outer = json.loads(protocol_path.read_text())
    p = outer["terrain_methods"]
    expected = {(a["name"], t, seed, arm) for a in p["actors"] for t in TERRAINS
                for seed in DEVELOPMENT_SEEDS for arm in ARMS}
    actor_hash = {a["name"]: a["checkpoint_sha256"] for a in p["actors"]}
    rows, locations, seen = [], [], set()
    protocol_hash = sha256(protocol_path)
    for root in roots:
        for receipt_path in sorted(root.rglob("campaign_result.json")):
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("phase") != "development":
                continue
            if receipt.get("protocol_sha256") != protocol_hash or receipt.get("status") != "PASS":
                raise ValueError("incomplete or mismatched terrain shard")
            for name in receipt["episode_files"]:
                path = (receipt_path.parent/name).resolve()
                if not path.is_relative_to(receipt_path.parent.resolve()):
                    raise ValueError("unsafe episode path")
                row = json.loads(path.read_text())
                identity = (row["actor"], row["terrain"], row["seed"], row["arm"])
                if identity not in expected or identity in seen:
                    raise ValueError("unexpected or duplicate declared terrain cell")
                if row["checkpoint_sha256"] != actor_hash[row["actor"]] or row["required_horizon_s"] != 40.:
                    raise ValueError("actor/horizon differs from frozen design")
                seen.add(identity)
                for member, digest in row["files_sha256"].items():
                    item = (path.parent/member).resolve()
                    if not item.is_relative_to(path.parent) or sha256(item) != digest:
                        raise ValueError("retained scientific payload changed")
                trace = row["trace"]
                reached = any(float(r["evaluator_goal_distance_m"]) <= .30 for r in trace)
                contacts = any(r["evaluator_wall_contact"] for r in trace)
                full = row["elapsed_simulation_s"] >= 40.-1e-6
                clean = reached and full and not (row["fell"] or contacts or row["nonfinite"])
                if (bool(row["goal_reached_evaluator"]) != reached or bool(row["collision"]) != contacts or
                        bool(row["success"]) != clean or bool(row["full_horizon"]) != full):
                    raise ValueError("reported episode outcome does not match retained evaluator trace")
                rows.append(row)
                locations.append({"path": str(path), "sha256": sha256(path)})
    if seen != expected:
        raise ValueError(f"terrain campaign incomplete: {len(seen)}/270 cells; no partial-cohort improvement claim")
    result = summarize(rows, "development")
    result.update(protocol_sha256=protocol_hash, complete_declared_cells=270, episode_sources=locations)
    by_identity = {(r["actor"], r["terrain"], r["seed"], r["arm"]): r for r in rows}
    rng = np.random.default_rng(900001)
    for arm in ARMS[1:]:
        # Resample the10 reset-seed clusters shared across all three actors and
        # terrain types. This interval is conditional on these fixed actors and
        # course families, not a population-level robot reliability interval.
        cluster_differences = []
        for seed in DEVELOPMENT_SEEDS:
            differences = []
            for actor in p["actors"]:
                for terrain in TERRAINS:
                    base = by_identity[(actor["name"], terrain, seed, "baseline")]
                    candidate = by_identity[(actor["name"], terrain, seed, arm)]
                    differences.append(int(candidate["success"])-int(base["success"]))
            cluster_differences.append(float(np.mean(differences)))
        samples = np.asarray(cluster_differences)[rng.integers(0, len(DEVELOPMENT_SEEDS), size=(10000, len(DEVELOPMENT_SEEDS)))].mean(axis=1)
        result["paired_vs_baseline"][arm].update(
            goal_rate_difference=float(np.mean(cluster_differences)),
            goal_rate_difference_95_percent_seed_cluster_bootstrap_interval=np.quantile(samples, [.025, .975]).tolist(),
            bootstrap_seed=900001, bootstrap_resamples=10000,
            interval_scope="conditional on three fixed actors and three fixed terrain families;10 reset-seed clusters, no physical/generalized safety interval")
    result["qualified_for_separately_preregistered_confirmation"] = []
    for actor in p["actors"]:
        for arm in ARMS:
            selected = [r for r in rows if r["actor"] == actor["name"] and r["arm"] == arm]
            if len(selected) == 30 and sum(r["success"] for r in selected) >= 27 and not any(r["fell"] or r["collision"] or r["nonfinite"] for r in selected):
                result["qualified_for_separately_preregistered_confirmation"].append({"actor": actor["name"], "arm": arm})
    write_json(output, result)
    print(json.dumps({k: result[k] for k in ("episodes", "arms", "paired_vs_baseline", "qualified_for_separately_preregistered_confirmation")}))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--create-protocol", action="store_true")
    p.add_argument("--collect", type=Path, nargs="+")
    p.add_argument("--template", type=Path)
    p.add_argument("--runtime-archive", type=Path)
    p.add_argument("--protocol", type=Path)
    p.add_argument("--phase", choices=("smoke", "development"))
    p.add_argument("--actor")
    p.add_argument("--terrain", choices=TERRAINS)
    p.add_argument("--output", type=Path, default=Path(os.environ.get("H34_OUTPUT_DIR", "terrain-methods-output")))
    a = p.parse_args(argv)
    if a.create_protocol:
        if not a.template or not a.runtime_archive:
            p.error("protocol creation requires --template and --runtime-archive")
        create_protocol(a)
        return 0
    if a.collect:
        if not a.protocol:
            p.error("collection requires --protocol")
        return collect_campaign(a.protocol, a.collect, a.output)
    if not a.protocol or not a.phase:
        p.error("run requires --protocol and --phase")
    scientific = validate_protocol(a.protocol)
    a.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write_json(a.output/"receipt.json", {"protocol_sha256": sha256(a.protocol), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_step_id": os.getenv("SLURM_STEP_ID"), "host": platform.node(), "phase": a.phase,
        "input_sha256": scientific["input_inventory"], "created_utc": datetime.now(timezone.utc).isoformat()})
    spec = importlib.util.spec_from_file_location("terrain_methods_launcher", ROOT/"scripts/bench/h34_campaign.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    with tempfile.TemporaryDirectory(prefix="bhl-terrain-methods-native-") as directory:
        launcher.safe_extract(scientific["runtime_archive"], Path(directory))
        runtime = Path(directory)/"runtime"
        receipt = json.loads((runtime/"runtime.json").read_text())
        if sha256(runtime/"fastlio_headless") != receipt["binary_sha256"]:
            raise ValueError("native binary checksum differs")
        (runtime/"fastlio_headless").chmod(0o700)
        actors = scientific["actors"]
        if a.actor:
            actors = [actor for actor in actors if actor["name"] == a.actor]
            if not actors:
                raise ValueError("undeclared actor")
        terrains = (a.terrain,) if a.terrain else TERRAINS
        seeds = DEVELOPMENT_SEEDS
        seconds = scientific["seconds"]
        if a.phase == "smoke":
            actors, terrains, seeds, seconds = actors[:1], ("flat",), (400000,), 8.
        rows, paths = [], []
        for actor in actors:
            for terrain in terrains:
                for seed in seeds:
                    for arm in map(str, np.random.default_rng(seed+500000).permutation(ARMS)):
                        stem = f"{actor['name']}--{terrain}--s{seed}--{arm}"
                        row = run_episode(scientific, actor, runtime, terrain, seed, arm, a.output/stem, seconds)
                        rows.append(row)
                        paths.append(stem+"/episode.json")
                        print(json.dumps({k: row[k] for k in ("actor", "terrain", "seed", "arm", "success", "fell", "collision", "native_tracked_frames")}), flush=True)
        validate_protocol(a.protocol)
        result = summarize(rows, a.phase)
        result.update(protocol_sha256=sha256(a.protocol), episode_files=paths, wall_seconds=time.monotonic()-started,
                      expected_episodes=len(actors)*len(terrains)*len(seeds)*len(ARMS),
                      actor_filter=a.actor, terrain_filter=a.terrain)
        if a.phase == "smoke" and not all(r["native_tracked_frames"] > 0 and not r["nonfinite"] and r["full_horizon"] for r in rows):
            result["status"] = "INCOMPLETE"
        write_json(a.output/"campaign_result.json", result)
        print(json.dumps(result), flush=True)
        return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
