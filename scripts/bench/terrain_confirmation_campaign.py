#!/usr/bin/env python3
"""Preregistered independent terrain confirmation; no development-data reuse."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import terrain_methods_campaign as development
from bhl_robust.research.native_lio import sha256
from bhl_robust.research.terrain_confirmation import (SCHEMA, SCOPE, SMOKE_SEED,
    QUALIFIED, CONFIRMATION_SEEDS, arms_for_actor, expected_cells, geometry_parameters, make_world,
    summarize, validate_development_summary)
from bhl_robust.research.terrain_methods import TERRAINS
from bhl_robust.research.terrain_confirmation_acquisition import validated_sensor_class, scan_time_problem
from native_navigation_campaign import write_json

FILES = ("scripts/bench/terrain_confirmation_campaign.py",
         "src/bhl_robust/research/terrain_confirmation.py",
         "src/bhl_robust/research/terrain_confirmation_acquisition.py")
SUMMARY_INPUT = "confirmation/development-summary.json"


def design():
    return dict(schema=SCHEMA, scope=SCOPE, confirmation_seeds=list(CONFIRMATION_SEEDS),
        smoke_seed=SMOKE_SEED, confirmation_seconds=40., smoke_seconds=12.,
        qualified=[dict(actor=a, arm=arm) for a, arm in QUALIFIED],
        terrains=list(TERRAINS), expected_confirmation_episodes=480, expected_smoke_episodes=24,
        development_summary_input=SUMMARY_INPUT,
        source_files=list(FILES),
        excluded_consumed_confirmation_seeds=list(range(420000, 420020)),
        acquisition_contract="Drop and retain original invalid-time scans, never fabricate points/times; keep all unconsumed IMU; existing350ms pose-age stop unchanged",
        arm_order="RNG(seed+500000) permutation of baseline plus only qualified arms for that actor",
        confirmation_gate="Per qualified actor/controller: >=54/60 clean goals, zero falls/contacts/nonfinite, positive paired clean-goal rate difference",
        smoke_gate="24/24 declared actor/terrain/controller episodes, full12s horizon, finite state, no falls, native tracked frames>0; no gain tuning",
        primary_metric="Matched candidate-minus-baseline clean-goal difference, 20 shared geometry/reset seed clusters",
        uncertainty="10000 percentile bootstrap samples of20 seed clusters, fixed seed900011; intervals conditional on fixed actors/families; five marginal descriptive comparisons",
        no_adaptation="Freeze before smoke; no geometry/gain/gate/seed/actor edits after outcomes; preserve negative outcomes; confirmation failures require new development cohort",
        geometric_parameters={f"{terrain}:{seed}": geometry_parameters(terrain, seed)
                              for terrain in TERRAINS for seed in (*CONFIRMATION_SEEDS, SMOKE_SEED)})


def create_protocol(args):
    original = json.loads(args.template.read_text())
    previous_sha = sha256(args.template)
    summary = json.loads(args.development_summary.read_text())
    validate_development_summary(summary, previous_sha)
    original_science = original["terrain_methods"]
    # Any changed learned actor, sensing, control, or evaluator starts a new
    # development experiment; confirmation must inherit the completed version.
    for name, digest in original_science["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("development scientific source changed before confirmation: " + name)
    original["terrain_methods"]["source_sha256"].update({name: sha256(ROOT/name) for name in FILES})
    original["additional_source"] = sorted(set(original["additional_source"]) | set(FILES))
    original["input_files"][SUMMARY_INPUT] = dict(path=str(args.development_summary.resolve()), sha256=sha256(args.development_summary))
    if args.runtime_archive:
        runtime_name = original_science["runtime_input"]
        if sha256(args.runtime_archive) != original["input_files"][runtime_name]["sha256"]:
            raise ValueError("confirmation native runtime must match development bytes")
        original["input_files"][runtime_name]["path"] = str(args.runtime_archive.resolve())
    original["terrain_confirmation"] = dict(design(), development_protocol_sha256=previous_sha,
        development_summary_sha256=sha256(args.development_summary),
        created_utc=datetime.now(timezone.utc).isoformat())
    original["campaign_id"] = "terrain-confirmation-20261011"
    original["scope"] = SCOPE
    original["execution"] = "fresh same-source24-episode smoke then9shards/480held-out confirmation episodes; no retries or scientific tuning"
    def job(name, phase, more=()):
        return dict(name=name, kind="smoke" if phase == "smoke" else "run",
            entrypoint=FILES[0], args=["--protocol", "{protocol}", "--phase", phase, *more],
            resources=dict(cpus=4, memory_gb=8, gpus=0, partition="share,eecs",
                           time_limit="01:00:00" if phase == "smoke" else "04:00:00", requeue=False),
            env={}, max_output_mb=512 if phase == "smoke" else 1024)
    original["jobs"] = [job("terrain-confirmation-smoke-20261011", "smoke")]
    for actor in original_science["actors"]:
        for terrain in TERRAINS:
            original["jobs"].append(job(f"terrain-confirmation-{actor['name']}-{terrain}-20261011", "confirmation",
                ("--actor", actor["name"], "--terrain", terrain)))
    write_json(args.output, original)
    print(json.dumps(dict(protocol=str(args.output), sha256=sha256(args.output), confirmation_episodes=480, smoke_episodes=24)))


def validate_design(outer):
    actual = dict(outer["terrain_confirmation"])
    for field in ("development_protocol_sha256", "development_summary_sha256", "created_utc"):
        actual.pop(field)
    if actual != design():
        raise ValueError("changed predeclared independent confirmation design")
    p = outer["terrain_methods"]
    if {a["name"] for a in p["actors"]} != {a for a, _ in QUALIFIED}:
        raise ValueError("confirmation actor cohort differs")
    if outer["input_files"][SUMMARY_INPUT]["sha256"] != outer["terrain_confirmation"]["development_summary_sha256"]:
        raise ValueError("development qualification input identity differs")


def validate_protocol(path):
    outer = json.loads(Path(path).read_text())
    validate_design(outer)
    p = development.validate_protocol(Path(path))
    relocated = bool(os.getenv("H34_WORK_DIR") or os.getenv("H34_INPUTS_DIR"))
    input_root = Path(os.getenv("H34_INPUTS_DIR", Path(path).parent / "inputs"))
    summary_path = input_root / SUMMARY_INPUT if relocated else Path(outer["input_files"][SUMMARY_INPUT]["path"])
    expected = outer["terrain_confirmation"]["development_summary_sha256"]
    if sha256(summary_path) != expected:
        raise ValueError("frozen development qualification evidence changed")
    validate_development_summary(json.loads(summary_path.read_text()), outer["terrain_confirmation"]["development_protocol_sha256"])
    p["input_inventory"][str(summary_path)] = expected
    return p


@contextmanager
def confirmation_world():
    """Reuse exact development dynamics/controller implementation in isolation."""
    original = development.make_world
    original_sensors = development.TerrainSensors
    development.make_world = make_world
    development.TerrainSensors = validated_sensor_class(original_sensors)
    try:
        yield
    finally:
        development.make_world = original
        development.TerrainSensors = original_sensors


def run_episode(p, actor, runtime, terrain, seed, arm, output, seconds):
    with confirmation_world():
        row = development.run_episode(p, actor, runtime, terrain, seed, arm, output, seconds)
    rejections = json.loads((output / "rejected-scans.json").read_text())
    row.update(schema="bhl-terrain-confirmation-episode-v2", scope=SCOPE,
               rejected_original_scan_count=rejections["rejected_count"],
               independent_confirmation=True, geometry_sha256=geometry_parameters(terrain, seed)["geometry_sha256"])
    row["files_sha256"]["episode.json"] = sha256(output / "episode.json")
    write_json(output / "confirmation-episode.json", row)
    return row


def rescore_episode(row, required_seconds=40.):
    """Recompute success from retained independent simulator traces."""
    if row.get("schema") != "bhl-terrain-confirmation-episode-v2" or row.get("ground_truth_inputs_to_controller") != []:
        raise ValueError("episode is not a declared estimator-only confirmation")
    if row.get("required_horizon_s") != required_seconds:
        raise ValueError("confirmation horizon differs")
    expected_geometry = geometry_parameters(row["terrain"], row["seed"])
    if row.get("geometry_evaluator") != expected_geometry or row.get("geometry_sha256") != expected_geometry["geometry_sha256"]:
        raise ValueError("new frozen course geometry does not match episode")
    trace = row["trace"]
    if not trace:
        if row.get("nonfinite") is not True or row.get("success") is not False:
            raise ValueError("missing evaluator trace without explicit nonfinite failure")
        return False
    xyz = np.asarray([r["evaluator_xyz_m"] for r in trace], dtype=float)
    tilt = np.asarray([r["evaluator_tilt_rad"] for r in trace], dtype=float)
    times = np.asarray([r["time_s"] for r in trace], dtype=float)
    if not (np.isfinite(xyz).all() and np.isfinite(tilt).all() and np.isfinite(times).all() and
            np.all(np.diff(times) > 0) and times[0] > 0):
        raise ValueError("malformed finite retained evaluator trace")
    distances = np.linalg.norm(xyz[:, :2] - np.array([5., 0.]), axis=1)
    if not np.allclose(distances, [r["evaluator_goal_distance_m"] for r in trace], atol=1e-9):
        raise ValueError("reported goal distances differ from independent pose")
    reached = bool(np.any(distances <= .30))
    contact = any(r["evaluator_wall_contact"] for r in trace)
    initial_z = float(row["known_start_pose"][2][3])
    fell = bool(np.any(tilt >= .78) or np.any(xyz[:, 2] < initial_z - .25))
    elapsed = float(row["elapsed_simulation_s"])
    # Nonfinite termination can occur after the last retainable finite state.
    if (not row["nonfinite"] and not np.isclose(times[-1], elapsed, atol=1e-8)) or elapsed > required_seconds + 1e-6:
        raise ValueError("retained horizon does not match elapsed simulation time")
    full = elapsed >= required_seconds - 1e-6
    success = bool(reached and full and not (fell or contact or row["nonfinite"]))
    if (bool(row["goal_reached_evaluator"]) != reached or bool(row["collision"]) != contact or
            bool(row["fell"]) != fell or bool(row["full_horizon"]) != full or bool(row["success"]) != success):
        raise ValueError("reported outcome disagrees with independently rescored trace")
    return success


def collect_campaign(protocol, roots, output):
    outer = json.loads(protocol.read_text())
    validate_design(outer)
    expected, seen, rows, sources = expected_cells("confirmation"), set(), [], []
    actor_hashes = {a["name"]: a["checkpoint_sha256"] for a in outer["terrain_methods"]["actors"]}
    for root in roots:
        for path in sorted(root.rglob("campaign_result.json")):
            receipt = json.loads(path.read_text())
            if receipt.get("phase") != "confirmation":
                continue
            if receipt.get("protocol_sha256") != sha256(protocol) or receipt.get("status") != "PASS":
                raise ValueError("incomplete or mismatched confirmation shard")
            for name in receipt["episode_files"]:
                member = (path.parent / name).resolve()
                if not member.is_relative_to(path.parent.resolve()):
                    raise ValueError("unsafe retained episode path")
                row = json.loads(member.read_text())
                identity = (row["actor"], row["terrain"], row["seed"], row["arm"])
                if identity not in expected or identity in seen:
                    raise ValueError("unexpected or duplicate confirmation episode")
                if row["checkpoint_sha256"] != actor_hashes[row["actor"]]:
                    raise ValueError("confirmation actor checkpoint changed")
                for filename, digest in row["files_sha256"].items():
                    payload = (member.parent / filename).resolve()
                    if not payload.is_relative_to(member.parent) or sha256(payload) != digest:
                        raise ValueError("retained scientific payload changed")
                rejected = json.loads((member.parent / "rejected-scans.json").read_text())
                if (row["rejected_original_scan_count"] != rejected["rejected_count"] or
                        rejected["rejected_count"] != len(rejected["rejected_scans"]) or
                        any(scan_time_problem(r["packet"], r["previous_valid_tail_s"]) != r["reason"] for r in rejected["rejected_scans"])):
                    raise ValueError("invalid original-scan rejection evidence")
                rescore_episode(row)
                seen.add(identity)
                rows.append(row)
                sources.append(dict(path=str(member), sha256=sha256(member)))
    if seen != expected:
        raise ValueError(f"confirmation incomplete: {len(seen)}/480; no partial confirmation claim")
    summary = summarize(rows, "confirmation")
    summary.update(protocol_sha256=sha256(protocol), episode_sources=sources,
                   development_summary_sha256=outer["terrain_confirmation"]["development_summary_sha256"])
    write_json(output, summary)
    print(json.dumps({k: summary[k] for k in ("episodes", "confirmed_combinations", "qualified_comparisons")}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-protocol", action="store_true")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--development-summary", type=Path)
    parser.add_argument("--runtime-archive", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--phase", choices=("smoke", "confirmation"))
    parser.add_argument("--actor")
    parser.add_argument("--terrain", choices=TERRAINS)
    parser.add_argument("--collect", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path(os.getenv("H34_OUTPUT_DIR", "terrain-confirmation-output")))
    args = parser.parse_args(argv)
    if args.create_protocol:
        if not args.template or not args.development_summary:
            parser.error("protocol creation requires frozen development template and complete collected summary")
        create_protocol(args)
        return 0
    if not args.protocol:
        parser.error("--protocol required")
    if args.collect:
        return collect_campaign(args.protocol, args.collect, args.output)
    if not args.phase:
        parser.error("--phase required")
    if args.phase == "smoke" and (args.actor or args.terrain):
        parser.error("smoke must cover all24 declared identities")
    p = validate_protocol(args.protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "receipt.json", dict(protocol_sha256=sha256(args.protocol),
        phase=args.phase, slurm_job_id=os.getenv("SLURM_JOB_ID"), host=platform.node(),
        source_input_sha256=p["input_inventory"], created_utc=datetime.now(timezone.utc).isoformat()))
    actors = [a for a in p["actors"] if not args.actor or a["name"] == args.actor]
    if not actors:
        raise ValueError("undeclared actor")
    terrains = (args.terrain,) if args.terrain else TERRAINS
    seeds = (SMOKE_SEED,) if args.phase == "smoke" else CONFIRMATION_SEEDS
    seconds = 12. if args.phase == "smoke" else 40.
    import h34_campaign
    rows, paths = [], []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="bhl-confirmation-runtime-") as directory:
        h34_campaign.safe_extract(p["runtime_archive"], Path(directory))
        runtime = Path(directory) / "runtime"
        receipt = json.loads((runtime / "runtime.json").read_text())
        if sha256(runtime / "fastlio_headless") != receipt["binary_sha256"]:
            raise ValueError("native runtime binary changed")
        (runtime / "fastlio_headless").chmod(0o700)
        for actor in actors:
            for terrain in terrains:
                for seed in seeds:
                    for arm in map(str, np.random.default_rng(seed+500000).permutation(arms_for_actor(actor["name"]))):
                        stem = f"{actor['name']}--{terrain}--s{seed}--{arm}"
                        row = run_episode(p, actor, runtime, terrain, seed, arm, args.output / stem, seconds)
                        rescore_episode(row, seconds)
                        rows.append(row)
                        paths.append(stem + "/confirmation-episode.json")
                        print(json.dumps({k: row[k] for k in ("actor", "terrain", "seed", "arm", "success", "fell", "collision", "native_tracked_frames")}), flush=True)
    validate_protocol(args.protocol)
    expected = {cell for cell in expected_cells(args.phase) if (not args.actor or cell[0] == args.actor) and
                (not args.terrain or cell[1] == args.terrain)}
    identities = {(r["actor"], r["terrain"], r["seed"], r["arm"]) for r in rows}
    if identities != expected or len(identities) != len(rows):
        raise ValueError("declared shard identities incomplete")
    result = summarize(rows, "smoke") if args.phase == "smoke" else dict(
        status="PASS", phase="confirmation", episodes=len(rows), scientific_status="UNCOLLECTED_CONFIRMATION_SHARD", scope=SCOPE)
    result.update(protocol_sha256=sha256(args.protocol), episode_files=paths,
                  actor_filter=args.actor, terrain_filter=args.terrain, wall_seconds=time.monotonic()-started)
    write_json(args.output / "campaign_result.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
