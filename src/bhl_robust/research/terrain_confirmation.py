"""Frozen confirmation cohort and genuinely new terrain fixtures.

Geometry is simulator/evaluator input only. Learned actors, sensing, estimated
pose, control gains and success definition are inherited without adjustment.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np

from .terrain_methods import TERRAINS
from .terrain_traversal import TraversalSettings, prepare_course

SCHEMA = "bhl-terrain-confirmation-v2"
CONFIRMATION_SEEDS = tuple(range(440000, 440020))
SMOKE_SEED = 450000
QUALIFIED = (("dr-default-s0", "heading"), ("dr-default-s0", "regulated"),
             ("dr-default-s1", "heading"), ("dr-default-s2", "heading"),
             ("dr-default-s2", "regulated"))
SCOPE = ("Independent held-out MuJoCo confirmation of five development-qualified "
         "actor/controller combinations on new procedural geometry and reserved reset seeds; "
         "fixed actors and simulator, not independent hardware or general robot safety validation")


def arms_for_actor(actor):
    arms = tuple(arm for name, arm in QUALIFIED if name == actor)
    if not arms:
        raise ValueError("actor did not qualify in the frozen development cohort")
    return ("baseline",) + arms


def expected_cells(phase):
    if phase not in ("smoke", "confirmation"):
        raise ValueError("undeclared confirmation phase")
    seeds = (SMOKE_SEED,) if phase == "smoke" else CONFIRMATION_SEEDS
    return {(actor, terrain, seed, arm) for actor in sorted({a for a, _ in QUALIFIED})
            for terrain in TERRAINS for seed in seeds for arm in arms_for_actor(actor)}


def geometry_parameters(terrain, seed):
    if terrain not in TERRAINS or seed not in (*CONFIRMATION_SEEDS, SMOKE_SEED):
        raise ValueError("terrain or seed outside reserved confirmation protocol")
    rng = np.random.default_rng(seed + 700000)
    onset = float(rng.uniform(.95, 1.15))
    friction = float(rng.uniform(.78, .82))
    treads = rng.uniform(.29, .41, 6).tolist()
    rise = float(rng.uniform(1.35, 1.65))
    plateau = float(rng.uniform(.25, .45))
    landmarks = [[float(x + rng.uniform(-.10, .10)),
                  float(side * rng.uniform(.92, 1.08)),
                  float(rng.uniform(.80, 1.10))]
                 for x, side in ((.55, -1), (1.50, 1), (2.40, -1),
                                 (3.35, 1), (4.35, -1), (5.35, 1))]
    value = dict(schema="bhl-confirmation-course-v1", terrain=terrain, seed=seed,
                 onset_m=onset, friction=friction, corridor_half_width_m=.75,
                 goal_xy_m=[5., 0.], exterior_landmark_centers_m=landmarks)
    if terrain == "small_steps":
        value.update(step_rise_m=.01, step_levels=[1, 2, 2, 3, 2, 1], treads_m=treads)
    elif terrain == "ramp":
        value.update(grade_deg=3., rise_length_m=rise, plateau_length_m=plateau)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["geometry_sha256"] = hashlib.sha256(encoded).hexdigest()
    return value


def make_world(terrain, seed, cache):
    from PIL import Image
    from bhl_robust.eval.multi_robot import _FLAT_WORLD
    cache = Path(cache)
    base = cache / "base.xml"
    base.write_text(_FLAT_WORLD)
    scene, _ = prepare_course(base, cache / "course", terrain, seed, TraversalSettings())
    root = ET.parse(scene).getroot()
    world, asset = root.find("worldbody"), root.find("asset")
    params = geometry_parameters(terrain, seed)
    for geom in world.findall("geom"):
        name = geom.get("name", "")
        if name.startswith("course_wall_"):
            geom.set("name", "wall_" + name)
        if name.startswith("terrain_"):
            geom.set("friction", f"{params['friction']:.9f} .005 .0001")
        if name.startswith("terrain_step"):
            world.remove(geom)
    if terrain == "small_steps":
        template = world.find("geom[@name='terrain_base']")
        contact = {name: template.get(name) for name in
                   ("group", "friction", "priority", "solref", "solimp", "rgba")}
        x = params["onset_m"]
        for k, (level, width) in enumerate(zip(params["step_levels"], params["treads_m"])):
            height = .01 * level
            ET.SubElement(world, "geom", name=f"terrain_step{k}", type="box",
                          size=f"{width/2:.12f} 1.0 {height/2:.12f}",
                          pos=f"{x+width/2:.12f} 0 {height/2:.12f}", **contact)
            x += width
    elif terrain == "ramp":
        xs = np.linspace(-1., 6., 513)
        onset, rise, plateau = params["onset_m"], params["rise_length_m"], params["plateau_length_m"]
        heights = np.tan(np.deg2rad(3.)) * (np.clip(xs-onset, 0, rise) -
                                            np.clip(xs-onset-rise-plateau, 0, rise))
        heights = np.maximum(heights, 0.)
        scale = float(heights.max())
        pixels = np.tile(np.rint(heights/scale*255).astype(np.uint8), (65, 1))
        Image.fromarray(pixels).save(cache / "course/ramp_height.png")
        asset.find("hfield[@name='terrain_hfield']").set("size", f"3.5 1.0 {scale:.12f} .05")
    for i, (x, y, z) in enumerate(params["exterior_landmark_centers_m"]):
        ET.SubElement(world, "geom", name=f"wall_confirmation_landmark_{i}", type="box",
                      group="0", pos=f"{x:.12f} {y:.12f} {z:.12f}",
                      size=".08 .08 .25", rgba=".2 .4 .7 1")
    return ET.tostring(root, encoding="unicode"), params


def validate_development_summary(summary, protocol_sha256):
    expected = [{"actor": a, "arm": arm} for a, arm in QUALIFIED]
    if (summary.get("status") != "PASS" or summary.get("episodes") != 270 or
            summary.get("complete_declared_cells") != 270 or
            summary.get("protocol_sha256") != protocol_sha256 or
            summary.get("qualified_for_separately_preregistered_confirmation") != expected):
        raise ValueError("complete frozen development qualification evidence required")


def summarize(rows, phase):
    """Compute complete matched-cohort effects, with shared-seed uncertainty."""
    identities = {(r["actor"], r["terrain"], r["seed"], r["arm"]): r for r in rows}
    if len(identities) != len(rows) or set(identities) != expected_cells(phase):
        raise ValueError("confirmation cohort incomplete, duplicate, or undeclared")
    result = dict(status="PASS", phase=phase, episodes=len(rows), scope=SCOPE,
                  scientific_status="SMOKE_ONLY" if phase == "smoke" else "MEASURED_HELD_OUT_SIMULATION_CONFIRMATION",
                  qualified_comparisons=[], confirmation_criteria="at least54/60 clean goals; zero falls, contacts, nonfinite; positive paired gain",
                  uncertainty_scope="20 shared geometry/reset-seed clusters conditional on fixed actors and terrain families; not hardware or population reliability")
    result["rejected_original_scan_count"] = sum(r.get("rejected_original_scan_count", 0) for r in rows)
    if phase == "smoke":
        result["gate_to_confirmation"] = all(r["full_horizon"] and not r["nonfinite"] and
            not r["fell"] and r["native_tracked_frames"] > 0 for r in rows)
        result["status"] = "PASS" if result["gate_to_confirmation"] else "NEGATIVE"
        return result
    for actor, arm in QUALIFIED:
        candidates = [identities[(actor, terrain, seed, arm)]
                      for seed in CONFIRMATION_SEEDS for terrain in TERRAINS]
        baselines = [identities[(actor, terrain, seed, "baseline")]
                     for seed in CONFIRMATION_SEEDS for terrain in TERRAINS]
        differences = np.array([int(b["success"])-int(a["success"])
                                for a, b in zip(baselines, candidates)]).reshape(20, 3).mean(axis=1)
        samples = differences[np.random.default_rng(900011).integers(0, 20, (10000, 20))].mean(axis=1)
        totals = lambda data: {"episodes": len(data), "clean_goals": sum(bool(r["success"]) for r in data),
            "falls": sum(bool(r["fell"]) for r in data), "contacts": sum(bool(r["collision"]) for r in data),
            "nonfinite": sum(bool(r["nonfinite"]) for r in data)}
        candidate, baseline = totals(candidates), totals(baselines)
        result["qualified_comparisons"].append(dict(actor=actor, arm=arm, candidate=candidate,
            matched_baseline=baseline, paired_goal_rate_difference=float(differences.mean()),
            paired_goal_rate_difference_95_percent_seed_cluster_interval=np.quantile(samples, [.025, .975]).tolist(),
            bootstrap_seed=900011, bootstrap_resamples=10000,
            confirmed=bool(candidate["clean_goals"] >= 54 and not any(candidate[k] for k in
                ("falls", "contacts", "nonfinite")) and differences.mean() > 0)))
    result["confirmed_combinations"] = sum(r["confirmed"] for r in result["qualified_comparisons"])
    result["baseline_unique_episodes"] = sum(r["arm"] == "baseline" for r in rows)
    result["candidate_unique_episodes"] = len(rows) - result["baseline_unique_episodes"]
    result["multiple_comparisons"] = "five prespecified engineering gates; descriptive marginal intervals, no family-wise statistical discovery claim"
    return result
