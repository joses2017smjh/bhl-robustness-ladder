"""Independent post-run H1 receipt, episode-contract and gate verification.

Does not execute or alter the frozen campaign. Use after extracting its input
archive and all raw records, with the protocol hash from submission.json.
This supplements the original frozen summarizer; it is not a changed gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ACTORS = ("armV5-s8", "armV5-s9", "armV5-s10")
SEEDS = tuple(range(120000, 120048))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    return True


def number(value, low=0., high=float("inf")):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_cells():
    return [{"name": f"{actor}-{mode}-{condition}", "actor": actor, "mode": mode,
             "condition": condition, "sensor_mode": "reactive" if condition == "nominal" else "reactive_dropout"}
            for actor in ACTORS for mode in ("none", "wz-lpf") for condition in ("nominal", "drop35")]


def verify_episode(e, seed, cell, p, recorded_root):
    require(finite_tree(e), "non-finite telemetry")
    require(type(e["seed"]) is int and e["seed"] == seed, "wrong reset/layout seed")
    require(e["gait"]["checkpoint_sha256"] == p["source_gait_sha256"], "wrong gait")
    require(e["gait"]["checkpoint"] == "models/gait.onnx", "wrong gait model path")
    require(e["gait"]["deploy"] == str(recorded_root / "models/deploy.yaml"), "wrong gait deploy path")
    require(e["policy"] == str(recorded_root / "models" / f"{cell['actor']}.onnx"), "wrong navigation actor")
    require(e["sensor_mode"] == cell["sensor_mode"], "wrong sensing condition")
    maze = e["maze"]
    require((maze["n"], maze["m"], maze["extra_openings"]) == (6, 6, 1), "wrong maze geometry")
    require(isinstance(maze["layout_sha256"], str) and len(maze["layout_sha256"]) == 16 and
            all(c in "0123456789abcdef" for c in maze["layout_sha256"]), "invalid maze hash")
    require(e["policy_map_integration"]["mode"] == "capture_pose", "wrong map integration")
    for key in ("updates_at_capture_pose", "updates_at_loop_top_pose"):
        require(type(e["policy_map_integration"][key]) is int and e["policy_map_integration"][key] >= 0,
                "invalid capture-pose counters")
    require(e["policy_map_integration"]["updates_at_loop_top_pose"] == 0, "capture-pose fallback occurred")
    filt = e["cmd_filter"]
    require(filt["mode"] == cell["mode"], "wrong filter mode")
    require(filt["tau_s"] == (.2 if cell["mode"] == "wz-lpf" else None), "wrong filter tau")
    if cell["mode"] == "wz-lpf":
        require(filt["channels"] == [2] and filt["alpha"] == .2, "wrong filter channel/rate")
    for field in ("actor_wz", "gait_wz"):
        stats = filt[field]
        require(type(stats["steps"]) is int and 0 <= stats["steps"] <= 4500, "invalid command step count")
        require(number(stats["flips_per_s"], 0., 25.001) and number(stats["saturated_share"], 0., 1.) and
                number(stats["mean_abs_wz"], 0., 1.001), "invalid command statistics")
    require(filt["actor_wz"]["steps"] == filt["gait_wz"]["steps"], "actor/gait command count mismatch")
    require(number(filt["max_tilt_rad"], 0., math.pi), "invalid peak tilt")
    require(type(e["success"]) is bool and type(e["clean_success"]) is bool, "goal flags must be booleans")
    require(e["outcome"] in ("goal", "fall", "time_out"), "unknown outcome")
    require(e["success"] == (e["outcome"] == "goal"), "goal flag/outcome disagree")
    require(type(e["wall_contact_steps"]) is int and e["wall_contact_steps"] >= 0, "invalid contact count")
    require(e["clean_success"] == (e["success"] and e["wall_contact_steps"] == 0), "clean-goal flag disagrees")
    require(number(e["elapsed_s"], 0., 180.) and number(e["path_length_m"]) and number(e["wall_seconds"]),
            "invalid episode duration/path/wall timing")
    if e["success"]:
        require(number(e["completion_s"], 0., 180.) and
                abs(e["completion_s"] - e["elapsed_s"] - .04) <= .011,
                "invalid successful completion time")
    else:
        require(e["completion_s"] is None, "failed episode has a completion time")
    require(e["goal_check"]["judge"]["reached"] is e["success"] and
            e["goal_check"]["judge"]["goal_radius_m"] == .3, "judge disagrees with goal result")
    require(number(e["goal_check"]["judge"]["final_true_dist_m"]) and
            (not e["success"] or e["goal_check"]["judge"]["final_true_dist_m"] <= .3005),
            "successful goal distance disagrees with the geometric judge")
    require(e["imu"]["source"] == "truth", "unexpected estimated-IMU intervention")
    require(all(e["pose_error"][k] == 0. for k in ("pose_bias_m", "pose_noise_m", "pose_yaw_deg")),
            "unexpected pose-error intervention")
    require(e["controller"] == {"cruise": .3, "turn_rate": .6, "inflate_m": .3, "replan_s": .4, "map_res": .1},
            "unexpected brake/controller settings")


def verify_receipt(receipt, proc, cell, protocol_sha, recorded_root):
    require(receipt["protocol_sha256"] == protocol_sha and receipt["cell"] == cell and
            receipt["cwd"] == str(recorded_root) and receipt["scope"] == "confirmation",
            "receipt scope/cell/protocol/cwd mismatch")
    require(type(proc["returncode"]) is int and proc["returncode"] == 0 and proc["inputs_still_valid"] is True,
            "failed process or changed inputs")
    argv = receipt["command"]
    require(isinstance(argv, list) and all(isinstance(s, str) for s in argv), "invalid process argv")
    expected = [str(recorded_root / "snapshot/scripts/bench/maze_explore.py"),
        "--upstream", receipt.get("upstream_expected", ""), "--gait", str(recorded_root / "models/deploy.yaml"),
        "--policy", str(recorded_root / "models" / f"{cell['actor']}.onnx"), "--policy-capture-pose",
        "--policy-cmd-filter", cell["mode"], "--policy-cmd-tau", "0.2",
        "--sensor-mode", cell["sensor_mode"], "--dropout-probability", ".35", "--seeds", "48",
        "--seed-start", "120000", "--n", "6", "--m", "6", "--extra-openings", "1",
        "--time-limit", "180.0", "--out-dir", str(recorded_root / "runs" / cell["name"]), "--no-overwrite"]
    require(len(argv) == len(expected) + 3 and argv[1:-2] == expected and argv[-2] == "--cache-dir" and
            Path(argv[0]).is_absolute() and argv[-1].startswith("/tmp/bhl-nav-confirm-"),
            "process command differs from frozen science argv")


def verify(campaign, expected_protocol_sha):
    require(digest(campaign / "protocol.json") == expected_protocol_sha, "trusted protocol hash mismatch")
    p = json.loads((campaign / "protocol.json").read_text())
    require(p["schema"] == "bhl-h1-confirmation-v1" and p["seeds"] == list(SEEDS) and p["cells"] == expected_cells(),
            "confirmation cohort/cell declaration changed")
    require(p["max_episodes"] == 576 and p["max_concurrent_evaluators"] == 2 and
            p["tau_s"] == .2 and p["time_limit_s"] == 180., "confirmation resource/science cap changed")
    for name, expected in p["input_sha256"].items():
        f = Path(name) if name.startswith("/") else campaign / name
        require(f.is_file() and digest(f) == expected, f"frozen input mismatch: {name}")
    for actor in ACTORS:
        require(digest(campaign / "models" / f"{actor}.onnx") == p["source_actors"][actor]["sha256"], "wrong actor bytes")
    require(digest(campaign / "models/gait.onnx") == p["source_gait_sha256"], "wrong gait bytes")
    rows, raw, problems, roots = {}, {}, [], set()
    hashes = {}
    all_layouts = {}
    for cell in p["cells"]:
        d = campaign / "runs" / cell["name"]
        valid = {}
        try:
            receipt = json.loads((d / "receipt.json").read_text())
            proc = json.loads((d / "process.json").read_text())
            recorded_root = Path(receipt["cwd"])
            require(recorded_root.is_absolute(), "recorded campaign root is relative")
            roots.add(str(recorded_root))
            receipt["upstream_expected"] = p["upstream"]  # verifier-owned expected value
            verify_receipt(receipt, proc, cell, expected_protocol_sha, recorded_root)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"{cell['name']} receipt: {exc}")
            recorded_root = Path("/invalid-receipt-root")
        observed = {f.name for f in d.glob("seed*.json")}
        require_files = {f"seed{seed}.json" for seed in SEEDS}
        if observed != require_files:
            problems.append(f"{cell['name']}: episode file set differs from exact48 cohort")
        for seed in SEEDS:
            f = d / f"seed{seed}.json"
            try:
                e = json.loads(f.read_text())
                verify_episode(e, seed, cell, p, recorded_root)
                maze_hash = e["maze"]["layout_sha256"]
                if seed in all_layouts and all_layouts[seed] != maze_hash:
                    raise ValueError("layout differs across actors/arms/conditions")
                all_layouts[seed] = maze_hash
                valid[seed] = e
                hashes[str(f.relative_to(campaign))] = digest(f)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                problems.append(f"{cell['name']} seed{seed}: {exc}")
        raw[cell["name"]] = valid
        rows[cell["name"]] = {"valid_episodes": len(valid), "goals": sum(e["success"] for e in valid.values()),
            "clean_goals": sum(e["clean_success"] for e in valid.values()),
            "falls": sum(e["outcome"] == "fall" for e in valid.values()),
            "wall_contact_steps": sum(e["wall_contact_steps"] for e in valid.values()),
            "median_gait_flips_per_s": statistics.median(e["cmd_filter"]["gait_wz"]["flips_per_s"] for e in valid.values()) if valid else None}
    if len(roots) != 1:
        problems.append("receipts do not share one original campaign root")
    passes = {a: all(rows[f"{a}-wz-lpf-{c}"]["goals"] >= n and rows[f"{a}-wz-lpf-{c}"]["falls"] == 0
                    for c, n in (("nominal", 40), ("drop35", 36))) for a in ACTORS}
    status = "INCOMPLETE" if problems else "PASS" if all(passes.values()) else "NEGATIVE"
    return {"verified_utc": datetime.now(timezone.utc).isoformat(), "status": status, "problems": problems,
        "protocol_sha256": expected_protocol_sha, "verifier_sha256": digest(__file__), "per_cell": rows,
        "per_actor_pass": passes, "valid_episode_records": sum(len(es) for es in raw.values()),
        "raw_episode_sha256": hashes, "recorded_roots": sorted(roots),
        "scope": "Independent post-run evidence contract verification; original576episode cohort and success gates unchanged. No new physics or candidate tuning."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--expected-protocol-sha256", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("output exists; choose a fresh verification receipt")
    try:
        report = verify(a.campaign.resolve(), a.expected_protocol_sha256)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"status": "INCOMPLETE", "problems": [str(exc)], "verifier_sha256": digest(__file__),
                  "scope": "Preflight/evidence verification failed; no performance result established."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "problems": report["problems"]}, indent=2))
    return 1 if report["status"] == "INCOMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
