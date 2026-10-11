"""Independently recompute the frozen three-seed H4 outcome from raw gates.

Accept extracted cell output directories or durable launch directories holding
outputs.tar.gz. Every declared seed must be complete and bound to the same
protocol and parent before >=2/3 joint qualifications can close the campaign.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

from h34_campaign import safe_extract
from h4_qualification import judge_records
from h4_train import PARENT_SHA256, TASK, check_protocol


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {value}")))


def check_raw_records(v2, v2x, rows, cell, deploy):
    if v2.get("variant") != "humanoid" or v2x.get("variant") != "humanoid":
        raise ValueError("gates must use the original 22-DoF humanoid variant")
    if v2.get("deploy") != v2x.get("deploy"):
        raise ValueError("v2 and v2x refer to different policies")
    recorded_deploy = Path(v2["deploy"])
    if recorded_deploy.parts[-2:] != ("exported", "deploy.yaml"):
        raise ValueError("gate deploy does not refer to this experiment's export")
    if deploy.get("policy_checkpoint_path") != str(recorded_deploy.with_name("policy.onnx")):
        raise ValueError("gate/deploy policy reference mismatch")
    expected_rule = {"turn_min_deg": 150.0, "drift_max_deg": 15.0, "seconds": 6.0,
                     "turn_seeds": [0, 1, 2], "turn_wz": 0.6, "turn_warm_s": 3.0,
                     "walk_cmd": [0.35, 0.0, 0.0], "walk_warm_s": 1.0, "walk_seed": 0}
    if v2.get("rule") != expected_rule:
        raise ValueError("v2 rule differs from the unchanged instrument")
    trials = v2["turns"] + [v2["walk"]] + v2x["turns"] + v2x["walks"]
    for trial in trials:
        for field in ("yaw_deg", "displacement_m", "seconds", "warm_s"):
            if isinstance(trial[field], bool) or not math.isfinite(float(trial[field])):
                raise ValueError(f"non-finite/non-numeric turn telemetry: {field}")
        if trial["displacement_m"] < 0:
            raise ValueError("negative displacement")
        fall = trial["fell_at_s"]
        if fall is not None and (isinstance(fall, bool) or not math.isfinite(float(fall)) or not 0 <= fall < trial["warm_s"] + trial["seconds"]):
            raise ValueError("invalid fall time")
    for row in rows:
        if row.get("label") != cell["run_name"]:
            raise ValueError("push CSV belongs to a different training cell")
        survival = float(row["survival_s"])
        if not math.isfinite(survival) or not 0 <= survival <= 12:
            raise ValueError("invalid push episode duration")
        fell = str(row["fell"]).lower() == "true"
        if not fell and abs(survival - 12.0) > 1e-9:
            raise ValueError("non-fallen push trial is truncated")
        if fell and survival >= 12:
            raise ValueError("fallen trial did not record an actual fall time")
        if not math.isclose(survival / .04, round(survival / .04), abs_tol=1e-6):
            raise ValueError("push duration is not on the unchanged policy clock")
        applied, survived = int(row["pushes_applied"]), int(row["pushes_survived"])
        expected_pushes = sum(t <= survival + 1e-9 for t in (3.0, 6.0, 9.0))
        if applied != expected_pushes or not 0 <= survived <= applied or (not fell and survived != applied):
            raise ValueError("push intervention delivery is inconsistent with episode duration")
        if float(row["terrain_difficulty"]) != 0:
            raise ValueError("push gate changed from flat terrain")
    return judge_records(copy.deepcopy(v2), copy.deepcopy(v2x), rows)


def check_cell(root, protocol, protocol_sha):
    import torch
    import yaml
    from bhl_robust.eval.heading_observable import FEATURE_CONTRACT, HEADING_KEY
    from bhl_robust.eval.gait_clock import load_env_yaml

    root = Path(root)
    receipt = read_json(root / "campaign_result.json")
    if receipt.get("phase") != "run":
        raise ValueError("smoke artifacts cannot count as scored seeds")
    if receipt.get("status") not in {"PASS", "NEGATIVE"}:
        raise ValueError("runner cell is incomplete")
    seed = receipt.get("seed")
    if isinstance(seed, bool) or seed not in (0, 1, 2):
        raise ValueError("training seed outside the declared cohort")
    index = receipt.get("cell")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 3:
        raise ValueError("missing/invalid protocol cell identity")
    cell = protocol["cells"][index]
    check_protocol(protocol, cell, "run")
    if seed != cell["seed"] or receipt.get("arm") != "R1HO" or receipt.get("task") != TASK:
        raise ValueError("receipt and declared training cell disagree")
    if receipt.get("protocol_sha256") != protocol_sha:
        raise ValueError("training receipt belongs to a different frozen protocol")
    if receipt.get("parent_sha256") != PARENT_SHA256:
        raise ValueError("training receipt belongs to a different parent checkpoint")
    if receipt.get("iterations_completed") != 1000 or receipt.get("final_iteration") != 999:
        raise ValueError("training did not reach the fixed final checkpoint")
    checkpoint = root / "model_999.pt"
    policy = root / "exported/policy.onnx"
    deploy_path = root / "exported/deploy.yaml"
    for path, key in ((checkpoint, "final_checkpoint_sha256"), (policy, "policy_sha256"), (deploy_path, "deploy_sha256")):
        if not path.is_file() or digest(path) != receipt.get(key):
            raise ValueError(f"{key} mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("iter") != 999:
        raise ValueError("checkpoint's saved iteration differs")
    tensors = state["model_state_dict"]
    if tuple(tensors["actor.0.weight"].shape) != (256, 79) or tuple(tensors["critic.0.weight"].shape) != (256, 82):
        raise ValueError("checkpoint actor/critic widths differ")
    if tuple(tensors["actor.6.weight"].shape) != (22, 128):
        raise ValueError("checkpoint action width differs from the 22-DoF model")
    if any(not torch.isfinite(value).all().item() for value in tensors.values()):
        raise ValueError("non-finite trained policy weights")
    for path in (root / "params/agent.yaml", root / "params/env.yaml"):
        if load_env_yaml(path).get("seed") != seed:
            raise ValueError("saved training configuration seed differs")
    guard = read_json(root / "training_guard.json")
    expected_guard = {"status": "PASS", "parent_sha256": PARENT_SHA256, "fresh_optimizer": True,
                      "starting_iteration": 0, "actor_width": 79, "critic_width": 82, "actions": 22,
                      "num_envs": 4096, "additional_iterations": 1000, "feature_contract": FEATURE_CONTRACT}
    if any(guard.get(key) != value for key, value in expected_guard.items()) or guard.get("recipe", {}).get("status") != "PASS":
        raise ValueError("training recipe/warm-start guard failed")
    export = read_json(root / "export_check.json")
    if export.get("status") != "PASS" or export.get("input_shape") != [256, 79] or export.get("output_shape") != [256, 22]:
        raise ValueError("export dimensions/parity guard failed")
    deploy = yaml.safe_load(deploy_path.read_text())
    if (deploy.get("num_observations"), deploy.get("num_actions"), deploy.get("num_joints"), deploy.get("history_length")) != (79, 22, 22, 0):
        raise ValueError("deploy widths differ from the declared model")
    if deploy.get(HEADING_KEY) != FEATURE_CONTRACT or receipt.get("feature_contract") != FEATURE_CONTRACT:
        raise ValueError("heading feature contract differs")
    v2_path, v2x_path, push_path = [root / "gates" / name for name in ("turn-v2.json", "turn-v2x.json", "push-60.csv")]
    v2, v2x = read_json(v2_path), read_json(v2x_path)
    with push_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    judged = check_raw_records(v2, v2x, rows, cell, deploy)
    instruments = {name: digest(REPO / name) for name in ("scripts/bench/turn_test.py", "src/bhl_robust/eval/run_eval.py", "src/bhl_robust/eval/harness.py")}
    if receipt.get("gates", {}).get("instrument_sha256") != instruments:
        raise ValueError("original evaluator source differs from the recorded instrument")
    return {"seed": seed, "cell": index, "run_name": cell["run_name"], "status": "PASS" if judged["joint_pass"] else "NEGATIVE",
            "reported_status": receipt["status"], "status_disagreement": receipt["status"] != ("PASS" if judged["joint_pass"] else "NEGATIVE"),
            **judged, "files": {str(path.relative_to(root)): digest(path) for path in
                                   (checkpoint, policy, deploy_path, v2_path, v2x_path, push_path)}}


def summarize(protocol_path, cell_directories):
    protocol_path = Path(protocol_path)
    protocol, protocol_sha = read_json(protocol_path), digest(protocol_path)
    for cell in protocol["cells"]:
        check_protocol(protocol, cell, "run")
    reports, problems, seen, checkpoint_hashes = [], [], set(), set()
    if len(cell_directories) != 3:
        problems.append(f"requires exactly three complete scored cell bundles; received {len(cell_directories)}")
    for directory in cell_directories:
        path = Path(directory)
        try:
            with tempfile.TemporaryDirectory(prefix="bhl-h4-summary-", dir="/tmp") as scratch:
                if (path / "outputs.tar.gz").is_file():
                    completion = read_json(path / "completion.json")
                    declared = completion.get("files", {}).get("outputs.tar.gz", {})
                    if declared.get("sha256") != digest(path / "outputs.tar.gz"):
                        raise ValueError("durable output archive checksum mismatch")
                    safe_extract(path / "outputs.tar.gz", scratch)
                    root = Path(scratch)
                else:
                    root = path
                report = check_cell(root, protocol, protocol_sha)
            if report["seed"] in seen:
                raise ValueError(f"duplicate training seed {report['seed']}")
            checkpoint_hash = report["files"]["model_999.pt"]
            if checkpoint_hash in checkpoint_hashes:
                raise ValueError("duplicate final checkpoint across purported independent seeds")
            seen.add(report["seed"])
            checkpoint_hashes.add(checkpoint_hash)
            report["bundle"] = str(path)
            reports.append(report)
        except Exception as exc:
            problems.append(f"{path}: {type(exc).__name__}: {exc}")
    if seen != {0, 1, 2}:
        problems.append(f"missing complete training seeds: {sorted({0, 1, 2} - seen)}")
    passed = sum(report["joint_pass"] for report in reports)
    status = "INCOMPLETE" if problems else "PASS" if passed >= 2 else "NEGATIVE"
    return {"schema": "h4-three-seed-independent-summary-v1", "task": "H4", "recipe": "R1HO",
            "status": status, "protocol_sha256": protocol_sha, "parent_sha256": PARENT_SHA256,
            "complete_seeds": len(reports), "joint_pass_seeds": passed, "required_joint_pass_seeds": 2,
            "seed_count": 3, "seeds": sorted(reports, key=lambda row: row["seed"]), "problems": problems,
            "scope": "22-DoF learned policies with simulated yaw; three independently seeded fine-tunes of the same selected R1H-s2 parent. All three final checkpoint bundles and original raw gates required; no smoke, hardware, or from-scratch claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cell-output", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.protocol, args.cell_output)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "complete_seeds", "joint_pass_seeds", "problems")}, indent=2))
    return 1 if report["status"] == "INCOMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
