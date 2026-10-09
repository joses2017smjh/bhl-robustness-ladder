"""Independent campaign verdicts, provenance failures, and raw-gate controls."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest
import torch
import yaml

from bhl_robust.eval.heading_observable import FEATURE_CONTRACT, HEADING_KEY
from bhl_robust.eval.harness import EvalConfig

REPO = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("h4_summary", REPO / "scripts/bench/h4_campaign_summary.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def write_json(path, record):
    path.write_text(json.dumps(record) + "\n")


def gate_records(seed):
    def trial(reset, sign):
        return dict(seed=reset, cmd=[0., 0., .6 * sign], warm_s=3., seconds=6.,
                    fell_at_s=None, yaw_deg=160. * sign, displacement_m=.1)
    def walk(reset):
        return dict(seed=reset, cmd=[.35, 0., 0.], warm_s=1., seconds=6.,
                    fell_at_s=None, yaw_deg=15., displacement_m=2.)
    common = dict(variant="humanoid", deploy=f"/tmp/original-s{seed}/output/exported/deploy.yaml")
    v2 = dict(common, protocol="v2", turns=[trial(s, sign) for s in range(3) for sign in (1, -1)], walk=walk(0),
              rule=dict(turn_min_deg=150., drift_max_deg=15., seconds=6., turn_seeds=[0, 1, 2], turn_wz=.6,
                        turn_warm_s=3., walk_cmd=[.35, 0., 0.], walk_warm_s=1., walk_seed=0))
    v2x = dict(common, protocol="v2x", turns=[trial(s, sign) for s in range(10, 15) for sign in (1, -1)],
               walks=[walk(s) for s in (10, 11, 12)],
               rule=dict(turn_min_deg=150., drift_max_deg=15., seconds=6., turn_seeds=[10, 11, 12, 13, 14],
                         walk_seeds=[10, 11, 12], min_turn_ok=9, min_walk_ok=2, turn_wz=.6,
                         turn_warm_s=3., walk_cmd=[.35, 0., 0.], walk_warm_s=1.))
    rows = [dict(label=f"h4-r1ho-s{seed}", command_vx=c[0], command_vy=c[1], command_wz=c[2], seed=s,
                 fell=False, survival_s=12., pushes_applied=3, pushes_survived=3, terrain_difficulty=0.)
            for c in EvalConfig().commands for s in range(10)]
    return v2, v2x, rows


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def campaign(tmp_path):
    protocol = json.loads((REPO / "results/h34-campaign-20261008/h4/protocol.json").read_text())
    path = tmp_path / "protocol.json"
    write_json(path, protocol)
    cells = []
    for seed in range(3):
        root = tmp_path / f"seed{seed}"
        for name in ("gates", "params", "exported"):
            (root / name).mkdir(parents=True, exist_ok=True)
        # Deliberately synthetic artifact fixtures exercise validation only.
        torch.save({"iter": 999, "model_state_dict": {
            "actor.0.weight": torch.full((256, 79), float(seed)),
            "critic.0.weight": torch.ones(256, 82), "actor.6.weight": torch.ones(22, 128)}}, root / "model_999.pt")
        (root / "exported/policy.onnx").write_bytes(b"synthetic artifact fixture " + bytes([seed]))
        deploy = dict(policy_checkpoint_path=f"/tmp/original-s{seed}/output/exported/policy.onnx",
                      num_observations=79, num_actions=22, num_joints=22, history_length=0,
                      **{HEADING_KEY: FEATURE_CONTRACT})
        (root / "exported/deploy.yaml").write_text(yaml.safe_dump(deploy))
        for name in ("agent", "env"):
            (root / f"params/{name}.yaml").write_text(yaml.safe_dump({"seed": seed}))
        write_json(root / "training_guard.json", dict(status="PASS", parent_sha256=summary.PARENT_SHA256,
            fresh_optimizer=True, starting_iteration=0, actor_width=79, critic_width=82, actions=22,
            num_envs=4096, additional_iterations=1000, feature_contract=FEATURE_CONTRACT, recipe={"status": "PASS"}))
        write_json(root / "export_check.json", dict(status="PASS", input_shape=[256, 79], output_shape=[256, 22]))
        v2, v2x, rows = gate_records(seed)
        write_json(root / "gates/turn-v2.json", v2)
        write_json(root / "gates/turn-v2x.json", v2x)
        write_rows(root / "gates/push-60.csv", rows)
        instruments = {name: summary.digest(REPO / name) for name in
                       ("scripts/bench/turn_test.py", "src/bhl_robust/eval/run_eval.py", "src/bhl_robust/eval/harness.py")}
        write_json(root / "campaign_result.json", dict(status="PASS", phase="run", seed=seed, cell=seed,
            arm="R1HO", task=summary.TASK, protocol_sha256=summary.digest(path), parent_sha256=summary.PARENT_SHA256,
            iterations_completed=1000, final_iteration=999, final_checkpoint_sha256=summary.digest(root / "model_999.pt"),
            policy_sha256=summary.digest(root / "exported/policy.onnx"), deploy_sha256=summary.digest(root / "exported/deploy.yaml"),
            feature_contract=FEATURE_CONTRACT, gates={"instrument_sha256": instruments}))
        cells.append(root)
    return path, cells


def mutate_json(path, mutate):
    record = json.loads(path.read_text())
    mutate(record)
    write_json(path, record)


def test_all_three_bundles_pass_and_two_of_three_boundary(campaign):
    path, cells = campaign
    report = summary.summarize(path, cells)
    assert (report["status"], report["complete_seeds"], report["joint_pass_seeds"]) == ("PASS", 3, 3)
    mutate_json(cells[2] / "gates/turn-v2.json", lambda d: d["walk"].update(yaw_deg=15.1))
    report = summary.summarize(path, cells)
    assert report["status"] == "PASS" and report["joint_pass_seeds"] == 2
    assert report["seeds"][2]["status_disagreement"]


def test_complete_negative_recomputed_despite_claimed_passes(campaign):
    path, cells = campaign
    for cell in cells[:2]:
        mutate_json(cell / "gates/turn-v2x.json", lambda d: (d["turns"][0].update(yaw_deg=149.9), d["turns"][2].update(yaw_deg=149.9)))
    report = summary.summarize(path, cells)
    assert (report["status"], report["complete_seeds"], report["joint_pass_seeds"]) == ("NEGATIVE", 3, 1)
    assert report["seeds"][0]["turn_ok"] == 8
    assert not report["problems"]


def test_two_passes_with_missing_third_seed_remain_incomplete(campaign):
    path, cells = campaign
    report = summary.summarize(path, cells[:2])
    assert report["status"] == "INCOMPLETE" and report["joint_pass_seeds"] == 2


@pytest.mark.parametrize("change", ["smoke", "protocol", "parent", "seed", "cell", "budget", "checkpoint", "export", "saved_seed", "instrument"])
def test_mixed_or_incomplete_provenance_refused(campaign, change):
    path, cells = campaign
    cell = cells[2]
    receipt = cell / "campaign_result.json"
    if change == "smoke":
        mutate_json(receipt, lambda d: d.update(phase="smoke"))
    elif change == "protocol":
        mutate_json(receipt, lambda d: d.update(protocol_sha256="0" * 64))
    elif change == "parent":
        mutate_json(receipt, lambda d: d.update(parent_sha256="0" * 64))
    elif change == "seed":
        mutate_json(receipt, lambda d: d.update(seed=1))
    elif change == "cell":
        mutate_json(receipt, lambda d: d.update(cell=1))
    elif change == "budget":
        mutate_json(receipt, lambda d: d.update(final_iteration=998))
    elif change == "checkpoint":
        (cell / "model_999.pt").write_bytes(b"changed bytes")
    elif change == "export":
        (cell / "exported/policy.onnx").write_bytes(b"different policy")
    elif change == "saved_seed":
        (cell / "params/agent.yaml").write_text("seed: 1\n")
    else:
        mutate_json(receipt, lambda d: d["gates"].update(instrument_sha256={}))
    assert summary.summarize(path, cells)["status"] == "INCOMPLETE"


def test_duplicate_cell_cannot_satisfy_repeated_seed_rule(campaign):
    path, cells = campaign
    report = summary.summarize(path, [cells[0], cells[1], cells[1]])
    assert report["status"] == "INCOMPLETE"
    assert any("duplicate training seed" in reason for reason in report["problems"])


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_direction", "truncated", "no_push", "wrong_label", "nonfinite"])
def test_raw_gate_controls(campaign, change):
    path, cells = campaign
    root = cells[2]
    if change in {"missing", "duplicate"}:
        mutate_json(root / "gates/turn-v2x.json", lambda d: d["turns"].pop() if change == "missing" else d["turns"].__setitem__(-1, copy.deepcopy(d["turns"][0])))
    elif change == "wrong_direction":
        mutate_json(root / "gates/turn-v2.json", lambda d: d["turns"][0].update(yaw_deg=-300.))
        report = summary.summarize(path, cells)
        assert report["status"] == "PASS" and not report["seeds"][2]["joint_pass"]
        return
    elif change == "nonfinite":
        mutate_json(root / "gates/turn-v2.json", lambda d: d["turns"][0].update(yaw_deg=float("inf")))
    else:
        rows = list(csv.DictReader((root / "gates/push-60.csv").open()))
        rows[0].update({"truncated": {"survival_s": 11.}, "no_push": {"pushes_applied": 0}, "wrong_label": {"label": "another-seed"}}[change])
        write_rows(root / "gates/push-60.csv", rows)
    assert summary.summarize(path, cells)["status"] == "INCOMPLETE"


def test_durable_bundle_archive_checksum_is_required(campaign, tmp_path):
    path, cells = campaign
    wrapper = tmp_path / "durable"
    wrapper.mkdir()
    with tarfile.open(wrapper / "outputs.tar.gz", "w:gz") as tar:
        for file in cells[2].rglob("*"):
            if file.is_file():
                tar.add(file, arcname=str(file.relative_to(cells[2])))
    write_json(wrapper / "completion.json", {"files": {"outputs.tar.gz": {"sha256": summary.digest(wrapper / "outputs.tar.gz")}}})
    assert summary.summarize(path, [cells[0], cells[1], wrapper])["status"] == "PASS"
    mutate_json(wrapper / "completion.json", lambda d: d["files"]["outputs.tar.gz"].update(sha256="bad"))
    assert summary.summarize(path, [cells[0], cells[1], wrapper])["status"] == "INCOMPLETE"
