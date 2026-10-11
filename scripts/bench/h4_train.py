"""Frozen R1HO fine-tuning cell: train, export, and unchanged H4 qualification.

Run through the H34 substrate with --protocol --phase smoke|run --cell N.
Every seed starts from the same R1H-s2 teacher; no teacher optimizer or
iteration is resumed. Only the final additional iteration 999 is scored.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TASK = "Velocity-BHL-Arms-TurnGaitHeadingObservable-v0"
PARENT_SHA256 = "4f1b0c353421dd9ab3f4eb3d6dc755142e036ab4cabfbc0a85a77ae8087bc014"
EXPORT_ATOL = 2e-5
EXPORT_RTOL = 1e-5
_APP = None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_protocol(protocol: dict, cell: dict, phase: str):
    h4 = protocol["h4"]
    expected = {"policy_dt_s": 0.04, "num_envs": 4096, "smoke_num_envs": 64,
                "smoke_iterations": 3, "feature_dimensions": 2, "heading_wz_threshold": 0.05,
                "reference": "reset_and_observable_command_change", "reference_command_tolerance": 0.0,
                "actor_observations": 79, "critic_observations": 82, "joints": 22,
                "oracle_yaw": True, "parent_recipe": "R1H-s2"}
    for key, value in expected.items():
        if h4.get(key) != value:
            raise ValueError(f"H4 protocol {key}: {h4.get(key)!r}, expected {value!r}")
    gate = {"required_seeds": 2, "seed_count": 3, "turn_v2_unchanged": True,
            "qualification_turn_min": 9, "qualification_turn_trials": 10,
            "walk_max_abs_yaw_deg": 15, "walk_min_pass": 2, "walk_trials": 3,
            "push_max_falls": 9, "push_trials": 60}
    if h4.get("gate") != gate:
        raise ValueError("H4 gate differs from the unchanged joint qualification")
    if len(protocol["cells"]) != 3 or {c["seed"] for c in protocol["cells"]} != {0, 1, 2}:
        raise ValueError("H4 must score all three declared fine-tuning seeds")
    if cell.get("teacher_checkpoint_sha256") != PARENT_SHA256:
        raise ValueError("declared parent checksum differs")
    if cell["seed"] not in (0, 1, 2) or cell["arm"] != "R1HO" or cell["task"] != TASK:
        raise ValueError("cell is not one of the frozen R1HO seeds")
    if cell["iterations"] != 1000 or cell["final_iteration"] != 999:
        raise ValueError("the final checkpoint or additional training budget changed")
    if phase not in ("smoke", "run"):
        raise ValueError("unknown H34 phase")


def check_env_recipe(candidate_path: Path, parent_path: Path, upstream: Path) -> dict:
    """Reject undeclared science changes; permit relocated paths and run metadata."""
    from bhl_robust.eval.gait_clock import load_env_yaml
    candidate, parent = load_env_yaml(candidate_path), load_env_yaml(parent_path)
    for group in ("policy", "critic"):
        removed = candidate["observations"][group].pop("command_latched_heading", None)
        if not removed or removed.get("func") != "bhl_robust.tasks.heading_observable_env_cfg:command_latched_heading":
            raise ValueError(f"missing command-latched heading term in {group}")
    old_root = str(parent["scene"]["robot"]["spawn"]["usd_path"]).split("/source/")[0]
    def normalize(value):
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        if isinstance(value, str):
            return value.replace(old_root, "<UPSTREAM>").replace(str(upstream), "<UPSTREAM>")
        return value
    for record in (candidate, parent):
        record.pop("seed", None)
        record["sim"].pop("log_dir", None)
        record["scene"]["num_envs"] = 4096
        record["scene"]["terrain"]["num_envs"] = 4096
    candidate, parent = normalize(candidate), normalize(parent)
    differences = []
    def compare(a, b, prefix=""):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differences.append(prefix + "/" + key)
                else:
                    compare(a[key], b[key], prefix + "/" + key)
        elif a != b:
            differences.append(prefix)
    compare(candidate, parent)
    if differences:
        raise ValueError(f"undeclared R1H recipe differences: {differences[:25]}")
    return {"status": "PASS", "allowed_changes": ["two appended observations", "seed", "smoke_num_envs",
                                                   "local log directory", "relocated identical upstream assets"]}


def check_export(actor, onnx: Path) -> dict:
    import numpy as np
    import onnxruntime as ort
    import torch
    obs = np.random.default_rng(100).standard_normal((256, 79)).astype(np.float32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx), sess_options=options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: obs})[0]
    with torch.no_grad():
        expected = actor(torch.from_numpy(obs)).numpy()
    if actual.shape != (256, 22) or not np.isfinite(actual).all():
        raise ValueError("export is not a finite 79D -> 22D policy")
    np.testing.assert_allclose(actual, expected, atol=EXPORT_ATOL, rtol=EXPORT_RTOL)
    return {"status": "PASS", "input_shape": [256, 79], "output_shape": [256, 22],
            "max_abs_error": float(np.max(np.abs(actual - expected))),
            "atol": EXPORT_ATOL, "rtol": EXPORT_RTOL, "test_seed": 100}


def run(args, output: Path, work: Path) -> dict:
    global _APP
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    cell = protocol["cells"][args.cell]
    check_protocol(protocol, cell, args.phase)
    inputs = {k: protocol_path.parent / cell[k] for k in
              ("teacher_checkpoint", "teacher_env", "teacher_deploy", "teacher_agent")}
    if digest(inputs["teacher_checkpoint"]) != PARENT_SHA256:
        raise ValueError("fixed R1H-s2 parent checkpoint hash differs")
    h4 = protocol["h4"]
    smoke = args.phase == "smoke"
    iterations = h4["smoke_iterations"] if smoke else cell["iterations"]
    num_envs = h4["smoke_num_envs"] if smoke else h4["num_envs"]
    upstream = Path(os.environ["UPSTREAM"]).resolve()
    sys.path.insert(0, str(upstream / "scripts/rsl_rl"))

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("H4 requires an allocated compatible GPU")
    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 5):
        raise RuntimeError(f"Isaac Sim 5.1 requires compute capability >=7.5; allocated {capability}")
    cuda_probe = torch.ones(16, device="cuda:0") * 2
    torch.cuda.synchronize()
    if float(cuda_probe.sum().item()) != 32.0:
        raise RuntimeError("actual CUDA kernel preflight failed")
    runtime_guard = {"status": "PASS", "gpu_name": torch.cuda.get_device_name(0),
                     "compute_capability": list(capability), "torch_version": torch.__version__,
                     "cuda_kernel_sum": 32.0}
    (output / "cuda_guard.json").write_text(json.dumps(runtime_guard, indent=2) + "\n")

    from isaaclab.app import AppLauncher
    launcher = AppLauncher({"headless": True, "device": "cuda:0",
                            "kit_args": os.environ.get("H34_KIT_ARGS", "")})
    app = launcher.app
    _APP = app
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        import yaml
        from bhl_robust import compat
        compat.apply()
        import berkeley_humanoid_lite.tasks  # noqa: F401
        import bhl_robust.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from bhl_robust.eval.gait_clock import load_env_yaml
        from bhl_robust.eval.heading_observable import expand_parent_state, FEATURE_CONTRACT, HEADING_KEY

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(cell["seed"])
        np.random.seed(cell["seed"])
        env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=num_envs)
        env_cfg.seed = cell["seed"]
        env_cfg.sim.log_dir = str(work / "isaaclab")
        parent_env = load_env_yaml(inputs["teacher_env"])
        # The published parent used DR scale 1.0. Copy these exact frozen
        # event parameter ranges instead of relying on launcher overrides.
        dr_keys = {
            "physics_material": ("static_friction_range", "dynamic_friction_range"),
            "add_base_mass": ("mass_distribution_params",),
            "add_all_joint_default_pos": ("pos_distribution_params",),
            "scale_all_actuator_torque_constant": ("stiffness_distribution_params", "damping_distribution_params"),
            "base_external_force_torque": ("force_range", "torque_range"),
        }
        for event, keys in dr_keys.items():
            for key in keys:
                getattr(env_cfg.events, event).params[key] = tuple(parent_env["events"][event]["params"][key])
        if abs(env_cfg.sim.dt * env_cfg.decimation - 0.04) > 1e-12:
            raise ValueError("R1H policy timing changed")
        agent = yaml.safe_load(inputs["teacher_agent"].read_text())
        agent.update(seed=cell["seed"], max_iterations=iterations, run_name=cell["run_name"],
                     resume=False, load_run=".*", load_checkpoint="model_.*.pt")
        # Explicit spelling of RSL3's already verified fallback resolution.
        agent["obs_groups"] = {"policy": ["policy"], "critic": ["critic"]}
        log_dir = work / "training"
        log_dir.mkdir(parents=True, exist_ok=True)
        env = gym.make(TASK, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env)
        # Isaac expands scene prim templates and fills terrain spacing during
        # construction. The parent's YAML was captured after that same step;
        # compare both instantiated configurations without relaxing physics.
        dump_yaml(str(log_dir / "params/env.yaml"), env_cfg)
        (log_dir / "params/agent.yaml").write_text(yaml.safe_dump(agent, sort_keys=False))
        recipe = check_env_recipe(log_dir / "params/env.yaml", inputs["teacher_env"], upstream)
        obs = env.get_observations()
        if tuple(obs["policy"].shape) != (num_envs, 79) or tuple(obs["critic"].shape) != (num_envs, 82):
            raise ValueError("live actor/critic observations do not have frozen widths")
        robot = env.unwrapped.scene["robot"]
        if robot.num_joints != 22:
            raise ValueError("live Isaac articulation is not the frozen 22-DoF humanoid")
        term = env.unwrapped.command_manager.get_term("base_velocity")
        error = term.robot.data.heading_w - term._h4_reference
        expected_features = torch.stack((torch.sin(error), torch.cos(error) - 1.0), dim=-1)
        expected_features = torch.where((term.vel_command_b[:, 2].abs() < 0.05)[:, None],
                                        expected_features, torch.zeros_like(expected_features))
        torch.testing.assert_close(obs["policy"][:, -2:], expected_features, atol=5e-7, rtol=1e-6)
        torch.testing.assert_close(obs["critic"][:, -2:], expected_features, atol=5e-7, rtol=1e-6)
        runner = OnPolicyRunner(env, copy.deepcopy(agent), log_dir=str(log_dir), device="cuda:0")
        policy = runner.alg.policy
        parent = torch.load(inputs["teacher_checkpoint"], map_location="cpu", weights_only=False)
        expanded = expand_parent_state(parent["model_state_dict"], policy.state_dict())
        policy.load_state_dict(expanded, strict=True)
        if runner.alg.optimizer.state or runner.current_learning_iteration != 0:
            raise ValueError("teacher optimizer/iteration was resumed")
        # Compare before learning, with arbitrary nonzero new features. Both
        # copied first-layer weights must still ignore the new columns.
        parent_actor = copy.deepcopy(policy.actor).cpu().eval()
        parent_actor[0] = torch.nn.Linear(77, 256)
        parent_actor.load_state_dict({k[len("actor."):]: v for k, v in parent["model_state_dict"].items()
                                      if k.startswith("actor.")})
        test_obs = torch.from_numpy(np.random.default_rng(100).standard_normal((256, 79)).astype(np.float32))
        initial_actor = copy.deepcopy(policy.actor).cpu().eval()
        with torch.no_grad():
            initial = initial_actor(test_obs)
            reference = parent_actor(test_obs[:, :77])
        torch.testing.assert_close(initial, reference, atol=EXPORT_ATOL, rtol=EXPORT_RTOL)
        initial_error = float((initial - reference).abs().max())
        training_guard = {"status": "PASS", "recipe": recipe, "parent_sha256": PARENT_SHA256,
                          "fresh_optimizer": True, "starting_iteration": 0,
                          "resolved_obs_groups": runner.cfg["obs_groups"],
                          "live_feature_formula_check": "PASS",
                          "initial_parent_max_abs_error": initial_error,
                          "actor_width": 79, "critic_width": 82, "actions": 22,
                          "num_envs": num_envs, "additional_iterations": iterations,
                          "feature_contract": FEATURE_CONTRACT,
                          "disclosure": "shared selected R1H-s2 initialization; independent PPO seeds, not from-scratch replication"}
        (output / "training_guard.json").write_text(json.dumps(training_guard, indent=2) + "\n")
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
        if runner.current_learning_iteration != iterations - 1:
            raise ValueError("the full additional training budget did not finish")
        final_name = f"model_{iterations - 1}.pt"
        runner.save(str(output / final_name))
        if runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        shutil.copytree(log_dir / "params", output / "params")
        for event in log_dir.glob("events.out.tfevents.*"):
            shutil.copy2(event, output / event.name)
        actor = copy.deepcopy(policy.actor).cpu().eval()
        export_dir = output / "exported"
        export_dir.mkdir()
        torch.onnx.export(actor, torch.zeros(1, 79), str(export_dir / "policy.onnx"),
                          input_names=["obs"], output_names=["actions"],
                          dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
                          opset_version=17, dynamo=False)
        torch.jit.trace(actor, torch.zeros(1, 79)).save(str(export_dir / "policy.pt"))
        export_check = check_export(actor, export_dir / "policy.onnx")
        (output / "export_check.json").write_text(json.dumps(export_check, indent=2) + "\n")
        deploy = yaml.safe_load(inputs["teacher_deploy"].read_text())
        deploy["policy_checkpoint_path"] = str(export_dir / "policy.onnx")
        deploy["num_observations"] = 79
        deploy["command_velocity"] = [0.0, 0.0, 0.0]
        deploy[HEADING_KEY] = FEATURE_CONTRACT
        deploy["gait_clock"]["source"] = str(output / "params/env.yaml")
        deploy_path = export_dir / "deploy.yaml"
        deploy_path.write_text(yaml.safe_dump(deploy, sort_keys=False))
        env.close()
        # CPU gates use the original instrument and the experiment's controller.
        from h4_qualification import evaluate
        gates = evaluate(deploy_path, upstream, work, output / "gates", cell["run_name"], smoke=smoke)
        return {"status": "PASS" if smoke or gates["joint_pass"] else "NEGATIVE",
                "task": TASK, "arm": "R1HO", "seed": cell["seed"], "phase": args.phase,
                "iterations_completed": iterations, "final_iteration": iterations - 1,
                "protocol_sha256": digest(protocol_path), "cell": args.cell,
                "parent_sha256": PARENT_SHA256, "final_checkpoint_sha256": digest(output / final_name),
                "policy_sha256": digest(export_dir / "policy.onnx"), "deploy_sha256": digest(deploy_path),
                "gates": gates, "label": "22-DoF learned policy; simulated yaw feature; MuJoCo evaluation",
                "feature_contract": FEATURE_CONTRACT}
    finally:
        # Main writes its authoritative receipt before Kit shutdown, whose
        # exit code can mask an exception on the existing v51 stack.
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--phase", required=True, choices=("smoke", "run"))
    parser.add_argument("--cell", type=int, required=True)
    args = parser.parse_args(argv)
    output = Path(os.environ["H34_OUTPUT_DIR"])
    work = Path(os.environ["H34_WORK_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args, output, work)
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "INCOMPLETE", "task": TASK, "arm": "R1HO", "phase": args.phase,
                  "cell": args.cell, "error": repr(exc)}
    (output / "campaign_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("H4-CELL:", result["status"], flush=True)
    if _APP is not None:
        _APP.close()
    return 0 if result["status"] in ("PASS", "NEGATIVE") else 1


if __name__ == "__main__":
    sys.exit(main())
