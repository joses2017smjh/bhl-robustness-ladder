"""Frozen H3 PPO cell: matched warm start, causal IMU packets, fixed final policy.

Launch only through the H34 Slurm substrate. A cell PASS validates execution;
the six-cell paired summary determines the scientific H3 outcome.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TEACHER_SHA256 = "7fde8dc0ddee142974d9d640feee710496bbb2079e841bf0c7b0abe8fe79e7df"


def resolve_input(protocol_path, cell, name):
    path = Path(cell[name])
    return path if path.is_absolute() else protocol_path.parent / path


def write_receipt(out, receipt):
    path = Path(out) / "campaign_result.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def export_actor(state, output, *, seed):
    import numpy as np
    import onnxruntime as ort
    import torch
    from bhl_robust.latency_history import actor_from_state, file_sha256

    actor = actor_from_state(state).cpu()
    generator = torch.Generator(device="cpu").manual_seed(7700 + seed)
    examples = torch.randn(256, 63, generator=generator)
    policy_path = Path(output) / "policy.onnx"
    torch.onnx.export(actor, examples[:1], str(policy_path), input_names=["obs"], output_names=["actions"],
                      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=17)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(policy_path), options, providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape[-1] != 63 or session.get_outputs()[0].shape[-1] != 12:
        raise ValueError("export has wrong observation/action dimensions")
    expected = actor(examples).detach().numpy()
    actual = session.run(None, {"obs": examples.numpy()})[0]
    max_abs = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, atol=1e-5, rtol=1e-5):
        raise ValueError(f"ONNX actor parity failed: {max_abs}")
    return dict(policy_sha256=file_sha256(policy_path), input_width=63, output_width=12,
                numeric_parity_samples=256, numeric_parity_max_abs=max_abs,
                numeric_parity_atol=1e-5, numeric_parity_rtol=1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "run"), required=True)
    parser.add_argument("--cell", type=int, required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.kit_args = os.environ.get("H34_KIT_ARGS", getattr(args, "kit_args", ""))
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    cell = protocol["cells"][args.cell]
    if cell.get("task") not in ("H3", "h3") and not str(cell.get("task", "")).startswith("Velocity-BHL-Biped-H3-"):
        raise ValueError("selected protocol cell is not H3")
    arm, seed = cell["arm"], int(cell["seed"])
    if arm not in ("history", "repeated_current") or seed not in (0, 1, 2):
        raise ValueError("cell is outside the frozen H3 arms/seeds")
    expected_h3 = dict(policy_dt_s=.04, num_envs=4096, smoke_num_envs=64, smoke_iterations=3,
                       physics_dt_s=.002, training_physics_dt_s=.005,
                       train_delay_steps=[0, 1], history_packets=4,
                       evaluation_delay_steps=[0, 1, 2, 3], evaluation_seeds=list(range(201000, 201005)),
                       episode_s=10., warm_s=1., qualified_progress_fraction=.5,
                       actor_observations=63, critic_observations=48, joints=12,
                       commands=[[.3, 0., 0.], [.5, 0., 0.], [-.2, 0., 0.],
                                 [0., .2, 0.], [0., 0., .5], [.3, 0., .5]],
                       expected_confirmation_episodes=720,
                       gate=dict(required_paired_seeds=2, primary_delay_steps=2, min_primary_qualified_gain=6,
                                 min_nominal_survival=27, max_nominal_qualified_loss=1))
    scientific = protocol.get("h3", {})
    for key, expected in expected_h3.items():
        if scientific.get(key) != expected:
            raise ValueError(f"protocol h3.{key} differs from frozen runner")
    if cell.get("iterations") != 500 or cell.get("final_iteration") != 499:
        raise ValueError("protocol requests a different training budget/final checkpoint")
    if cell.get("teacher_checkpoint_sha256") != TEACHER_SHA256:
        raise ValueError("protocol names a different teacher checksum")
    expected_task = "Velocity-BHL-Biped-H3-History-v0" if arm == "history" else "Velocity-BHL-Biped-H3-Feedforward-v0"
    if cell.get("task") != expected_task:
        raise ValueError("protocol task id does not match the scientific arm")
    output = Path(os.environ["H34_OUTPUT_DIR"])
    work = Path(os.environ["H34_WORK_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if (output / "campaign_result.json").exists():
        raise FileExistsError("refusing to overwrite a cell receipt")
    from bhl_robust.latency_history import file_sha256
    teacher_path = resolve_input(protocol_path, cell, "teacher_checkpoint")
    teacher_deploy = resolve_input(protocol_path, cell, "teacher_deploy")
    teacher_hash = file_sha256(teacher_path)
    if teacher_hash != TEACHER_SHA256:
        raise ValueError("teacher hash differs from the frozen dr-default-s0 checkpoint")
    receipt = dict(schema="h3-training-cell-v1", task="H3", arm=arm, seed=seed, phase=args.phase,
                   cell=args.cell, status="INCOMPLETE", execution_status="STARTED",
                   science_status="NON_SCORED_SMOKE" if args.phase == "smoke" else "UNPAIRED_PENDING_SUMMARY",
                   protocol_sha256=file_sha256(protocol_path), teacher_sha256=teacher_hash,
                   teacher_deploy_sha256=file_sha256(teacher_deploy),
                   started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   num_envs=64 if args.phase == "smoke" else 4096,
                   requested_iterations=3 if args.phase == "smoke" else 500,
                   policy_dt=.04, training_physics_dt=.005, evaluation_physics_dt=.002,
                   actor_width=63, critic_width=48, training_delay_steps=[0, 1], training_delay_ms=[0, 40],
                   warmstart="same teacher actor/critic/std; new18 actorweights zero; fresh optimizer",
                   job_id=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename)
    write_receipt(output, receipt)
    simulation_app = None
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; H3 training requires an allocated compatible GPU")
        capability = torch.cuda.get_device_capability()
        if capability < (7, 5):
            raise RuntimeError(f"GPU capability {capability} is below the frozen supported minimum 7.5")
        # Availability alone is insufficient on the cluster's V100 hosts with
        # this CUDA build. Execute and synchronize a real kernel before Kit.
        kernel_value = float((torch.ones(16, device="cuda") * 2).sum().item())
        torch.cuda.synchronize()
        if kernel_value != 32:
            raise RuntimeError("CUDA kernel guard produced the wrong result")
        receipt["cuda_guard"] = dict(gpu_name=torch.cuda.get_device_name(), capability=list(capability),
                                     kernel_value=kernel_value, minimum_capability=[7, 5])
        write_receipt(output, receipt)
        launcher = AppLauncher(args)
        simulation_app = launcher.app
        import gymnasium as gym
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from bhl_robust import compat
        compat.apply()
        import bhl_robust.tasks  # noqa: F401
        from bhl_robust.tasks.history_latency_env_cfg import HistoryLatencyEnvCfg, FeedforwardLatencyEnvCfg, packet_state
        from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped.agents.rsl_rl_ppo_cfg import BerkeleyHumanoidLiteBipedPPORunnerCfg
        from bhl_robust.latency_history import expanded_teacher_state, actor_from_state

        cfg = HistoryLatencyEnvCfg() if arm == "history" else FeedforwardLatencyEnvCfg()
        cfg.seed = seed
        cfg.scene.num_envs = receipt["num_envs"]
        cfg.sim.device = args.device
        if abs(float(cfg.sim.dt) * int(cfg.decimation) - .04) > 1e-12:
            raise ValueError("Isaac policy rate is not 25 Hz")
        if hasattr(cfg.sim, "log_dir"):
            cfg.sim.log_dir = str(work / "isaaclab")
        agent = BerkeleyHumanoidLiteBipedPPORunnerCfg()
        agent.seed, agent.device = seed, args.device
        agent.max_iterations = receipt["requested_iterations"]
        agent.experiment_name = "h3"
        agent.run_name = f"h3-{arm}-s{seed}"
        agent.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        agent.resume = False
        # Fixed teacher-compatible MLP PPO; no inherited shell overlays.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        env = ManagerBasedRLEnv(cfg=cfg)
        wrapped = RslRlVecEnvWrapper(env)
        observations = wrapped.get_observations()
        if tuple(observations["policy"].shape) != (receipt["num_envs"], 63) or tuple(observations["critic"].shape) != (receipt["num_envs"], 48):
            raise ValueError("actor/critic observations do not match frozen dimensions")
        actor_terms = list(env.observation_manager.active_terms["policy"])
        critic_terms = list(env.observation_manager.active_terms["critic"])
        expected_terms = ["velocity_commands", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"]
        if actor_terms != expected_terms + ["h3_past_imu"] or critic_terms != expected_terms + ["base_lin_vel"]:
            raise ValueError("observation term order differs from the teacher/deploy path")
        receipt["actor_terms"], receipt["critic_terms"] = actor_terms, critic_terms
        log_dir = work / "training"
        runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=str(log_dir), device=agent.device)
        checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
        state = expanded_teacher_state(checkpoint["model_state_dict"])
        runner.alg.policy.load_state_dict(state, strict=True)
        if runner.current_learning_iteration != 0 or runner.alg.optimizer.state:
            raise ValueError("warm start accidentally resumed iteration/optimizer state")
        old_actor = actor_from_state(checkpoint["model_state_dict"], width=45)
        new_actor = actor_from_state(state, width=63)
        examples = torch.randn(256, 63, generator=torch.Generator().manual_seed(3033))
        initial_delta = float(torch.max(torch.abs(old_actor(examples[:, :45]) - new_actor(examples))).detach())
        if initial_delta > 1e-5:
            raise ValueError("history expansion changes teacher initial actions")
        sensing = packet_state(env)
        if sensing.mode != ("history" if arm == "history" else "feedforward") or sensing.max_delay_steps != 1:
            raise ValueError("wrong actual sensing arm/delay queue")
        receipt["initial_action_parity_max_abs"] = initial_delta
        receipt["fresh_optimizer"] = True
        receipt["actual_policy_shape"] = list(observations["policy"].shape)
        receipt["actual_critic_shape"] = list(observations["critic"].shape)
        dump_yaml(str(log_dir / "params" / "env.yaml"), cfg)
        dump_yaml(str(log_dir / "params" / "agent.yaml"), agent)
        started = time.perf_counter()
        runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
        receipt["training_wall_s"] = time.perf_counter() - started
        final_iteration = agent.max_iterations - 1
        final_path = log_dir / f"model_{final_iteration}.pt"
        trained = torch.load(final_path, map_location="cpu", weights_only=False)
        if trained.get("iter") != final_iteration:
            raise ValueError("fixed final checkpoint was not completed")
        if tuple(trained["model_state_dict"]["actor.0.weight"].shape) != (256, 63):
            raise ValueError("trained actor shape changed")
        shutil.copy2(final_path, output / "final-checkpoint.pt")
        receipt["checkpoint_sha256"] = file_sha256(output / "final-checkpoint.pt")
        receipt["checkpoint_iteration"] = final_iteration
        receipt["completed_iterations"] = final_iteration + 1
        receipt["export"] = export_actor(trained["model_state_dict"], output, seed=seed)
        from omegaconf import OmegaConf
        deploy = OmegaConf.load(teacher_deploy)
        deploy.num_observations, deploy.history_length = 63, 0
        deploy.policy_checkpoint_path = str(output / "policy.onnx")
        deploy.physics_dt, deploy.policy_dt = .002, .04
        deploy.h3 = dict(arm=arm, training_seed=seed, history_packets=4, imu_columns=[3, 9],
                         packet_padding="first captured packet", teacher_sha256=teacher_hash)
        OmegaConf.save(deploy, output / "deploy.yaml")
        params = output / "params"
        shutil.copytree(log_dir / "params", params)
        for events in log_dir.glob("events.out.tfevents.*"):
            shutil.copy2(events, output / events.name)
        env.close()
        from history_latency_eval import evaluate_cell, validate_evaluation
        upstream = Path(os.environ.get("UPSTREAM", REPO / "external" / "Berkeley-Humanoid-Lite"))
        evaluation = evaluate_cell(output / "deploy.yaml", arm=arm, training_seed=seed,
                                   out=output, upstream=upstream, smoke=args.phase == "smoke")
        if args.phase == "run":
            issues = validate_evaluation(evaluation, arm=arm, seed=seed)
            if issues:
                raise ValueError(f"incomplete/invalid held-out grid: {issues}")
        receipt["evaluation_episodes"] = evaluation["num_episodes"]
        receipt["evaluation_sha256"] = file_sha256(output / "latency-evaluation.json")
        receipt["status"] = "PASS"
        receipt["execution_status"] = "COMPLETE"
        receipt["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_receipt(output, receipt)
        print("H3 CELL EXECUTION PASS; scientific result requires all six paired cells", flush=True)
        return 0
    except Exception as exc:
        receipt.update(status="INCOMPLETE", execution_status="FAILED", error=f"{type(exc).__name__}: {exc}",
                       finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_receipt(output, receipt)
        traceback.print_exc()
        return 1
    finally:
        if simulation_app is not None:
            # Isaac shutdown may hard-exit. The durable receipt is written first.
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
