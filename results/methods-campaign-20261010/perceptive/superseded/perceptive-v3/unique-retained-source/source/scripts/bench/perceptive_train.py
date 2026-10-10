#!/usr/bin/env python3
"""Frozen PPO teacher, measured-LiDAR students and matched degradation checks.

Training is actual Isaac Lab physics. Results explicitly distinguish execution,
teacher qualification and student evaluation. No upstream paper score is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def unpack(obs):
    import torch
    from bhl_robust.research.perceptive_policy import HISTORY, RAYS, CELLS
    proprio, height = obs["policy"], obs["teacher"]
    points = obs["measured"].reshape(len(proprio), HISTORY, RAYS, 5)
    if proprio.shape[-1] != 45 or height.shape[-1] != CELLS or obs["critic"].shape[-1] != 48:
        raise ValueError("actual teacher/student observation dimensions changed")
    critic = torch.cat((obs["critic"][:, :45], height, obs["critic"][:, -3:]), -1)
    return proprio, height, points, critic


def mean_action(model, packet, teacher_mode):
    p, h, points, _ = packet
    return model(p, h if teacher_mode else points)


def train(model, env, *, iterations, teacher=None, seed=0, steps=24, smoke=False):
    import torch
    from bhl_robust.research.perceptive_policy import generalized_advantage, edge_reconstruction_loss, trajectory_penalty
    teacher_mode = teacher is None
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    observations, _ = env.reset(seed=seed)
    packet = unpack(observations)
    logs = []
    for iteration in range(iterations):
        storage = {k: [] for k in ("proprio", "height", "points", "critic", "action", "logp", "value", "reward", "done", "teacher")}
        for _ in range(steps):
            with torch.no_grad():
                mean = mean_action(model, packet, teacher_mode)
                std = model.log_std.clamp(-4., 0.).exp().expand_as(mean)
                distribution = torch.distributions.Normal(mean, std)
                action = distribution.sample()
                value = model.value(packet[3])
                target = mean if teacher_mode else teacher(packet[0], packet[1])
                old = packet
                observations, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                packet = unpack(observations)
                # Match the standard RSL time-limit bootstrap without crossing
                # the reset boundary in GAE or next-state disagreement.
                reward = reward + .99 * value * truncated.float()
                if not teacher_mode and model.arm == "query_memory_trajectory":
                    reward = reward + trajectory_penalty(model(packet[0], packet[2]),
                        teacher(packet[0], packet[1]), done)
                row = dict(zip(("proprio", "height", "points", "critic"), old))
                row.update(action=action, logp=distribution.log_prob(action).sum(-1), value=value,
                           reward=reward, done=done, teacher=target)
                for key in storage:
                    storage[key].append(row[key].detach().clone())
        data = {k: torch.stack(v) for k, v in storage.items()}
        with torch.no_grad():
            advantage, returns = generalized_advantage(data["reward"], data["value"], data["done"], model.value(packet[3]))
            advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
        flat = {k: v.flatten(0, 1) for k, v in data.items()}
        adv, ret = advantage.flatten(), returns.flatten()
        losses = []
        for epoch in range(1 if smoke else 4):
            permutation = torch.randperm(len(adv), device=adv.device)
            for ids in permutation.split(128 if not teacher_mode else 512):
                p, h, points = flat["proprio"][ids], flat["height"][ids], flat["points"][ids]
                mean = model(p, h if teacher_mode else points)
                dist = torch.distributions.Normal(mean, model.log_std.clamp(-4., 0.).exp().expand_as(mean))
                ratio = (dist.log_prob(flat["action"][ids]).sum(-1)-flat["logp"][ids]).exp()
                actor_loss = -torch.minimum(ratio*adv[ids], ratio.clamp(.8, 1.2)*adv[ids]).mean()
                value_loss = (model.value(flat["critic"][ids])-ret[ids]).square().mean()
                loss = actor_loss + .5*value_loss - .01*dist.entropy().sum(-1).mean()
                imitation, reconstruction = mean.new_zeros(()), mean.new_zeros(())
                if not teacher_mode:
                    imitation = (mean-flat["teacher"][ids]).square().mean()
                    reconstruction = edge_reconstruction_loss(model.reconstruct(p, points), h)
                    loss = loss + imitation + .2*reconstruction
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite perceptive PPO loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
        logs.append(dict(iteration=iteration, mean_reward=float(data["reward"].mean()),
                         episode_terminations=int(data["done"].sum()), mean_loss=sum(losses)/len(losses),
                         terrain_level=float(env.scene.terrain.terrain_levels.float().mean()),
                         sensor_delay_steps=int(env.cfg.perception_delay_steps)))
        print(json.dumps(logs[-1]), flush=True)
    return logs


def evaluate(model, env, *, teacher_mode, seed, steps=250, dropout=0., delay=0):
    """Score the first episode of each environment; never count reset survivors."""
    import torch
    from bhl_robust.tasks.perceptive_env_cfg import point_state
    env.cfg.perception_dropout, env.cfg.perception_delay_steps = dropout, delay
    point_state(env).delay_steps = delay
    observations, _ = env.reset(seed=seed)
    packet = unpack(observations)
    start = env.scene["robot"].data.root_pos_w[:, :2].clone()
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    survived_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    integrated_target = torch.zeros(env.num_envs, device=env.device)
    integrated_error = torch.zeros_like(integrated_target)
    position = start.clone()
    latencies, reconstruction_errors = [], []
    with torch.no_grad():
        for _ in range(steps):
            torch.cuda.synchronize()
            then = time.perf_counter()
            action = mean_action(model, packet, teacher_mode)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter()-then)
            if not teacher_mode:
                error = (model.reconstruct(packet[0], packet[2])-packet[1]).abs().mean(-1)
                reconstruction_errors.extend(error[alive].cpu().tolist())
            before = env.scene["robot"].data.root_pos_w[:, :2].clone()
            velocity = env.scene["robot"].data.root_lin_vel_b[:, :2]
            commands = packet[0][:, :2]
            integrated_target += torch.linalg.vector_norm(commands, dim=-1)*.04*alive
            integrated_error += torch.linalg.vector_norm(velocity-commands, dim=-1)*.04*alive
            survived_steps += alive.long()
            observations, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            # Auto-reset positions must not enter displacement measurements.
            position[alive] = torch.where(done[:, None], before,
                env.scene["robot"].data.root_pos_w[:, :2])[alive]
            alive &= ~done
            packet = unpack(observations)
    progress = torch.linalg.vector_norm(position-start, dim=-1)
    qualified = alive & (integrated_target > .1) & (progress >= .5*integrated_target)
    import numpy as np
    rows = [dict(environment=i, survived=bool(alive[i]), qualified=bool(qualified[i]),
                 completed_policy_steps=int(survived_steps[i]), displacement_m=float(progress[i]),
                 commanded_distance_m=float(integrated_target[i]), integrated_tracking_error_m=float(integrated_error[i]))
            for i in range(env.num_envs)]
    return dict(seed=seed, dropout=dropout, delay_steps=delay, episodes=env.num_envs,
                survivors=int(alive.sum()), qualified=int(qualified.sum()), rows=rows,
                inference_p95_ms=float(np.quantile(latencies, .95)*1000),
                reconstruction_mae_m=float(np.mean(reconstruction_errors)) if reconstruction_errors else None,
                scope="first-episode Isaac terrain velocity test, not navigation goals or physical traversal")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "run"), required=True)
    parser.add_argument("--cell", type=int, default=0)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.kit_args = os.environ.get("H34_KIT_ARGS", "")
    p = json.loads(args.protocol.read_text())
    settings = p["perceptive"]
    seed = settings["seeds"][args.cell]
    out = Path(os.environ["H34_OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema="bhl-perceptive-training-v1", status="INCOMPLETE", phase=args.phase,
                   seed=seed, protocol_sha256=digest(args.protocol), jobs_completed=[],
                   job_id=os.environ.get("SLURM_JOB_ID"), started_utc=datetime.now(timezone.utc).isoformat(),
                   scientific_status="SMOKE_ONLY" if args.phase == "smoke" else "DEVELOPMENT_ONLY",
                   scope="12-DoF Isaac simulation; ideal ray-LiDAR with explicit noise/dropout/delay; no RGB-stereo or physical sensing claim")
    save_json(out/"campaign_result.json", receipt)
    app, env = None, None
    try:
        import torch
        from bhl_robust.research.perceptive_policy import TerrainTeacher, TerrainStudent, ARMS
        if settings["arms"] != list(ARMS) or settings["seeds"] != [0, 1, 2]:
            raise ValueError("perceptive matched ablation arms/seeds changed")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5):
            raise RuntimeError("allocated compatible GPU required")
        torch.ones(8, device="cuda").sum().item()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))
        app = AppLauncher(args).app
        from bhl_robust import compat
        compat.apply()
        from isaaclab.envs import ManagerBasedRLEnv
        from bhl_robust.tasks.perceptive_env_cfg import PerceptiveTerrainEnvCfg
        cfg = PerceptiveTerrainEnvCfg()
        cfg.seed, cfg.sim.device = seed, args.device
        cfg.scene.num_envs = settings["smoke_num_envs"] if args.phase == "smoke" else settings["num_envs"]
        cfg.episode_length_s = 20.
        if abs(cfg.sim.dt*cfg.decimation-.04) > 1e-9:
            raise ValueError("perceptive policy timestep changed")
        env = ManagerBasedRLEnv(cfg)
        from isaaclab.utils.io import dump_yaml
        dump_yaml(str(out/"env.yaml"), cfg)
        expected_terms = ["velocity_commands", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions"]
        if (list(env.observation_manager.active_terms["policy"]) != expected_terms
                or list(env.observation_manager.active_terms["critic"]) != expected_terms+["base_lin_vel"]):
            raise ValueError("warm-start observation term order changed")
        inputs = Path(os.getenv("H34_INPUTS_DIR", args.protocol.parent/"inputs"))
        warm = inputs/"perceptive_warmstart.pt"
        if digest(warm) != p["input_files"]["perceptive_warmstart.pt"]["sha256"]:
            raise ValueError("warm-start gait checkpoint changed")
        teacher = TerrainTeacher().to(args.device)
        teacher.warm_start(torch.load(warm, map_location=args.device, weights_only=False)["model_state_dict"])
        iterations = settings["smoke_iterations"] if args.phase == "smoke" else settings["teacher_iterations"]
        logs = train(teacher, env, iterations=iterations, seed=seed, smoke=args.phase == "smoke")
        save_json(out/"teacher-training.json", logs)
        torch.save(dict(state_dict=teacher.state_dict(), seed=seed, iterations=iterations,
                        protocol_sha256=receipt["protocol_sha256"]), out/"teacher.pt")
        teacher.eval().requires_grad_(False)
        check = evaluate(teacher, env, teacher_mode=True, seed=510000+seed,
                         steps=24 if args.phase == "smoke" else 250)
        save_json(out/"teacher-development.json", check)
        receipt["teacher_checkpoint_sha256"] = digest(out/"teacher.pt")
        receipt["jobs_completed"].append("teacher")
        teacher_ready = check["qualified"] >= .8*check["episodes"]
        receipt["teacher_qualified"] = bool(teacher_ready)
        if args.phase == "run" and not teacher_ready:
            receipt.update(status="NEGATIVE", scientific_status="TEACHER_QUALIFICATION_FAILED",
                           student_training="UNRUN_AFTER_FAILED_TEACHER_GATE")
            save_json(out/"campaign_result.json", receipt)
            return 0
        # Every student sees identical reset/random seeds and the same teacher.
        outcomes = {}
        for arm in ARMS:
            torch.manual_seed(seed+730000)
            torch.cuda.manual_seed_all(seed+730000)
            student = TerrainStudent(arm).to(args.device)
            student.from_teacher(teacher)
            env.cfg.perception_dropout, env.cfg.perception_delay_steps = .1, 1
            from bhl_robust.tasks.perceptive_env_cfg import point_state
            point_state(env).delay_steps = 1
            logs = train(student, env, iterations=settings["smoke_iterations"] if args.phase == "smoke"
                         else settings["student_iterations"], teacher=teacher, seed=seed+520000,
                         smoke=args.phase == "smoke")
            save_json(out/(arm+"-training.json"), logs)
            torch.save(dict(state_dict=student.state_dict(), arm=arm, seed=seed,
                            schema="bhl-perceptive-student-v1", protocol_sha256=receipt["protocol_sha256"],
                            teacher_sha256=receipt["teacher_checkpoint_sha256"]), out/(arm+".pt"))
            student.eval()
            from bhl_robust.research.perceptive_policy import export_student
            deployment = export_student(student, out/(arm+"-actor.jit"))
            save_json(out/(arm+"-deployment.json"), deployment)
            checks = []
            for dropout, delay in ((0., 0), (.5, 0), (0., 2), (.5, 2)):
                checks.append(evaluate(student, env, teacher_mode=False, seed=530000+seed,
                                       dropout=dropout, delay=delay, steps=24 if args.phase == "smoke" else 250))
            save_json(out/(arm+"-development.json"), checks)
            outcomes[arm] = dict(checkpoint_sha256=digest(out/(arm+".pt")),
                                  episodes=sum(x["episodes"] for x in checks),
                                  qualified=sum(x["qualified"] for x in checks),
                                  survivors=sum(x["survivors"] for x in checks))
            receipt["jobs_completed"].append(arm)
            save_json(out/"campaign_result.json", receipt)
        receipt.update(status="PASS", student_results=outcomes,
                       acceptance_meaning="execution complete; improvement requires paired multi-seed summary and independent confirmation")
    except Exception as error:
        receipt.update(status="INCOMPLETE", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(out/"campaign_result.json", receipt)
        if env is not None:
            env.close()
        if app is not None:
            app.close()
    return 0 if receipt["status"] in ("PASS", "NEGATIVE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
