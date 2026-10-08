"""Mission7 train/evaluate/smoke with RSL-RL PPO and measured promotion gates."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from bhl_robust.mission.env import MissionEnv
from bhl_robust.mission.layout import SPLITS, STAGES, generate
from bhl_robust.mission.policy import MissionPolicy
from bhl_robust.mission.sensors import ARMS, FAILURES, MissionSensors


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".pending")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def tensor(obs):
    return TensorDict({"policy": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)}, batch_size=[1])


def environment(args, cache, **overrides):
    values = dict(arm=args.arm, stage=args.stage, split="train", seed=args.seed)
    values.update(overrides)
    return MissionEnv(args.repo, cache, **values)


def evaluate(args, policy, output, *, split="validation", count=16, failure="normal", video=None):
    if not 1 <= count <= SPLITS[split][1]:
        raise ValueError("evaluation count must fit split")
    env = environment(args, output.parent/"eval-cache", split=split, failure=failure, seed=1000+args.seed)
    rows, weights = [], []
    renderer = writer = None
    for index in range(count):
        obs = env.reset(index)
        if video is not None and index == 0:
            import mujoco
            import imageio.v2 as imageio
            renderer = mujoco.Renderer(env.model, height=540, width=960)
            writer = imageio.get_writer(str(video), fps=5)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [4, 4, .3]
            camera.distance, camera.azimuth, camera.elevation = 15, 120, -65
        try:
            while True:
                with torch.inference_mode():
                    packed = tensor(obs)
                    action = policy.act_inference(packed)[0].numpy()
                    weights.append(policy.actor.encode(packed["policy"])[1][0, -1].tolist())
                if args.no_button:
                    action[3] = -10
                obs, _, done, _ = env.step(action)
                if renderer is not None:
                    renderer.update_scene(env.runner.d, camera=camera)
                    writer.append_data(renderer.render())
                if done:
                    break
        finally:
            if renderer is not None:
                renderer.close()
                writer.close()
                renderer = writer = None
        rows.append(env.metrics())
        write_json(output, {"complete": False, "split": split, "episodes": rows})
    rate = float(np.mean([row["success"] for row in rows]))
    result = {"schema": "bhl-mission7-eval-v1", "complete": True,
              "stage": args.stage, "arm": args.arm, "seed": args.seed,
              "split": split, "failure": failure, "no_button": args.no_button,
              "success_rate": rate, "episode_count": count, "episodes": rows,
              "mean_fusion_weights_lidar_stereo": np.mean(weights, axis=0).tolist(),
              "sensor_provenance": MissionSensors.provenance()}
    write_json(output, result)
    return result


def load_policy(path, obs, args, previous=False):
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    if checkpoint["arm"] != args.arm or checkpoint["seed"] != args.seed:
        raise ValueError("checkpoint arm/seed mismatch")
    expected = STAGES[STAGES.index(args.stage)-1] if previous else args.stage
    if checkpoint["stage"] != expected:
        raise ValueError("checkpoint stage mismatch")
    policy = MissionPolicy(obs)
    policy.load_state_dict(checkpoint["model"])
    return policy


def train(args):
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    env = environment(args, args.out/"train-cache")
    obs = tensor(env.reset())
    if args.stage != "approach":
        if args.previous_gate is None:
            raise ValueError("later stages require the previous stage's passed validation gate")
        gate = json.loads(args.previous_gate.read_text())
        if not (gate.get("passed") is True and gate.get("kind") == "policy_validation"
                and gate.get("split") == "validation" and gate.get("episode_count", 0) >= 16
                and gate.get("arm") == args.arm and gate.get("seed") == args.seed
                and gate.get("stage") == STAGES[STAGES.index(args.stage)-1]):
            raise ValueError("previous stage did not pass measured validation")
        checkpoint = Path(gate["checkpoint"])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != gate["checkpoint_sha256"]:
            raise ValueError("passed checkpoint hash mismatch")
        policy = load_policy(checkpoint, obs, args, previous=True)
    else:
        if args.previous_gate:
            raise ValueError("approach starts from scratch")
        policy = MissionPolicy(obs)
    initial = evaluate(args, policy, args.out/"initial-validation.json", count=16)
    ppo = PPO(policy, num_learning_epochs=4, num_mini_batches=4, learning_rate=3e-4,
              schedule="adaptive", entropy_coef=.01, gamma=.995, lam=.95)
    ppo.init_storage("rl", 1, args.horizon, obs, [5])
    started, episodes, steps = time.monotonic(), 0, 0
    thresholds = {}
    config = {"arm": args.arm, "seed": args.seed, "stage": args.stage,
              "updates": args.updates, "horizon": args.horizon, "control_dt": .2,
              "actor_critic_inputs": "same_sensor_only_history", "algorithm": "installed_rsl_rl.PPO",
              "gait_deploy": str(env.deploy),
              "gait_sha256": hashlib.sha256((env.deploy.parent/"policy.onnx").read_bytes()).hexdigest(),
              "sensor_provenance": MissionSensors.provenance()}
    write_json(args.out/"config.json", config)
    with (args.out/"learning.jsonl").open("x") as log, (args.out/"episodes.jsonl").open("x") as episode_log:
        for update in range(args.updates):
            total_reward = 0.
            with torch.no_grad():
                for _ in range(args.horizon):
                    action = ppo.act(obs)[0].numpy()
                    next_obs, reward, done, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    if done:
                        episode_log.write(json.dumps({"update": update, "steps": steps, **env.metrics()}, allow_nan=False)+"\n")
                        episode_log.flush()
                        episodes += 1
                        next_obs = env.reset()
                    obs = tensor(next_obs)
                    # The specified finite mission deadline is a terminal failure,
                    # not an artificial rollout truncation needing bootstrap.
                    ppo.process_env_step(obs, torch.tensor([reward]), torch.tensor([done]), {})
                ppo.compute_returns(obs)
            losses = ppo.update()
            if not np.isfinite(list(losses.values())).all() or not all(torch.isfinite(p).all() for p in policy.parameters()):
                raise FloatingPointError("nonfinite PPO update")
            row = {"update": update+1, "steps": steps, "episodes": episodes,
                   "rollout_reward": total_reward, "elapsed_s": time.monotonic()-started, **losses}
            log.write(json.dumps(row, allow_nan=False)+"\n")
            log.flush()
            if (update+1) % 25 == 0 or update+1 == args.updates:
                torch.save({**config, "update": update+1, "model": policy.state_dict(),
                            "optimizer": ppo.optimizer.state_dict()}, args.out/f"model_{update+1}.pt")
                print(json.dumps(row), flush=True)
            if (update+1) % 100 == 0 and update+1 != args.updates:
                check = evaluate(args, policy, args.out/f"validation-{update+1}.json", count=16)
                for threshold in (.3, .5, .8):
                    if check["success_rate"] >= threshold:
                        thresholds.setdefault(str(threshold), steps)
    checkpoint = (args.out/f"model_{args.updates}.pt").resolve()
    final = evaluate(args, policy, args.out/"validation.json", count=16)
    passed = final["success_rate"] >= .30 and (final["success_rate"] >= initial["success_rate"]+.125
             or initial["success_rate"] >= .30 and final["success_rate"] >= initial["success_rate"]-.125)
    for threshold in (.3, .5, .8):
        if final["success_rate"] >= threshold:
            thresholds.setdefault(str(threshold), steps)
    gate = {"kind": "policy_validation", "passed": passed, "split": "validation",
            "arm": args.arm, "seed": args.seed, "stage": args.stage, "episode_count": 16,
            "initial_success_rate": initial["success_rate"], "success_rate": final["success_rate"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "steps": steps, "threshold_first_measured_steps": thresholds,
            "gate_rule": "success>=.30; improve >=.125, or retain already-passing transferred behavior within .125"}
    write_json(args.out/"gate.json", gate)
    print("MISSION7_POLICY_" + ("PASS" if passed else "FAIL"), flush=True)
    return 0 if passed else 2


def smoke(args):
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    results = []
    # Every sensor arm and all objective types must step/reset; short horizons
    # force real terminal/reset transitions in PPO's collection loop.
    for arm in ARMS:
        for stage in ("approach", "doors", "transport"):
            env = environment(args, args.out/f"cache-{arm}-{stage}", arm=arm, stage=stage, max_seconds=.6)
            obs = tensor(env.reset(0))
            policy = MissionPolicy(obs)
            ppo = PPO(policy, num_learning_epochs=1, num_mini_batches=2, schedule="fixed")
            ppo.init_storage("rl", 1, 8, obs, [5])
            resets = 0
            with torch.no_grad():
                for _ in range(8):
                    action = ppo.act(obs)[0].numpy()
                    next_obs, reward, done, _ = env.step(action)
                    if done:
                        next_obs = env.reset(1)
                        resets += 1
                    obs = tensor(next_obs)
                    ppo.process_env_step(obs, torch.tensor([reward]), torch.tensor([done]), {})
                ppo.compute_returns(obs)
            losses = ppo.update()
            passed = bool(resets and np.isfinite(list(losses.values())).all() and torch.isfinite(obs["policy"]).all()
                          and all(torch.isfinite(p).all() for p in policy.parameters()))
            results.append({"arm": arm, "stage": stage, "passed": passed, "resets": resets, "losses": losses})
            write_json(args.out/"smoke.json", {"complete": False, "cells": results})
    # Upright locomotion smoke is separate from finite random actions/PPO.
    env = environment(args, args.out/"standing-cache", max_seconds=2.)
    env.reset(0)
    while True:
        _, _, done, _ = env.step(np.zeros(5))
        if done:
            break
    standing = env.metrics()
    # Contact fixtures deliberately reposition the robot; these are scoring /
    # physics controls, never demonstrations or learned-policy success evidence.
    import mujoco
    contacts = []
    for kind in ("correct", "wrong", "no_intent", "no_contact"):
        control = environment(args, args.out/f"contact-{kind}", stage="doors", max_seconds=.6)
        control.reset(0)
        side = control.layout.correct_sides[0]*(-1 if kind == "wrong" else 1)
        if kind != "no_contact":
            control.runner.d.qpos[control.slot.qpos_adr:control.slot.qpos_adr+2] = control.layout.plate(0, side)
            mujoco.mj_forward(control.model, control.runner.d)
        control.step([0, 0, 0, 0 if kind == "no_intent" else 2, 0])
        opened = control.state.open[0]
        contacts.append({"control": kind, "door_open": opened,
                         "wrong_activations": control.state.wrong_buttons,
                         "passed": opened == (kind == "correct") and
                         (kind != "wrong" or control.state.wrong_buttons == 1)})
    placements = []
    for inside in (True, False):
        control = environment(args, args.out/f"placement-{inside}", stage="transport", max_seconds=1.6)
        control.reset(0)
        control.state.acquired = True
        control.state.open[:] = [True, True]
        control.state.crossed[:] = [True, True]
        control._gates()
        goal = control.layout.xy(control.layout.route[-1])
        if inside:
            control.runner.d.qpos[control.runner.parcel_q:control.runner.parcel_q+3] = [*goal, .25]
        while True:
            _, _, done, _ = control.step([0, 0, 0, 0, 0])
            if done:
                break
        row = control.metrics()
        placements.append({"object_in_drop_zone": inside, "passed": row["success"] == inside, "metrics": row})
    passed = (all(r["passed"] for r in results+contacts+placements) and standing["failure"] == "timeout"
              and standing["collisions"] == 0)
    write_json(args.out/"smoke.json", {"kind": "infrastructure_smoke", "complete": True,
               "passed": passed, "cells": results, "standing": standing, "physical_contact_controls": contacts,
               "placement_controls": placements,
               "policy_success_claim": False})
    print("MISSION7_SMOKE_"+("PASS" if passed else "FAIL"), flush=True)
    return 0 if passed else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("smoke", "train", "eval", "manifest"))
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, default="both")
    p.add_argument("--stage", choices=STAGES, default="approach")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--previous-gate", type=Path)
    p.add_argument("--infrastructure-gate", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--failure", choices=FAILURES, default="normal")
    p.add_argument("--no-button", action="store_true")
    p.add_argument("--video", type=Path)
    args = p.parse_args()
    if args.updates < 1 or args.horizon < 4 or args.horizon % 4:
        p.error("positive updates and horizon divisible by four required")
    # All produced artifacts must remain in the authoritative repository.
    args.repo = args.repo.resolve(strict=True)
    if not args.out.resolve().is_relative_to(args.repo):
        p.error("output must remain inside --repo")
    if args.video and (args.video.exists() or not args.video.resolve().is_relative_to(args.repo)):
        p.error("video must be a fresh path inside --repo")
    if args.out.exists():
        p.error("output already exists; choose a fresh run path")
    torch.set_num_threads(1)
    if args.infrastructure_gate:
        gate = json.loads(args.infrastructure_gate.read_text())
        if not (gate.get("kind") == "infrastructure_smoke" and gate.get("passed") is True and gate.get("complete") is True):
            p.error("infrastructure smoke did not pass")
    if args.mode == "train":
        return train(args)
    if args.mode == "smoke":
        return smoke(args)
    if args.mode == "manifest":
        layouts = [generate(split, i).metadata() for split, (_, n) in SPLITS.items() for i in range(n)]
        hashes = [l["geometry_sha256"] for l in layouts]
        assert len(hashes) == len(set(hashes)), "duplicate geometry across manifest"
        write_json(args.out, {"schema": "bhl-mission7-layouts-v1", "splits": SPLITS, "layouts": layouts})
        return 0
    if not args.checkpoint:
        p.error("eval needs --checkpoint")
    env = environment(args, args.out.parent/"load-cache", split=args.split)
    obs = tensor(env.reset(0))
    policy = load_policy(args.checkpoint, obs, args)
    evaluate(args, policy, args.out, split=args.split, count=args.episodes, failure=args.failure, video=args.video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
