"""Train limb agents with skrl's MAPPO or IPPO.

The single file that knows about agents. Everything else -- the env, the
rewards, the terminations, the curriculum -- is the same code the PPO baseline
runs, wrapped by `LimbMarlEnv` so that the action factorisation is the only
difference. That is the point: a Tier 1 row has to be answerable as "the split
helped" or "it did not", and any second difference makes it unanswerable.

Not yet run. G-B4's online half must pass first (`slurm/88_marl_gate.sbatch`);
if the action slices do not reassemble into exactly the single-agent vector,
this trains a different robot.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_iterations", type=int, default=6000)
parser.add_argument("--run_name", type=str, default="marl")
parser.add_argument("--partition", type=str, default="limb4",
                    choices=("limb4", "limb2", "legs2", "limb1"))
parser.add_argument("--algo", type=str, default="mappo", choices=("mappo", "ippo"))
parser.add_argument("--ablate-arm-deviation", action="store_true")
parser.add_argument("--hparams", type=str, default="rsl", choices=("rsl", "skrl-default"),
                    help="rsl: every PPO setting and the network read from the task's own "
                         "rsl-rl runner config, so a skrl row differs from the PPO control "
                         "only in the factorisation. skrl-default: what the first block "
                         "(marl-* runs, 2026-08/09) trained with, kept to reproduce it.")
parser.add_argument("--rollouts", type=int, default=None, help="override the hparam set")
parser.add_argument("--learning-rate", type=float, default=None, help="override the hparam set")
parser.add_argument("--write-interval", type=int, default=60,
                    help="timesteps between TensorBoard writes; a 2-iteration gate "
                         "needs it below 48 or it writes nothing to check")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import bhl_robust.tasks  # noqa: F401,E402
from bhl_robust.limb_partition import PARTITIONS  # noqa: E402
from bhl_robust.tasks.limb_marl import LimbMarlEnv, ablate_arm_deviation  # noqa: E402

from skrl.memories.torch import RandomMemory  # noqa: E402
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model  # noqa: E402
from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG  # noqa: E402
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG  # noqa: E402
from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa: E402
from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa: E402
from skrl.trainers.torch import SequentialTrainer  # noqa: E402
from skrl.utils import set_seed  # noqa: E402


_ACT = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}


def _mlp(inp: int, hidden: list[int], out: int, act: str = "elu") -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), _ACT[act]()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


#: The first block's network. Its docstring called it "the baseline's actor
#: shape"; the baseline is [256, 128, 128]. Kept only under --hparams skrl-default.
SKRL_DEFAULT_DIMS = [256, 256, 128]


def rsl_matched(task: str):
    """The PPO control's settings, read from the task's own rsl-rl runner config.

    The first block ran skrl's defaults beside an rsl-rl control, and they
    differ in nearly everything but the clip ratio: entropy 0 against 0.008,
    a fixed learning rate against the KL-adaptive schedule, 8 epochs x 2
    mini-batches against 5 x 4, gradient clip 0.5 against 1.0, no time-limit
    bootstrap, observation and value normalisation rsl-rl does not use, and a
    wider network. Every skrl row -- limb1 included -- then survived 34-80
    steps where PPO survived ~225, so the comparison was among handicapped
    policies. Reading the values from the same object rsl-rl reads keeps them
    identical by construction rather than by transcription.
    """
    import importlib

    entry = gym.spec(task).kwargs["rsl_rl_cfg_entry_point"]
    if isinstance(entry, str):
        mod, _, name = entry.partition(":")
        entry = getattr(importlib.import_module(mod), name)
    rcfg = entry()
    alg, pol = rcfg.algorithm, rcfg.policy
    hp = {
        "rollouts": rcfg.num_steps_per_env,
        "learning_epochs": alg.num_learning_epochs,
        "mini_batches": alg.num_mini_batches,
        "discount_factor": alg.gamma,
        "lambda": alg.lam,
        "learning_rate": alg.learning_rate,
        "grad_norm_clip": alg.max_grad_norm,
        "ratio_clip": alg.clip_param,
        "value_clip": alg.clip_param,
        "clip_predicted_values": bool(alg.use_clipped_value_loss),
        "entropy_loss_scale": alg.entropy_coef,
        "value_loss_scale": alg.value_loss_coef,
        # rsl-rl adds gamma * V(s) to the reward on time-outs.
        "time_limit_bootstrap": True,
    }
    if alg.schedule == "adaptive":
        hp["learning_rate_scheduler"] = KLAdaptiveLR
        hp["learning_rate_scheduler_kwargs"] = {"kl_threshold": alg.desired_kl}
    net = (list(pol.actor_hidden_dims), list(pol.critic_hidden_dims), pol.activation)
    return hp, net, bool(getattr(rcfg, "empirical_normalization", False))


class Policy(GaussianMixin, Model):
    def __init__(self, obs_space, act_space, device, n_act, hidden, act):
        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(self, clip_actions=False)
        self.net = _mlp(self.num_observations, hidden, n_act, act)
        # log std 0 is rsl-rl's init_noise_std = 1.0.
        self.log_std_parameter = nn.Parameter(torch.zeros(n_act))

    def compute(self, inputs, role=""):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, obs_space, act_space, device, hidden, act):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = _mlp(self.num_observations, hidden, 1, act)

    def compute(self, inputs, role=""):
        return self.net(inputs["states"]), {}


def main() -> None:
    set_seed(args_cli.seed)

    cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    if args_cli.ablate_arm_deviation:
        found = ablate_arm_deviation(cfg)
        print(f"[marl] joint_deviation_arms ablated: {found}")
        if not found:
            # Loud, not silent. The work order requires this ablation; a term
            # that quietly is not there means it is still on under another name.
            raise SystemExit("joint_deviation_arms not found -- refusing to "
                             "train with an unablated arm penalty")

    base = gym.make(args_cli.task, cfg=cfg, disable_env_checker=True)
    env = LimbMarlEnv(base, partition=args_cli.partition)
    device = env.device
    policy_terms = list(base.unwrapped.observation_manager.active_terms.get("policy", []))
    print(f"[marl] {args_cli.algo} on {args_cli.partition}: "
          f"{env.possible_agents} widths={env.num_actions}")
    # One line a gate can parse: what the constructed env and wrapper actually
    # are, not what the command line asked for.
    print(f"[marl-gate] task={args_cli.task} partition={args_cli.partition} "
          f"n_dof={env.n_dof} agents={len(env.possible_agents)} "
          f"act={env.num_actions} obs={env.num_observations[env.possible_agents[0]]} "
          f"state={env.num_states} policy_terms={policy_terms} "
          f"joints={ {a: [env.joint_names[i] for i in idx][:1] for a, idx in env.partition.items()} }",
          flush=True)

    from gymnasium import spaces
    obs_spaces, act_spaces = {}, {}
    for a in env.possible_agents:
        obs_spaces[a] = spaces.Box(-float("inf"), float("inf"),
                                   (env.num_observations[a],))
        act_spaces[a] = spaces.Box(-1.0, 1.0, (env.num_actions[a],))
    state_space = spaces.Box(-float("inf"), float("inf"), (env.num_states,))

    if args_cli.hparams == "rsl":
        hp, (actor_dims, critic_dims, act), normalize = rsl_matched(args_cli.task)
    else:
        hp = {"rollouts": 24, "learning_rate": 1.0e-3}
        actor_dims = critic_dims = SKRL_DEFAULT_DIMS
        act, normalize = "elu", True
    if args_cli.rollouts is not None:
        hp["rollouts"] = args_cli.rollouts
    if args_cli.learning_rate is not None:
        hp["learning_rate"] = args_cli.learning_rate
    rollouts = hp["rollouts"]
    print(f"[marl-gate] hparams={args_cli.hparams} actor={actor_dims} critic={critic_dims} "
          f"act={act} normalize={normalize} "
          f"{ {k: (v.__name__ if isinstance(v, type) else v) for k, v in hp.items()} }", flush=True)

    memories, models = {}, {}
    for a in env.possible_agents:
        memories[a] = RandomMemory(memory_size=rollouts,
                                   num_envs=env.num_envs, device=device)
        models[a] = {
            "policy": Policy(obs_spaces[a], act_spaces[a], device,
                             env.num_actions[a], actor_dims, act),
            # MAPPO's critic reads the shared state; IPPO's reads the agent's
            # own observation. That is the entire difference between the two
            # rows, which is why they are otherwise identical here.
            "value": Value(state_space if args_cli.algo == "mappo" else obs_spaces[a],
                           act_spaces[a], device, critic_dims, act),
        }

    if args_cli.algo == "mappo":
        agent_cfg = MAPPO_DEFAULT_CONFIG.copy()
    else:
        agent_cfg = IPPO_DEFAULT_CONFIG.copy()
    agent_cfg.update(hp)
    if normalize:
        agent_cfg.update({
            "state_preprocessor": RunningStandardScaler,
            "state_preprocessor_kwargs": {"size": env.num_states, "device": device},
            "value_preprocessor": RunningStandardScaler,
            "value_preprocessor_kwargs": {"size": 1, "device": device},
        })
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "skrl", "marl", f"{stamp}_{args_cli.run_name}")
    agent_cfg["experiment"] = {"directory": log_dir, "experiment_name": "",
                               "write_interval": args_cli.write_interval,
                               "checkpoint_interval": 500}

    kw = dict(possible_agents=env.possible_agents, models=models,
              memories=memories, cfg=agent_cfg,
              observation_spaces=obs_spaces, action_spaces=act_spaces,
              device=device)
    if args_cli.algo == "mappo":
        kw["shared_observation_spaces"] = {a: state_space for a in env.possible_agents}
        agent = MAPPO(**kw)
    else:
        agent = IPPO(**kw)
    print(f"[marl-gate] trainer_agent={type(agent).__name__} log_dir={log_dir}", flush=True)

    trainer = SequentialTrainer(
        env=env, agents=agent,
        cfg={"timesteps": args_cli.max_iterations * rollouts,
             "headless": True,
             # Isaac Lab reports its curriculum and per-term episode sums under
             # infos["log"]; skrl's trainer forwards infos["episode"] by default
             # and silently drops the rest. Every MARL run before this line
             # logged no terrain level -- the work order's primary metric --
             # so the split could only be compared on reward.
             "environment_info": "log"},
    )
    trainer.train()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
