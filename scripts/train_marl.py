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
                    choices=("limb4", "limb2", "limb1"))
parser.add_argument("--algo", type=str, default="mappo", choices=("mappo", "ippo"))
parser.add_argument("--ablate-arm-deviation", action="store_true")
parser.add_argument("--rollouts", type=int, default=24)
parser.add_argument("--learning-rate", type=float, default=1.0e-3)
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
from skrl.trainers.torch import SequentialTrainer  # noqa: E402
from skrl.utils import set_seed  # noqa: E402


def _mlp(inp: int, out: int) -> nn.Sequential:
    """The baseline's actor shape, so width is not a second variable."""
    return nn.Sequential(
        nn.Linear(inp, 256), nn.ELU(),
        nn.Linear(256, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, out),
    )


class Policy(GaussianMixin, Model):
    def __init__(self, obs_space, act_space, device, n_act):
        Model.__init__(self, obs_space, act_space, device)
        GaussianMixin.__init__(self, clip_actions=False)
        self.net = _mlp(self.num_observations, n_act)
        self.log_std_parameter = nn.Parameter(torch.zeros(n_act))

    def compute(self, inputs, role=""):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, obs_space, act_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = _mlp(self.num_observations, 1)

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
    print(f"[marl] {args_cli.algo} on {args_cli.partition}: "
          f"{env.possible_agents} widths={env.num_actions}")

    from gymnasium import spaces
    obs_spaces, act_spaces = {}, {}
    for a in env.possible_agents:
        obs_spaces[a] = spaces.Box(-float("inf"), float("inf"),
                                   (env.num_observations[a],))
        act_spaces[a] = spaces.Box(-1.0, 1.0, (env.num_actions[a],))
    state_space = spaces.Box(-float("inf"), float("inf"), (env.num_states,))

    memories, models = {}, {}
    for a in env.possible_agents:
        memories[a] = RandomMemory(memory_size=args_cli.rollouts,
                                   num_envs=env.num_envs, device=device)
        models[a] = {
            "policy": Policy(obs_spaces[a], act_spaces[a], device,
                             env.num_actions[a]),
            # MAPPO's critic reads the shared state; IPPO's reads the agent's
            # own observation. That is the entire difference between the two
            # rows, which is why they are otherwise identical here.
            "value": Value(state_space if args_cli.algo == "mappo" else obs_spaces[a],
                           act_spaces[a], device),
        }

    if args_cli.algo == "mappo":
        agent_cfg = MAPPO_DEFAULT_CONFIG.copy()
    else:
        agent_cfg = IPPO_DEFAULT_CONFIG.copy()
    agent_cfg.update({
        "rollouts": args_cli.rollouts,
        "learning_rate": args_cli.learning_rate,
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.num_states, "device": device},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": device},
    })
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "skrl", "marl", f"{stamp}_{args_cli.run_name}")
    agent_cfg["experiment"] = {"directory": log_dir, "experiment_name": "",
                               "write_interval": 60, "checkpoint_interval": 500}

    kw = dict(possible_agents=env.possible_agents, models=models,
              memories=memories, cfg=agent_cfg,
              observation_spaces=obs_spaces, action_spaces=act_spaces,
              device=device)
    if args_cli.algo == "mappo":
        kw["shared_observation_spaces"] = {a: state_space for a in env.possible_agents}
        agent = MAPPO(**kw)
    else:
        agent = IPPO(**kw)

    trainer = SequentialTrainer(
        env=env, agents=agent,
        cfg={"timesteps": args_cli.max_iterations * args_cli.rollouts,
             "headless": True},
    )
    trainer.train()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
