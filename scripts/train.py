"""Training entrypoint for the robustness overlays.

VENDORED from external/Berkeley-Humanoid-Lite/scripts/rsl_rl/train.py @984741a
with exactly three additions, each marked `# [overlay]` below:
  1. sys.path wiring for upstream's local `cli_args` module and this repo's src/
  2. `import bhl_robust.tasks`, which registers the overlay task ids. It must
     land after AppLauncher starts SimulationApp, since the configs it pulls in
     import isaaclab at module scope.
  3. `apply_strategy_flags` after hydra returns, so CoopLift ablation flags
     that `__post_init__` already consumed still change the constructed env.

Vendoring rather than patching keeps external/ pristine and re-pinnable.
"""
# [overlay] path wiring -- must precede `import cli_args`
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_UPSTREAM = _os.environ.get("UPSTREAM", _os.path.join(_HERE, "..", "external", "Berkeley-Humanoid-Lite"))
_sys.path.insert(0, _os.path.join(_UPSTREAM, "scripts", "rsl_rl"))
_sys.path.insert(0, _os.path.join(_HERE, "..", "src"))


"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pickle
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import berkeley_humanoid_lite.tasks  # noqa: F401

# [overlay] register push / terrain task ids
import bhl_robust.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def dump_pickle(filename: str, data: object):
    """Saves data into a PKL file safely.

    The function creates any missing directory along the file's path.
    """
    # check ending
    if not filename.endswith("pkl"):
        filename += ".pkl"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # save data
    with open(filename, "wb") as f:
        pickle.dump(data, f)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # Write Isaac Lab internal logs to the project folder to avoid /tmp permission issues.
    isaaclab_log_dir = os.path.join(log_root_path, "isaaclab")
    os.makedirs(isaaclab_log_dir, exist_ok=True)
    if hasattr(env_cfg.sim, "log_dir"):
        env_cfg.sim.log_dir = isaaclab_log_dir
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # [overlay] Hydra writes CoopLift flags after @configclass __post_init__.
    # Re-apply so nopriv/notrack/staged actually change the constructed env.
    from bhl_robust.tasks.coop_depth_env_cfg import apply_depth_flags
    from bhl_robust.tasks.coop_lift_env_cfg import apply_strategy_flags
    apply_strategy_flags(env_cfg)
    # Same failure mode, same fix, for the depth arm's `drop_object_pose`: an
    # override that only reached the dumped yaml would train a policy identical
    # to the sighted control and label it as the blind one.
    apply_depth_flags(env_cfg)

    # [overlay] Symmetry augmentation. `symmetry_cfg` defaults to None on the
    # algorithm cfg, and hydra cannot set attributes on a None node, so the
    # whole object is attached here instead of being overridden field by field.
    # BHL_SYMMETRY=aug | loss | aug+loss | metrics.
    # [overlay] Algorithm variants, env-selected for the same reason symmetry is:
    # `policy` and `rnd_cfg` default to concrete objects or None on the agent
    # cfg, and Hydra cannot construct a different class or attribute onto a None
    # node from the command line.
    #
    # BHL_POLICY=recurrent swaps the MLP actor for an LSTM one. That is the
    # right move exactly when the observation does not contain what the policy
    # needs right now -- the occluded lift withholds the object pose, so what a
    # robot saw two steps ago is the only thing that locates the payload.
    #
    # BHL_RND=<weight> adds a Random Network Distillation exploration bonus.
    # The lift plateaus at a 4 cm hover across every recipe tried, and a
    # plateau that survives fourteen reward ablations looks more like a policy
    # that stopped exploring than one that is being mis-rewarded.
    pol_mode = os.environ.get("BHL_POLICY", "").strip()
    if pol_mode == "recurrent":
        from isaaclab_rl.rsl_rl import RslRlPpoActorCriticRecurrentCfg
        old_pol = agent_cfg.policy
        agent_cfg.policy = RslRlPpoActorCriticRecurrentCfg(
            init_noise_std=old_pol.init_noise_std,
            actor_hidden_dims=old_pol.actor_hidden_dims,
            critic_hidden_dims=old_pol.critic_hidden_dims,
            activation=old_pol.activation,
            rnn_type="lstm", rnn_hidden_dim=256, rnn_num_layers=1,
        )
        print("[overlay] policy: ActorCriticRecurrent (lstm, 256)")

    rnd_w = os.environ.get("BHL_RND", "").strip()
    if rnd_w:
        from isaaclab_rl.rsl_rl import RslRlRndCfg
        agent_cfg.algorithm.rnd_cfg = RslRlRndCfg(weight=float(rnd_w))
        print(f"[overlay] RND exploration bonus, weight {rnd_w}")

    sym_mode = os.environ.get("BHL_SYMMETRY", "").strip()
    if sym_mode:
        from isaaclab_rl.rsl_rl import RslRlSymmetryCfg
        agent_cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation="aug" in sym_mode,
            use_mirror_loss="loss" in sym_mode,
            mirror_loss_coeff=float(os.environ.get("BHL_MIRROR_COEFF", "1.0")),
            data_augmentation_func="bhl_robust.tasks.symmetry:bhl_symmetry",
        )
        print(f"[overlay] symmetry: {sym_mode}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
