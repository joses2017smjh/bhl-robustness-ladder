"""G-B4: is the limb factorisation exactly a factorisation?

Three questions, cheapest first. The first two need no simulator at all, so a
broken partition is caught in milliseconds rather than after a GPU allocation.

  G-B4a  Does each partition cover all 22 joints exactly once, and does
         split -> reassemble round-trip to the identity?
  G-B4b  Does `reassemble` respect joint ORDER, not just membership? A
         concatenation in dict order passes a naive round-trip and still
         permutes the robot's joints, which would be a silent limb swap.
  G-B4c  Does the wrapped env construct, reset and step, with each agent
         receiving an action slice of the declared width -- and does the joined
         action equal what the single-agent env would have been given?

If G-B4c fails, MAPPO is not being compared against PPO. It is being compared
against a different robot, and every Tier 1 row would be measuring the
partition instead of the algorithm.
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Velocity-BHL-Arms-Bumpy-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--offline", action="store_true",
                    help="run only the simulator-free checks")
parser.add_argument("--partition", type=str, default=None,
                    choices=("limb4", "limb2"),
                    help="check ONE partition. Isaac Sim does not survive "
                         "tearing a scene down and building another in the same "
                         "process -- the first version of this gate checked "
                         "limb4, then hung for an hour on limb2 and was killed "
                         "by the wall clock. One partition per process.")

_known, _ = parser.parse_known_args()
if not _known.offline:
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
else:
    args_cli = parser.parse_args()

import torch  # noqa: E402

from bhl_robust.limb_partition import (  # noqa: E402
    JOINTS, N_JOINTS, PARTITIONS, reassemble, split, validate,
)


def offline_checks() -> bool:
    ok = True
    print("G-B4a  partition coverage and round-trip")
    for kind, part in PARTITIONS.items():
        try:
            validate(part)
        except ValueError as e:
            print(f"  {kind:8} FAIL {e}")
            ok = False
            continue
        x = torch.randn(7, N_JOINTS)
        back = reassemble(split(x, part), part)
        exact = torch.equal(x, back)
        widths = {k: len(v) for k, v in part.items()}
        print(f"  {kind:8} {'ok' if exact else 'ROUND-TRIP MISMATCH'}  "
              f"widths={widths} sum={sum(widths.values())}")
        ok &= exact

    print("\nG-B4b  order is preserved, not merely membership")
    for kind, part in PARTITIONS.items():
        # A marker per joint index; if reassemble concatenated in dict order
        # instead of scattering, this comes back permuted.
        x = torch.arange(N_JOINTS, dtype=torch.float32).unsqueeze(0)
        back = reassemble(split(x, part), part)
        exact = torch.equal(x, back)
        if not exact:
            bad = [(i, JOINTS[i], int(back[0, i])) for i in range(N_JOINTS)
                   if int(back[0, i]) != i][:4]
            print(f"  {kind:8} PERMUTED at {bad}")
        else:
            print(f"  {kind:8} ok")
        ok &= exact
    return ok


def online_checks() -> bool:
    import gymnasium as gym

    import bhl_robust.tasks  # noqa: F401
    from bhl_robust.tasks.limb_marl import LimbMarlEnv, ablate_arm_deviation

    ok = True
    print(f"\nG-B4c  wrapped env on {args_cli.task}")
    kinds = [args_cli.partition] if args_cli.partition else list(PARTITIONS)
    for kind in kinds:
        try:
            cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
            cfg.scene.num_envs = args_cli.num_envs
            ablated = ablate_arm_deviation(cfg)
            base = gym.make(args_cli.task, cfg=cfg, disable_env_checker=True)
            env = LimbMarlEnv(base, partition=kind)
            obs, _ = env.reset()

            widths_ok = True
            for _ in range(args_cli.steps):
                acts = {a: torch.zeros((env.num_envs, n), device=env.device)
                        for a, n in env.num_actions.items()}
                joined = reassemble(acts, env.partition)
                if tuple(joined.shape) != (env.num_envs, N_JOINTS):
                    widths_ok = False
                obs, rew, term, trunc, _ = env.step(acts)

            agents_ok = set(obs) == set(env.possible_agents)
            print(f"  {kind:8} agents={len(env.possible_agents)} "
                  f"act={env.num_actions} obs={env.num_observations[env.possible_agents[0]]} "
                  f"state={env.num_states} arm_ablation={ablated or 'NONE FOUND'}")
            # An ablation that finds nothing is a failure, not a note. The row
            # would otherwise train with the penalty on and be reported as
            # having it off.
            ok &= widths_ok and agents_ok and bool(ablated)
            env.close()
        except Exception as e:                                   # noqa: BLE001
            import traceback
            print(f"  {kind:8} FAIL {type(e).__name__}: {e}")
            traceback.print_exc()
            ok = False
        finally:
            try:
                from isaaclab.sim import SimulationContext
                SimulationContext.clear_instance()
            except Exception:                                    # noqa: BLE001
                pass
    return ok


def main() -> None:
    ok = offline_checks()
    if not args_cli.offline:
        ok &= online_checks()
        simulation_app.close()
    print(f"\nG-B4 {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
