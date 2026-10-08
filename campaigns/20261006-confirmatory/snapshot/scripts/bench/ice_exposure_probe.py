"""Do the trained B3 policies actually walk on the ice? Measured per physics step.

The placement probe (``ice_placement_probe.py``) answered where the patches are
relative to the robots at reset: a median 0.8 m away, reachable. It did not ask
whether a *trained* policy, acting, ever puts a foot on one. If it does not, the
placed-ice rung measured flat ground with a friction decoration nobody touched,
and the blind/depth comparison on it says nothing about ice.

This loads a trained ``ppo-ice-placed-*`` checkpoint in the task it was trained
on (resolved from the run's ``params/env.yaml``, cross-checked against its run
name), runs ``--num-envs`` environments for one full episode with the policy
acting, and records, for every env's *first* episode and at every *physics* step
(a hook on ``scene.update``, which ``ManagerBasedRLEnv.step`` calls once per
decimation substep):

  contact    per-foot ContactSensor filtered against that env's six patches;
             foot on ice = filtered normal force > ``--force-threshold``.
             Primary method whenever the filtered sensors build.
  geometric  foot body xy inside a patch footprint AND the sole within
             ``--z-window`` of the patch top. The foot body frame is the ankle,
             not the sole, so the sole offset is calibrated in-run as the median
             foot-body height over ground while the foot is in contact off-ice.
             Primary method only if the filtered sensors are unavailable.
  hybrid     foot body xy inside a footprint AND the robot's own
             ``contact_forces`` sensor reports that foot in contact. Reported
             for cross-checking; never the verdict.

and the env's terrain curriculum level at the start of the episode.

Output JSON (``--out``) carries ``episodes_with_any_ice_contact_fraction``,
``mean_on_ice_step_fraction``, ``n_episodes`` and ``verdict`` (EXPOSED iff at
least half the episodes touch a patch and the mean on-ice step fraction is at
least 0.10), plus the per-method and per-episode breakdowns.

``--resolve-only`` resolves and prints the task without starting Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

ARM_TASKS = {
    "blind": "Velocity-BHL-Biped-Ice-v0",
    "depth": "Velocity-BHL-Biped-Ice-Depth-v0",
    "visible": "Velocity-BHL-Biped-IceVisible-v0",
}
VISIBLE_RGB = (0.35, 0.72, 0.95)
FEET = ("leg_left_ankle_roll", "leg_right_ankle_roll")
EXPOSED_EPISODE_FRACTION = 0.5
EXPOSED_STEP_FRACTION = 0.10


# ----------------------------------------------------------------- params side
def load_params_yaml(path: Path) -> dict:
    """Isaac Lab dumps configs with python/tuple and python/object tags."""
    import yaml

    class _Loader(yaml.SafeLoader):
        pass

    def _py(loader, _suffix, node):
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        return loader.construct_scalar(node)

    _Loader.add_multi_constructor("tag:yaml.org,2002:python/", _py)
    with open(path) as fh:
        return yaml.load(fh, Loader=_Loader)


def resolve_task(run_dir: Path) -> dict:
    """Task id from what the run's env.yaml contains, checked against its name."""
    env = load_params_yaml(run_dir / "params" / "env.yaml")
    agent = load_params_yaml(run_dir / "params" / "agent.yaml")
    scene = env.get("scene", {})
    patches = sorted(k for k in scene if re.fullmatch(r"ice_\d+", str(k)))
    if not patches:
        raise SystemExit(f"{run_dir.name}: no ice_* patches in params/env.yaml -- not a B3 run")
    reset_ice = env.get("events", {}).get("reset_ice") or {}
    if "reset_ice_patches" not in str(reset_ice.get("func", "")):
        raise SystemExit(f"{run_dir.name}: no reset_ice_patches event -- pre-placement (72 m) run")
    rgb = tuple(float(x) for x in scene["ice_0"]["spawn"]["visual_material"]["diffuse_color"])
    if "depth_cam" in scene:
        arm, evidence = "depth", "scene.depth_cam present"
    elif all(abs(a - b) < 1e-3 for a, b in zip(rgb, VISIBLE_RGB)):
        arm, evidence = "visible", f"ice_0 diffuse {rgb} is VISIBLE_ICE_RGBA"
    else:
        arm, evidence = "blind", f"no depth_cam, ice_0 diffuse {rgb}"
    run_name = str(agent.get("run_name", ""))
    m = re.search(r"ppo-ice-placed-(blind|depth|visible)-s(\d+)", run_name)
    if m is None or m.group(1) != arm:
        raise SystemExit(f"{run_dir.name}: params say '{arm}' ({evidence}) but run_name is {run_name!r}")
    return {
        "task": ARM_TASKS[arm], "arm": arm, "seed": int(m.group(2)), "run_name": run_name,
        "evidence": evidence, "patches": patches, "env": env, "agent": agent,
    }


def trained_fingerprint(env: dict, patches: list[str]) -> dict:
    """The numbers that decide where the ice is and how long an episode is."""
    tg = env["scene"]["terrain"]["terrain_generator"]
    out = {
        "sim.dt": float(env["sim"]["dt"]),
        "decimation": int(env["decimation"]),
        "episode_length_s": float(env["episode_length_s"]),
        "terrain.num_rows": int(tg["num_rows"]),
        "terrain.num_cols": int(tg["num_cols"]),
        "terrain.size": [float(x) for x in tg["size"]],
        "terrain.sub_terrains": sorted(tg["sub_terrains"].keys()),
        "terrain.max_init_terrain_level": env["scene"]["terrain"].get("max_init_terrain_level"),
    }
    for p in patches:
        s = env["scene"][p]
        out[f"{p}.pos"] = [float(x) for x in s["init_state"]["pos"]]
        out[f"{p}.size"] = [float(x) for x in s["spawn"]["size"]]
        out[f"{p}.friction"] = [float(s["spawn"]["physics_material"]["static_friction"]),
                                float(s["spawn"]["physics_material"]["dynamic_friction"])]
    return out


def live_fingerprint(cfg, patches: list[str]) -> dict:
    tg = cfg.scene.terrain.terrain_generator
    out = {
        "sim.dt": float(cfg.sim.dt),
        "decimation": int(cfg.decimation),
        "episode_length_s": float(cfg.episode_length_s),
        "terrain.num_rows": int(tg.num_rows),
        "terrain.num_cols": int(tg.num_cols),
        "terrain.size": [float(x) for x in tg.size],
        "terrain.sub_terrains": sorted(tg.sub_terrains.keys()),
        "terrain.max_init_terrain_level": cfg.scene.terrain.max_init_terrain_level,
    }
    for p in patches:
        s = getattr(cfg.scene, p)
        out[f"{p}.pos"] = [float(x) for x in s.init_state.pos]
        out[f"{p}.size"] = [float(x) for x in s.spawn.size]
        out[f"{p}.friction"] = [float(s.spawn.physics_material.static_friction),
                                float(s.spawn.physics_material.dynamic_friction)]
    return out


def fingerprint_mismatches(trained: dict, live: dict) -> list[str]:
    bad = []
    for k, v in trained.items():
        w = live.get(k)
        if isinstance(v, list) and v and isinstance(v[0], float):
            ok = isinstance(w, list) and len(w) == len(v) and all(abs(a - b) < 1e-6 for a, b in zip(v, w))
        elif isinstance(v, float):
            ok = w is not None and abs(v - float(w)) < 1e-9
        else:
            ok = v == w
        if not ok:
            bad.append(f"{k}: trained {v!r} vs live {w!r}")
    return bad


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="a logs/rsl_rl/biped/*_ppo-ice-placed-* run directory")
    p.add_argument("--checkpoint", default="model_5999.pt", help="checkpoint file inside --run-dir")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="JSON output path (required unless --resolve-only)")
    p.add_argument("--force-threshold", type=float, default=1.0,
                   help="N; matches the scene contact_forces sensor's force_threshold")
    p.add_argument("--z-window", type=float, default=0.02,
                   help="m; geometric method: sole within this of the patch top")
    p.add_argument("--no-ice-contact-sensor", action="store_true",
                   help="do not add the filtered per-foot patch contact sensors (geometric fallback)")
    p.add_argument("--max-init-level", type=int, default=-1,
                   help="override scene.terrain.max_init_terrain_level; -1 keeps the trained value")
    p.add_argument("--resolve-only", action="store_true",
                   help="resolve the task from params and exit without starting Isaac Sim")
    return p


# ------------------------------------------------------------------ sim side
def run(args, app_args, resolved: dict) -> dict:
    from isaaclab.app import AppLauncher

    app_args.headless = True
    launcher = AppLauncher(app_args)
    _ = launcher.app

    import gymnasium as gym
    import torch

    from bhl_robust import compat as _bhl_compat
    _bhl_compat.apply()
    import berkeley_humanoid_lite.tasks  # noqa: F401
    import bhl_robust.tasks  # noqa: F401
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    task = resolved["task"]
    patches = resolved["patches"]
    ckpt = (args.run_dir / args.checkpoint).resolve(strict=True)

    cfg = gym.spec(task).kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    live_names = sorted(k for k in vars(cfg.scene) if re.fullmatch(r"ice_\d+", str(k)))
    mismatches = fingerprint_mismatches(trained_fingerprint(resolved["env"], patches),
                                        live_fingerprint(cfg, live_names))
    if live_names != patches:
        mismatches.append(f"patch names: trained {patches} vs live {live_names}")
    if args.max_init_level >= 0:
        cfg.scene.terrain.max_init_terrain_level = args.max_init_level

    # Filtered contact is one-to-many only, so one sensor per foot. Named
    # foot_ice_contact_*: reset_ice_patches treats every scene key that starts
    # with "ice_" as a patch and would try to teleport a sensor.
    use_contact = not args.no_ice_contact_sensor
    if use_contact:
        for foot in FEET:
            setattr(cfg.scene, f"foot_ice_contact_{foot}", ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/robot/{foot}",
                update_period=0.0,
                history_length=1,
                track_air_time=False,
                force_threshold=args.force_threshold,
                filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/{p}" for p in patches],
            ))

    env = gym.make(task, cfg=cfg, disable_env_checker=True)
    u = env.unwrapped
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    try:  # same guarded migration as train_play.py; absent on v51
        from importlib.metadata import version as _pkg_version

        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
    except ImportError:
        pass
    trained_dims = resolved["agent"].get("policy", {}).get("actor_hidden_dims")
    live_dims = list(getattr(getattr(agent_cfg, "policy", None), "actor_hidden_dims", []) or [])
    if trained_dims is not None and list(trained_dims) != live_dims:
        mismatches.append(f"actor_hidden_dims: trained {trained_dims} vs live {live_dims}")

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=u.device)

    dev = u.device
    n = u.num_envs
    robot = u.scene["robot"]
    rfoot_ids, rfoot_names = robot.find_bodies(list(FEET), preserve_order=True)
    cs = u.scene["contact_forces"]
    cfoot_ids, cfoot_names = cs.find_bodies(list(FEET), preserve_order=True)
    assets = [u.scene[p] for p in patches]
    half = torch.tensor([float(getattr(cfg.scene, p).spawn.size[0]) / 2.0 for p in patches], device=dev)
    half_y = torch.tensor([float(getattr(cfg.scene, p).spawn.size[1]) / 2.0 for p in patches], device=dev)
    thick = torch.tensor([float(getattr(cfg.scene, p).spawn.size[2]) for p in patches], device=dev)

    contact_sensors, contact_note = [], "disabled by --no-ice-contact-sensor"
    if use_contact:
        contact_sensors = [u.scene[f"foot_ice_contact_{f}"] for f in FEET]
        counts = [int(s.contact_physx_view.filter_count) for s in contact_sensors]
        if all(c == len(patches) for c in counts):
            contact_note = f"filtered sensors built, filter_count {counts}"
        else:
            contact_note = f"filter_count {counts} != {len(patches)} patches; contact method disabled"
            contact_sensors = []
    method = "contact" if contact_sensors else "geometric"

    def body_pos():
        d = robot.data
        return d.body_link_pos_w if hasattr(d, "body_link_pos_w") else d.body_pos_w

    def patch_state():
        pos = torch.stack([a.data.root_pos_w for a in assets], dim=1)  # (N, P, 3)
        return pos, pos[..., 2] + thick / 2.0

    # --------------------------------------------------------- episode start
    _obs = wrapped.get_observations()
    obs = _obs[0] if isinstance(_obs, tuple) else _obs
    u.episode_length_buf[:] = 0
    origins0 = u.scene.env_origins.clone()
    levels0 = u.scene.terrain.terrain_levels.clone() if hasattr(u.scene.terrain, "terrain_levels") \
        else torch.full((n,), -1, device=dev)
    types0 = u.scene.terrain.terrain_types.clone() if hasattr(u.scene.terrain, "terrain_types") \
        else torch.full((n,), -1, device=dev)
    root0 = robot.data.root_pos_w[:, :3].clone()
    ppos0, ptop0 = patch_state()
    offsets = torch.tensor([list(getattr(cfg.scene, p).init_state.pos) for p in patches], device=dev)
    placement_err = (ppos0 - (origins0[:, None, :] + offsets[None])).norm(dim=-1)  # (N, P)
    d_xy = (root0[:, None, :2] - ppos0[..., :2]).abs()
    edge = torch.stack([(d_xy[..., 0] - half).clamp(min=0), (d_xy[..., 1] - half_y).clamp(min=0)], -1)
    spawn_edge_dist = edge.norm(dim=-1).min(dim=1).values
    standing_h = (body_pos()[:, rfoot_ids, 2] - origins0[:, None, 2]).clone()  # (N, 2)

    # ------------------------------------------------ per-physics-step hook
    rec: dict[str, list] = {k: [] for k in (
        "alive", "over", "dz", "h_ground", "net_contact", "ice_f", "ice_net")}
    state = {"armed": False, "calls": 0, "alive": None, "root_xy": root0[:, :2].clone()}
    orig_update = u.scene.update

    def hooked_update(dt):
        orig_update(dt)
        if not state["armed"]:
            return
        state["calls"] += 1
        feet = body_pos()[:, rfoot_ids, :]  # (N, 2, 3)
        ppos, ptop = patch_state()
        rel = (feet[:, :, None, :2] - ppos[:, None, :, :2]).abs()  # (N, 2, P, 2)
        inside = (rel[..., 0] <= half) & (rel[..., 1] <= half_y)  # (N, 2, P)
        dz = feet[:, :, 2:3] - ptop[:, None, :]  # (N, 2, P)
        dz_in = torch.where(inside, dz, torch.full_like(dz, math.inf)).min(dim=-1).values
        net = cs.data.net_forces_w[:, cfoot_ids, :].norm(dim=-1)  # (N, 2)
        rec["alive"].append(state["alive"])
        rec["over"].append(inside.any(dim=-1))
        rec["dz"].append(dz_in)
        rec["h_ground"].append(feet[:, :, 2] - u.scene.env_origins[:, None, 2])
        rec["net_contact"].append(net > args.force_threshold)
        if contact_sensors:
            rec["ice_f"].append(torch.stack(
                [s.data.force_matrix_w[:, 0].norm(dim=-1).sum(dim=-1) for s in contact_sensors], dim=1))
            rec["ice_net"].append(torch.stack(
                [s.data.net_forces_w[:, 0].norm(dim=-1) for s in contact_sensors], dim=1))
        state["root_xy"] = robot.data.root_pos_w[:, :2].clone()

    u.scene.update = hooked_update

    # ------------------------------------------------------------- rollout
    T = int(u.max_episode_length)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    ep_steps = torch.zeros(n, dtype=torch.long, device=dev)
    timed_out = torch.zeros(n, dtype=torch.bool, device=dev)
    final_xy = torch.zeros(n, 2, device=dev)
    act_abs, fwd_speed, cmd_speed, policy_steps = 0.0, 0.0, 0.0, 0
    t0 = time.time()
    with torch.inference_mode():
        for _ in range(T + 5):
            if not bool(alive.any()):
                break
            actions = policy(obs)
            act_abs += float(actions[alive].abs().mean())
            cmd = u.command_manager.get_command("base_velocity")
            cmd_speed += float(cmd[alive, :2].norm(dim=-1).mean())
            state["alive"] = alive.clone()
            state["armed"] = True
            obs, _, dones, extras = wrapped.step(actions)
            state["armed"] = False
            policy_steps += 1
            fwd_speed += float(robot.data.root_lin_vel_b[alive, 0].mean()) if bool(alive.any()) else 0.0
            ep_steps += alive.long()
            done = dones.bool() & alive
            if bool(done.any()):
                to = extras.get("time_outs")
                if to is not None:
                    timed_out |= done & to.bool()
                final_xy[done] = state["root_xy"][done]
                alive &= ~done
    wall = time.time() - t0
    closed = ~alive
    final_xy[alive] = state["root_xy"][alive]

    # ------------------------------------------------------------ reduce
    A = torch.stack(rec["alive"])  # (S, N)
    S = A.shape[0]
    over = torch.stack(rec["over"])  # (S, N, 2)
    dz = torch.stack(rec["dz"])
    hg = torch.stack(rec["h_ground"])
    netc = torch.stack(rec["net_contact"])
    A2 = A[..., None].expand_as(over)

    calib = hg[A2 & netc & ~over]
    calib_src = "median foot-body height over ground, in contact off-ice"
    if calib.numel() < 50:
        calib = standing_h.flatten()
        calib_src = "foot-body height over ground at reset (too few off-ice contact samples)"
    sole_off = float(calib.median())

    on = {
        "geometric": over & ((dz - sole_off) <= args.z_window),
        "hybrid": over & netc,
        "over_xy": over,
    }
    ice_share = None
    ice_dz = None
    if contact_sensors:
        ice_f = torch.stack(rec["ice_f"])
        ice_net = torch.stack(rec["ice_net"])
        on["contact"] = ice_f > args.force_threshold
        m = A2 & on["contact"]
        if bool(m.any()):
            ice_share = float((ice_f[m] / ice_net[m].clamp(min=1e-6)).clamp(max=1.0).mean())
            dzc = dz[m]
            dzc = dzc[torch.isfinite(dzc)]
            ice_dz = float(dzc.median()) if dzc.numel() else None

    n_sub = A.sum(dim=0).clamp(min=1).float()  # (N,)
    per_method = {}
    per_env_any, per_env_frac = {}, {}
    for name, flag in on.items():
        step_on = (flag.any(dim=-1) & A)  # (S, N)
        any_ep = step_on.any(dim=0)
        frac = step_on.sum(dim=0).float() / n_sub
        per_env_any[name], per_env_frac[name] = any_ep, frac
        per_method[name] = {
            "episodes_with_any_ice_contact_fraction": float(any_ep.float().mean()),
            "mean_on_ice_step_fraction": float(frac.mean()),
            "median_on_ice_step_fraction": float(frac.median()),
        }
    agree = None
    if contact_sensors:
        c, g = on["contact"].any(-1), on["geometric"].any(-1)
        agree = float(((c == g) & A).sum() / A.sum().clamp(min=1))

    prim_any, prim_frac = per_env_any[method], per_env_frac[method]
    ep_frac_any = float(prim_any.float().mean())
    mean_frac = float(prim_frac.mean())
    verdict = ("EXPOSED" if ep_frac_any >= EXPOSED_EPISODE_FRACTION and mean_frac >= EXPOSED_STEP_FRACTION
               else "NOT_EXPOSED")

    by_level = {}
    for lv in sorted(set(int(x) for x in levels0.tolist())):
        sel = levels0 == lv
        by_level[str(lv)] = {
            "n": int(sel.sum()),
            "episodes_with_any_ice_contact_fraction": float(prim_any[sel].float().mean()),
            "mean_on_ice_step_fraction": float(prim_frac[sel].mean()),
        }

    walked = (final_xy - root0[:, :2]).norm(dim=-1)
    episodes = []
    for e in range(n):
        episodes.append({
            "env_id": e,
            "terrain_level": int(levels0[e]),
            "terrain_type": int(types0[e]),
            "env_origin": [round(float(x), 3) for x in origins0[e].tolist()],
            "policy_steps": int(ep_steps[e]),
            "physics_steps": int(A[:, e].sum()),
            "closed": bool(closed[e]),
            "timed_out": bool(timed_out[e]),
            "spawn_dist_to_nearest_patch_m": round(float(spawn_edge_dist[e]), 3),
            "distance_walked_m": round(float(walked[e]), 3),
            "any_ice_contact": bool(prim_any[e]),
            "on_ice_step_fraction": round(float(prim_frac[e]), 4),
            **{f"{k}_any": bool(per_env_any[k][e]) for k in on},
            **{f"{k}_frac": round(float(per_env_frac[k][e]), 4) for k in on},
        })

    decim = int(cfg.decimation)
    return {
        "status": "ok",
        "task": task,
        "arm": resolved["arm"],
        "seed_of_run": resolved["seed"],
        "run_dir": str(args.run_dir.resolve()),
        "run_name": resolved["run_name"],
        "task_evidence": resolved["evidence"],
        "checkpoint": str(ckpt),
        "stack": os.environ.get("BHL_STACK", "unknown"),
        "num_envs": n,
        "n_episodes": n,
        "episodes_closed": int(closed.sum()),
        "episodes_timed_out": int(timed_out.sum()),
        "max_episode_length_policy_steps": T,
        "decimation": decim,
        "physics_dt": float(u.physics_dt),
        "policy_steps_run": policy_steps,
        "physics_steps_recorded": S,
        "hook_calls": state["calls"],
        "hook_ok": state["calls"] == policy_steps * decim == S,
        "method": method,
        "contact_sensor_note": contact_note,
        "force_threshold_N": args.force_threshold,
        "z_window_m": args.z_window,
        "sole_offset_m": sole_off,
        "sole_offset_source": calib_src,
        "sole_offset_samples": int(calib.numel()),
        "median_foot_body_minus_patch_top_in_ice_contact_m": ice_dz,
        "mean_ice_share_of_foot_force_in_ice_contact": ice_share,
        "contact_vs_geometric_step_agreement": agree,
        "episodes_with_any_ice_contact_fraction": ep_frac_any,
        "mean_on_ice_step_fraction": mean_frac,
        "verdict": verdict,
        "verdict_rule": (f"EXPOSED iff episodes_with_any_ice_contact_fraction >= {EXPOSED_EPISODE_FRACTION} "
                         f"and mean_on_ice_step_fraction >= {EXPOSED_STEP_FRACTION} ({method} method)"),
        "methods": per_method,
        "by_terrain_level": by_level,
        "terrain_level_note": ("ICE_TERRAINS_CFG tiles are all zero-slope, so the level moves the tile "
                               "(env_origins) but not the geometry; max_init_terrain_level "
                               f"{cfg.scene.terrain.max_init_terrain_level}"),
        "params_match": not mismatches,
        "params_mismatches": mismatches,
        "patch_placement_max_err_m": float(placement_err.max()),
        "median_spawn_dist_to_nearest_patch_m": float(spawn_edge_dist.median()),
        "policy_acting": {
            "mean_abs_action": act_abs / max(policy_steps, 1),
            "mean_forward_speed_mps": fwd_speed / max(policy_steps, 1),
            "mean_command_speed_mps": cmd_speed / max(policy_steps, 1),
            "median_distance_walked_m": float(walked.median()),
        },
        "foot_bodies_robot": list(rfoot_names),
        "foot_bodies_contact_sensor": list(cfoot_names),
        "rollout_wall_s": round(wall, 1),
        "episodes": episodes,
    }


def main() -> None:
    parser = build_parser()
    args, rest = parser.parse_known_args()
    resolved = resolve_task(args.run_dir)
    if args.resolve_only:
        print(json.dumps({k: resolved[k] for k in ("task", "arm", "seed", "run_name", "evidence", "patches")},
                         indent=2))
        return
    if args.out is None:
        parser.error("--out is required unless --resolve-only")
    from isaaclab.app import AppLauncher

    app_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(app_parser)
    app_args = app_parser.parse_args(rest)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ice-exposure] run {args.run_dir.name} -> task {resolved['task']} ({resolved['evidence']})", flush=True)
    try:
        result = run(args, app_args, resolved)
        code = 0
    except BaseException as exc:  # noqa: BLE001 -- the JSON must say why
        result = {"status": "error", "task": resolved["task"], "arm": resolved["arm"],
                  "run_dir": str(args.run_dir), "error": repr(exc), "traceback": traceback.format_exc()}
        code = 1
        print(result["traceback"], file=sys.stderr, flush=True)
    args.out.write_text(json.dumps(result, indent=2))
    if result["status"] == "ok":
        print(f"ICE-EXPOSURE-RESULT arm={result['arm']} task={result['task']} method={result['method']} "
              f"n_episodes={result['n_episodes']} closed={result['episodes_closed']} "
              f"any_contact_frac={result['episodes_with_any_ice_contact_fraction']:.3f} "
              f"mean_on_ice_frac={result['mean_on_ice_step_fraction']:.4f} verdict={result['verdict']} "
              f"params_match={result['params_match']} hook_ok={result['hook_ok']}", flush=True)
    else:
        print(f"ICE-EXPOSURE-RESULT arm={result['arm']} status=error {result['error']}", flush=True)
    # Isaac Sim's shutdown hard-exits and can hang; the JSON is already on disk.
    os._exit(code)


if __name__ == "__main__":
    main()
