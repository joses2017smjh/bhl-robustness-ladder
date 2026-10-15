"""Actual MuJoCo transitions, plus paired frozen-maze governor experiments.

Maze hooks are compiled from the campaign's immutable source at runtime with
exact checked insertion points. No historical repository or snapshot is edited.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from common import ROOT, UPSTREAM, digest, load_controller, setup, validate, write_new
from world_model import Governor, METHODS, MODEL_METHODS, SCHEMA, StateHistory, proprio
from navigation_recovery import NavigationRecovery, RecoveryGovernor, MODES as RECOVERY_MODES, SETTINGS as RECOVERY_SETTINGS


def deploy_path(campaign, gait_name, training_seed):
    return Path(campaign) / "models" / f"{gait_name}-s{training_seed}" / "deploy.yaml"


def yaw_of_quaternion(q):
    w,x,y,z = q
    return float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))


def local_displacement(before_xy, after_xy, before_yaw):
    delta = np.asarray(after_xy)-np.asarray(before_xy)
    c,s = np.cos(before_yaw),np.sin(before_yaw)
    return np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]],dtype=np.float32)


def excitation(step, dt, episode_index, rng, cache):
    """Prospective command coverage of chatter, sustained turns and stops."""
    now = step*dt
    if now < 1.:
        return np.zeros(3)
    mode = episode_index % 4
    if mode == 0:
        block = int((now-1.)/.6)
        if block != cache.get("block"):
            cache.update(block=block, cmd=np.array([rng.choice([.075,.15,.2,.25,.3,.4,.45]), 0., rng.uniform(-.7,.7)]))
        return cache["cmd"].copy()
    if mode == 1:
        # Navigation-like rapid full-turn reversals, then recovery segments.
        period = (.08, .16, .24)[(int((now-1.)/2.) % 3)]
        return np.array([.3, 0., (-1 if int((now-1.)/period)%2 else 1)*1.])
    if mode == 2:
        # Stale-sensor-like immediate stops and restarts.
        phase = int((now-1.)/.8) % 4
        return np.zeros(3) if phase in (1, 3) else np.array([.35, 0., .6 if phase == 0 else -.6])
    phase = int((now-1.)/1.5) % 4
    # Sustained competent walking/turning coverage, rather than spending turn
    # phases entirely inside the low-level gait's measured speed deadband.
    return np.array([.3, 0., (.0, .3, .0, -.3)[phase]])


def collect(args):
    campaign = setup(args.campaign)
    from bhl_robust.eval.harness import EvalConfig
    from bhl_robust.eval.mjcf_assets import prepare_mjcf
    from terrain_suite import TERRAINS, TerrainEnv, prepare_terrain
    from adaptive_command import AdaptiveCommand
    cfg, controller = load_controller(deploy_path(campaign, args.gait_name, args.training_seed))
    if args.out.exists() or args.out.with_suffix(".json").exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    qualification = args.qualification or campaign/'qualification.json'
    if args.behavior_policy!='raw' and not qualification.is_file():
        raise FileNotFoundError('Reference-behavior collection requires gait qualification: '+str(qualification))
    qualification_data = json.loads(qualification.read_text()) if qualification.is_file() else {}
    provenance_paths={'wrapper_sha256':Path(__file__),
        'world_model_sha256':Path(__file__).with_name('world_model.py'),
        'adaptive_source_sha256':Path(__file__).with_name('adaptive_command.py')}
    if qualification.is_file():provenance_paths['qualification_sha256']=qualification
    provenance={key:digest(path) for key,path in provenance_paths.items()}
    protocol_path=campaign/'protocol.json'
    protocol_hash=digest(protocol_path) if protocol_path.is_file() else None
    rows = {key: [] for key in ("state", "command", "requested_command", "behavior", "next_core", "velocity", "displacement", "sink", "event", "seed", "episode")}
    episode_rows = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="bhl-dynamics-") as tmp:
        base = prepare_mjcf(UPSTREAM, Path(tmp)/"base", args.variant, terrain=False)
        for ti, terrain in enumerate(("flat",)+tuple(TERRAINS)):
            for ei in range(args.episodes_per_terrain):
                reset_seed = args.seed_start+ei
                terrain_seed = 10000+reset_seed
                directory = Path(tmp)/f"{terrain}-{ei}"
                scene = base if terrain == "flat" else prepare_terrain(base, directory, terrain, args.severity, terrain_seed)
                env = TerrainEnv(cfg, scene)
                rng = np.random.default_rng(reset_seed)
                obs = env.reset(rng, EvalConfig())
                controller.prev_actions[:] = 0.
                controller.policy_observations[:] = 0.
                history = StateHistory()
                block = ei//4
                adaptive_behavior = bool(block%2) if args.behavior_policy=='mixed' else args.behavior_policy=='adaptive'
                pushed = bool((block//2)%2) if args.behavior_policy=='mixed' else bool(block%2)
                reference = AdaptiveCommand(qualification=qualification_data,gait_name=args.gait_name,
                                             training_seed=args.training_seed)
                ep = {key: [] for key in ("state", "command", "requested_command", "behavior", "next_core", "velocity", "displacement", "tilt", "sink", "event")}
                command_rng = np.random.default_rng(30000+reset_seed)
                cache, fell, pushes = {}, False, 0
                for step in range(int(args.seconds/float(cfg.policy_dt))):
                    dt = float(cfg.policy_dt)
                    desired = excitation(step, dt, ei, command_rng, cache)
                    # Pushes are prospective and balanced by reset-seed, not selected from evaluation failures.
                    if pushed and step*dt > 1. and step%max(1, int(3./dt)) == 0:
                        env.push(args.push_speed, rng)
                        pushes += 1
                    obs = env.robot_observations(desired)
                    requested = desired.copy()
                    if adaptive_behavior:
                        desired = reference.filter(requested,obs,controller.prev_actions.copy(),dt)
                        obs[-3:] = desired
                    core = proprio(obs, controller.prev_actions)
                    state = history.state(core)
                    before_xy = env.base_xy.copy()
                    before_yaw = yaw_of_quaternion(env.mj_data.qpos[3:7])
                    target = controller.update(obs)
                    if not np.isfinite(target).all():
                        raise RuntimeError("nonfinite low-level policy action during dynamics collection")
                    env.step(target)
                    next_core = proprio(env.robot_observations(desired), controller.prev_actions)
                    if not np.isfinite(next_core).all():
                        raise RuntimeError("nonfinite physical state during dynamics collection")
                    fell = env.tilt_rad >= .78 or env.sink_m > .25
                    after_yaw = yaw_of_quaternion(env.mj_data.qpos[3:7])
                    yaw_rate = ((after_yaw-before_yaw+np.pi)%(2*np.pi)-np.pi)/dt
                    velocity = np.concatenate([env.base_lin_vel_yaw_frame(),[yaw_rate]])
                    displacement = local_displacement(before_xy,env.base_xy,before_yaw)
                    for key, value in (("state", state), ("command", desired.copy()), ("next_core", next_core),
                                       ("requested_command",requested),("behavior",int(adaptive_behavior)),
                                       ("tilt", env.tilt_rad), ("sink", env.sink_m), ("velocity",velocity),
                                       ("displacement",displacement),("event",float(fell))):
                        ep[key].append(value)
                    history.advance(core, desired)
                    if fell:
                        break
                transitions = len(ep["state"])
                episode_number = ti*args.episodes_per_terrain+ei
                for key in ("state", "command", "requested_command", "behavior", "next_core", "velocity", "displacement", "sink", "event"):
                    rows[key].extend(ep[key])
                rows["seed"].extend([reset_seed]*transitions)
                rows["episode"].extend([episode_number]*transitions)
                row = {"episode": episode_number, "terrain": terrain, "terrain_seed": terrain_seed,
                       "reset_seed": reset_seed, "training_seed": args.training_seed, "command_regime": ei%4,
                       "push_regime": pushed,"behavior_source":"adaptive" if adaptive_behavior else "raw",
                       "steps": transitions, "fell": bool(fell), "pushes": pushes,
                       "distance_x_m": float(env.base_xy[0]), "max_tilt_rad": float(max(ep["tilt"])),
                       "immediate_event_rows": int(sum(ep["event"])),
                       "mean_body_vx_m_s": float(np.mean(np.asarray(ep["velocity"])[:,0]))}
                episode_rows.append(row)
                print(json.dumps(row), flush=True)
    metadata = {"schema": SCHEMA, "gait_name": args.gait_name, "training_seed": args.training_seed,
                "deploy": str(deploy_path(campaign, args.gait_name, args.training_seed)),
                "checkpoint_sha256": digest(cfg.policy_checkpoint_path), "dt": float(cfg.policy_dt),
                "variant": args.variant, "terrain_types": ["flat"]+list(TERRAINS), "severity": args.severity,
                "collection_seed_start": args.seed_start,
                "collection_push_block_by_four":args.behavior_policy!='mixed',
                "collection_push_block_episodes":8 if args.behavior_policy=='mixed' else 4,
                "reference_mix_by_four_blocks":args.behavior_policy=='mixed',
                "behavior_policy":args.behavior_policy,
                **provenance,"protocol_sha256":protocol_hash,
                "qualification_sha256":provenance.get('qualification_sha256'),
                "collection_source_stability_checked":True,
                "episodes": len(episode_rows), "transitions": len(rows["state"]), "wall_s": time.monotonic()-started,
                "risk_label": "immediate NEXT state termination from tilt>=0.78 or local_sink>0.25; no future behavior-policy severity label",
                "proprio_input": "IMU projected gravity/gyro, all acted joint positions/velocities, previous gait action and proprio/command history; no terrain ID",
                "label_privilege": "Simulator yaw-frame velocity, step-local displacement, tilt and local-terrain sink for supervised targets only; truth IMU sensors are simulated. None are added to model input."}
    for key,path in provenance_paths.items():
        if digest(path)!=provenance[key]:
            raise RuntimeError('Collection source/provenance changed during rollout: '+str(path))
    if protocol_hash is not None and digest(protocol_path)!=protocol_hash:
        raise RuntimeError('Campaign protocol changed during collection')
    arrays = {k: np.asarray(v, dtype=np.int64 if k in ("seed", "episode", "behavior") else np.float32) for k, v in rows.items()}
    # Opening in exclusive mode prevents concurrent workers overwriting transition evidence.
    with args.out.open("xb") as f:
        np.savez_compressed(f, **arrays, metadata=np.array(json.dumps(metadata)))
    write_new(args.out.with_suffix(".json"), {"metadata": metadata, "episodes": episode_rows})
    print(json.dumps({"dataset": str(args.out), "transitions": len(rows["state"]),
                      "falls": sum(e["fell"] for e in episode_rows)}), flush=True)
    return 0


def patched_run_seed(maze, governor_kind, checkpoint, qualification=None,gait_name=None,training_seed=None,consumed_development=False,recovery_mode="none"):
    source = inspect.getsource(maze.run_seed)
    replacements = [
        ("        # plan on a timer,", "        recovery.update(now, xy_e, settle_s=args.settle_s)\n        # plan on a timer,"),
        ("if learned is None and (plan is None or now - last_plan_t >= args.replan_s):", "if (learned is None or recovery.active) and (plan is None or now - last_plan_t >= args.replan_s):"),
        ("            new_plan = planner.plan(xy_e, goal)\n", "            new_plan = planner.plan(xy_e, goal)\n            recovery.record_plan(bool(new_plan))\n            if recovery.active and not new_plan:\n                plan = None  # do not pursue an obsolete blocked route\n"),
        ("        if learned is not None:\n            # the gym", "        if learned is not None and not recovery.active:\n            # the gym"),
        ("arrived_last = learned is None and state == \"arrived\" and is_goal", "arrived_last = (learned is None or recovery.active) and state == \"arrived\" and is_goal"),
        ("    dt = float(cfg.policy_dt)\n", "    dt = float(cfg.policy_dt)\n    recovery = _NavigationRecovery(_upgrade_recovery_mode, time_limit_s=args.time_limit)\n    governor = _RecoveryGovernor(_UpgradeGovernor(_upgrade_kind, _upgrade_checkpoint, qualification=_upgrade_qualification, gait_name=_upgrade_gait_name, training_seed=_upgrade_training_seed, consumed_development=_upgrade_consumed_development), recovery, _upgrade_model_methods)\n"),
        ("            cmd_stats[\"actor\"].update(actor_cmd[2])\n", "            if not recovery.active:\n                cmd_stats[\"actor\"].update(actor_cmd[2])\n"),
        ("                trace[-1][\"cmd_actor\"] = [round(float(c), 3) for c in actor_cmd]\n", "                trace[-1][\"cmd_navigator\"] = [round(float(c), 3) for c in actor_cmd]\n                trace[-1][\"cmd_actor\"] = None if recovery.active else trace[-1][\"cmd_navigator\"]\n"),
        ("            cmd_stats[\"gait\"].update(command[2])\n", "            pass  # executed gait command counted after the governor hook\n"),
        ("        target = controller.update(obs)\n", "        command = governor.filter(command, obs, controller.prev_actions.copy(), dt)\n        obs[-3:] = command\n        if cmd_stats is not None and now >= args.settle_s:\n            cmd_stats['gait'].update(command[2])\n        target = controller.update(obs)\n"),
        ("            if pose.active:\n", "            trace[-1]['governor'] = governor.last_info.copy()\n            trace[-1]['navigation_recovery'] = recovery.last_info.copy()\n            if pose.active:\n"),
        ("        runner.step([target])\n", "        before_position = runner.d.qpos[slot.qpos_adr:slot.qpos_adr+2].copy()\n        before_yaw = yaw\n        runner.step([target])\n        q_after = runner.d.qpos[slot.qpos_adr+3:slot.qpos_adr+7]\n        yaw_after = yaw_of(q_after)\n        v_world = runner.d.qvel[slot.qvel_adr:slot.qvel_adr+2]\n        c_after,s_after = np.cos(yaw_after),np.sin(yaw_after)\n        yaw_rate = ((yaw_after-before_yaw+np.pi)%(2*np.pi)-np.pi)/dt\n        v_body = np.array([c_after*v_world[0]+s_after*v_world[1],-s_after*v_world[0]+c_after*v_world[1],yaw_rate])\n        achieved_displacement = _upgrade_local_displacement(before_position,runner.d.qpos[slot.qpos_adr:slot.qpos_adr+2],before_yaw)\n        governor.observe_transition(achieved_velocity=v_body, achieved_displacement=achieved_displacement)\n"),
        ("    return result\n", "    result['navigation_recovery'] = recovery.summary()\n    result['control'] += '; navigation recovery mode=' + _upgrade_recovery_mode + ', with planner commands distinguished from learned actor commands'\n    result['governor'] = governor.summary()\n    result['governor']['maze_source_sha256'] = _upgrade_source_sha256\n    result['governor']['sim_state_access'] = 'High-level actor and recovery retain existing oracle pose/goal and lidar occupancy; low-level governor sees only upstream robot observation and action history, including truth simulated IMU'\n    return result\n")]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError(f"immutable maze source differs at required hook: {before!r}")
        source = source.replace(before, after)
    namespace = dict(vars(maze))
    namespace.update(_UpgradeGovernor=Governor, _NavigationRecovery=NavigationRecovery, _RecoveryGovernor=RecoveryGovernor,
                     _upgrade_recovery_mode=recovery_mode, _upgrade_model_methods=MODEL_METHODS, _upgrade_kind=governor_kind,
                     _upgrade_checkpoint=checkpoint, _upgrade_source_sha256=digest(maze.__file__),
                     _upgrade_qualification=qualification,_upgrade_gait_name=gait_name,
                     _upgrade_training_seed=training_seed,_upgrade_local_displacement=local_displacement,
                     _upgrade_consumed_development=consumed_development)
    exec(compile(source, str(Path(__file__).resolve())+"::run_seed_hooks", "exec"), namespace)
    return namespace["run_seed"]


def evaluate(args):
    campaign = setup(args.campaign)
    import maze_explore as maze
    if args.governor in MODEL_METHODS:
        checkpoint = args.checkpoint or campaign/"worldmodels"/f"{args.gait_name}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        fitted = Governor("ensemble", checkpoint,consumed_development=args.consumed_development).model
        gait_hash = digest(deploy_path(campaign, args.gait_name, args.training_seed).parent/"policy.onnx")
        if fitted.gait_name != args.gait_name or gait_hash not in fitted.policy_sha256s:
            raise ValueError("world model provenance does not include the evaluated frozen gait checkpoint")
    else:
        checkpoint = None
    kind = "baseline" if args.governor in ("baseline", "yaw_lpf") else args.governor
    qualification = args.qualification or campaign/"qualification.json"
    if (args.governor in MODEL_METHODS or args.governor in ('adaptive','adaptive_curvature')) and not qualification.is_file():
        raise FileNotFoundError(qualification)
    if args.governor in MODEL_METHODS or args.governor in ('adaptive','adaptive_curvature'):
        policy_qualification=json.loads(qualification.read_text()).get('policies',{}).get(f'{args.gait_name}-s{args.training_seed}',{})
        expected_hash=digest(deploy_path(campaign,args.gait_name,args.training_seed).parent/'policy.onnx')
        if policy_qualification.get('policy_sha256')!=expected_hash:
            raise ValueError('Qualification provenance does not match the evaluated gait checkpoint')
    maze.run_seed = patched_run_seed(maze, kind, checkpoint, qualification if qualification.is_file() else None,args.gait_name,args.training_seed,args.consumed_development,args.recovery_mode)
    gait = deploy_path(campaign, args.gait_name, args.training_seed)
    command = ["maze_explore.py", "--upstream", str(UPSTREAM), "--gait", str(gait),
               "--seeds", str(args.seeds), "--seed-start", str(args.seed_start),
               "--n", "6", "--m", "6", "--extra-openings", "1", "--time-limit", str(args.time_limit),
               "--sensor-mode", args.sensor_mode, "--dropout-probability", ".35",
               "--no-overwrite", "--out-dir", str(args.out_dir)]
    if args.navigator != "astar":
        actor = campaign/"models"/args.navigator/"actor.onnx"
        command += ["--policy", str(actor), "--policy-capture-pose", "--policy-cmd-filter",
                    "wz-lpf" if args.governor == "yaw_lpf" else "none"]
    elif args.governor == "yaw_lpf":
        raise ValueError("existing yaw_lpf ablation applies to learned navigator only")
    if args.variant != "biped":
        command += ["--variant", args.variant]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_new(args.out_dir/"upgrade_receipt.json", {
        "protocol_sha256": digest(campaign/"protocol.json") if (campaign/"protocol.json").exists() else None,
        "gait_name": args.gait_name, "training_seed": args.training_seed, "navigator": args.navigator,
        "recovery_mode": args.recovery_mode, "recovery_settings": RECOVERY_SETTINGS,
        "recovery_source_sha256": digest(Path(__file__).with_name("navigation_recovery.py")),
        "governor": args.governor, "gait_sha256": digest(gait.parent/"policy.onnx"),
        "consumed_development":args.consumed_development,
        "model_sha256": digest(checkpoint) if checkpoint else None, "maze_source_sha256": digest(maze.__file__),
        "qualification_sha256":digest(qualification) if qualification.is_file() else None,
        "wrapper_sha256": digest(__file__), "world_model_sha256": digest(Path(__file__).with_name("world_model.py")),
        "adaptive_source_sha256":digest(Path(__file__).with_name("adaptive_command.py")),
        "command": command, "reset_seeds": list(range(args.seed_start, args.seed_start+args.seeds)),
        "slurm_job_id": os.getenv("SLURM_JOB_ID")})
    with tempfile.TemporaryDirectory(prefix="bhl-nav-upgrade-") as cache:
        command += ["--cache-dir", cache]
        previous_argv = sys.argv
        sys.argv = command
        try:
            result = maze.main()
        finally:
            sys.argv = previous_argv
    write_new(args.out_dir/"upgrade_process.json", {"returncode": result, "completed": result == 0})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    collect_parser = subs.add_parser("collect")
    eval_parser = subs.add_parser("eval", aliases=["evaluate"])
    for p in (collect_parser, eval_parser):
        p.add_argument("--campaign", type=Path, default=ROOT)
        p.add_argument("--gait-name", choices=("dr-default", "dr-s1.5"))
        p.add_argument("--training-seed", type=int, choices=(0, 1, 2))
        p.add_argument("--seed-start", type=int)
        p.add_argument("--variant", choices=("biped", "humanoid"), default="biped")
    collect_parser.add_argument("--collection-cell", type=int)
    collect_parser.add_argument("--out", type=Path)
    collect_parser.add_argument("--episodes-per-terrain", type=int, default=8)
    collect_parser.add_argument("--seconds", type=float, default=12.)
    collect_parser.add_argument("--severity", type=float, default=.5)
    collect_parser.add_argument("--push-speed", type=float, default=.5)
    collect_parser.add_argument('--behavior-policy',choices=('mixed','raw','adaptive'),default='mixed')
    collect_parser.add_argument('--qualification',type=Path)
    eval_parser.add_argument("--cell", "--evaluate-cell", dest="cell",type=int)
    eval_parser.add_argument("--navigator", choices=("astar", "armV5-s8", "armV5-s9", "armV5-s10"))
    eval_parser.add_argument("--governor", choices=METHODS+("yaw_lpf",))
    eval_parser.add_argument("--recovery-mode", choices=RECOVERY_MODES, default="none")
    eval_parser.add_argument("--checkpoint", type=Path)
    eval_parser.add_argument("--qualification", type=Path)
    eval_parser.add_argument('--consumed-development',action='store_true')
    eval_parser.add_argument("--out-dir", type=Path)
    eval_parser.add_argument("--seeds", type=int, default=24)
    eval_parser.add_argument("--time-limit", type=float, default=180.)
    eval_parser.add_argument("--sensor-mode", choices=("reactive", "reactive_dropout"), default="reactive")
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    protocol = validate(campaign) if (campaign/"protocol.json").exists() else None
    if args.mode == "collect" and args.collection_cell is not None:
        settings = protocol["data_collection"] if protocol else {}
        seeds = settings.get("policy_training_seeds",[0,1,2])
        if protocol is None or not 0 <= args.collection_cell < len(settings["gaits"])*len(seeds):
            parser.error("--collection-cell requires frozen protocol and valid collection index")
        settings = protocol["data_collection"]
        args.gait_name = settings["gaits"][args.collection_cell//len(seeds)]
        args.training_seed = seeds[args.collection_cell%len(seeds)]
        args.seed_start = settings["seed_starts"][f"{args.gait_name}-s{args.training_seed}"]
        args.episodes_per_terrain = settings["episodes_per_terrain"]
        args.seconds = settings["seconds"]
        args.severity = settings.get("severity",args.severity)
        args.push_speed = settings.get('push_speed',args.push_speed)
        args.behavior_policy = settings.get('behavior_policy',args.behavior_policy)
        args.out = campaign/"data"/f"{args.gait_name}-s{args.training_seed}.npz"
    if args.mode in ("eval", "evaluate") and args.cell is not None:
        if protocol is None or not 0 <= args.cell < len(protocol["navigation_cells"]):
            parser.error("--cell requires frozen protocol and valid navigation cell index")
        cell, settings = protocol["navigation_cells"][args.cell], protocol["navigation_settings"]
        args.gait_name, args.training_seed = cell["gait"], cell["training_seed"]
        args.navigator = "astar" if cell["method"] == "astar" else cell["actor"]
        args.governor = "baseline" if cell["method"] == "astar" else cell["method"]
        args.sensor_mode = cell["sensor_mode"]
        args.seed_start, args.seeds = min(settings["maze_seeds"]), len(settings["maze_seeds"])
        if settings["maze_seeds"] != list(range(args.seed_start, args.seed_start+args.seeds)):
            parser.error("maze seed declaration must be consecutive for immutable benchmark wrapper")
        args.time_limit = settings["time_limit_s"]
        args.out_dir = campaign/"navigation"/cell["name"]
    if any(getattr(args, k) is None for k in ("gait_name", "training_seed", "seed_start")):
        parser.error("explicit runs need --gait-name, --training-seed, --seed-start")
    if args.mode == "collect":
        if args.out is None or args.episodes_per_terrain < 1 or args.seconds < .04 or args.push_speed < 0:
            parser.error("positive episode count/duration and nonnegative push speed required")
        result = collect(args)
        if protocol:
            validate(campaign)
        return result
    if args.out_dir is None or args.navigator is None or args.governor is None or args.seeds < 1 or args.time_limit < .04:
        parser.error("positive episode count/duration required")
    result = evaluate(args)
    if protocol:
        validate(campaign)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
