"""Why does a 22-DoF gait not turn from a standstill?  (diagnostic, not a gate)

Three instruments, one question -- is yaw tracking under a sustained pure-turn
command (0, 0, wz) learned at all, and if so, is the failure a sim2sim gap?

  mujoco  CPU MuJoCo replay through the same path as turn_test.py (build_multi +
          RlController + ContactRunner). After a zero-command settle, a pure
          turn is held for --seconds. Per run it records: the base yaw turned;
          per foot the number of lift-offs (contact -> airborne for >= 20 ms,
          contacts sampled every 5 ms), total air time and peak clearance; how
          much of the base yaw happened while a foot was airborne (stepping) vs
          in double stance (pivoting/shuffling); foot yaw accumulated while in
          contact (sole pivot); leg-joint motion (RMS about the window mean) in
          the turn window vs the last 2 s of the settle; and the action response
          to the command: at every turn step the policy is also evaluated on the
          same observation with the command zeroed, and ||a(cmd) - a(0)|| is
          averaged (does the network react to wz at all?).
  tb      Reads a run's TensorBoard scalars and normalises the episode-summed
          command metrics per step (Isaac Lab accumulates error_vel_* as
          |err| / max_command_step over the episode, and Episode_Reward/* is the
          episode sum / episode_length_s, so both scale with episode length).
  isaac   Isaac Sim replay probe (GPU job only; `isaaclab.app` must import).
          Builds the stock 22-DoF velocity env with DR as trained, observation
          noise off, standing envs off, heading control off, no resampling;
          zero command for --warm seconds, then half the envs get (0, 0, +wz)
          and half (0, 0, -wz) for --seconds. Per env: yaw turned in the
          commanded direction, fell, foot lift-offs (Isaac contact sensor).
          The exported TorchScript actor (exported/policy.pt) is the policy,
          i.e. the network the ONNX / MuJoCo replay uses.
          If the Isaac probe turns >= 150 deg where MuJoCo does not, the failure
          is a sim2sim gap; if Isaac also stands, pure-turn tracking was never
          learned.

Every output is labelled: learned policy, replayed; no scripted control.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]

LEG_GROUPS = {
    "hip_roll": ("leg_left_hip_roll_joint", "leg_right_hip_roll_joint"),
    "hip_yaw": ("leg_left_hip_yaw_joint", "leg_right_hip_yaw_joint"),
    "hip_pitch": ("leg_left_hip_pitch_joint", "leg_right_hip_pitch_joint"),
    "knee": ("leg_left_knee_pitch_joint", "leg_right_knee_pitch_joint"),
    "ankle_pitch": ("leg_left_ankle_pitch_joint", "leg_right_ankle_pitch_joint"),
    "ankle_roll": ("leg_left_ankle_roll_joint", "leg_right_ankle_roll_joint"),
}
FEET = ("left", "right")
MIN_AIR_S = 0.020          # an airborne bout shorter than this is contact chatter, not a step
SAMPLE_EVERY = 10          # physics substeps between contact samples (0.5 ms * 10 = 5 ms)


def yaw_of(q):
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def mat_yaw(xmat):
    return math.atan2(xmat[3], xmat[0])


def count_liftoffs(contact: np.ndarray, dt: float, min_air_s: float = MIN_AIR_S) -> tuple[int, float]:
    """(lift-offs, airborne seconds) of a boolean contact trace sampled every dt.

    A lift-off is a contact -> airborne transition whose airborne bout lasts at
    least min_air_s; a bout still open at the end of the trace counts if long
    enough. Bouts before the first contact are not lift-offs."""
    n, air, i, seen_contact = 0, 0.0, 0, False
    c = np.asarray(contact, dtype=bool)
    while i < len(c):
        if c[i]:
            seen_contact = True
            i += 1
            continue
        j = i
        while j < len(c) and not c[j]:
            j += 1
        bout = (j - i) * dt
        if seen_contact and bout >= min_air_s - 1e-12:
            n += 1
            air += bout
        i = j
    return n, air


# ----------------------------------------------------------------------------- MuJoCo

def mujoco_run(deploy: Path, upstream: Path, cache: Path, wz: float, warm: float, seconds: float, seed: int) -> dict:
    import mujoco
    from omegaconf import OmegaConf
    from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
    from bhl_robust.eval.multi_robot import build_multi
    from team_airlock import ContactRunner, CpuPolicy

    cfg = OmegaConf.load(deploy)
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    model, slots = build_multi(upstream, cache / "humanoid", 1, ["t"], variant="humanoid", world="flat")
    ctrl = RlController(cfg)
    ctrl.policy = policy
    runner = ContactRunner(model, slots, [cfg], [ctrl])
    runner.reset(np.random.default_rng(seed))
    slot = slots[0]
    owners = np.array([0 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "").startswith(slot.prefix)
                       else -1 for g in range(model.ngeom)])
    runner.configure_contacts(owners)

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_body = {f: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{slot.prefix}leg_{f}_ankle_roll") for f in FEET}
    foot_geoms = {f: {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == foot_body[f] and model.geom_contype[g]}
                  for f in FEET}
    joints = list(cfg.joints)
    jidx = {g: [joints.index(n) for n in names] for g, names in LEG_GROUPS.items()}

    trace = {"t": [], "left": [], "right": [], "yaw": [], "fyaw_l": [], "fyaw_r": [], "fz_l": [], "fz_r": []}
    clock = {"t": 0.0}
    sub_dt = float(cfg.physics_dt)
    counter = {"k": 0}

    def hook(d):
        counter["k"] += 1
        clock["t"] += sub_dt
        if counter["k"] % SAMPLE_EVERY:
            return
        on = {"left": False, "right": False}
        for c in range(d.ncon):
            g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
            other = g2 if g1 == floor else g1 if g2 == floor else None
            if other is None:
                continue
            for f in FEET:
                if other in foot_geoms[f]:
                    on[f] = True
        trace["t"].append(clock["t"])
        trace["left"].append(on["left"])
        trace["right"].append(on["right"])
        trace["yaw"].append(yaw_of(d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]))
        trace["fyaw_l"].append(mat_yaw(d.xmat[foot_body["left"]]))
        trace["fyaw_r"].append(mat_yaw(d.xmat[foot_body["right"]]))
        trace["fz_l"].append(float(d.xpos[foot_body["left"], 2]))
        trace["fz_r"].append(float(d.xpos[foot_body["right"], 2]))

    runner.substep_hook = hook
    dt = float(cfg.policy_dt)
    n_obs = int(cfg.num_observations)
    qpos_steps, resp, resp_leg, act_norm = [], [], [], []
    leg_ids = [i for i, n in enumerate(joints) if n.startswith("leg_")]
    fell = None
    for step in range(int(round((warm + seconds) / dt))):
        now = step * dt
        cmd = np.zeros(3) if now < warm else np.array([0.0, 0.0, wz])
        obs = runner.observe(0, cmd)
        target = ctrl.update(obs)
        if now >= warm:
            a = ctrl.policy_actions[0].copy()
            cf = ctrl.policy_observations.copy()
            off = cf.shape[1] - n_obs          # the latest frame starts with the 3 command entries
            cf[0, off:off + 3] = 0.0
            a0 = np.asarray(policy.forward(cf)).reshape(-1)
            resp.append(float(np.linalg.norm(a - a0)))
            resp_leg.append(float(np.linalg.norm((a - a0)[leg_ids])))
            act_norm.append(float(np.linalg.norm(a)))
        qpos_steps.append((now, runner.d.qpos[slot.qpos_adr + 7:slot.qpos_adr + 7 + len(joints)].copy()))
        runner.step([target])
        if runner.tilt(0) >= 0.78:
            fell = round(now, 2)
            break

    t = np.array(trace["t"])
    samp_dt = sub_dt * SAMPLE_EVERY
    turn_mask = t >= warm
    settle_mask = (t >= max(0.0, warm - 2.0)) & (t < warm)
    yaw = np.unwrap(np.array(trace["yaw"]))
    out = {"seed": seed, "cmd": [0.0, 0.0, wz], "warm_s": warm, "seconds": seconds, "fell_at_s": fell}
    if turn_mask.sum() < 2 or not any(tt >= warm for tt, _ in qpos_steps):
        out["note"] = "fell before the turn command started; no turn window"
        return out
    i0 = int(np.argmax(turn_mask))
    out["yaw_deg"] = round(math.degrees(yaw[-1] - yaw[i0]), 1)
    feet = {}
    for f, key in (("left", "fyaw_l"), ("right", "fyaw_r")):
        c = np.array(trace[f])
        fz = np.array(trace["fz_l" if f == "left" else "fz_r"])
        fy = np.unwrap(np.array(trace[key]))
        base_z = float(np.median(fz[settle_mask])) if settle_mask.any() else float(fz[0])
        lo_t, air_t = count_liftoffs(c[turn_mask], samp_dt)
        lo_s, air_s = count_liftoffs(c[settle_mask], samp_dt)
        dfy = np.diff(fy[turn_mask])
        both_in = c[turn_mask][1:] & c[turn_mask][:-1]
        feet[f] = {"liftoffs_turn": lo_t, "air_s_turn": round(air_t, 3), "liftoffs_last2s_settle": lo_s,
                   "air_s_last2s_settle": round(air_s, 3),
                   "peak_clearance_m_turn": round(float(fz[turn_mask].max() - base_z), 4) if turn_mask.any() else None,
                   "foot_yaw_in_contact_deg": round(math.degrees(float(np.sum(dfy[both_in]))), 1)}
    out["feet"] = feet
    cl, cr = np.array(trace["left"])[turn_mask], np.array(trace["right"])[turn_mask]
    dyaw = np.diff(yaw[turn_mask])
    airborne = ~(cl & cr)[1:]
    out["yaw_deg_while_a_foot_airborne"] = round(math.degrees(float(np.sum(dyaw[airborne]))), 1)
    out["yaw_deg_in_double_stance"] = round(math.degrees(float(np.sum(dyaw[~airborne]))), 1)
    q_t = np.array([q for (tt, q) in qpos_steps if tt >= warm])
    q_s = np.array([q for (tt, q) in qpos_steps if warm - 2.0 <= tt < warm])
    motion = {}
    for g, ids in jidx.items():
        def rms(Q):
            return float(np.sqrt(np.mean((Q[:, ids] - Q[:, ids].mean(axis=0)) ** 2))) if len(Q) else None
        motion[g] = {"rms_rad_turn": round(rms(q_t), 4) if len(q_t) else None,
                     "rms_rad_last2s_settle": round(rms(q_s), 4) if len(q_s) else None}
    if len(q_t):
        hy = jidx["hip_yaw"]
        motion["hip_yaw"]["mean_left_minus_right_rad_turn"] = round(float(np.mean(q_t[:, hy[0]] - q_t[:, hy[1]])), 4)
    out["joint_motion"] = motion
    out["action_response"] = {"mean_norm_a_cmd_minus_a_zero": round(float(np.mean(resp)), 4) if resp else None,
                              "mean_norm_legs_only": round(float(np.mean(resp_leg)), 4) if resp_leg else None,
                              "mean_norm_a": round(float(np.mean(act_norm)), 4) if act_norm else None}
    return out


def summarise_mujoco(all_runs: list[dict]) -> dict:
    runs = [r for r in all_runs if "note" not in r]          # runs with a turn window
    ok = [r for r in all_runs if r["fell_at_s"] is None]
    lo = [r["feet"]["left"]["liftoffs_turn"] + r["feet"]["right"]["liftoffs_turn"] for r in runs]
    signed = [r["yaw_deg"] * math.copysign(1.0, r["cmd"][2]) for r in runs]
    return {"n_runs": len(all_runs), "n_fell": len(all_runs) - len(ok), "n_fell_before_turn": len(all_runs) - len(runs),
            "yaw_deg_in_commanded_direction": [round(s, 1) for s in signed],
            "n_ge_150deg": sum(s >= 150.0 and r["fell_at_s"] is None for s, r in zip(signed, runs)),
            "liftoffs_turn_total_per_run": lo,
            "median_liftoffs_turn": float(np.median(lo)) if lo else None,
            "median_action_response": round(float(np.median([r["action_response"]["mean_norm_a_cmd_minus_a_zero"] for r in runs
                                                              if r["action_response"]["mean_norm_a_cmd_minus_a_zero"] is not None])), 3)
            if runs else None,
            "median_hip_yaw_rms_turn": float(np.median([r["joint_motion"]["hip_yaw"]["rms_rad_turn"] for r in runs])) if runs else None,
            "yaw_share_while_airborne": [round(r["yaw_deg_while_a_foot_airborne"] / r["yaw_deg"], 2) if abs(r["yaw_deg"]) > 5 else None
                                         for r in runs]}


def main_mujoco(args) -> int:
    report = {"instrument": "MuJoCo replay of a LEARNED policy (no scripted control), pure-turn command after a zero-command settle",
              "wz": args.wz, "warm_s": args.warm, "seconds": args.seconds, "seeds": args.seeds, "runs": {}, "summary": {}}
    for label, deploy in zip(args.labels, args.deploy):
        runs = []
        for seed in args.seeds:
            for sign in (1.0, -1.0):
                r = mujoco_run(deploy, args.upstream, args.cache_dir, sign * args.wz, args.warm, args.seconds, seed)
                runs.append(r)
                if "note" in r:
                    print(f"  {label} s{seed} wz{sign * args.wz:+.2f}: {r['note']} (fell {r['fell_at_s']})", flush=True)
                    continue
                f = r["feet"]
                print(f"  {label} s{seed} wz{sign * args.wz:+.2f}: yaw {r['yaw_deg']:+7.1f} deg | lift-offs L{f['left']['liftoffs_turn']}"
                      f" R{f['right']['liftoffs_turn']} (settle L{f['left']['liftoffs_last2s_settle']} R{f['right']['liftoffs_last2s_settle']})"
                      f" | airborne-yaw {r['yaw_deg_while_a_foot_airborne']:+.1f} double-stance-yaw {r['yaw_deg_in_double_stance']:+.1f}"
                      f" | foot-pivot L{f['left']['foot_yaw_in_contact_deg']:+.1f} R{f['right']['foot_yaw_in_contact_deg']:+.1f}"
                      f" | |a(cmd)-a(0)| {r['action_response']['mean_norm_a_cmd_minus_a_zero']:.3f}"
                      f" | hip-yaw rms {r['joint_motion']['hip_yaw']['rms_rad_turn']:.3f} | fell {r['fell_at_s']}", flush=True)
        report["runs"][label] = runs
        report["summary"][label] = summarise_mujoco(runs)
        s = report["summary"][label]
        print(f"MUJOCO {label}: {s['n_ge_150deg']}/{s['n_runs']} runs >= 150 deg, median lift-offs {s['median_liftoffs_turn']},"
              f" median |a(cmd)-a(0)| {s['median_action_response']}, falls {s['n_fell']}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


# ----------------------------------------------------------------------------- TensorBoard

TB_TAGS = ("Metrics/base_velocity/error_vel_yaw", "Metrics/base_velocity/error_vel_xy",
           "Episode_Reward/track_ang_vel_z_exp", "Episode_Reward/feet_air_time", "Episode_Reward/track_lin_vel_xy_exp",
           "Metrics/base_velocity/pure_turn_env", "Metrics/base_velocity/direct_wz_env", "Train/mean_episode_length",
           "Episode_Termination/base_orientation", "Episode_Termination/time_out", "Policy/mean_noise_std")


def per_step(metric_sum: float, ep_len_steps: float, max_command_steps: float) -> float:
    """Isaac Lab's velocity-command metric is sum_t |err_t| / max_command_steps over the episode;
    return the per-step mean |err| (rad/s for error_vel_yaw)."""
    return metric_sum * max_command_steps / ep_len_steps if ep_len_steps > 0 else float("nan")


def kernel_mean(reward_per_s: float, episode_length_s: float, ep_len_steps: float, step_dt: float, weight: float) -> float:
    """Episode_Reward/<term> = episode sum / episode_length_s, each step adding weight * value * step_dt;
    return the mean per-step value of the term (for an exp kernel: its mean in [0, 1])."""
    return reward_per_s * episode_length_s / (ep_len_steps * step_dt * weight) if ep_len_steps > 0 else float("nan")


def main_tb(args) -> int:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    report = {"instrument": "TensorBoard scalars of Isaac training runs (learned policies, stochastic rollouts)",
              "note": "yaw_err_per_step_rad_s and yaw_kernel_mean are APPROXIMATE: ratios of episode means "
                      "(metric mean / mean episode length), not per-episode means of the ratio",
              "window": args.window, "at": args.at, "runs": {}}
    for run in args.run_dirs:
        ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
        ea.Reload()
        have = set(ea.Tags()["scalars"])
        rows = {}
        for it in args.at:
            row = {}
            for tag in TB_TAGS:
                if tag not in have:
                    continue
                v = np.array([s.value for s in ea.Scalars(tag)])
                if not len(v):
                    continue
                k = min(it, len(v) - 1)
                row[tag.split("/")[-1]] = float(v[max(0, k - args.window):k + args.window + 1].mean())
            L = row.get("mean_episode_length", float("nan"))
            if "error_vel_yaw" in row:
                row["yaw_err_per_step_rad_s"] = per_step(row["error_vel_yaw"], L, args.max_command_steps)
            if "track_ang_vel_z_exp" in row:
                row["yaw_kernel_mean"] = kernel_mean(row["track_ang_vel_z_exp"], args.episode_length_s, L, args.step_dt,
                                                     args.yaw_weight)
            rows[str(it)] = {k: round(v, 4) for k, v in row.items()}
        report["runs"][run.name] = rows
        last = rows[str(args.at[-1])]
        print(f"TB {run.name} @ {args.at[-1]}: ep_len {last.get('mean_episode_length')}, falls/reset "
              f"{last.get('base_orientation')}, yaw err/step {last.get('yaw_err_per_step_rad_s')} rad/s, "
              f"yaw kernel {last.get('yaw_kernel_mean')}, air-time {last.get('feet_air_time')}, "
              f"noise std {last.get('mean_noise_std')}, pure_turn_env {last.get('pure_turn_env')}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


# ----------------------------------------------------------------------------- Isaac

def _nanmed(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.median(x)) if x.size else float("nan")


def main_isaac(args) -> int:
    """Isaac Sim replay probe. GPU job only; see the module docstring."""
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True).app

    import gymnasium as gym
    import torch
    import berkeley_humanoid_lite.tasks  # noqa: F401
    import bhl_robust.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    report = {"instrument": "Isaac Sim replay of a LEARNED policy (exported TorchScript actor), pure-turn command",
              "task": args.task, "wz": args.wz, "warm_s": args.warm, "seconds": args.seconds, "num_envs": args.num_envs,
              "runs": {}}
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.observations.policy.enable_corruption = False
    c = env_cfg.commands.base_velocity
    c.heading_command = False
    c.rel_standing_envs = 0.0
    c.rel_heading_envs = 0.0
    c.resampling_time_range = (1.0e6, 1.0e6)
    c.ranges.lin_vel_x = (0.0, 0.0)
    c.ranges.lin_vel_y = (0.0, 0.0)
    c.ranges.ang_vel_z = (0.0, 0.0)
    c.debug_vis = False
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), args.warm + args.seconds + 5.0)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    dt = float(env.step_dt)
    term = env.command_manager.get_term("base_velocity")
    robot = env.scene["robot"]
    sensor = env.scene.sensors["contact_forces"]
    feet_ids = sensor.find_bodies(".*_ankle_roll")[0]
    n = env.num_envs
    sign = torch.where(torch.arange(n, device=env.device) < n // 2, 1.0, -1.0)
    for label, run_dir in zip(args.labels, args.run_dirs):
        actor = torch.jit.load(str(Path(run_dir) / "exported" / "policy.pt"), map_location=env.device).eval()
        obs, _ = env.reset()
        term.vel_command_b[:] = 0.0
        obs = env.observation_manager.compute()
        fell = torch.zeros(n, dtype=torch.bool, device=env.device)
        yaws, contacts = [], []
        steps_warm, steps_turn = int(round(args.warm / dt)), int(round(args.seconds / dt))
        for k in range(steps_warm + steps_turn):
            if k == steps_warm:
                term.vel_command_b[:, 0:2] = 0.0
                term.vel_command_b[:, 2] = sign * args.wz
                obs = env.observation_manager.compute()
            with torch.inference_mode():
                act = actor(obs["policy"])
            obs, _, terminated, truncated, _ = env.step(act)
            fell |= terminated
            if k >= steps_warm:
                yaws.append(robot.data.heading_w.clone())
                contacts.append((sensor.data.current_contact_time[:, feet_ids] > 0.0).clone())
        yaw = torch.stack(yaws).cpu().numpy()
        yaw = np.unwrap(yaw, axis=0)
        turned = np.degrees(yaw[-1] - yaw[0]) * sign.cpu().numpy()
        cont = torch.stack(contacts).cpu().numpy()          # (T, n, 2)
        lift = np.array([[count_liftoffs(cont[:, e, f], dt, min_air_s=dt)[0] for f in range(cont.shape[2])]
                         for e in range(n)]).sum(axis=1)
        fell_np = fell.cpu().numpy()
        ok = (~fell_np) & (turned >= 150.0)
        res = {"yaw_deg_in_commanded_direction": [round(float(v), 1) for v in turned],
               "fell": [bool(v) for v in fell_np], "liftoffs_turn": [int(v) for v in lift],
               "n_ge_150deg_no_fall": int(ok.sum()), "n": n,
               "median_yaw_deg_plus": _nanmed(turned[: n // 2][~fell_np[: n // 2]]),
               "median_yaw_deg_minus": _nanmed(turned[n // 2:][~fell_np[n // 2:]]),
               "median_liftoffs": _nanmed(lift[~fell_np]), "n_fell": int(fell_np.sum()),
               "note": "medians over envs that did not fall; yaw in the commanded direction (+dir: first half, -dir: second half)"}
        report["runs"][label] = res
        print(f"ISAAC {label}: {res['n_ge_150deg_no_fall']}/{n} envs >= 150 deg (no fall); median yaw +{res['median_yaw_deg_plus']:.1f} / "
              f"-dir {res['median_yaw_deg_minus']:.1f} deg; median lift-offs {res['median_liftoffs']}; falls {res['n_fell']}", flush=True)
        if args.out:          # written after every run: Kit can die later and still exit 0
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(f"wrote {args.out} ({len(report['runs'])}/{len(args.labels)} runs)", flush=True)
    env.close()
    app.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("mujoco")
    m.add_argument("--deploy", type=Path, nargs="+", required=True)
    m.add_argument("--labels", nargs="+", required=True)
    m.add_argument("--upstream", type=Path, required=True)
    m.add_argument("--cache-dir", type=Path, required=True)
    m.add_argument("--seeds", type=int, nargs="+", default=[0])
    m.add_argument("--wz", type=float, default=0.6)
    m.add_argument("--warm", type=float, default=3.0)
    m.add_argument("--seconds", type=float, default=6.0)
    m.add_argument("--out", type=Path, default=None)
    t = sub.add_parser("tb")
    t.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    t.add_argument("--at", type=int, nargs="+", default=[1000, 3000, 5949])
    t.add_argument("--window", type=int, default=50)
    t.add_argument("--max-command-steps", type=float, default=250.0, help="resampling 10 s / step_dt 0.04 s (humanoid)")
    t.add_argument("--episode-length-s", type=float, default=20.0)
    t.add_argument("--step-dt", type=float, default=0.04)
    t.add_argument("--yaw-weight", type=float, default=2.0, help="track_ang_vel_z_exp weight of the runs (TurnBoth/TurnCmd 2.0; stock 1.0)")
    t.add_argument("--out", type=Path, default=None)
    i = sub.add_parser("isaac")
    i.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    i.add_argument("--labels", nargs="+", required=True)
    i.add_argument("--task", default="Velocity-Berkeley-Humanoid-Lite-v0")
    i.add_argument("--num-envs", type=int, default=32)
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--wz", type=float, default=0.6)
    i.add_argument("--warm", type=float, default=3.0)
    i.add_argument("--seconds", type=float, default=6.0)
    i.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.mode in ("mujoco", "isaac") and len(args.labels) != len(getattr(args, "deploy", None) or args.run_dirs):
        ap.error("--labels must match --deploy / --run-dirs one-to-one")
    return {"mujoco": main_mujoco, "tb": main_tb, "isaac": main_isaac}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
