"""Why do TaskV2 CubeToShelf training episodes last ~5 steps?

`spawn_quat_probe.py` (21342562 re-probe, results/spawn_quat_probe.json) said
the configured spawn is STANDING -- 1 of 27 bodies below z = 0, R22 = +1 -- yet
`v2stand-cubetoshelf-blind-s0` (21352980_0) logged mean episode length 5.00
with Episode_Termination/fallen 1.0000. The probe measured the spawn after at
most two physics steps and used `matrix_from_quat` (native quaternion order);
the termination uses `coop_lift_mdp._tilt_from_quat`, which unpacks the root
quaternion as (w, x, y, z). Those two claims have never been measured on the
same rollout. This script does that.

For ONE task per process (Isaac Lab allows one SimulationContext per process),
it builds the env at `--num_envs`, resets it onto the configured default spawn,
and for `--steps` policy steps runs each condition:

  zero     action_manager gets a zero action. With use_default_offset=True and
           scale 0.25 this targets default_joint_pos for the ACTION joints.
  pd_hold  the action manager is bypassed: wherever env.step applies actions
           (every physics substep, or once per policy step on backends that
           handle decimation themselves) each robot's joint position target is set to its own `default_joint_pos` for ALL
           joints (action-set or not). For the action joints this is the same
           target as `zero`, so (a) vs (b) disagreeing isolates something
           outside the action path (non-action joints, clip, offset drift).
  gauss    (extra, training-like) actions ~ N(0, 1), the std the TaskV2 actor
           is initialised with (init_std = 1.0), fresh every step.

Episodes are NOT reset when a term fires. `termination_manager.compute` is
wrapped: the real terms are evaluated (so `get_term` is exactly what training
sees), their values are recorded, and then the done buffers are cleared so
`env.step` does not call `_reset_idx`. That keeps the trajectory continuous for
all `--steps` steps, and the first step each term fires is the episode length
the trainer would have seen. Everything else goes through the stock
`ManagerBasedRLEnv.step` (decimation, actuators, sensors) unchanged.

Logged per step (step 0 = after reset, before any action), per robot, per env:
  tilt_native  acos(R22) with R from isaaclab.utils.math.matrix_from_quat, which
               reads the stack's own quaternion order (xyzw on Lab 3.0)
  tilt_term    coop_lift_mdp._tilt_from_quat -- what `fallen` compares to 0.78
  tilt_pg      acos(-projected_gravity_b.z)
  base_z       root z above the env origin;  max_body_z
  n_below      bodies with z < 0 (relative to the env origin);  min_body_z
  ankle_z      z of each *ankle_roll* body;  shoulder_z  z of each *shoulder_pitch*
  foot_force   |net contact force| on each foot body from contact_a / contact_b
  joint_err    max |joint_pos - default_joint_pos|
and per env: every termination term's value, object z above the origin.

Output: one JSON per condition, `<out_dir>/<task>__<condition>.json`, with
`first_fire` (first step each term fires, overall and per env), the time
series, and `verdict`:
  STANDS                      no term fired in `--steps` steps and max native
                              tilt < 0.5 rad
  TERMINATES: <term> @ step N the first term to fire and its step
  TILTS: ...                  nothing fired but native tilt reached >= 0.5 rad
plus `diagnosis`, machine-generated notes comparing tilt_term with tilt_native
at the step `fallen` first fired.

Grep line per condition:  SPAWN-DIAG <task> <condition> <verdict>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

DEFAULT_OUT = os.path.join(os.environ.get("REPO", "."), "results",
                           "repo-gpu-20260923", "spawn_diag")

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="TaskV2-BHL-CubeToShelf-Blind-v0",
                    help="registered gym id (one per process)")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200, help="policy steps per condition")
parser.add_argument("--conditions", nargs="+", default=["zero", "pd_hold", "gauss"],
                    choices=["zero", "pd_hold", "gauss"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--tilt_stand", type=float, default=0.5,
                    help="native tilt (rad) at or above which a robot is not standing")
parser.add_argument("--out_dir", default=os.environ.get("SPAWN_DIAG_OUT") or DEFAULT_OUT)
parser.add_argument("--init_z_offset", type=float, default=0.0,
                    help="metres added to both robots' configured init_state z before the env is built "
                         "(spawn-height sweep; 0 = as configured)")
parser.add_argument("--tag", default="", help="suffix for the output file names, e.g. z-0.095")

# --help must not boot Isaac Sim: answer it before importing isaaclab.
if any(a in ("-h", "--help") for a in sys.argv[1:]):
    parser.print_help()
    sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

import bhl_robust.tasks  # noqa: F401,E402  registers the ids
from bhl_robust.quat_order import quat_order  # noqa: E402
from bhl_robust.tasks.coop_lift_mdp import _t, _tilt_from_quat  # noqa: E402

ROBOTS = ("robot_a", "robot_b")


def _r(x, nd: int = 5):
    """Tensor/float/list -> JSON-friendly nested lists of rounded floats."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, bool):
        return x
    if isinstance(x, float):
        return None if not math.isfinite(x) else round(x, nd)
    return x


def _mx(vals, default: float = -1.0) -> float:
    """Max of a list that may hold None (non-finite values), or `default`."""
    vals = [v for v in (vals or []) if v is not None]
    return max(vals) if vals else default


def _idx(names, keys):
    """Indices of names containing the first key in `keys` that matches any."""
    for k in keys:
        hit = [i for i, n in enumerate(names) if k in n.lower()]
        if hit:
            return hit
    return []


class Probe:
    """Per-step state reader for one built env."""

    def __init__(self, env):
        self.env = env
        self.u = env.unwrapped
        keys = list(self.u.scene.keys())
        self.robots = [r for r in ROBOTS if r in keys]
        self.has_object = "object" in keys
        self.meta = {}
        for r in self.robots:
            art = self.u.scene[r]
            names = list(art.body_names)
            m = {
                "body_names": names,
                "ankle_idx": _idx(names, ("ankle_roll", "ankle", "foot")),
                "shoulder_idx": _idx(names, ("shoulder_pitch", "shoulder")),
                "sensor": None, "foot_sensor_idx": [], "foot_sensor_names": [],
            }
            sname = "contact_" + r.split("_")[-1]
            if sname in keys:
                sensor = self.u.scene[sname]
                snames = list(sensor.body_names or [])
                fidx = _idx(snames, ("ankle_roll", "foot", "ankle"))
                m.update(sensor=sname, foot_sensor_idx=fidx,
                         foot_sensor_names=[snames[i] for i in fidx])
            self.meta[r] = m

    def snapshot(self) -> dict:
        u = self.u
        org = u.scene.env_origins
        out = {}
        for r in self.robots:
            art = u.scene[r]
            m = self.meta[r]
            q = _t(art.data.root_quat_w)
            R = matrix_from_quat(q)
            tilt_native = torch.acos(R[:, 2, 2].clamp(-1.0, 1.0))
            tilt_term = _tilt_from_quat(art)
            pg = _t(art.data.projected_gravity_b)
            tilt_pg = torch.acos((-pg[:, 2]).clamp(-1.0, 1.0))
            root = _t(art.data.root_pos_w)
            bz = _t(art.data.body_pos_w)[..., 2] - org[:, 2:3]
            jerr = (_t(art.data.joint_pos) - _t(art.data.default_joint_pos)).abs().max(dim=1).values
            row = {
                "tilt_native": tilt_native, "tilt_term": tilt_term, "tilt_pg": tilt_pg,
                "base_z": root[:, 2] - org[:, 2],
                "max_body_z": bz.max(dim=1).values, "min_body_z": bz.min(dim=1).values,
                "n_below": (bz < 0.0).sum(dim=1),
                "ankle_z": bz[:, m["ankle_idx"]] if m["ankle_idx"] else None,
                "shoulder_z": bz[:, m["shoulder_idx"]] if m["shoulder_idx"] else None,
                "joint_err": jerr,
                "foot_force": None,
            }
            if m["sensor"] and m["foot_sensor_idx"]:
                f = _t(u.scene[m["sensor"]].data.net_forces_w)
                if f is not None:
                    row["foot_force"] = f[:, m["foot_sensor_idx"], :].norm(dim=-1)
            out[r] = {k: _r(v) for k, v in row.items()}
        if self.has_object:
            oz = _t(u.scene["object"].data.root_pos_w)[:, 2] - org[:, 2]
            out["object_z"] = _r(oz)
        return out


def _term_info(tm) -> dict:
    info = {}
    for name in tm.active_terms:
        cfg = tm.get_term_cfg(name)
        params = {}
        for k, v in (cfg.params or {}).items():
            params[k] = getattr(v, "name", None) or repr(v)
        info[name] = {"func": getattr(cfg.func, "__name__", repr(cfg.func)),
                      "module": getattr(cfg.func, "__module__", None),
                      "time_out": bool(getattr(cfg, "time_out", False)),
                      "params": params}
    return info


def _hold_default(u, robots):
    """Replacement for action_manager.apply_action: PD-hold default joints."""
    def apply_action():
        for r in robots:
            art = u.scene[r]
            tgt = _t(art.data.default_joint_pos)
            if hasattr(art, "set_joint_position_target_index"):
                art.set_joint_position_target_index(target=tgt)
            else:                                          # Isaac Lab 2.x
                art.set_joint_position_target(tgt)
    return apply_action


def run_condition(env, probe: Probe, cond: str, steps: int, seed: int) -> dict:
    u = env.unwrapped
    tm = u.termination_manager
    am = u.action_manager
    terms = list(tm.active_terms)
    env.reset(seed=seed)
    for r in probe.robots:
        try:
            u.scene[r].update(dt=0.0)
        except Exception:                                   # noqa: BLE001
            pass

    q0 = {r: _r(_t(u.scene[r].data.root_quat_w)) for r in probe.robots}
    series = [dict(step=0, terms={t: [False] * u.num_envs for t in terms}, **probe.snapshot())]

    captured = {}
    orig_compute = tm.compute

    def compute_no_reset():
        orig_compute()
        for t in terms:
            captured[t] = tm.get_term(t).clone()
        # Clear the done buffers so env.step does not reset anyone. The term
        # values above are exactly what training would have acted on.
        for buf in ("_terminated_buf", "_truncated_buf"):
            if hasattr(tm, buf):
                getattr(tm, buf)[:] = False
        return torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)

    tm.compute = compute_no_reset
    if cond == "pd_hold":
        am.apply_action = _hold_default(u, probe.robots)
    gen = torch.Generator(device=u.device).manual_seed(seed + 1)
    dim = am.total_action_dim
    zero = torch.zeros((u.num_envs, dim), device=u.device)
    err = None
    t0 = time.time()
    try:
        for k in range(1, steps + 1):
            if cond == "gauss":
                act = torch.randn((u.num_envs, dim), device=u.device, generator=gen)
            else:
                act = zero
            env.step(act)
            row = dict(step=k, terms={t: captured[t].bool().cpu().tolist() for t in terms})
            row.update(probe.snapshot())
            series.append(row)
    except Exception as exc:                                # noqa: BLE001
        import traceback
        traceback.print_exc()
        err = f"{type(exc).__name__}: {exc}"
    finally:
        # Restore the class methods (instance attributes shadow them).
        for obj, attr in ((tm, "compute"), (am, "apply_action")):
            if attr in vars(obj):
                delattr(obj, attr)
    wall = time.time() - t0

    # ---------------------------------------------------------------- analysis
    n = u.num_envs
    first_fire = {}
    for t in terms:
        per_env = [None] * n
        for row in series[1:]:
            for e, v in enumerate(row["terms"][t]):
                if v and per_env[e] is None:
                    per_env[e] = row["step"]
        hits = [s for s in per_env if s is not None]
        first_fire[t] = {"step": min(hits) if hits else None, "per_env": per_env,
                         "envs_fired": len(hits)}
    # training episode length implied per env = first step any term fires
    implied = []
    for e in range(n):
        s = [first_fire[t]["per_env"][e] for t in terms if first_fire[t]["per_env"][e] is not None]
        implied.append(min(s) if s else None)

    def _max_over(key, upto=None):
        best, at = -1.0, None
        for row in series[: (upto + 1) if upto is not None else None]:
            for r in probe.robots:
                m = _mx(row[r][key], default=None)
                if m is not None and m > best:
                    best, at = m, row["step"]
        return best, at

    max_native, max_native_at = _max_over("tilt_native")
    first_tilt_step = None
    for row in series:
        if any(v is not None and v >= args_cli.tilt_stand
               for r in probe.robots for v in row[r]["tilt_native"]):
            first_tilt_step = row["step"]
            break

    fired = sorted((ff["step"], t) for t, ff in first_fire.items() if ff["step"] is not None)
    if err is not None and not fired:
        verdict = f"ERROR: {err}"
    elif not fired and max_native < args_cli.tilt_stand:
        verdict = "STANDS"
    elif fired:
        s0 = fired[0][0]
        names = "+".join(t for s, t in fired if s == s0)
        verdict = f"TERMINATES: {names} @ step {s0}"
    else:
        verdict = (f"TILTS: native tilt {max_native:.3f} rad >= {args_cli.tilt_stand} "
                   f"first at step {first_tilt_step}; no termination fired")

    diagnosis = []
    s0row = series[0]
    for r in probe.robots:
        tn = s0row[r]["tilt_native"]
        tt = s0row[r]["tilt_term"]
        diagnosis.append(
            f"{r} at reset: tilt_native {tn}, tilt_term {tt}, n_below {s0row[r]['n_below']}, "
            f"base_z {s0row[r]['base_z']}, root_quat_w {q0[r][0]} (stack order {quat_order()})")
        if _mx(tt) > 0.78 and 0.0 <= _mx(tn) < args_cli.tilt_stand:
            diagnosis.append(
                f"{r}: _tilt_from_quat reads {_mx(tt):.3f} rad at reset while the native "
                f"tilt is {_mx(tn):.3f} rad -- the fallen term sees a fall that the "
                f"orientation does not have (quaternion-order mismatch in _tilt_from_quat).")
    if "fallen" in first_fire and first_fire["fallen"]["step"] is not None:
        s = first_fire["fallen"]["step"]
        row = series[s]
        for r in probe.robots:
            diagnosis.append(
                f"fallen first fires at step {s} ({s * u.step_dt:.3f} s): {r} tilt_term "
                f"{row[r]['tilt_term']} vs tilt_native {row[r]['tilt_native']}, "
                f"base_z {row[r]['base_z']}, n_below {row[r]['n_below']}")
        tn_max = _mx([_mx(row[r]["tilt_native"]) for r in probe.robots])
        tt_max = _mx([_mx(row[r]["tilt_term"]) for r in probe.robots])
        if tt_max > 0.78 and tn_max <= 0.78:
            diagnosis.append(
                f"AT FIRST FIRE the termination's tilt ({tt_max:.3f}) exceeds 0.78 but no "
                f"robot's native tilt does ({tn_max:.3f}): the robot has not physically "
                f"fallen by the stack's own quaternion convention.")
        elif tn_max > 0.78:
            diagnosis.append(
                f"AT FIRST FIRE a robot's native tilt is {tn_max:.3f} > 0.78: it has "
                f"physically tipped over by step {s}.")

    return {
        "task": args_cli.task, "condition": cond, "num_envs": n, "steps": steps,
        "seed": seed, "error": err, "wall_s": round(wall, 2),
        "stack": {"quat_order": quat_order(), "bhl_stack": os.environ.get("BHL_STACK"),
                  "step_dt": u.step_dt, "physics_dt": u.physics_dt,
                  "decimation": u.cfg.decimation,
                  "max_episode_length": int(u.max_episode_length)},
        "init_z_offset": args_cli.init_z_offset,
        "configured_init_rot": {r: list(getattr(u.cfg.scene, r).init_state.rot)
                                for r in probe.robots},
        "configured_init_pos": {r: list(getattr(u.cfg.scene, r).init_state.pos)
                                for r in probe.robots},
        "root_quat_w_at_reset": q0,
        "termination_terms": _term_info(tm),
        "first_fire": first_fire,
        "implied_episode_length_per_env": implied,
        "max_tilt_native": _r(max_native), "max_tilt_native_step": max_native_at,
        "first_step_tilt_native_ge_threshold": first_tilt_step,
        "tilt_stand_threshold": args_cli.tilt_stand,
        "verdict": verdict,
        "diagnosis": diagnosis,
        "bodies": {r: {k: probe.meta[r][k] for k in
                       ("ankle_idx", "shoulder_idx", "sensor", "foot_sensor_names")}
                   | {"ankle_names": [probe.meta[r]["body_names"][i] for i in probe.meta[r]["ankle_idx"]],
                      "shoulder_names": [probe.meta[r]["body_names"][i] for i in probe.meta[r]["shoulder_idx"]],
                      "n_bodies": len(probe.meta[r]["body_names"])}
                   for r in probe.robots},
        "timeseries": series,
    }


def main() -> None:
    os.makedirs(args_cli.out_dir, exist_ok=True)
    cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    if args_cli.init_z_offset:
        for r in ("robot_a", "robot_b"):
            rc_ = getattr(cfg.scene, r)
            x, y, z = rc_.init_state.pos
            rc_.init_state.pos = (x, y, z + args_cli.init_z_offset)
            print(f"  {r}: init_state z {z:+.4f} -> {z + args_cli.init_z_offset:+.4f}", flush=True)
    env = gym.make(args_cli.task, cfg=cfg, disable_env_checker=True)
    env.reset(seed=args_cli.seed)
    probe = Probe(env)
    u = env.unwrapped
    print(f"task {args_cli.task}  envs {u.num_envs}  quat_order {quat_order()}  "
          f"step_dt {u.step_dt}  terms {list(u.termination_manager.active_terms)}  "
          f"action_dim {u.action_manager.total_action_dim}", flush=True)
    for r in probe.robots:
        m = probe.meta[r]
        print(f"  {r}: ankle bodies {[m['body_names'][i] for i in m['ankle_idx']]}  "
              f"shoulder bodies {[m['body_names'][i] for i in m['shoulder_idx']]}  "
              f"foot sensor bodies {m['foot_sensor_names']}", flush=True)

    rc = 0
    for cond in args_cli.conditions:
        res = run_condition(env, probe, cond, args_cli.steps, args_cli.seed)
        suffix = f"__{args_cli.tag}" if args_cli.tag else ""
        path = os.path.join(args_cli.out_dir, f"{args_cli.task}__{cond}{suffix}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
        ff = {t: v["step"] for t, v in res["first_fire"].items()}
        print(f"SPAWN-DIAG {args_cli.task} {cond} {res['verdict']}  first_fire {ff}  "
              f"implied_ep_len {res['implied_episode_length_per_env']}  "
              f"max_tilt_native {res['max_tilt_native']}", flush=True)
        for line in res["diagnosis"]:
            print(f"    {line}", flush=True)
        print(f"wrote {path}", flush=True)
        if res["error"]:
            rc = 1
    sys.stdout.flush()
    os._exit(rc)   # Kit can hang on a clean close; the JSONs are already on disk.


if __name__ == "__main__":
    main()
