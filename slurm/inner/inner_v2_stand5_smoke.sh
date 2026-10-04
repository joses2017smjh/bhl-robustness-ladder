#!/bin/bash
# Inside the container (via bhl_exec), for slurm/repo20260923/gpu_v2_stand5_smoke.sbatch:
# the in-Isaac build probe of CubeToShelfStand5 (TaskV2-BHL-CubeToShelfStand5-Blind-v0).
# One task per Python process (one SimulationContext per process; Stand4's config
# is instantiated for the diff and the expected term lists, never built). Writes
# one JSON, $STAND5_OUT/probe_${STAND5_JOB}.json, BEFORE env.close() /
# simulation_app.close() (Isaac's shutdown hard-exits the interpreter, so the exit
# code is never read). The verdict is computed on the host by the sbatch
# (stand5_mdp.smoke_verdict5), which recomputes the observation and curriculum
# clauses from the raw records written here.
#
# Label: SCRIPTED probe (zero actions, teleported cube poses, no policy). It checks
# wiring -- the two observation terms (place, width, axis order on the live stack)
# and the curriculum's promotion test -- never behaviour, and is never a result.
#
# Stages: cfg_diff (Stand4 vs Stand5 cfg.to_dict(); Stand4's expected term lists),
# build (Stand5's managers), obs (cube at rest in free space in five known
# orientations, one step: the observation tails, both terms, the raw quaternions
# and Isaac Lab's own matrix_from_quat reference), curriculum (after a fresh reset
# the cube in four pose groups, one step, then the curriculum term and Stand4's
# coop_lift_mdp.lift_height_curriculum called directly on chosen envs at known
# levels and pinch distances; the probe's changes are restored), steps (30
# zero-action steps: the env's own resets call the curriculum with int32 ids).
#
# Reads: STAND5_OUT, STAND5_JOB, TASK, NUM_ENVS, SEED, PROBE_STEPS, ENABLE_CAMERAS.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
: "${STAND5_OUT:?STAND5_OUT must be forwarded}"
: "${STAND5_JOB:?STAND5_JOB must be forwarded}"
: "${TASK:?TASK must be forwarded}"
OUT="$STAND5_OUT/probe_${STAND5_JOB}.json"
[ -e "$OUT" ] && { echo "inner_v2_stand5_smoke: refusing to overwrite $OUT"; exit 1; }
PROBE_PY=$(mktemp "${TMPDIR:-/tmp}/stand5_probe_XXXXXX.py")
trap 'rm -f "$PROBE_PY"' EXIT
cat > "$PROBE_PY" <<'PY'
"""CubeToShelfStand5 build probe (smoke, v60). SCRIPTED, wiring only, not a result."""
import argparse, json, traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

res = {"task": args.task, "num_envs": args.num_envs, "steps": args.steps, "seed": args.seed,
       "label": "SCRIPTED probe (zero actions, teleported cube); wiring only, never a result",
       "status": "error", "stage": "start"}
env = None


def fname(f):
    return f"{f.__module__}:{f.__name__}"


def scalars(d):
    return {k: v for k, v in d.items() if isinstance(v, (bool, int, float, str))}


def rows(t):
    return t.detach().double().cpu().tolist()


try:
    import bhl_robust.tasks  # noqa: F401  registers the ids
    from isaaclab.managers import (CurriculumTermCfg, ObservationTermCfg, RewardTermCfg,
                                   TerminationTermCfg)
    from isaaclab.utils.math import matrix_from_quat
    from bhl_robust.quat_order import native_quat, quat_order
    from bhl_robust.tasks import coop_lift_mdp as coop
    from bhl_robust.tasks import stand5_mdp as s5
    s4 = s5.s4

    torch.manual_seed(args.seed)
    res["quat_order"] = quat_order()

    # (1) configs: Stand5 differs from Stand4 only where it says it does
    res["stage"] = "cfg_diff"
    cfg4 = gym.spec(s5.TASK4_ID).kwargs["env_cfg_entry_point"]()
    cfg5 = gym.spec(args.task).kwargs["env_cfg_entry_point"]()
    f4, f5 = s5.flatten_dump(cfg4.to_dict()), s5.flatten_dump(cfg5.to_dict())
    res["cfg_diff_keys"] = sorted(k for k in set(f4) | set(f5) if f4.get(k) != f5.get(k))
    res["cfg_diff_check"] = s5.diff_check(res["cfg_diff_keys"], s5.CFG_DIFF_ALLOWED5)

    def active(grp, kind):
        return [n for n, t in vars(grp).items() if isinstance(t, kind)]

    lh4 = cfg4.curriculum.lift_height
    res["stand4_expected"] = {
        "reward_terms": active(cfg4.rewards, RewardTermCfg),
        "termination_terms": active(cfg4.terminations, TerminationTermCfg),
        "curriculum_terms": active(cfg4.curriculum, CurriculumTermCfg),
        "obs_terms": {g: active(getattr(cfg4.observations, g), ObservationTermCfg)
                      for g in s5.OBS_GROUPS},
        "lift_height_func": fname(lh4.func), "lift_height_params": scalars(lh4.params)}
    del cfg4, f4, f5

    # (2) build Stand5; the terms its managers actually hold
    res["stage"] = "build"
    cfg5.scene.num_envs = args.num_envs
    cfg5.sim.device = app_launcher.device
    cfg5.seed = args.seed
    env = gym.make(args.task, cfg=cfg5, disable_env_checker=True)
    env.reset()
    u = env.unwrapped
    dev, N = u.device, u.num_envs
    rm, tm, cm, om = (u.reward_manager, u.termination_manager, u.curriculum_manager,
                      u.observation_manager)
    lt, pt, st = rm.get_term_cfg("lifting_object"), rm.get_term_cfg("placed"), tm.get_term_cfg("success")

    def lift_height_cfg():
        return cm._term_cfgs[list(cm.active_terms).index("lift_height")]

    lh = lift_height_cfg()
    ocfg = {g: dict(zip(om.active_terms[g], om._group_obs_term_cfgs[g])) for g in s5.OBS_GROUPS}
    res["managers"] = {
        "reward_terms": list(rm.active_terms), "termination_terms": list(tm.active_terms),
        "curriculum_terms": list(cm.active_terms),
        "obs_terms": {g: list(om.active_terms[g]) for g in s5.OBS_GROUPS},
        "obs_term_dims": {g: [list(d) for d in om.group_obs_term_dim[g]] for g in s5.OBS_GROUPS},
        "obs_group_dims": {g: list(om.group_obs_dim[g]) for g in s5.OBS_GROUPS},
        "lifting_object": {"func": fname(lt.func), "weight": float(lt.weight)},
        "placed": {"func": fname(pt.func), "weight": float(pt.weight)},
        "success": {"func": fname(st.func)},
        "lift_height": {"func": fname(lh.func), "params": scalars(lh.params)},
        "new_obs": {g: {n: {"func": fname(c.func), "robot": c.params["robot_cfg"].name,
                            "noise": repr(c.noise), "scale": repr(c.scale),
                            "clip": None if c.clip is None else [float(v) for v in c.clip]}
                        for n, c in ocfg[g].items() if n in s5.OBS_TERMS5}
                    for g in s5.OBS_GROUPS},
    }

    obj = u.scene["object"]
    ids = torch.arange(N, device=dev)
    still = torch.zeros((N, 6), device=dev)
    zero = torch.zeros((N, u.action_space.shape[-1]), device=dev)

    def put_cube(pos_local, quats_wxyz):
        """Teleport every env's cube (env-local position, (w, x, y, z) literal
        through native_quat), at rest."""
        pos = u.scene.env_origins + torch.tensor(pos_local, device=dev, dtype=torch.float32)
        quat = torch.tensor([native_quat(q) for q in quats_wxyz], device=dev, dtype=torch.float32)
        obj.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=ids)
        obj.write_root_velocity_to_sim_index(root_velocity=still, env_ids=ids)

    # (3) observation: the cube at rest in free space in five known orientations
    # (s5.AXIS_CASES), one zero-action step; with zero angular velocity and no
    # contact the orientation is what was written. Recorded raw; the host
    # recomputes A4/A5 (s5.obs_stage_eval).
    res["stage"] = "obs"
    cases = [s5.axis_case(i) for i in range(N)]
    put_cube([[0.0, 0.0, s5.FREE_Z]] * N, [s5.AXIS_CASES[c] for c in cases])
    obs, *_ = env.step(zero)
    valid = ~u.reset_buf.bool()
    cube_q = s5._t(obj.data.root_quat_w).clone()
    robot_q = {r: s5._t(u.scene[r].data.root_quat_w).clone() for _, r in s5.OBS_TERM_ROBOTS}
    zc = matrix_from_quat(cube_q)[:, :, 2]                     # Isaac Lab's own convention
    terms, ref = {}, {}
    for name, robot in s5.OBS_TERM_ROBOTS:
        terms[name] = s5.object_zaxis_in_root(u, robot_cfg=ocfg["policy"][name].params["robot_cfg"])
        ref[name] = torch.einsum("nji,nj->ni", matrix_from_quat(robot_q[robot]), zc)
    res["obs"] = {"cases": cases, "valid": [bool(v) for v in valid.cpu()],
                  "cube_quat_native": rows(cube_q),
                  "robot_quat_native": {r: rows(q) for r, q in robot_q.items()},
                  "term_values": {n: rows(v) for n, v in terms.items()},
                  "isaac_ref_values": {n: rows(v) for n, v in ref.items()},
                  "world_zaxis_isaac": rows(zc),
                  "tail": {g: rows(obs[g][:, -len(s5.OBS_TERMS5) * s5.OBS_TERM_DIM:])
                           for g in s5.OBS_GROUPS},
                  "obs_width": {g: int(obs[g].shape[-1]) for g in s5.OBS_GROUPS}}

    # (4) curriculum: after a fresh reset the cube in s5.CURR_GROUPS poses, one
    # zero-action step, then the Stand5 term and Stand4's function called
    # directly on envs picked by the flags of the live state, at known levels
    # and pinch distances. Recorded raw; the host recomputes A6
    # (s5.curriculum_stage_eval). What the probe changes is restored after.
    res["stage"] = "curriculum"
    env.reset()
    groups = [s5.curr_group(i) for i in range(N)]
    poses = [s5.curr_pose_local(i) for i in range(N)]
    put_cube([p for p, _ in poses], [q for _, q in poses])
    env.step(zero)
    valid = ~u.reset_buf.bool()
    lh = lift_height_cfg()
    p_loc, _ = s4.sm._obj_state(u)
    w, x, y, z = s4._cube_quat_wxyz(u)
    roll_ok = s4.roll_proof_mask(p_loc, w, x, y, z, clearance=lh.params["clearance"],
                                 tilt_max_deg=lh.params["tilt_max_deg"])
    z_world = s5._t(obj.data.root_pos_w)[:, 2].clone()
    spawn = float(u.cfg.object_spawn_z)
    p4 = {k: v for k, v in lh.params.items() if k not in ("clearance", "tilt_max_deg")}
    lo_h, hi_h = float(lh.params["min_height"]), float(lh.params["max_height"])
    saved = {"lift_h": getattr(u, "_bhl_lift_h", None), "pinch_d": getattr(u, "_bhl_pinch_d", None),
             "minimal_height": rm.get_term_cfg("lifting_object").params["minimal_height"]}
    centre_lo = z_world > spawn + lo_h
    sel_roll = torch.nonzero(roll_ok & valid).squeeze(-1)
    sel_centre = torch.nonzero(~roll_ok & centre_lo & valid).squeeze(-1)
    sel_all = torch.nonzero(valid).squeeze(-1).int()           # the dtype _reset_idx passes
    plan = [("c1_roll_proof_pinch_far", sel_roll, lo_h, 1.0),
            ("c2_centre_high_pinched_not_roll_proof", sel_centre, lo_h, 0.0),
            ("c3_same_envs_from_the_cap", sel_centre, hi_h, 0.0),
            ("c4_all_valid_int32_ids", sel_all, lo_h, 0.0)]
    calls = []
    for name, sel, pre, pinch in plan:
        u._bhl_pinch_d = torch.full((N,), pinch, device=dev)
        u._bhl_lift_h = pre
        r5 = float(lh.func(u, sel, **lh.params))
        mh5 = float(rm.get_term_cfg("lifting_object").params["minimal_height"])
        h5 = float(u._bhl_lift_h)
        u._bhl_lift_h = pre
        r4 = float(coop.lift_height_curriculum(u, sel, **p4))
        calls.append({"name": name, "ids": [int(i) for i in sel.cpu()], "ids_dtype": str(sel.dtype),
                      "pre": pre, "pinch_d": pinch, "r5": r5, "minimal_height_after_r5": mh5,
                      "lift_h_after_r5": h5, "r4": r4})
    for attr, key in (("_bhl_lift_h", "lift_h"), ("_bhl_pinch_d", "pinch_d")):
        if saved[key] is None:
            if hasattr(u, attr):
                delattr(u, attr)
        else:
            setattr(u, attr, saved[key])
    t_cfg = rm.get_term_cfg("lifting_object")
    t_cfg.params["minimal_height"] = saved["minimal_height"]
    rm.set_term_cfg("lifting_object", t_cfg)
    res["curriculum"] = {
        "groups": groups, "valid": [bool(v) for v in valid.cpu()], "spawn_z": spawn,
        "params": scalars(lh.params), "stand4_params": scalars(p4),
        "p_local": rows(p_loc), "quat_native": rows(s5._t(obj.data.root_quat_w)),
        "z_world": [float(v) for v in z_world.cpu()],
        "roll_ok": [bool(v) for v in roll_ok.cpu()],
        "clearance": [float(v) for v in s4.support_clearance(p_loc, w, x, y, z).cpu()],
        "tilt_deg": [float(v) for v in s4.cube_tilt_deg(w, x, y, z).cpu()],
        "restored": {"lift_h": None if saved["lift_h"] is None else float(saved["lift_h"]),
                     "minimal_height": float(saved["minimal_height"])},
        "calls": calls}

    # (5) zero-action steps: the env's own resets call the curriculum (int32 ids)
    res["stage"] = "steps"
    trace = []
    for _ in range(args.steps):
        env.step(zero)
        trace.append({"n_reset": int(u.reset_buf.sum()),
                      "lift_h": float(getattr(u, "_bhl_lift_h", float("nan"))),
                      "minimal_height": float(rm.get_term_cfg("lifting_object").params["minimal_height"])})
    res["steps"] = {"n_steps": args.steps, "n_resets": sum(t["n_reset"] for t in trace),
                    "trace": trace}
    res["status"] = "ok"
    res["stage"] = "done"
except Exception:                                                  # noqa: BLE001
    res["traceback"] = traceback.format_exc()

with open(args.out, "w") as f:
    json.dump(res, f, indent=1, default=str)
print(f"STAND5-PROBE status={res['status']} stage={res['stage']} order={res.get('quat_order')} "
      f"cfg_diff={res.get('cfg_diff_check')} "
      f"obs_width={res.get('obs', {}).get('obs_width')} "
      f"curriculum_calls={[(c['name'], len(c['ids']), c['r5'], c['r4']) for c in res.get('curriculum', {}).get('calls', [])]} "
      f"steps_resets={res.get('steps', {}).get('n_resets')} -> {args.out}", flush=True)
if res["status"] != "ok":
    print(res.get("traceback", ""), flush=True)
try:
    if env is not None:
        env.close()
except Exception:                                                  # noqa: BLE001
    pass
simulation_app.close()
PY
CAM=(); [ "${ENABLE_CAMERAS:-0}" = "1" ] && CAM=(--enable_cameras)
echo "=== inner_v2_stand5_smoke: $TASK | envs ${NUM_ENVS:-16} | steps ${PROBE_STEPS:-30} | seed ${SEED:-100} | stack ${BHL_STACK:-?} | $(date) ==="
set +e
timeout 1500 "$PY" "$PROBE_PY" --task "$TASK" --num_envs "${NUM_ENVS:-16}" \
    --steps "${PROBE_STEPS:-30}" --seed "${SEED:-100}" --out "$OUT" ${CAM[@]+"${CAM[@]}"} 2>&1 \
    | tee "$STAND5_OUT/probe_${STAND5_JOB}.log" | grep -E "^STAND5-PROBE|^Traceback|^[A-Za-z]+Error:" || true
set -e
[ -s "$OUT" ] && echo "wrote $OUT" || echo "inner_v2_stand5_smoke: no probe JSON (see $STAND5_OUT/probe_${STAND5_JOB}.log)"
exit 0
