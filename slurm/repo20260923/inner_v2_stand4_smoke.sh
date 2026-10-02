#!/bin/bash
# Inside the container (via bhl_exec), for gpu_v2_stand4_smoke.sbatch: the
# in-Isaac build probe of CubeToShelfStand4 (TaskV2-BHL-CubeToShelfStand4-Blind-v0).
# One task per Python process (one SimulationContext per process; Stand3's config
# is instantiated for the diff, never built). Writes one JSON,
# $STAND4_OUT/probe_${STAND4_JOB}.json, BEFORE env.close() / simulation_app.close()
# (Isaac's shutdown hard-exits the interpreter, so the exit code is never read).
# The verdict is computed on the host by the sbatch (stand4_mdp.smoke_verdict).
#
# Label: SCRIPTED probe (zero actions, teleported cube poses, no policy). It
# checks wiring -- terms, colliders, the filtered contact sensor, per-step
# logging, and (stage 8, "seat", added on review 2026-10-02) the success chain:
# a cube resting flat on a deck must seat, stay released and fire `success` with
# `placed` paying; one rolled 90 deg (tilt 90 deg, body z vs world up) must not
# -- never behaviour, and is never a result.
#
# Reads: STAND4_OUT, STAND4_JOB, TASK, NUM_ENVS, SEED, PROBE_STEPS, ENABLE_CAMERAS.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
: "${STAND4_OUT:?STAND4_OUT must be forwarded}"
: "${STAND4_JOB:?STAND4_JOB must be forwarded}"
: "${TASK:?TASK must be forwarded}"
OUT="$STAND4_OUT/probe_${STAND4_JOB}.json"
[ -e "$OUT" ] && { echo "inner_v2_stand4_smoke: refusing to overwrite $OUT"; exit 1; }
PROBE_PY=$(mktemp "${TMPDIR:-/tmp}/stand4_probe_XXXXXX.py")
trap 'rm -f "$PROBE_PY"' EXIT
cat > "$PROBE_PY" <<'PY'
"""CubeToShelfStand4 build probe (smoke, v60). SCRIPTED, wiring only, not a result."""
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

STAND3 = "TaskV2-BHL-CubeToShelfStand3-Blind-v0"
res = {"task": args.task, "num_envs": args.num_envs, "steps": args.steps, "seed": args.seed,
       "label": "SCRIPTED probe (zero actions, teleported cube); wiring only, never a result",
       "status": "error", "stage": "start"}
env = None


def fname(f):
    return f"{f.__module__}:{f.__name__}"


def scalars(d):
    return {k: v for k, v in d.items() if isinstance(v, (bool, int, float, str))}


def flatten(d, pre=""):
    """Dotted keys -> repr(value); an empty mapping is a leaf so it still shows."""
    out = {}
    if isinstance(d, dict):
        if not d and pre:
            out[pre] = "{}"
        for k, v in d.items():
            out.update(flatten(v, f"{pre}.{k}" if pre else str(k)))
    else:
        out[pre] = repr(d)
    return out


def current_stage():
    try:
        import isaaclab.sim as sim_utils
        return sim_utils.get_current_stage()
    except Exception:                                               # noqa: BLE001
        import omni.usd
        return omni.usd.get_context().get_stage()


try:
    import bhl_robust.tasks  # noqa: F401  registers the ids
    from bhl_robust.quat_order import native_quat, quat_order
    from bhl_robust.tasks import stand4_mdp as s4

    torch.manual_seed(args.seed)
    res["quat_order"] = quat_order()

    # (1) configs: Stand4 differs from Stand3 only where it says it does
    res["stage"] = "cfg_diff"
    cfg3 = gym.spec(STAND3).kwargs["env_cfg_entry_point"]()
    cfg4 = gym.spec(args.task).kwargs["env_cfg_entry_point"]()
    f3, f4 = flatten(cfg3.to_dict()), flatten(cfg4.to_dict())
    res["cfg_diff_keys"] = sorted(k for k in set(f3) | set(f4) if f3.get(k) != f4.get(k))
    res["cfg_diff_check"] = s4.cfg_diff_check(res["cfg_diff_keys"])
    res["usd_paths"] = {"stand3": {r: str(getattr(cfg3.scene, r).spawn.usd_path) for r in s4.RELEASE_ROBOTS},
                        "stand4": {r: str(getattr(cfg4.scene, r).spawn.usd_path) for r in s4.RELEASE_ROBOTS}}
    del cfg3, f3, f4

    # (2) build Stand4; the terms its managers actually hold
    res["stage"] = "build"
    cfg4.scene.num_envs = args.num_envs
    cfg4.sim.device = app_launcher.device
    cfg4.seed = args.seed
    env = gym.make(args.task, cfg=cfg4, disable_env_checker=True)
    env.reset()
    u = env.unwrapped
    dev, N = u.device, u.num_envs
    rm, tm, cm = u.reward_manager, u.termination_manager, u.curriculum_manager
    lt, pt, st = rm.get_term_cfg("lifting_object"), rm.get_term_cfg("placed"), tm.get_term_cfg("success")
    res["managers"] = {
        "reward_terms": list(rm.active_terms), "termination_terms": list(tm.active_terms),
        "curriculum_terms": list(cm.active_terms),
        "lifting_object": {"func": fname(lt.func), "weight": float(lt.weight), "params": scalars(lt.params)},
        "placed": {"func": fname(pt.func), "weight": float(pt.weight), "params": scalars(pt.params)},
        "success": {"func": fname(st.func), "params": scalars(st.params)},
    }

    # (3) hand colliders on the spawned robots (USD of env_0, the clone source;
    # step 7 is the all-envs physical proof)
    res["stage"] = "colliders"
    stage = current_stage()
    hc = {}
    for r in s4.RELEASE_ROBOTS:
        for side in ("left", "right"):
            path = f"/World/envs/env_0/{r}/arm_{side}_hand_link/hand_collider"
            prim = stage.GetPrimAtPath(path)
            ok = bool(prim and prim.IsValid())
            hc[path] = {"valid": ok,
                        "collision_api": bool(ok and "PhysicsCollisionAPI" in prim.GetAppliedSchemas()),
                        "approximation": (str(prim.GetAttribute("physics:approximation").Get())
                                          if ok else None)}
    res["hand_colliders"] = hc

    # (4) the cube's contact sensor, filtered one-to-many against 26 robot links
    res["stage"] = "sensor"
    sens = u.scene[s4.CUBE_SENSOR]
    view = sens.contact_view
    res["sensor"] = {"force_matrix_shape": list(s4._t(sens.data.force_matrix_w).shape),
                     "filter_count": int(view.filter_count), "sensor_count": int(view.sensor_count)}
    try:
        res["sensor"]["filter_paths_head"] = [str(p) for p in list(view.filter_paths)[:s4.N_RELEASE_BODIES]]
    except Exception as exc:                                        # noqa: BLE001
        res["sensor"]["filter_paths_error"] = repr(exc)

    # (5) per-step logging over zero-action steps
    res["stage"] = "steps"
    zero = torch.zeros((N, u.action_space.shape[-1]), device=dev)
    shapes_ok, in_extras, natural = True, True, []
    for _ in range(args.steps):
        env.step(zero)
        c = getattr(u, s4.CACHE_KEY, {})
        in_extras = in_extras and (u.extras.get(s4.EXTRAS_KEY) is c)
        for k in s4.CACHE_KEYS:
            v = c.get(k)
            shapes_ok = shapes_ok and (isinstance(v, float) if k == "minimal_height" else
                                       (isinstance(v, torch.Tensor) and tuple(v.shape) == (N,)))
        f = c.get("robot_force_max")
        natural.append(float(f.nan_to_num(nan=-1.0).max()) if isinstance(f, torch.Tensor) else None)
    c = getattr(u, s4.CACHE_KEY, {})
    res["cache"] = {"n_steps": args.steps, "keys_missing": [k for k in s4.CACHE_KEYS if k not in c],
                    "in_extras": bool(in_extras), "shapes_ok": bool(shapes_ok),
                    "last_means": {k: (float(v.float().mean()) if isinstance(v, torch.Tensor) else v)
                                   for k, v in c.items()},
                    "max_robot_force_per_step": natural}
    res["diag_values"] = {n: float(getattr(s4, fn)(u, None)) for n, fn in (
        ("pinch_dist", "pinch_distance_mean"), ("cube_tilt_deg", "cube_tilt_mean"),
        ("corner_clear", "corner_clearance_mean"), ("roll_ok", "roll_ok_share"),
        ("lift_paid", "lift_paid_share"), ("released", "released_share"),
        ("robot_force_max", "robot_force_max_mean"))}

    obj = u.scene["object"]
    ids = torch.arange(N, device=dev)
    ident = torch.tensor(native_quat((1.0, 0.0, 0.0, 0.0)), device=dev, dtype=torch.float32).repeat(N, 1)
    still = torch.zeros((N, 6), device=dev)

    def put_cube(pos_w):
        obj.write_root_pose_to_sim_index(root_pose=torch.cat([pos_w, ident], dim=-1), env_ids=ids)
        obj.write_root_velocity_to_sim_index(root_velocity=still, env_ids=ids)

    def mags():
        return torch.linalg.vector_norm(s4._t(sens.data.force_matrix_w).reshape(N, -1, 3), dim=-1)

    # (6) cube in free space, 2.5 m up: no robot link touches it -> released.
    # Envs that reset during the step are excluded (the reset puts the cube back).
    res["stage"] = "free_space"
    put_cube(u.scene.env_origins + torch.tensor([0.0, 0.0, 2.5], device=dev))
    env.step(zero)
    valid = ~u.reset_buf.bool()
    fm = s4._t(sens.data.force_matrix_w)
    worst, rel = s4.robot_force_max(fm), s4.released_mask(fm)
    res["free_space"] = {"n_valid": int(valid.sum()),
                         "max_force_valid": float(worst[valid].max()) if bool(valid.any()) else None,
                         "all_released_valid": bool(rel[valid].all()) if bool(valid.any()) else False}

    # (7) cube teleported onto robot_a's left hand-link origin: the overlay hull
    # (and the forearm capsule) must show on their columns of the matrix, and
    # the release predicate and the termination's cached reading must agree.
    res["stage"] = "overlap"
    ra = u.scene["robot_a"]
    hid = ra.find_bodies("arm_left_hand_link")[0][0]
    put_cube(s4._t(ra.data.body_pos_w)[:, hid, :].clone())
    col = s4.RELEASE_FILTER_EXPRS.index("{ENV_REGEX_NS}/robot_a/arm_left_hand_link")
    valid = torch.ones(N, dtype=torch.bool, device=dev)
    best_hand = torch.zeros(N, device=dev)
    best_any = torch.zeros(N, device=dev)
    inconsistent = cache_mismatch = 0
    per_step = []
    for _ in range(4):
        env.step(zero)
        valid &= ~u.reset_buf.bool()
        m = mags()
        m0 = m.nan_to_num(nan=0.0)
        best_hand = torch.maximum(best_hand, m0[:, col])
        best_any = torch.maximum(best_any, m0.max(dim=-1).values)
        rel = s4.released_mask(s4._t(sens.data.force_matrix_w))
        contact = (m0 >= s4.RELEASE_FORCE_N).any(dim=-1) | torch.isnan(m).any(dim=-1)
        inconsistent += int((valid & (contact == rel)).sum())
        cached = getattr(u, s4.CACHE_KEY)["released"]
        cache_mismatch += int((valid & (cached != rel)).sum())
        per_step.append({"n_valid": int(valid.sum()), "n_contact_valid": int((valid & contact).sum()),
                         "hand_col_max": float(m0[:, col].max()), "any_col_max": float(m0.max())})
    res["overlap"] = {"column": col, "n_valid": int(valid.sum()),
                      "n_hand_contact": int((valid & (best_hand >= s4.RELEASE_FORCE_N)).sum()),
                      "n_any_contact": int((valid & (best_any >= s4.RELEASE_FORCE_N)).sum()),
                      "hand_col_max_per_env": [float(v) for v in best_hand.cpu()],
                      "any_col_max_per_env": [float(v) for v in best_any.cpu()],
                      "inconsistent": inconsistent, "cache_mismatch": cache_mismatch,
                      "per_step": per_step}

    # (8) seat: the success chain with the real sensor (review 2026-10-02).
    # Fresh reset, then each env's cube rests on a deck (s4.seat_pose_local:
    # flat / rolled90 control / tilt10 consistency-only), zero velocity, zero
    # actions for s4.SMOKE_SEAT_STEPS steps. After every step: the success
    # termination, reset_buf, the cache (hold, seated, released, placed_now,
    # tilt_deg) and the `placed` reward of that step. s4.seat_stage_eval follows
    # each env until its first reset (a success resets its env on the step it
    # fires); the host verdict recomputes it from these raw records.
    res["stage"] = "seat"
    env.reset()
    hold_before = getattr(u, s4.HOLD_KEY, None)
    poses = [s4.seat_pose_local(i) for i in range(N)]
    seat_pos = u.scene.env_origins + torch.tensor([p for p, _ in poses], device=dev, dtype=torch.float32)
    seat_quat = torch.tensor([native_quat(q) for _, q in poses], device=dev, dtype=torch.float32)
    obj.write_root_pose_to_sim_index(root_pose=torch.cat([seat_pos, seat_quat], dim=-1), env_ids=ids)
    obj.write_root_velocity_to_sim_index(root_velocity=still, env_ids=ids)
    i_placed = list(rm.active_terms).index("placed")
    kind = {"success": bool, "reset": bool, "hold": int, "seated": bool, "released": bool,
            "placed_now": bool, "tilt_deg": float, "placed_reward": float}
    records, seat_diag = [], []
    for _ in range(s4.SMOKE_SEAT_STEPS):
        env.step(zero)
        c = getattr(u, s4.CACHE_KEY)
        raw = {"success": tm.get_term("success"), "reset": u.reset_buf, "hold": c["hold"],
               "seated": c["seated"], "released": c["released"], "placed_now": c["placed_now"],
               "tilt_deg": c["tilt_deg"], "placed_reward": rm._step_reward[:, i_placed]}
        records.append({k: [kind[k](x) for x in raw[k].detach().cpu().tolist()]
                        for k in s4.SEAT_RECORD_KEYS})
        p_loc, spd = s4.sm._obj_state(u)
        seat_diag.append({"z": [round(float(v), 4) for v in p_loc[:, 2].cpu()],
                          "speed": [round(float(v), 4) for v in spd.cpu()],
                          "robot_force_max": [float(v) for v in c["robot_force_max"].cpu()]})
    groups = [s4.seat_group(i) for i in range(N)]
    res["seat"] = {"steps": s4.SMOKE_SEAT_STEPS, "groups": groups,
                   "sides": [s4.seat_side(i) for i in range(N)],
                   "poses_local": [list(p) for p, _ in poses],
                   "roll_deg": [s4.SEAT_ROLL_DEG[g] for g in groups],
                   "hold_before": (None if hold_before is None
                                   else [int(v) for v in hold_before.cpu()]),
                   "records": records, "diag": seat_diag,
                   "summary": s4.seat_stage_eval(groups, records)}
    res["status"] = "ok"
    res["stage"] = "done"
except Exception:                                                  # noqa: BLE001
    res["traceback"] = traceback.format_exc()

with open(args.out, "w") as f:
    json.dump(res, f, indent=1, default=str)
seat_short = {k: v for k, v in res.get("seat", {}).get("summary", {}).items()
              if k not in ("fire_events", "inconsistent_head", "fire_step")}
print(f"STAND4-PROBE status={res['status']} stage={res['stage']} order={res.get('quat_order')} "
      f"cfg_diff={res.get('cfg_diff_check')} sensor={res.get('sensor', {}).get('force_matrix_shape')} "
      f"filters={res.get('sensor', {}).get('filter_count')} "
      f"colliders={[v.get('collision_api') for v in res.get('hand_colliders', {}).values()]} "
      f"cache_missing={res.get('cache', {}).get('keys_missing')} free={res.get('free_space')} "
      f"overlap_hand={res.get('overlap', {}).get('n_hand_contact')}/{res.get('overlap', {}).get('n_valid')} "
      f"seat={seat_short} "
      f"-> {args.out}", flush=True)
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
echo "=== inner_v2_stand4_smoke: $TASK | envs ${NUM_ENVS:-16} | steps ${PROBE_STEPS:-30} | seed ${SEED:-100} | stack ${BHL_STACK:-?} | $(date) ==="
set +e
timeout 1500 "$PY" "$PROBE_PY" --task "$TASK" --num_envs "${NUM_ENVS:-16}" \
    --steps "${PROBE_STEPS:-30}" --seed "${SEED:-100}" --out "$OUT" ${CAM[@]+"${CAM[@]}"} 2>&1 \
    | tee "$STAND4_OUT/probe_${STAND4_JOB}.log" | grep -E "^STAND4-PROBE|^Traceback|^[A-Za-z]+Error:" || true
set -e
[ -s "$OUT" ] && echo "wrote $OUT" || echo "inner_v2_stand4_smoke: no probe JSON (see $STAND4_OUT/probe_${STAND4_JOB}.log)"
exit 0
