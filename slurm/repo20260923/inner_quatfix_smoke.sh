#!/bin/bash
# Inside the container (via bhl_exec), for gpu_quatfix_smoke.sbatch: the
# in-Isaac quaternion probe of the 2026-09-28 payload-quaternion fix. One task
# per Python process (one SimulationContext per process). Writes one JSON,
# $QUATFIX_OUT/probe_${QUATFIX_JOB}.json, BEFORE simulation_app.close() (Isaac's
# shutdown hard-exits the interpreter, so the exit code is never read). The
# verdict is computed on the host by the sbatch, from this JSON and the
# training run's TensorBoard -- see the sbatch header for the rule.
#
# Reads: QUATFIX_OUT, QUATFIX_JOB, TASK, NUM_ENVS, SEED, PROBE_STEPS, ENABLE_CAMERAS.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
: "${QUATFIX_OUT:?QUATFIX_OUT must be forwarded}"
: "${QUATFIX_JOB:?QUATFIX_JOB must be forwarded}"
: "${TASK:?TASK must be forwarded}"
OUT="$QUATFIX_OUT/probe_${QUATFIX_JOB}.json"
[ -e "$OUT" ] && { echo "inner_quatfix_smoke: refusing to overwrite $OUT"; exit 1; }
PROBE_PY=$(mktemp "${TMPDIR:-/tmp}/quatfix_probe_XXXXXX.py")
trap 'rm -f "$PROBE_PY"' EXIT
cat > "$PROBE_PY" <<'PY'
"""Payload-quaternion probe (2026-09-28 fix), one TaskV2 task, v60.

Label: SCRIPTED probe (zero / N(0,1) actions, no policy); checks config and
reward wiring only, not behaviour.
"""
import argparse, json, math, traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=20)
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
       "label": "SCRIPTED probe (zero / N(0,1) actions); config + reward wiring only",
       "status": "error"}


def flt(t):
    return [float(v) for v in t.detach().flatten().cpu()]


try:
    import bhl_robust.tasks  # noqa: F401  registers the ids
    from bhl_robust.quat_order import native_quat, quat_order
    from bhl_robust.tasks import coop_lift_mdp as coop
    from bhl_robust.tasks import stand_mdp as sm
    from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz

    torch.manual_seed(args.seed)
    res["quat_order"] = quat_order()
    res["native_identity"] = list(native_quat((1.0, 0.0, 0.0, 0.0)))

    cfg = gym.spec(args.task).kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = app_launcher.device
    cfg.seed = args.seed
    res["cfg_object_rot"] = [float(v) for v in cfg.scene.object.init_state.rot]
    res["cfg_robot_rots"] = {n: [float(v) for v in getattr(cfg.scene, n).init_state.rot]
                             for n in ("robot_a", "robot_b") if hasattr(cfg.scene, n)}
    env = gym.make(args.task, cfg=cfg, disable_env_checker=True)
    env.reset()
    u = env.unwrapped
    dev = u.device
    obj = u.scene["object"]

    def live():
        """(object_tilt_l2, _cube_tilt_deg, Isaac's own (1-R22)^2, Isaac's own acos(R22) deg)."""
        q = coop._t(obj.data.root_quat_w).clone()
        r22 = matrix_from_quat(q)[..., 2, 2].clamp(-1.0, 1.0)
        return (coop.object_tilt_l2(u), sm._cube_tilt_deg(u),
                torch.square(1.0 - r22), torch.rad2deg(torch.acos(r22)), q)

    ot, td, ot_ref, td_ref, q = live()
    res["reset"] = {"root_quat_w_env0": flt(q[0]), "object_tilt_max": max(flt(ot)),
                    "cube_tilt_deg_max": max(flt(td)),
                    "cube_tilted_share": sm.cube_tilted_share(u, None)}
    err_ot, err_td = [float((ot - ot_ref).abs().max())], [float((td - td_ref).abs().max())]
    tilt_max_seen = [max(flt(td))]
    n_act = u.action_space.shape[-1]
    for _ in range(args.steps):
        env.step(torch.randn((args.num_envs, n_act), device=dev))
        ot, td, ot_ref, td_ref, _q = live()
        err_ot.append(float((ot - ot_ref).abs().max()))
        err_td.append(float((td - td_ref).abs().max()))
        tilt_max_seen.append(max(flt(td)))
    res["live_agreement"] = {"max_abs_err_object_tilt": max(err_ot), "max_abs_err_tilt_deg": max(err_td),
                             "max_cube_tilt_deg_seen": max(tilt_max_seen), "n_checks": len(err_ot)}

    # Known rotations, built by Isaac Lab's own writer (native order), fed
    # through the repo's two readers on a stand-in scene.
    names = ["identity", "roll_x_90", "pitch_y_90", "yaw_z_90", "roll_x_180"]
    rpy = torch.tensor([[0, 0, 0], [math.pi / 2, 0, 0], [0, math.pi / 2, 0],
                        [0, 0, math.pi / 2], [math.pi, 0, 0]], dtype=torch.float32, device=dev)
    qk = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])

    class _Obj:
        class data:
            root_quat_w = qk

    fake = type("E", (), {"scene": {"object": _Obj}})()
    res["known"] = {"names": names, "quats_native": [flt(r) for r in qk],
                    "object_tilt": flt(coop.object_tilt_l2(fake)),
                    "cube_tilt_deg": flt(sm._cube_tilt_deg(fake)),
                    "want_object_tilt": [0.0, 1.0, 1.0, 0.0, 4.0],
                    "want_cube_tilt_deg": [0.0, 90.0, 90.0, 0.0, 180.0],
                    "isaac_identity_equals_cfg_rot":
                        [round(v, 6) for v in flt(qk[0])] == [round(v, 6) for v in res["cfg_object_rot"]]}
    res["status"] = "ok"
    env.close()
except Exception:                                                 # noqa: BLE001
    res["traceback"] = traceback.format_exc()

with open(args.out, "w") as f:
    json.dump(res, f, indent=1)
print(f"QUATFIX-PROBE status={res['status']} order={res.get('quat_order')} "
      f"cfg_object_rot={res.get('cfg_object_rot')} reset={res.get('reset')} "
      f"known_object_tilt={res.get('known', {}).get('object_tilt')} "
      f"known_cube_tilt_deg={res.get('known', {}).get('cube_tilt_deg')} "
      f"live={res.get('live_agreement')} -> {args.out}", flush=True)
if res["status"] != "ok":
    print(res.get("traceback", ""), flush=True)
simulation_app.close()
PY
CAM=(); [ "${ENABLE_CAMERAS:-0}" = "1" ] && CAM=(--enable_cameras)
echo "=== inner_quatfix_smoke: $TASK | envs ${NUM_ENVS:-16} | steps ${PROBE_STEPS:-20} | seed ${SEED:-100} | stack ${BHL_STACK:-?} | $(date) ==="
set +e
timeout 1200 "$PY" "$PROBE_PY" --task "$TASK" --num_envs "${NUM_ENVS:-16}" \
    --steps "${PROBE_STEPS:-20}" --seed "${SEED:-100}" --out "$OUT" ${CAM[@]+"${CAM[@]}"} 2>&1 \
    | tee "$QUATFIX_OUT/probe_${QUATFIX_JOB}.log" | grep -E "^QUATFIX-PROBE|^Traceback|^[A-Za-z]+Error:" || true
set -e
[ -s "$OUT" ] && echo "wrote $OUT" || echo "inner_quatfix_smoke: no probe JSON (see $QUATFIX_OUT/probe_${QUATFIX_JOB}.log)"
exit 0
