#!/usr/bin/env python3
"""Isaac scripted eval for C0 / C2 / C5.

Must run under SimulationApp (sbatch). Deformable cells are cost-gated.
Success is the geometric predicate, not a rendered clip.
"""

from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rung", default="C0", choices=("C0", "C0B", "C0F", "C2", "C2F", "C5"))
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--max_steps", type=int, default=80)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--garment", default=None,
                    help="catalog garment for the one-garment rungs (C0, C0F); default shirt_a")
parser.add_argument("--policy", default="scripted", choices=("scripted", "hold"),
                    help="hold: the arm never moves -- the control for free-base balance")
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import bhl_robust.tasks  # noqa: F401,E402
from bhl_robust.cloth.cost import report_cost  # noqa: E402
from bhl_robust.cloth.garments import GARMENT_BY_NAME  # noqa: E402
from bhl_robust.cloth.layout import basket_center  # noqa: E402
from bhl_robust.cloth.metrics import EpisodeMetrics, RunMetrics  # noqa: E402
from bhl_robust.cloth.scripted import scripted_action  # noqa: E402
from bhl_robust.tasks.cloth_sort_env_cfg import build_cfg  # noqa: E402
from bhl_robust.tasks.cloth_sort_mdp import _local_pos  # noqa: E402


TASK = {
    "C0": "ClothSort-BHL-Rigid-Oracle-v0",
    # Free base in the knee-1.0 stance with the leg controller: a balance probe, not a sort.
    "C0B": "ClothSort-BHL-RigidBalance-Oracle-v0",
    # Fixed base: diagnostic, labelled as such in every result it produces.
    "C0F": "ClothSort-BHL-RigidFixedBase-Oracle-v0",
    "C2": "ClothSort-BHL-Deformable-Oracle-v0",
    "C2F": "ClothSort-BHL-DeformableFixedBase-Oracle-v0",
    "C5": "ClothSort-BHL-RigidFive-Oracle-v0",
}
GARMENT = {
    "C0": "shirt_a",
    "C0F": "shirt_a",
    "C0B": "shirt_a",
    "C2": "shirt_a",
    "C2F": "shirt_a",
    "C5": "sock_a",
}


def _xy(env, name: str = "garment_0") -> np.ndarray:
    p = _local_pos(env.unwrapped, name)[0, :2]
    return p.detach().cpu().numpy()


def _term_fired(tm, name: str) -> bool:
    """Did the named termination term fire this step?

    Read through the public ``get_term``. ``_term_dones`` is a
    ``(num_envs, n_terms)`` tensor here, not a dict keyed by name, so the old
    ``"success" in tm._term_dones`` raised ``RuntimeError`` and a bare
    ``except: pass`` turned that into a silent ``False``. ``success_rate``
    and ``fall_rate`` could therefore never report anything but 0.0 -- a
    headline number that was structurally incapable of moving.

    Raises rather than defaulting if the term is absent: silence is what made
    the previous version wrong.
    """
    active = list(tm.active_terms)
    if name not in active:
        raise KeyError(f"termination term {name!r} is not active; have {active}")
    return bool(tm.get_term(name)[0].item())


def main() -> None:
    tid = TASK[args_cli.rung]
    physics = "deformable" if args_cli.rung in ("C2", "C2F") else "rigid"
    n_env = args_cli.num_envs
    if physics == "deformable":
        n_env = min(n_env, 8)
        rep = report_cost(
            num_envs=n_env, iterations=args_cli.episodes,
            steps_per_iter=args_cli.max_steps, physics="deformable", max_hours=4.0,
        )
        print(json.dumps(rep.as_dict(), indent=2))
        if not rep.accepted:
            raise SystemExit(rep.reason)

    cfg = build_cfg(
        gym.spec(tid).kwargs["env_cfg_entry_point"],
        num_envs=n_env, device=app_launcher.device,
    )
    # Clip mode (BHL_CAMERA_CLIP=1): short RL steps with the sweep schedule
    # latched across them, and a static camera sensor over the table -- a camera
    # sensor, not the viewport, which draws stale poses (docs/ISAAC_RENDER.md 10).
    clip_on = os.environ.get("BHL_CAMERA_CLIP", "0") == "1"
    clip_dir = os.environ.get("BHL_CLIP_DIR", str(Path(args_cli.out or "clip").with_suffix("")) + "_frames")
    if args_cli.garment:
        if args_cli.rung not in ("C0", "C0B", "C0F", "C2", "C2F"):
            raise SystemExit("--garment applies to the one-garment rungs C0, C0F, C2 and C2F")
        from bhl_robust.tasks.cloth_sort_env_cfg import use_garment
        use_garment(cfg, args_cli.garment)
    if args_cli.policy == "hold":
        cfg.actions.sweep.hold = True
    if clip_on:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg
        cfg.actions.sweep.latch = True
        cfg.decimation = 10
        cfg.sim.render_interval = cfg.decimation
        cfg.scene.clip_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/clip_cam", height=360, width=640, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=16.0),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        )
    env = gym.make(tid, cfg=cfg, disable_env_checker=True)
    if clip_on:
        from PIL import Image
        os.makedirs(clip_dir, exist_ok=True)
        cam = env.unwrapped.scene["clip_cam"]
        o = env.unwrapped.scene.env_origins
        eye = torch.tensor([0.35, 1.05, 0.95], device=o.device) + o
        target = torch.tensor([-0.28, 0.22, 0.28], device=o.device) + o
        cam.set_world_poses_from_view(eye, target)
        n_frames = 0
        args_cli.max_steps = max(args_cli.max_steps, int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation)) + 2)
    spec = GARMENT_BY_NAME[args_cli.garment or GARMENT[args_cli.rung]]
    metrics = RunMetrics(
        physics="isaac_deformable" if physics == "deformable" else "isaac_rigid",
        policy=args_cli.policy,
        num_envs=n_env,
        n_deformables=1 if physics == "deformable" else 0,
    )
    sweep = env.unwrapped.action_manager.get_term("sweep")
    trace_path = None
    t0 = time.perf_counter()
    env_steps = 0
    fars: list[float] = []
    nonfinite_eps: list[bool] = []
    finite_success: list[bool] = []
    for ep in range(args_cli.episodes):
        env.reset()
        em = EpisodeMetrics(n_garments=5 if args_cli.rung == "C5" else 1)
        done = False
        step = 0
        # Garment motion is read only between steps that did not end the
        # episode: Isaac resets an env inside the step that terminates it, so a
        # position read after that step is the reset position, not where the
        # sweep left the garment.
        g0 = _xy(env).copy()
        prev, far, measured = g0.copy(), 0.0, False
        nonfinite0 = int(sweep.nonfinite[0].item())
        while not done and step < args_cli.max_steps:
            # Each env's own garment, read the way the sweep term reads it:
            # through the robot's actual root pose, with the garment's yaw. This
            # used to send every env the action computed for env 0's garment.
            g_t, yaw_t = sweep._garment_in_layout_frame()
            g_np, yaw_np = g_t.detach().cpu().numpy(), yaw_t.detach().cpu().numpy()
            act_np = np.stack([scripted_action(g_np[i], spec, garment_yaw=float(yaw_np[i])) for i in range(n_env)])
            act = torch.tensor(act_np, dtype=torch.float32, device=env.unwrapped.device)
            _, _, term, trunc, _ = env.step(act)
            if clip_on and ep == 0:
                rgb = torch.as_tensor(cam.data.output["rgb"][:])[0, ..., :3]
                Image.fromarray(rgb.detach().cpu().numpy().astype("uint8")).save(
                    os.path.join(clip_dir, f"frame_{n_frames:04d}.png"))
                n_frames += 1
            env_steps += n_env
            step += 1
            em.n_sweeps += 1
            # The sweep term flags a plan it refused. Without this the Isaac
            # eval published invalid_trajectory_rate = 0 whatever happened --
            # the same structurally pinned metric success and fall_rate were.
            em.invalid_trajectory += int(bool(sweep._invalid[0].item()))
            done = bool(term[0] or trunc[0])
            if not done:
                cur = _xy(env)
                em.displacement_per_sweep.append(float(np.linalg.norm(cur - prev)))
                far = max(far, float(np.linalg.norm(cur - g0)))
                prev = cur.copy()
                measured = True
        em.env_steps = step
        tm = env.unwrapped.termination_manager
        em.success = _term_fired(tm, "success")
        em.fell = _term_fired(tm, "fallen")
        ep_nonfinite = int(sweep.nonfinite[0].item()) - nonfinite0
        nonfinite_eps.append(ep_nonfinite > 0)
        finite_success.append(bool(em.success) and ep_nonfinite == 0)
        # Only a success has a time to success. This used to be set whenever the
        # episode ended, so a fall at step 1 reported "1 step to success".
        if em.success:
            em.steps_to_success = step
        # An episode that ended on its first step has no between-step position to
        # read, so its garment travel is unknown, not zero. 21300348 reported 0.0
        # travel for four episodes that each ended by falling on step one.
        em.final_distance = float(np.linalg.norm(prev - basket_center(spec.target_basket)[:2])) if measured else float('nan')
        fars.append(far if measured else None)
        metrics.episodes.append(em)
        if ep == 0 and sweep.trace_on:
            # BHL_SWEEP_TRACE=1: the first episode's plans and substep samples.
            from bhl_robust.tasks.cloth_sort_mdp import QUAT_ORDER
            trace_path = str(Path(args_cli.out or "isaac_eval.json").with_suffix("")) + "_trace.json"
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            Path(trace_path).write_text(json.dumps({
                "task": tid, "rung": args_cli.rung, "garment": spec.name, "quat_order": QUAT_ORDER,
                "sim_dt": float(cfg.sim.dt), "decimation": int(cfg.decimation),
                "arm_joints": list(sweep._table.joints), "p_hand": [float(v) for v in sweep._table.p_hand],
                "success": bool(em.success), "fell": bool(em.fell), "sweeps": int(step),
                "plans": sweep.trace_plans, "samples": sweep.trace_samples,
            }))
            sweep.trace_on = False
    elapsed = time.perf_counter() - t0
    metrics.env_steps_per_s = env_steps / max(elapsed, 1e-9)
    payload = metrics.as_dict()
    got = [f for f in fars if f is not None]
    payload["garment_travel_measured_episodes"] = len(got)
    payload["max_garment_travel_mean"] = float(np.mean(got)) if got else None
    payload["garment_moved_episodes"] = int(sum(f > 0.02 for f in got)) if got else None
    dists = [e.final_distance for e in metrics.episodes if e.final_distance == e.final_distance]
    payload["final_distance_mean"] = float(np.mean(dists)) if dists else None
    if clip_on:
        payload["clip_frames"] = n_frames
        payload["clip_dir"] = clip_dir
    if trace_path:
        payload["trace_file"] = trace_path
    # A "success" in an episode whose robot state went non-finite is not a sort.
    payload["nonfinite_episodes"] = int(sum(nonfinite_eps))
    payload["success_rate_finite"] = float(np.mean(finite_success)) if finite_success else None
    payload["newton_njmax"] = os.environ.get("BHL_NEWTON_NJMAX") or "preset"
    payload["newton_nconmax"] = os.environ.get("BHL_NEWTON_NCONMAX") or "preset"
    payload["newton_coupling"] = (os.environ.get("BHL_NEWTON_COUPLING") or "one_way") if physics == "deformable" else "n/a"
    payload["rung"] = args_cli.rung
    payload["garment"] = spec.name
    payload["task"] = tid
    from bhl_robust.cloth.schedule import MACRO_STEP_S
    payload["macro_step_s"] = MACRO_STEP_S
    print(json.dumps(payload, indent=2))
    if args_cli.out:
        Path(args_cli.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args_cli.out).write_text(json.dumps(payload, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
