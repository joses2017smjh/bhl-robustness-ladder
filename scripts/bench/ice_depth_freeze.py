"""Does the ice depth arm use its depth image as a live sensor? A freeze test.

Why this exists
---------------
On the placed-ice rung and its no-ice control the depth-camera biped beat the
blind one. The ice terrain is 100% flat tiles and the Isaac depth camera
ray-casts only `/World/ground`, so the 16x16 image (obs 45..300 of 301) can
only encode the camera's height, pitch and roll. Offline, the trained depth
actors move their actions 10-20x more through the image than through
projected_gravity for a 2 deg tilt. H1: the depth arm uses the image as a clean
height/tilt sensor. This closed-loop MuJoCo test asks whether the image
*content* matters while walking.

Arms, per depth checkpoint (all on the flat MuJoCo biped scene, one robot):

  A  live depth: the ego camera is rendered every policy step through
     `MultiRunner.enable_depth_obs` / `depth_for` (the repo's replay path for
     301-wide policies; needs EGL, i.e. a GPU node).
  B  frozen depth: the image seen at t=0 of the same episode (robot upright at
     the nominal spawn height, before any physics step), held constant for the
     whole episode. `--nominal-source rendered` takes it from the renderer
     (GPU); `--nominal-source analytic` synthesises it by ray-casting the flat
     floor plane from the t=0 camera pose (no GL: runs on the login node).
  Bsettle  (secondary) live depth for the 1 s settle window, then frozen at
     the image of that instant. The t=0 camera is ~5 cm above where the robot
     settles, so B's frozen image is also a *biased* one; Bsettle asks whether
     an unbiased frozen image is enough. Same rules as B, reported separately.
  C  analytic live depth (ORACLE): every step the image is synthesised from the
     simulator's ground-truth camera pose against the z=0 floor plane, i.e. the
     nominal image plus exactly the change a height/tilt shift produces, and
     nothing else (no self-occlusion, no render artefacts). No GL needed.
  blind  reference: the matched blind checkpoint (45 observations).

Honest labels: LEARNED = the actor MLP read from `model_5999.pt` (bit-equal to
the exported ONNX where one exists; checked in-run). SCRIPTED = the velocity
commands, the frozen image of arm B, the analytic image of arm C. ORACLE = arm
C and the analytic nominal read the simulator's true camera pose.

Protocol (matched to `bhl_robust.eval.run_eval` / `harness.run_episode`):
harness command set (6 commands incl. vx 0.3), seeds 0-9, 12 s episodes,
pushes off, 1 s settle, fall = tilt > 0.78 rad or sink > 0.25 m, lin-vel error
in the yaw frame, displacement = |final xy - start xy|. Every arm of a seed sees
the identical initial state (same `default_rng(seed)` draws as run_episode):
the design is paired.

Predeclared rules (decided 2026-09-26, before any arm-A number exists):

  gate      arm A fall rate > 0.50 -> INCONCLUSIVE, B not interpreted. Labelled
            "sim2sim gap" if the matched blind checkpoint (same ring, same
            seed) has fall rate <= 0.20 ("walks"), else "harness" (both fail).
  (i)   "uses depth as a live sensor"  iff  fall(B) - fall(A) >= 0.30  or
            disp(B) <= 0.50 * disp(A)
  (ii)  "image content not needed"     iff  |fall(B) - fall(A)| <= 0.10  and
            |disp(B) - disp(A)| <= 0.20 * disp(A)
  (iii) otherwise inconclusive.
  C (secondary) "height/tilt image reproduces live depth" iff C is within
            0.10 fall rate and 20% displacement of A; otherwise the MuJoCo
            render carries content beyond height/tilt.
  pooled    episodes of gate-passing checkpoints only, same rules; none
            passing -> pooled INCONCLUSIVE.
  parity    the MLP loader must match exported/policy.onnx within 1e-3 (max
            abs action over 64 random inputs) where an ONNX exists, else that
            checkpoint aborts.

What this cannot tell apart: the critic also received depth during training
(env.yaml has a `depth` term in the critic group). A closed-loop actor test
shows whether the *actor* consumes the image online; it cannot say whether the
training advantage came from a better critic.

Usage:
  ice_depth_freeze.py run --run-dir <biped run> --arms B C --nominal-source analytic \
      --out part.json [--seeds 10 --episode-s 12 --commands all]
  ice_depth_freeze.py summarise --parts dir/*.json --out verdict.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "external" / "Berkeley-Humanoid-Lite"
RUNS_DIR = UPSTREAM / "logs" / "rsl_rl" / "biped"
# Tracked; identical to the Sep-17 placed export apart from the checkpoint path
# (overridden) and the random command_velocity (unused by update()).
DEFAULT_TEMPLATE = REPO / "results" / "deploy_ice_blind.yaml"
CKPT_NAME = "model_5999.pt"

# ---------------------------------------------------------------- predeclared
GATE_A_FALL_MAX = 0.50
BLIND_WALKS_FALL_MAX = 0.20
LIVE_FALL_DELTA = 0.30
LIVE_DISP_RATIO = 0.50
SAME_FALL_TOL = 0.10
SAME_DISP_TOL = 0.20
NOMINAL_DIFF_TOL = 0.02       # per pooled cell, scaled units (0.02 = 12 cm)
ONNX_PARITY_TOL = 1e-3

# Same numbers as harness.TILT_LIMIT_RAD / MAX_SINK_M, repeated so the pure
# functions import without MuJoCo; `run` asserts they still agree.
TILT_LIMIT_RAD = 0.78
MAX_SINK_M = 0.25

# Depth term geometry (depth_env_cfg / MultiRunner).
DEPTH_RES = 64
DEPTH_SIDE = 16
DEPTH_CLIP = 6.0
PROPRIO_DIM = 45

LABELS = {
    "learned": "actor MLP from model_5999.pt (deterministic mean action)",
    "scripted": "velocity commands (harness set); arm B frozen t=0 image; arm C analytic image",
    "oracle": "arm C and the analytic nominal use the simulator's ground-truth camera pose "
              "to ray-cast the z=0 floor plane (privileged state)",
    "sim": "MuJoCo flat biped scene, uniform friction (no ice patches); pushes off",
}

RUN_RE = re.compile(r"ppo-ice-(placed|control)-(blind|depth)-s(\d+)$")

# ============================================================ pure functions


def parse_run(name: str):
    """'..._ppo-ice-placed-depth-s1' -> ('placed', 'depth', 1), else None."""
    m = RUN_RE.search(name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def plane_depth_raw(cam_pos, cam_xmat, fovy_deg: float, res: int = DEPTH_RES,
                    no_hit: float = 200.0, ray_max: float | None = None,
                    plane_z: float = 0.0) -> np.ndarray:
    """Planar depth (distance along the optical axis) of the plane z=plane_z.

    Pinhole model of a MuJoCo camera: it looks down its own -Z with +Y up;
    `cam_xmat` holds the camera axes as columns. Row 0 is the TOP of the image,
    as `mujoco.Renderer.render()` returns it. Pixels whose ray misses the plane
    (at or above the horizon) get `no_hit`, which is what MuJoCo's depth buffer
    reports for empty sky (the far plane, zfar * extent). With `ray_max`, a hit
    farther than that along the ray also becomes `no_hit` (the Isaac ray
    caster's max_distance).
    """
    R = np.asarray(cam_xmat, dtype=np.float64).reshape(3, 3)
    p = np.asarray(cam_pos, dtype=np.float64)
    f = 0.5 * res / math.tan(math.radians(fovy_deg) / 2.0)
    c = (np.arange(res) + 0.5 - res / 2.0) / f
    xc = np.broadcast_to(c[None, :], (res, res))          # +x right
    yc = np.broadcast_to(-c[:, None], (res, res))         # +y up, row 0 top
    dirs = np.stack([xc, yc, -np.ones((res, res))], axis=-1)  # unit optical z
    dw = dirs @ R.T                                       # world directions
    dz = dw[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (plane_z - p[2]) / dz
    hit = (dz < 0) & (t > 0)
    if ray_max is not None:
        ray_len = t * np.linalg.norm(dw, axis=-1)
        hit &= ray_len <= ray_max
    return np.where(hit, t, no_hit)


def pool_scale(raw: np.ndarray, side: int = DEPTH_SIDE, clip: float = DEPTH_CLIP,
               convention: str = "multirunner") -> np.ndarray:
    """Pool a raw 64x64 depth image to side x side and scale to [0, 1].

    "multirunner": what `MultiRunner._depth_obs` does -- nan/inf -> clip, pool
    the RAW depths, then /clip and clamp. Arm A is fed this, so B and C use it.
    "isaac": what `depth_env_cfg.depth_obs` sees -- misses and hits beyond
    max_distance are already `clip` before pooling. The two differ only in
    pooled cells that mix sky/far ground with near ground.
    """
    d = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=clip, posinf=clip)
    if convention == "isaac":
        d = np.minimum(d, clip)
    elif convention != "multirunner":
        raise ValueError(f"unknown convention {convention!r}")
    res = d.shape[0]
    pool = res // side
    if pool * side != res:
        raise ValueError(f"cannot pool {res} to {side}")
    d = d.reshape(side, pool, side, pool).mean(axis=(1, 3))
    return np.clip(d.reshape(-1) / clip, 0.0, 1.0).astype(np.float32)


def analytic_depth_obs(cam_pos, cam_xmat, fovy_deg, no_hit=200.0,
                       convention="multirunner") -> np.ndarray:
    """The 256-vector a flat-ground camera at this pose produces."""
    ray_max = DEPTH_CLIP if convention == "isaac" else None
    raw = plane_depth_raw(cam_pos, cam_xmat, fovy_deg, no_hit=no_hit, ray_max=ray_max)
    return pool_scale(raw, convention=convention)


def summarise_arm(episodes: list[dict]) -> dict:
    """Fall rate, mean displacement, tracking error over a list of episodes."""
    n = len(episodes)
    if n == 0:
        return {"n": 0}
    fin = lambda k: [e[k] for e in episodes if e.get(k) is not None and np.isfinite(e[k])]  # noqa: E731
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
    by_cmd: dict[str, dict] = {}
    for e in episodes:
        k = f"{e['command'][0]:+.1f},{e['command'][1]:+.1f},{e['command'][2]:+.1f}"
        by_cmd.setdefault(k, []).append(e)
    return {
        "n": n,
        "fall_rate": float(np.mean([bool(e["fell"]) for e in episodes])),
        "mean_disp_m": mean(fin("distance_m")),
        "mean_lin_vel_err": mean(fin("lin_vel_err")),
        "mean_yaw_rate_err": mean(fin("yaw_rate_err")),
        "mean_survival_s": mean(fin("survival_s")),
        "by_command": {
            k: {"n": len(v), "fall_rate": float(np.mean([bool(e["fell"]) for e in v])),
                "mean_disp_m": float(np.mean([e["distance_m"] for e in v])),
                "mean_lin_vel_err": mean([e["lin_vel_err"] for e in v
                                          if e["lin_vel_err"] is not None
                                          and np.isfinite(e["lin_vel_err"])])}
            for k, v in sorted(by_cmd.items())
        },
    }


def _close(x: dict, a: dict) -> bool:
    """x within SAME_FALL_TOL fall rate and SAME_DISP_TOL relative disp of a."""
    return (abs(x["fall_rate"] - a["fall_rate"]) <= SAME_FALL_TOL + 1e-12
            and abs(x["mean_disp_m"] - a["mean_disp_m"])
            <= SAME_DISP_TOL * a["mean_disp_m"] + 1e-12)


def classify(A: dict | None, B: dict | None, blind: dict | None,
             C: dict | None = None) -> dict:
    """Apply the predeclared rules to arm summaries (each from summarise_arm)."""
    out: dict = {"rule_version": "2026-09-26"}
    if C is not None and A is not None and A.get("n") and C.get("n"):
        out["C_vs_A"] = ("height/tilt image reproduces live depth" if _close(C, A)
                         else "MuJoCo render carries content beyond height/tilt")
    if not A or not A.get("n"):
        out.update(outcome="INCONCLUSIVE", reason="arm A (live depth) not run", gate="not evaluated")
        return out
    if not B or not B.get("n"):
        out.update(outcome="INCONCLUSIVE", reason="arm B (frozen depth) not run", gate="not evaluated")
        return out
    blind_walks = None if not blind or not blind.get("n") else (
        blind["fall_rate"] <= BLIND_WALKS_FALL_MAX)
    out["blind_walks"] = blind_walks
    if A["fall_rate"] > GATE_A_FALL_MAX:
        if blind_walks is True:
            why = "sim2sim gap: live-depth arm falls while the blind reference walks"
        elif blind_walks is False:
            why = "harness: live-depth arm and blind reference both fall"
        else:
            why = "live-depth arm falls; blind reference missing"
        out.update(outcome="INCONCLUSIVE", reason=why, gate="FAIL")
        return out
    out["gate"] = "PASS"
    d_fall = B["fall_rate"] - A["fall_rate"]
    a_disp, b_disp = A["mean_disp_m"], B["mean_disp_m"]
    out["d_fall_B_minus_A"] = d_fall
    out["disp_ratio_B_over_A"] = (b_disp / a_disp) if a_disp > 0 else None
    if d_fall >= LIVE_FALL_DELTA - 1e-12 or (a_disp > 0 and b_disp <= LIVE_DISP_RATIO * a_disp):
        out.update(outcome="(i)", reason="uses depth as a live sensor")
    elif _close(B, A):
        out.update(outcome="(ii)", reason="image content not needed")
    else:
        out.update(outcome="(iii)", reason="inconclusive: between the predeclared bands")
    return out


# ============================================================ policy loading


class MlpPolicy:
    """rsl_rl ActorCritic actor (ELU MLP, no normaliser) read from model_*.pt.

    Upstream's TorchPolicy expects a pickled nn.Module; an rsl_rl checkpoint is
    a dict, so it cannot load these. The weights are the same ones the ONNX
    export wraps -- verified by `onnx_parity`.
    """

    def __init__(self, ckpt_path: str | Path):
        import torch
        sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)["model_state_dict"]
        idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
        self.layers = [(sd[f"actor.{i}.weight"].numpy().astype(np.float32),
                        sd[f"actor.{i}.bias"].numpy().astype(np.float32)) for i in idx]
        self.in_dim = int(self.layers[0][0].shape[1])
        self.out_dim = int(self.layers[-1][0].shape[0])

    def forward(self, obs: np.ndarray) -> np.ndarray:
        h = np.asarray(obs, dtype=np.float32).reshape(-1, self.in_dim)
        for j, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if j < len(self.layers) - 1:
                h = np.where(h > 0, h, np.expm1(np.minimum(h, 0.0))).astype(np.float32)
        return h


def onnx_parity(policy: MlpPolicy, onnx_path: Path, n: int = 64) -> float:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(str(onnx_path), so)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, policy.in_dim)).astype(np.float32)
    if policy.in_dim > PROPRIO_DIM:
        x[:, PROPRIO_DIM:] = rng.uniform(0, 1, (n, policy.in_dim - PROPRIO_DIM))
    key = s.get_inputs()[0].name
    y = np.concatenate([s.run(None, {key: x[i:i + 1]})[0] for i in range(n)])
    return float(np.abs(policy.forward(x) - y).max())


def make_cfg(template: Path, ckpt: Path, width: int):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(template)
    cfg.policy_checkpoint_path = str(ckpt)
    cfg.num_observations = int(width)
    if int(cfg.get("history_length", 0)) != 0:
        raise ValueError("template has history_length != 0; the ice arms have none")
    return cfg


def make_controller(cfg, policy: MlpPolicy):
    from bhl_robust.eval.depth_controller import DepthRlController

    class CkptController(DepthRlController):
        def load_policy(self):   # noqa: D401 -- weights already in memory
            self.policy = policy

    c = CkptController(cfg)
    c.load_policy()
    return c


# ============================================================ simulation


@dataclass
class Episode:
    arm: str
    command: tuple
    seed: int
    fell: bool
    survival_s: float
    lin_vel_err: float
    yaw_rate_err: float
    distance_m: float
    mean_cam_height_m: float
    settle_cam_height_m: float
    # arm A only: rendered vs analytic, scaled units
    t0_render_vs_analytic_max: float | None = None
    step_render_vs_analytic_max: float | None = None
    frac_steps_render_diff_gt_tol: float | None = None


def _yaw_frame_vel(qvel_lin, quat):
    w, x, y, z = (float(c) for c in quat)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = math.cos(-yaw), math.sin(-yaw)
    v = qvel_lin
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class Sim:
    """One robot on the flat multi-world scene with the ego camera attached."""

    def __init__(self, cache_dir: Path):
        import mujoco
        from bhl_robust.eval.mjcf_assets import EGO_CAM_NAME
        from bhl_robust.eval.multi_robot import build_multi
        self.mj = mujoco
        self.model, self.slots = build_multi(UPSTREAM, cache_dir, 1, ["robot"],
                                             world="flat", ego_camera=True)
        self.cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                     f"r0_{EGO_CAM_NAME}")
        if self.cam < 0:
            raise RuntimeError("ego camera missing from the composed model")
        self.fovy = float(self.model.cam_fovy[self.cam])
        # What MuJoCo's depth buffer reports for sky: the far clipping plane.
        self.no_hit = float(self.model.vis.map.zfar * self.model.stat.extent)

    def analytic(self, d, convention="multirunner"):
        return analytic_depth_obs(d.cam_xpos[self.cam], d.cam_xmat[self.cam], self.fovy,
                                  no_hit=self.no_hit, convention=convention)

    def nominal_state(self):
        """Camera pose at t=0: base at qpos0, identity orientation."""
        d = self.mj.MjData(self.model)
        self.mj.mj_resetData(self.model, d)
        s = self.slots[0]
        d.qpos[s.qpos_adr + 3:s.qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.mj.mj_forward(self.model, d)
        return d


def run_episode_arm(sim: Sim, runner, ctrl, arm: str, command, seed: int, ecfg,
                    nominal_source: str) -> Episode:
    """Mirror of harness.run_episode (pushes off) with a depth arm."""
    mj = sim.mj
    s = runner.slots[0]
    d = runner.d
    rng = np.random.default_rng(seed)
    runner.reset(rng)                                  # joint noise N(0, 0.02)
    d.qvel[s.qvel_adr:s.qvel_adr + 2] = rng.normal(0.0, ecfg.init_vel_noise, 2)
    mj.mj_forward(runner.m, d)
    ctrl.prev_actions[:] = 0.0
    ctrl.policy_observations[:] = 0.0

    dt = runner.cfgs[0].policy_dt
    n_steps = int(ecfg.episode_s / dt)
    settle = int(ecfg.settle_s / dt)
    spawn_z = float(d.qpos[s.qpos_adr + 2])
    start_xy = d.qpos[s.qpos_adr:s.qpos_adr + 2].copy()

    frozen = None
    t0_diff = None
    if arm == "B":
        if nominal_source == "rendered":
            frozen = runner.depth_for(0)
            t0_diff = float(np.abs(frozen - sim.analytic(d)).max())
        else:
            frozen = sim.analytic(d)

    obs = runner.observe(0, (0.0, 0.0, 0.0))          # run_episode's first obs
    lin_errs, yaw_errs, cam_h, settle_h = [], [], [], []
    step_diffs = []
    fell, survival = False, n_steps
    for t in range(n_steps):
        if arm == "A":
            depth = runner.depth_for(0)
            diff = float(np.abs(depth - sim.analytic(d)).max())
            step_diffs.append(diff)
            if t == 0:
                t0_diff = diff
        elif arm == "B":
            depth = frozen
        elif arm == "Bsettle":
            if t <= settle:
                live = (runner.depth_for(0) if nominal_source == "rendered"
                        else sim.analytic(d))
                if t == settle:
                    frozen = live
                depth = live
            else:
                depth = frozen
        elif arm == "C":
            depth = sim.analytic(d)
        else:
            depth = None
        targets = ctrl.update(obs, depth)
        runner.step([targets])
        obs = runner.observe(0, command)

        z = float(d.qpos[s.qpos_adr + 2])
        tilt = runner.tilt(0)
        if tilt > TILT_LIMIT_RAD or (spawn_z - z) > MAX_SINK_M:
            fell, survival = True, t
            break
        h = float(d.cam_xpos[sim.cam][2])
        if t < settle:
            settle_h.append(h)
        else:
            q = d.qpos[s.qpos_adr + 3:s.qpos_adr + 7]
            v = _yaw_frame_vel(d.qvel[s.qvel_adr:s.qvel_adr + 3], q)
            lin_errs.append(float(np.linalg.norm(v - np.asarray(command[:2]))))
            yaw_errs.append(abs(float(d.sensordata[s.gyro_adr + 2]) - command[2]))
            cam_h.append(h)
    m = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
    return Episode(
        arm=arm, command=tuple(float(c) for c in command), seed=int(seed), fell=fell,
        survival_s=survival * dt, lin_vel_err=m(lin_errs), yaw_rate_err=m(yaw_errs),
        distance_m=float(np.linalg.norm(d.qpos[s.qpos_adr:s.qpos_adr + 2] - start_xy)),
        mean_cam_height_m=m(cam_h), settle_cam_height_m=m(settle_h),
        t0_render_vs_analytic_max=t0_diff,
        step_render_vs_analytic_max=max(step_diffs) if step_diffs else None,
        frac_steps_render_diff_gt_tol=(float(np.mean(np.asarray(step_diffs) > NOMINAL_DIFF_TOL))
                                       if step_diffs else None),
    )


def _jsonable(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def cmd_run(args) -> int:
    from bhl_robust.eval import harness
    from bhl_robust.eval.harness import EvalConfig
    from bhl_robust.eval.multi_robot import MultiRunner
    assert harness.TILT_LIMIT_RAD == TILT_LIMIT_RAD and harness.MAX_SINK_M == MAX_SINK_M

    run_dir = Path(args.run_dir)
    meta = parse_run(run_dir.name)
    if meta is None:
        print(f"ERROR: {run_dir.name} is not a ppo-ice-{{placed,control}}-{{blind,depth}}-sN run")
        return 2
    ring, kind, rseed = meta
    ckpt = run_dir / args.checkpoint
    if not ckpt.is_file():
        print(f"ERROR: missing {ckpt}")
        return 2
    policy = MlpPolicy(ckpt)
    width = policy.in_dim
    if width not in (PROPRIO_DIM, PROPRIO_DIM + DEPTH_SIDE * DEPTH_SIDE):
        print(f"ERROR: {width}-wide actor is neither blind (45) nor 16x16 depth (301)")
        return 2
    arms = list(args.arms)
    if kind == "blind":
        bad = [a for a in arms if a != "blind"]
        if bad:
            print(f"note: blind checkpoint; ignoring depth arms {bad}")
        arms = ["blind"]
    else:
        arms = [a for a in arms if a != "blind"]

    record: dict = {
        "schema": "ice_depth_freeze/part/v1",
        "labels": LABELS,
        "run": run_dir.name, "ring": ring, "kind": kind, "train_seed": rseed,
        "checkpoint": str(ckpt), "obs_width": width,
        "predeclared": predeclared(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    onnx = run_dir / "exported" / "policy.onnx"
    if onnx.is_file():
        err = onnx_parity(policy, onnx)
        record["onnx_parity_max_abs"] = err
        print(f"[parity] {run_dir.name}: MLP vs exported ONNX max|diff| = {err:.2e}")
        if err > ONNX_PARITY_TOL:
            record["error"] = f"ONNX parity {err:.3e} > {ONNX_PARITY_TOL}"
            _write(args.out, record)
            print(f"ERROR: {record['error']}")
            return 3
    else:
        record["onnx_parity_max_abs"] = None
        record["onnx_note"] = "no exported/policy.onnx; model_5999.pt loaded directly"

    ecfg = EvalConfig(episode_s=args.episode_s, seeds=tuple(range(args.seeds)))
    commands = ecfg.commands if args.commands == "all" else tuple(
        tuple(float(v) for v in c.split(",")) for c in args.commands.split(";"))
    record["protocol"] = {
        "episode_s": ecfg.episode_s, "settle_s": ecfg.settle_s, "seeds": list(ecfg.seeds),
        "commands": [list(c) for c in commands], "push_speed": 0.0,
        "init_joint_noise": ecfg.init_joint_noise, "init_vel_noise": ecfg.init_vel_noise,
        "tilt_limit_rad": TILT_LIMIT_RAD, "max_sink_m": MAX_SINK_M,
        "nominal_source": args.nominal_source if "B" in arms else None,
        "paired": "every arm of a (command, seed) starts from the identical state",
    }

    sim = Sim(Path(args.cache_dir))
    cfg = make_cfg(Path(args.template), ckpt, width)
    dn = sim.nominal_state()
    nom_mr = sim.analytic(dn, "multirunner")
    nom_is = sim.analytic(dn, "isaac")
    record["nominal"] = {
        "cam_pos": [float(v) for v in dn.cam_xpos[sim.cam]],
        "cam_fovy_deg": sim.fovy, "mujoco_no_hit_depth_m": sim.no_hit,
        "analytic_image_multirunner": [round(float(v), 5) for v in nom_mr],
        "multirunner_vs_isaac_convention_max_abs": float(np.abs(nom_mr - nom_is).max()),
        "multirunner_vs_isaac_cells_differing": int((np.abs(nom_mr - nom_is) > 1e-6).sum()),
    }
    print(f"[nominal] cam z={dn.cam_xpos[sim.cam][2]:.3f} m  image range "
          f"[{nom_mr.min():.3f}, {nom_mr.max():.3f}]  MR-vs-Isaac pooling max diff "
          f"{record['nominal']['multirunner_vs_isaac_convention_max_abs']:.3f} "
          f"({record['nominal']['multirunner_vs_isaac_cells_differing']} cells)")

    record["arms"] = {}
    for arm in arms:
        ctrl = make_controller(cfg, policy)
        runner = MultiRunner(sim.model, sim.slots, [cfg], [ctrl])
        if arm == "A" or (arm in ("B", "Bsettle") and args.nominal_source == "rendered"):
            runner.enable_depth_obs([width])           # builds a GL renderer
        eps = []
        t0 = time.time()
        for command in commands:
            for seed in ecfg.seeds:
                e = run_episode_arm(sim, runner, ctrl, arm, command, seed, ecfg,
                                    args.nominal_source)
                eps.append(asdict(e))
                print(f"{run_dir.name[-26:]} arm={arm:5s} cmd=({command[0]:+.1f},{command[1]:+.1f},"
                      f"{command[2]:+.1f}) seed={seed} fell={int(e.fell)} surv={e.survival_s:5.2f}s "
                      f"lin_err={e.lin_vel_err:.3f} dist={e.distance_m:.2f}", flush=True)
        if runner._depth_r is not None:
            try:
                runner._depth_r.close()
            except Exception:
                pass
        summ = summarise_arm(eps)
        if arm == "A":
            vals = [e["t0_render_vs_analytic_max"] for e in eps if e["t0_render_vs_analytic_max"] is not None]
            stp = [e["frac_steps_render_diff_gt_tol"] for e in eps if e["frac_steps_render_diff_gt_tol"] is not None]
            summ["render_check"] = {
                "t0_max_abs": max(vals) if vals else None,
                "t0_within_tol": (max(vals) <= NOMINAL_DIFF_TOL) if vals else None,
                "mean_frac_steps_gt_tol": float(np.mean(stp)) if stp else None,
                "tol": NOMINAL_DIFF_TOL,
            }
        if arm == "B" and args.nominal_source == "rendered":
            vals = [e["t0_render_vs_analytic_max"] for e in eps if e["t0_render_vs_analytic_max"] is not None]
            summ["rendered_nominal_vs_analytic_max_abs"] = max(vals) if vals else None
        summ["wall_s"] = round(time.time() - t0, 1)
        record["arms"][arm] = {"summary": summ, "episodes": eps}
        print(f"== {run_dir.name} arm {arm}: n={summ['n']} fall={summ['fall_rate']:.3f} "
              f"disp={summ['mean_disp_m']:.3f} m lin_err={summ['mean_lin_vel_err']:.3f} "
              f"({summ['wall_s']} s)", flush=True)
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write(args.out, record)
    print(f"json -> {args.out}")
    return 0


def _write(path, record):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_jsonable(record), indent=1) + "\n")


def predeclared() -> dict:
    return {
        "decided": "2026-09-26",
        "gate": f"A fall rate > {GATE_A_FALL_MAX} -> INCONCLUSIVE (sim2sim gap if matched blind "
                f"fall rate <= {BLIND_WALKS_FALL_MAX}, else harness)",
        "i": f"fall(B)-fall(A) >= {LIVE_FALL_DELTA} or disp(B) <= {LIVE_DISP_RATIO}*disp(A)",
        "ii": f"|fall(B)-fall(A)| <= {SAME_FALL_TOL} and |disp(B)-disp(A)| <= {SAME_DISP_TOL}*disp(A)",
        "iii": "otherwise",
        "C_secondary": f"C within {SAME_FALL_TOL} fall and {SAME_DISP_TOL:.0%} disp of A",
        "Bsettle_secondary": "(i)/(ii)/(iii) rules applied to Bsettle in place of B; reported, "
                             "never replaces the primary B outcome",
        "pooled": "episodes of gate-passing checkpoints only; none -> INCONCLUSIVE",
        "onnx_parity_tol": ONNX_PARITY_TOL,
        "render_check_tol": NOMINAL_DIFF_TOL,
        "matched_blind": "same ring (placed/control), same training seed",
    }


# ============================================================ summarise


def build_verdict(parts: list[dict]) -> dict:
    """Pair depth checkpoints with matched blind ones and apply the rules.

    Parts of the same run (e.g. arms B and C written by separate processes)
    are merged first; an arm present in two parts of one run is refused.
    """
    merged: dict[str, dict] = {}
    for p in parts:
        q = merged.setdefault(p["run"], {**{k: v for k, v in p.items() if k != "arms"}, "arms": {}})
        if p.get("error"):
            q["error"] = p["error"]
        if (q.get("protocol") or {}).get("nominal_source") is None and p.get("protocol"):
            q["protocol"] = p["protocol"]
        for a, v in (p.get("arms") or {}).items():
            if a in q["arms"]:
                raise ValueError(f"arm {a} of {p['run']} appears in two parts")
            q["arms"][a] = v
    parts = list(merged.values())
    blind ={(p["ring"], p["train_seed"]): p for p in parts if p.get("kind") == "blind"}
    depth = [p for p in parts if p.get("kind") == "depth"]
    per = []
    pool: dict[str, list] = {"A": [], "B": [], "Bsettle": [], "C": [], "blind": []}
    for p in sorted(depth, key=lambda q: (q["ring"], q["train_seed"])):
        arms = p.get("arms", {})
        S = {a: arms[a]["summary"] for a in arms}
        bp = blind.get((p["ring"], p["train_seed"]))
        bs = bp["arms"]["blind"]["summary"] if bp and "blind" in bp.get("arms", {}) else None
        if p.get("error"):
            v = {"outcome": "INCONCLUSIVE", "reason": p["error"], "gate": "not evaluated"}
        else:
            v = classify(S.get("A"), S.get("B"), bs, S.get("C"))
            if S.get("Bsettle"):
                vs = classify(S.get("A"), S.get("Bsettle"), bs)
                v["Bsettle_outcome"] = f"{vs['outcome']} ({vs['reason']})"
        row = {
            "run": p["run"], "ring": p["ring"], "train_seed": p["train_seed"],
            "blind_run": bp["run"] if bp else None,
            "nominal_source": (p.get("protocol") or {}).get("nominal_source"),
            "arms": {a: {k: S[a].get(k) for k in ("n", "fall_rate", "mean_disp_m",
                                                   "mean_lin_vel_err", "mean_yaw_rate_err")}
                     for a in S},
            "blind": None if bs is None else {k: bs.get(k) for k in (
                "n", "fall_rate", "mean_disp_m", "mean_lin_vel_err")},
            "render_check": S.get("A", {}).get("render_check"),
            **v,
        }
        per.append(row)
        if v.get("gate") == "PASS":
            for a in ("A", "B", "Bsettle", "C"):
                if a in arms:
                    pool[a] += arms[a]["episodes"]
            if bp and "blind" in bp.get("arms", {}):
                pool["blind"] += bp["arms"]["blind"]["episodes"]
    if pool["A"]:
        PS = {a: summarise_arm(pool[a]) for a in pool if pool[a]}
        pv = classify(PS.get("A"), PS.get("B"), PS.get("blind"), PS.get("C"))
        if PS.get("Bsettle"):
            vs = classify(PS.get("A"), PS.get("Bsettle"), PS.get("blind"))
            pv["Bsettle_outcome"] = f"{vs['outcome']} ({vs['reason']})"
        pooled = {"n_checkpoints": sum(r.get("gate") == "PASS" for r in per),
                  "arms": {a: {k: PS[a].get(k) for k in ("n", "fall_rate", "mean_disp_m",
                                                          "mean_lin_vel_err")} for a in PS},
                  **pv}
    else:
        pooled = {"n_checkpoints": 0, "outcome": "INCONCLUSIVE",
                  "reason": "no checkpoint passed the validity gate (or arm A not run)"}
    # CPU preview: B against C where A is absent. Not a predeclared test.
    preview = []
    for p in depth:
        arms = p.get("arms", {})
        if "A" not in arms and "B" in arms and "C" in arms:
            b, c = arms["B"]["summary"], arms["C"]["summary"]
            preview.append({"run": p["run"], "B": {k: b[k] for k in ("fall_rate", "mean_disp_m")},
                            "C": {k: c[k] for k in ("fall_rate", "mean_disp_m")},
                            "note": "C (analytic live, oracle) stands in for A; NOT the predeclared test"})
    return {"schema": "ice_depth_freeze/verdict/v1", "labels": LABELS,
            "predeclared": predeclared(), "per_checkpoint": per, "pooled": pooled,
            "cpu_preview_B_vs_C": preview,
            "cannot_discriminate": "the critic also saw depth in training; this actor-only "
                                   "closed-loop test does not separate H1 from a critic effect"}


def cmd_summarise(args) -> int:
    files = sorted({f for g in args.parts for f in glob.glob(g)})
    parts, kept, skipped = [], [], []
    for f in files:
        j = json.loads(Path(f).read_text())
        if str(j.get("schema", "")).startswith("ice_depth_freeze/part/"):
            parts.append(j)
            kept.append(f)
        else:
            skipped.append(f)
    files = kept
    if skipped:
        print(f"skipped {len(skipped)} non-part JSON(s): {skipped}")
    v = build_verdict(parts)
    v["parts"] = files
    _write(args.out, v)
    for r in v["per_checkpoint"]:
        a = r["arms"]
        fmt = lambda k: ("-" if k not in a else f"fall={a[k]['fall_rate']:.2f} disp={a[k]['mean_disp_m']:.2f}")  # noqa: E731
        bl = r["blind"]
        bls = "-" if bl is None else "fall=%.2f disp=%.2f" % (bl["fall_rate"], bl["mean_disp_m"])
        print(f"ICE-DEPTH-FREEZE {r['run']}: {r['outcome']} ({r['reason']}) | A {fmt('A')} | "
              f"B {fmt('B')} | Bsettle {fmt('Bsettle')} | C {fmt('C')} | blind {bls} | "
              f"C_vs_A: {r.get('C_vs_A', '-')} | Bsettle: {r.get('Bsettle_outcome', '-')}")
    p = v["pooled"]
    print(f"ICE-DEPTH-FREEZE POOLED ({p['n_checkpoints']} gate-passing checkpoints): "
          f"{p['outcome']} ({p['reason']})" + (f" | C_vs_A: {p['C_vs_A']}" if "C_vs_A" in p else "")
          + (f" | Bsettle: {p['Bsettle_outcome']}" if "Bsettle_outcome" in p else ""))
    for r in v["cpu_preview_B_vs_C"]:
        print(f"CPU-PREVIEW (not the test) {r['run']}: B fall={r['B']['fall_rate']:.2f} "
              f"disp={r['B']['mean_disp_m']:.2f} | C fall={r['C']['fall_rate']:.2f} "
              f"disp={r['C']['mean_disp_m']:.2f}")
    print(f"json -> {args.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evaluate one checkpoint")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--checkpoint", default=CKPT_NAME)
    r.add_argument("--arms", nargs="+", default=["A", "B", "C"],
                   choices=["A", "B", "Bsettle", "C", "blind"])
    r.add_argument("--nominal-source", choices=["analytic", "rendered"], default="analytic")
    r.add_argument("--seeds", type=int, default=10)
    r.add_argument("--episode-s", type=float, default=12.0)
    r.add_argument("--commands", default="all",
                   help="'all' (harness set) or 'vx,vy,wz;vx,vy,wz'")
    r.add_argument("--template", default=str(DEFAULT_TEMPLATE),
                   help="deploy yaml whose PD/timing fields every arm shares")
    r.add_argument("--cache-dir", required=True)
    r.add_argument("--out", required=True)
    s = sub.add_parser("summarise", help="apply the predeclared rules to part JSONs")
    s.add_argument("--parts", nargs="+", required=True)
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_summarise(args)


if __name__ == "__main__":
    sys.exit(main())
