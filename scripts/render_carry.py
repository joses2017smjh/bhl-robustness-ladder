"""Render a trained cooperative-lift policy carrying crates in MuJoCo.

§5 had no clip of the thing it measures. `docs/gifs/squat_pick.gif` is the
scripted kinematics check and says so; every number in the section came from
PhysX-side TensorBoard scalars. This renders the actual policy, in the other
simulator, which is the standard this repo holds every other section to.

The crew sizes are not three views of one demo. The policy controls exactly two
robots, so:

* 2 robots, one crate -- the policy as trained.
* 4 robots, two crates -- two independent pairs sharing a world, a solver and a
  clock. Whether one pair's contact disturbs the other is a physics question
  that two separate rollouts cannot ask.
* 3 robots -- a pair, and one robot given a crate of its own with its partner
  slots fed its own state. That last one is the control the section never had:
  if a lone robot lifts the crate too, "cooperative" was doing no work.

Clip conventions are the paired-GIF conventions, because a reader should not
have to learn a second visual language for this section: a fallen robot goes
dark, a red border comes on and stays on, and once the whole crew is down the
last frame holds for the remainder of the clip instead of showing bodies
settling. The default run is 12 s against a trained horizon of 8 s, with the
horizon marked on the timeline -- a hold that only lasts as long as the episode
did is a different result from one that keeps holding.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from bhl_robust.eval.coop_replay import (
    EPISODE_STEPS,
    PAYLOADS,
    POLICY_DT,
    TILT_LIMIT,
    CoopActor,
    CrewRunner,
    build_crew,
)
from bhl_robust.eval.livery import PAYLOAD_RGBA, SHELL_RGBA
from bhl_robust.eval.video import EpisodeRecorder

# `reaching_fine` uses std = 0.12; the curriculum counts a pinch below 0.20 m.
PINCH_GATE_M = 0.20

# Full-scale of the lift gauge. The staged curriculum ramps its height target to
# 0.22 m, so a gauge that tops out there reads as "how much of the asked-for
# lift happened" rather than as an arbitrary bar.
LIFT_FULL_M = 0.22
_POV_RES = 184

_DEAD_SCALE = 0.30


def kill_colour(model: mujoco.MjModel, prefix: str) -> None:
    """Darken one robot's geoms in place, so a fall reads without a caption."""
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if b != 0 and name.startswith(prefix):
            model.geom_rgba[g, :3] *= _DEAD_SCALE


class CarryRecorder(EpisodeRecorder):
    """EpisodeRecorder with a per-crate lift gauge and a run timeline.

    The gauges exist because the interesting quantity is a few centimetres of
    crate height, and a few centimetres at 880 px wide is one pixel of crate.
    §5's whole finding is that pinch moved and height did not, so the clip has
    to make height legible or it cannot show the finding at all.
    """

    def __init__(self, model, path, fps, n_crates: int, steps: int,
                 horizon: int, width: int = 1180, height: int = 620,
                 caption: str = "", band: int = 86, pov: int = 0):
        self._band = band
        # The POV column is appended to the right of the third-person scene, so
        # the two are the same clip on the same clock. Cutting between them
        # would let a reader assume a correspondence the clip never showed.
        self._pov = pov
        super().__init__(model, path, fps=fps, width=width,
                         height=height + band, caption=caption, track_body="")
        self._scene_h = height
        self._scene_w = width - (pov + 12 if pov else 0)
        self.renderer.close()
        self.renderer = mujoco.Renderer(model, height=height,
                                        width=self._scene_w)
        self.n_crates = n_crates
        self.steps, self.horizon = steps, horizon
        self._shell = tuple(int(255 * c) for c in SHELL_RGBA[:3])
        self._payload = tuple(int(255 * c) for c in PAYLOAD_RGBA[:3])

    def _hud(self, lifts, pinched, t: int) -> np.ndarray:
        band = np.full((self._band, self.width, 3), 16, dtype=np.uint8)
        gh, pad, gw, gap = 52, 26, 30, 46
        y1 = 10 + gh
        for k in range(self.n_crates):
            x0 = pad + k * (gw + gap)
            band[10:y1, x0:x0 + gw] = (42, 44, 50)
            fill = int(round(gh * min(max(lifts[k] / LIFT_FULL_M, 0.0), 1.0)))
            if fill:
                band[y1 - fill:y1, x0:x0 + gw] = self._payload
            # Pinch lamp under the gauge: orange once the hands are inside the
            # 0.20 m gate the curriculum counts as a pinch.
            lamp = self._shell if pinched[k] else (58, 58, 64)
            band[y1 + 6:y1 + 14, x0:x0 + gw] = lamp

        # Timeline, with a tick at the trained episode length.
        tx0 = pad + self.n_crates * (gw + gap) + 14
        tw = self.width - tx0 - pad
        if tw > 40:
            band[y1 + 2:y1 + 8, tx0:tx0 + tw] = (48, 50, 56)
            done = int(round(tw * min(t / max(self.steps - 1, 1), 1.0)))
            if done:
                band[y1 + 2:y1 + 8, tx0:tx0 + done] = self._shell
            hx = tx0 + int(round(tw * min(self.horizon / max(self.steps, 1), 1.0)))
            if tx0 <= hx < tx0 + tw:
                band[y1 - 6:y1 + 16, hx:hx + 2] = (215, 215, 225)
        return band

    _POV_RULES = ((255, 255, 255), (235, 104, 52), (42, 120, 214))

    def _pov_column(self, panes: tuple) -> np.ndarray:
        """The robot's own three views, stacked, ruled in three hues.

        No glyph rendering is available here, so each pane is identified by a
        3 px rule above it -- white for the colour view, orange for the raw
        depth frame, blue for the 8x8 the policy is handed. The README caption
        carries the words. Nearest-neighbour upscaling is deliberate on the
        bottom pane: smoothing an 8x8 observation into something that looks
        like an image would misrepresent what the network receives.
        """
        w = self._pov
        col = np.full((self._scene_h, w, 3), 16, dtype=np.uint8)
        y = 8
        for img, hue in zip(panes, self._POV_RULES):
            # Round the scale factor up and crop, so a 64 px frame and an
            # 8 px one both fill the column instead of leaving a dark margin.
            k = max(1, -(-w // img.shape[1]))
            up = np.repeat(np.repeat(img, k, axis=0), k, axis=1)[:w, :w]
            h = up.shape[0]
            col[y:y + 3, :] = hue
            y += 7
            col[y:y + h, :up.shape[1]] = up
            y += h + 12
        return col

    def capture_frame(self, data, lifts, pinched, t: int,
                      flash: str | None = None,
                      pov: tuple | None = None) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.camera)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:self._scene_h, :self._scene_w] = self.renderer.render()
        if pov is not None and self._pov:
            frame[:self._scene_h, self._scene_w + 12:] = self._pov_column(pov)
        frame[self._scene_h:] = self._hud(lifts, pinched, t)
        frame[self._scene_h - 2:self._scene_h] = (40, 40, 46)
        if flash:
            colour = {"push": (255, 190, 0), "fall": (220, 40, 40)}.get(flash)
            if colour is not None:
                b = 10
                frame[:b, :, :] = colour
                frame[-b:, :, :] = colour
                frame[:, :b, :] = colour
                frame[:, -b:, :] = colour
        return frame

    def write(self, frame: np.ndarray) -> None:
        try:
            self._proc.stdin.write(frame.astype(np.uint8).tobytes())
            self.n_frames += 1
        except BrokenPipeError:
            pass


def find_checkpoint(run_dir: Path, iteration: int | None) -> Path:
    if iteration is not None:
        p = run_dir / f"model_{iteration}.pt"
        if not p.is_file():
            raise SystemExit(f"no checkpoint {p}")
        return p
    cands = sorted(run_dir.glob("model_*.pt"),
                   key=lambda q: int(q.stem.split("_")[1]))
    if not cands:
        raise SystemExit(f"no model_*.pt under {run_dir}")
    return cands[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="a logs/rsl_rl/coop_lift/<run> directory")
    p.add_argument("--iteration", type=int, default=None)
    p.add_argument("--robots", type=int, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gif", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300,
                   help=f"policy steps at {POLICY_DT:g} s; trained horizon is "
                        f"{EPISODE_STEPS}")
    p.add_argument("--payload", default="cube", choices=sorted(PAYLOADS),
                   help="which object the policy was trained on")
    p.add_argument("--pov", action="store_true",
                   help="append the robot's own colour and depth views")
    p.add_argument("--pov-robot", type=int, default=0,
                   help="which robot's head the POV column comes from")
    p.add_argument("--gif-width", type=int, default=880)
    p.add_argument("--gif-fps", type=int, default=12)
    args = p.parse_args()

    ckpt = find_checkpoint(args.run_dir, args.iteration)
    # Cameras are mounted unconditionally: they are massless and change no geom
    # index, so the physics of a blind replay is unaffected, and building the
    # scene before the checkpoint is read keeps the two independent.
    model, slots, crates = build_crew(args.upstream, args.cache_dir, args.robots,
                                      ego_camera=True, payload=args.payload)
    actor = CoopActor(ckpt)
    run = CrewRunner(model, slots, crates, actor)
    if args.pov:
        run.enable_pov(_POV_RES)
    run.reset(np.random.default_rng(args.seed))

    solo = [c for c in crates if c.slot_b is None]
    print(f"checkpoint  {ckpt.parent.name}/{ckpt.name}")
    print(f"actor       {actor.n_obs} obs -> 44 act  ({actor.layout})")
    print(f"vision      {'on, ' + str(len(slots)) + ' ego depth cameras' if run.wants_depth else 'off'}")
    print(f"payload     {args.payload}")
    print(f"crew        {args.robots} robots, {len(crates)} crate(s), "
          f"{len(solo)} solo attempt(s)")
    print(f"spawn       crate z = {run.d.xpos[crates[0].body_id][2]:.3f} m, "
          f"base z = {run.d.qpos[slots[0].qpos_adr + 2]:+.3f} m "
          f"(feet planted, Isaac uses a -0.070 m pelvis drop)")

    cap = (f"{args.robots} x 22 DoF  ·  cooperative lift, trained policy  ·  "
           f"{ckpt.parent.name.split('_', 1)[-1]}")
    rec = CarryRecorder(model, args.out, fps=1.0 / POLICY_DT,
                        n_crates=len(crates), steps=args.steps,
                        horizon=EPISODE_STEPS,
                        width=1180 + (_POV_RES + 12 if args.pov else 0),
                        height=620, caption=cap,
                        pov=_POV_RES if args.pov else 0)
    rec._track_id = -1
    rec.camera.distance = 2.3 + 0.55 * args.robots
    rec.camera.elevation = -13.0
    rec.camera.azimuth = 14.0
    rec.camera.lookat[:] = [0.0, 0.0, 0.30]

    n_c = len(crates)
    best_lift = np.zeros(n_c)
    best_pinch = np.full(n_c, np.inf)
    pinch_steps = np.zeros(n_c)
    fell_at = [None] * len(slots)
    crew_down_at = None
    frozen = None

    def crate_is_down(c) -> bool:
        members = [c.slot_a] + ([c.slot_b] if c.slot_b is not None else [])
        return any(fell_at[i] is not None for i in members)

    for t in range(args.steps):
        if frozen is not None:
            # Every pair is on the floor. Holding the last frame is the paired
            # clips' convention and it is also the honest one: what follows is
            # ragdoll settling, not policy behaviour.
            rec.write(frozen)
            continue

        run.step()
        for i in range(len(slots)):
            if fell_at[i] is None and run.tilt(i) > TILT_LIMIT:
                fell_at[i] = t * POLICY_DT
                kill_colour(model, slots[i].prefix)

        lifts, pinched = np.zeros(n_c), [False] * n_c
        for k, c in enumerate(crates):
            d = run.pinch_distance(c)
            lifts[k] = run.crate_lift(c)
            pinched[k] = d < PINCH_GATE_M
            best_pinch[k] = min(best_pinch[k], d)
            best_lift[k] = max(best_lift[k], lifts[k])
            pinch_steps[k] += float(pinched[k])

        down = any(f is not None for f in fell_at)
        pov = ((run.pov_rgb(args.pov_robot),
                run.pov_depth_rgb(args.pov_robot),
                run.pov_depth_obs_rgb(args.pov_robot))
               if args.pov else None)
        frame = rec.capture_frame(run.d, lifts, pinched, t,
                                  "fall" if down else None, pov=pov)
        rec.write(frame)

        if all(crate_is_down(c) for c in crates):
            crew_down_at = t * POLICY_DT
            frozen = frame

    err = rec.close()
    print(f"video -> {args.out}  ({rec.n_frames} frames)" if not err else f"ffmpeg: {err}")
    if crew_down_at is not None:
        print(f"froze  at t = {crew_down_at:.1f} s (every pair down)")

    rows = []
    print(f"\n{'crate':<7}{'robots':<10}{'closest pinch':>14}{'fine kernel':>13}"
          f"{'peak lift':>11}{'in-pinch':>10}  fell")
    for k, c in enumerate(crates):
        who = f"{c.slot_a},{c.slot_b}" if c.slot_b is not None else f"{c.slot_a} solo"
        members = [c.slot_a] + ([c.slot_b] if c.slot_b is not None else [])
        down = [str(i) for i in members if fell_at[i] is not None]
        kern = float(1.0 - np.tanh(best_pinch[k] / 0.12))
        frac = pinch_steps[k] / args.steps
        print(f"{k:<7}{who:<10}{best_pinch[k]:>13.3f}m{kern:>13.3f}"
              f"{best_lift[k] * 100:>10.1f}cm{frac:>9.0%}  "
              f"{','.join(down) if down else '-'}")
        rows.append(dict(
            run=ckpt.parent.name, iteration=int(ckpt.stem.split("_")[1]),
            crew=args.robots, crate=k, solo=int(c.slot_b is None),
            closest_pinch_m=round(float(best_pinch[k]), 4),
            fine_kernel=round(kern, 4),
            peak_lift_m=round(float(best_lift[k]), 4),
            in_pinch_frac=round(float(frac), 4),
            fell=len(down),
            steps=args.steps,
        ))

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        new = not args.csv.is_file()
        with args.csv.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"csv   -> {args.csv}")

    if args.gif is not None and args.out.is_file():
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from render_multi import mp4_to_gif
        mp4_to_gif(args.out, args.gif, seconds=args.steps * POLICY_DT,
                   width=args.gif_width, fps=args.gif_fps)
        print(f"gif   -> {args.gif}  ({args.gif.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
