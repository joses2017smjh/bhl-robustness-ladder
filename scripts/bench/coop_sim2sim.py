"""Score cooperative-lift policies in MuJoCo, many seeds, no video.

§5 is the one section of this project whose every number came from the
simulator the policy was trained in. §1 exists to say why that is not enough:
the same locomotion checkpoint that looks best in PhysX is not the one that
survives MuJoCo, and the ranking inverts. A lift is a contact-dominated task,
which is exactly the regime where two solvers have the most room to disagree,
so the ranking in §5 has been resting on the weaker of the two available kinds
of evidence.

This produces the other kind. For each run it replays the trained actor in
MuJoCo across N seeds and reports the same three quantities §5's table reports
-- closest pinch, peak lift, fall rate -- so the columns can be set beside each
other rather than argued about.

Two controls are scored alongside, because they are cheap and they are the
questions a reader asks first:

* `solo`: one robot, its partner observation slots fed its own state. If a lone
  robot lifts the crate as well as a pair does, the word "cooperative" was not
  earning its place in the section title.
* `crew4`: two independent pairs in one world, one solver, one clock. Two
  separate rollouts cannot tell you whether one pair's contact disturbs the
  other; this can.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from bhl_robust.eval.coop_replay import (
    EPISODE_STEPS,
    POLICY_DT,
    TILT_LIMIT,
    CoopActor,
    CrewRunner,
    build_crew,
)

PINCH_GATE_M = 0.20
FINE_STD = 0.12


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


def score(model, slots, crates, runner: CrewRunner, seed: int, steps: int):
    """One episode. Returns one row per crate."""
    runner.reset(np.random.default_rng(seed))
    n_c = len(crates)
    best_pinch = np.full(n_c, np.inf)
    best_lift = np.zeros(n_c)
    held_lift = np.zeros(n_c)
    pinch_steps = np.zeros(n_c)
    fell = [False] * len(slots)

    for t in range(steps):
        runner.step()
        for i in range(len(slots)):
            if runner.tilt(i) > TILT_LIMIT:
                fell[i] = True
        for k, c in enumerate(crates):
            d = runner.pinch_distance(c)
            best_pinch[k] = min(best_pinch[k], d)
            best_lift[k] = max(best_lift[k], runner.crate_lift(c))
            pinch_steps[k] += float(d < PINCH_GATE_M)
            # The lift that is still there at the trained horizon, as opposed to
            # the peak. A toss and a hold have the same peak.
            if t == steps - 1:
                held_lift[k] = runner.crate_lift(c)

    rows = []
    for k, c in enumerate(crates):
        members = [c.slot_a] + ([c.slot_b] if c.slot_b is not None else [])
        rows.append(dict(
            seed=seed, crate=k, solo=int(c.slot_b is None),
            closest_pinch_m=float(best_pinch[k]),
            fine_kernel=float(1.0 - np.tanh(best_pinch[k] / FINE_STD)),
            peak_lift_m=float(best_lift[k]),
            held_lift_m=float(held_lift[k]),
            in_pinch_frac=float(pinch_steps[k] / steps),
            fell=int(any(fell[i] for i in members)),
        ))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, nargs="+", required=True)
    p.add_argument("--iteration", type=int, default=None)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--steps", type=int, default=EPISODE_STEPS)
    p.add_argument("--crews", type=int, nargs="+", default=(2, 3, 4),
                   help="crew sizes; 3 contributes the solo control")
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()

    # One compiled model per crew size, reused across runs and seeds. Compiling
    # a four-robot 22-DoF scene is the expensive part of this script. Ego
    # cameras go on every build: they are massless, they shift no geom index, and
    # a blind policy simply never has them rendered.
    built = {n: build_crew(args.upstream, args.cache_dir, n, ego_camera=True)
             for n in args.crews}

    out, skipped = [], []
    for rd in args.run_dir:
        ckpt = find_checkpoint(rd, args.iteration)
        try:
            actor = CoopActor(ckpt)
        except RuntimeError as e:
            # A run whose observation config this replay cannot assemble is a
            # skip, not a crash. Six of the coop runs are 150-wide standing-spawn
            # or `notrack` policies, and dying on the first of them would throw
            # away the scores of every run after it in the list.
            skipped.append((rd.name, str(e)))
            print(f"  skip   {rd.name}: {e}", flush=True)
            continue
        arm = rd.name.split("_", 1)[-1]
        for n in args.crews:
            model, slots, crates = built[n]
            runner = CrewRunner(model, slots, crates, actor)
            for s in range(args.seeds):
                for row in score(model, slots, crates, runner, s, args.steps):
                    row.update(run=rd.name, arm=arm, crew=n,
                               iteration=int(ckpt.stem.split("_")[1]),
                               vision=int(runner.wants_depth))
                    out.append(row)
            print(f"  scored {arm:<28} crew={n}  obs={actor.n_obs}  "
                  f"vision={'on' if runner.wants_depth else 'off'}  "
                  f"{args.seeds} seeds x {args.steps} steps", flush=True)
            runner.close()

    if not out:
        raise SystemExit("no run was scorable; nothing to report")

    print(f"\n{'arm':<26}{'crew':>5}{'pinch m':>10}{'kernel':>9}"
          f"{'peak cm':>9}{'held cm':>9}{'in-pinch':>10}{'fall':>7}   n")
    for rd in args.run_dir:
        arm = rd.name.split("_", 1)[-1]
        for n in args.crews:
            for solo in (0, 1):
                sel = [r for r in out
                       if r["run"] == rd.name and r["crew"] == n and r["solo"] == solo]
                if not sel:
                    continue
                tag = f"{n} solo" if solo else f"{n}"
                print(f"{arm:<26}{tag:>5}"
                      f"{np.mean([r['closest_pinch_m'] for r in sel]):>10.3f}"
                      f"{np.mean([r['fine_kernel'] for r in sel]):>9.3f}"
                      f"{np.mean([r['peak_lift_m'] for r in sel]) * 100:>9.1f}"
                      f"{np.mean([r['held_lift_m'] for r in sel]) * 100:>9.1f}"
                      f"{np.mean([r['in_pinch_frac'] for r in sel]):>9.0%}"
                      f"{np.mean([r['fell'] for r in sel]):>7.2f}   {len(sel)}")

    if skipped:
        print(f"\nskipped {len(skipped)} run(s) this replay cannot assemble:")
        for name, why in skipped:
            print(f"  {name}: {why}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f"\ncsv -> {args.csv}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
