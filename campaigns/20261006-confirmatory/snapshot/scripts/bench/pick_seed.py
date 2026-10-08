"""Pick the seed a clip should run, by scoring every seed first.

A clip is one rollout. Choosing which one by hand, or leaving it at seed 0,
means the caption and the file can disagree with the table beside them -- the
first cut of the ball transfer clip landed on 1.9 cm while the text beside it
said 9.1 cm, because 9.1 cm was the best of six seeds and seed 3 was not it.

This prints a per-seed table and names the seed that best represents the claim:
`hold` for the longest-standing rollout (what a section about a working lift
should show), `lift` for the highest peak (what a section about a ceiling
should show). Both are reported either way, so the choice is visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bhl_robust.eval.coop_replay import (
    POLICY_DT,
    TILT_LIMIT,
    CoopActor,
    CrewRunner,
    build_crew,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--robots", type=int, default=2)
    p.add_argument("--payload", default="cube")
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--criterion", choices=("hold", "lift"), default="hold")
    args = p.parse_args()

    ck = sorted(args.run_dir.glob("model_*.pt"),
                key=lambda q: int(q.stem.split("_")[1]))[-1]
    rows = []
    for seed in range(args.seeds):
        model, slots, crates = build_crew(args.upstream, args.cache_dir,
                                          args.robots, ego_camera=True,
                                          payload=args.payload)
        run = CrewRunner(model, slots, crates, CoopActor(ck))
        run.reset(np.random.default_rng(seed))
        # `upright` is training's own orientation termination and nothing more.
        # It cannot see a robot that sinks with its torso level, which is what
        # the ladder pair does for every seed -- so the base's descent from its
        # planted spawn is reported beside it. A large drop with no fall is a
        # collapse the fall criterion is blind to, not a rollout that held.
        base0 = [float(run.d.xpos[s.body_id][2]) for s in slots]
        # Peak lift stops accruing at the fall. A payload flung upward by a
        # toppling robot is not a lift, and letting it count picked seed 4 of
        # the ball transfer at "27.6 cm" -- a number the clip itself renders as
        # 9.1 cm, because the recorder freezes on the fall and the sweep did
        # not. Whatever the clip shows is what the sweep has to score.
        peak, fell, sink = 0.0, None, 0.0
        for t in range(args.steps):
            run.step()
            if fell is None:
                peak = max(peak, max(run.crate_lift(c) for c in crates))
            sink = max(sink, max(z0 - float(run.d.xpos[s.body_id][2])
                                 for s, z0 in zip(slots, base0)))
            if fell is None and max(run.tilt(i) for i in range(len(slots))) > TILT_LIMIT:
                fell = t * POLICY_DT
        rows.append((seed, 100 * peak,
                     fell if fell is not None else args.steps * POLICY_DT,
                     100 * sink))
        run.close()

    print(f"{args.payload} x{args.robots}  {args.run_dir.name}/{ck.name}")
    for s, pk, hold, sink in rows:
        print(f"  seed {s:2d}  peak {pk:5.1f} cm  upright {hold:5.2f} s  "
              f"base sank {sink:5.1f} cm")
    key = (lambda r: (r[2], r[1])) if args.criterion == "hold" else (lambda r: (r[1], r[2]))
    best = max(rows, key=key)
    pk = np.array([r[1] for r in rows])
    print(f"  median {np.median(pk):.1f} cm  mean {np.mean(pk):.1f} cm")
    print(f"BEST_SEED={best[0]}  # by {args.criterion}: "
          f"{best[1]:.1f} cm, upright {best[2]:.2f} s, sank {best[3]:.1f} cm")


if __name__ == "__main__":
    main()
