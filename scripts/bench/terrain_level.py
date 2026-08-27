"""Read a run's terrain-level curve from its TensorBoard event file.

G-B2 has now failed twice on evidence that was not what it claimed to be. The
probe piped training through `tail -40`, so its log kept one
`Curriculum/terrain_levels` line out of three hundred, and `tail -1` on that
single surviving line was reported as the final terrain level. Scraping a
truncated stdout stream is the wrong source: the trainer already writes every
iteration to an event file, and that file cannot be truncated by a pipe.

It also prints a control run alongside, because "still at level 0 after 300
iterations" is only a statement about the terrain if a robot on terrain known
to be walkable has left level 0 by the same iteration. Without that column the
probe cannot distinguish "these stairs are a wall" from "PPO has been running
for 300 iterations".
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

TAG = "Curriculum/terrain_levels"


def curve(run: Path) -> list[float]:
    from tensorboard.backend.event_processing import event_accumulator

    events = sorted(glob.glob(str(run / "events.out.tfevents.*")))
    if not events:
        raise SystemExit(f"no event file under {run}")
    ea = event_accumulator.EventAccumulator(events[-1], size_guidance={"scalars": 0})
    ea.Reload()
    if TAG not in ea.Tags()["scalars"]:
        raise SystemExit(f"{run.name} logs no {TAG}")
    return [x.value for x in ea.Scalars(TAG)]


def describe(label: str, v: list[float], at: int) -> float:
    """Print the curve and return the level at iteration `at`."""
    idx = min(at, len(v)) - 1
    step = max(1, len(v) // 8)
    print(f"  {label:22} n={len(v):4d}  @{at}={v[idx]:.4f}  "
          f"max={max(v):.4f}  last={v[-1]:.4f}")
    print(f"    {' '.join('%.3f' % x for x in v[::step])}")
    return v[idx]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True,
                   help="the probe run directory")
    p.add_argument("--control", type=Path, default=None,
                   help="a run on terrain known to be walkable, same length")
    p.add_argument("--at", type=int, default=300,
                   help="iteration to judge at")
    p.add_argument("--threshold", type=float, default=0.05)
    args = p.parse_args()

    print(f"=== {TAG} at iteration {args.at}")
    probe = describe(args.run.name, curve(args.run), args.at)

    if args.control is None:
        print("\nno control given -- this can say the probe did not promote, "
              "not that the terrain is why")
        raise SystemExit(0 if probe > args.threshold else 1)

    ctrl = describe(args.control.name, curve(args.control), args.at)
    print()
    if probe > args.threshold:
        print(f"G-B2 PASS | probe reached {probe:.4f} by iteration {args.at}")
        raise SystemExit(0)
    if ctrl <= args.threshold:
        print(f"G-B2 INCONCLUSIVE | probe at {probe:.4f}, but the control is "
              f"at {ctrl:.4f} too -- {args.at} iterations is not long enough "
              f"for this curriculum to promote on any terrain, so this says "
              f"nothing about the stairs. Re-probe longer, do not change the "
              f"riser height on this evidence.")
        raise SystemExit(2)
    print(f"G-B2 FAIL | probe pinned at {probe:.4f} while the control reached "
          f"{ctrl:.4f} at the same iteration -- the terrain is the difference")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
