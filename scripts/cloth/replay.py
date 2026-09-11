#!/usr/bin/env python3
"""Replay / render entry for cloth-sort.

Isaac replay uses the existing ``scripts/train_play.py`` path and the registered
gym ids. This wrapper only prints the exact command so nobody invents a
second play entrypoint.
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="ClothSort-BHL-Rigid-Oracle-v0")
    p.add_argument("--run-name", default=None)
    args = p.parse_args()
    cmd = (
        "BHL_STACK=v60 ENABLE_CAMERAS=1 python scripts/train_play.py "
        f"--task {args.task} --headless --enable_cameras"
    )
    if args.run_name:
        cmd += f" --load_run {args.run_name}"
    print(cmd)
    print("Submit via the existing train_play sbatch pattern, not a new renderer.")


if __name__ == "__main__":
    main()
