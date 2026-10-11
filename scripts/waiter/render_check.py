#!/usr/bin/env python3
"""Refuse to run a Waiter camera job on a node that does not render the head camera like the demonstrations.

The phase 3 policy was trained on frames rendered through the GPU's EGL driver. On a node where that driver is not
available to the job (the MIG slices of dgxh-1: MIG has no graphics support), MuJoCo silently falls back to Mesa's
software renderer; the image then differs on more than half of its pixels and the policy does markedly worse
(found 2026-10-10: runs on dgxh-1). This check renders the reset frame of two demonstration seeds and compares it
with the stored demonstration frames (results/waiter-20261005/render_reference/reset_frames.npz).

PASS iff, for each seed, the mean absolute difference is <= 0.5 grey levels and at most 0.5% of the pixels differ by
more than 8 levels in any channel (GPU nodes measured: <= 1 level on < 0.1% of pixels; software rendering: mean 3-5,
3-5% of pixels over 8). Exit code 0 on PASS, 1 on FAIL.

usage: render_check.py [--deploy <deploy.yaml>] [--upstream <dir>] [--reference <npz>]
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

MAX_MEAN_ABS = 0.5
MAX_FRAC_OVER_8 = 0.005


def compare(img: np.ndarray, ref: np.ndarray) -> dict:
    d = np.abs(img.astype(np.int16) - ref.astype(np.int16))
    mean_abs, frac = float(d.mean()), float((d.max(-1) > 8).mean())
    return {"mean_abs": mean_abs, "frac_over_8": frac, "max": int(d.max()),
            "ok": bool(img.shape == ref.shape and mean_abs <= MAX_MEAN_ABS and frac <= MAX_FRAC_OVER_8)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", type=Path,
                    default=REPO / "results/waiter-20261005/wbc1c/training/waiter-wbc1c-s7.deploy.yaml")
    ap.add_argument("--upstream", type=Path, default=REPO / "external/Berkeley-Humanoid-Lite")
    ap.add_argument("--reference", type=Path, default=REPO / "results/waiter-20261005/render_reference/reset_frames.npz")
    a = ap.parse_args(argv)
    from bhl_robust.eval.waiter_env import WaiterEnv
    env = WaiterEnv(a.deploy, a.upstream, f"/tmp/waiter-render-check-{os.getpid()}", render=True)
    host, ok = socket.gethostname().split(".")[0], True
    with np.load(a.reference) as ref:
        for key in ref.files:
            r = compare(env.reset(int(key.split("_")[1]))["image"], ref[key])
            ok &= r["ok"]
            print(f"[render-check] {host} {key}: mean |diff| {r['mean_abs']:.3f}, pixels over 8 levels "
                  f"{100 * r['frac_over_8']:.2f}%, max {r['max']} -> {'ok' if r['ok'] else 'DIFFERENT'}", flush=True)
    env.close()
    print(f"[render-check] {'PASS' if ok else 'FAIL: this node does not render like the demonstrations'} "
          f"(MUJOCO_GL={os.environ.get('MUJOCO_GL')}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
