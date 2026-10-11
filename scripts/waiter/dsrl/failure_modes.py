#!/usr/bin/env python3
"""Dev diagnostic: sort the failed episodes of a trace_base.py run by what went wrong.

Each failed episode gets one label, from the object poses logged at every decision (ORACLE poses, analysis only):
  robot fell
  wrong tumbler picked up (the other colour was the first one carried), with or without carrying the tray to a table
  no tumbler picked up:                target knocked over or pushed | target untouched
  target picked up, never on the tray: tipped over | still held at the end | upright over the tray, not resting on its
                                       floor | put down on table A outside the tray | on the ground
  target on the tray, not delivered:   tray never lifted | tumbler lost from the tray | tray dropped | carried to the
                                       wrong table | out of time on the way | at the table, not set down | set down,
                                       not finished
"Picked up" = rose 3 cm above its own start height while upright, at a decision; the env's stage 1 only needs 1.5 cm
at any step, which a bump also produces. Heights are relative to each object's start, so no table height is needed.

usage: failure_modes.py <trace.json> [--list]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

TRAY_HALF = 0.13          # waiter_scene.TRAY_HALF (both axes); checked in tests/test_waiter_dsrl.py
CUP_R = 0.038             # waiter_scene.CUP_R
LIFT_M = 0.015            # the env's "tumbler lifted" margin above the table
PICK_M = 0.03             # picked up: this far above its own start height, upright
CARRIED_M = 1.5           # the tray ended this far from its start: it was carried to a table
TIPPED_DEG = 30.0
MOVED_M = 0.03
GROUND_DROP_M = 0.20


def in_tray(cup, tray) -> bool:
    """Tumbler centre inside the tray floor (tray frame; the env's test, with the tray's logged yaw)."""
    dx, dy = cup[0] - tray[0], cup[1] - tray[1]
    c, s = math.cos(tray[4]), math.sin(tray[4])
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return abs(lx) < TRAY_HALF - 0.5 * CUP_R and abs(ly) < TRAY_HALF - 0.5 * CUP_R


def first_pick(ep: dict):
    """Which tumbler was picked up first: "target", "other" or None."""
    tr = ep["trace"]
    tgt, oth = f"tumbler_{ep['target_colour']}", f"tumbler_{'blue' if ep['target_colour'] == 'red' else 'red'}"
    for s in tr:
        for name, key in (("target", tgt), ("other", oth)):
            if s[key][2] - tr[0][key][2] > PICK_M and s[key][3] < 45.0:
                return name
    return None


def classify(ep: dict) -> str:
    tr = ep["trace"]
    tgt = f"tumbler_{ep['target_colour']}"
    c0, t0 = tr[0][tgt], tr[0]["tray"]
    ce, te = tr[-1][tgt], tr[-1]["tray"]
    moved = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])  # noqa: E731
    if ep["fell"]:
        return "robot fell"
    pick = first_pick(ep)
    if pick == "other":
        return "wrong tumbler picked up" + (" and carried to a table" if moved(te, t0) > CARRIED_M else "")
    if pick is None:
        if ce[3] > TIPPED_DEG or moved(ce, c0) > MOVED_M:
            return "no tumbler picked up: target knocked over or pushed"
        return "no tumbler picked up: target untouched"
    stage = ep["stage_max"]
    if stage <= 1:
        if ce[2] - c0[2] < -GROUND_DROP_M:
            return "target picked up: ended on the ground"
        if ce[3] > TIPPED_DEG:
            return "target picked up: tipped over " + ("in the tray" if in_tray(ce, te) else "outside the tray")
        if in_tray(ce, te):
            # upright inside the tray's floor but never counted as resting on it: hung on the finger or the rim
            return "target picked up: upright over the tray, never resting on its floor"
        if ce[2] - c0[2] > LIFT_M:
            return "target picked up: still held away from the tray at the end"
        return "target picked up: put down on table A outside the tray"
    # the target tumbler was on the tray at some point
    if not in_tray(ce, te) or ce[3] > TIPPED_DEG:
        return "tray: tumbler lost from the tray"
    if te[2] - t0[2] < -GROUND_DROP_M or te[3] > 45.0:
        return "tray: tray dropped"
    if stage == 2:
        return "tray: tray never lifted"
    if stage == 3:
        by = tr[-1]["base"][1]
        if abs(by) > 1.0 and (by > 0) != (ep["target_side"] == "left"):
            return "tray: carried to the wrong table"
        return "tray: out of time on the way to the table"
    return "tray: at the table, not set down" if stage == 4 else "tray: set down, not finished"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--list", action="store_true", help="also print one line per failed episode")
    a = ap.parse_args(argv)
    res = json.loads(a.trace.read_text())
    eps = res["episodes"]
    fails = [e for e in eps if not e["success"]]
    labels = Counter(classify(e) for e in fails)
    picks = Counter(first_pick(e) for e in eps)
    ok = sum(e["success"] for e in eps if first_pick(e) == "target")
    print(f"{a.trace.name}: {len(eps) - len(fails)}/{len(eps)} successes ({res['arm']} arm, {res['exec_horizon']} "
          f"executed steps, dev seeds {res['seeds'][0]}-{res['seeds'][-1]}); {len(fails)} failures:")
    for label, n in labels.most_common():
        print(f"  {n:3d}  {label}")
    print(f"first tumbler picked up: target {picks['target']}, other {picks['other']}, none {picks[None]}; "
          f"success after picking the target: {ok}/{picks['target']}")
    if a.list:
        for e in fails:
            print(f"    seed {e['seed']} {e['target_colour']:4s} -> {e['target_side']:5s} stage<={e['stage_max']} "
                  f"| {classify(e)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
