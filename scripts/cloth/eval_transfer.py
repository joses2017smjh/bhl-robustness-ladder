#!/usr/bin/env python3
"""C3: zero-shot transfer gap. Rigid success minus deformable success.

Without a deformable Isaac run this scores the kinematic rigid policy against
itself as a sanity check (gap must be 0) and writes a template for the real
C3 cell. It will not invent a cloth success rate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from bhl_robust.cloth.metrics import transfer_gap


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rigid", type=Path, required=True, help="C0 or C1 metrics JSON")
    p.add_argument("--deformable", type=Path, default=None,
                   help="C2/C3 metrics JSON. Omit to record 'not yet measured'.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rigid = json.loads(args.rigid.read_text())
    # Accept either a flat metrics dict or the C1 {eval: ...} wrapper.
    rigid_rate = rigid.get("success_rate", rigid.get("eval", {}).get("success_rate"))
    if rigid_rate is None:
        raise SystemExit(f"no success_rate in {args.rigid}")

    if args.deformable is None or not args.deformable.exists():
        payload = {
            "rigid_success": rigid_rate,
            "zero_shot_deformable_success": None,
            "transfer_gap": None,
            "note": "deformable cell not yet measured; refuse to invent it",
        }
    else:
        soft = json.loads(args.deformable.read_text())
        soft_rate = soft.get("success_rate", soft.get("eval", {}).get("success_rate"))
        payload = {
            "rigid_success": rigid_rate,
            "zero_shot_deformable_success": soft_rate,
            "transfer_gap": transfer_gap(rigid_rate, soft_rate),
        }
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
