"""Waiter dev diagnostic scripts/waiter/dsrl/failure_modes.py: the label each kind of failed episode gets, on
synthetic traces, and its two scene constants against waiter_scene.py. No simulator."""
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "waiter" / "dsrl"))
import failure_modes as F  # noqa: E402

Z = 0.46


def _snap(red=(0.30, 0.33, Z, 0.0), blue=(0.30, -0.33, Z, 0.0), tray=(0.36, 0.0, Z, 0.0, 0.0), base=(0.0, 0.0, 0.0),
          t=0.0):
    return {"t": t, "base": list(base), "locked": True, "tumbler_red": list(red) + [0.0], "tumbler_blue": list(blue) + [0.0],
            "tray": list(tray), "pivot_left": [0, 0, 0], "pivot_right": [0, 0, 0]}


def _ep(stage, *snaps, fell=False, side="left"):
    return {"seed": 1300, "target_colour": "red", "target_side": side, "success": False, "fell": fell,
            "stage_max": stage, "trace": [_snap()] + list(snaps)}


def test_constants_match_the_scene():
    tree = ast.parse((REPO / "src/bhl_robust/eval/waiter_scene.py").read_text())
    ns: dict = {}
    keep = [n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("TRAY_HALF", "CUP_R")]
    exec(compile(ast.Module(body=keep, type_ignores=[]), "waiter_scene", "exec"), ns)
    assert ns["TRAY_HALF"] == (F.TRAY_HALF, F.TRAY_HALF) and ns["CUP_R"] == F.CUP_R


def test_in_tray_uses_the_tray_yaw():
    tray = [0.36, 0.0, Z, 0.0, 0.0]
    assert F.in_tray([0.36 + 0.10, 0.0, Z, 0.0], tray) and not F.in_tray([0.36 + 0.12, 0.0, Z, 0.0], tray)
    turned = [0.36, 0.0, Z, 0.0, 0.7854]                       # 45 deg: a corner now points along x
    assert F.in_tray([0.36 + 0.10, 0.10, Z, 0.0], tray) and not F.in_tray([0.36 + 0.10, 0.10, Z, 0.0], turned)
    assert F.in_tray([0.36 + 0.14, 0.0, Z, 0.0], turned) and not F.in_tray([0.36 + 0.14, 0.0, Z, 0.0], tray)


UP = _snap(red=(0.30, 0.33, Z + 0.05, 3.0))                    # the target (red) picked up
OTHER_UP = _snap(blue=(0.30, -0.33, Z + 0.05, 3.0))            # the other tumbler picked up


def test_first_pick():
    assert F.first_pick(_ep(0, _snap())) is None
    assert F.first_pick(_ep(1, _snap(red=(0.30, 0.33, Z + 0.02, 25.0)))) is None        # a bump: 2 cm, tilted
    assert F.first_pick(_ep(1, _snap(red=(0.30, 0.33, Z + 0.05, 80.0)))) is None        # lying on something
    assert F.first_pick(_ep(1, UP, OTHER_UP)) == "target" and F.first_pick(_ep(1, OTHER_UP, UP)) == "other"


def test_labels():
    far = dict(tray=(-0.3, 2.3, Z, 1.0, 1.57), base=(-0.3, 1.9, 1.57))
    cases = [
        (_ep(1, UP, _snap(), fell=True), "robot fell"),
        (_ep(0, _snap()), "no tumbler picked up: target untouched"),
        (_ep(1, _snap(red=(0.34, 0.33, Z, 0.0))), "no tumbler picked up: target knocked over or pushed"),
        (_ep(1, _snap(red=(0.30, 0.33, Z, 80.0))), "no tumbler picked up: target knocked over or pushed"),
        (_ep(0, OTHER_UP, _snap(blue=(0.36, 0.0, Z + 0.01, 1.0))), "wrong tumbler picked up"),
        (_ep(1, OTHER_UP, _snap(blue=(-0.3, 2.3, Z + 0.01, 1.0), **far)), "wrong tumbler picked up and carried to a table"),
        (_ep(1, UP, _snap(red=(0.50, 0.60, 0.02, 85.0))), "target picked up: ended on the ground"),
        (_ep(1, UP, _snap(red=(0.36, 0.05, Z + 0.02, 70.0))), "target picked up: tipped over in the tray"),
        (_ep(1, UP, _snap(red=(0.30, 0.25, Z, 88.0))), "target picked up: tipped over outside the tray"),
        (_ep(1, UP, _snap(red=(0.36, 0.05, Z + 0.03, 4.0))),
         "target picked up: upright over the tray, never resting on its floor"),
        (_ep(1, UP, _snap(red=(0.30, 0.25, Z + 0.08, 6.0))), "target picked up: still held away from the tray at the end"),
        (_ep(1, UP, _snap(red=(0.31, 0.30, Z, 1.0))), "target picked up: put down on table A outside the tray"),
        (_ep(2, UP, _snap(red=(0.36, 0.05, Z + 0.01, 2.0))), "tray: tray never lifted"),
        (_ep(3, UP, _snap(red=(0.36, 0.30, Z, 2.0))), "tray: tumbler lost from the tray"),
        (_ep(3, UP, _snap(red=(0.2, 1.0, 0.05, 3.0), tray=(0.2, 1.0, 0.04, 2.0, 0.0))), "tray: tray dropped"),
        (_ep(3, UP, _snap(red=(-0.3, -2.0, 0.61, 2.0), tray=(-0.3, -2.0, 0.6, 3.0, -1.57), base=(-0.3, -1.6, -1.57))),
         "tray: carried to the wrong table"),
        (_ep(3, UP, _snap(red=(-0.1, 0.8, 0.61, 2.0), tray=(-0.1, 0.8, 0.6, 3.0, 1.57), base=(-0.2, 0.5, 1.57))),
         "tray: out of time on the way to the table"),
        (_ep(4, UP, _snap(red=(-0.3, 2.2, 0.61, 2.0), tray=(-0.3, 2.2, 0.6, 3.0, 1.57), base=(-0.3, 1.9, 1.57))),
         "tray: at the table, not set down"),
        (_ep(5, UP, _snap(red=(-0.3, 2.3, Z + 0.01, 2.0), **far)), "tray: set down, not finished"),
    ]
    for ep, want in cases:
        assert F.classify(ep) == want, (want, F.classify(ep))


def test_cli_counts(tmp_path, capsys):
    eps = [_ep(0, _snap()), _ep(0, OTHER_UP, _snap()), _ep(1, UP, _snap(red=(0.30, 0.25, Z, 88.0))),
           dict(_ep(6, UP, _snap()), success=True)]
    p = tmp_path / "trace.json"
    p.write_text(json.dumps({"arm": "base", "exec_horizon": 10, "seeds": [1300, 1301, 1302, 1303], "episodes": eps}))
    assert F.main([str(p), "--list"]) == 0
    out = capsys.readouterr().out
    assert "1/4 successes" in out and "  1  no tumbler picked up: target untouched" in out
    assert "  1  wrong tumbler picked up" in out and "  1  target picked up: tipped over outside the tray" in out
    assert "first tumbler picked up: target 2, other 1, none 1; success after picking the target: 1/2" in out
