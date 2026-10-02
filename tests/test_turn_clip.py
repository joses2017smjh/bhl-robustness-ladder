"""scripts/bench/turn_clip.py: pure helpers, refusal to overwrite, and that `simulate` still repeats
turn_test.run_command step for step (no simulator needed)."""
import inspect
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "bench"))

import turn_clip  # noqa: E402
import turn_test  # noqa: E402


def _body(fn, stop_at_return):
    """Source lines of fn's body, trailing comments and blank lines dropped, whitespace stripped."""
    out = []
    for line in inspect.getsource(fn).splitlines()[1:]:
        line = line.split("  #")[0].strip()
        if not line:
            continue
        if stop_at_return and line.startswith("return"):
            break
        out.append(line)
    return out


def test_simulate_repeats_run_command_step_for_step():
    """Every line of run_command up to its return appears, in order, in simulate (which may add lines)."""
    ref = _body(turn_test.run_command, stop_at_return=True)
    got = _body(turn_clip.simulate, stop_at_return=False)
    it = iter(got)
    missing = [line for line in ref if not any(line == g for g in it)]
    assert not missing, f"turn_clip.simulate no longer repeats run_command: {missing}"
    assert len(ref) > 25


def test_scored_yaw_and_match_boundary():
    scored = {"turns": [{"name": "turn+_s0", "yaw_deg": 196.8}, {"name": "turn-_s0", "yaw_deg": -206.2}]}
    assert turn_clip.scored_yaw(scored, "turn+_s0") == 196.8
    assert turn_clip.scored_yaw(scored, "turn-_s0") == -206.2
    assert turn_clip.scored_yaw(scored, "turn+_s1") is None
    assert turn_clip.matches(197.3, 196.8)            # exactly the 0.5 deg tolerance
    assert not turn_clip.matches(197.31, 196.8)
    assert not turn_clip.matches(196.8, None)


def test_card_follows_the_v2_per_run_criterion():
    assert turn_clip.card_for(150.0, 0.6, None)[1] is True
    assert turn_clip.card_for(149.9, 0.6, None)[1] is False
    assert turn_clip.card_for(-150.0, -0.6, None)[1] is True
    assert turn_clip.card_for(200.0, -0.6, None)[1] is False      # wrong direction
    text, ok = turn_clip.card_for(190.0, 0.6, 4.2)
    assert ok is False and text.startswith("FELL at 4.2")


def test_compose_has_even_sides_and_both_panels():
    rgb = np.full((96, 96, 3), 200, np.uint8)
    left = turn_clip.side_panel(rgb, "Left", "sub", 123.0, 0.6, None, None)
    right = turn_clip.side_panel(rgb, "Right", "sub", 4.0, 0.6, "turned +4\N{DEGREE SIGN}", False)
    assert left.size == (96, 96 + turn_clip.HEAD_H) and isinstance(left, Image.Image)
    frame = turn_clip.compose(left, right, "banner", ["line one", "line two"])
    h, w, c = frame.shape
    assert c == 3 and h % 2 == 0 and w % 2 == 0
    assert w >= 96 * 2 + 6 and h == turn_clip.BANNER_H + 96 + turn_clip.HEAD_H + turn_clip.FOOT_H


def test_refuses_to_overwrite_an_existing_clip(tmp_path, monkeypatch, capsys):
    gif = tmp_path / "clip.gif"
    gif.write_bytes(b"GIF89a")
    dummy = tmp_path / "none.yaml"
    argv = ["turn_clip.py", "--upstream", str(tmp_path), "--cache-dir", str(tmp_path),
            "--mp4", str(tmp_path / "clip.mp4"), "--gif", str(gif)]
    for side in ("left", "right"):
        argv += [f"--{side}-deploy", str(dummy), f"--{side}-scored", str(dummy),
                 f"--{side}-title", "t", f"--{side}-subtitle", "s"]
    monkeypatch.setattr(sys, "argv", argv)
    assert turn_clip.main() == 3
    assert "REFUSING" in capsys.readouterr().out
    assert gif.read_bytes() == b"GIF89a"
