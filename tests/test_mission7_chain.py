"""Chain-window classification for the Mission 7 route probe.

Pins the review finding that the window describing a stall must start at the
stall condition: a synthetic episode that walks 4 m and then stands still is
'progressing' over the whole post-stage window and a stall over the window
anchored at the stall. No MuJoCo, no policy weights.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from mission7_route_handoff_probe import summarize_chain  # noqa: E402

HZ = 25
JOINTS = 22


def synthetic_chain(walk_s=10.0, stall_s=20.0, speed=0.4):
    """Walk along +x with a cycling gait, then stand still with frozen targets."""
    n_walk, n_stall = int(walk_s * HZ), int(stall_s * HZ)
    n = n_walk + n_stall
    t = np.arange(n) / HZ
    x = np.where(t < walk_s, speed * t, speed * walk_s)
    base = np.stack([x, np.zeros(n), np.where(t < walk_s, speed, 0.0), np.zeros(n), np.zeros(n)], 1)
    phase = 2 * np.pi * 1.5 * t
    gait = 0.6 * np.sin(phase)[:, None] * np.ones((1, JOINTS))
    tgt = np.where((t < walk_s)[:, None], gait, 0.0)
    jpos = np.roll(tgt, 1, axis=0) * 0.5  # never tracks fully; that is normal for this gait
    contact = np.ones(n)
    foot = lambda side: np.stack([contact, x + side * 0.1, np.full(n, side * 0.1), np.full(n, 0.05)], 1)
    return {
        "t": t.tolist(), "cmd_in": np.tile([0.3, 0.0, 0.0], (n, 1)).tolist(),
        "prev_in": tgt.tolist(), "jpos_in": jpos.tolist(), "jvel_in": np.gradient(jpos, axis=0).tolist(),
        "raw_out": (tgt / 0.25).tolist(), "tgt_out": tgt.tolist(), "ctrl_pre": np.zeros((n, JOINTS)).tolist(),
        "foot_l": foot(+1).tolist(), "foot_r": foot(-1).tolist(), "base": base.tolist(), "phase": ["advance"] * n,
    }


class StallAnchoredWindow(unittest.TestCase):
    def setUp(self):
        self.chain = synthetic_chain()
        self.defaults = np.zeros(JOINTS)
        self.effort = np.full(JOINTS, 4.0)
        self.reference = summarize_chain(self.chain, 1.0, 9.5, self.defaults, self.effort)

    def test_reference_window_is_walking(self):
        self.assertEqual(self.reference["mechanism"], "progressing")
        self.assertGreater(self.reference["target_range_mean_rad"], 1.0)

    def test_whole_window_hides_the_stall(self):
        whole = summarize_chain(self.chain, 1.0, 30.0, self.defaults, self.effort, reference=self.reference)
        self.assertEqual(whole["mechanism"], "progressing")
        self.assertGreater(whole["base_progress_m"], 3.5)

    def test_stall_anchored_window_labels_the_stall(self):
        anchored = summarize_chain(self.chain, 10.0, 30.0, self.defaults, self.effort, reference=self.reference)
        self.assertEqual(anchored["mechanism"], "frozen_targets")
        self.assertLess(anchored["base_progress_m"], 0.05)
        self.assertLess(anchored["target_range_ratio_vs_reference"], 0.05)

    def test_short_window_is_reported_not_labelled(self):
        short = summarize_chain(self.chain, 10.0, 10.5, self.defaults, self.effort, reference=self.reference)
        self.assertEqual(short["mechanism"], "window_too_short")


if __name__ == "__main__":
    unittest.main()
