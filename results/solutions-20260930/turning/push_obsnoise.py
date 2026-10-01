"""Matched push regression (run_eval, 0.5 m/s every 3 s, 12 s, 6 commands x seeds 0-9 = 60) with Isaac's training
observation corruption added to the policy input (uniform: ang vel 0.3, gravity 0.05, joint pos 0.05, joint vel 2.0).
Report-only diagnostic; repo untouched (monkeypatch in this process only)."""
import sys
import numpy as np
from berkeley_humanoid_lite_lowlevel.policy import rl_controller as rc

mult = float(sys.argv[1])
HALF = np.concatenate([np.zeros(3), np.full(3, 0.3), np.full(3, 0.05), np.full(22, 0.05), np.full(22, 2.0), np.zeros(22)]).astype(np.float32)
rng = np.random.default_rng(12345)
_orig = rc.OnnxPolicy.forward
def noisy_forward(self, observations):
    if mult > 0:
        observations = observations + (mult * HALF * rng.uniform(-1, 1, observations.shape)).astype(np.float32)
    return _orig(self, observations)
rc.OnnxPolicy.forward = noisy_forward
from bhl_robust.eval import run_eval
sys.exit(run_eval.main(sys.argv[2:]))
