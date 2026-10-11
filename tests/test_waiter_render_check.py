"""scripts/waiter/render_check.py: the comparison rule and the stored reference frames. No simulator."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "waiter"))
import render_check as R  # noqa: E402

REF = REPO / "results/waiter-20261005/render_reference"


def test_reference_frames_are_the_recorded_ones():
    meta = json.loads((REF / "reset_frames.json").read_text())
    with np.load(REF / "reset_frames.npz") as z:
        assert sorted(z.files) == ["seed_3000", "seed_3001"]
        for key in z.files:
            assert z[key].shape == (224, 224, 3) and z[key].dtype == np.uint8
            assert hashlib.sha256(z[key].tobytes()).hexdigest() == meta["sha256"][key]


def test_compare_rule():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    assert R.compare(ref, ref) == {"mean_abs": 0.0, "frac_over_8": 0.0, "max": 0, "ok": True}
    gpu = ref.copy()                                           # another GPU: one grey level on a few pixels
    gpu[:5, :9, 0] = np.clip(gpu[:5, :9, 0].astype(int) + 1, 0, 255)
    assert R.compare(gpu, ref)["ok"]
    soft = ref.astype(int)                                     # a software renderer: small shifts on over half the
    soft[::2] += 4                                             # pixels, large ones on a few percent of them
    soft[:10] += 60
    r = R.compare(np.clip(soft, 0, 255).astype(np.uint8), ref)
    assert not r["ok"] and r["mean_abs"] > R.MAX_MEAN_ABS and r["frac_over_8"] > R.MAX_FRAC_OVER_8
    few = ref.copy()                                           # 1% of the pixels far off: the mean passes, the
    few[:2, :, :] = 255 - few[:2, :, :]                        # share of large differences does not
    r = R.compare(few, ref)
    assert not r["ok"] and r["frac_over_8"] > R.MAX_FRAC_OVER_8
    assert (R.MAX_MEAN_ABS, R.MAX_FRAC_OVER_8) == (0.5, 0.005)


def test_launchers_run_the_check_and_skip_the_mig_node():
    for f in ("scripts/waiter/dsrl/dsrl_gpu.sbatch", "scripts/waiter/vla/rollout_vla.sbatch"):
        s = (REPO / f).read_text()
        assert "#SBATCH --exclude=dgxh-1" in s and "scripts/waiter/render_check.py" in s
        assert s.index("render_check.py\"") < s.index("exec \"$LEARNER_PY\"" if "dsrl" in f else "rollout_vla.py\" --env-python")
