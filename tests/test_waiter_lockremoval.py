"""Option 3 of the stance-lock work (2026-10-14): the predeclared candidate rule (scripts/waiter/lockremoval_select.py),
its write-once record, the second scored range in expert_gate.py, and the launchers. No simulator."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "waiter"))

import lockremoval_select as S  # noqa: E402


def _row(run, dev, disp):
    return {"run": run, "deploy": f"/x/{run}.dual.deploy.yaml", "dev_successes": dev, "q4_mean_disp_m": disp}


def test_choose_most_dev_then_least_drift_then_lowest_seed():
    rows = [_row("waiter-wbc1i-s27", 20, 0.05), _row("waiter-wbc1i-s28", 27, 0.07), _row("waiter-wbc1i-s29", 27, 0.04)]
    pick = S.choose(rows, 26)
    assert pick["candidate"] == "waiter-wbc1i-s29" and pick["scored"] and pick["dev_successes"] == 27
    tie = [_row("waiter-wbc1i-s31", 26, 0.05), _row("waiter-wbc1i-s30", 26, 0.05)]
    assert S.choose(tie, 26)["candidate"] == "waiter-wbc1i-s30"
    low = [_row("waiter-wbc1i-s27", 25, 0.03), _row("waiter-wbc1i-s28", None, 0.01)]   # a missing dev check is out
    pick = S.choose(low, 26)
    assert pick["candidate"] == "waiter-wbc1i-s27" and not pick["scored"]
    assert S.choose([_row("waiter-wbc1i-s27", None, 0.0)], 26) == {"candidate": None, "scored": False}


def test_main_writes_once(tmp_path, capsys):
    q4 = tmp_path / "q4.json"
    q4.write_text(json.dumps({"episodes": [{"fell_at_s": None, "disp_m": 0.04}, {"fell_at_s": None, "disp_m": 0.06},
                                           {"fell_at_s": 1.0, "disp_m": None}]}))
    dev = tmp_path / "gate.json"
    dev.write_text(json.dumps({"successes": 28, "n": 32}))
    out = tmp_path / "sel" / "selection.json"
    args = ["--candidate", f"waiter-wbc1i-s30=/d/s30.yaml={q4}={dev}",
            "--candidate", f"waiter-wbc1i-s31=/d/s31.yaml={q4}={tmp_path / 'missing.json'}", "--out", str(out)]
    assert S.main(args) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == "SELECT: waiter-wbc1i-s30 /d/s30.yaml"
    rec = json.loads(out.read_text())
    assert rec["candidate"] == "waiter-wbc1i-s30" and rec["scored"] and rec["candidates"][0]["q4_mean_disp_m"] == pytest.approx(0.05)
    with pytest.raises(SystemExit, match="never overwritten"):
        S.main(args)


def test_second_scored_range_and_launchers():
    import importlib
    eg = importlib.import_module("expert_gate")
    assert eg.RANGES["lock2"] == (9000, 9099) and eg.RANGES["gate"] == (2000, 2099)
    t = (REPO / "slurm/repo20260923/gpu_waiter_wbc.sbatch").read_text()
    assert 'if [ "$VARIANT" = 1i ]; then SEED0=${WAITER_1I_SEED0:-$SEED0}; RES_NAME=${WAITER_1I_RES:-$RES_NAME}; fi' in t
    s = (REPO / "slurm/repo20260923/waiter_lockremoval_select.sbatch").read_text()
    assert "GATE_START_ONLY=1 GATE_SEED0=9000 GATE_SEED1=9099 GATE_RANGE=lock2" in s and "GATE_MIN_SUCCESS=80" in s
    assert '--min-dev "${SEL_MIN_DEV:-26}"' in s and '"SELECT: none"*)' in s
    assert "| 9000-9099 |" in (REPO / "docs/WAITER_INTERFACES.md").read_text()
