"""Waiter phase 3 gate scorer (scripts/waiter/vla/phase3_gate.py): the verdict from synthetic scored results, the
input checks (gate hash, checkpoint, seeds, instruction ids, stance lock) and the write-once rule. No simulator."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "waiter" / "vla"))
import phase3_gate as G  # noqa: E402


def _rule(tmp_path, min_own=30, min_held=25):
    r = {"rule": "test rule", "checkpoint": "/x/sft1/step_017000", "checkpoint_data_hash": "abc", "seed0": 4000,
         "seed1": 4099, "held_out_template": "T5", "held_out_instruction_id": 5, "min_own": min_own,
         "min_held_out": min_held}
    p = tmp_path / "rule.json"
    p.write_text(json.dumps(r))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def _result(sha, ids, k, held_out_all=False, **over):
    eps = [{"seed": s, "success": i < k, "fell": False, "template_id": "T5" if (held_out_all or s % 6 == 5) else "T0",
            "held_out_template": bool(held_out_all or s % 6 == 5)} for i, s in enumerate(range(4000, 4100))]
    r = {"purpose": "scored", "env": "waiter", "synthetic": False, "gate": {"sha256": sha},
         "checkpoint": "/nfs/other/path/step_017000", "checkpoint_data_hash": "abc", "seeds": list(range(4000, 4100)),
         "instruction_ids": ids, "episodes": eps, "completed": True, "counterfactual_slot": None, "stance_lock": True,
         "fixed_base": False}
    r.update(over)
    return r


def _run(tmp_path, own, held, rule_path):
    (tmp_path / "own.json").write_text(json.dumps(own))
    (tmp_path / "held.json").write_text(json.dumps(held))
    return G.main(["--rule", str(rule_path), "--own", str(tmp_path / "own.json"), "--held-out",
                   str(tmp_path / "held.json"), "--out", str(tmp_path / "verdict.json")])


def test_pass_fail_and_write_once(tmp_path):
    rule, sha = _rule(tmp_path)
    assert _run(tmp_path, _result(sha, [None], 45), _result(sha, [5], 25, held_out_all=True), rule) == 0
    v = json.loads((tmp_path / "verdict.json").read_text())
    assert v["verdict"] == "PASS" and v["own_instruction"]["successes"] == 45 and v["held_out_template"]["successes"] == 25
    assert v["rule_sha256"] == sha and v["own_instruction"]["seen_templates_only"]["n"] < 100
    with pytest.raises(SystemExit):                                   # never overwritten
        _run(tmp_path, _result(sha, [None], 45), _result(sha, [5], 25, held_out_all=True), rule)
    (tmp_path / "verdict.json").unlink()
    assert _run(tmp_path, _result(sha, [None], 45), _result(sha, [5], 24, held_out_all=True), rule) == 0
    assert json.loads((tmp_path / "verdict.json").read_text())["verdict"] == "FAIL"


@pytest.mark.parametrize("which, change", [
    ("own", {"purpose": "dev"}), ("own", {"gate": {"sha256": "0" * 64}}), ("own", {"checkpoint_data_hash": "zzz"}),
    ("own", {"checkpoint": "/x/step_020000"}), ("own", {"stance_lock": False}), ("own", {"instruction_ids": [0]}),
    ("held", {"instruction_ids": [None]}), ("held", {"completed": False}), ("held", {"counterfactual_slot": "side"}),
    ("held", {"seeds": list(range(4000, 4050))}),
])
def test_invalid_inputs_write_nothing(tmp_path, which, change):
    rule, sha = _rule(tmp_path)
    own, held = _result(sha, [None], 60), _result(sha, [5], 60, held_out_all=True)
    (own if which == "own" else held).update(change)
    with pytest.raises(SystemExit) as e:
        _run(tmp_path, own, held, rule)
    assert "INVALID" in str(e.value) and not (tmp_path / "verdict.json").exists()


def test_held_out_run_must_use_the_held_out_template(tmp_path):
    rule, sha = _rule(tmp_path)
    with pytest.raises(SystemExit) as e:
        _run(tmp_path, _result(sha, [None], 60), _result(sha, [5], 60, held_out_all=False), rule)
    assert "not template T5" in str(e.value)
