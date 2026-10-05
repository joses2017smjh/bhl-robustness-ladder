"""scripts/bench/er_obs_local.py (ER-OBS-1-OW, open-weight observers): the parts that need no GPU or model."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from bhl_robust.eval import er_obs as E            # noqa: E402
from bhl_robust.eval.er_obs import client as C     # noqa: E402

spec = importlib.util.spec_from_file_location("er_obs_local_t", REPO / "scripts/bench/er_obs_local.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

GOOD = '{"cube_point": [512, 300], "lifted_clear": true, "on_floor": false, "robot_contact": true, "seated_flat": false}'


def test_same_prompt_parser_and_scoring_as_the_gemini_arm():
    assert L.C is C and L.C.TEXT == C.TEXT
    r = L.classify_text(GOOD, "x/y")
    assert r["kind"] == "ok" and r["answer"]["cube_point"] == [512.0, 300.0] and r["booleans"]["robot_contact"]
    assert L.classify_text("```json\n" + GOOD + "\n```", "x/y")["kind"] == "ok"
    assert L.classify_text("The cube is on the plinth.", "x/y")["kind"] == "non_json"
    bad_point = L.classify_text(GOOD.replace("[512, 300]", "[-1, -1]"), "x/y")
    assert bad_point["kind"] == "schema_violation" and bad_point["booleans"] is not None   # C1 still reads them


def test_decoding_and_models_are_declared():
    assert L.MAX_NEW_TOKENS == 256 and L.DTYPE == "bfloat16" and L.LATENCY_CALLS == 50
    assert L.MODELS == {"molmo2-er": ("allenai/Molmo2-ER", "dab22564403d2607855bb1fffb0721285b445081"),
                        "qwen3-vl-8b": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")}
    assert "OPEN-WEIGHT VLM observer" in L.LABEL and "Gemini" not in L.LABEL


def test_runs_refuse_a_revision_override_and_an_unpinned_model(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="--revision is for the development probe"):
        L.main(["calls", "--model", "qwen3-vl-8b", "--revision", "abc", "--frames", str(tmp_path), "--mode", "smoke",
                "--out", str(tmp_path / "o")])
    monkeypatch.setitem(L.MODELS, "qwen3-vl-8b", ("Qwen/Qwen3-VL-8B-Instruct", None))
    import types
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=lambda *a, **k: None))
    with pytest.raises(SystemExit, match="no pinned revision"):
        L.Observer("qwen3-vl-8b")


def test_probe_refuses_the_scored_frames(tmp_path):
    from test_er_obs import make_frames_dir
    run = make_frames_dir(tmp_path / "frames-run", "run", list(E.SCORED_SEEDS), 20, hw=(6, 8))
    with pytest.raises(SystemExit, match="SMOKE frames dir only"):
        L.main(["probe", "--model", "qwen3-vl-8b", "--revision", "abc", "--frames", str(run)])


def test_verdict_scores_failed_calls_as_the_gemini_arm_does(tmp_path):
    from test_er_obs import make_frames_dir
    import hashlib
    fr = make_frames_dir(tmp_path / "frames", "smoke", [900, 901], 14)
    man = json.loads((fr / "manifest.json").read_text())
    recs = []
    for f in man["frames"]:
        for variant in E.VARIANTS:
            png = (fr / f["files"]["rgb" if variant == "main" else "c1"]).read_bytes()
            if variant == "main":
                ans = {**{b: f["labels"][b] for b in E.BOOLEANS}, "cube_point": [500, 500]}
                r = L.classify_text(json.dumps(ans), "m")
            else:
                r = {"kind": "timeout", "booleans": None}          # every C1 call failed: scored CORRECT on C1
            r.update(frame_id=f["id"], variant=variant, image_sha256=hashlib.sha256(png).hexdigest())
            recs.append(r)
    calls = tmp_path / "calls.json"
    calls.write_text(json.dumps({"mode": "smoke", "model": "m", "records": recs, "prompt_schema_sha256": C.PROMPT_SCHEMA_SHA256,
                                 "frames_manifest_sha256": hashlib.sha256((fr / "manifest.json").read_bytes()).hexdigest()}))
    out = tmp_path / "v.json"
    assert L.main(["verdict", "accuracy", "--frames", str(fr), "--calls", str(calls), "--out", str(out)]) == 0
    v = json.loads(out.read_text())
    assert v["verdict"] == "PIPELINE_CHECK" and v["mode"] == "smoke"
    assert all(v["main_balanced_accuracy"][b]["ba"] in (1.0, None) for b in E.BOOLEANS)
    assert all(v["c1_balanced_accuracy"][b]["ba"] in (1.0, None) for b in E.BOOLEANS)   # failures help nothing
    assert v["checks"]["c1_ba_all_le_0.60"] is False
    with pytest.raises(FileExistsError):
        L.main(["verdict", "accuracy", "--frames", str(fr), "--calls", str(calls), "--out", str(out)])
