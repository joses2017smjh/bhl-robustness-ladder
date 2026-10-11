"""The pilot must not mix simulation completion with estimator/hardware claims."""
import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest

PATH = Path(__file__).parents[1] / "scripts/bench/sensor_campaign.py"
SPEC = importlib.util.spec_from_file_location("sensor_campaign_test", PATH)
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


def wheel(tmp_path, names):
    path = tmp_path / "runtime.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, "fixture")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_verified_runtime_stays_private(tmp_path):
    path, sha = wheel(tmp_path, ["onnxruntime/__init__.py", "onnxruntime_gpu-1.22.0.dist-info/METADATA"])
    target = tmp_path / "private"
    receipt = CAMPAIGN.unpack_runtime(path, sha, target)
    assert receipt["package"] == "onnxruntime-gpu==1.22.0"
    assert (target / "onnxruntime/__init__.py").read_text() == "fixture"


@pytest.mark.parametrize("names", [["../escape"], ["/escape"], ["other_package/__init__.py"],
                                   ["onnxruntime/x", "onnxruntime/x"], ["onnxruntime/..\\escape"]])
def test_runtime_rejects_path_and_package_changes(tmp_path, names):
    path, sha = wheel(tmp_path, names)
    with pytest.raises(ValueError):
        CAMPAIGN.unpack_runtime(path, sha, tmp_path / "private")
    assert not (tmp_path / "private").exists()


def test_runtime_rejects_wrong_hash(tmp_path):
    path, _ = wheel(tmp_path, ["onnxruntime/__init__.py"])
    with pytest.raises(ValueError, match="hash"):
        CAMPAIGN.unpack_runtime(path, "0" * 64, tmp_path / "private")


def test_stages_preserve_target_model_and_stereo_gpu(tmp_path):
    research = {"capture": {"pilot": {"frames_per_scene": 24}},
                "timing": {"pilot": {"warmup": 30, "timed": 200}}}
    stages = CAMPAIGN.stage_commands(tmp_path, tmp_path / "protocol.json", tmp_path / "replay",
                                   tmp_path / "out", tmp_path / "model.onnx", "pilot", research)
    assert [name for name, _ in stages] == ["capture", "stereo", "geometry", "geometry_ffs", "pose_readiness"]
    stereo = stages[1][1]
    assert stereo[stereo.index("--provider") + 1] == "CUDAExecutionProvider"
    assert stereo[stereo.index("--model") + 1] == str(tmp_path / "model.onnx")
    assert stereo[stereo.index("--warmup") + 1] == "30"
    assert stereo[stereo.index("--timed") + 1] == "200"
    ffs_geometry = stages[3][1]
    assert ffs_geometry[ffs_geometry.index("--stereo-method") + 1] == "c_fast_foundationstereo"


def test_unknown_phase_rejected(tmp_path):
    with pytest.raises(ValueError):
        CAMPAIGN.stage_commands(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, "confirmation", {})


def test_fusion_pass_cannot_conceal_missing_terrain():
    with pytest.raises(ValueError, match="map stage incomplete"):
        CAMPAIGN.validate_stage_result("geometry", {"status": "PASS", "terrain_status": "INCOMPLETE"})
    with pytest.raises(ValueError, match="hazard"):
        CAMPAIGN.validate_stage_result("geometry", {"status": "PASS", "terrain_status": "PASS"},
                                       {"terrain": {"status": "PASS", "hazard_evaluation_ready": False}})
    CAMPAIGN.validate_stage_result("geometry", {"status": "PASS", "terrain_status": "PASS"},
                                   {"terrain": {"status": "PASS", "hazard_evaluation_ready": True}})


def test_pose_readiness_cannot_become_unrun_science_result():
    valid = {"status": "PASS", "scientific_status": "BLOCKED_RUNTIME", "estimated_pose_outputs": 0,
             "closed_loop_episodes": 0}
    CAMPAIGN.validate_stage_result("pose_readiness", valid)
    with pytest.raises(ValueError, match="zero unrun"):
        CAMPAIGN.validate_stage_result("pose_readiness", {**valid, "closed_loop_episodes": 432})
