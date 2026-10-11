"""Local trained assets for physics tests, without mutable training-run dirs."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def frozen_humanoid_repo(tmp_path):
    """MissionEnv's expected layout, with pinned geometry and the exact gait.

    The production environment finds the gait under upstream/logs. Keep that
    lookup unchanged by materializing its inputs in a disposable repo rather
    than writing training outputs into the pinned submodule.
    """
    checkout = Path(__file__).resolve().parents[1]
    fixture = checkout / "tests/fixtures/arms-dr1.0-s0"
    receipt = json.loads((fixture / "provenance.json").read_text())
    for name in ("deploy.yaml", "policy.onnx"):
        assert hashlib.sha256((fixture / name).read_bytes()).hexdigest() == receipt["files_sha256"][name]
    repo = tmp_path / "frozen-humanoid-repo"
    upstream = repo / "external/Berkeley-Humanoid-Lite"
    upstream.mkdir(parents=True)
    (upstream / "source").symlink_to(checkout / "external/Berkeley-Humanoid-Lite/source", target_is_directory=True)
    export = upstream / "logs/rsl_rl/humanoid" / receipt["run_directory"] / "exported"
    export.mkdir(parents=True)
    shutil.copyfile(fixture / "policy.onnx", export / "policy.onnx")
    # RlController opens the path from the config; its loader does not resolve
    # relative paths. Only this asset pointer changes in the disposable copy.
    lines = (fixture / "deploy.yaml").read_text().splitlines(keepends=True)
    lines[0] = f"policy_checkpoint_path: {json.dumps(str(export / 'policy.onnx'))}\n"
    (export / "deploy.yaml").write_text("".join(lines))
    return repo
