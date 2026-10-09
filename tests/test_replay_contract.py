"""Reject evaluator leakage, corrupt recordings and nonrigid frame definitions."""
import json

import numpy as np
import pytest

from bhl_robust.research.replay_contract import member, transform, audit_replay


@pytest.mark.parametrize("name", ["evaluator/truth.npz", "../inference/image.png", "/inference/image.png", "inference/..\\image.png"])
def test_inference_path_cannot_consume_evaluator_or_escape(tmp_path, name):
    with pytest.raises(ValueError):
        member(tmp_path, name, "inference")


def test_link_rejected(tmp_path):
    (tmp_path / "inference").mkdir()
    (tmp_path / "original").write_text("image")
    (tmp_path / "inference/image").symlink_to(tmp_path / "original")
    with pytest.raises(ValueError):
        member(tmp_path, "inference/image", "inference")


def test_parent_link_cannot_alias_evaluator(tmp_path):
    (tmp_path / "evaluator").mkdir()
    (tmp_path / "evaluator/image").write_text("truth")
    (tmp_path / "inference").symlink_to(tmp_path / "evaluator", target_is_directory=True)
    with pytest.raises(ValueError):
        member(tmp_path, "inference/image", "inference")


def test_proper_rigid_frames(tmp_path):
    assert np.array_equal(transform(np.eye(4)), np.eye(4))
    for bad in (np.diag([2., 1., 1., 1.]), np.diag([-1., 1., 1., 1.]), np.full((4, 4), np.nan)):
        with pytest.raises(ValueError):
            transform(bad)


def test_legacy_sector_logs_not_raw_replay(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "maze_trace", "lidar_sector_m": [1.] * 36}))
    with pytest.raises(ValueError, match="unsupported"):
        audit_replay(tmp_path)
