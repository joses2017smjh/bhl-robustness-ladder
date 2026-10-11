"""Enforce durable shared scope, immutable source and all-nine-shard dispatch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location("terrain_confirmation_dispatch", Path(__file__).resolve().parents[1] / "scripts/bench/terrain_confirmation_dispatch.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.fixture
def plan(tmp_path, monkeypatch):
    root = tmp_path / "share"
    campaign, control = root / "terrain-confirmation-v3", root / "dispatch"
    campaign.mkdir(parents=True)
    control.mkdir()
    monkeypatch.setattr(module, "SHARED_ROOT", root)
    monkeypatch.setattr(module, "CAMPAIGN", campaign)
    monkeypatch.setattr(module, "__file__", str(control / "terrain_confirmation_dispatch.py"))
    sources = {}
    for name in (*module.HELPERS, "terrain_confirmation_dispatch.py"):
        (control / name).write_text(name)
        sources[name] = module.digest(control / name)
    jobs, expected_jobs = [], []
    for actor in ("dr-default-s0", "dr-default-s1", "dr-default-s2"):
        for terrain in ("flat", "small_steps", "ramp"):
            name = actor + "-" + terrain
            jobs.append(dict(name=name, kind="run", args=["--actor", actor, "--terrain", terrain]))
            expected_jobs.append(dict(name=name, expected_fields=dict(phase="confirmation", episodes=40 if actor == "dr-default-s1" else 60,
                actor_filter=actor, terrain_filter=terrain)))
    (campaign / "protocol.json").write_text(json.dumps(dict(jobs=jobs)))
    (campaign / "frozen-campaign.tar.gz").write_bytes(b"test-only-archive")
    intake = dict(protocol_sha256=module.digest(campaign / "protocol.json"), archive_sha256=module.digest(campaign / "frozen-campaign.tar.gz"))
    (campaign / "intake.json").write_text(json.dumps(intake))
    value = dict(schema="bhl-terrain-confirmation-dispatch-v1", campaign_dir=str(campaign), max_concurrency=3,
        reserved_bytes_per_job=1024**3, minimum_free_bytes=8*1024**3, poll_seconds=30, source_sha256=sources,
        jobs=expected_jobs, smoke=dict(expected_fields=dict(phase="smoke", episodes=24, gate_to_confirmation=True)), **intake)
    path = control / "plan.json"
    path.write_text(json.dumps(value))
    return path, value


def test_complete_plan_accepts_three_bounded_shards(plan):
    path, value = plan
    assert module.validate_plan(path, module.digest(path)) == value


@pytest.mark.parametrize("field,value", [("max_concurrency", 4), ("minimum_free_bytes", 0),
    ("reserved_bytes_per_job", 128*1024**2), ("campaign_dir", "/nfs/stak/users/sanchej7/old")])
def test_home_escape_or_relaxed_reserve_rejected(plan, field, value):
    path, data = plan
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.validate_plan(path, module.digest(path))


def test_partial_cohort_cannot_dispatch(plan):
    path, data = plan
    data["jobs"].pop()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="nine"):
        module.validate_plan(path, module.digest(path))


def test_partial_or_negative_smoke_cannot_dispatch(plan):
    path, data = plan
    data["smoke"]["expected_fields"]["gate_to_confirmation"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="smoke"):
        module.validate_plan(path, module.digest(path))


def test_changed_source_or_plan_hash_rejected(plan):
    path, _ = plan
    with pytest.raises(ValueError, match="pinned"):
        module.validate_plan(path, "0"*64)
    (path.parent / "methods_archive.py").write_text("changed")
    with pytest.raises(ValueError, match="source"):
        module.validate_plan(path, module.digest(path))
