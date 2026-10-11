"""Pinned observer overlays and CPU finalization failure-preservation controls."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

REPO = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/bench" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finalizer, launch = load("h34_finalize"), load("h34_campaign")


@pytest.fixture
def frozen(tmp_path):
    repo = tmp_path / "training-source"
    (repo / "scripts/bench").mkdir(parents=True)
    (repo / "slurm/repo20261008").mkdir(parents=True)
    shutil.copyfile(REPO / "scripts/bench/h34_campaign.py", repo / "scripts/bench/h34_campaign.py")
    (repo / "scripts/runner.py").write_text("# unchanged frozen training source\n")
    (repo / "slurm/repo20261008/h34_campaign.sbatch").write_text("#!/bin/bash\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "scripts", "slurm"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"], check=True)
    protocol = {"schema_version": 1, "campaign_id": "h3-fixture", "task": "H3",
                "runtime": {"stack": "cpu", "python": sys.executable, "share_root": "/nfs/hpc/share/sanchej7"},
                "jobs": [{"name": "smoke", "kind": "smoke", "entrypoint": "scripts/runner.py", "args": [],
                          "resources": {"partition": "share", "cpus": 1, "memory_gb": 2, "gpus": 0, "time_limit": "00:05:00"}}]}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    persistent = tmp_path / "frozen"
    launch.freeze(repo, protocol_path, persistent)
    observer_repo = tmp_path / "observer-source"
    for name in finalizer.OBSERVER_FILES:
        target = observer_repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# observer fixture\n")
    observer_out = tmp_path / "observer"
    receipt = finalizer.freeze_observer(observer_repo, observer_out)
    return persistent, observer_repo, observer_out, receipt


def test_new_observer_overlay_preserves_frozen_campaign_bytes(frozen, tmp_path):
    persistent, _, observer, receipt = frozen
    before = {p.name: finalizer.digest(p) for p in persistent.iterdir() if p.is_file()}
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    campaign, _, provenance = finalizer.stage(persistent, observer / "observer.tar.gz", receipt["archive_sha256"], scratch)
    assert provenance["original_source_files_verified"] == 3
    assert (campaign / "source/scripts/runner.py").read_text() == "# unchanged frozen training source\n"
    assert all((campaign / "source" / name).is_file() for name in finalizer.OBSERVER_FILES)
    assert before == {p.name: finalizer.digest(p) for p in persistent.iterdir() if p.is_file()}


def test_import_does_not_mutate_pinned_source_and_restores_caller_setting(tmp_path, monkeypatch):
    helper = tmp_path / "helper.py"
    helper.write_text("answer = 42\n")
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    module = finalizer.load_module(helper, "fixture_pinned_helper")
    assert module.answer == 42
    assert sys.dont_write_bytecode is False
    assert sorted(p.name for p in tmp_path.iterdir()) == ["helper.py"]
    helper.write_text("raise RuntimeError('failed import')\n")
    with pytest.raises(RuntimeError, match="failed import"):
        finalizer.load_module(helper, "fixture_failed_helper")
    assert sys.dont_write_bytecode is False
    assert not (tmp_path / "__pycache__").exists()


@pytest.mark.parametrize("change", ["observer_hash", "protocol", "campaign_archive", "observer_contents"])
def test_changed_pins_refused(frozen, tmp_path, change):
    persistent, _, observer, receipt = frozen
    expected = receipt["archive_sha256"]
    if change == "observer_hash":
        expected = "bad"
    elif change == "protocol":
        (persistent / "protocol.json").write_text("{}")
    elif change == "campaign_archive":
        (persistent / "frozen-campaign.tar.gz").write_bytes(b"changed")
    else:
        extracted = tmp_path / "repack"
        extracted.mkdir()
        finalizer.read_archive(observer / "observer.tar.gz", extracted)
        (extracted / "scripts/bench/h34_collect.py").write_text("changed observer bytes")
        with tarfile.open(observer / "observer.tar.gz", "w:gz") as tar:
            for path in extracted.rglob("*"):
                if path.is_file():
                    tar.add(path, arcname=str(path.relative_to(extracted)))
        expected = finalizer.digest(observer / "observer.tar.gz")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ValueError):
        finalizer.stage(persistent, observer / "observer.tar.gz", expected, scratch)


def test_observer_cannot_replace_training_code(frozen, tmp_path):
    persistent, _, observer, receipt = frozen
    extracted = tmp_path / "repack"
    extracted.mkdir()
    finalizer.read_archive(observer / "observer.tar.gz", extracted)
    malicious = extracted / "scripts/runner.py"
    malicious.write_text("replaced training code")
    with tarfile.open(observer / "observer.tar.gz", "w:gz") as tar:
        for file in extracted.rglob("*"):
            if file.is_file():
                tar.add(file, arcname=str(file.relative_to(extracted)))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ValueError, match="inventory"):
        finalizer.stage(persistent, observer / "observer.tar.gz", finalizer.digest(observer / "observer.tar.gz"), scratch)


def test_container_never_requests_gpu_and_preserves_readonly_inputs(tmp_path):
    protocol = {"runtime": {"sif": "/shared/image.sif", "python": "/shared/python", "share_root": "/shared"}}
    command, env = finalizer.container_command(protocol, "/frozen", "/new-output", tmp_path, tmp_path / "campaign", "hash")
    assert "--nv" not in command
    assert "/frozen:/frozen:ro" in command
    assert "/new-output:/new-output:rw" in command
    assert f"{tmp_path}/campaign:{tmp_path}/campaign:ro" in command
    assert env["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == ""
    assert env["APPTAINERENV_OMP_NUM_THREADS"] == "2"
    assert "APPTAINERENV_HOME" not in env


@pytest.mark.parametrize("status,expected", [("PASS", 0), ("NEGATIVE", 0), ("INCOMPLETE", 1)])
def test_worker_preserves_recomputed_status(tmp_path, monkeypatch, status, expected):
    source = tmp_path / "source"
    (source / "scripts/bench").mkdir(parents=True)
    # Fixture collector has no simulated or fabricated robot-performance claims.
    (source / "scripts/bench/h34_collect.py").write_text(
        "import json\ndef collect(persistent,out):\n"
        " out.mkdir(); report={'status':" + repr(status) + "}; (out/'collection.json').write_text(json.dumps(report)); (out/'scientific-summary.json').write_text(json.dumps(report)); return report\n")
    out = tmp_path / "out"
    out.mkdir()
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    (persistent / "protocol.json").write_text(json.dumps({"runtime": {"share_root": "/fixture/shared"}}))
    monkeypatch.setattr(finalizer.os, "statvfs", lambda path: SimpleNamespace(f_flag=finalizer.os.ST_RDONLY))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert finalizer.worker(persistent, out, source, "observer-pin") == expected
    assert json.loads((out / "finalization.json").read_text())["status"] == status


def test_finalization_failure_is_durable_and_cannot_mutate_campaign(frozen, tmp_path):
    persistent, _, observer, receipt = frozen
    target = tmp_path / "failed-finalization"
    assert finalizer.finalize(persistent, observer / "observer.tar.gz", "wrong pin", target) == 1
    result = json.loads((target / "finalization.json").read_text())
    completion = json.loads((target / "completion.json").read_text())
    assert result["status"] == completion["status"] == "INCOMPLETE"
    assert "observer archive" in " ".join(result["problems"])
    with pytest.raises(ValueError, match="separate"):
        finalizer.finalize(persistent, observer / "observer.tar.gz", receipt["archive_sha256"], persistent / "new")


def test_worker_refuses_writable_campaign_mounts(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "scripts/bench").mkdir(parents=True)
    (source / "scripts/bench/h34_collect.py").write_text("def collect(*args):\n raise AssertionError('collector must not run')\n")
    persistent = tmp_path / "persistent"
    persistent.mkdir()
    (persistent / "protocol.json").write_text(json.dumps({"runtime": {"share_root": str(tmp_path)}}))
    output = tmp_path / "out"
    output.mkdir()
    monkeypatch.setattr(finalizer.os, "statvfs", lambda path: SimpleNamespace(f_flag=0))
    assert finalizer.worker(persistent, output, source, "pin") == 1
    result = json.loads((output / "finalization.json").read_text())
    assert result["status"] == "INCOMPLETE" and "read-only" in result["problems"][0]
    assert not (output / "collection").exists()


def test_existing_output_is_never_overwritten(frozen, tmp_path):
    persistent, _, observer, receipt = frozen
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep.txt").write_text("original")
    with pytest.raises(FileExistsError):
        finalizer.finalize(persistent, observer / "observer.tar.gz", receipt["archive_sha256"], target)
    assert (target / "keep.txt").read_text() == "original"
