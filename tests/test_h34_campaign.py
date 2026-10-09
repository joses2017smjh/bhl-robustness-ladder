"""Launch provenance, isolation, and failed-result controls for H3/H4."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


spec = importlib.util.spec_from_file_location("h34_campaign", Path(__file__).parents[1] / "scripts/bench/h34_campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def protocol():
    return {"schema_version": 1, "campaign_id": "h3-test", "runtime": {"stack": "cpu", "python": sys.executable, "share_root": "/nfs/hpc/share/sanchej7"},
            "jobs": [{"name": "smoke", "kind": "smoke", "entrypoint": "scripts/runner.py", "args": ["{protocol}", "{cell}"],
                      "resources": {"partition": "share", "cpus": 1, "memory_gb": 2, "gpus": 0, "time_limit": "00:05:00"}}]}


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a//b", "", "./file", 123])
def test_paths_cannot_escape_or_alias(name):
    with pytest.raises(ValueError):
        campaign.relative_name(name)


def test_runtime_isolation_and_array_forwarding(tmp_path, monkeypatch):
    p = protocol()
    p["runtime"]["sif"] = "/readonly/image.sif"
    p["jobs"][0]["resources"]["gpus"] = 1
    p["jobs"][0]["env"] = {"CUSTOM": "a,b", "SETTING": "{output}/metric.json"}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-abc")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    cmd, env = campaign.runtime_command(p, p["jobs"][0], tmp_path)
    assert "--containall" in cmd and "--cleanenv" in cmd and "--nv" in cmd
    assert "/nfs/hpc/share/sanchej7:/nfs/hpc/share/sanchej7:ro" in cmd
    assert cmd[-1] == "3"
    assert env["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == "MIG-abc"
    assert env["APPTAINERENV_CUSTOM"] == "a,b"  # no comma-splitting --env flags
    assert env["APPTAINERENV_H34_KIT_ARGS"] == f"--portable-root {tmp_path}/kit"
    assert env["APPTAINERENV_PYTHONPATH"].split(":")[0] == f"{tmp_path}/source/src"
    assert "APPTAINERENV_HOME" not in env


@pytest.mark.parametrize("override", [{"gpus": 2}, {"cpus": 0}, {"memory_gb": 129}, {"time_limit": "49:00:00"}, {"array": "0-5%3"}, {"requeue": "false"}, {"requeue": 0}])
def test_unbounded_resources_refused(override):
    p = protocol()
    p["jobs"][0]["resources"].update(override)
    with pytest.raises(ValueError):
        campaign.validate_protocol(p)


def test_v100_and_gpu_array_overbooking_refused():
    p = protocol()
    p["runtime"]["sif"] = "/image.sif"
    p["jobs"][0]["resources"].update(gpus=1, partition="dgx2")
    with pytest.raises(ValueError, match="V100"):
        campaign.validate_protocol(p)
    p["jobs"][0]["resources"].update(partition="gpu", array="0-5%2")
    with pytest.raises(ValueError, match="concurrency"):
        campaign.validate_protocol(p)


@pytest.mark.parametrize("env", [{"HOME": "/share"}, {"PYTHONPATH": "/dirty"}, {"CUDA_VISIBLE_DEVICES": "0"}, {"APPTAINERENV_HOME": "/share"}, {"SLURM_JOB_ID": "1"}, {"H34_OUTPUT_DIR": "/share"}, {"XDG_CACHE_HOME": "/share"}])
def test_job_cannot_disable_runtime_isolation(env):
    p = protocol()
    p["jobs"][0]["env"] = env
    with pytest.raises(ValueError):
        campaign.validate_protocol(p)


@pytest.mark.parametrize("kind", ["symlink", "parent", "duplicate"])
def test_hostile_archive_refused_before_extract(tmp_path, kind):
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("../escape" if kind == "parent" else "source/a")
        if kind == "symlink":
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        else:
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
            if kind == "duplicate":
                tar.addfile(info, io.BytesIO(b"y"))
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError):
        campaign.safe_extract(path, out)
    assert not list(out.rglob("*"))


def frozen(tmp_path, status="PASS"):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "slurm/repo20261008").mkdir(parents=True)
    (repo / "scripts/runner.py").write_text(
        "import json,os,pathlib\n"
        f"pathlib.Path(os.environ['H34_OUTPUT_DIR'],'campaign_result.json').write_text(json.dumps({{'status':{status!r}}}))\n")
    (repo / "slurm/repo20261008/h34_campaign.sbatch").write_text("#!/bin/bash\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "scripts", "slurm"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"], check=True)
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(protocol()))
    target = tmp_path / "persistent"
    intake = campaign.freeze(repo, p, target)
    return repo, target, intake


def test_frozen_source_and_launcher_mutation_detected(tmp_path):
    _, target, _ = frozen(tmp_path)
    out = tmp_path / "extract"
    out.mkdir()
    campaign.safe_extract(target / "frozen-campaign.tar.gz", out)
    assert campaign.verify_source(out) == 2
    (out / "source/scripts/runner.py").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        campaign.verify_source(out)
    (target / "h34_campaign.sbatch").write_text("changed")
    with pytest.raises(ValueError, match="frozen input changed"):
        campaign.load_intake(target)


def test_input_checkpoints_are_frozen_and_verified(tmp_path):
    repo, _, _ = frozen(tmp_path)
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"a real fixture checkpoint")
    p = protocol()
    p["input_files"] = {"teacher.pt": str(teacher)}
    config = tmp_path / "with-input.json"
    config.write_text(json.dumps(p))
    target = tmp_path / "second"
    campaign.freeze(repo, config, target)
    out = tmp_path / "extract"
    out.mkdir()
    campaign.safe_extract(target / "frozen-campaign.tar.gz", out)
    assert (out / "inputs/teacher.pt").read_bytes() == teacher.read_bytes()
    assert campaign.verify_source(out) == 3
    (out / "inputs/teacher.pt").write_bytes(b"bad")
    with pytest.raises(ValueError):
        campaign.verify_source(out)


def test_oversized_input_refused_before_durable_writes(tmp_path):
    repo, _, _ = frozen(tmp_path)
    teacher = tmp_path / "oversized.pt"
    with teacher.open("wb") as stream:
        stream.truncate(512 * 1024 ** 2 + 1)  # sparse fixture; no half-GiB write
    p = protocol()
    p["input_files"] = {"teacher.pt": str(teacher)}
    config = tmp_path / "oversized.json"
    config.write_text(json.dumps(p))
    target = tmp_path / "refused"
    with pytest.raises(ValueError, match="per-file cap"):
        campaign.freeze(repo, config, target)
    assert not target.exists()


@pytest.mark.parametrize("status,expected", [("PASS", 0), ("NEGATIVE", 1), ("INCOMPLETE", 1)])
def test_smoke_requires_scientific_pass_and_preserves_result(tmp_path, monkeypatch, status, expected):
    _, target, intake = frozen(tmp_path, status)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    assert campaign.execute(target, "smoke", intake["archive_sha256"]) == expected
    receipt = json.loads((target / "smoke-fixture/completion.json").read_text())
    assert receipt["status"] == status and receipt["exit_status"] == 0
    assert (target / "smoke-fixture/outputs.tar.gz").is_file()


def test_training_requires_same_archive_smoke_dependency(tmp_path):
    _, target, _ = frozen(tmp_path)
    p = json.loads((target / "protocol.json").read_text())
    p["jobs"][0]["kind"] = "run"
    (target / "protocol.json").write_text(json.dumps(p))
    intake = json.loads((target / "intake.json").read_text())
    intake["protocol_sha256"] = campaign.digest(target / "protocol.json")
    (target / "intake.json").write_text(json.dumps(intake))
    with pytest.raises(ValueError, match="requires"):
        campaign.submit(target, "smoke")
    with pytest.raises(ValueError, match="fresh smoke"):
        campaign.submit(target, "smoke", "afterok:123")


@pytest.mark.parametrize("requeue", [None, True, False])
def test_submission_does_not_inherit_interactive_context(tmp_path, monkeypatch, requeue):
    _, target, _ = frozen(tmp_path)
    if requeue is not None:
        p = json.loads((target / "protocol.json").read_text())
        p["jobs"][0]["resources"]["requeue"] = requeue
        (target / "protocol.json").write_text(json.dumps(p))
        intake = json.loads((target / "intake.json").read_text())
        intake["protocol_sha256"] = campaign.digest(target / "protocol.json")
        (target / "intake.json").write_text(json.dumps(intake))
    monkeypatch.setenv("SLURM_JOB_ID", "desktop")
    monkeypatch.setenv("TMPDIR", "/scratch/desktop-only")
    captured = {}
    def fake_run(command, **kwargs):
        captured.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, "12345\n", "")
    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    receipt = campaign.submit(target, "smoke")
    assert receipt["job_id"] == "12345"
    assert "SLURM_JOB_ID" not in captured["env"]
    assert captured["env"]["TMPDIR"] == "/tmp"
    assert "--export=NONE" in captured["command"]
    assert ("--no-requeue" in captured["command"]) == (requeue is False)
    with pytest.raises(FileExistsError):
        campaign.submit(target, "smoke")
