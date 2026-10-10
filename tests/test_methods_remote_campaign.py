"""Remote execution must preserve source pins, explicit gates and raw evidence."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).parents[1]/"scripts/bench/methods_remote_campaign.py"
SPEC = importlib.util.spec_from_file_location("methods_remote_campaign_tested", SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def frozen(tmp_path, monkeypatch, *, kind="smoke"):
    monkeypatch.setattr(M, "AUTHORIZED_ROOT", tmp_path/"durable")
    campaign = tmp_path/"original"
    campaign.mkdir()
    protocol = {"jobs": [{"name": "declared-job", "kind": kind, "resources": {"gpus": 1}}]}
    (campaign/"protocol.json").write_text(json.dumps(protocol))
    (campaign/"h34_campaign.sbatch").write_text("# original frozen launcher\n")
    manifest = {"source/scripts/bench/h34_campaign.py": {"sha256": "code", "bytes": 4},
                "inputs/actor.onnx": {"sha256": "actor", "bytes": 1}}
    archive = campaign/"frozen-campaign.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, value in (("manifest.json", json.dumps(manifest).encode()),
                            ("source/scripts/bench/h34_campaign.py", b"pass")):
            member = tarfile.TarInfo(name)
            member.size = len(value)
            stream.addfile(member, io.BytesIO(value))
    intake = {"archive_sha256": M.digest(archive), "archive_bytes": archive.stat().st_size,
              "protocol_sha256": M.digest(campaign/"protocol.json"),
              "launcher_sha256": M.digest(campaign/"h34_campaign.sbatch")}
    (campaign/"intake.json").write_text(json.dumps(intake))
    def publication(path, name, expected):
        M.checked_file(path, expected)
        return {"asset_name": name, **expected, "url": "https://example.invalid/fixture"}
    monkeypatch.setattr(M, "publish_asset", publication)
    return campaign, M.AUTHORIZED_ROOT/"new-job"


def prepared(tmp_path, monkeypatch):
    campaign, durable = frozen(tmp_path, monkeypatch)
    result = M.prepare(campaign, durable, "declared-job")
    return campaign, durable, Path(result["plan"]), result["plan_sha256"]


def test_changed_source_never_reaches_publication(tmp_path, monkeypatch):
    campaign, durable = frozen(tmp_path, monkeypatch)
    with (campaign/"frozen-campaign.tar.gz").open("ab") as stream:
        stream.write(b"changed")
    monkeypatch.setattr(M, "publish_asset", lambda *a: pytest.fail("changed source cannot be published"))
    with pytest.raises(ValueError, match="pinned file changed"):
        M.prepare(campaign, durable, "declared-job")
    assert not durable.exists()


def test_development_requires_explicit_matching_scientific_smoke(tmp_path, monkeypatch):
    campaign, durable = frozen(tmp_path, monkeypatch, kind="run")
    monkeypatch.setattr(M, "publish_asset", lambda *a: pytest.fail("unqualified development cannot be published"))
    with pytest.raises(ValueError, match="qualified smoke"):
        M.prepare(campaign, durable, "declared-job")


def test_prepare_retains_metadata_but_never_large_source_archive(tmp_path, monkeypatch):
    _, durable, path, pin = prepared(tmp_path, monkeypatch)
    _, plan = M.load_plan(path, pin)
    assert not (durable/"frozen-campaign.tar.gz").exists()
    assert plan["job"]["kind"] == "smoke"
    assert sum(p.stat().st_size for p in durable.iterdir()) < M.COMPACT_CAP


@pytest.mark.parametrize("target", ("plan", "protocol", "bootstrap"))
def test_changed_plan_metadata_or_bootstrap_rejected(tmp_path, monkeypatch, target):
    _, durable, path, pin = prepared(tmp_path, monkeypatch)
    if target == "plan":
        path.write_text("{}")
    elif target == "protocol":
        (durable/"protocol.json").write_text("{}")
    else:
        plan = json.loads(path.read_text())
        plan["bootstrap_sha256"] = "0"*64
        path.write_text(json.dumps(plan))
        pin = M.digest(path)
    with pytest.raises(ValueError):
        M.load_plan(path, pin)


def test_unallocated_execution_never_downloads(tmp_path, monkeypatch):
    _, _, path, pin = prepared(tmp_path, monkeypatch)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(M, "gh", lambda *a: pytest.fail("unallocated execution reached GitHub"))
    with pytest.raises(ValueError, match="Slurm"):
        M.run(path, pin)


def test_failed_readiness_is_negative_not_runtime_crash_and_blocks_rerun(tmp_path, monkeypatch):
    campaign, durable, path, pin = prepared(tmp_path, monkeypatch)
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    def download(*args):
        destination = Path(args[args.index("--dir")+1])/args[args.index("--pattern")+1]
        shutil.copyfile(campaign/"frozen-campaign.tar.gz", destination)
    monkeypatch.setattr(M, "gh", download)
    def execute(local, name, source):
        job = local/(name+"-123456")
        job.mkdir()
        (job/"launch.json").write_text(json.dumps({"archive_sha256": source, "job": {"kind": "smoke"}}))
        (job/"runtime.log").write_text("real runtime completed; scientific readiness failed\n")
        (job/"outputs.tar.gz").write_bytes(b"raw measured negative evidence")
        (job/"campaign_result.json").write_text(json.dumps({"status": "NEGATIVE"}))
        completion = {"status": "NEGATIVE", "exit_status": 0, "error": None, "archive_sha256": source,
            "files": {p.name: {"sha256": M.digest(p), "bytes": p.stat().st_size} for p in job.iterdir()}}
        (job/"completion.json").write_text(json.dumps(completion))
        return 1
    monkeypatch.setattr(M, "extract_runner", lambda *a: SimpleNamespace(execute=execute))
    assert M.run(path, pin) == 1
    job = durable/"declared-job-123456"
    result = json.loads((job/"remote-result.json").read_text())
    assert result["status"] == "NEGATIVE" and result["runtime_exit_status"] == 0
    assert result["outcome"] == "FAILED_SCIENTIFIC_READINESS_GATE"
    assert result["follow_on"] == "UNRUN_AFTER_NEGATIVE_SMOKE"
    assert (job/"publication.json").exists() and not (job/"outputs.tar.gz").exists()
    with pytest.raises(ValueError, match="readiness PASS"):
        M.gate_receipt(job, json.loads(path.read_text())["scientific_identity"])
    with pytest.raises(FileExistsError):
        M.run(path, pin)


def test_failed_publication_retains_original_local_science(tmp_path):
    retained = None
    with pytest.raises(RuntimeError, match="publication failed"):
        with M.retained_workspace() as work:
            retained = work
            (work/"raw-measurements.txt").write_text("must remain inspectable")
            raise RuntimeError("publication failed")
    assert (retained/"raw-measurements.txt").read_text() == "must remain inspectable"
    shutil.rmtree(retained)


def test_successful_gate_is_bound_to_original_runtime_and_input_identity(tmp_path):
    identity = {"files": {"inputs/actor.onnx": {"sha256": "actor", "bytes": 1}},
                "runtime": {"python": "/frozen/python"}, "sif_sha256": "container"}
    (tmp_path/"launch.json").write_text(json.dumps({"job": {"kind": "smoke"}, "archive_sha256": "source"}))
    (tmp_path/"campaign_result.json").write_text(json.dumps({"status": "PASS"}))
    completion = {"status": "PASS", "exit_status": 0, "error": None, "finished_utc": "2026-10-10T00:00:00Z",
        "archive_sha256": "source", "files": {p.name: {"sha256": M.digest(p), "bytes": p.stat().st_size}
                                               for p in tmp_path.iterdir()}}
    (tmp_path/"completion.json").write_text(json.dumps(completion))
    (tmp_path/"remote-result.json").write_text(json.dumps({"status": "PASS", "runner_returncode": 0,
        "runtime_exit_status": 0, "publication": {"sha256": "verified-raw"}, "source_archive_sha256": "source",
        "scientific_identity_sha256": M.identity_sha256(identity)}))
    assert len(M.gate_receipt(tmp_path, identity)) == 4
    changed = dict(identity, sif_sha256="different-container")
    with pytest.raises(ValueError, match="readiness PASS"):
        M.gate_receipt(tmp_path, changed)
