"""Adversarial checks for durable result collection and publication boundaries."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tarfile

import pytest

SPEC = importlib.util.spec_from_file_location("methods_finalize_tested", Path(__file__).parents[1]/"scripts/bench/methods_finalize.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_changed_finalization_plan_rejected_before_read(tmp_path):
    plan = tmp_path/"plan.json"
    plan.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        M.validate_plan(plan, "0"*64)


@pytest.mark.parametrize("replacement", [
    {"asset_name": "../other.tar.gz"}, {"source_archive_sha256": "changed"},
    {"sha256": "different"}, {"bytes": 6},
])
def test_publication_mismatch_never_downloads(tmp_path, monkeypatch, replacement):
    job = tmp_path/"job"
    job.mkdir()
    raw = {"bytes": 5, "sha256": "expected"}
    (job/"completion.json").write_text(json.dumps({"files": {"outputs.tar.gz": raw}}))
    publication = {"asset_name": "original.tar.gz", "source_archive_sha256": "source", **raw, **replacement}
    (job/"publication.json").write_text(json.dumps(publication))
    def forbidden(*args):
        pytest.fail("malformed provenance must not reach GitHub")
    monkeypatch.setattr(M.ARCHIVE, "gh", forbidden)
    with pytest.raises(ValueError):
        M.obtain_raw(job, "source", tmp_path)


def test_download_checksum_rechecked(tmp_path, monkeypatch):
    job = tmp_path/"job"
    job.mkdir()
    raw = {"bytes": 5, "sha256": "0"*64}
    (job/"completion.json").write_text(json.dumps({"files": {"outputs.tar.gz": raw}}))
    (job/"publication.json").write_text(json.dumps({"asset_name": "original.tar.gz", "source_archive_sha256": "source", **raw}))
    (tmp_path/"original.tar.gz").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checksum"):
        M.obtain_raw(job, "source", tmp_path)


def test_unsubmitted_job_stays_incomplete_and_never_collects(tmp_path, monkeypatch):
    (tmp_path/"intake.json").write_text("{}")
    plan = {"track": "terrain", "archive_sha256": "source", "result_dir": str(tmp_path/"result")}
    monkeypatch.setattr(M, "validate_plan", lambda *args: (plan, tmp_path, [{"name": "run-0"}]))
    monkeypatch.setattr(M, "verified_campaign_view", lambda campaign, *args: campaign)
    monkeypatch.setattr(M.DISPATCH, "submitted_id", lambda *args: None)
    monkeypatch.setattr(M, "collect_track", lambda *args: pytest.fail("missing cohort cannot be scored"))
    monkeypatch.setattr(M, "publish_summary", lambda *args: {"url": "test"})
    assert M.run(tmp_path/"plan.json", "pin") == 1
    result = json.loads((tmp_path/"result/collection.json").read_text())
    assert result["status"] == "INCOMPLETE"
    assert result["observed_jobs"] == []
    assert "never submitted" in result["problems"][0]["error"]


def test_summary_publication_refuses_replacement(tmp_path, monkeypatch):
    (tmp_path/"collection.json").write_text("{}")
    calls = []
    def gh(*args):
        calls.append(args)
        name = "methods-final-terrain-20261010-"+M.SAFE.digest(tmp_path/"collection.json")[:12]+".tar.gz"
        return json.dumps({"assets": [{"name": name}]}).encode()
    monkeypatch.setattr(M.ARCHIVE, "gh", gh)
    with pytest.raises(ValueError, match="already exists"):
        M.publish_summary(tmp_path, "terrain")
    assert len(calls) == 1 and calls[0][0] == "api"


@pytest.mark.parametrize("field,value", [("job", {"name": "other"}), ("source_files_verified", 99), ("job_id", "9")])
def test_exact_declared_launch_required(tmp_path, field, value):
    launch = {"job": {"name": "declared"}, "source_files_verified": 100, "job_id": "8"}
    launch[field] = value
    path = tmp_path/"launch.json"
    path.write_text(json.dumps(launch))
    (tmp_path/"completion.json").write_text(json.dumps({"files": {"launch.json": {"bytes": path.stat().st_size, "sha256": M.SAFE.digest(path)}}}))
    with pytest.raises(ValueError, match="exact frozen job"):
        M.verify_launch(tmp_path, {"name": "declared"}, {"source_files": 100}, "8")


def source_fixture(tmp_path):
    campaign, remote, work = (tmp_path/name for name in ("campaign", "release", "work"))
    for path in (campaign, remote, work):
        path.mkdir()
    code = b"original scientific source\n"
    import hashlib
    manifest = {"source/example.py": {"bytes": len(code), "sha256": hashlib.sha256(code).hexdigest()}}
    (campaign/"protocol.json").write_text(json.dumps({"jobs": []}))
    (campaign/"manifest.json").write_text(json.dumps(manifest))
    (campaign/"h34_campaign.sbatch").write_text("# original launcher\n")
    archive = remote/"original.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, payload in (("protocol.json", (campaign/"protocol.json").read_bytes()),
                              ("manifest.json", (campaign/"manifest.json").read_bytes()), ("source/example.py", code)):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))
    intake = dict(archive_sha256=M.SAFE.digest(archive), archive_bytes=archive.stat().st_size,
                  protocol_sha256=M.SAFE.digest(campaign/"protocol.json"),
                  launcher_sha256=M.SAFE.digest(campaign/"h34_campaign.sbatch"),
                  source_files=1, uncompressed_input_bytes=len(code))
    (campaign/"intake.json").write_text(json.dumps(intake))
    return campaign, archive, work, intake


def test_absent_source_restored_to_node_view_and_original_metadata_preserved(tmp_path, monkeypatch):
    campaign, archive, work, intake = source_fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in campaign.iterdir()}
    def download(*args):
        assert args[args.index("--pattern")+1] == "methods-source-"+intake["archive_sha256"]+".tar.gz"
        shutil.copyfile(archive, Path(args[args.index("--dir")+1])/args[args.index("--pattern")+1])
    monkeypatch.setattr(M.ARCHIVE, "gh", download)
    view = M.verified_campaign_view(campaign, intake, work)
    assert view.parent == work and view != campaign
    assert M.SAFE.digest(view/"frozen-campaign.tar.gz") == intake["archive_sha256"]
    assert before == {p.name: p.read_bytes() for p in campaign.iterdir()}


@pytest.mark.parametrize("existing", ("corrupt", "dangling_link"))
def test_existing_bad_source_never_falls_back_to_release(tmp_path, monkeypatch, existing):
    campaign, _, work, intake = source_fixture(tmp_path)
    target = campaign/"frozen-campaign.tar.gz"
    if existing == "corrupt":
        target.write_bytes(b"wrong source")
    else:
        target.symlink_to(tmp_path/"missing")
    monkeypatch.setattr(M.ARCHIVE, "gh", lambda *args: pytest.fail("existing source cannot trigger fallback"))
    with pytest.raises(ValueError):
        M.verified_campaign_view(campaign, intake, work)


def test_downloaded_source_bytes_must_match_original_intake(tmp_path, monkeypatch):
    campaign, _, work, intake = source_fixture(tmp_path)
    def wrong_download(*args):
        (Path(args[args.index("--dir")+1])/args[args.index("--pattern")+1]).write_bytes(b"wrong source")
    monkeypatch.setattr(M.ARCHIVE, "gh", wrong_download)
    with pytest.raises(ValueError, match="checksum"):
        M.verified_campaign_view(campaign, intake, work)
    assert not (campaign/"frozen-campaign.tar.gz").exists()


def test_collectors_use_verified_view_while_jobs_keep_original_identity(tmp_path, monkeypatch):
    campaign = tmp_path/"campaign"
    campaign.mkdir()
    job = campaign/"run-0-12"
    job.mkdir()
    for name in ("intake.json",):
        (campaign/name).write_text("{}")
    for name in ("campaign_result.json", "completion.json", "launch.json", "publication.json"):
        (job/name).write_text("{}")
    view = tmp_path/"verified-view"
    plan = {"track": "perceptive", "archive_sha256": "source", "result_dir": str(tmp_path/"result")}
    monkeypatch.setattr(M, "validate_plan", lambda *args: (plan, campaign, [{"name": "run-0"}]))
    monkeypatch.setattr(M, "verified_campaign_view", lambda *args: view)
    monkeypatch.setattr(M.DISPATCH, "submitted_id", lambda *args: "12")
    monkeypatch.setattr(M.DISPATCH, "scheduler_state", lambda *args: ("COMPLETED", "0:0"))
    monkeypatch.setattr(M.DISPATCH, "checked_result", lambda *args: ({"phase": "run"},
        {"files": {"campaign_result.json": {"sha256": "measured"}}}))
    monkeypatch.setattr(M, "verify_launch", lambda *args: None)
    monkeypatch.setattr(M, "obtain_raw", lambda *args: None)
    def collect(track, source, jobs, *args):
        assert source == view and jobs == [job]
        return {"status": "PASS"}
    monkeypatch.setattr(M, "collect_track", collect)
    monkeypatch.setattr(M, "publish_summary", lambda *args: {"url": "fixture"})
    assert M.run(tmp_path/"plan.json", "pin") == 0
    receipt = json.loads((tmp_path/"result/collection.json").read_text())
    assert receipt["source_archive_origin"] == "SHA_VERIFIED_RELEASE_RESTORED_IN_NODE_TMP"
