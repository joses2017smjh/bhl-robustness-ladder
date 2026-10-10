"""A pinned collector bootstrap must reject malformed cohorts and source packs."""
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest


SOURCE = Path(__file__).parents[1]/"scripts/bench/methods_finalize_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("methods_finalize_bootstrap_tested", SOURCE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
TRACKS = ("terrain", "stereo", "sensors", "perceptive")


def frozen_plan(tmp_path, *, member="scripts/bench/methods_finalize.py", link=False):
    archive = tmp_path/"source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo(member)
        payload = b"# fixture: no actual collection or publication\n"
        if link:
            info.type = tarfile.SYMTYPE
            info.linkname = "/tmp/outside"
            stream.addfile(info)
        else:
            info.size = len(payload)
            stream.addfile(info, io.BytesIO(payload))
    rows = []
    for track in TRACKS:
        child = tmp_path/(track+".json")
        child.write_text(json.dumps({"schema": "bhl-methods-finalization-v1", "track": track}))
        rows.append({"track": track, "plan_path": str(child), "plan_sha256": M.digest(child)})
    plan = {"schema": "bhl-methods-finalization-bootstrap-v1", "bootstrap_sha256": M.digest(SOURCE),
            "source_archive": str(archive), "source_archive_sha256": M.digest(archive),
            "python": "unused-fixture-python", "collections": rows}
    path = tmp_path/"plan.json"
    path.write_text(json.dumps(plan))
    return path, plan


def forbid_subprocess(monkeypatch):
    monkeypatch.setattr(M.subprocess, "run", lambda *args, **kwargs: pytest.fail("invalid input reached collection"))


@pytest.mark.parametrize("member,link", [("../escaped.py", False), ("/tmp/escaped.py", False),
                                         ("scripts\\escaped.py", False), ("symlink", True)])
def test_archive_traversal_or_link_never_reaches_collection(tmp_path, monkeypatch, member, link):
    path, _ = frozen_plan(tmp_path, member=member, link=link)
    forbid_subprocess(monkeypatch)
    with pytest.raises(ValueError, match="unsafe"):
        M.run(path, M.digest(path))


def test_source_archive_tamper_is_rejected(tmp_path, monkeypatch):
    path, plan = frozen_plan(tmp_path)
    with Path(plan["source_archive"]).open("ab") as stream:
        stream.write(b"altered")
    forbid_subprocess(monkeypatch)
    with pytest.raises(ValueError, match="archive changed"):
        M.run(path, M.digest(path))


@pytest.mark.parametrize("tracks", [(), ("terrain",)*4, ("terrain", "stereo", "sensors", "../escaped")])
def test_missing_duplicate_or_unknown_track_cannot_report_pass(tmp_path, monkeypatch, tracks):
    path, plan = frozen_plan(tmp_path)
    rows = {row["track"]: row for row in plan["collections"]}
    plan["collections"] = [dict(rows.get(track, rows["terrain"]), track=track) for track in tracks]
    path.write_text(json.dumps(plan))
    forbid_subprocess(monkeypatch)
    with pytest.raises(ValueError):
        M.run(path, M.digest(path))


def test_child_plan_track_must_match_its_label(tmp_path, monkeypatch):
    path, plan = frozen_plan(tmp_path)
    row = plan["collections"][0]
    child = Path(row["plan_path"])
    child.write_text(json.dumps({"schema": "bhl-methods-finalization-v1", "track": "stereo"}))
    row["plan_sha256"] = M.digest(child)
    path.write_text(json.dumps(plan))
    forbid_subprocess(monkeypatch)
    with pytest.raises(ValueError):
        M.run(path, M.digest(path))


def test_negative_return_preserves_other_tracks_and_reports_incomplete(tmp_path, monkeypatch):
    path, _ = frozen_plan(tmp_path)
    observed = []
    def collect(command, **kwargs):
        local_plan = Path(command[command.index("--plan")+1])
        track = json.loads(local_plan.read_text())["track"]
        observed.append(track)
        assert local_plan.parent.name.startswith("bhl-finalize-source-")
        assert Path(command[1]).is_file()
        return SimpleNamespace(returncode=1 if track == "terrain" else 0)
    monkeypatch.setattr(M.subprocess, "run", collect)
    assert M.run(path, M.digest(path)) == 1
    receipt = json.loads((tmp_path/"bootstrap-result.json").read_text())
    assert observed == list(TRACKS)
    assert receipt["status"] == "INCOMPLETE"
    assert [row["returncode"] for row in receipt["collections"]] == [1, 0, 0, 0]
