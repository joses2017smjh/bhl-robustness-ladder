"""Durable transport validation, preservation and full-array completeness."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest


spec = importlib.util.spec_from_file_location("h34_collect", Path(__file__).parents[1] / "scripts/bench/h34_collect.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def write_json(path, value, *, sorted_keys=True):
    path.write_text(json.dumps(value, indent=2, sort_keys=sorted_keys) + "\n")


def archive(path, files):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def fixture(tmp_path, *, task="H4", cells=None):
    root = tmp_path / "persistent"
    root.mkdir()
    resources = dict(partition="share", cpus=1, memory_gb=2, gpus=0, time_limit="00:05:00")
    if cells is None:
        cells = ([dict(arm="R1HO", seed=s) for s in range(3)] if task == "H4" else
                 [dict(arm=arm, seed=s) for arm in ("history", "repeated_current") for s in range(3)])
    jobs = [dict(name="fresh-smoke", kind="smoke", entrypoint="scripts/runner.py", resources=resources.copy()),
            dict(name="full", kind="run", entrypoint="scripts/runner.py", max_output_mb=1,
                 resources=dict(resources, array=f"0-{len(cells) - 1}%1"))]
    protocol = dict(schema_version=1, campaign_id="test-campaign", task=task, cells=cells, jobs=jobs,
                    runtime=dict(stack="cpu", python=sys.executable, share_root="/readonly"))
    write_json(root / "protocol.json", protocol)
    source = {"source/scripts/runner.py": b"# frozen runtime\n", "inputs/teacher.pt": b"frozen teacher"}
    manifest = {name: dict(bytes=len(data), sha256=collector.hashlib.sha256(data).hexdigest()) for name, data in source.items()}
    write_json(root / "manifest.json", manifest)
    archive(root / "frozen-campaign.tar.gz", dict(source, **{"protocol.json": (root / "protocol.json").read_bytes(),
                                                            "manifest.json": (root / "manifest.json").read_bytes()}))
    (root / "h34_campaign.sbatch").write_text("#!/bin/bash\n")
    intake = dict(archive_sha256=collector.digest(root / "frozen-campaign.tar.gz"),
                  archive_bytes=(root / "frozen-campaign.tar.gz").stat().st_size,
                  protocol_sha256=collector.digest(root / "protocol.json"),
                  launcher_sha256=collector.digest(root / "h34_campaign.sbatch"),
                  source_files=len(manifest), uncompressed_input_bytes=sum(map(len, source.values())))
    write_json(root / "intake.json", intake)
    smoke = dict(status="SUBMITTED", returncode=0, job_id="300", archive_sha256=intake["archive_sha256"],
                 resources=jobs[0]["resources"], command=["sbatch"])
    run = dict(status="SUBMITTED", returncode=0, job_id="310", archive_sha256=intake["archive_sha256"],
               resources=jobs[1]["resources"], command=["sbatch", "--dependency=afterok:300"])
    write_json(root / "fresh-smoke-submission.json", smoke)
    write_json(root / "full-submission.json", run)
    for i, cell in enumerate(cells):
        add_cell(root, i, cell, protocol, intake)
    return root, protocol, intake


def add_cell(root, index, cell, protocol, intake, *, job_id=None, status="PASS"):
    job_id = job_id or str(321 + index)
    directory = root / f"full-{job_id}"
    directory.mkdir()
    result = dict(status=status, task=protocol["task"], phase="run", cell=index,
                  arm=cell["arm"], seed=cell["seed"], protocol_sha256=intake["protocol_sha256"])
    write_json(directory / "campaign_result.json", result)
    launch = dict(job_id=job_id, archive_sha256=intake["archive_sha256"], source_files_verified=intake["source_files"],
                  job=protocol["jobs"][1], command=[sys.executable, "runner.py", "--phase", "run", "--cell", str(index)])
    write_json(directory / "launch.json", launch)
    (directory / "runtime.log").write_text("all iterations complete\n")
    # Deliberately serialize the inner receipt with a different key order.
    payload = {"campaign_result.json": (json.dumps(result, indent=2, sort_keys=False) + "\n").encode(),
               "exported/deploy.yaml": b"policy_checkpoint_path: /original/scratch/exported/policy.onnx\n",
               "model_999.pt": b"a final checkpoint"}
    archive(directory / "outputs.tar.gz", payload)
    completion = dict(status=status, exit_status=0, error=None, archive_sha256=intake["archive_sha256"],
                      output_bytes=sum(map(len, payload.values())),
                      files={p.name: dict(bytes=p.stat().st_size, sha256=collector.digest(p)) for p in directory.iterdir()})
    write_json(directory / "completion.json", completion)
    return directory


def update_completion(directory):
    completion = collector.read_json(directory / "completion.json")
    completion["files"] = {p.name: dict(bytes=p.stat().st_size, sha256=collector.digest(p))
                           for p in directory.iterdir() if p.name != "completion.json"}
    write_json(directory / "completion.json", completion)


@pytest.mark.parametrize("task,status", [("H4", "PASS"), ("H4", "NEGATIVE"), ("H3", "PASS")])
def test_full_array_recomputes_verdict_and_preserves_exact_bytes(tmp_path, monkeypatch, task, status):
    root, protocol, intake = fixture(tmp_path, task=task)
    before = {str(p.relative_to(root)): collector.digest(p) for p in root.rglob("*") if p.is_file()}
    def summarize(scientific_task, protocol_path, cells):
        assert scientific_task == task
        assert collector.digest(protocol_path) == intake["protocol_sha256"]
        assert len(cells) == len(protocol["cells"])
        assert all((cell / "exported/deploy.yaml").read_bytes().startswith(b"policy_checkpoint_path: /original/") for cell in cells)
        return dict(status=status, protocol_sha256=intake["protocol_sha256"], problems=[])
    monkeypatch.setattr(collector, "scientific_summary", summarize)
    out = tmp_path / "collection"
    report = collector.collect(root, out)
    assert report["status"] == status and report["complete_cells"] == len(protocol["cells"])
    assert collector.read_json(out / "collection.json")["status"] == status
    assert collector.read_json(out / "scientific-summary.json")["status"] == status
    after = {str(p.relative_to(root)): collector.digest(p) for p in root.rglob("*") if p.is_file()}
    assert before == after
    for row in report["cells"]:
        original = out / "originals" / row["launch"]
        assert (original / "completion.json").read_bytes() == (root / row["launch"] / "completion.json").read_bytes()
        assert (original / "campaign_result.json").read_bytes() != (out / row["directory"] / "campaign_result.json").read_bytes()


def test_missing_cell_is_incomplete_and_never_summarized(tmp_path, monkeypatch):
    root, _, _ = fixture(tmp_path)
    collector.shutil.rmtree(root / "full-323")
    monkeypatch.setattr(collector, "scientific_summary", lambda *args: pytest.fail("partial science cohort"))
    result = collector.collect(root, tmp_path / "out")
    assert result["status"] == "INCOMPLETE" and result["complete_cells"] == 2
    assert any("missing complete scored cells: [2]" in p for p in result["problems"])


def test_failed_cell_evidence_retained_and_gate_not_claimed(tmp_path):
    root, _, _ = fixture(tmp_path)
    path = root / "full-321/completion.json"
    completion = collector.read_json(path)
    completion.update(status="INCOMPLETE", exit_status=1, error="training failed")
    write_json(path, completion)
    out = tmp_path / "out"
    report = collector.collect(root, out)
    assert report["status"] == "INCOMPLETE"
    assert (out / "originals/full-321/completion.json").read_bytes() == path.read_bytes()
    assert (out / "originals/full-321/outputs.tar.gz").read_bytes() == (root / "full-321/outputs.tar.gz").read_bytes()


def test_duplicate_seed_cell_rejected(tmp_path):
    root, protocol, intake = fixture(tmp_path)
    add_cell(root, 0, protocol["cells"][0], protocol, intake, job_id="999")
    report = collector.collect(root, tmp_path / "out")
    assert report["status"] == "INCOMPLETE" and any("duplicate full-array cell" in p for p in report["problems"])


@pytest.mark.parametrize("mutation", ["archive", "protocol", "launcher", "completion_archive", "checksum", "wrong_arm", "smoke", "cell_command", "source_count", "output_bytes", "dependency", "run_resources"])
def test_mixed_or_malformed_provenance_is_incomplete(tmp_path, mutation):
    root, _, _ = fixture(tmp_path)
    directory = root / "full-321"
    if mutation in {"archive", "protocol", "launcher"}:
        name = {"archive": "frozen-campaign.tar.gz", "protocol": "protocol.json", "launcher": "h34_campaign.sbatch"}[mutation]
        with (root / name).open("ab") as stream:
            stream.write(b"changed")
    elif mutation in {"completion_archive", "output_bytes"}:
        path = directory / "completion.json"
        receipt = collector.read_json(path)
        receipt["archive_sha256" if mutation == "completion_archive" else "output_bytes"] = "wrong" if mutation == "completion_archive" else 1000000000
        write_json(path, receipt)
    elif mutation == "checksum":
        (directory / "outputs.tar.gz").write_bytes(b"corrupt")
    elif mutation in {"wrong_arm", "smoke"}:
        path = directory / "campaign_result.json"
        result = collector.read_json(path)
        result["arm" if mutation == "wrong_arm" else "phase"] = "wrong" if mutation == "wrong_arm" else "smoke"
        write_json(path, result)
        update_completion(directory)
    elif mutation in {"cell_command", "source_count"}:
        path = directory / "launch.json"
        launch = collector.read_json(path)
        if mutation == "cell_command":
            launch["command"][-1] = "1"
        else:
            launch["source_files_verified"] = 999
        write_json(path, launch)
        update_completion(directory)
    else:
        path = root / "full-submission.json"
        submission = collector.read_json(path)
        if mutation == "dependency":
            submission["command"][-1] = "--dependency=afterok:999"
        else:
            submission["resources"]["array"] = "0-1%1"
        write_json(path, submission)
    report = collector.collect(root, tmp_path / "out")
    assert report["status"] == "INCOMPLETE" and report["problems"]


def test_summary_rejects_forged_science_protocol(tmp_path, monkeypatch):
    root, _, _ = fixture(tmp_path)
    monkeypatch.setattr(collector, "scientific_summary", lambda *args: dict(status="PASS", protocol_sha256="foreign"))
    assert collector.collect(root, tmp_path / "out")["status"] == "INCOMPLETE"


def test_destination_must_be_new_and_outside_originals(tmp_path):
    root, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        collector.collect(root, root / "overwrite")
    out = tmp_path / "existing"
    out.mkdir()
    (out / "valuable.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        collector.collect(root, out)
    assert (out / "valuable.txt").read_text() == "keep"


@pytest.mark.parametrize("kind", ["traversal", "symlink", "hardlink", "duplicate", "parent_collision", "oversize"])
def test_unsafe_archive_rejected_before_extraction(tmp_path, kind):
    path = tmp_path / "hostile.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("../escape" if kind == "traversal" else "a")
        if kind in {"symlink", "hardlink"}:
            info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        else:
            info.size = 10 if kind == "oversize" else 1
            tar.addfile(info, io.BytesIO(b"x" * info.size))
            if kind in {"duplicate", "parent_collision"}:
                info2 = tarfile.TarInfo("a" if kind == "duplicate" else "a/b")
                info2.size = 1
                tar.addfile(info2, io.BytesIO(b"y"))
    destination = tmp_path / "safe-output"
    with pytest.raises(ValueError):
        collector.safe_extract(path, destination, cap=5)
    assert not destination.exists() and not (tmp_path / "escape").exists()


@pytest.mark.parametrize("name", ["", "/absolute", "../up", "a/../b", "a//b", "./file", "a\\b", 42])
def test_bad_relative_name(name):
    with pytest.raises(ValueError):
        collector.relative_name(name)


def test_durable_original_copy_is_bounded(tmp_path, monkeypatch):
    root, _, _ = fixture(tmp_path)
    with (root / "full-321/runtime.log").open("wb") as stream:
        stream.truncate(2 * 1024 ** 2)
    monkeypatch.setattr(collector, "MAX_RECEIPT_OVERHEAD", 1024)
    out = tmp_path / "out"
    result = collector.collect(root, out)
    assert result["status"] == "INCOMPLETE"
    assert not (out / "originals/full-321/runtime.log").exists()
    assert any("exceeds cap" in p for p in result["problems"])


def test_copied_and_extracted_evidence_share_one_cap(tmp_path, monkeypatch):
    root, _, _ = fixture(tmp_path)
    initial = sum(p.stat().st_size for p in root.iterdir() if p.is_file())
    first = root / "full-321"
    copied = sum(p.stat().st_size for p in first.iterdir())
    expanded = collector.read_json(first / "completion.json")["output_bytes"]
    monkeypatch.setattr(collector, "MAX_COLLECTION_BYTES", initial + copied + expanded - 1)
    result = collector.collect(root, tmp_path / "out")
    assert result["status"] == "INCOMPLETE" and result["complete_cells"] == 0
    assert not (tmp_path / "out/cells/cell-0-R1HO-s0").exists()
    assert any("uncompressed archive exceeds cap" in p for p in result["problems"])
