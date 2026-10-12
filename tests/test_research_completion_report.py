"""Prevent unattended publication from promoting incomplete or negative science."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

SPEC = importlib.util.spec_from_file_location("research_completion_report", Path(__file__).resolve().parents[1] / "scripts/bench/research_completion_report.py")
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))
    return path


def native_fixture(gated=False):
    ready = {arm: not (gated and arm == "stereo_inertial") for arm in reporter.ARMS}
    rows, jobs = [], []
    for seed in (730000, 730001):
        for condition in sorted(reporter.CONDITIONS):
            name = f"s{seed}-{condition}"
            jobs.append({"kind": "run", "name": name, "args": ["--seed", str(seed), "--condition", condition]})
            cells = [{"case": case, "seed": seed, "condition": condition, "arm": arm}
                     for case in ("straight", "occluders") for arm in sorted(reporter.ARMS)]
            measured = [cell for cell in cells if ready[cell["arm"]]]
            unrun = [dict(cell, reason="UNRUN_AFTER_NEGATIVE_NOMINAL_ARM_SMOKE") for cell in cells if not ready[cell["arm"]]]
            shard = {"status": "PASS", "episodes": len(measured), "arm_development_ready": ready,
                     "completed_cells": measured, "unrun_cells": unrun}
            rows.append((name, shard, reporter.digest_bytes(encoded(shard))))
    groups = {arm + "/" + condition: {"episodes": 4 if ready[arm] else 0, "clean_goals": 2 if ready[arm] else 0,
                                       "falls": 0, "contacts": 0}
              for arm in reporter.ARMS for condition in reporter.CONDITIONS}
    summary = {"status": "PASS", "declared_cells": 200, "measured_cells": 160 if gated else 200,
               "unrun_after_nominal_gate": 40 if gated else 0, "missing_cells": [], "arm_development_ready": ready,
               "source_results": [{"sha256": row[2]} for row in rows], "by_arm_condition": groups}
    return summary, rows, {"jobs": jobs}


def terrain_fixture():
    qualified = [{"actor": actor, "arm": arm} for actor in ("dr-default-s0", "dr-default-s1", "dr-default-s2")
                 for arm in (("heading",) if actor.endswith("s1") else ("heading", "regulated"))]
    cfg = {"qualified": qualified, "terrains": ["flat", "small_steps", "ramp"], "confirmation_seeds": list(range(440000, 440020))}
    rows, sources = [], []
    for actor in ("dr-default-s0", "dr-default-s1", "dr-default-s2"):
        for terrain in cfg["terrains"]:
            paths = [f"{actor}--{terrain}--s{seed}--{arm}/confirmation-episode.json"
                     for seed in cfg["confirmation_seeds"] for arm in ["baseline"] + [r["arm"] for r in qualified if r["actor"] == actor]]
            shard = {"episodes": len(paths), "episode_files": paths, "protocol_sha256": "a" * 64,
                     "actor_filter": actor, "terrain_filter": terrain}
            rows.append((actor + terrain, shard, reporter.digest_bytes(encoded(shard))))
            sources.extend({"path": "/original/raw/" + p, "sha256": reporter.digest_bytes(p.encode())} for p in paths)
    # All five are complete scientific negatives despite successful collection.
    comparisons = [dict(row, candidate={"episodes": 60, "clean_goals": 53, "falls": 0, "contacts": 0, "nonfinite": 0},
                        matched_baseline={"episodes": 60, "clean_goals": 30, "falls": 0, "contacts": 0, "nonfinite": 0},
                        confirmed=False) for row in qualified]
    summary = {"status": "PASS", "episodes": 480, "baseline_unique_episodes": 180, "candidate_unique_episodes": 300,
               "qualified_comparisons": comparisons, "confirmed_combinations": 0, "episode_sources": sources,
               "protocol_sha256": "a" * 64}
    return summary, rows, {"terrain_confirmation": cfg}


def test_native_complete_and_gated_cells_remain_distinct():
    measured = reporter.validate_native(*native_fixture())
    gated = reporter.validate_native(*native_fixture(True))
    assert measured["measured"] == 200 and measured["unrun_after_negative_gate"] == 0
    assert gated["measured"] == 160 and gated["unrun_after_negative_gate"] == 40


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "undeclared", "gated-measured", "wrong-count", "wrong-source", "group-count"])
def test_native_invalid_cohort_rejected(mutation):
    summary, rows, protocol = native_fixture()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[0][1]["completed_cells"][1] = rows[0][1]["completed_cells"][0]
    elif mutation == "undeclared":
        rows[0][1]["completed_cells"][0]["seed"] = 999999
    elif mutation == "gated-measured":
        summary["arm_development_ready"]["stereo"] = False
    elif mutation == "wrong-count":
        summary["measured_cells"] = 199
    elif mutation == "wrong-source":
        summary["source_results"][0]["sha256"] = "0" * 64
    else:
        summary["by_arm_condition"]["stereo/nominal"]["episodes"] = 5
    with pytest.raises(ValueError):
        reporter.validate_native(summary, rows, protocol)


def raw_binding_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "SHARED_ROOT", tmp_path)
    summary, rows, protocol = native_fixture()
    bindings, sources, evidence = [], [], {}
    for name, result, compact_sha in rows:
        # This reproduces the frozen runner: archive unsorted raw JSON, then
        # retain a compact receipt with sorted keys and identical values.
        raw_path = tmp_path / "raw" / (name + ".json")
        raw_path.parent.mkdir(exist_ok=True)
        raw_path.write_text(json.dumps(result, indent=2) + "\n")
        raw_sha = reporter.digest(raw_path)
        assert raw_sha != compact_sha
        bindings.append({"name": name, "raw_result_path": str(raw_path), "raw_result_sha256": raw_sha,
                         "compact_result_sha256": compact_sha})
        sources.append({"path": "/collector/raw/" + name + "/campaign_result.json", "sha256": raw_sha})
        evidence["native/shards/" + name + "/campaign_result.json"] = encoded(result)
    summary["source_results"] = sources
    evidence["native/scientific-summary.json"] = encoded(summary)
    manifest = {"schema": "bhl-native-result-serialization-bindings-v1", "controller_plan_sha256": "c" * 64,
                "scientific_summary_sha256": reporter.digest_bytes(encoded(summary)), "shards": bindings}
    path = write(tmp_path / "bindings.json", manifest)
    spec = {"plan_sha256": "c" * 64, "raw_result_bindings": {"path": str(path), "sha256": reporter.digest(path)}}
    return spec, summary, rows, protocol, evidence, manifest


def test_native_raw_serialization_bridge_keeps_both_byte_identities(tmp_path, monkeypatch):
    spec, summary, rows, protocol, evidence, manifest = raw_binding_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="bind every shard"):
        reporter.validate_native(summary, rows, protocol)
    rebound = reporter.bind_native_raw_results(spec, summary, rows, evidence)
    assert reporter.validate_native(summary, rebound, protocol)["measured"] == 200
    assert all(before[1] == after[1] and before[2] != after[2] for before, after in zip(rows, rebound))
    assert len([key for key in evidence if key.startswith("native/raw-results/")]) == 20
    for row in manifest["shards"]:
        assert evidence["native/raw-results/" + row["name"] + "/campaign_result.json"] == Path(row["raw_result_path"]).read_bytes()
    assert evidence["native/raw-result-bindings.json"] == Path(spec["raw_result_bindings"]["path"]).read_bytes()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "wrong-plan", "wrong-summary",
                                      "wrong-compact", "wrong-raw", "missing-file", "changed-file",
                                      "wrong-source", "duplicate-source", "wrong-source-name", "unpinned-manifest"])
def test_native_raw_binding_rejects_incomplete_or_changed_provenance(tmp_path, monkeypatch, mutation):
    spec, summary, rows, protocol, evidence, manifest = raw_binding_fixture(tmp_path, monkeypatch)
    if mutation == "missing":
        manifest["shards"].pop()
    elif mutation == "duplicate":
        manifest["shards"][1] = manifest["shards"][0]
    elif mutation == "extra":
        manifest["shards"].append(dict(manifest["shards"][0], name="extra"))
    elif mutation == "wrong-plan":
        manifest["controller_plan_sha256"] = "d" * 64
    elif mutation == "wrong-summary":
        manifest["scientific_summary_sha256"] = "d" * 64
    elif mutation == "wrong-compact":
        manifest["shards"][0]["compact_result_sha256"] = "d" * 64
    elif mutation == "wrong-raw":
        manifest["shards"][0]["raw_result_sha256"] = "d" * 64
    elif mutation == "missing-file":
        Path(manifest["shards"][0]["raw_result_path"]).unlink()
    elif mutation == "changed-file":
        Path(manifest["shards"][0]["raw_result_path"]).write_text("{}")
    elif mutation == "wrong-source":
        summary["source_results"][0]["sha256"] = "d" * 64
    elif mutation == "duplicate-source":
        summary["source_results"][1] = summary["source_results"][0]
    elif mutation == "wrong-source-name":
        summary["source_results"][0]["path"] = "/other/name.json"
    else:
        manifest["schema"] = "changed"
    path = write(Path(spec["raw_result_bindings"]["path"]), manifest)
    if mutation != "unpinned-manifest":
        spec["raw_result_bindings"]["sha256"] = reporter.digest(path)
    with pytest.raises(ValueError):
        reporter.bind_native_raw_results(spec, summary, rows, evidence)


@pytest.mark.parametrize("mutation", ["outcome", "boolean-as-integer", "integer-as-float", "duplicate-key", "nonfinite"])
def test_native_raw_binding_never_accepts_semantic_change_even_with_rehashed_manifest(tmp_path, monkeypatch, mutation):
    spec, summary, rows, protocol, evidence, manifest = raw_binding_fixture(tmp_path, monkeypatch)
    pin = manifest["shards"][0]
    raw_path = Path(pin["raw_result_path"])
    raw = json.loads(raw_path.read_text())
    if mutation == "outcome":
        raw["episodes"] = 9
    elif mutation == "boolean-as-integer":
        raw["arm_development_ready"]["stereo"] = 1
    elif mutation == "integer-as-float":
        raw["episodes"] = 10.0
    if mutation == "duplicate-key":
        raw_path.write_text(raw_path.read_text().replace('"status": "PASS"', '"status": "FAIL", "status": "PASS"'))
    elif mutation == "nonfinite":
        raw["episodes"] = float("nan")
        raw_path.write_text(json.dumps(raw))
    else:
        raw_path.write_text(json.dumps(raw))
    pin["raw_result_sha256"] = reporter.digest(raw_path)
    summary["source_results"][0]["sha256"] = pin["raw_result_sha256"]
    evidence["native/scientific-summary.json"] = encoded(summary)
    manifest["scientific_summary_sha256"] = reporter.digest_bytes(encoded(summary))
    path = write(Path(spec["raw_result_bindings"]["path"]), manifest)
    spec["raw_result_bindings"]["sha256"] = reporter.digest(path)
    with pytest.raises(ValueError, match="semantically|duplicate JSON|nonfinite JSON"):
        reporter.bind_native_raw_results(spec, summary, rows, evidence)


def test_terrain_collection_pass_keeps_negative_science():
    result = reporter.validate_terrain(*terrain_fixture())
    assert result["measured"] == 480
    assert result["confirmed_combinations"] == 0


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "wrong-source", "wrong-shard", "wrong-gate", "wrong-denominator", "wrong-total"])
def test_terrain_invalid_confirmation_rejected(mutation):
    summary, rows, protocol = terrain_fixture()
    if mutation == "duplicate":
        rows[0][1]["episode_files"][1] = rows[0][1]["episode_files"][0]
    elif mutation == "missing":
        rows.pop()
    elif mutation == "wrong-source":
        summary["episode_sources"][0]["path"] = "/not/a/declared-episode.json"
    elif mutation == "wrong-shard":
        rows[0][1]["actor_filter"] = "other"
    elif mutation == "wrong-gate":
        summary["qualified_comparisons"][0]["confirmed"] = True
    elif mutation == "wrong-denominator":
        summary["qualified_comparisons"][0]["candidate"]["episodes"] = 59
    else:
        summary["confirmed_combinations"] = 1
    with pytest.raises(ValueError):
        reporter.validate_terrain(summary, rows, protocol)


def archive(path, members):
    with tarfile.open(path, "w:gz") as stream:
        for name, value in members:
            info = tarfile.TarInfo(name)
            info.size = len(value)
            stream.addfile(info, io.BytesIO(value))
    return path


def test_final_archive_must_match_exact_local_summary_and_protocol(tmp_path):
    evidence = {"native/scientific-summary.json": b'{"status":"PASS"}', "native/protocol.json": b'{}'}
    path = archive(tmp_path / "final.tar.gz", [(key.split("/")[-1], value) for key, value in evidence.items()])
    reporter.verify_final_archive(path, evidence, "native")
    evidence["native/scientific-summary.json"] = b'{"status":"NEGATIVE"}'
    with pytest.raises(ValueError, match="differs"):
        reporter.verify_final_archive(path, evidence, "native")


def test_final_archive_duplicate_summary_cannot_hide_tampering(tmp_path):
    path = archive(tmp_path / "final.tar.gz", [("scientific-summary.json", b"{}"), ("scientific-summary.json", b"{}"), ("protocol.json", b"{}")])
    with pytest.raises(ValueError, match="duplicate"):
        reporter.verify_final_archive(path, {"native/scientific-summary.json": b"{}", "native/protocol.json": b"{}"}, "native")


def publication():
    name = "native-controls-final-20261011-test.tar.gz"
    return {"asset_name": name, "bytes": 10, "sha256": "a" * 64, "verified_utc": "2026-10-11T21:00:00Z",
            "verification": "fresh download SHA256", "url": f"https://github.com/{reporter.REPOSITORY}/releases/download/{reporter.TAG}/{name}"}


@pytest.mark.parametrize("mutation", ["no-verification", "wrong-host", "oversized"])
def test_unverified_or_unbounded_final_publication_rejected(mutation):
    value = publication()
    if mutation == "no-verification":
        value.pop("verified_utc")
    elif mutation == "wrong-host":
        value["url"] = "https://example.org/final.tar.gz"
    else:
        value["bytes"] = reporter.ARCHIVE_CAP + 1
    with pytest.raises(ValueError):
        reporter.checked_publication(value, final=True)


def test_controller_failure_never_requires_or_claims_final_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "SHARED_ROOT", tmp_path)
    directory = tmp_path / "controller"
    plan = write(directory / "plan.json", {"campaign_dir": str(tmp_path / "campaign")})
    spec = {"directory": str(directory), "plan_sha256": reporter.digest(plan)}
    write(directory / "state.json", {"plan_sha256": spec["plan_sha256"], "status": "FAILED_NO_RETRY", "error": "native execution failed", "jobs": []})
    evidence = {}
    result = reporter.inspect_controller("native", spec, evidence, tmp_path, None, None)
    assert result["complete"] is False and result["error"] == "native execution failed"
    assert "summary" not in result


def test_pinned_recovery_preserves_failed_original_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "SHARED_ROOT", tmp_path)
    campaign = tmp_path / "campaign"
    name, jid = "native-controls-v2-s730000-c04-20261011", "21776269"
    directory = campaign / (name + "-" + jid)
    launch = write(directory / "launch.json", {"archive_sha256": "b" * 64})
    raw_bytes = 3 * 1024**3
    old = write(directory / "completion.json", {"status": "INCOMPLETE", "exit_status": 0, "archive_sha256": "b" * 64,
                "error": f"output exceeds predeclared archive cap ({raw_bytes} bytes)", "output_bytes": raw_bytes,
                "files": {"launch.json": {"sha256": reporter.digest(launch), "bytes": launch.stat().st_size}}})
    original_bytes = old.read_bytes()
    recovery_root = tmp_path / "recovery"
    recovery = {"job_name": name, "job_id": jid}
    part = dict(publication(), files=["raw.bin"])
    receipt = {"schema": "bhl-native-controls-packaging-recovery-v1", "status": "COMPLETE", "scientific_rerun": False,
               "recovered_scientific_status": "PASS", "original_job_id": jid, "source_archive_sha256": "b" * 64,
               "protocol_sha256": "c" * 64, "original_completion_sha256": reporter.digest(old),
               "original_runtime_exit_status": 0, "raw_output_bytes": raw_bytes, "parts": [part]}
    for key, value in (("result", {"status": "PASS", "episodes": 10, "phase": "development", "expected_declared_cells": 10}),
                       ("inventory", {"raw.bin": {"sha256": "f" * 64, "bytes": raw_bytes}})):
        path = write(recovery_root / (key + ".json"), value)
        recovery[key], recovery[key + "_sha256"] = str(path), reporter.digest(path)
    receipt.update(result_sha256=recovery["result_sha256"], inventory_sha256=recovery["inventory_sha256"])
    path = write(recovery_root / "receipt.json", receipt)
    recovery["receipt"], recovery["receipt_sha256"] = str(path), reporter.digest(path)
    plan = {"jobs": [{"name": name}], "campaign_dir": str(campaign), "archive_sha256": "b" * 64,
            "protocol_sha256": "c" * 64, "recovery": recovery}
    pub = {"schema": "bhl-native-recovered-tar-volumes-v1", "parts": [part], "inventory": recovery["inventory"],
           "inventory_sha256": recovery["inventory_sha256"], "raw_output_bytes": raw_bytes}
    state = {"jobs": [{"name": name, "job_id": jid, "state": "COMPLETE", "publication": pub,
                       "packaging_recovered": True, "recovery_receipt_sha256": recovery["receipt_sha256"]}]}
    evidence = {}
    rows = reporter.shard_results("native", {}, plan, state, evidence)
    assert rows[0][1]["episodes"] == 10 and old.read_bytes() == original_bytes
    Path(recovery["result"]).write_text('{"status":"PASS","episodes":9}')
    with pytest.raises(ValueError, match="recovered"):
        reporter.shard_results("native", {}, plan, state, {})


def multipart_fixture(tmp_path):
    campaign, name, jid = tmp_path / "campaign", "native-c05", "456"
    directory = campaign / (name + "-" + jid)
    result = write(directory / "campaign_result.json", {"status": "PASS", "episodes": 10})
    archive_pin = {"sha256": "c" * 64, "bytes": 25}
    completion = {"status": "PASS", "exit_status": 0, "error": None, "archive_sha256": "b" * 64,
                  "files": {"campaign_result.json": {"sha256": reporter.digest(result), "bytes": result.stat().st_size},
                            "outputs.tar.gz": archive_pin}}
    write(directory / "completion.json", completion)
    part_name = "native-controls-v3-c05-cccccccccccc.tar.gz.part000"
    part = {"asset_name": part_name, "sha256": "d" * 64, "bytes": 25,
            "url": f"https://github.com/{reporter.REPOSITORY}/releases/download/{reporter.TAG}/{part_name}"}
    pub = dict(archive_pin, schema="bhl-native-raw-publication-v1", parts=[part], verified_utc="now",
               verification="each part independently downloaded and SHA256 verified; whole archive SHA256 retained")
    transport = {"schema": "bhl-native-transport-result-v1", "status": "PASS", "runner_returncode": 0,
                 "runtime_exit_status": 0, "source_archive_sha256": "b" * 64, "protocol_sha256": "e" * 64,
                 "plan_sha256": "f" * 64, "original_max_output_mb": 2048, "operational_max_output_mb": 8192,
                 "publication": pub}
    write(directory / "remote-result.json", transport)
    plan = {"jobs": [{"name": name}], "campaign_dir": str(campaign), "archive_sha256": "b" * 64, "protocol_sha256": "e" * 64}
    state = {"jobs": [{"name": name, "job_id": jid, "state": "COMPLETE", "publication": pub}]}
    return {"plan_sha256": "f" * 64}, plan, state, directory


def test_multipart_worker_shard_validates_original_archive_and_transport(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "SHARED_ROOT", tmp_path)
    spec, plan, state, directory = multipart_fixture(tmp_path)
    rows = reporter.shard_results("native", spec, plan, state, {})
    assert rows[0][1]["episodes"] == 10
    transport = reporter.read(directory / "remote-result.json")
    transport["source_archive_sha256"] = "a" * 64
    write(directory / "remote-result.json", transport)
    with pytest.raises(ValueError, match="transport"):
        reporter.shard_results("native", spec, plan, state, {})


@pytest.mark.parametrize("mutation", ["whole-hash", "missing-part", "part-order", "part-url"])
def test_multipart_incomplete_or_wrong_original_bytes_rejected(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(reporter, "SHARED_ROOT", tmp_path)
    spec, plan, state, directory = multipart_fixture(tmp_path)
    pub = state["jobs"][0]["publication"]
    if mutation == "whole-hash":
        pub["sha256"] = "a" * 64
    elif mutation == "missing-part":
        pub["parts"] = []
    elif mutation == "part-order":
        pub["parts"][0]["asset_name"] = "wrong.tar.gz.part002"
    else:
        pub["parts"][0]["url"] = "https://example.org/raw"
    transport = reporter.read(directory / "remote-result.json")
    transport["publication"] = pub
    write(directory / "remote-result.json", transport)
    with pytest.raises(ValueError, match="multipart"):
        reporter.shard_results("native", spec, plan, state, {})


def test_queued_controller_without_state_is_not_started(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "SHARED_ROOT", tmp_path)
    directory = tmp_path / "controller"
    plan = write(directory / "plan.json", {"campaign_dir": str(tmp_path / "campaign")})
    spec = {"directory": str(directory), "plan_sha256": reporter.digest(plan)}
    result = reporter.inspect_controller("native", spec, {}, tmp_path, None, None)
    assert result["status"] == "NOT_STARTED" and result["complete"] is False
    assert result["status"] not in reporter.FAILURES


def test_failure_report_retains_reason_and_no_completion_claim(tmp_path):
    result = {"observed_utc": "2026-10-11T22:00:00+00:00", "status": "INCOMPLETE", "campaigns": {
        "native": {"status": "FAILED_NO_RETRY", "complete": False, "error": "execution failed"},
        "terrain": {"status": "WAITING_FOR_SHARED_SPACE", "complete": False, "completed_shards": 5}}}
    identifier = reporter.write_bundle(tmp_path / "bundle", result, {"native/state.json": b"{}"})
    markdown = (tmp_path / "bundle/report.md").read_text()
    assert "INCOMPLETE" in markdown and "execution failed" in markdown
    assert "Measured **" not in markdown
    manifest = reporter.read(tmp_path / "bundle/manifest.json")
    assert "report.md" in manifest["files"] and "report.json" in manifest["files"]
    assert "/" not in identifier
    with pytest.raises(FileExistsError):
        reporter.write_bundle(tmp_path / "bundle", result, {})


def test_complete_negative_terrain_report_does_not_upgrade_gate():
    summary, rows, protocol = terrain_fixture()
    counts = reporter.validate_terrain(summary, rows, protocol)
    value = {"observed_utc": "now", "status": "INCOMPLETE", "campaigns": {
        "native": {"status": "RUNNING", "complete": False},
        "terrain": {"status": "COMPLETE", "complete": True, "summary": summary, "counts": counts,
                    "publication": publication(), "source_archive_sha256": "b" * 64}}}
    markdown = reporter.render_report(value, "../evidence")
    assert "**0/5**" in markdown and markdown.count("| NEGATIVE |") == 5
    assert "180 unique baseline and 300 candidate" in markdown


def test_guard_refuses_symlink_receipt_and_outside_shared_root(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "SHARED_ROOT", tmp_path)
    path = write(tmp_path / "file.json", {})
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        reporter.read(link)
    with pytest.raises(ValueError):
        reporter.scoped(tmp_path.parent / "elsewhere")
