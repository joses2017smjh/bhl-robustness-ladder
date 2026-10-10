"""Evidence-boundary tests for complete and partial sensor-method collections."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

PATH = Path(__file__).resolve().parents[1]/"scripts/bench/sensor_methods_collect.py"
spec = importlib.util.spec_from_file_location("sensor_methods_collect_tests", PATH)
COLLECT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(COLLECT)


def encoded(value):
    return (json.dumps(value, sort_keys=True)+"\n").encode()


def protocol():
    cells = [dict(id=f"family-{g}-{condition}", geometry_group=f"group-{g}", family="family",
                  geometry_seed=g, condition=condition)
             for g in range(6) for condition in ("nominal", "low_texture", "lidar_half")]
    return dict(sensor_methods=dict(cells=cells, frames=150, replay_gate={"maximum_ate_rmse_m": .1}, model_input="model.onnx"),
                input_files={"model.onnx": {"sha256": "model-hash"}}, jobs=[])


def payload():
    p = protocol()
    cell = p["sensor_methods"]["cells"][0]
    intake = dict(protocol_sha256="protocol-hash", archive_sha256="source-hash", source_files=4)
    runtimes = {name: dict(receipt_sha256=name+"-receipt", value=dict(binary_sha256=name+"-binary", upstream_commit=name+"-commit"))
                for name in ("lio", "livo")}
    files, frames = {}, []
    for index in range(150):
        frame = dict(frame_id=f"{index:05d}", timestamp_s=index*.2)
        for key in ("left", "right", "lidar", "imu", "truth"):
            name = f"inference/{index}-{key}"
            frame[key] = name
            files["replay/"+name] = (key+str(index)).encode()
        frames.append(frame)
    manifest = dict(cell=cell, origin="simulation", protocol_sha256="protocol-hash", sequences=[dict(id=cell["id"], frames=frames)],
                    file_sha256={name.removeprefix("replay/"): COLLECT.sha(value) for name, value in files.items()})
    files["replay/manifest.json"] = encoded(manifest)
    plan = dict(schema="bhl-native-livo-methods-v1", methods=list(COLLECT.NATIVE), sequences=[cell["id"]], gate=p["sensor_methods"]["replay_gate"])
    files["native-plan.json"] = encoded(plan)
    depth = dict(mae_m=.1, rmse_m=.2, coverage=.8, near_obstacle_pixel_recall=.75)
    terrain = dict(ground_height_rmse_m=.02, ground_coverage=.6, declared_95_percent_interval_coverage=.9,
                   ground_hazard_recall=.8, hazard_recall_including_unresolved=.5, false_traversable_ground_cells=3,
                   unresolved_ground_hazards=4, sensor_conflict_cells=2)
    mapping = dict(status="PASS", frames=150, rows=[dict(frame_id=f["frame_id"],
        depth={m:dict(native_support=depth.copy(), common_support=depth.copy()) for m in COLLECT.DEPTH},
        terrain={m:terrain.copy() for m in COLLECT.MAPS}) for f in frames],
        depth_latency={m:dict(p95_ms=20.) for m in COLLECT.DEPTH},
        provenance={"c_fast_foundationstereo": dict(model_sha256="model-hash", provider="CUDAExecutionProvider",
            provider_execution_evidence=dict(status="VERIFIED_CUDA_KERNEL_EXECUTION", kernel_events_by_provider={"CUDAExecutionProvider": 1}))})
    files["mapping/metrics.json"] = encoded(mapping)
    rows = []
    for method in COLLECT.NATIVE:
        prefix = f"native/{cell['id']}/{method}/"
        raw = [dict(frame_index=i, tracked=True, visual_points=int(method == "fast_livo2")) for i in range(150)]
        files[prefix+"native_frames.jsonl"] = b"".join(encoded(f) for f in raw)
        files[prefix+"trajectory.json"] = encoded(dict(poses=150))
        rt = runtimes["lio" if method == "fast_lio2" else "livo"]
        keys = ("lidar", "imu") if method == "fast_lio2" else ("left", "lidar", "imu")
        receipt = dict(method=COLLECT.METHOD_NAMES[method], input_manifest_sha256=COLLECT.sha(files["replay/manifest.json"]),
            runtime_sha256=rt["value"]["binary_sha256"], runtime_receipt_sha256=rt["receipt_sha256"],
            upstream_commit=rt["value"]["upstream_commit"], native_frames_sha256=COLLECT.sha(files[prefix+"native_frames.jsonl"]),
            input_files_sha256={f[key]:manifest["file_sha256"][f[key]] for f in frames for key in keys})
        if method == "fast_lio2":
            receipt["config_sha256"] = "lio-config"
        else:
            receipt["config"] = dict(visual_enabled=method == "fast_livo2", parameter=.1)
        files[prefix+"estimator_receipt.json"] = encoded(receipt)
        measured = dict(frames=150, tracked_frames=150, estimator_receipt_sha256=COLLECT.sha(files[prefix+"estimator_receipt.json"]),
            trajectory_sha256=COLLECT.sha(files[prefix+"trajectory.json"]), frames_with_visual_measurements=150 if method == "fast_livo2" else 0,
            native_compute_p95_ms=10.)
        rows.append(dict(method=method, sequence=cell["id"], native=measured,
            evaluation=dict(replay_readiness_gate="PASS", segments=[dict(metrics=dict(ate_translation_m=dict(rmse=.01)))])))
    native = dict(status="PASS", method_cells=3, rows=rows, closed_loop_episodes=0,
        manifest_sha256=COLLECT.sha(files["replay/manifest.json"]), protocol_sha256=COLLECT.sha(files["native-plan.json"]))
    files["native/campaign_result.json"] = encoded(native)
    result = dict(schema="bhl-sensor-methods-cell-v1", status="PASS", phase="run",
        scientific_status="DEVELOPMENT_CELL_NO_GLOBAL_VERDICT", cell=cell, protocol_sha256="protocol-hash",
        capture=dict(frames=150, manifest_sha256=COLLECT.sha(files["replay/manifest.json"])))
    files["campaign_result.json"] = encoded(result)
    records = {name: ([json.loads(line) for line in data.splitlines()] if name.endswith(".jsonl") else json.loads(data))
               for name, data in files.items() if name.endswith((".json", ".jsonl"))}
    hashes = {name:COLLECT.sha(data) for name, data in files.items()}
    return p, intake, runtimes, result, records, hashes, files


def test_complete_raw_payload_validates_all_three_actual_runtime_receipts():
    p, intake, runtimes, result, records, hashes, _ = payload()
    value = COLLECT.validate_payload(records, hashes, result, p, intake, runtimes)
    assert value["native_cells"] == 3 and value["frame_count"] == 150
    assert value["measurements"]["map/lidar/hazard_recall_including_unresolved"] == .5


@pytest.mark.parametrize("fault", ("missing_frame", "wrong_runtime", "wrong_truth_hash", "missing_fusion", "fake_cuda", "changed_gate", "visual_in_ablation"))
def test_raw_payload_rejects_changed_or_incomplete_scientific_evidence(fault):
    p, intake, runtimes, result, records, hashes, _ = payload()
    prefix = f"native/{result['cell']['id']}/fast_livo2_no_visual/"
    if fault == "missing_frame":
        records[prefix+"native_frames.jsonl"].pop()
    elif fault == "wrong_runtime":
        records[prefix+"estimator_receipt.json"]["runtime_sha256"] = "other-binary"
    elif fault == "wrong_truth_hash":
        records["replay/manifest.json"]["file_sha256"]["inference/0-truth"] = "wrong"
    elif fault == "missing_fusion":
        del records["mapping/metrics.json"]["rows"][0]["terrain"]["sgbm_gated_fusion"]
    elif fault == "fake_cuda":
        records["mapping/metrics.json"]["provenance"]["c_fast_foundationstereo"]["provider_execution_evidence"]["kernel_events_by_provider"] = {}
    elif fault == "changed_gate":
        records["native-plan.json"]["gate"]["maximum_ate_rmse_m"] = 10.
    else:
        records[prefix+"native_frames.jsonl"][0]["visual_points"] = 1
    with pytest.raises(ValueError):
        COLLECT.validate_payload(records, hashes, result, p, intake, runtimes)


def summary_rows():
    p = protocol()
    keys = {key for _, left, right in COLLECT.paired_comparisons() for key in (left, right)}
    rows = []
    for cell in p["sensor_methods"]["cells"]:
        measurements = {key:0. for key in keys}
        # Deliberately unequal condition effects; all three conditions must stay grouped.
        measurements["native/fast_livo2/ate_rmse_m"] = dict(nominal=0., low_texture=0., lidar_half=30.)[cell["condition"]]
        rows.append(dict(cell=cell, measurements=measurements, frame_count=150, native_cells=3,
                         native_config_sha256={m:"fixed" for m in COLLECT.NATIVE},
                         qualification={m:"PASS" for m in COLLECT.NATIVE}))
    return p, rows


def test_group_bootstrap_uses_six_geometry_groups_with_all_conditions_paired():
    p, rows = summary_rows()
    result = COLLECT.aggregate(rows, p)
    estimate = result["paired_comparisons"]["livo_minus_fast_lio2/ate_rmse_m"]
    assert result["status"] == "PASS" and result["observed_native_cells"] == 54
    assert estimate["groups_with_both_metrics"] == 6
    assert estimate["mean_difference"] == estimate["lower_95"] == estimate["upper_95"] == 10.


def test_partial_cohort_has_observed_counts_without_global_inference():
    p, rows = summary_rows()
    result = COLLECT.aggregate(rows[:-1], p)
    assert result["status"] == "INCOMPLETE" and result["observed_camera_frames"] == 2550
    assert result["complete_geometry_groups"] == 5 and result["paired_comparisons"] == {}
    assert result["observed_native_qualification"]["fast_livo2"] == dict(qualified=17, evaluated=17)


def test_duplicate_cell_quarantines_both_copies_and_cannot_complete():
    p, rows = summary_rows()
    result = COLLECT.aggregate(rows+[copy.deepcopy(rows[0])], p)
    assert result["status"] == "INCOMPLETE" and result["observed_cells"] == 17
    assert result["problems"] and result["paired_comparisons"] == {}


def test_changed_native_parameters_and_missing_metric_groups_are_not_silent():
    p, rows = summary_rows()
    rows[0]["native_config_sha256"]["fast_livo2"] = "changed"
    assert COLLECT.aggregate(rows, p)["status"] == "INCOMPLETE"
    rows[0]["native_config_sha256"]["fast_livo2"] = "fixed"
    rows[0]["measurements"]["native/fast_livo2/ate_rmse_m"] = None
    result = COLLECT.aggregate(rows, p)
    assert result["paired_comparisons"]["livo_minus_fast_lio2/ate_rmse_m"]["status"] == "UNAVAILABLE_METRIC_GROUPS"


def write_archive(path, files):
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def test_archive_stream_rejects_links_and_nonfinite_json(tmp_path):
    archive = tmp_path/"linked.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("linked")
        member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(member)
    with pytest.raises(ValueError, match="unsafe"):
        COLLECT.archive_records(archive, 1024)
    archive = tmp_path/"nonfinite.tar.gz"
    write_archive(archive, {"metric.json": b'{"metric": NaN}'})
    with pytest.raises(ValueError, match="nonfinite"):
        COLLECT.archive_records(archive, 1024)


def test_completion_verified_mirror_is_required_and_hash_checked(tmp_path):
    job, mirror = tmp_path/"job", tmp_path/"mirror"
    job.mkdir(); mirror.mkdir()
    raw = mirror/"raw.tar.gz"
    raw.write_bytes(b"original completed archive")
    specification = dict(bytes=raw.stat().st_size, sha256=COLLECT.SAFE.digest(raw))
    completion = dict(files={"outputs.tar.gz": specification})
    publication = dict(schema="bhl-methods-archive-v1", source_archive_sha256="source", asset_name=raw.name, **specification)
    (job/"publication.json").write_bytes(encoded(publication))
    assert COLLECT.find_archive(job, completion, "source", [mirror]) == raw
    with pytest.raises(ValueError, match="explicit verified mirror"):
        COLLECT.find_archive(job, completion, "source", [])
    raw.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        COLLECT.find_archive(job, completion, "source", [mirror])


def test_full_job_completion_checks_source_and_archived_raw_payload(tmp_path):
    p, intake, runtimes, result, _, _, files = payload()
    declared = dict(name="sensor-cell", kind="run", args=["--phase", "run", "--cell", "0"], max_output_mb=16)
    p["jobs"] = [declared]
    job = tmp_path/"sensor-cell-123"; job.mkdir()
    launch = dict(job=declared, job_id="123", command=declared["args"], source_files_verified=4, archive_sha256="source-hash")
    (job/"launch.json").write_bytes(encoded(launch))
    (job/"campaign_result.json").write_bytes(encoded(result))
    (job/"runtime.log").write_text("completed")
    write_archive(job/"outputs.tar.gz", files)
    completion = dict(status="PASS", exit_status=0, error=None, finished_utc="2026-10-10T00:00:00Z", archive_sha256="source-hash",
        output_bytes=sum(map(len, files.values())), files={f.name:dict(bytes=f.stat().st_size, sha256=COLLECT.SAFE.digest(f)) for f in job.iterdir()})
    (job/"completion.json").write_bytes(encoded(completion))
    assert COLLECT.collect_job(job, p, intake, runtimes, [])["frame_count"] == 150
    completion["archive_sha256"] = "different-source"
    (job/"completion.json").write_bytes(encoded(completion))
    with pytest.raises(ValueError, match="frozen source"):
        COLLECT.collect_job(job, p, intake, runtimes, [])


def test_protocol_cannot_split_conditions_across_geometry_groups():
    p = protocol()
    p["sensor_methods"]["cells"][0]["geometry_group"] = "seventh-group"
    with pytest.raises(ValueError, match="six geometry groups"):
        COLLECT.validate_protocol(p)
