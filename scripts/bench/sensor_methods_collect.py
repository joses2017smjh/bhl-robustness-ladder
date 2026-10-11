#!/usr/bin/env python3
"""Verify raw sensor-method evidence and summarize six paired geometry groups.

No downloads, credentials, submissions or scientific reruns. Missing evidence
produces INCOMPLETE; execution completeness never implies method improvement.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h34_collect as SAFE

CONDITIONS = {"nominal", "low_texture", "lidar_half"}
DEPTH = ("sgbm", "c_fast_foundationstereo")
NATIVE = ("fast_lio2", "fast_livo2", "fast_livo2_no_visual")
MAPS = ("lidar", *DEPTH, *(m+s for m in DEPTH for s in ("_simple_fusion", "_gated_fusion")))
METHOD_NAMES = {"fast_lio2": "fast_lio2_lidar_imu", "fast_livo2": "fast_livo2_camera_lidar_imu",
                "fast_livo2_no_visual": "fast_livo2_lidar_imu_ablation"}


def parse(data):
    def invalid(value):
        raise ValueError("nonfinite JSON: "+value)
    return json.loads(data, parse_constant=invalid)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_protocol(protocol):
    settings = protocol["sensor_methods"]
    cells = settings["cells"]
    if settings["frames"] != 150 or len(cells) != 18 or len({c["id"] for c in cells}) != 18:
        raise ValueError("exactly 18 unique declared 150-frame cells required")
    groups = defaultdict(list)
    for cell in cells:
        if PurePosixPath(cell["id"]).name != cell["id"]:
            raise ValueError("unsafe cell id")
        groups[cell["geometry_group"]].append(cell)
    if len(groups) != 6:
        raise ValueError("six geometry groups required")
    for group in groups.values():
        if (len(group) != 3 or {c["condition"] for c in group} != CONDITIONS or
                len({(c["family"], c["geometry_seed"]) for c in group}) != 1):
            raise ValueError("all three conditions must retain the same geometry group")
    return cells


def checked_file(path, specification):
    path = SAFE.regular_file(path)
    if path.stat().st_size != specification["bytes"] or SAFE.digest(path) != specification["sha256"]:
        raise ValueError("completion checksum/size mismatch: "+str(path))
    return path


def find_archive(job, completion, source_hash, mirror_dirs):
    specification = completion["files"]["outputs.tar.gz"]
    if (job/"outputs.tar.gz").exists():
        return checked_file(job/"outputs.tar.gz", specification)
    publication = SAFE.read_json(job/"publication.json")
    name = publication["asset_name"]
    if (PurePosixPath(name).name != name or publication.get("schema") != "bhl-methods-archive-v1" or
            publication.get("source_archive_sha256") != source_hash or
            publication.get("sha256") != specification["sha256"] or
            publication.get("bytes") != specification["bytes"]):
        raise ValueError("publication/completion lineage mismatch")
    matches = [Path(directory)/name for directory in mirror_dirs if (Path(directory)/name).exists()]
    if not matches:
        raise ValueError("raw archive missing; provide an explicit verified mirror directory")
    return checked_file(matches[0], specification)


def archive_records(archive, cap):
    """Single streaming pass; retain JSON only, hash all original raw evidence."""
    hashes, records, total = {}, {}, 0
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            SAFE.relative_name(member.name)
            if (not member.isfile() or member.name in hashes or member.size < 0 or
                    member.size > 512*1024**2 or len(hashes) >= SAFE.MAX_MEMBERS):
                raise ValueError("unsafe or duplicate raw archive member")
            total += member.size
            if total > cap:
                raise ValueError("raw archive exceeds declared output cap")
            keep = member.name.endswith((".json", "/native_frames.jsonl"))
            if keep and member.size > SAFE.MAX_JSON_BYTES:
                raise ValueError("raw JSON evidence exceeds cap")
            h, chunks = hashlib.sha256(), []
            with stream.extractfile(member) as source:
                for chunk in iter(lambda: source.read(1024*1024), b""):
                    h.update(chunk)
                    if keep:
                        chunks.append(chunk)
            hashes[member.name] = h.hexdigest()
            if keep:
                value = b"".join(chunks)
                records[member.name] = ([parse(line) for line in value.splitlines()] if member.name.endswith(".jsonl")
                                        else parse(value))
    for name in hashes:
        if any(str(p) in hashes for p in PurePosixPath(name).parents if str(p) != "."):
            raise ValueError("archive file/directory collision")
    return records, hashes, total


def runtime_receipts(campaign, protocol):
    result = {}
    with tarfile.open(campaign/"frozen-campaign.tar.gz", "r:gz") as outer:
        for kind in ("lio", "livo"):
            name = "inputs/"+protocol["sensor_methods"][kind+"_input"]
            found = []
            with outer.extractfile(name) as fileobj, tarfile.open(fileobj=fileobj, mode="r|gz") as inner:
                for member in inner:
                    if member.name in ("runtime.json", "runtime/runtime.json"):
                        if not member.isfile() or member.size > SAFE.MAX_JSON_BYTES:
                            raise ValueError("invalid frozen native runtime receipt")
                        data = inner.extractfile(member).read()
                        found.append(dict(receipt_sha256=sha(data), value=parse(data)))
            if len(found) != 1 or found[0]["value"].get("status") != "PASS":
                raise ValueError("one passing frozen native runtime receipt required")
            result[kind] = found[0]
    return result


def mean_optional(values):
    present = [v for v in values if v is not None]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in present):
        raise ValueError("nonfinite/nonnumeric metric")
    return float(np.mean(present)) if present else None


def validate_payload(records, hashes, result, protocol, intake, runtimes):
    cells = validate_protocol(protocol)
    cell = result["cell"]
    if cell not in cells or result.get("protocol_sha256") != intake["protocol_sha256"]:
        raise ValueError("cell identity or protocol mismatch")
    if (result.get("schema") != "bhl-sensor-methods-cell-v1" or result.get("status") != "PASS" or
            result.get("phase") != "run" or result.get("scientific_status") != "DEVELOPMENT_CELL_NO_GLOBAL_VERDICT" or
            records["campaign_result.json"] != result):
        raise ValueError("only complete development cells can be collected")
    manifest = records["replay/manifest.json"]
    frames = manifest["sequences"][0]["frames"]
    ids = [f["frame_id"] for f in frames]
    if (manifest.get("cell") != cell or manifest.get("protocol_sha256") != intake["protocol_sha256"] or
            manifest.get("origin") != "simulation" or len(manifest["sequences"]) != 1 or
            manifest["sequences"][0]["id"] != cell["id"] or len(frames) != 150 or len(set(ids)) != 150 or
            result["capture"] != dict(frames=150, manifest_sha256=hashes["replay/manifest.json"])):
        raise ValueError("capture manifest/count/hash mismatch")
    files = manifest["file_sha256"]
    if not files:
        raise ValueError("original sensor and truth hashes required")
    for name, expected in files.items():
        SAFE.relative_name(name)
        if hashes["replay/"+name] != expected:
            raise ValueError("raw sensor or evaluator truth changed")
    for frame in frames:
        if any(frame[key] not in files for key in ("left", "right", "lidar", "imu", "truth")):
            raise ValueError("frame has unhashed sensor/truth input")
    mapping = records["mapping/metrics.json"]
    if (mapping.get("status") != "PASS" or mapping.get("frames") != 150 or
            [r["frame_id"] for r in mapping["rows"]] != ids):
        raise ValueError("mapping frame cohort mismatch")
    learned = mapping["provenance"]["c_fast_foundationstereo"]
    expected_model = protocol["input_files"][protocol["sensor_methods"]["model_input"]]["sha256"]
    execution = learned.get("provider_execution_evidence", {})
    if (learned["model_sha256"] != expected_model or learned["provider"] != "CUDAExecutionProvider" or
            execution.get("status") != "VERIFIED_CUDA_KERNEL_EXECUTION" or
            execution.get("kernel_events_by_provider", {}).get("CUDAExecutionProvider", 0) < 1):
        raise ValueError("pretrained model or required execution provider mismatch")
    measurements = {}
    available = {}
    for row in mapping["rows"]:
        if set(row["depth"]) != set(DEPTH) or set(row["terrain"]) != set(MAPS):
            raise ValueError("missing depth or sensor-fusion ablation")
    for method in DEPTH:
        for support in ("native_support", "common_support"):
            for metric in ("mae_m", "rmse_m", "coverage", "near_obstacle_pixel_recall"):
                key = f"depth/{method}/{support}/{metric}"
                values = [r["depth"][method][support][metric] for r in mapping["rows"]]
                measurements[key] = mean_optional(values)
                available[key] = sum(v is not None for v in values)
        measurements[f"depth/{method}/p95_ms"] = mapping["depth_latency"][method]["p95_ms"]
    for method in MAPS:
        for metric in ("ground_height_rmse_m", "ground_coverage", "declared_95_percent_interval_coverage",
                       "ground_hazard_recall", "hazard_recall_including_unresolved", "false_traversable_ground_cells",
                       "unresolved_ground_hazards", "sensor_conflict_cells"):
            key = f"map/{method}/{metric}"
            values = [r["terrain"][method][metric] for r in mapping["rows"]]
            measurements[key] = mean_optional(values)
            available[key] = sum(v is not None for v in values)
    native = records["native/campaign_result.json"]
    if (native.get("status") != "PASS" or native.get("method_cells") != 3 or
            [r["method"] for r in native["rows"]] != list(NATIVE) or native.get("closed_loop_episodes") != 0 or
            native.get("manifest_sha256") != hashes["replay/manifest.json"] or
            native.get("protocol_sha256") != hashes["native-plan.json"]):
        raise ValueError("native three-arm cohort or input lineage mismatch")
    expected_plan = dict(schema="bhl-native-livo-methods-v1", methods=list(NATIVE), sequences=[cell["id"]],
                         gate=protocol["sensor_methods"]["replay_gate"])
    if records["native-plan.json"] != expected_plan:
        raise ValueError("native gate changed")
    qualification, configs = {}, {}
    for row in native["rows"]:
        method, measured = row["method"], row["native"]
        prefix = f"native/{cell['id']}/{method}/"
        receipt = records[prefix+"estimator_receipt.json"]
        raw = records[prefix+"native_frames.jsonl"]
        runtime = runtimes["lio" if method == "fast_lio2" else "livo"]
        if (row["sequence"] != cell["id"] or measured["frames"] != 150 or len(raw) != 150 or
                [f["frame_index"] for f in raw] != list(range(150)) or
                measured["tracked_frames"] != sum(f["tracked"] is True for f in raw) or
                measured["estimator_receipt_sha256"] != hashes[prefix+"estimator_receipt.json"] or
                measured["trajectory_sha256"] != hashes[prefix+"trajectory.json"] or
                receipt["method"] != METHOD_NAMES[method] or receipt["input_manifest_sha256"] != hashes["replay/manifest.json"] or
                receipt["runtime_sha256"] != runtime["value"]["binary_sha256"] or
                receipt["runtime_receipt_sha256"] != runtime["receipt_sha256"] or
                receipt["upstream_commit"] != runtime["value"]["upstream_commit"]):
            raise ValueError("native runtime, frame count, or artifact receipt mismatch")
        expected_inputs = {f[key] for f in frames for key in (("lidar", "imu") if method == "fast_lio2" else ("left", "lidar", "imu"))}
        if set(receipt["input_files_sha256"]) != expected_inputs:
            raise ValueError("native inference input set differs from original sensor cohort")
        for name, expected in receipt["input_files_sha256"].items():
            if files.get(name) != expected or "truth" in name:
                raise ValueError("native estimator input lineage or truth boundary mismatch")
        if receipt.get("native_frames_sha256", hashes[prefix+"native_frames.jsonl"]) != hashes[prefix+"native_frames.jsonl"]:
            raise ValueError("native frame bytes changed")
        if method != "fast_lio2":
            count = sum(f["visual_points"] > 0 for f in raw)
            if measured["frames_with_visual_measurements"] != count or (method.endswith("no_visual") and count):
                raise ValueError("visual participation/ablation mismatch")
            if receipt["config"]["visual_enabled"] != (method == "fast_livo2"):
                raise ValueError("native camera ablation configuration mismatch")
        configs[method] = receipt.get("config_sha256") or sha(json.dumps(receipt["config"], sort_keys=True).encode())
        score = row["evaluation"]
        qualification[method] = score["replay_readiness_gate"]
        if qualification[method] not in ("PASS", "NEGATIVE"):
            raise ValueError("native scientific gate is incomplete")
        segments = score["segments"]
        if len(segments) != 1:
            raise ValueError("unexpected native map-reset segmentation")
        metrics = segments[0]["metrics"]
        measurements[f"native/{method}/ate_rmse_m"] = metrics["ate_translation_m"]["rmse"] if metrics else None
        measurements[f"native/{method}/tracking_fraction"] = measured["tracked_frames"]/150
        measurements[f"native/{method}/p95_ms"] = measured["native_compute_p95_ms"]
        measurements[f"native/{method}/qualified"] = float(qualification[method] == "PASS")
    return dict(cell=cell, measurements=measurements, metric_available_frames=available, qualification=qualification,
                native_config_sha256=configs, manifest_sha256=hashes["replay/manifest.json"], frame_count=150, native_cells=3)


def collect_job(job, protocol, intake, runtimes, mirrors):
    job = Path(job)
    if job.is_symlink():
        raise ValueError("linked job directory")
    completion, launch = SAFE.read_json(job/"completion.json"), SAFE.read_json(job/"launch.json")
    if (completion.get("status") != "PASS" or completion.get("exit_status") != 0 or completion.get("error") or
            not completion.get("finished_utc") or completion["archive_sha256"] != intake["archive_sha256"] or
            launch["archive_sha256"] != intake["archive_sha256"] or launch["source_files_verified"] != intake["source_files"]):
        raise ValueError("incomplete execution or frozen source mismatch")
    declared = [j for j in protocol["jobs"] if j["kind"] == "run" and j == launch["job"]]
    if len(declared) != 1 or job.name != declared[0]["name"]+"-"+str(launch["job_id"]):
        raise ValueError("undeclared or mismatched launch")
    if not {"launch.json", "runtime.log", "campaign_result.json", "outputs.tar.gz"} <= set(completion["files"]):
        raise ValueError("incomplete durable artifacts")
    for name, specification in completion["files"].items():
        if PurePosixPath(name).name != name:
            raise ValueError("unsafe completion filename")
        if name != "outputs.tar.gz":
            checked_file(job/name, specification)
    result = SAFE.read_json(job/"campaign_result.json")
    index = protocol["sensor_methods"]["cells"].index(result["cell"])
    for command in (declared[0]["args"], launch["command"]):
        if (command.count("--cell") != 1 or command[command.index("--cell")+1] != str(index) or
                command.count("--phase") != 1 or command[command.index("--phase")+1] != "run"):
            raise ValueError("launch command/cell identity mismatch")
    archive = find_archive(job, completion, intake["archive_sha256"], mirrors)
    records, hashes, total = archive_records(archive, declared[0].get("max_output_mb", 512)*1024**2)
    if total != completion["output_bytes"]:
        raise ValueError("raw output byte count mismatch")
    value = validate_payload(records, hashes, result, protocol, intake, runtimes)
    value["evidence"] = dict(job=str(job), archive=str(archive), archive_sha256=completion["files"]["outputs.tar.gz"]["sha256"],
                             completion_sha256=SAFE.digest(job/"completion.json"), source_archive_sha256=intake["archive_sha256"])
    return value


def paired_comparisons():
    comparisons = []
    for support in ("native_support", "common_support"):
        for metric in ("mae_m", "rmse_m", "coverage", "near_obstacle_pixel_recall"):
            comparisons.append((f"foundation_minus_sgbm/{support}/{metric}",
                f"depth/c_fast_foundationstereo/{support}/{metric}", f"depth/sgbm/{support}/{metric}"))
    comparisons.append(("foundation_minus_sgbm/p95_ms", "depth/c_fast_foundationstereo/p95_ms", "depth/sgbm/p95_ms"))
    for method in DEPTH:
        for metric in ("ground_height_rmse_m", "ground_coverage", "hazard_recall_including_unresolved", "false_traversable_ground_cells"):
            comparisons.append((f"gated_minus_simple/{method}/{metric}",
                f"map/{method}_gated_fusion/{metric}", f"map/{method}_simple_fusion/{metric}"))
    for baseline in ("fast_lio2", "fast_livo2_no_visual"):
        for metric in ("ate_rmse_m", "tracking_fraction", "p95_ms", "qualified"):
            comparisons.append((f"livo_minus_{baseline}/{metric}", f"native/fast_livo2/{metric}", f"native/{baseline}/{metric}"))
    return comparisons


def aggregate(rows, protocol, problems=()):
    cells = validate_protocol(protocol)
    problems = list(problems)
    identities = [r["cell"]["id"] for r in rows]
    if len(identities) != len(set(identities)):
        duplicates = {name for name in identities if identities.count(name) > 1}
        problems.append(dict(error="duplicate cell evidence; all copies quarantined", cells=sorted(duplicates)))
        rows = [r for r in rows if r["cell"]["id"] not in duplicates]
        identities = [r["cell"]["id"] for r in rows]
    expected = {c["id"] for c in cells}
    if not set(identities) <= expected or any(r["cell"] not in cells for r in rows):
        raise ValueError("unexpected cell evidence")
    if rows and any(r["native_config_sha256"] != rows[0]["native_config_sha256"] for r in rows):
        problems.append(dict(error="native configuration changed between cells"))
    complete = set(identities) == expected and not problems
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cell"]["geometry_group"]].append(row)
    group_means = {}
    for name, group in grouped.items():
        if len(group) == 3 and {r["cell"]["condition"] for r in group} == CONDITIONS:
            group_means[name] = {key: (mean_optional([r["measurements"][key] for r in group])
                                if all(r["measurements"][key] is not None for r in group) else None)
                                for key in group[0]["measurements"]}
    comparisons = {}
    if complete:
        for name, left, right in paired_comparisons():
            values = [g[left]-g[right] for g in group_means.values() if g[left] is not None and g[right] is not None]
            record = dict(groups_with_both_metrics=len(values), declared_groups=6,
                          direction="signed first-minus-second difference; interpret each metric separately")
            if len(values) == 6:
                array = np.asarray(values)
                rng = np.random.default_rng(20261010)
                draws = array[rng.integers(0, 6, size=(2000, 6))].mean(axis=1)
                record.update(status="DESCRIPTIVE_DEVELOPMENT_ESTIMATE", mean_difference=float(array.mean()),
                              lower_95=float(np.quantile(draws, .025)), upper_95=float(np.quantile(draws, .975)),
                              bootstrap_seed=20261010, samples=2000)
            else:
                record.update(status="UNAVAILABLE_METRIC_GROUPS", mean_difference=None, lower_95=None, upper_95=None)
            comparisons[name] = record
    return dict(schema="bhl-sensor-methods-collection-v1", status="PASS" if complete else "INCOMPLETE",
        scientific_status="COMPLETE_DEVELOPMENT_COMPARISON_NO_CONFIRMATION" if complete else "PARTIAL_EVIDENCE_NO_GLOBAL_VERDICT",
        observed_cells=len(rows), expected_cells=18, observed_camera_frames=sum(r["frame_count"] for r in rows),
        expected_camera_frames=2700, observed_native_cells=sum(r["native_cells"] for r in rows), expected_native_cells=54,
        complete_geometry_groups=len(group_means), expected_geometry_groups=6,
        missing_cells=sorted(expected-set(identities)), problems=list(problems), cells=rows,
        observed_native_qualification={method: dict(qualified=sum(r["qualification"][method] == "PASS" for r in rows),
            evaluated=len(rows)) for method in NATIVE}, group_means=group_means, paired_comparisons=comparisons,
        analysis_unit="geometry variant; equal mean of all three conditions within each group; group bootstrap only, never frame bootstrap",
        metric_summary="equal frame means within each cell; missing metric denominators retained; latency is mean cell p95, not pooled p95",
        caveats=["six development geometry variants from previously used families; descriptive uncertainty, no unseen-family confirmation",
                 "conditional pose/depth error must be read with tracking/coverage",
                 "unknown hazards remain unresolved; pixel recall is not obstacle instance detection",
                 "no closed-loop navigation, falls, contacts or physical-hardware outcomes are measured"])


def collect(campaign, job_dirs, mirror_dirs=()):
    campaign = Path(campaign)
    intake = SAFE.read_json(campaign/"intake.json")
    protocol = SAFE.read_json(campaign/"protocol.json")
    validate_protocol(protocol)
    manifest, _ = SAFE.verify_frozen_archive(campaign, intake)
    for name, specification in protocol["input_files"].items():
        if manifest["inputs/"+name]["sha256"] != specification["sha256"]:
            raise ValueError("frozen input differs from predeclared pin")
    runtimes = runtime_receipts(campaign, protocol)
    rows, problems = [], []
    for job in job_dirs:
        try:
            rows.append(collect_job(Path(job), protocol, intake, runtimes, mirror_dirs))
        except (ValueError, OSError, KeyError, IndexError, TypeError, tarfile.TarError) as error:
            problems.append(dict(job=str(job), error=f"{type(error).__name__}: {error}"))
    result = aggregate(rows, protocol, problems)
    result.update(source_archive_sha256=intake["archive_sha256"], protocol_sha256=intake["protocol_sha256"],
                  input_sha256={k:v["sha256"] for k,v in protocol["input_files"].items()},
                  collector_sha256=SAFE.digest(__file__))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--campaign-dir", required=True, type=Path)
    p.add_argument("--job-dirs", nargs="*", type=Path, default=[])
    p.add_argument("--mirror-dir", action="append", type=Path, default=[])
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    result = collect(args.campaign_dir, args.job_dirs, args.mirror_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:result[k] for k in ("status", "observed_cells", "expected_cells", "observed_native_cells", "missing_cells")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
