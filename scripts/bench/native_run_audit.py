"""Independent raw terrain/stereo outcome audits for the dated run publisher."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tarfile

import numpy as np


def verify_retained_payloads(campaign,intake,expected_manifest_sha256=None):
    """Verify every saved frozen payload while distinguishing retired gzip bytes."""
    campaign=Path(campaign)
    inventory_path=campaign/"retained-member-inventory.json"
    receipt=json.loads((campaign/"successful-packaging-reclamation.json").read_text())
    inventory=json.loads(inventory_path.read_text())
    def digest(path):
        with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()
    if receipt["inventory_sha256"]!=digest(inventory_path):raise ValueError("retained frozen inventory hash differs")
    if inventory["original_archive_sha256"]!=intake["archive_sha256"] or receipt["removed_archive_sha256"]!=intake["archive_sha256"]:raise ValueError("retired source lineage differs")
    manifest_path=campaign/"manifest.json";manifest=json.loads(manifest_path.read_text())
    pinned_manifest=intake.get("source_and_input_manifest_sha256",intake.get("source_manifest_sha256_unchanged",expected_manifest_sha256))
    if expected_manifest_sha256 and pinned_manifest!=expected_manifest_sha256:raise ValueError("frozen publisher manifest pin differs from immutable intake")
    if not pinned_manifest or digest(manifest_path)!=inventory["original_manifest_sha256"] or digest(manifest_path)!=pinned_manifest:raise ValueError("original frozen manifest hash differs from immutable intake")
    if receipt["retained_source_archive_sha256"]!=inventory["retained_source_archive_sha256"]:raise ValueError("retained source archive lineage differs")
    members=inventory["members"]
    if set(members)!=set(manifest)|{"protocol.json","manifest.json"}:raise ValueError("retained frozen payload inventory differs")
    by_archive=defaultdict(list)
    for name,row in members.items():
        if name in manifest and {"bytes":row["bytes"],"sha256":row["sha256"]}!=manifest[name]:raise ValueError("retained member differs from original manifest")
        reconstruction=row["reconstruction"];path=Path(reconstruction["path"])
        if reconstruction["kind"]=="surviving_exact_file":
            if path.is_symlink() or path.stat().st_size!=row["bytes"] or digest(path)!=row["sha256"]:raise ValueError("surviving exact frozen input differs")
        elif reconstruction["kind"]=="retained_source_archive_member":by_archive[path].append((name,row,reconstruction["member"]))
        else:raise ValueError("unknown frozen-payload reconstruction kind")
    for path,rows in by_archive.items():
        if path.is_symlink() or path.stat().st_size!=inventory["retained_source_archive_bytes"] or digest(path)!=inventory["retained_source_archive_sha256"]:raise ValueError("retained frozen-source archive hash differs")
        expected={member:(name,row) for name,row,member in rows}
        if len(expected)!=len(rows):raise ValueError("duplicate retained member reference")
        with tarfile.open(path) as archive:
            saved=archive.getmembers()
            if len({m.name for m in saved})!=len(saved) or not set(expected)<={m.name for m in saved} or any(not m.isfile() for m in saved):raise ValueError("retained source archive inventory differs")
            for member in saved:
                if member.name not in expected:continue
                _,row=expected[member.name]
                digestor=hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b""):digestor.update(chunk)
                if member.size!=row["bytes"] or digestor.hexdigest()!=row["sha256"]:raise ValueError("retained frozen source payload checksum differs")
    for name in ("protocol.json","manifest.json"):
        if digest(campaign/name)!=members[name]["sha256"]:raise ValueError("canonical frozen receipt differs from retained payload")
    return {"original_archive_rehashed":False,"original_compressed_stream":"retired; exact original digest retained as lineage",
        "logical_frozen_payloads_verified":len(members),"original_archive_sha256":intake["archive_sha256"],
        "original_manifest_sha256":inventory["original_manifest_sha256"],"retained_inventory_sha256":digest(inventory_path),
        "retained_source_archive_sha256":inventory["retained_source_archive_sha256"]}


def load_json(archive, name):
    return json.load(archive.extractfile(name))


def save(output, result, lines):
    output.mkdir(parents=True, exist_ok=False)
    (output/"observer.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (output/"report.md").write_text("\n".join(lines)+"\n")
    return result


def terrain(job_root, output):
    outer=json.loads((job_root/"campaign_result.json").read_text())
    episodes=[];qualified=[]
    with tarfile.open(job_root/"outputs.tar.gz") as archive:
        if load_json(archive,"campaign_result.json")!=outer:raise ValueError("terrain outer/raw result differs")
        for stage in outer["stages"]:
            child=load_json(archive,stage["result"])
            parent=str(Path(stage["result"]).parent)
            current=[]
            for name in child.get("episode_files",[]):
                row=load_json(archive,parent+"/"+name);trace=row["trace"]
                collision=any(t["wall_contact"] for t in trace)
                fell=any(t["tilt_evaluator_rad"]>=.78 or t["local_sink_evaluator_m"]>.25 for t in trace)
                crossed=any(t["progress_evaluator_m"]>=5. and t["goal_crossed"] for t in trace)
                success=bool(crossed and row["elapsed_s"]>=40.-1e-6 and not fell and not collision
                             and not row["nonfinite"] and row["forward_progress_m"] is not None)
                if (success,row["fell"],row["collision"],row["goal_crossed"])!=(row["success"],fell,collision,crossed):
                    raise ValueError("terrain outcome differs from actual command/evaluator trace")
                raw=archive.extractfile(parent+"/"+row["raw_scan_file"]).read()
                if hashlib.sha256(raw).hexdigest()!=row["raw_scan_sha256"]:raise ValueError("terrain raw scan hash mismatch")
                squared=0.;cells=0;truth_cells=0
                with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:
                    if len(arrays["timestamp_s"])!=row["raw_scan_count"]:raise ValueError("terrain scan count differs")
                    if int(arrays["raw_offsets"][-1])!=row["raw_point_count"]:raise ValueError("terrain raw point count differs")
                    height=arrays["inferred_maps"][:,0];truth=arrays["evaluator_truth_height_m"]
                    if height.shape!=truth.shape or len(height)!=len(row["terrain_map_metrics"]):raise ValueError("terrain map shapes differ")
                    for estimated,reference,metrics in zip(height,truth,row["terrain_map_metrics"]):
                        valid=np.isfinite(reference)&np.isfinite(estimated);errors=estimated[valid].astype(float)-reference[valid]
                        count=int(valid.sum());rmse=float(np.sqrt(np.square(errors).mean())) if count else None
                        if count!=metrics["observed_height_cells"] or (rmse is None)!=(metrics["height_rmse_m"] is None):
                            raise ValueError("terrain coverage differs from retained raw maps")
                        if rmse is not None and not math.isclose(rmse,metrics["height_rmse_m"],rel_tol=1e-4,abs_tol=2e-6):
                            raise ValueError("terrain RMSE differs from retained raw maps")
                        squared+=float(np.square(errors).sum());cells+=count;truth_cells+=int(np.isfinite(reference).sum())
                summary={k:row[k] for k in ("actor","terrain","group","arm","success","fell","collision","nonfinite",
                    "failure","elapsed_s","forward_progress_m","raw_scan_count","raw_point_count")}
                summary.update(stage=stage["name"],height_observed_cells=cells,height_truth_cells=truth_cells,
                    height_rmse_m=math.sqrt(squared/cells) if cells else None)
                episodes.append(summary);current.append(summary)
            if len(current)!=child["episodes"]:raise ValueError("terrain stage count differs")
            if stage["name"]=="screen":
                by_actor=defaultdict(list)
                for row in current:by_actor[row["actor"]].append(row)
                for actor,rows in by_actor.items():
                    identities={(r["terrain"],r["group"]) for r in rows}
                    if len(rows)==6 and len(identities)==6 and {r["terrain"] for r in rows}=={"flat","small_steps","ramp"} and len({r["group"] for r in rows})==2 and all(r["success"] for r in rows):qualified.append(actor)
                if sorted(qualified)!=sorted(outer["qualified_actors"]):raise ValueError("terrain qualification differs from six physical outcomes")
            if stage["name"]=="confirmation" and not qualified and current:raise ValueError("confirmation ran without a qualified actor")
    status="SMOKE_ONLY" if outer["phase"]=="smoke" else outer["status"]
    result={"audit_status":"PASS","scientific_status":status,"qualified_actors":qualified,"episodes":episodes,
        "scope":"actual learned-biped simulation; ideal 3D rays/IMU; no SLAM, stereo fusion or hardware trials",
        "height_scope":"all-scene dense vertical-ray geometry, including wall tops; not ground-only surface accuracy",
        "smoke_nested_qualified_actors_ignored":True}
    lines=["# Actual sensor-informed terrain traversal","","Scientific status: **"+status+"**.","",
        "Stopping short is a timeout even without falls. Qualification and confirmation are reported separately; shared actors/groups are paired clusters. All sensors and physics are simulated.","",
        "|Stage|Actor|Terrain|Group|Arm|Goal|Fall|Contact|Progress(m)|Height RMSE(m)|","|---|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in episodes:
        progress=row["forward_progress_m"];rmse=row["height_rmse_m"]
        lines.append("|"+"|".join([row["stage"],row["actor"],row["terrain"],str(row["group"]),row["arm"],str(int(row["success"])),str(int(row["fell"])),str(int(row["collision"])),f"{progress:.3f}" if progress is not None else "unknown",f"{rmse:.4f}" if rmse is not None else "unobserved"])+"|")
    lines.extend(["","Height error compares raw inferred maps with independent dense scene geometry after commands were computed. Wall-top/side aliasing is included; these values are not ground-only accuracy. Smoke never qualifies actors."])
    return save(output,result,lines)


def stereo(job_root,output,*,expected_binary_sha256):
    spec=importlib.util.spec_from_file_location("shared_native_pose_observer",Path(__file__).with_name("native_navigation_observe.py"))
    observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
    outer=json.loads((job_root/"campaign_result.json").read_text());results=[]
    with tarfile.open(job_root/"outputs.tar.gz") as archive:
        if load_json(archive,"campaign_result.json")!=outer:raise ValueError("stereo outer/raw result differs")
        if outer["method"]!="orb_slam3_stereo_pose_plus_lidar_obstacle_brake":raise ValueError("actual stereo-pose-plus-LiDAR campaign required")
        for name in outer["episode_files"]:
            row=load_json(archive,name);prefix=str(Path(name).parent)
            stream=load_json(archive,prefix+"/native/stream_receipt.json")
            if (stream["schema"]!="bhl-native-orb-navigation-stream-v1" or stream["ground_truth_inputs"]!=[]
                    or stream["binary_sha256"]!=expected_binary_sha256 or not stream["no_lidar_or_imu_to_stereo_estimator"]):
                raise ValueError("stereo estimator input/runtime receipt differs")
            native=[json.loads(line) for line in archive.extractfile(prefix+"/native/native_frames.jsonl")]
            manifest=load_json(archive,prefix+"/stereo/image_manifest.json")
            images={}
            for image in manifest["images"]:
                images[float(image["timestamp_s"])]=image
                for side in ("left","right"):
                    path=prefix+"/stereo/"+Path(image[side]).name
                    if hashlib.sha256(archive.extractfile(path).read()).hexdigest()!=image[side+"_sha256"]:
                        raise ValueError("original stereo image hash mismatch")
            body_lidar=np.eye(4);body_lidar[:3,3]=[.10,0,.38]
            camera_imu=np.linalg.inv(np.asarray(stream["T_B_C_left"]))@body_lidar@np.linalg.inv(np.asarray(stream["T_I_L"]))
            first_map=None;changed=False
            for frame in native:
                image=images[float(frame["timestamp_s"])]
                for side in ("left","right"):
                    if frame["original_"+side+"_sha256"]!=image[side+"_sha256"]:raise ValueError("native input differs from original stereo pair")
                if frame["native_ORB_tracking_state"]==2 and frame["native_ORB_map_id"] is not None:
                    if first_map is None:first_map=frame["native_ORB_map_id"]
                    elif first_map!=frame["native_ORB_map_id"]:changed=True
                accepted=frame["native_ORB_tracking_state"]==2 and frame["native_ORB_map_id"] is not None and not changed
                if accepted!=frame["tracked"] or frame["map_reset_id"]!=int(changed):raise ValueError("native map/loss stop policy differs")
                if accepted and not np.allclose(np.asarray(frame["native_ORB_T_W_C"])@camera_imu,frame["T_W_I"],rtol=0,atol=1e-9):
                    raise ValueError("native camera/IMU calibration differs")
            truth_raw=archive.extractfile(prefix+"/evaluator_truth.npz").read()
            if hashlib.sha256(truth_raw).hexdigest()!=row["evaluator_truth_sha256"]:raise ValueError("stereo evaluator truth hash differs")
            if hashlib.sha256(archive.extractfile(prefix+"/inference_sensors.npz").read()).hexdigest()!=row["inference_sensors_sha256"]:
                raise ValueError("stereo raw LiDAR/IMU hash differs")
            with np.load(io.BytesIO(truth_raw),allow_pickle=False) as truth:
                result=observer.score_episode(row,native,truth["timestamp_s"],truth["T_W_I"])
            result.update(original_stereo_pairs=len(manifest["images"]),map_changed=changed,
                capture_compute_seconds=manifest["total_capture_wall_seconds"],
                unconsumed_capture_seconds=manifest["unbilled_unconsumed_capture_seconds_at_horizon"])
            results.append(result)
    if len(results)!=outer["episodes"] or sum(r["success"] for r in results)!=outer["goals"]:raise ValueError("stereo campaign outcome count differs")
    status="SMOKE_ONLY" if outer["phase"]=="smoke" else outer["status"]
    result={"audit_status":"PASS","scientific_status":status,"episodes":results,
        "scope":"actual native ORB stereo pose plus raw LiDAR obstacle brake and frozen 12-DoF learned gait; simulated body-attached images, known start/external routes; no hardware claim"}
    lines=["# Actual native stereo estimated-pose navigation","","Scientific status: **"+status+"**.","",
        result["scope"]+". Smoke outcomes are excluded from qualification.","",
        "|Case|Group|Goal|Fall|Contact|Original stereo pairs|Post-init tracking|Client p95(ms)|","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        tracking=row["post_initialization_tracking_fraction"];latency=row["client_wall_p95_ms"]
        lines.append("|"+"|".join([row["case"],str(row["seed"]),str(int(row["success"])),str(int(row["fell"])),str(int(row["collision"])),str(row["original_stereo_pairs"]),f"{tracking:.3f}" if tracking is not None else "unknown",f"{latency:.2f}" if latency is not None else "unknown"])+"|")
    lines.extend(["","Original PNG hashes, camera-to-IMU transforms, native map IDs and causal response arrivals are independently checked. Drift scoring uses evaluator-only gravity/yaw/translation alignment with fixed metric scale1; no evaluator correction enters navigation."])
    return save(output,result,lines)
