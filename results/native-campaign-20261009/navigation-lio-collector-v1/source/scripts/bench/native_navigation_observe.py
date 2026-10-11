#!/usr/bin/env python3
"""Independently audit completed native LIO navigation, without rerunning it."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from bhl_robust.research.pose_metrics import Trajectory,score_trajectories,transform
from bhl_robust.research.native_lio import UPSTREAM_COMMIT

SPEC=importlib.util.spec_from_file_location("navigation_reference_interpolation",Path(__file__).with_name("native_slam_campaign.py"))
REFERENCE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(REFERENCE)


def sha256(value):return hashlib.sha256(value).hexdigest()


def score_episode(row,native,truth_times,truth_poses):
    """Truth is opened only by this offline observer, never fed to navigation."""
    if len(native)!=row["native_frames"]:raise ValueError("native frame count differs from episode")
    stamps=[];poses=[];tracked=[];reference_times=[];reference_poses=[]
    for frame in native:
        stamp=float(frame["timestamp_s"]);tracking=frame["tracked"]
        if type(tracking) is not bool or (not tracking and frame["T_W_I"] is not None):
            raise ValueError("native lost frame must retain explicit null pose")
        stamps.append(stamp);tracked.append(tracking)
        # Trajectory's dense storage uses identity only with tracked=False;
        # null native source remains unchanged and excluded from metric scoring.
        poses.append(transform(frame["T_W_I"]) if tracking else np.eye(4))
        reference=REFERENCE.interpolate_truth(truth_times,truth_poses,stamp)
        if reference is not None:reference_times.append(stamp);reference_poses.append(reference)
    post=[frame for frame in native if frame["timestamp_s"]>=5.]
    result={"case":row["case"],"seed":row["seed"],"success":row["success"],
        "fell":row["fell"],"collision":row["collision"],"nonfinite":row["nonfinite"],
        "native_frames":len(native),"native_tracked_frames":sum(tracked),
        "post_initialization_tracking_fraction":sum(f["tracked"] for f in post)/len(post) if post else None,
        "client_wall_p95_ms":row["client_wall_p95_ms"],"dropped_scans":row["dropped_scans"],
        "final_true_goal_distance_m":row["trace"][-1]["evaluator_goal_distance_m"] if row["trace"] else None,
        "command_reason_steps":dict(Counter(t["reason"] for t in row["trace"])),
        "metrics":None,"metric_pose_origin":"IMU; evaluator-only gravity/yaw/translation alignment, fixed metric scale1"}
    try:
        if any(f["map_reset_id"]!=0 for f in native):raise ValueError("native map reset needs separate segment scoring")
        result["metrics"]=score_trajectories(
            Trajectory(np.asarray(stamps),np.asarray(poses),np.asarray(tracked,bool)),
            Trajectory(np.asarray(reference_times),np.asarray(reference_poses)),
            alignment="gravity_yaw_translation",max_difference_s=1e-8,end_time_s=row["elapsed_simulation_s"])
    except ValueError as error:result["unscorable_reason"]=str(error)
    # Causal delivery is independently checked against retained trace metadata.
    for trace in row["trace"]:
        consumption=trace.get("native_last_consumed_at_s");arrival=trace.get("native_available_at_s")
        if arrival is not None and arrival>trace["timestamp_s"]+1e-8:raise ValueError("native estimate became visible before arrival")
        if consumption is not None and consumption>trace["timestamp_s"]+1e-8:
            raise ValueError("controller consumption occurs in the future")
    return result


def observe(job_root,output):
    job_root,output=Path(job_root),Path(output)
    completion=json.loads((job_root/"completion.json").read_text())
    if completion["status"] not in ("PASS","NEGATIVE"):raise ValueError("completed native run required")
    for name,expected in completion["files"].items():
        path=job_root/name
        if path.stat().st_size!=expected["bytes"] or sha256(path.read_bytes())!=expected["sha256"]:
            raise ValueError("completion artifact hash mismatch: "+name)
    results=[];plot_rows=[]
    with tarfile.open(job_root/"outputs.tar.gz") as archive:
        campaign=json.load(archive.extractfile("campaign_result.json"))
        expected_binary=[value for name,value in campaign["input_sha256"].items() if name.endswith("/fastlio_headless")]
        if campaign["method"]!="fast_lio2_lidar_imu" or len(expected_binary)!=1:raise ValueError("audited native LIO campaign required")
        for name in campaign["episode_files"]:
            row=json.load(archive.extractfile(name));prefix=str(Path(name).parent)
            stream=json.load(archive.extractfile(prefix+"/native/stream_receipt.json"))
            if stream["status"]!="PASS" or stream["ground_truth_inputs"]:raise ValueError("valid truth-free native stream required")
            if stream["upstream_commit"]!=UPSTREAM_COMMIT or stream["runtime_binary_sha256"]!=expected_binary[0]:
                raise ValueError("native estimator differs from frozen runtime")
            for raw_name in ("input.bin","native_frames.jsonl"):
                if sha256(archive.extractfile(prefix+"/native/"+raw_name).read())!=stream[raw_name+"_sha256"]:
                    raise ValueError("native stream artifact hash mismatch")
            native=[json.loads(line) for line in archive.extractfile(prefix+"/native/native_frames.jsonl")]
            truth_bytes=archive.extractfile(prefix+"/evaluator_truth.npz").read()
            if sha256(truth_bytes)!=row["evaluator_truth_sha256"]:raise ValueError("evaluator truth hash mismatch")
            if sha256(archive.extractfile(prefix+"/inference_sensors.npz").read())!=row["inference_sensors_sha256"]:
                raise ValueError("raw causal sensor hash mismatch")
            with np.load(io.BytesIO(truth_bytes),allow_pickle=False) as truth:
                results.append(score_episode(row,native,truth["timestamp_s"],truth["T_W_I"]))
            plot_rows.append((row,native))
    if len(results)!=campaign["episodes"] or sum(r["success"] for r in results)!=campaign["goals"]:
        raise ValueError("whole campaign outcome count mismatch")
    result={"schema":"bhl-native-navigation-observer-v1","status":"PASS","campaign_phase":campaign["phase"],
            "scientific_status":"SMOKE_ONLY" if campaign["phase"]=="smoke" else campaign["status"],
            "source_runner_status":campaign["status"],"episodes":results,"ground_truth_use":"offline scoring only; no estimator/planner feedback",
            "source_archive_sha256":completion["files"]["outputs.tar.gz"]["sha256"],
            "scope":"Actual LIO learned-gait simulation; fixed prescribed routes, ideal simulated sensors, one frozen actor; smoke excluded from qualification"}
    output.mkdir(parents=True,exist_ok=False)
    (output/"observer.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    lines=["# Actual native LIO navigation", "", "Scientific status: **"+result["scientific_status"]+"**.", "",
        "Smoke outcomes are excluded from qualification. Development uses one frozen learned gait, scripted external waypoints and ideal simulated sensors; these are not physical-robot results.", "",
        "|Case|Seed|Clean goal|Fall|Contact|Post-init tracking|ATE RMSE(m)|Client p95(ms)|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        tracking=row["post_initialization_tracking_fraction"];metrics=row["metrics"]
        lines.append("|"+"|".join([row["case"],str(row["seed"]),str(int(row["success"])),str(int(row["fell"])),str(int(row["collision"])),
            f"{tracking:.3f}" if tracking is not None else "unknown",f"{metrics['ate_translation_m']['rmse']:.4f}" if metrics else "unscorable",
            f"{row['client_wall_p95_ms']:.2f}" if row["client_wall_p95_ms"] is not None else "unknown"])+"|")
    lines.extend(["", "ATE aligns native IMU poses to independent truth with gravity-preserving yaw/translation and fixed scale1. This evaluator alignment never enters the controller.", "", "Raw archive SHA256: `"+result["source_archive_sha256"]+"`.", ""])
    (output/"report.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    present=[case for case in ("straight","dogleg","occluders") if any(row["case"]==case for row,_ in plot_rows)]
    fig,axes=plt.subplots(1,len(present),figsize=(4*len(present),4),squeeze=False)
    for ax,case in zip(axes[0],present):
        for row,native in plot_rows:
            if row["case"]!=case:continue
            xy=np.asarray([t["evaluator_xy"] for t in row["trace"]]);route=np.asarray(row["route_external"])
            line,=ax.plot(xy[:,0],xy[:,1],label=f"seed{row['seed']} true")
            if row["initial_registration"] is not None:
                registration=transform(row["initial_registration"]);t_imu_body=np.linalg.inv(transform(row["T_body_imu"]))
                estimates=np.asarray([(registration@transform(f["T_W_I"])@t_imu_body)[:2,3] for f in native if f["tracked"]])
                if len(estimates):ax.plot(estimates[:,0],estimates[:,1],"--",color=line.get_color(),label=f"seed{row['seed']} native")
            ax.scatter(route[:,0],route[:,1],marker="x",c="black")
        ax.set_title(case);ax.set_aspect("equal",adjustable="datalim");ax.set_xlabel("world x(m)");ax.set_ylabel("world y(m)");ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    outcome="smoke only; outcomes excluded" if campaign["phase"]=="smoke" else f"{campaign['goals']}/{campaign['episodes']} clean goals"
    fig.suptitle(f"Actual native LIO closed loop — {outcome}")
    fig.tight_layout();fig.savefig(output/"actual-navigation-paths.png",dpi=150);plt.close(fig)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--job-root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();result=observe(args.job_root,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!="episodes"}),flush=True)
