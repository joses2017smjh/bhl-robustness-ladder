#!/usr/bin/env python3
"""Audit the fixed nine-episode development cohort from evaluator poses."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np


def audit(job_root,output):
    job_root,output=Path(job_root),Path(output)
    completion=json.loads((job_root/"completion.json").read_text());archive_path=job_root/"outputs.tar.gz"
    with archive_path.open("rb") as stream:actual_sha=hashlib.file_digest(stream,"sha256").hexdigest()
    if actual_sha!=completion["files"]["outputs.tar.gz"]["sha256"]:raise ValueError("raw archive hash differs")
    episodes={};truths={}
    with tarfile.open(archive_path,"r|gz") as archive:
        for member in archive:
            prefix=str(Path(member.name).parent)
            if member.name.endswith("/episode.json"):episodes[prefix]=json.load(archive.extractfile(member))
            elif member.name.endswith("/evaluator_truth.npz"):
                data=archive.extractfile(member).read()
                with np.load(io.BytesIO(data),allow_pickle=False) as truth:
                    truths[prefix]=(hashlib.sha256(data).hexdigest(),truth["timestamp_s"].copy(),truth["T_W_I"].copy())
    results=[]
    expected={(case,seed) for case in ("straight","dogleg","occluders") for seed in range(380000,380003)}
    observed=[(row["case"],row["seed"]) for row in episodes.values()]
    if len(observed)!=9 or set(observed)!=expected or set(episodes)!=set(truths):
        raise ValueError("complete declared nine-episode development cohort required")
    for prefix,row in episodes.items():
        truth_sha,times,poses=truths[prefix]
        if truth_sha!=row["evaluator_truth_sha256"]:raise ValueError("independent evaluator matrix hash differs")
        trace=row["trace"]
        if len(times)!=len(trace):raise ValueError("trace/evaluator lengths differ")
        np.testing.assert_allclose(times,[t["timestamp_s"] for t in trace],rtol=0,atol=1e-9)
        body=poses@np.linalg.inv(np.asarray(row["T_body_imu"]))
        positions=body[:,:3,3]
        distance=np.linalg.norm(positions[:,:2]-np.asarray(row["route_external"][-1]),axis=1)
        np.testing.assert_allclose(distance,[t["evaluator_goal_distance_m"] for t in trace],rtol=0,atol=1e-8)
        np.testing.assert_allclose(positions[:,:2],[t["evaluator_xy"] for t in trace],rtol=0,atol=1e-8)
        tilt=np.arccos(np.clip(body[:,2,2],-1,1))
        np.testing.assert_allclose(tilt,[t["evaluator_tilt_rad"] for t in trace],rtol=0,atol=1e-7)
        reached=bool(np.any(distance<=.30));fell=bool(np.any(tilt>=.78) or np.any(positions[:,2]<row["known_start_pose"][2][3]-.25))
        contact=bool(any(t["evaluator_contact"] for t in trace))
        finite=bool(np.isfinite(body).all() and np.isfinite(np.asarray([t["command"] for t in trace])).all())
        # MultiRunner.reset preserves qpos0 translation exactly; only joints
        # are perturbed. This makes the declared start height the actual reset
        # height, without adding any pose feedback to the controller.
        full=bool(len(times)==1000 and abs(times[-1]-40.)<1e-6)
        success=bool(reached and not(fell or contact) and finite and full)
        if success!=row["success"] or fell!=row["fell"] or contact!=row["collision"] or reached!=row["goal_reached_evaluator"]:
            raise ValueError("reconstructed navigation outcome differs")
        results.append({"case":row["case"],"seed":row["seed"],"success":success,"full_horizon":full,
            "reached":reached,"fell":fell,"contact_trace":contact,"finite_truth_and_commands":finite,
            "truth_frames":len(times),"minimum_goal_distance_m":float(distance.min()),"final_goal_distance_m":float(distance[-1]),
            "maximum_tilt_rad":float(tilt.max()),"minimum_base_height_m":float(positions[:,2].min()),
            "declared_reset_base_height_m":row["known_start_pose"][2][3],"evaluator_truth_sha256":truth_sha})
    report={"schema":"bhl-native-navigation-outcome-audit-v1","status":"PASS","raw_archive_sha256":actual_sha,
        "episodes":results,"clean_goals":sum(r["success"] for r in results),
        "scope":"DEVELOPMENT_ONLY_NO_CONFIRMATION. Offline reconstruction from separate evaluator matrices at policy ticks; goal distance, base tilt/height and full horizon recomputed; contact flags crosschecked from retained physics traces; no independent reconstruction of between-tick contacts/joint state, resimulation or estimator/controller feedback"}
    with output.open("x") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--job-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=audit(a.job_root,a.output);print(json.dumps({"status":r["status"],"clean_goals":r["clean_goals"],"episodes":len(r["episodes"])}))
