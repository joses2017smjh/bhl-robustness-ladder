#!/usr/bin/env python3
"""Run frozen native replay inside an existing Slurm CPU allocation step.

This utility never submits or edits a batch job. The caller must cancel any
pending batch twin before launch. Physics, rendering and training entrypoints
are deliberately unsupported; this route runs native replay and its evaluator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


def now():return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def write_json(path,value):
    with Path(path).open("x") as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write("\n")


def allocation_resources(text,allocation,uid):
    """Validate actual owner/running CPU+RAM capacity, retaining original text."""
    fields=dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)",text))
    owner=re.search(r"\((\d+)\)$",fields.get("UserId",""))
    if fields.get("JobId")!=str(allocation) or fields.get("JobState")!="RUNNING" or not owner or int(owner[1])!=uid:
        raise ValueError("existing running allocation owned by the current user required")
    cpus=int(fields.get("NumCPUs","0"))
    memory=re.search(r"(?:^|,)mem=([0-9.]+)([KMGT]?)",fields.get("AllocTRES",""))
    if not memory:raise ValueError("actual allocated memory must be recorded")
    mib=float(memory[1])*{"":1,"K":1/1024,"M":1,"G":1024,"T":1024**2}[memory[2]]
    if cpus<2 or mib<16*1024:raise ValueError("native step requires at least2allocatedCPUs and16GiB")
    return {"job_id":str(allocation),"cpus":cpus,"memory_mib":mib,"node_list":fields.get("NodeList"),
            "partition":fields.get("Partition"),"end_time":fields.get("EndTime"),"scontrol_oneliner":text.strip()}


def validate_smoke_step(campaign,intake,protocol,receipt_path):
    campaign=Path(campaign).resolve();receipt_path=Path(receipt_path).resolve()
    if receipt_path.parent!=campaign:raise ValueError("fresh smoke step receipt must belong to this frozen campaign")
    receipt=json.loads(receipt_path.read_text())
    smoke_names={j["name"] for j in protocol["jobs"] if j["kind"]=="smoke"}
    if receipt.get("status")!="ALLOCATION_STEP_COMPLETE" or receipt.get("returncode")!=0 or receipt.get("job_name") not in smoke_names:
        raise ValueError("successful actual fresh smoke allocation step required")
    if receipt.get("archive_sha256")!=intake["archive_sha256"] or not re.fullmatch(r"\d+",str(receipt.get("job_id",""))):
        raise ValueError("fresh smoke source/allocation pin differs")
    environment_path=Path(receipt.get("step_environment_path","")).resolve()
    if environment_path.parent!=campaign or digest(environment_path)!=receipt.get("step_environment_sha256"):
        raise ValueError("fresh smoke actual step environment hash differs")
    environment=json.loads(environment_path.read_text())
    if not re.fullmatch(r"\d+",str(receipt.get("slurm_step_id",""))) or environment.get("SLURM_STEP_ID")!=receipt["slurm_step_id"] or environment.get("SLURM_JOB_ID")!=receipt["job_id"]:
        raise ValueError("fresh smoke must retain its actual Slurm step identity")
    completion_path=campaign/(receipt["job_name"]+"-"+receipt["job_id"])/"completion.json"
    completion=json.loads(completion_path.read_text())
    if completion.get("status")!="PASS" or completion.get("exit_status")!=0 or completion.get("archive_sha256")!=intake["archive_sha256"]:
        raise ValueError("same-archive fresh smoke completion must PASS")
    if digest(completion_path)!=receipt.get("completion_sha256"):
        raise ValueError("fresh smoke completion hash differs")
    return {"path":str(receipt_path),"sha256":digest(receipt_path),"job_name":receipt["job_name"],
            "allocation_job_id":receipt["job_id"],"slurm_step_id":receipt["slurm_step_id"]}


# All caller-controlled values are passed as argv, never interpolated into shell.
STEP_WRAPPER=r'''set -euo pipefail
snapshot=$1
python=$2
launcher=$3
campaign=$4
job=$5
pin=$6
"$python" - "$snapshot" <<'PY'
import datetime,json,os,pathlib,socket,sys
data={key:os.environ.get(key) for key in ("SLURM_JOB_ID","SLURM_STEP_ID","SLURM_STEP_NODELIST","SLURM_CPUS_PER_TASK","SLURM_NTASKS","CUDA_VISIBLE_DEVICES")}
data.update(host=socket.gethostname(),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
if not data["SLURM_JOB_ID"] or not data["SLURM_STEP_ID"]:raise SystemExit("actual Slurm step environment required")
with pathlib.Path(sys.argv[1]).open("x") as stream:json.dump(data,stream,indent=2,sort_keys=True);stream.write("\n")
PY
exec /bin/bash --noprofile --norc "$launcher" "$campaign" "$job" "$pin"
'''


def run_step(campaign_dir,job_name,allocation="21739893",fresh_smoke=None):
    if not re.fullmatch(r"\d+",str(allocation)):raise ValueError("numeric existing allocation required")
    spec=importlib.util.spec_from_file_location("native_step_frozen_intake",Path(__file__).with_name("h34_campaign.py"))
    launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
    campaign,intake,protocol=launch.load_intake(campaign_dir)
    job=launch.get_job(protocol,job_name);resources=job["resources"]
    if job["entrypoint"]!="scripts/bench/native_slam_campaign.py" or resources.get("gpus",0)!=0 or resources["cpus"]>2 or resources["memory_gb"]>16:
        raise ValueError("only bounded CPU native replay entrypoints may use this allocation utility")
    fresh=None
    if job["kind"]!="smoke":
        if fresh_smoke is None:raise ValueError("full replay requires an actual same-archive fresh smoke step receipt")
        fresh=validate_smoke_step(campaign,intake,protocol,fresh_smoke)
    raw_snapshot=subprocess.check_output(["scontrol","show","job",str(allocation),"--oneliner"],text=True)
    snapshot=allocation_resources(raw_snapshot,allocation,os.getuid())
    stem=job_name+"-allocation-step-"+str(allocation)
    receipt_path=campaign/(stem+".json");environment_path=campaign/(stem+"-environment.json");log_path=campaign/(stem+".log")
    start_path=campaign/(stem+"-start.json");target=campaign/(job_name+"-"+str(allocation))
    if any(p.exists() for p in (receipt_path,environment_path,log_path,start_path,target)):
        raise ValueError("allocation attempt evidence already exists; use a new campaign version")
    command=["srun","--jobid="+str(allocation),"--overlap","-n","1","-c","2","--mem=16G","--chdir=/tmp",
        "/bin/bash","--noprofile","--norc","-c",STEP_WRAPPER,"native-replay-step",str(environment_path),
        protocol["runtime"]["python"],str(campaign/"h34_campaign.sbatch"),str(campaign),job_name,intake["archive_sha256"]]
    base={"schema":"bhl-native-replay-allocation-step-v1","job_id":str(allocation),"allocation_job_id":str(allocation),
        "job_name":job_name,"archive_sha256":intake["archive_sha256"],"protocol_sha256":intake["protocol_sha256"],
        "allocation_resources":snapshot,"requested_step_resources":{"cpus":2,"memory_gib":16,"additional_gpus":0},
        "fresh_smoke_step":fresh,"command":command,"host_submitting":socket.gethostname(),
        "utility_sha256":digest(Path(__file__)),"intake_verifier_sha256":digest(Path(__file__).with_name("h34_campaign.py")),
        "batch_submission_receipts_modified":False,"step_environment_path":str(environment_path)}
    write_json(start_path,{**base,"status":"ALLOCATION_STEP_STARTED_NOT_COMPLETE","started_utc":now()})
    env={k:v for k,v in os.environ.items() if not k.startswith(("SLURM_","APPTAINERENV_"))}
    env["TMPDIR"]="/tmp";env["PYTHONDONTWRITEBYTECODE"]="1"
    returncode=125;failure=None
    try:
        with log_path.open("xb") as stream:
            process=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        returncode=process.returncode
    except BaseException as error:
        failure=repr(error)
        raise
    finally:
        actual=json.loads(environment_path.read_text()) if environment_path.exists() else {}
        completion_path=target/"completion.json"
        valid_identity=actual.get("SLURM_JOB_ID")==str(allocation) and bool(re.fullmatch(r"\d+",str(actual.get("SLURM_STEP_ID",""))))
        status="ALLOCATION_STEP_COMPLETE" if valid_identity and failure is None else "ALLOCATION_STEP_INCOMPLETE"
        receipt={**base,"status":status,"returncode":returncode,"failure":failure,"finished_utc":now(),
            "slurm_step_id":actual.get("SLURM_STEP_ID"),"step_environment":actual,
            "step_environment_sha256":digest(environment_path) if environment_path.exists() else None,
            "completion_sha256":digest(completion_path) if completion_path.exists() else None,
            "log_sha256":digest(log_path) if log_path.exists() else None}
        write_json(receipt_path,receipt)
    return receipt_path,receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir",type=Path,required=True);parser.add_argument("--job",required=True)
    parser.add_argument("--allocation",default="21739893");parser.add_argument("--fresh-smoke-step-receipt",type=Path)
    args=parser.parse_args();path,result=run_step(args.campaign_dir,args.job,args.allocation,args.fresh_smoke_step_receipt)
    print(json.dumps({"receipt":str(path),**{k:result[k] for k in ("status","returncode","job_id","job_name","slurm_step_id","archive_sha256")}},indent=2),flush=True)
    raise SystemExit(0 if result["status"]=="ALLOCATION_STEP_COMPLETE" and result["returncode"]==0 else 1)
