#!/usr/bin/env python3
"""Publish verified terminal native-campaign evidence from an isolated clone."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
SOURCE_FILES=["scripts/bench/native_run_publish.py","scripts/bench/native_run_audit.py",
    "scripts/bench/native_navigation_observe.py","scripts/bench/native_slam_collect.py",
    "scripts/bench/native_slam_campaign.py","scripts/bench/pose_research.py","scripts/bench/h34_campaign.py",
    "src/bhl_robust/__init__.py","src/bhl_robust/research/__init__.py",
    "src/bhl_robust/research/native_orb.py","src/bhl_robust/research/native_lio.py",
    "src/bhl_robust/research/pose_metrics.py"]
REPOSITORY="joses2017smjh/bhl-robustness-ladder"
RELEASE_TAG="native-campaign-evidence-20261009"
RESULTS=Path("results/native-campaign-20261009")


def digest(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def safe_name(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or str(p)!=name:raise ValueError("unsafe evidence path")
    return p


def run(argv,*,cwd=None,stdout=None):
    process=subprocess.run(list(map(str,argv)),cwd=cwd,stdout=stdout or subprocess.PIPE,
        stderr=subprocess.PIPE,text=stdout is None)
    if process.returncode:
        # Commands never contain credentials. Do not echo subprocess stderr or
        # auth environment, which may contain configuration-specific material.
        raise RuntimeError(f"{Path(str(argv[0])).name} operation failed (exit {process.returncode})")
    return process.stdout if stdout is None else None


def freeze(source,plan_path,durable):
    source,plan_path,durable=map(lambda p:Path(p).resolve(),(source,plan_path,durable))
    plan=json.loads(plan_path.read_text());validate_plan(plan)
    for key,setting in (("commit_name","user.name"),("commit_email","user.email")):
        if not plan.get(key):plan[key]=run([plan.get("git","/bin/git"),"-C",source,"config","--get",setting]).strip()
        if not plan[key]:raise ValueError("existing configured Git commit identity required")
    durable.mkdir(parents=True,exist_ok=False)
    for name in SOURCE_FILES:
        expected=digest(source/name)
        target=durable/"source"/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,target)
        if digest(target)!=expected:raise ValueError("publisher source changed during freeze: "+name)
    write(durable/"plan.json",plan)
    files=[durable/"plan.json",*sorted((durable/"source").rglob("*"))]
    manifest={str(p.relative_to(durable)):{"sha256":digest(p),"bytes":p.stat().st_size}
              for p in files if p.is_file()}
    write(durable/"manifest.json",manifest)
    receipt={"schema":"bhl-native-publisher-bundle-v1","manifest_sha256":digest(durable/"manifest.json"),
        "source_files":len(SOURCE_FILES),"bundle_bytes":sum(r["bytes"] for r in manifest.values()),
        "scope":"publication source/configuration only; no native binary or private experiment pack"}
    write(durable/"intake.json",receipt)
    return receipt


def validate_plan(plan):
    if plan.get("schema")!="bhl-native-publish-plan-v1" or plan.get("repository")!=REPOSITORY or plan.get("release_tag")!=RELEASE_TAG:
        raise ValueError("explicitly authorized repository/release plan required")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+",plan["collection_id"]):raise ValueError("unsafe collection ID")
    labels=set()
    for source in plan["campaigns"]:
        if source["kind"] not in {"lio_navigation","terrain","stereo_navigation","native_replay"}:raise ValueError("unknown scientific collector")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+",source["label"]) or source["label"] in labels:raise ValueError("duplicate/unsafe collection label")
        labels.add(source["label"])
        if not re.fullmatch(r"\d+(?:\.\d+)?",source["job_id"]):raise ValueError("unsafe job ID")
        if not re.fullmatch(r"[0-9a-f]{64}",source["source_archive_sha256"]):raise ValueError("exact experiment source SHA required")
        if not Path(source["campaign_dir"]).is_absolute():raise ValueError("absolute durable campaign path required")
        if source.get("raw_asset_name") and not re.fullmatch(r"[a-zA-Z0-9_.-]+\.tar\.gz",source["raw_asset_name"]):raise ValueError("unsafe raw asset filename")
    if not labels:raise ValueError("at least one predeclared campaign required")
    return plan


def verify_bundle(bundle,expected):
    bundle=Path(bundle).resolve()
    if digest(bundle/"manifest.json")!=expected:raise ValueError("publisher manifest changed")
    manifest=json.loads((bundle/"manifest.json").read_text())
    actual={str(p.relative_to(bundle)) for p in (bundle/"source").rglob("*") if p.is_file()}|{"plan.json"}
    if actual!=set(manifest):raise ValueError("publisher source inventory changed")
    for name,row in manifest.items():
        safe_name(name);p=bundle/name
        if p.is_symlink() or p.stat().st_size!=row["bytes"] or digest(p)!=row["sha256"]:raise ValueError("publisher source hash changed")
    return validate_plan(json.loads((bundle/"plan.json").read_text()))


def validate_completion(source):
    campaign=Path(source["campaign_dir"]);intake=json.loads((campaign/"intake.json").read_text())
    if intake["archive_sha256"]!=source["source_archive_sha256"]:raise ValueError("experiment source pin changed")
    if digest(campaign/"protocol.json")!=intake["protocol_sha256"]:raise ValueError("experiment protocol hash changed")
    frozen=campaign/"frozen-campaign.tar.gz"
    if frozen.exists():
        if digest(frozen)!=intake["archive_sha256"]:raise ValueError("experiment frozen source hash changed")
        source_verification={"original_archive_rehashed":True,"archive_sha256":intake["archive_sha256"]}
    elif (campaign/"successful-packaging-reclamation.json").exists():
        audit=load("retained_native_source_audit",ROOT/"scripts/bench/native_run_audit.py")
        source_verification=audit.verify_retained_payloads(campaign,intake,source.get("source_manifest_sha256"))
    else:raise ValueError("frozen source packaging unavailable without verified retained-payload lineage")
    submission_path=Path(source["allocation_step"]) if source.get("allocation_step") else campaign/(source["job_name"]+"-submission.json")
    submission=json.loads(submission_path.read_text())
    expected="ALLOCATION_STEP_COMPLETE" if source.get("allocation_step") else "SUBMITTED"
    if submission["status"]!=expected or str(submission["job_id"])!=source["job_id"] or submission["archive_sha256"]!=intake["archive_sha256"]:
        raise ValueError("experiment submission identity/source changed")
    job=campaign/(source["job_name"]+"-"+source["job_id"])
    completion=json.loads((job/"completion.json").read_text());launch=json.loads((job/"launch.json").read_text())
    pinned_manifest=source.get("source_manifest_sha256") or intake.get("source_and_input_manifest_sha256") or intake.get("source_manifest_sha256_unchanged")
    if pinned_manifest and digest(campaign/"manifest.json")!=pinned_manifest:raise ValueError("original manifest differs from frozen publisher/intake pin")
    for receipt in (launch,completion):
        recorded_manifest=receipt.get("source_and_input_manifest_sha256",receipt.get("source_manifest_sha256"))
        if recorded_manifest and pinned_manifest and recorded_manifest!=pinned_manifest:raise ValueError("executed manifest differs from frozen publisher/intake pin")
    if completion["archive_sha256"]!=intake["archive_sha256"] or launch["archive_sha256"]!=intake["archive_sha256"]:
        raise ValueError("completed experiment source changed")
    if str(launch["job_id"])!=source["job_id"] or launch["job"]["name"]!=source["job_name"]:raise ValueError("completed job identity changed")
    for name,row in completion["files"].items():
        if PurePosixPath(name).name!=name:raise ValueError("unsafe completion filename")
        path=job/name
        if path.is_symlink() or path.stat().st_size!=row["bytes"] or digest(path)!=row["sha256"]:raise ValueError("completion evidence checksum changed: "+name)
    result=json.loads((job/"campaign_result.json").read_text())
    if result["status"] not in {"PASS","NEGATIVE","INCOMPLETE"} or result["status"]!=completion["status"]:
        raise ValueError("scientific/completion status differs")
    if result["status"]!="INCOMPLETE" and completion["exit_status"]!=0:raise ValueError("claimed completed result has nonzero runtime exit")
    return job,completion,result,submission_path,source_verification


def guard_raw_archive(path):
    """Only original research data; never native/runtime/model/private packs."""
    with tarfile.open(path) as archive:
        names=set()
        for member in archive:
            safe_name(member.name)
            if not member.isfile() or member.name in names:raise ValueError("raw archive has duplicate/non-file evidence")
            names.add(member.name)
            suffix=Path(member.name).suffix.lower()
            if suffix in {".so",".a",".onnx",".pt",".pth",".sif",".gz"} or "runtime" in PurePosixPath(member.name).parts:
                raise ValueError("native binary/model/private packaging excluded from publication")
            beginning=archive.extractfile(member).read(8)
            if beginning.startswith((b"\x7fELF",b"!<arch>")):raise ValueError("native binary excluded from publication")


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def collect_one(source,work):
    output=work/source["label"]
    record={"label":source["label"],"kind":source["kind"],"job_id":source["job_id"],
        "source_archive_sha256":source["source_archive_sha256"],"scientific_status":"INCOMPLETE",
        "audit_status":"INCOMPLETE","measured_episodes":None,"scope":"simulation only; smoke excluded from qualification","problems":[]}
    try:
        job,completion,campaign,submission_path,source_verification=validate_completion(source)
        record["source_verification"]=source_verification
        raw=job/"outputs.tar.gz"
        if raw.exists():guard_raw_archive(raw)
        if campaign["status"]=="INCOMPLETE":
            output.mkdir()
            record["problems"].append("Original runner reported INCOMPLETE; no complete cohort or missing episode count is inferred")
        elif source["kind"]=="lio_navigation":
            bundle=Path(source["collector_bundle"])
            if digest(bundle/"manifest.json")!=source["collector_manifest_sha256"]:raise ValueError("LIO collector bundle pin changed")
            collector=load("pinned_lio_collector",bundle/"source/scripts/bench/native_navigation_collect.py")
            collector.collect(bundle,Path(source["campaign_dir"]),source["job_name"],output)
            observed=json.loads((output/"observer.json").read_text());record.update(audit_status="PASS",scientific_status=observed["scientific_status"],measured_episodes=len(observed["episodes"]))
        elif source["kind"]=="native_replay":
            collector=load("pinned_replay_collector",ROOT/"scripts/bench/native_slam_collect.py")
            observed=collector.collect(Path(source["campaign_dir"]),source["job_name"],output,source.get("allocation_step"))
            record.update(audit_status="PASS",scientific_status=observed["scientific_status"],measured_episodes=0)
        else:
            audit=load("pinned_native_raw_audit",ROOT/"scripts/bench/native_run_audit.py")
            if source["kind"]=="terrain":observed=audit.terrain(job,output)
            else:
                promotion=json.loads((Path(source["campaign_dir"])/"promotion.json").read_text())
                if promotion["archive_sha256"]!=source["source_archive_sha256"]:raise ValueError("stereo runtime promotion source pin differs")
                observed=audit.stereo(job,output,expected_binary_sha256=promotion["runtime_verification"]["binary_sha256"])
            record.update(audit_status="PASS",scientific_status=observed["scientific_status"],measured_episodes=len(observed["episodes"]))
        evidence=output/"verified-evidence";evidence.mkdir(exist_ok=True)
        for name in ("completion.json","launch.json","campaign_result.json"):shutil.copyfile(job/name,evidence/name)
        shutil.copyfile(submission_path,evidence/"submission.json")
        shutil.copyfile(Path(source["campaign_dir"])/"protocol.json",evidence/"protocol.json")
        for name in ("intake.json","manifest.json","retained-member-inventory.json","successful-packaging-reclamation.json"):
            path=Path(source["campaign_dir"])/name
            if path.exists():shutil.copyfile(path,evidence/name)
        if raw.exists():record["raw"]={"path":str(raw),"sha256":digest(raw),"bytes":raw.stat().st_size}
    except Exception as error:
        # Keep explicit failure evidence, without publishing an unverified raw
        # archive or silently replacing incomplete science with zero episodes.
        record.update(scientific_status="INCOMPLETE",audit_status="INCOMPLETE",measured_episodes=None)
        record.pop("raw",None);record["problems"].append(type(error).__name__+": "+str(error))
        output.mkdir(exist_ok=True)
    write(output/"publication-observation.json",record)
    return record


def release_asset(gh,path,name,checksum,work):
    release=json.loads(run([gh,"api",f"repos/{REPOSITORY}/releases/tags/{RELEASE_TAG}"]))
    matching=[asset for asset in release["assets"] if asset["name"]==name]
    if not matching:
        staged=work/name
        try:os.link(path,staged)
        except OSError:shutil.copyfile(path,staged)
        if digest(staged)!=checksum:raise ValueError("staged release payload checksum mismatch")
        run([gh,"release","upload",RELEASE_TAG,staged,"--repo",REPOSITORY])
        staged.unlink()
        release=json.loads(run([gh,"api",f"repos/{REPOSITORY}/releases/tags/{RELEASE_TAG}"]))
        matching=[asset for asset in release["assets"] if asset["name"]==name]
    if len(matching)!=1:raise ValueError("unique release asset not found")
    asset=matching[0]
    if asset["size"]!=Path(path).stat().st_size:raise ValueError("release asset size mismatch")
    if asset.get("digest")=="sha256:"+checksum:method="github_server_sha256"
    else:
        downloaded=work/("verify-"+name)
        with downloaded.open("xb") as stream:run([gh,"api",f"repos/{REPOSITORY}/releases/assets/{asset['id']}","-H","Accept: application/octet-stream"],stdout=stream)
        if digest(downloaded)!=checksum:raise ValueError("release asset checksum mismatch")
        downloaded.unlink();method="downloaded_sha256"
    return {"url":asset["browser_download_url"],"sha256":checksum,"asset_id":asset["id"],"verification":method}


def build_index(existing,records,collection_id):
    indexed={row["label"]+"-"+row["job_id"]:row for row in existing.get("campaigns",[])}
    for row in records:indexed[row["label"]+"-"+row["job_id"]]=row
    return {"schema":"bhl-native-public-collection-v1","collection_id":collection_id,
        "updated_utc":datetime.now(timezone.utc).isoformat(),"scope":"actual simulated campaigns; smoke/replay/navigation/traversal kept explicit; no hardware validation",
        "campaigns":list(indexed.values())}


def report(index):
    lines=["# Native run collection — 2026-10-09","",index["scope"]+".","",
        "PASS means the experiment's declared scientific gate passed; NEGATIVE is a completed failed gate; INCOMPLETE means missing, interrupted or unverifiable evidence. Smoke is labeled SMOKE_ONLY and never qualifies an actor. Replay has zero closed-loop episodes.","",
        "|Campaign|Job|Scientific status|Audit|Measured episodes|Evidence|","|---|---:|---|---|---:|---|"]
    for row in index["campaigns"]:
        count="unknown" if row["measured_episodes"] is None else str(row["measured_episodes"])
        path="collected/"+row["label"]+"-"+row["job_id"]+"/publication-observation.json"
        lines.append(f"|{row['label']}|{row['job_id']}|{row['scientific_status']}|{row['audit_status']}|{count}|[verified collection]({path})|")
    lines.extend(["","Each unique result directory contains source/job/completion hashes, derived audits and original outcome receipts. Raw archives above50MiB are linked from the verified prerelease; smaller raw archives are retained beside their collection. Native binaries and private frozen input packs are excluded.",""])
    return "\n".join(lines)


def push_records(plan,records,collected,work):
    git=plan.get("git","/bin/git");url="https://github.com/"+REPOSITORY+".git"
    for attempt in range(4):
        checkout=work/f"publication-clone-{attempt}"
        run([git,"clone","--depth","1","--branch","main",url,checkout])
        # Use the existing gh credential store through this clone's config.
        # No token enters arguments/logs and no user/global config is modified.
        gh=str(plan["gh"])
        if not re.fullmatch(r"/[a-zA-Z0-9_./-]+",gh):raise ValueError("safe absolute gh executable path required")
        run([git,"config","credential.helper",""],cwd=checkout)
        run([git,"config","credential.https://github.com.helper","!"+gh+" auth git-credential"],cwd=checkout)
        for row in records:
            destination=checkout/RESULTS/"collected"/(row["label"]+"-"+row["job_id"])
            if destination.exists():
                previous=json.loads((destination/"publication-observation.json").read_text())
                if previous["source_archive_sha256"]!=row["source_archive_sha256"]:raise ValueError("published job source identity collision")
                shutil.rmtree(destination)
            shutil.copytree(collected/row["label"],destination)
        index_path=checkout/RESULTS/"collection-index.json"
        existing=json.loads(index_path.read_text()) if index_path.exists() else {}
        index=build_index(existing,records,plan["collection_id"])
        write(index_path,index)
        (checkout/RESULTS/"NATIVE_RUN_COLLECTION_2026-10-09.md").write_text(report(index))
        run([git,"add","--",str(RESULTS/"collected"),str(RESULTS/"collection-index.json"),str(RESULTS/"NATIVE_RUN_COLLECTION_2026-10-09.md")],cwd=checkout)
        changed=run([git,"diff","--cached","--name-only"],cwd=checkout).strip()
        if not changed:return {"status":"ALREADY_PUBLISHED","commit":run([git,"rev-parse","HEAD"],cwd=checkout).strip()}
        run([git,"-c","user.name="+plan["commit_name"],"-c","user.email="+plan["commit_email"],"commit","-m","Publish verified native campaign outcomes for 2026-10-09"],cwd=checkout)
        proc=subprocess.run([git,"push","origin","HEAD:main"],cwd=checkout,capture_output=True,text=True)
        if proc.returncode==0:return {"status":"PUSHED","commit":run([git,"rev-parse","HEAD"],cwd=checkout).strip()}
        # A clean new clone of the latest main handles concurrent remote work;
        # never force-push, reset or modify the original working repository.
        shutil.rmtree(checkout)
    raise RuntimeError("normal fastforward push failed after four clean-clone attempts")


def publish(bundle,manifest_sha256,output):
    plan=verify_bundle(bundle,manifest_sha256);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/"src"))
    result={"schema":"bhl-native-publication-v1","status":"INCOMPLETE","publisher_manifest_sha256":manifest_sha256,
        "repository":REPOSITORY,"release_tag":RELEASE_TAG,"campaigns":[],"problems":[]}
    try:
        with tempfile.TemporaryDirectory(prefix="bhl-native-publisher-") as temporary:
            work=Path(temporary);collected=work/"collected";collected.mkdir()
            records=[collect_one(source,collected) for source in plan["campaigns"]]
            asset_names={source["label"]:source.get("raw_asset_name") for source in plan["campaigns"]}
            for row in records:
                raw=row.get("raw")
                if raw:
                    asset_name=asset_names[row["label"]] or f"{row['label']}-{row['job_id']}-raw.tar.gz"
                    if raw["bytes"]>50*1024**2:
                        row["raw_publication"]=release_asset(plan["gh"],Path(raw["path"]),asset_name,raw["sha256"],work)
                    else:
                        shutil.copyfile(raw["path"],collected/row["label"]/asset_name)
                        if digest(collected/row["label"]/asset_name)!=raw["sha256"]:raise ValueError("copied raw payload checksum mismatch")
                        row["raw_publication"]={"relative_path":asset_name,"sha256":raw["sha256"],"verification":"local_completed_archive_sha256"}
                # Keep durable source paths in provenance, never credentials.
                write(collected/row["label"]/"publication-observation.json",row)
            published=push_records(plan,records,collected,work)
            result.update(status="PASS",campaigns=records,publication=published)
            shutil.copytree(collected,output/"collected",ignore=shutil.ignore_patterns("*.tar.gz"))
    except Exception as error:result["problems"].append(type(error).__name__+": "+str(error))
    write(output/"publication.json",result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest="command",required=True)
    create=commands.add_parser("freeze");create.add_argument("--source-root",type=Path,required=True);create.add_argument("--plan",type=Path,required=True);create.add_argument("--durable",type=Path,required=True)
    execute=commands.add_parser("publish");execute.add_argument("--bundle",type=Path,required=True);execute.add_argument("--manifest-sha256",required=True);execute.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=="freeze":result=freeze(args.source_root,args.plan,args.durable)
    else:result=publish(args.bundle,args.manifest_sha256,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!="campaigns"}),flush=True)
    return 0 if result.get("status","PASS")=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
