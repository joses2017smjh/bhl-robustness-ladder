"""Freeze and run bounded H3/H4 jobs without writing to shared training trees.

The protocol supplies scientific entrypoints and acceptance rules. This module
checks provenance and launch resources; an exit code is never a science result.
Only ``output/`` is archived: runners must copy complete metrics, configuration,
and selected checkpoints there, while transient training files stay in work/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def relative_name(name):
    if not isinstance(name, str):
        raise ValueError(f"unsafe relative path: {name!r}")
    p = PurePosixPath(name)
    if not isinstance(name, str) or not name or p.is_absolute() or ".." in p.parts or str(p) != name:
        raise ValueError(f"unsafe relative path: {name!r}")
    return p


def validate_protocol(protocol):
    if protocol.get("schema_version") != 1:
        raise ValueError("protocol schema_version must be 1")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", protocol.get("campaign_id", "")):
        raise ValueError("invalid campaign_id")
    runtime = protocol["runtime"]
    for key in ("python", "share_root"):
        if not Path(runtime[key]).is_absolute():
            raise ValueError(f"runtime {key} must be absolute")
    if runtime.get("stack", "v51") not in {"v51", "v60", "cpu"}:
        raise ValueError("unknown runtime stack")
    names = set()
    jobs = protocol["jobs"]
    if not jobs or len(jobs) > 32:
        raise ValueError("campaign requires 1..32 bounded jobs")
    for job in jobs:
        name = job["name"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in names:
            raise ValueError("job names must be unique safe identifiers")
        names.add(name)
        if job["kind"] not in {"smoke", "run", "evaluate"}:
            raise ValueError("unknown job kind")
        relative_name(job["entrypoint"])
        relative_name(job.get("result_path", "campaign_result.json"))
        if not isinstance(job.get("args", []), list) or not all(isinstance(x, str) for x in job.get("args", [])):
            raise ValueError("job args must be a list of strings")
        r = job["resources"]
        if "requeue" in r and not isinstance(r["requeue"], bool):
            raise ValueError("requeue must be an explicit boolean")
        if not isinstance(r["cpus"], int) or not 1 <= r["cpus"] <= 16:
            raise ValueError("cpus must be in 1..16")
        if not isinstance(r["memory_gb"], int) or not 1 <= r["memory_gb"] <= 128:
            raise ValueError("memory_gb must be in 1..128")
        if r.get("gpus", 0) not in {0, 1}:
            raise ValueError("each job may allocate at most one GPU")
        if r.get("array") and not re.fullmatch(r"(?:\d+(?:-\d+)?)(?:,\d+(?:-\d+)?)*(?:%[12])?", r["array"]):
            raise ValueError("array must use numeric cells and at most two simultaneous tasks")
        if r.get("gpus", 0) and r.get("array") and not r["array"].endswith("%1"):
            raise ValueError("H3 and H4 GPU arrays must each cap concurrency at one")
        if not re.fullmatch(r"[0-4]?\d:[0-5]\d:[0-5]\d", r["time_limit"]) or tuple(map(int, r["time_limit"].split(":"))) > (48, 0, 0):
            raise ValueError("wall time must be HH:MM:SS and at most 48 hours")
        if r.get("gpus", 0):
            if not runtime.get("sif") or not Path(runtime["sif"]).is_absolute():
                raise ValueError("GPU jobs require an absolute SIF path")
            if "dgx2" in r["partition"].split(","):
                raise ValueError("V100 dgx2 is unsupported by the frozen PyTorch/Isaac stack")
        env = job.get("env", {})
        if not isinstance(env, dict) or any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", k) or not isinstance(v, str) for k, v in env.items()):
            raise ValueError("job env must map uppercase names to strings")
        forbidden = {"HOME", "PATH", "PYTHONPATH", "PYTHONHOME", "UPSTREAM", "REPO", "TMPDIR", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH",
                     "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "CUDA_CACHE_PATH", "OV_CACHE", "PYTHONDONTWRITEBYTECODE"}
        if forbidden.intersection(env) or any(k.startswith(("SLURM_", "APPTAINER", "H34_")) for k in env):
            raise ValueError("job env cannot override isolation or GPU masks")
        if not 1 <= job.get("max_output_mb", 512) <= 2048:
            raise ValueError("max_output_mb must be in 1..2048")
    return protocol


def git_output(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def source_files(repo):
    """Tracked runtime files, including initialized pinned nested submodules."""
    roots = [(Path(repo), Path("."))]
    submodules = git_output(repo, "submodule", "status", "--recursive")
    for row in submodules.splitlines():
        # check_output.strip removes the initial space on the first clean row.
        if row.startswith(("-", "+", "U")):
            raise ValueError("every upstream submodule must be initialized at its pin")
        fields = row.split()
        roots.append((Path(repo) / fields[1], Path(fields[1])))
    files = {}
    prefixes = ("src/", "scripts/", "slurm/", "assets/", "configs/", "config/")
    for root, prefix in roots:
        for name in git_output(root, "ls-files", "-z").split("\0"):
            if not name:
                continue
            src = root / name
            if not src.is_file():
                continue
            if prefix == Path(".") and not (name.startswith(prefixes) or name in {"pyproject.toml", "uv.lock", ".gitmodules", "requirements-test.txt"}):
                continue
            # Dereference only links to other immutable repository inputs.
            if not src.resolve().is_relative_to(Path(repo).resolve()):
                raise ValueError(f"source link escapes repository: {src}")
            files[str(prefix / name)] = src
    return files


def freeze(repo, protocol_path, persistent):
    repo, persistent = Path(repo).resolve(), Path(persistent).resolve()
    protocol = validate_protocol(json.loads(Path(protocol_path).read_text()))
    files = source_files(repo)
    # Newly implemented runners can be explicitly listed before their commit;
    # hashes and dirty-state receipt keep that provenance honest.
    for name in protocol.get("additional_source", []):
        relative_name(name)
        p = repo / name
        if not p.is_file() or not p.resolve().is_relative_to(repo):
            raise ValueError(f"missing additional source: {name}")
        files[name] = p
    for job in protocol["jobs"]:
        if job["entrypoint"] not in files:
            raise ValueError(f"entrypoint absent from frozen source: {job['entrypoint']}")
    staged = {"source/" + name: path for name, path in files.items()}
    for name, row in protocol.get("input_files", {}).items():
        relative_name(name)
        path = Path(row if isinstance(row, str) else row["path"])
        if not path.is_file() or path.is_symlink() or (isinstance(row, dict) and digest(path) != row["sha256"]):
            raise ValueError(f"declared input missing or hash mismatch: {name}")
        staged["inputs/" + name] = path
    total_bytes = sum(path.stat().st_size for path in staged.values())
    if total_bytes > 2 * 1024 ** 3 or any(path.stat().st_size > 512 * 1024 ** 2 for path in staged.values()):
        raise ValueError("frozen inputs exceed the 2 GiB total or 512 MiB per-file cap")
    existing_parent = persistent.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < total_bytes + 64 * 1024 ** 2:
        raise ValueError("insufficient durable space for bounded frozen inputs")
    persistent.mkdir(parents=True, exist_ok=False)
    write_json(persistent / "protocol.json", protocol)
    manifest = {name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in sorted(staged.items())}
    write_json(persistent / "manifest.json", manifest)
    archive = persistent / "frozen-campaign.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(persistent / "protocol.json", arcname="protocol.json")
        tar.add(persistent / "manifest.json", arcname="manifest.json")
        for name, path in sorted(staged.items()):
            info = tar.gettarinfo(str(path), arcname=name)
            # Store dereferenced links as regular files.
            info.type, info.linkname, info.size = tarfile.REGTYPE, "", path.stat().st_size
            with path.open("rb") as stream:
                tar.addfile(info, stream)
    launcher = repo / "slurm/repo20261008/h34_campaign.sbatch"
    shutil.copyfile(launcher, persistent / "h34_campaign.sbatch")
    runtime = protocol["runtime"]
    receipt = {"created_utc": now(), "source_commit": git_output(repo, "rev-parse", "HEAD"),
               "source_status": git_output(repo, "status", "--porcelain"),
               "submodules": git_output(repo, "submodule", "status", "--recursive"),
               "archive_sha256": digest(archive), "protocol_sha256": digest(persistent / "protocol.json"),
               "launcher_sha256": digest(persistent / "h34_campaign.sbatch"),
               "sif_sha256": digest(runtime["sif"]) if runtime.get("sif") else None,
               "source_files": len(manifest), "uncompressed_input_bytes": total_bytes,
               "archive_bytes": archive.stat().st_size, "status": "FROZEN_NOT_SUBMITTED"}
    write_json(persistent / "intake.json", receipt)
    return receipt


def load_intake(persistent):
    p = Path(persistent).resolve()
    intake = json.loads((p / "intake.json").read_text())
    for name, key in (("frozen-campaign.tar.gz", "archive_sha256"), ("protocol.json", "protocol_sha256"), ("h34_campaign.sbatch", "launcher_sha256")):
        if digest(p / name) != intake[key]:
            raise ValueError(f"frozen input changed: {name}")
    return p, intake, validate_protocol(json.loads((p / "protocol.json").read_text()))


def get_job(protocol, name):
    return next(j for j in protocol["jobs"] if j["name"] == name)


def submit(persistent, name, dependency=None):
    p, intake, protocol = load_intake(persistent)
    job = get_job(protocol, name)
    r = job["resources"]
    if job["kind"] == "run" and not dependency:
        raise ValueError("training run requires an afterok smoke dependency")
    if dependency and not re.fullmatch(r"afterok:\d+", dependency):
        raise ValueError("dependency must be afterok:<jobid>")
    if job["kind"] == "run":
        expected_id = dependency.split(":")[1]
        matching_smoke = False
        for smoke in (j for j in protocol["jobs"] if j["kind"] == "smoke"):
            receipt = p / f"{smoke['name']}-submission.json"
            if receipt.is_file():
                data = json.loads(receipt.read_text())
                matching_smoke |= data.get("job_id") == expected_id and data.get("archive_sha256") == intake["archive_sha256"]
        if not matching_smoke:
            raise ValueError("run dependency must be a fresh smoke submission of this frozen archive")
    cmd = ["sbatch", "--parsable", "--ntasks=1", "--account=eecs", f"--job-name={name}",
           f"--partition={r['partition']}", f"--cpus-per-task={r['cpus']}", f"--mem={r['memory_gb']}G",
           f"--time={r['time_limit']}", f"--output={p}/{name}-%j.out", f"--error={p}/{name}-%j.out",
           "--export=NONE"]
    if r.get("requeue") is False:
        # A preempted immutable attempt must retain its exclusive receipt
        # directory. Restarting under the same job ID would collide with it.
        cmd.append("--no-requeue")
    for key in ("constraint", "exclude"):
        if r.get(key):
            cmd.append(f"--{key}={r[key]}")
    if r.get("gpus", 0):
        cmd.append("--gres=gpu:1")
    if r.get("array"):
        cmd.append(f"--array={r['array']}")
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.extend([str(p / "h34_campaign.sbatch"), str(p), name, intake["archive_sha256"]])
    write_json(p / f"{name}-submission-start.json", {"created_utc": now(), "command": cmd})
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "APPTAINERENV_"))}
    env["TMPDIR"] = "/tmp"
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    match = re.fullmatch(r"(\d+)(?:;[^\n]+)?\n?", proc.stdout)
    receipt = {"submitted_utc": now(), "job_id": match[1] if match and proc.returncode == 0 else None,
               "command": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
               "resources": r, "archive_sha256": intake["archive_sha256"], "status": "SUBMITTED" if match and proc.returncode == 0 else "SUBMISSION_FAILED"}
    write_json(p / f"{name}-submission.json", receipt)
    return receipt


def safe_extract(archive, destination):
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = set()
        for member in members:
            relative_name(member.name)
            if member.name in names or not member.isfile() or member.size > 512 * 1024 * 1024:
                raise ValueError("archive contains duplicate, non-file, or oversized member")
            names.add(member.name)
        if sum(m.size for m in members) > 2 * 1024 ** 3:
            raise ValueError("frozen source exceeds 2 GiB")
        for member in members:
            target = Path(destination) / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as stream, target.open("xb") as out:
                shutil.copyfileobj(stream, out)


def verify_source(scratch):
    root = Path(scratch)
    manifest = json.loads((root / "manifest.json").read_text())
    actual = {str(x.relative_to(root)) for subtree in ("source", "inputs") for x in (root / subtree).rglob("*") if x.is_file()}
    if actual != set(manifest):
        raise ValueError("frozen source file inventory changed")
    for name, row in manifest.items():
        relative_name(name)
        path = root / name
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"frozen source hash mismatch: {name}")
    return len(manifest)


def runtime_command(protocol, job, scratch):
    runtime = protocol["runtime"]
    s = Path(scratch)
    replacements = {"{source}": str(s / "source"), "{work}": str(s / "work"),
                    "{output}": str(s / "output"), "{protocol}": str(s / "protocol.json"),
                    "{campaign}": str(s), "{cell}": os.environ.get("SLURM_ARRAY_TASK_ID", "0")}
    def expand(value):
        for token, path in replacements.items():
            value = value.replace(token, path)
        return value
    upstream = s / "source/external/Berkeley-Humanoid-Lite"
    paths = [s / "source/src", upstream / "source/berkeley_humanoid_lite",
             upstream / "source/berkeley_humanoid_lite_assets", upstream / "source/berkeley_humanoid_lite_lowlevel"]
    variables = {"PYTHONPATH": ":".join(map(str, paths)), "PYTHONDONTWRITEBYTECODE": "1",
                 "UPSTREAM": str(upstream), "REPO": str(s / "source"),
                 "TMPDIR": str(s / "tmp"), "XDG_CACHE_HOME": str(s / "cache"),
                 "CUDA_CACHE_PATH": str(s / "cache/cuda"), "OV_CACHE": str(s / "cache/ov"),
                 "XDG_CONFIG_HOME": str(s / "config"), "XDG_DATA_HOME": str(s / "data"),
                 "H34_OUTPUT_DIR": str(s / "output"), "H34_WORK_DIR": str(s / "work"),
                 "H34_PROTOCOL": str(s / "protocol.json"), "H34_KIT_ARGS": f"--portable-root {s}/kit",
                 "H34_STACK": runtime.get("stack", "v51"), "H34_CELL": os.environ.get("SLURM_ARRAY_TASK_ID", "0"),
                 "OMP_NUM_THREADS": str(job["resources"]["cpus"]),
                 "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "ACCEPT_EULA": "Y", "OMNI_KIT_ACCEPT_EULA": "YES"}
    variables.update({k: expand(v) for k, v in job.get("env", {}).items()})
    for key in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS"):
        if os.environ.get(key):
            variables[key] = os.environ[key]
    argv = [runtime["python"], str(s / "source" / job["entrypoint"]), *[expand(x) for x in job.get("args", [])]]
    env = {k: v for k, v in os.environ.items() if not k.startswith("APPTAINERENV_")}
    if runtime.get("sif"):
        env.update({"APPTAINERENV_" + k: v for k, v in variables.items()})
        command = ["apptainer", "exec", "--containall", "--cleanenv", "--home", str(s / "home"),
                   "--bind", f"{runtime['share_root']}:{runtime['share_root']}:ro", "--bind", f"{s}:{s}",
                   "--bind", f"{s}/source:{s}/source:ro", "--bind", f"{s}/protocol.json:{s}/protocol.json:ro",
                   "--pwd", str(s / "work")]
        if (s / "inputs").is_dir():
            command.extend(["--bind", f"{s}/inputs:{s}/inputs:ro"])
        if job["resources"].get("gpus", 0):
            command.append("--nv")
        command.extend([runtime["sif"], *argv])
    else:
        env.update(variables)
        command = argv
    return command, env


def execute(persistent, name, archive_sha):
    p, intake, protocol = load_intake(persistent)
    if archive_sha != intake["archive_sha256"]:
        raise ValueError("submitted archive pin differs from intake")
    job = get_job(protocol, name)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    target = p / f"{name}-{job_id}"
    target.mkdir(exist_ok=False)
    scratch = Path(tempfile.mkdtemp(prefix=f"bhl-h34-{name}-{job_id}-", dir="/tmp"))
    safe_extract(p / "frozen-campaign.tar.gz", scratch)
    verified = verify_source(scratch)
    for directory in ("home", "cache/cuda", "cache/ov", "tmp", "config", "data", "kit", "work", "output"):
        (scratch / directory).mkdir(parents=True, exist_ok=True)
    sif = protocol["runtime"].get("sif")
    if sif and digest(sif) != intake["sif_sha256"]:
        raise ValueError("runtime image changed after freeze")
    command, env = runtime_command(protocol, job, scratch)
    write_json(target / "launch.json", {"started_utc": now(), "host": os.uname().nodename, "job_id": job_id,
               "command": command, "source_files_verified": verified, "archive_sha256": archive_sha,
               "scratch": str(scratch), "job": job})
    proc = None
    def interrupted(signum, frame):
        if proc is not None:
            proc.terminate()
        raise InterruptedError(f"scheduler signal {signum}")
    old_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    rc, error = 1, None
    try:
        with (target / "runtime.log").open("xb") as log:
            proc = subprocess.Popen(command, cwd=scratch / "work", env=env, stdout=log, stderr=subprocess.STDOUT)
            rc = proc.wait()
    except BaseException as exc:
        error = str(exc)
        if proc is not None:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        outputs = [x for x in (scratch / "output").rglob("*") if x.is_file()]
        if any(x.is_symlink() for x in (scratch / "output").rglob("*")):
            error = "output links refused"
        output_bytes = sum(x.stat().st_size for x in outputs)
        if output_bytes > job.get("max_output_mb", 512) * 1024 ** 2:
            error = f"output exceeds predeclared archive cap ({output_bytes} bytes)"
        if not error:
            with tarfile.open(target / "outputs.tar.gz", "w:gz") as tar:
                for path in sorted(outputs):
                    tar.add(path, arcname=str(path.relative_to(scratch / "output")))
        result_path = scratch / "output" / job.get("result_path", "campaign_result.json")
        try:
            result = json.loads(result_path.read_text()) if result_path.is_file() else {"status": "INCOMPLETE", "problems": ["runner produced no scientific result"]}
        except (ValueError, OSError) as exc:
            result = {"status": "INCOMPLETE", "problems": [f"invalid runner result: {exc}"]}
        if rc != 0 and result.get("status") != "INCOMPLETE":
            result = {"status": "INCOMPLETE", "problems": [f"runtime exited {rc}"], "partial_result": result}
        if error or result.get("status") not in {"PASS", "NEGATIVE", "INCOMPLETE"}:
            result = {"status": "INCOMPLETE", "problems": [error or "invalid scientific status"]}
        write_json(target / "campaign_result.json", result)
        write_json(target / "completion.json", {"finished_utc": now(), "exit_status": rc, "status": result["status"],
                   "error": error, "output_bytes": output_bytes, "archive_sha256": archive_sha,
                   "files": {x.name: {"sha256": digest(x), "bytes": x.stat().st_size} for x in target.iterdir() if x.is_file()},
                   "scope": "Runner completion or phase verdict only, never inferred from scheduler completion. Paired H3 acceptance requires all six cells; H4 recipe acceptance requires the predeclared repeated-seed summary."})
    accepted = result["status"] == "PASS" if job["kind"] == "smoke" else result["status"] != "INCOMPLETE"
    return 0 if rc == 0 and accepted else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--repo", required=True, type=Path)
    f.add_argument("--protocol", required=True, type=Path)
    f.add_argument("--persistent-dir", required=True, type=Path)
    s = sub.add_parser("submit")
    s.add_argument("--persistent-dir", required=True, type=Path)
    s.add_argument("--job", required=True)
    s.add_argument("--dependency")
    e = sub.add_parser("execute")
    e.add_argument("--persistent-dir", required=True, type=Path)
    e.add_argument("--job", required=True)
    e.add_argument("--archive-sha", required=True)
    a = p.parse_args()
    if a.action == "freeze":
        print(json.dumps(freeze(a.repo, a.protocol, a.persistent_dir), indent=2))
    elif a.action == "submit":
        result = submit(a.persistent_dir, a.job, a.dependency)
        print(json.dumps(result, indent=2))
        return 0 if result["job_id"] else 1
    else:
        return execute(a.persistent_dir, a.job, a.archive_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
