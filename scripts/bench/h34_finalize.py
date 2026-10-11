"""CPU finalization from pinned campaign and observer archives, without Git.

Observer code can add only the four explicitly named post-processing files;
the original frozen training source, protocol, teachers, and jobs stay intact.
Submit afterany on the full array so failed/missing cells become INCOMPLETE.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile

OBSERVER_FILES = (
    "scripts/bench/h34_collect.py",
    "scripts/bench/h4_campaign_summary.py",
    "scripts/bench/h34_finalize.py",
    "slurm/repo20261008/h34_finalize.sbatch",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Direct CLI users need the same immutable inventory as batch users.
    # importlib otherwise adds __pycache__ before source verification, even
    # though the imported helper's source bytes have not changed.
    previous_bytecode_setting = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module


def read_archive(archive, out, *, allowed=None, total_cap=2 * 1024 ** 3):
    """Validate the entire inventory before any bytes are extracted."""
    out = Path(out)
    with tarfile.open(archive, "r:gz") as tar:
        members, names = tar.getmembers(), set()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or str(name) != member.name or not member.isfile() or member.name in names:
                raise ValueError("archive has unsafe, duplicate, or non-file entries")
            if member.size < 0 or member.size > 512 * 1024 ** 2 or (allowed is not None and member.name not in allowed):
                raise ValueError("archive member exceeds allowed observer inventory or size")
            names.add(member.name)
        if sum(member.size for member in members) > total_cap:
            raise ValueError("archive exceeds bounded extraction cap")
        for member in members:
            target = out / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as stream, target.open("xb") as target_stream:
                shutil.copyfileobj(stream, target_stream)
    return names


def freeze_observer(repo, out):
    repo, out = Path(repo).resolve(), Path(out).resolve()
    files = {}
    for name in OBSERVER_FILES:
        path = repo / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 2 * 1024 ** 2:
            raise ValueError(f"observer file missing or oversized: {name}")
        files[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "observer-manifest.json", {"schema": "h34-observer-v1", "files": files})
    archive = out / "observer.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out / "observer-manifest.json", arcname="observer-manifest.json")
        for name in OBSERVER_FILES:
            tar.add(repo / name, arcname=name)
    shutil.copyfile(repo / OBSERVER_FILES[-1], out / "h34_finalize.sbatch")
    receipt = {"created_utc": now(), "status": "FROZEN_OBSERVER_NOT_SUBMITTED",
               "archive_sha256": digest(archive), "archive_bytes": archive.stat().st_size,
               "manifest_sha256": digest(out / "observer-manifest.json"), "files": files}
    atomic_json(out / "observer-intake.json", receipt)
    return receipt


def stage(persistent, observer_archive, observer_sha, scratch):
    persistent, scratch = Path(persistent).resolve(), Path(scratch).resolve()
    intake = json.loads((persistent / "intake.json").read_text())
    for filename, key in (("frozen-campaign.tar.gz", "archive_sha256"), ("protocol.json", "protocol_sha256"), ("h34_campaign.sbatch", "launcher_sha256")):
        if digest(persistent / filename) != intake[key]:
            raise ValueError(f"frozen campaign pin mismatch: {filename}")
    if digest(observer_archive) != observer_sha:
        raise ValueError("observer archive differs from submitted checksum")
    campaign = scratch / "campaign"
    campaign.mkdir()
    read_archive(persistent / "frozen-campaign.tar.gz", campaign)
    helper = load_module(campaign / "source/scripts/bench/h34_campaign.py", "h34_frozen_launch_helper")
    count = helper.verify_source(campaign)
    if digest(campaign / "protocol.json") != intake["protocol_sha256"]:
        raise ValueError("extracted campaign protocol mismatch")
    observer = scratch / "observer"
    observer.mkdir()
    names = read_archive(observer_archive, observer, allowed=set(OBSERVER_FILES) | {"observer-manifest.json"}, total_cap=8 * 1024 ** 2)
    if names != set(OBSERVER_FILES) | {"observer-manifest.json"}:
        raise ValueError("observer is missing its exact allowed inventory")
    manifest = json.loads((observer / "observer-manifest.json").read_text())
    if manifest.get("schema") != "h34-observer-v1" or set(manifest.get("files", {})) != set(OBSERVER_FILES):
        raise ValueError("invalid observer manifest")
    for name, row in manifest["files"].items():
        path = observer / name
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"observer file checksum mismatch: {name}")
        target = campaign / "source" / name
        if target.exists():
            if digest(target) != row["sha256"]:
                raise ValueError(f"observer would replace frozen source: {name}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    protocol = json.loads((campaign / "protocol.json").read_text())
    if protocol.get("task") not in {"H3", "H4"}:
        raise ValueError("finalizer only accepts H3/H4 protocols")
    # Six full H3 cells or three H4 cells, plus at most three smoke bundles.
    cap = 9 if protocol["task"] == "H3" else 6
    bundles = list(persistent.glob("*/outputs.tar.gz"))
    if len(bundles) > cap or sum(p.stat().st_size for p in bundles) > cap * 512 * 1024 ** 2:
        raise ValueError("campaign output archives exceed predeclared count/byte bounds")
    return campaign, protocol, {"archive_sha256": intake["archive_sha256"], "protocol_sha256": intake["protocol_sha256"],
                               "observer_sha256": observer_sha, "observer_manifest": manifest,
                               "original_source_files_verified": count, "task": protocol["task"]}


def container_command(protocol, persistent, output, scratch, campaign, observer_sha):
    runtime = protocol["runtime"]
    s, c, p, o = map(Path, (scratch, campaign, persistent, output))
    upstream = c / "source/external/Berkeley-Humanoid-Lite"
    variables = {"PYTHONPATH": ":".join(map(str, (c / "source/src", upstream / "source/berkeley_humanoid_lite",
                  upstream / "source/berkeley_humanoid_lite_assets", upstream / "source/berkeley_humanoid_lite_lowlevel"))),
                 "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                 "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TMPDIR": str(s / "tmp"),
                 "XDG_CACHE_HOME": str(s / "cache"), "XDG_CONFIG_HOME": str(s / "config"),
                 "XDG_DATA_HOME": str(s / "data"), "UPSTREAM": str(upstream), "REPO": str(c / "source")}
    env = {k: v for k, v in os.environ.items() if not k.startswith("APPTAINERENV_")}
    env.update({"APPTAINERENV_" + k: v for k, v in variables.items()})
    command = ["apptainer", "exec", "--containall", "--cleanenv", "--home", str(s / "home"),
               "--bind", f"{runtime['share_root']}:{runtime['share_root']}:ro", "--bind", f"{s}:{s}",
               "--bind", f"{c}:{c}:ro", "--bind", f"{p}:{p}:ro", "--bind", f"{o}:{o}:rw", "--pwd", str(s),
               runtime["sif"], runtime["python"], str(c / "source/scripts/bench/h34_finalize.py"),
               "worker", "--persistent-dir", str(p), "--out", str(o), "--source", str(c / "source"),
               "--observer-sha", observer_sha]
    return command, env


def worker(persistent, output, source, observer_sha):
    output, source = Path(output), Path(source)
    import sys
    sys.path.insert(0, str(source / "scripts/bench"))
    sys.path.insert(0, str(source / "src"))
    collector = load_module(source / "scripts/bench/h34_collect.py", "h34_pinned_cpu_collector")
    try:
        protocol = json.loads((Path(persistent) / "protocol.json").read_text())
        roots = {"campaign": Path(persistent), "source": source, "shared_runtime": Path(protocol["runtime"]["share_root"])}
        isolation = {key + "_readonly": bool(os.statvfs(path).f_flag & os.ST_RDONLY) for key, path in roots.items()}
        isolation["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        isolation["requested_gpus"] = 0
        atomic_json(output / "runtime-isolation.json", isolation)
        if not all(isolation[key + "_readonly"] for key in roots) or isolation["cuda_visible_devices"]:
            raise ValueError("CPU finalizer requires read-only campaign/source/shared runtime and no GPU visibility")
        report = collector.collect(Path(persistent), output / "collection")
        if report.get("status") not in {"PASS", "NEGATIVE", "INCOMPLETE"}:
            raise ValueError("collector returned an invalid scientific verdict")
        summary_path = output / "collection/scientific-summary.json"
        if report["status"] in {"PASS", "NEGATIVE"}:
            if not summary_path.is_file() or json.loads(summary_path.read_text()).get("status") != report["status"]:
                raise ValueError("complete collection lacks its matching recomputed scientific summary")
        result = {"status": report["status"], "observer_sha256": observer_sha,
                  "finished_utc": now(), "collection": "collection/collection.json",
                  "scientific_summary": "collection/scientific-summary.json" if summary_path.exists() else None,
                  "scope": "Independent CPU recomputation of complete frozen campaign; no additional training or GPU allocation."}
        if report.get("problems"):
            result["problems"] = report["problems"]
    except Exception as exc:
        result = {"status": "INCOMPLETE", "finished_utc": now(), "observer_sha256": observer_sha,
                  "problems": [f"{type(exc).__name__}: {exc}"]}
    atomic_json(output / "finalization.json", result)
    return 1 if result["status"] == "INCOMPLETE" else 0


def finalize(persistent, observer_archive, observer_sha, output):
    persistent, output = Path(persistent).resolve(), Path(output).resolve()
    if output.is_relative_to(persistent) or persistent.is_relative_to(output):
        raise ValueError("finalization output must be separate from the frozen campaign")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "finalization.json", {"status": "INCOMPLETE", "started_utc": now(), "stage": "STARTED",
                "observer_sha256": observer_sha, "problems": ["CPU finalization has not completed."]})
    rc, error, command = 1, None, None
    process = None
    def interrupted(signum, frame):
        if process is not None:
            process.terminate()
        raise InterruptedError(f"scheduler signal {signum}")
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        with tempfile.TemporaryDirectory(prefix="bhl-h34-finalize-", dir="/tmp") as local:
            scratch = Path(local)
            campaign, protocol, provenance = stage(persistent, observer_archive, observer_sha, scratch)
            for name in ("home", "tmp", "cache", "config", "data"):
                (scratch / name).mkdir()
            atomic_json(output / "provenance.json", provenance)
            runtime = protocol["runtime"]
            intake = json.loads((persistent / "intake.json").read_text())
            if digest(runtime["sif"]) != intake["sif_sha256"]:
                raise ValueError("shared SIF differs from frozen campaign image")
            command, env = container_command(protocol, persistent, output, scratch, campaign, observer_sha)
            atomic_json(output / "launch.json", {"started_utc": now(), "command": command,
                        "host": os.uname().nodename, "job_id": os.environ.get("SLURM_JOB_ID"), "gpus": 0})
            with (output / "runtime.log").open("xb") as log:
                process = subprocess.Popen(command, env=env, cwd=scratch, stdout=log, stderr=subprocess.STDOUT)
                rc = process.wait()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        result = json.loads((output / "finalization.json").read_text())
        if error or rc != 0:
            # Preserve collector's detailed incomplete report if it exists.
            result.update(status="INCOMPLETE", finished_utc=now())
            if error:
                result.setdefault("problems", []).append(error)
            atomic_json(output / "finalization.json", result)
        atomic_json(output / "completion.json", {"finished_utc": now(), "exit_status": rc, "status": result["status"],
                    "error": error, "observer_sha256": observer_sha,
                    "files": {str(p.relative_to(output)): {"sha256": digest(p), "bytes": p.stat().st_size}
                              for p in output.rglob("*") if p.is_file()}})
    return 1 if result["status"] == "INCOMPLETE" else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_subparsers(dest="action", required=True)
    freeze = actions.add_parser("freeze-observer")
    freeze.add_argument("--repo", required=True, type=Path)
    freeze.add_argument("--out", required=True, type=Path)
    run = actions.add_parser("finalize")
    run.add_argument("--persistent-dir", required=True, type=Path)
    run.add_argument("--observer-archive", required=True, type=Path)
    run.add_argument("--observer-sha", required=True)
    run.add_argument("--out", required=True, type=Path)
    w = actions.add_parser("worker")
    w.add_argument("--persistent-dir", required=True, type=Path)
    w.add_argument("--out", required=True, type=Path)
    w.add_argument("--source", required=True, type=Path)
    w.add_argument("--observer-sha", required=True)
    args = p.parse_args()
    if args.action == "freeze-observer":
        print(json.dumps(freeze_observer(args.repo, args.out), indent=2))
        return 0
    if args.action == "worker":
        return worker(args.persistent_dir, args.out, args.source, args.observer_sha)
    return finalize(args.persistent_dir, args.observer_archive, args.observer_sha, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
