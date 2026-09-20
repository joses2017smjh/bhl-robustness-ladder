"""Submit one explicit weekend experiment and preserve its scheduler receipt.

No broad grid is implicit. Supply --after for afterok dependencies; failed
prerequisites cancel their dependent jobs instead of retaining dead queue rows.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--partition", help="Default: share for CPUs, gpu,ampere for GPUs")
    p.add_argument("--constraint")
    p.add_argument("--time", default="01:00:00")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default="64G")
    p.add_argument("--after", action="append", default=[])
    p.add_argument("--aftercorr", help="Corresponding task in a prerequisite Slurm array")
    p.add_argument("--afterany", action="append", default=[], help="Reporting jobs only: run after any outcome")
    p.add_argument("--array")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--container-cpu", action="store_true")
    p.add_argument("--submit", action="store_true")
    p.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    a.partition = a.partition or ("share" if a.cpu or a.container_cpu else "gpu,ampere")
    args = a.args[1:] if a.args[:1] == ["--"] else a.args
    if not args:
        p.error("Pass the batch script arguments after --")
    out = ROOT / "results/weekend-20260919"
    if a.cpu and a.container_cpu:
        p.error("Choose plain CPU or container CPU")
    batch = ROOT / ("slurm/weekend_setup.sbatch" if a.container_cpu else
                    "slurm/weekend_cpu.sbatch" if a.cpu else "slurm/weekend_gpu.sbatch")
    cmd = ["sbatch", "--parsable", f"--job-name={a.name}", f"--partition={a.partition}",
           f"--time={a.time}", f"--cpus-per-task={a.cpus}", f"--mem={a.mem}",
           f"--chdir={ROOT}", f"--output={out}/%x-%A_%a.out", f"--error={out}/%x-%A_%a.out"]
    if a.constraint:
        cmd.append(f"--constraint={a.constraint}")
    if sum(bool(x) for x in (a.after, a.aftercorr, a.afterany)) > 1:
        p.error("Choose afterok, afterany, or a corresponding array dependency")
    if a.after:
        if not all(item.isdigit() for item in a.after):
            p.error("Dependency IDs must be numeric")
        cmd += ["--dependency=afterok:" + ":".join(a.after), "--kill-on-invalid-dep=yes"]
    if a.aftercorr:
        if not a.aftercorr.isdigit() or not a.array:
            p.error("aftercorr requires a numeric array ID and --array")
        cmd += ["--dependency=aftercorr:" + a.aftercorr, "--kill-on-invalid-dep=yes"]
    if a.afterany:
        if not all(item.isdigit() for item in a.afterany):
            p.error("afterany requires numeric IDs")
        cmd += ["--dependency=afterany:" + ":".join(a.afterany)]
    if a.array:
        cmd.append(f"--array={a.array}")
    cmd += [str(batch), *args]
    print(shlex.join(cmd), flush=True)
    if not a.submit:
        return
    out.mkdir(parents=True, exist_ok=True)
    # A fresh batch allocation must not inherit the OOD/interactive job mask.
    env = {k:v for k,v in os.environ.items() if not k.startswith("SLURM_")
           and k not in {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "TMPDIR"}}
    receipt = subprocess.check_output(cmd, env=env, text=True).strip()
    jobid = receipt.split(";")[0]
    if not jobid.isdigit():
        raise RuntimeError(f"Unrecognized sbatch receipt: {receipt!r}")
    hashes = {}
    for folder in ("src", "scripts", "slurm"):
        for path in sorted((ROOT/folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".sbatch"}:
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    row = {"job_id":jobid, "name":a.name, "submitted_at":dt.datetime.now(dt.timezone.utc).isoformat(),
           "command":cmd, "dependencies":a.after, "aftercorr":a.aftercorr, "afterany":a.afterany,
           "source_sha256":hashes}
    with (out/"submissions.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print("SUBMITTED_JOB_ID=" + jobid, flush=True)


if __name__ == "__main__":
    main()
