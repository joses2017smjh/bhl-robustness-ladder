#!/usr/bin/env python3
"""One-shot allocated reporting; output only, no simulation or GitHub mutations."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


root = Path(sys.argv[1]).resolve()
expected_report, expected_runner = sys.argv[2:4]
if digest(root / 'report.py') != expected_report or digest(__file__) != expected_runner:
    raise SystemExit('Frozen reporting helper changed; do not produce a report')
plan = json.loads((root / 'report-plan.json').read_text())
job = os.environ.get('SLURM_JOB_ID', '')
if not job.isdecimal():
    raise SystemExit('Allocated Slurm reporting job required')
output = root / ('outputs-' + job)
command = [sys.executable, str(root / 'report.py'), '--campaign', plan['campaign'], '--output', str(output)]
write(root / ('attempt-' + job + '-start.json'), dict(started_utc=datetime.now(timezone.utc).isoformat(),
    job_id=job, parent_job_id=plan['parent_job_id'], hostname=os.uname().nodename,
    report_sha256=expected_report, runner_sha256=expected_runner, command=command))
result = subprocess.run(command)
write(root / ('attempt-' + job + '-result.json'), dict(finished_utc=datetime.now(timezone.utc).isoformat(),
    job_id=job, parent_job_id=plan['parent_job_id'], status='COMPLETE' if result.returncode == 0 else 'FAILED',
    returncode=result.returncode, output=str(output), report_sha256=expected_report,
    report_evidence_sha256=digest(output / 'report-evidence.json') if (output / 'report-evidence.json').exists() else None,
    scope='Independent compact-evidence reporting only; no simulations, parameter changes, retries or GitHub writes'))
raise SystemExit(result.returncode)
