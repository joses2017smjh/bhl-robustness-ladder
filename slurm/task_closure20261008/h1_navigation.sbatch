#!/bin/bash
#SBATCH --job-name=bhl-h1-confirm
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
# CPU-only, no GPU allocation/training/Mission 7 episodes.
# Submit using sbatch --output/--error <persistent-home>/slurm-%j.out \
#   h1_navigation.sbatch <frozen archive> <archive sha256> <persistent dir>
set -euo pipefail
ARCHIVE=$1
EXPECTED_SHA=$2
PERSIST=$3
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
CAMPAIGN=$(mktemp -d /tmp/bhl-h1-confirm-${SLURM_JOB_ID:-local}-XXXXXX)
mkdir -p "$PERSIST"
export H1_CAMPAIGN="$CAMPAIGN" H1_PERSIST="$PERSIST" H1_ARCHIVE="$ARCHIVE" H1_ARCHIVE_SHA="$EXPECTED_SHA"

finish() {
    local exit_status=$?
    trap - EXIT
    set +e
    # Preserve the complete episode evidence even for a partial failed run.
    if [ -d "$CAMPAIGN/runs" ]; then
        tar -czf "$PERSIST/episodes-${SLURM_JOB_ID:-local}.tar.gz.partial" -C "$CAMPAIGN" runs
        if [ $? -eq 0 ]; then
            mv "$PERSIST/episodes-${SLURM_JOB_ID:-local}.tar.gz.partial" "$PERSIST/episodes-${SLURM_JOB_ID:-local}.tar.gz"
        fi
    fi
    export H1_EXIT_STATUS="$exit_status"
    "$PY" - <<'PY'
import datetime, hashlib, json, os, pathlib, platform
c=pathlib.Path(os.environ['H1_CAMPAIGN']); p=pathlib.Path(os.environ['H1_PERSIST'])
report=c/'confirmation_report.json'
if report.is_file():
    data=json.loads(report.read_text())
else:
    data={'status':'INCOMPLETE','problems':['Job did not produce the final validated report.'],
          'scope':'Frozen H1 confirmation; no success inferred from scheduler completion.'}
target=p/'confirmation_report.json'
if not target.exists():
    with target.open('x') as f: json.dump(data,f,indent=2);f.write('\n')
files={x.name:{'sha256':hashlib.sha256(x.read_bytes()).hexdigest(),'bytes':x.stat().st_size}
       for x in p.iterdir() if x.is_file() and (x.suffix in {'.json','.gz'} or x.name.endswith('.sbatch'))}
receipt={'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'job_id':os.getenv('SLURM_JOB_ID'),'host':platform.node(),'exit_status':int(os.environ['H1_EXIT_STATUS']),
         'status':data['status'],'raw_episode_records':len(list((c/'runs').glob('*/seed*.json'))),
         'planned_episodes':576,'files':files,'local_campaign':str(c),
         'scope':'CPU-only NavGym confirmation, no training, no Mission 7 episodes, no hardware.'}
with (p/'completion.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
PY
    exit "$exit_status"
}
trap finish EXIT

"$PY" - <<'PY'
import hashlib,os,pathlib
p=pathlib.Path(os.environ['H1_ARCHIVE'])
if hashlib.sha256(p.read_bytes()).hexdigest()!=os.environ['H1_ARCHIVE_SHA']:
    raise SystemExit('Frozen archive SHA256 mismatch; refusing execution.')
print('Frozen archive SHA256 verified')
PY
tar -xzf "$ARCHIVE" -C "$CAMPAIGN"
RUNNER=$CAMPAIGN/snapshot/scripts/bench/navigation_confirmation.py
"$PY" - "$CAMPAIGN" <<'PY'
import importlib.util,pathlib,sys,platform,json
c=pathlib.Path(sys.argv[1]); runner=c/'snapshot/scripts/bench/navigation_confirmation.py'
s=importlib.util.spec_from_file_location('h1',runner);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
p=m.validate(c)
assert p['max_episodes']==576 and p['max_concurrent_evaluators']==2
assert p['seeds']==list(range(120000,120048)) and p['tau_s']==0.2
print(json.dumps({'preflight':'PASS','host':platform.node(),'python':sys.version,'input_files':len(p['input_sha256'])}))
PY
set +e
"$PY" "$RUNNER" run --campaign "$CAMPAIGN"
RUN_RC=$?
"$PY" "$RUNNER" summary --campaign "$CAMPAIGN"
SUMMARY_RC=$?
set -e
if [ "$RUN_RC" -ne 0 ] || [ "$SUMMARY_RC" -ne 0 ]; then
    exit 1
fi
