#!/bin/bash
#SBATCH --signal=B:TERM@30
# CPU-only: validate the frozen stereo-recovery source, write the scientific
# protocol and freeze (dispatch prepare). Never submits the GPU campaign.
set -euo pipefail
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RESEARCH=/nfs/hpc/share/sanchej7/humanoid-research-20261011
FOLLOW=$RESEARCH/followups-20261014-stereo
SOURCE=$FOLLOW/repo
COMMIT=7638a6479d4e5487c54ab359f92a698777784626
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
CAMPAIGN=$FOLLOW/stereo-recovery-v2-20261014
cd "$SOURCE"
export PYTHONPATH=$SOURCE/src
[[ $(git rev-parse HEAD) == $COMMIT* ]]
[[ -z $(git status --porcelain) ]]
set +e
"$PY" -m pytest -q -p no:cacheprovider tests/test_processing_schedule.py tests/test_processing_schedule_campaign.py tests/test_processing_schedule_dispatch.py tests/test_navigation_sensor_controls.py tests/test_euroc_native_validation.py tests/test_native_orb.py tests/test_native_orb_inertial.py tests/test_native_lio.py tests/test_native_navigation_campaign.py tests/test_native_stereo_navigation_campaign.py tests/test_h34_campaign.py tests/test_h34_collect.py tests/test_h34_finalize.py tests/test_orb_atlas_telemetry_build.py tests/test_stereo_relocalization.py tests/test_stereo_recovery_campaign.py
TEST_RC=$?
set -e
mkdir -p "$FOLLOW/validation"
"$PY" - "$TEST_RC" "$FOLLOW/validation" <<'PY'
from datetime import datetime,timezone
import json,os,pathlib,subprocess,sys
p=pathlib.Path(sys.argv[2])/f"stereo-recovery-validation-{os.environ['SLURM_JOB_ID']}.json"
with p.open('x') as f:
 json.dump({'job_id':os.environ['SLURM_JOB_ID'],'host':os.uname().nodename,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'pytest_exit_code':int(sys.argv[1]),'finished_utc':datetime.now(timezone.utc).isoformat()},f,indent=2);f.write('\n')
PY
if [[ $TEST_RC != 0 ]]; then exit "$TEST_RC"; fi
[[ -z $(git status --porcelain) ]]
mkdir -p "$FOLLOW/protocol-v2"
"$PY" scripts/bench/stereo_recovery_campaign.py --write-protocol "$FOLLOW/protocol-v2/scientific-protocol.json"
"$PY" scripts/bench/stereo_recovery_dispatch.py prepare --repo "$SOURCE" \
  --scientific-protocol "$FOLLOW/protocol-v2/scientific-protocol.json" \
  --orb-runtime "$FOLLOW/native-atlas-build-v1/runtime.tar.gz" \
  --orb-build-result "$FOLLOW/native-atlas-build-v1/build-result.json" \
  --campaign "$CAMPAIGN"
[[ -z $(git status --porcelain) ]]
