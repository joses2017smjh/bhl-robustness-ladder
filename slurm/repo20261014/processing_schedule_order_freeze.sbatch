#!/bin/bash
# CPU validation and freeze of the order-confirmation-20261014 design.
# Usage: sbatch ... processing_schedule_order_freeze.sbatch <expected-commit> [attempt-tag, default v1]
# Runs the processing-schedule freeze tests plus the new analysis tests,
# replays the new collector and analysis on the retained v1 collector inputs
# (must reproduce v1's published collection), writes the scientific protocol
# and runs `processing_schedule_dispatch.py prepare`. Never submits a GPU job.
#SBATCH --signal=B:TERM@30
set -euo pipefail
EXPECTED_COMMIT=${1:?expected frozen source commit required}
ATTEMPT=${2:-v1}
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RESEARCH=/nfs/hpc/share/sanchej7/humanoid-research-20261011
SOURCE=$RESEARCH/followups-20261014-order/repo
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
DESIGN=order-confirmation-20261014
VALIDATION=$RESEARCH/processing-schedule-order-validation-$ATTEMPT-20261014
PROTOCOL_DIR=$RESEARCH/processing-schedule-order-protocol-$ATTEMPT-20261014
CAMPAIGN=$RESEARCH/processing-schedule-order-$ATTEMPT-20261014
V1=$RESEARCH/processing-schedule-v1-20261012
cd "$SOURCE"
export PYTHONPATH=$SOURCE/src
[[ $(git rev-parse HEAD) == "$EXPECTED_COMMIT" ]]
[[ -z $(git status --porcelain) ]]
[[ ! -e $PROTOCOL_DIR && ! -e $CAMPAIGN ]]
mkdir "$VALIDATION"
set +e
"$PY" -m pytest -q -p no:cacheprovider tests/test_processing_schedule.py tests/test_processing_schedule_campaign.py \
    tests/test_processing_schedule_dispatch.py tests/test_processing_schedule_order_analysis.py \
    tests/test_navigation_sensor_controls.py tests/test_euroc_native_validation.py tests/test_native_orb.py \
    tests/test_native_orb_inertial.py tests/test_native_lio.py tests/test_native_navigation_campaign.py \
    tests/test_native_stereo_navigation_campaign.py tests/test_h34_campaign.py tests/test_h34_collect.py \
    tests/test_h34_finalize.py > "$VALIDATION/pytest.log" 2>&1
TEST_RC=$?
"$PY" scripts/bench/processing_schedule_campaign.py --collect "$V1/collector-inputs" \
    --qualified-smoke "$V1/qualified-smoke.json" --output "$VALIDATION/v1-replay-collection.json" > "$VALIDATION/v1-replay.log" 2>&1 &&
"$PY" scripts/bench/processing_schedule_order_analysis.py --collection "$VALIDATION/v1-replay-collection.json" \
    --output "$VALIDATION/v1-replay-analysis" >> "$VALIDATION/v1-replay.log" 2>&1
REPLAY_RC=$?
set -e
tail -3 "$VALIDATION/pytest.log"
"$PY" - "$TEST_RC" "$REPLAY_RC" "$VALIDATION" "$SOURCE" <<'PY'
from datetime import datetime, timezone
import json, os, pathlib, subprocess, sys
test_rc, replay_rc, validation, source = int(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
replay = dict(status="NOT_RUN")
if replay_rc == 0:
    new = json.loads((validation / "v1-replay-collection.json").read_text())
    old = json.loads((source / "results/processing-schedule-20261012/collection.json").read_text())
    top = {k: new[k] == old[k] for k in old if k not in ("paired", "source_results")}
    paired = {k: new["paired"][k] == old["paired"][k] for k in old["paired"] if k != "pairs"}
    pairs = len(old["paired"]["pairs"]) == len(new["paired"]["pairs"]) and all(
        {k: b[k] for k in a} == a for a, b in zip(old["paired"]["pairs"], new["paired"]["pairs"]))
    lineage = [a["sha256"] for a in old["source_results"]] == [b["sha256"] for b in new["source_results"]]
    analysis = json.loads((validation / "v1-replay-analysis/analysis.json").read_text())
    lidar = analysis["arms"]["lidar_inertial"]["goals"]
    replay = dict(top_level_fields_equal=top, paired_summary_equal=paired, pair_fields_equal=pairs, source_result_hashes_equal=lineage,
                  order_audit=new["order_audit"], analysis_verdict=analysis["verdict"]["status"],
                  lidar_goals=dict(legacy=lidar["legacy_goals"], bounded=lidar["bounded_goals"], pairs=lidar["pairs"],
                                   mcnemar_p=lidar["mcnemar"]["p_value"]))
    replay["status"] = "REPRODUCED" if all(top.values()) and all(paired.values()) and pairs and lineage else "MISMATCH"
receipt = dict(schema="bhl-processing-schedule-order-validation-v1", job_id=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename,
               source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               source_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
               pytest_exit_code=test_rc, pytest_summary=(validation / "pytest.log").read_text().strip().splitlines()[-1],
               v1_replay_exit_code=replay_rc, v1_replay=replay, finished_utc=datetime.now(timezone.utc).isoformat())
with (validation / "validation.json").open("x") as stream:
    json.dump(receipt, stream, indent=2); stream.write("\n")
print(json.dumps(dict(pytest=receipt["pytest_summary"], replay=replay["status"]), indent=2))
if test_rc or replay_rc or replay["status"] != "REPRODUCED":
    raise SystemExit("validation failed; nothing frozen")
PY
[[ -z $(git status --porcelain) ]]
mkdir "$PROTOCOL_DIR"
"$PY" scripts/bench/processing_schedule_campaign.py --write-protocol "$PROTOCOL_DIR/scientific-protocol.json" --design "$DESIGN"
"$PY" scripts/bench/processing_schedule_dispatch.py prepare --repo "$SOURCE" \
    --scientific-protocol "$PROTOCOL_DIR/scientific-protocol.json" \
    --profile-episode "$RESEARCH/processing-schedule-profile-input-20261012/straight-s730000-nominal-stereo" \
    --campaign "$CAMPAIGN" > "$VALIDATION/prepare.json"
"$PY" - "$CAMPAIGN" "$PROTOCOL_DIR/scientific-protocol.json" "$VALIDATION" "$EXPECTED_COMMIT" "$DESIGN" <<'PY'
from datetime import datetime, timezone
import hashlib, json, pathlib, sys
campaign, protocol, validation, commit, design = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), sys.argv[4], sys.argv[5]
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
intake = json.loads((campaign / "intake.json").read_text())
launch = json.loads((campaign / "protocol.json").read_text())
scientific = json.loads(protocol.read_text())
checks = dict(status=intake["status"] == "FROZEN_NOT_SUBMITTED", commit=intake["source_commit"] == commit,
              clean=intake["source_status"] == "", design=intake["design"] == launch["design"] == scientific["design"] == design,
              protocol_hash=intake["scientific_protocol_sha256"] == digest(protocol),
              counts=(intake["new_scored_episodes"], intake["fresh_smoke_episodes"], intake["unscored_warmup_episodes"], intake["declared_jobs"]) == (96, 6, 50, 27),
              archive=intake["archive_sha256"] == digest(campaign / "frozen-campaign.tar.gz"),
              not_submitted=not (campaign / "submission-start.json").exists())
receipt = dict(schema="bhl-processing-schedule-order-freeze-v1", checks=checks, campaign=str(campaign), design=design,
               archive_sha256=intake["archive_sha256"], archive_bytes=intake["archive_bytes"],
               scientific_protocol=str(protocol), scientific_protocol_sha256=digest(protocol),
               launch_protocol_sha256=intake["protocol_sha256"], source_commit=intake["source_commit"],
               quota_before_freeze=intake["quota_before_freeze"],
               submit_command=["/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python", "scripts/bench/processing_schedule_dispatch.py",
                               "submit", "--campaign", str(campaign)],
               frozen_utc=datetime.now(timezone.utc).isoformat())
with (validation / "freeze.json").open("x") as stream:
    json.dump(receipt, stream, indent=2); stream.write("\n")
print(json.dumps(receipt, indent=2))
if not all(checks.values()):
    raise SystemExit("frozen intake failed verification")
PY
