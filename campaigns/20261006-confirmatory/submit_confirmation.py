"""Submit once; dependent IDs always come from Slurm's actual receipts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from dr_confirmatory import validate
from navgym_confirmatory import validate_inputs, write_new


def main():
    campaign = Path(__file__).resolve().parent
    if (campaign / 'submission.json').exists():
        raise FileExistsError('already submitted; inspect submission.json, do not duplicate arrays')
    validate(campaign)
    validate_inputs(campaign)
    smoke = json.loads((campaign / 'smoke/verification.json').read_text())
    if smoke['status'] != 'PASS':
        raise ValueError('deployment smoke did not pass')
    (campaign / 'jobs').mkdir(exist_ok=True)
    submitted = {}
    try:
        for key, file, dependency in (
            ('locomotion', 'dr_confirmation.sbatch', None),
            ('navigation', 'navigation_confirmation.sbatch', 'locomotion'),
            ('final_verdict', 'finalize_confirmation.sbatch', 'both')):
            command = ['sbatch', '--parsable']
            if dependency:
                ids = [submitted['locomotion']] if dependency == 'locomotion' else [submitted['locomotion'], submitted['navigation']]
                command += ['--dependency=afterany:' + ':'.join(ids)]
            command += [str(campaign / file)]
            submitted[key] = subprocess.check_output(command, text=True).strip().split(';')[0]
    finally:
        # Even a partial submission has an immutable receipt and cannot be blindly duplicated.
        write_new(campaign / 'submission.json', {'submitted_utc': datetime.now(timezone.utc).isoformat(),
                  'jobs': submitted, 'state': 'SUBMITTED' if len(submitted) == 3 else 'PARTIAL_SUBMISSION',
                  'fresh_results': 'No pending/running job is considered successful evidence.'})
    print(json.dumps(submitted))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
