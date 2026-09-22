"""One parallelized PlateSafe Transport or Doors route episode."""
from pathlib import Path
import json, os

from bhl_robust.mission.approach_debug import DebugEnv, PlateSafeRouteController

root = Path(os.environ['MISSION7_REPO'])
out = Path(os.environ['MISSION7_OUT'])
stage = os.environ.get('MISSION7_STAGE', 'transport')
index = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
out.mkdir(parents=True, exist_ok=True)
env = DebugEnv(root, out / f'cache-{stage}-{index}', stage=stage, split='validation', seed=1000)
env.reset(index)
controller = PlateSafeRouteController(env, contact_hold_s=1.0)
while True:
    _, _, done, _ = env.step(controller.action())
    if done:
        break
row = env.metrics()
row.update(layout_index=index, stage=stage,
           controller='plate_safe_contact_brake_1.0s',
           plate_contacts_seen=sorted(set(controller.plate_contacts_seen)))
(out / f'{stage}-{index}.json').write_text(json.dumps(row, indent=2) + '\n')
print(json.dumps({'stage': stage, 'index': index, 'success': row['success'],
                  'failure': row['failure']}, sort_keys=True), flush=True)
