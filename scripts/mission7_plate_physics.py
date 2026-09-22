"""Contact-parameter intervention on the ten retained plate-fall replays."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from bhl_robust.mission.approach_debug import DebugEnv, physical_sample
from mission7_overnight import write


def run(a, out):
    source=json.loads((a.campaign/'fullroute/legacy-doors.json').read_text())
    selected=[(i,e) for i,e in enumerate(source['episodes']) if e.get('fall')]
    if a.smoke: selected=selected[:1]
    env=DebugEnv(a.repo,out/'cache',stage='doors',split='validation',seed=1000)
    rows=[]
    next_reset=0
    for index,episode in selected:
        for reset_index in range(next_reset,index+1):
            env.reset(reset_index)
        next_reset=index+1
        r=env.runner
        for geom in range(env.model.ngeom):
            name=__import__('mujoco').mj_id2name(env.model,__import__('mujoco').mjtObj.mjOBJ_GEOM,geom) or ''
            if name.startswith('plate_'):
                # Geometry, activation force threshold, and plate locations are
                # unchanged; only sliding friction is an explicit physics test.
                env.model.geom_friction[geom,0]=.20
        samples=[]
        for ref in episode['diagnostic_trace']:
            env.state.open=[t is not None and r.d.time+1e-8>=t for t in episode['activation_s']];env._gates()
            command=np.asarray(ref['command']);r.step([env.controller.update(r.observe(0,command))])
            samples.append(physical_sample(env,command,ref['phase']))
        falls=[s for s in samples if s['tilt']>=.78]
        rows.append(dict(layout_index=index,fall=bool(falls),first_fall_s=falls[0]['time_s'] if falls else None,
                         maximum_tilt=max(s['tilt'] for s in samples),replay_elapsed_s=samples[-1]['time_s']))
        write(out/'result.json',dict(complete=False,episodes=rows))
    write(out/'result.json',dict(complete=True,status='COMPLETED_DIAGNOSTIC',intervention='plate sliding friction 1.0 -> 0.20; geometry and activation unchanged',episodes=rows,
        summary=dict(episodes=len(rows),falls=sum(x['fall'] for x in rows),no_fall=sum(not x['fall'] for x in rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();a.repo=a.repo.resolve();a.campaign=a.campaign.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True);run(a,a.out);print('MISSION7_PLATE_PHYSICS_COMPLETE',flush=True)
