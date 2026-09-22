"""Paired geometry interventions on exact recorded door-fall command sequences."""
import json
import numpy as np
import mujoco
from bhl_robust.mission.approach_debug import DebugEnv,physical_sample,wrap
from mission7_overnight import write


def replay(a,out):
    original=json.loads((a.campaign/'fullroute/legacy-doors.json').read_text())
    assert original['complete']
    results=[]
    for variant in ('unchanged','no_plates','open_floor'):
        env=DebugEnv(a.repo,out/(variant+'-cache'),stage='doors',split='validation',seed=1000)
        for index,episode in enumerate(original['episodes']):
            # Reset every preceding layout, including non-falls, to reproduce
            # the original simulator reset RNG and per-episode sensor seeds.
            env.reset(index)
            if not episode['fall']:continue
            r=env.runner
            for geom in range(env.model.ngeom):
                name=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,geom) or ''
                remove=(variant=='no_plates' and name.startswith('plate_')) or (
                    variant=='open_floor' and name.startswith(('wall_','door_','plate_','goal_post')))
                if remove:env.model.geom_contype[geom]=env.model.geom_conaffinity[geom]=0
            trace=episode['diagnostic_trace'][:5] if a.smoke else episode['diagnostic_trace']
            samples=[];error=0.
            for reference in trace:
                env.state.open=[t is not None and r.d.time+1e-8>=t for t in episode['activation_s']]
                env._gates()
                command=np.asarray(reference['command'])
                r.step([env.controller.update(r.observe(0,command))])
                sample=physical_sample(env,command,reference['phase'])
                sample['plate_contacts']=sorted({mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,int(g))
                    for contact in r.d.contact for g in (contact.geom1,contact.geom2)
                    if (mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or '').startswith('plate_')})
                samples.append(sample)
                assert abs(sample['time_s']-reference['time_s'])<1e-7
                if not np.isfinite(r.d.qpos).all() or r.d.warning.number.sum():raise FloatingPointError('replay physics')
                error=max(error,float(np.linalg.norm(np.asarray(sample['xy'])-reference['xy'])),
                          abs(sample['tilt']-reference['tilt']),abs(wrap(sample['yaw']-reference['yaw'])))
            first_fall=next((s['time_s'] for s in samples if s['tilt']>=.78),None)
            result=dict(variant=variant,layout_index=index,layout_seed=episode['layout']['seed'],
                original_fall_s=episode['elapsed_s'],replay_seconds=samples[-1]['time_s'],
                fall=first_fall is not None,first_fall_s=first_fall,final_tilt=samples[-1]['tilt'],
                maximum_pose_difference=error,matched_original=error<1e-6 if variant=='unchanged' else None,
                plate_contact_frames=sum(bool(s['plate_contacts']) for s in samples),trace=samples)
            results.append(result);write(out/'episodes.json',dict(complete=False,episodes=results))
            if a.smoke:break
        if variant=='unchanged' and not all(r['matched_original'] for r in results):
            write(out/'episodes.json',dict(complete=True,episodes=results))
            write(out/'result.json',dict(complete=True,status='INVALID_REPLAY',unchanged_replay_matches=False,
                interpretation_allowed=False,reason='Unchanged geometry did not reproduce the recorded trajectory; no interventions performed.'))
            return
    matched=all(r['matched_original'] for r in results if r['variant']=='unchanged')
    summaries={v:dict(trials=sum(r['variant']==v for r in results),falls=sum(r['fall'] for r in results if r['variant']==v))
               for v in ('unchanged','no_plates','open_floor')}
    write(out/'episodes.json',dict(complete=True,episodes=results))
    write(out/'result.json',dict(complete=True,status='COMPLETED_DIAGNOSTIC',unchanged_replay_matches=matched,
        summaries=summaries,interpretation_allowed=matched,
        limitation='Open-loop recorded commands and recorded gate-opening times, ending at the original fall time. This is a geometry intervention, not a new route-success evaluation. Benchmark geometry is unchanged.'))
