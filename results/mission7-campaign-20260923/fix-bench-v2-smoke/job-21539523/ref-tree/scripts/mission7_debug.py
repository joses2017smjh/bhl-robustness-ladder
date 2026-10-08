"""Mission7 command, reward and optimization diagnostics. Never releases sensors."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib,json,time
from pathlib import Path
import numpy as np
import mujoco
import torch
from bhl_robust.mission.approach_debug import DebugEnv,BearingController,TranslationController,RecoveryTranslationController,MeasuredRouteController,RecoveryRouteController,physical_sample,classify_fall,command_action,wrap
from bhl_robust.mission.policy import MissionPolicy
from mission7_overnight import pack,write


def describe(values):
    x=np.asarray(values,float)
    return dict(mean=float(x.mean()),median=float(np.median(x)),std=float(x.std()),min=float(x.min()),max=float(x.max()))


def summarize(rows):
    return dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
        success_rate=float(np.mean([r['success'] for r in rows])),falls=sum(r['fall'] for r in rows),
        collisions=sum(r['collisions'] for r in rows),failures=dict(Counter(r['failure'] or 'success' for r in rows)),
        mean_return=float(np.mean([r['return_total'] for r in rows])),
        mean_diagnostic_return=float(np.mean([r['diagnostic_return_total'] for r in rows])),
        final_distance=describe([r['final_goal_distance_m'] for r in rows]),
        minimum_distance=describe([r['minimum_goal_distance_m'] for r in rows]),
        fraction_closed=describe([r['fraction_distance_closed'] for r in rows]),
        initial_distance=describe([r['initial_goal_distance_m'] for r in rows]))


def gait(a,out):
    commands=[(f'forward_{v:.2f}',[v,0,0]) for v in (.1,.2,.3,.4)]
    commands += [(f'lateral_{v:+.2f}',[0,v,0]) for v in (-.35,-.3,-.2,-.1,.1,.2,.3,.35)]
    commands += [(f'yaw_{v:+.2f}',[0,0,v]) for v in (-.35,-.2,.2,.35)]
    durations=(.08,.2,.4,.52,.8,1.,2.)
    if getattr(a,'extra',False):
        commands=[(f'backward_{v:+.2f}',[v,0,0]) for v in (-.3,-.4)]
        commands += [(f'yaw_{v:+.2f}',[0,0,v]) for v in (-1.2,-.8,-.4,.4,.8,1.2)]
        durations=(.8,2.)
    if getattr(a,'endurance',False):
        commands=[('forward_0.30',[.3,0,0]),('backward_-0.30',[-.3,0,0]),
            ('lateral_+0.35',[0,.35,0]),('lateral_-0.30',[0,-.3,0]),
            ('mixed_positive',[.04,.35,0]),('mixed_negative',[.04,-.3,0]),
            ('lateral_turn_positive',[0,.35,.1]),('lateral_turn_negative',[0,-.3,.1])]
        durations=(5.,10.)
    if a.smoke:commands=commands[:1];durations=(.08,)
    rows=[]
    env=DebugEnv(a.repo,out/'gait-cache',stage='approach',seed=3000+a.index,hold_ticks=1)
    for name,command in commands:
        for duration in durations:
            env.reset(0);r,s=env.runner,env.slot
            # Open-floor system identification, distinct from a maze rollout.
            for geom in range(env.model.ngeom):
                name_g=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,geom) or ''
                if name_g.startswith(('wall_','door_','plate_','goal_post')):
                    env.model.geom_contype[geom]=env.model.geom_conaffinity[geom]=0
                    env.model.geom_group[geom]=5
            r.d.qpos[s.qpos_adr:s.qpos_adr+2]=env.rng.normal(0,.015,2)
            mujoco.mj_forward(env.model,r.d)
            warm_ticks=30;command_ticks=round(duration/env.dt);brake_ticks=75
            samples=[]
            for tick in range(warm_ticks+command_ticks+brake_ticks):
                phase=('settle' if tick<warm_ticks else 'brake' if tick>=warm_ticks+command_ticks
                       else 'turn' if command[2] else 'accelerate' if tick<warm_ticks+10 else 'advance')
                cmd=command if warm_ticks<=tick<warm_ticks+command_ticks else [0,0,0]
                targets=env.controller.update(r.observe(0,np.asarray(cmd)))
                r.step([targets])
                if not np.isfinite(r.d.qpos).all() or r.d.warning.number.sum():raise FloatingPointError('gait physics')
                samples.append(physical_sample(env,cmd,phase))
            start=samples[warm_ticks-1];release=samples[warm_ticks+command_ticks-1];end=samples[-1]
            active=samples[warm_ticks:warm_ticks+command_ticks];braking=samples[warm_ticks+command_ticks:]
            direction=np.asarray(command[:2],float);speed=np.linalg.norm(direction)
            direction=direction/speed if speed else np.array([1.,0.])
            # Report displacement in the robot heading at command onset.
            y=start['yaw'];rot=np.array([[np.cos(y),-np.sin(y)],[np.sin(y),np.cos(y)]])
            direction=rot@direction;lateral=np.array([-direction[1],direction[0]])
            delta=np.asarray(release['xy'])-start['xy'];after=np.asarray(end['xy'])-release['xy']
            onsets=[x['time_s']-start['time_s'] for x in active if np.dot(x['velocity'],direction)>max(.03,.2*speed)] if speed else [x['time_s']-start['time_s'] for x in active if abs(x['yaw_rate'])>.2*abs(command[2])]
            stop=None
            for j in range(len(braking)-7):
                if all(np.linalg.norm(x['velocity'])<.04 and abs(x['yaw_rate'])<.08 for x in braking[j:j+8]):
                    stop=braking[j]['time_s']-release['time_s'];break
            commanded_heading=command[2]*duration
            row=dict(command_name=name,command=command,duration_s=duration,reset_seed=3000+a.index,
                commanded_displacement_m=float(speed*duration),along_displacement_m=float(delta@direction),
                actual_speed_m_s=float(np.mean([np.dot(x['velocity'],direction) for x in active])),
                lateral_drift_m=float(delta@lateral),heading_change_rad=wrap(release['yaw']-start['yaw']),
                heading_error_rad=wrap(release['yaw']-start['yaw']-commanded_heading),
                onset_delay_s=onsets[0] if onsets else None,braking_distance_m=float(np.linalg.norm(after)),
                braking_path_m=float(sum(np.linalg.norm(np.asarray(b['xy'])-c['xy']) for c,b in zip([release]+braking,braking))),
                braking_stop_s=stop,overshoot_m=float(delta@direction-speed*duration),
                fall=any(x['tilt']>=.78 for x in samples),final_upright=end['tilt']<.78,
                recovered=any(x['tilt']>=.78 for x in samples) and all(x['tilt']<.78 for x in samples[-25:]),
                fall_classification=classify_fall(samples),trace=samples,arena='open floor; benchmark obstacles disabled only in this fixture')
            rows.append(row);write(out/'result.json',dict(complete=False,episodes=rows))
    write(out/'result.json',dict(complete=True,status='COMPLETED_DIAGNOSTIC',episodes=rows,physics_dt=.0005,gait_dt=.04,current_mission_hold_s=.2))


def gait_extra(a,out):
    rows=[]
    for seed in range(1 if a.smoke else 3):
        folder=out/f'seed-{seed}';folder.mkdir()
        args=argparse.Namespace(**{**vars(a),'index':seed,'extra':True})
        gait(args,folder);rows+=json.loads((folder/'result.json').read_text())['episodes']
    write(out/'result.json',dict(complete=True,status='COMPLETED_DIAGNOSTIC',episodes=rows,
        supported_training_range=dict(vx=[-1,1],vy=[-.5,.5],yaw=[-1.5,1.5]),
        note='Extra yaw commands exceed current Mission7 +/-0.4 cap, but remain inside the frozen gait training range. Open-floor identification only.'))


def gait_endurance(a,out):
    rows=[]
    for seed in range(1 if a.smoke else 3):
        folder=out/f'seed-{seed}';folder.mkdir()
        args=argparse.Namespace(**{**vars(a),'index':seed,'endurance':True})
        gait(args,folder);rows+=json.loads((folder/'result.json').read_text())['episodes']
    write(out/'result.json',dict(complete=True,status='COMPLETED_DIAGNOSTIC',episodes=rows,
        note='48 open-floor trials: eight commands, 5/10 s, three resets. All within Mission7 caps. Separates sustained/mixed-command instability from route transitions and interaction fixtures.'))


def evaluate_policy(a,out,checkpoint,*,count=32,distance=.4):
    env=DebugEnv(a.repo,out/'eval-cache',stage='approach',split='validation',seed=1000,approach_distance=distance)
    rows=[]; policy=None
    for index in range(count):
        obs=env.reset(index)
        if policy is None:
            policy=MissionPolicy(pack(obs,env));policy.load_state_dict(checkpoint['model']);policy.eval()
        if a.smoke:env.max_seconds=.2
        while True:
            with torch.inference_mode():action=policy.act_inference(pack(obs,env))[0].numpy()
            obs,_,done,_=env.step(action)
            if done:break
        rows.append(env.metrics());write(out/'result.json',dict(complete=False,episodes=rows))
    result=dict(complete=True,status='COMPLETED_DIAGNOSTIC',summary=summarize(rows),episodes=rows)
    write(out/'result.json',result)
    return result


def c4(a,out):
    summaries={}
    for update in (75,100,125):
        path=a.campaign/'preserved-c4'/f'model_{update}.pt'
        expected=json.loads((a.campaign/'preserved-c4/sha256.json').read_text())[path.name]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
        folder=out/f'update-{update}';folder.mkdir()
        result=evaluate_policy(a,folder,torch.load(path,map_location='cpu',weights_only=False),count=1 if a.smoke else 32)
        summaries[str(update)]={**result['summary'],'checkpoint_sha256':expected}
    write(out/'result.json',dict(complete=True,status='COMPLETED_DIAGNOSTIC',checkpoints=summaries,split='validation'))


def approach(a,out):
    settings=json.loads((a.campaign/'gait-summary.json').read_text())['controller']
    cases={}
    for name in ('bearing','translation'):
        rows=[]
        for seed in range(1 if a.smoke or name=='bearing' else 2):
            env=DebugEnv(a.repo,out/f'{name}-{seed}-cache',stage='approach',split='validation',seed=1000+seed)
            for index in range(1 if a.smoke else 32):
                env.reset(index)
                controller=BearingController(env,**settings) if name=='bearing' else TranslationController(env)
                if a.smoke:env.max_seconds=.2
                while True:
                    _,_,done,_=env.step(controller.action())
                    if done:break
                row=env.metrics();row.update(layout_index=index,reset_seed=1000+seed)
                rows.append(row);write(out/f'{name}.json',dict(complete=False,episodes=rows))
        cases[name]=dict(summary=summarize(rows),episodes=rows)
        write(out/f'{name}.json',dict(complete=True,**cases[name]))
    stats=cases['translation']['summary']
    write(out/'result.json',dict(complete=True,status='COMPLETED_DIAGNOSTIC',summary=stats,
        cases={k:v['summary'] for k,v in cases.items()},
        reliable=stats['success_rate']>=.75 and stats['falls']/stats['episodes']<=.1,controller='translation',
        bearing_controller=settings,
        reliability_gate='>=75% success and <=10% falls in 64 validation episodes; diagnostic only'))


def rewards(a,out):
    settings=json.loads((a.campaign/'gait-summary.json').read_text())['controller']
    checkpoint=torch.load(a.campaign/'preserved-c4/model_100.pt',map_location='cpu',weights_only=False)
    summaries={};tables={}
    for distance in (.65,.40):
        for control in ('standstill','random','scripted','untrained','c4_100'):
            torch.manual_seed(0);rng=np.random.default_rng(501);rows=[]
            env=DebugEnv(a.repo,out/f'{control}-{distance}-cache',stage='approach',split='validation',seed=1000,approach_distance=distance)
            policy=None
            for index in range(1 if a.smoke else 16):
                obs=env.reset(index);controller=TranslationController(env)
                if policy is None and control in ('untrained','c4_100'):
                    policy=MissionPolicy(pack(obs,env))
                    if control=='c4_100':policy.load_state_dict(checkpoint['model'])
                    policy.eval()
                if a.smoke:env.max_seconds=.2
                while True:
                    if control=='standstill':action=np.zeros(5)
                    elif control=='random':action=rng.normal(0,.5,5)
                    elif control=='scripted':action=controller.action()
                    else:
                        with torch.inference_mode():action=policy.act_inference(pack(obs,env))[0].numpy()
                    obs,_,done,_=env.step(action)
                    if done:break
                rows.append(env.metrics())
                write(out/f'{control}-{distance}.json',dict(complete=False,episodes=rows))
            key=f'{control}-{distance}';summaries[key]=summarize(rows)
            samples=[s for r in rows for s in r['decision_rewards']]
            terms=samples[0]['reward_terms'].keys();absolute=sum(abs(v) for s in samples for v in s['reward_terms'].values())
            tables[key]={term:{**describe([r['reward_terms'].get(term,0.) for r in rows]),
                'decision_nonzero_fraction':float(np.mean([s['reward_terms'][term]!=0 for s in samples])),
                'absolute_contribution_fraction':sum(abs(s['reward_terms'][term]) for s in samples)/max(absolute,1e-9)} for term in terms}
            write(out/f'{key}.json',dict(complete=True,summary=summaries[key],episodes=rows))
            write(out/'result.json',dict(complete=False,summaries=summaries,reward_terms=tables))
    lines=['# Matched controller returns','', 'Validation layouts 0–15 and identical reset seeds within each spawn distance. Original sparse scoring; C4 update 100 is an easier-stage checkpoint.','', '| Start m | Controller | Success | Mean return | Median final distance m | Falls |','|---:|---|---:|---:|---:|---:|']
    for key,r in summaries.items():
        control,distance=key.rsplit('-',1);lines.append(f"| {distance} | {control} | {r['successes']}/{r['episodes']} | {r['mean_return']:.4f} | {r['final_distance']['median']:.3f} | {r['falls']} |")
    (out/'returns.md').write_text('\n'.join(lines)+'\n')
    write(out/'result.json',dict(complete=True,status='COMPLETED_DIAGNOSTIC',summaries=summaries,reward_terms=tables,
        interpretation='Standstill is not a global reward optimum when reliable completion is possible; compare its expected return with the actual movement controllers. No upright/survival reward exists.'))


def approach_recovery(a,out):
    rows=[]
    for seed in range(1 if a.smoke else 2):
        env=DebugEnv(a.repo,out/f'controller-{seed}-cache',stage='approach',split='validation',seed=1000+seed)
        for index in range(1 if a.smoke else 32):
            env.reset(index);controller=RecoveryTranslationController(env)
            if a.smoke:env.max_seconds=.2
            while True:
                _,_,done,_=env.step(controller.action())
                if done:break
            row=env.metrics();row.update(layout_index=index,reset_seed=1000+seed,recovery_pulse_used=controller.kicked)
            rows.append(row);write(out/'result.json',dict(complete=False,episodes=rows))
    stats=summarize(rows)
    write(out/'result.json',dict(complete=True,status='COMPLETED_DIAGNOSTIC',summary=stats,episodes=rows,
        reliable=stats['success_rate']>=.75 and stats['falls']/stats['episodes']<=.1,
        controller='translation with one 0.3 m/s, 0.4 s forward pulse on stalled positive-lateral starts',
        reliability_gate='unchanged: >=75% success and <=10% falls over the same 64 validation episodes'))


def fullroute(a,out):
    from mission7_overnight import InteractionController
    summaries={};falls=[]
    for variant in (('recovery',) if a.mode=='fullroute_recovery' else ('legacy','measured')):
        for stage in ('doors','transport'):
            rows=[];env=DebugEnv(a.repo,out/f'{variant}-{stage}-cache',stage=stage,split='validation',seed=1000)
            for index in range(1 if a.smoke else 16):
                env.reset(index);controller=(InteractionController if variant=='legacy' else RecoveryRouteController if variant=='recovery' else MeasuredRouteController)(env)
                if a.smoke:env.max_seconds=.2
                while True:
                    action=controller.action()
                    if variant=='legacy':
                        env.phase='acquire' if action[4]>.5 else 'release' if action[4]<-.5 else 'switch' if action[3]>.5 else 'advance'
                    _,_,done,_=env.step(action)
                    if done:break
                row=env.metrics();row.update(layout_index=index,variant=variant,controller_route_index=controller.waypoint)
                rows.append(row)
                if row['fall_classification']:
                    falls.append(dict(stage=stage,variant=variant,layout_index=index,action_hold_s=.2,**row['fall_classification']))
                write(out/f'{variant}-{stage}.json',dict(complete=False,episodes=rows))
                write(out/'falls.json',dict(complete=False,falls=falls))
            summaries[f'{variant}-{stage}']=summarize(rows)
            write(out/f'{variant}-{stage}.json',dict(complete=True,summary=summaries[f'{variant}-{stage}'],episodes=rows))
    write(out/'falls.json',dict(complete=True,falls=falls))
    write(out/'result.json',dict(complete=True,status='COMPLETED_DIAGNOSTIC',summaries=summaries,
        unchanged_benchmark=True,split='validation',comparison='16 paired layouts; same seed; only scripted navigation commands differ'))


fullroute_recovery=fullroute

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('gait','gait_extra','gait_endurance','c4','approach','approach_recovery','rewards','fullroute','fullroute_recovery','fall_replay','train','train_exploration'))
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    if not a.campaign.resolve().is_relative_to(a.repo.resolve()):p.error('stay inside repo')
    out=a.campaign/(f'gait-{a.index}' if a.mode=='gait' else f'P{a.index+1}' if a.mode=='train' else f'P{a.index+5}' if a.mode=='train_exploration' else a.mode)
    if out.exists():p.error('refusing to overwrite diagnostic output')
    out.mkdir(parents=True)
    if a.mode in ('gait','gait_extra','gait_endurance','c4','approach','approach_recovery','rewards','fullroute','fullroute_recovery'):globals()[a.mode](a,out)
    elif a.mode in ('train','train_exploration'):
        from mission7_debug_train import train
        train(a,out)
    elif a.mode=='fall_replay':
        from mission7_debug_replay import replay
        replay(a,out)
    else:raise NotImplementedError('mode not qualified yet')
    print('MISSION7_DEBUG_COMPLETE '+a.mode,flush=True)


if __name__=='__main__':main()
