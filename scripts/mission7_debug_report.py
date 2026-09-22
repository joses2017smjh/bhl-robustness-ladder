"""Readable, incremental outcome plots; raw experiment files remain unchanged."""
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mission7_overnight import write


def episode_records(path):
    if path.suffix=='.jsonl':
        with path.open() as f:
            for line in f:
                try:yield json.loads(line)
                except json.JSONDecodeError:break  # An actively written last line.
    else:
        yield from json.loads(path.read_text()).get('episodes',[])


def optimization_report(out):
    keys=('surrogate','value_function','entropy','analytic_kl_post_update',
          'clip_fraction_post_update','explained_variance_after',
          'gradient_norm_before_clip_mean','learning_rate')
    fig,axes=plt.subplots(4,2,figsize=(12,13));summaries={}
    for name in ('P1','P2','P3','P4','P5','P6'):
        path=out/name/'learning.jsonl'
        if not path.exists():continue
        rows=list(episode_records(path))
        if not rows:continue
        summary=dict(updates=rows[-1]['update'],training_successes=rows[-1]['training_successes'],
                     simulated_training_s=rows[-1]['simulated_training_s'],windows={})
        for lo,hi in ((1,50),(51,100),(101,150),(151,200)):
            window=[r for r in rows if lo<=r['update']<=hi]
            if not window:continue
            stats={}
            for key in keys:
                values=[r[key] for r in window if r.get(key) is not None]
                if values:stats[key]=dict(mean=float(np.mean(values)),median=float(np.median(values)),min=min(values),max=max(values))
            stats['action_std_range']=[min(min(r['action_std']) for r in window),max(max(r['action_std']) for r in window)]
            summary['windows'][f'{lo}-{hi}']=stats
        summaries[name]=summary
        for ax,key in zip(axes.flat,keys):
            ax.plot([r['update'] for r in rows],[r.get(key) for r in rows],label=name,linewidth=.8)
            ax.set(xlabel='Update',ylabel=key);ax.grid(alpha=.2)
    if summaries:
        for index in (1,6,7):axes.flat[index].set_yscale('log')
        axes.flat[3].set_yscale('symlog',linthresh=.01)
        axes.flat[5].set_yscale('symlog',linthresh=1.)
        axes.flat[4].set_ylim(0,1)
        axes[0,0].legend();fig.tight_layout();fig.savefig(out/'new-ppo-dynamics.png',dpi=160);fig.savefig(out/'new-ppo-dynamics.pdf')
    plt.close(fig)
    write(out/'new-ppo-dynamics.json',summaries)


def approach_paths(out):
    labels={(1,0):'+x',(-1,0):'−x',(0,1):'+y',(0,-1):'−y'}
    fig,axes=plt.subplots(2,2,figsize=(10,9));groups={}
    for axis,(direction,label) in zip(axes.flat,labels.items()):
        axis.add_patch(plt.Circle((0,0),.36,color='green',alpha=.08))
        for sign in (-1,1):axis.add_patch(plt.Circle((.52,sign*.35),.045,color='black'))
        for name,path,color in [('translation',out/'approach/translation.json','tab:orange'),
                                ('recovery',out/'approach_recovery/result.json','tab:blue')]:
            es=json.loads(path.read_text())['episodes'];selected=[]
            for e in es:
                route=e['layout']['route'];d=tuple(route[-1][i]-route[-2][i] for i in range(2))
                if d!=direction:continue
                selected.append(e);goal=np.asarray(route[-1])*e['layout']['cell_m']
                points=np.asarray([r['xy'] for r in e['diagnostic_trace']])-goal
                axis.plot(points[:,0],points[:,1],color=color,alpha=.35,linewidth=.7)
            successes=sum(e['success'] for e in selected)
            axis.plot([],[],color=color,label=f'{name}: {successes}/{len(selected)}')
            groups[f'{name}/{label}']=dict(episodes=len(selected),successes=successes,
                median_path_m=float(np.median([e['distance_m'] for e in selected])),
                median_time_closest_s=float(np.median([e['time_closest_s'] for e in selected])))
        axis.set(xlabel='x relative to goal (m)',ylabel='y relative to goal (m)',title=f'Goal direction {label}',xlim=(-.9,.9),ylim=(-.9,.9),aspect='equal')
        axis.legend(fontsize=8);axis.grid(alpha=.2)
    fig.suptitle('Paired Approach control: same 32 layouts × two reset seeds')
    fig.tight_layout();fig.savefig(out/'approach/controller-paths.png',dpi=170);fig.savefig(out/'approach/controller-paths.pdf');plt.close(fig)
    write(out/'approach/directional-comparison.json',groups)


def diagnostic_rewards(out):
    def corr(x,y):
        return float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-10 and np.std(y)>1e-10 else None
    result={}
    lines=['# New training reward decomposition','',
           'Completed training episodes only; excludes unfinished episodes at rollout boundaries. Frequencies are per high-level decision.',
           '', '| Cell | Term | Mean | Median | Std | Nonzero decisions | Absolute contribution | Corr(final distance) | Corr(goal entry) |',
           '|---|---|---:|---:|---:|---:|---:|---|---|']
    for name in ('P1','P2','P3','P4','P5','P6'):
        path=out/name/'episodes.jsonl'
        if not path.exists():continue
        episodes=list(episode_records(path))
        if not episodes:continue
        values=defaultdict(list);nonzero=Counter();absolute=Counter();decisions=0
        for e in episodes:
            for key,value in {**e['reward_terms'],'privileged_distance_progress':e['privileged_distance_progress']}.items():values[key].append(value)
            assert np.isclose(sum(e['reward_terms'].values())+e['privileged_distance_progress'],e['diagnostic_return_total'])
            if name!='P1':assert np.isclose(e['privileged_distance_progress'],10*(e['initial_goal_distance_m']-e['final_goal_distance_m']))
            for step in e['decision_rewards']:
                decisions+=1
                for key,value in step['reward_terms'].items():nonzero[key]+=int(value!=0);absolute[key]+=abs(value)
        denominator=sum(absolute.values());terms={}
        for key,v in values.items():
            terms[key]=dict(mean=float(np.mean(v)),median=float(np.median(v)),std=float(np.std(v)),
                nonzero_decision_fraction=nonzero[key]/decisions,absolute_contribution_fraction=absolute[key]/max(denominator,1e-9),
                correlation_final_distance=corr(v,[e['final_goal_distance_m'] for e in episodes]),
                correlation_goal_entry=corr(v,[e['goal_region_entered'] for e in episodes]))
            r=terms[key]
            lines.append(f"| {name} | {key} | {r['mean']:.4f} | {r['median']:.4f} | {r['std']:.4f} | {r['nonzero_decision_fraction']:.1%} | {r['absolute_contribution_fraction']:.1%} | {r['correlation_final_distance']} | {r['correlation_goal_entry']} |")
        result[name]=dict(episodes=len(episodes),decisions=decisions,terms=terms)
    write(out/'new-reward-audit.json',result);(out/'new-reward-audit.md').write_text('\n'.join(lines)+'\n')


def validation_commands(out):
    records=[];arrivals=[]
    for name in ('P1','P2','P3','P4','P5','P6'):
        for path in sorted((out/name).glob('validation-*.json')):
            data=json.loads(path.read_text())
            if not data['complete']:continue
            commands=np.asarray([r['command'] for e in data['episodes'] for r in e['diagnostic_trace']])
            supported=(np.abs(commands[:,0])>=.3)|(commands[:,1]>=.3)|(commands[:,1]<=-.2)
            records.append(dict(cell=name,update=int(path.stem.split('-')[1]),samples=len(commands),
                measured_range_fraction=float(supported.mean()),minimum_command=commands.min(0).tolist(),
                maximum_command=commands.max(0).tolist(),mean_abs_command=np.abs(commands).mean(0).tolist(),
                success=data['summary']['successes']))
            episodes=[]
            for i,e in enumerate(data['episodes']):
                goal=np.asarray(e['layout']['route'][-1])*e['layout']['cell_m'];dwell=maximum=inside_s=0.
                for r in e['diagnostic_trace']:
                    inside=np.linalg.norm(np.asarray(r['xy'])-goal)<.36
                    inside_s+=.04*inside
                    dwell=dwell+.04 if inside and r['tilt']<.78 and np.linalg.norm(r['velocity'])<.2 else 0.
                    maximum=max(maximum,dwell)
                assert (maximum+1e-9>=1.)==e['success'],(name,path,i,maximum,e['success'])
                episodes.append(dict(layout_index=i,success=e['success'],failure=e['failure'],
                    goal_entered=e['goal_region_entered'],time_inside_s=float(inside_s),maximum_valid_dwell_s=maximum))
            arrivals.append(dict(cell=name,update=int(path.stem.split('-')[1]),
                entered=sum(e['goal_entered'] for e in episodes),success=sum(e['success'] for e in episodes),episodes=episodes))
    write(out/'validation-command-audit.json',dict(records=records,
        criterion='|vx| >=0.30 OR vy >=0.30 OR vy <=-0.20, from isolated command response; a heuristic, not a guaranteed mobility classifier. Samples are weighted by 40 ms physical intervals.'))
    lines=['# Deterministic validation command audit','',
           'Measured-range criterion: |vx| ≥0.30 OR vy ≥0.30 OR vy ≤−0.20 m/s. This is an isolated-command guide, not proof of motion under combined commands.',
           '', '| Cell | Update | Success /16 | vx min/max | vy min/max | Fraction in measured movement ranges |',
           '|---|---:|---:|---|---|---:|']
    for r in records:
        lo,hi=r['minimum_command'],r['maximum_command']
        lines.append(f"| {r['cell']} | {r['update']} | {r['success']} | {lo[0]:.3f}/{hi[0]:.3f} | {lo[1]:.3f}/{hi[1]:.3f} | {r['measured_range_fraction']:.1%} |")
    (out/'validation-command-audit.md').write_text('\n'.join(lines)+'\n')
    write(out/'validation-arrival-audit.json',dict(records=arrivals,scoring_reconstruction_matches=True,
        note='Reconstructed from 40 ms goal distance, tilt and velocity; independently checked against every recorded validation success.'))


def report(out):
    checkpoint=json.loads((out/'c4/result.json').read_text())
    c4={u:json.loads((out/f'c4/update-{u}/result.json').read_text())['episodes'] for u in (75,100,125)}
    directions={}
    for u,episodes in c4.items():
        groups=defaultdict(lambda:[0,0])
        for ep in episodes:
            route=ep['layout']['route'];key=str(tuple(route[-1][i]-route[-2][i] for i in range(2)))
            groups[key][0]+=int(ep['success']);groups[key][1]+=1
        directions[str(u)]=dict(groups)
    write(out/'c4/directional-outcomes.json',dict(complete=True,checkpoints=directions))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for u,es in c4.items():
        axes[0].plot(sorted(e['final_goal_distance_m'] for e in es),np.arange(1,33)/32,label=f'update {u}')
        axes[1].scatter([e['initial_goal_distance_m'] for e in es],[e['minimum_goal_distance_m'] for e in es],label=f'update {u}',alpha=.7,s=18)
    axes[0].axvline(.36,color='black',linestyle=':');axes[0].set(xlabel='Final distance (m)',ylabel='Empirical cumulative fraction',title='C4: same 32 validation layouts')
    axes[1].axhline(.36,color='black',linestyle=':');axes[1].set(xlabel='Initial distance (m)',ylabel='Minimum distance (m)',title='C4: distance achieved');axes[0].legend();axes[1].legend();fig.tight_layout();fig.savefig(out/'c4/checkpoint-distances.png',dpi=170);fig.savefig(out/'c4/checkpoint-distances.pdf');plt.close(fig)
    results={}
    for mode in ('approach','approach_recovery','rewards','fullroute','fullroute_recovery','gait_endurance','fall_replay','P1','P2','P3','P4','P5','P6'):
        path=out/mode/'result.json'
        if mode=='fall_replay' and (out/'replay-on-original-node/fall_replay/result.json').exists():path=out/'replay-on-original-node/fall_replay/result.json'
        if path.exists():
            data=json.loads(path.read_text());results[mode]={k:v for k,v in data.items() if k not in ('episodes','reward_terms')}
    write(out/'current-outcomes.json',dict(checkpoints=checkpoint,studies=results,sensor_release=False))
    reward=results.get('rewards',{})
    if reward.get('complete'):
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
        names=['standstill','random','scripted','untrained','c4_100']
        for ax,distance in zip(axes,(.65,.4)):
            means=[reward['summaries'][f'{n}-{distance}']['mean_return'] for n in names]
            ax.bar(range(5),means);ax.set_xticks(range(5),names,rotation=25);ax.axhline(0,color='black',linewidth=.7);ax.set_title(f'Start distance {distance} m')
        axes[0].set_ylabel('Mean original sparse return');fig.suptitle('Matched controller comparison, 16 validation layouts per cell');fig.tight_layout();fig.savefig(out/'rewards/controller-returns.png',dpi=170);fig.savefig(out/'rewards/controller-returns.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7));has_learning=False
    for name in ('P1','P2','P3','P4','P5','P6'):
        path=out/name/'learning-curve.json'
        if not path.exists():continue
        has_learning=True;curve=json.loads(path.read_text())
        for ax,key in zip(axes.flat,('success_rate','final_distance','fraction_closed','falls')):
            values=[r[key]['median'] if isinstance(r[key],dict) else r[key]/r['episodes'] if key=='falls' else r[key] for r in curve]
            ax.plot([r['update'] for r in curve],values,'o-',label=name);ax.set(xlabel='Update',ylabel=key);ax.grid(alpha=.2)
    if has_learning:
        axes[0,0].set(ylabel='Success rate',ylim=(0,1))
        axes[0,1].set_ylabel('Median final goal distance (m)')
        axes[1,0].set(ylabel='Median fraction of initial distance closed',ylim=(0,1))
        axes[1,1].set(ylabel='Fall rate',ylim=(0,1))
        fig.suptitle('Six privileged Approach controls: fixed 16-layout validation, seed 0')
        axes[0,0].legend();fig.tight_layout();fig.savefig(out/'approach-learning-curves.png',dpi=170);fig.savefig(out/'approach-learning-curves.pdf')
    plt.close(fig)
    optimization_report(out)
    diagnostic_rewards(out)
    validation_commands(out)
    if results.get('approach_recovery',{}).get('complete'):approach_paths(out)
    falls=[]
    for folder in ('approach','approach_recovery','fullroute','fullroute_recovery','P1','P2','P3','P4','P5','P6'):
        files=list((out/folder).glob('*.json'))
        # Only episode files, never duplicate summary result.json copies.
        if folder=='approach':files=[out/folder/'bearing.json',out/folder/'translation.json']
        if folder=='approach_recovery':files=[out/folder/'result.json']
        if folder=='fullroute':files=[p for p in files if p.name.startswith(('legacy-','measured-'))]
        if folder=='fullroute_recovery':files=[p for p in files if p.name.startswith('recovery-')]
        if folder.startswith('P'):files=[p for p in files if p.name.startswith('validation-')]+[out/folder/'episodes.jsonl']
        for path in files:
            if not path.exists():continue
            for e in episode_records(path):
                if not e['fall']:continue
                t=e['diagnostic_trace'];i=next(j for j,r in enumerate(t) if r['tilt']>=.78);recent=t[max(0,i-15):i+1];last=recent[-1]
                contacts={n for r in recent for n in r['contacts']}
                norms=np.array([np.linalg.norm(r['command'][:2]) for r in recent])
                phase=last['phase'];category='other'
                if any(n.startswith('door_') for n in contacts):category='door collision'
                elif contacts:category='wall/post collision'
                elif phase in ('acquire','release'):category='object interaction'
                elif phase=='recover':category='post-interaction recovery'
                elif norms[-1]<.05 and norms.max()>.15:category='during braking'
                elif abs(last['command'][2])>.1:category='during turning'
                elif norms.max()-norms.min()>.15:category='during acceleration'
                falls.append(dict(source=str(path.relative_to(out)),layout_seed=e['layout']['seed'],update=e.get('update'),steps=e.get('steps'),time_s=last['time_s'],category=category,
                    command=last['command'],speed_bin=round(float(norms[-1])/.1)*.1,turn_bin=round(abs(last['command'][2])/.1)*.1,
                    action_hold_s=e.get('action_hold_s',.2),phase=phase,contacts=sorted(contacts),
                    confidence='temporal association in preceding 0.6 s; not established causality'))
    endurance=out/'gait_endurance/result.json'
    if endurance.exists():
        data=json.loads(endurance.read_text());groups=defaultdict(list)
        for trial in data['episodes']:
            groups[(trial['command_name'],trial['duration_s'])].append(trial)
            if not trial['fall']:continue
            event=trial['fall_classification'];command=event['command']
            falls.append(dict(source=str(endurance.relative_to(out)),layout_seed=trial['reset_seed'],
                command_name=trial['command_name'],time_s=event['time_s'],category=event['category'],command=command,
                speed_bin=round(float(np.linalg.norm(command[:2]))/.1)*.1,turn_bin=round(abs(command[2])/.1)*.1,
                action_hold_s=trial['duration_s'],phase=event['phase'],contacts=event['contacts'],
                confidence='temporal association in preceding 0.4 s; commanded trial duration is shown as hold'))
        table=['# Long-duration open-floor response','',
               '| Command | Hold s | Trials | Falls | Recovered after fall | Final upright | Mean along displacement |',
               '|---|---:|---:|---:|---:|---:|---:|']
        summaries=[]
        for (name,duration),trials in sorted(groups.items()):
            row=dict(command_name=name,duration_s=duration,trials=len(trials),falls=sum(t['fall'] for t in trials),
                recovered=sum(t['recovered'] for t in trials),final_upright=sum(t['final_upright'] for t in trials),
                mean_along_displacement_m=float(np.mean([t['along_displacement_m'] for t in trials])))
            summaries.append(row)
            table.append(f"| {name} | {duration} | {len(trials)} | {row['falls']} | {row['recovered']} | {row['final_upright']} | {row['mean_along_displacement_m']:.3f} m |")
        write(out/'gait-endurance-summary.json',dict(complete=data['complete'],groups=summaries))
        (out/'gait-endurance.md').write_text('\n'.join(table)+'\n')
    complete=endurance.exists() and all(results.get(name,{}).get('complete',False) for name in ('fullroute','fullroute_recovery','P1','P2','P3','P4','P5','P6'))
    write(out/'fall-events.json',dict(complete=complete,events=falls))
    lines=['# Fall classification','', '| Study/episode file | Layout/reset seed | Time s | Category | Command vx/vy/yaw | Hold s | Phase |','|---|---:|---:|---|---|---:|---|']
    for e in falls:lines.append(f"| {e['source']} | {e['layout_seed']} | {e['time_s']:.2f} | {e['category']} | {e['command']} | {e['action_hold_s']} | {e['phase']} |")
    (out/'fall-events.md').write_text('\n'.join(lines)+'\n')
    replay=out/'fall_replay/episodes.json'
    if (out/'replay-on-original-node/fall_replay/episodes.json').exists():replay=out/'replay-on-original-node/fall_replay/episodes.json'
    if replay.exists():
        data=json.loads(replay.read_text())
        table=['# Counterfactual fall replay','',
               'Recorded commands end at the original fall time. Baseline trajectory matching is required before causal interpretation.',
               '', '| Variant | Layout | Fell | First fall s | Command at fall | Phase | Plate-contact frames | Exact baseline match |',
               '|---|---:|---|---:|---|---|---:|---|']
        for row in data['episodes']:
            first=next((r for r in row['trace'] if r['tilt']>=.78),None)
            table.append(f"| {row['variant']} | {row['layout_index']} | {row['fall']} | {row['first_fall_s']} | {first['command'] if first else '—'} | {first['phase'] if first else '—'} | {row['plate_contact_frames']} | {row['matched_original']} |")
        (out/'fall-replay.md').write_text('\n'.join(table)+'\n')
    if falls:
        counts=Counter((e['category'],round(e['speed_bin'],1)) for e in falls);categories=sorted({e['category'] for e in falls});speeds=sorted({round(e['speed_bin'],1) for e in falls})
        fig,ax=plt.subplots(figsize=(8,4));grid=np.array([[counts[c,s] for s in speeds] for c in categories]);im=ax.imshow(grid,aspect='auto',vmin=0)
        ax.set_yticks(range(len(categories)),categories);ax.set_xticks(range(len(speeds)),speeds);ax.set(xlabel='Commanded translation speed bin (m/s)',title='Observed falls by phase/contact association and speed');fig.colorbar(im,ax=ax,label='Fall count');fig.tight_layout();fig.savefig(out/'fall-heatmap.png',dpi=170);plt.close(fig)
    print('REPORT_UPDATED',list(results),len(falls),'fall events')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();report(a.campaign)
