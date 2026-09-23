"""Reproducible offline analyses of existing Mission7 logs and command trials."""
import argparse,json,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bhl_robust.mission.overnight import TERMS
from mission7_debug import describe
from mission7_overnight import write


def correlation(x,y):
    x=np.asarray(x);y=np.asarray(y)
    return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and x.std()>1e-10 and y.std()>1e-10 else None


def old_training_falls(previous,out):
    rows=[]
    for name in ['A1','A2','A3','A4','A5','B1','C2','C3','C4']:
        for index,line in enumerate((previous/name/'episodes.jsonl').read_text().splitlines()):
            episode=json.loads(line)
            if episode['failure']!='fall':continue
            last=episode['trace'][-1];command=(np.asarray(last['action'][:3])*[.4,.35,.4]).tolist()
            category='during turning' if abs(command[2])>.1 else 'braking / stationary' if np.linalg.norm(command[:2])<.04 else 'other'
            rows.append(dict(run=name,episode_index=index,layout_seed=episode['layout']['seed'],
                time_s=episode['elapsed_s'],command=command,action_hold_s=.2,phase_proxy=category,
                collision_intervals=episode['collisions'],
                confidence='Coarse ~1 s trace and terminal sample only; exact acceleration/braking/contact cause was not recorded.'))
    write(out/'previous-training-falls.json',dict(complete=True,falls=rows))
    lines=['# Historical PPO falls','',
        'All nine prior training logs: three falls, all in A2. Classification is a coarse command association, not an established cause.',
        '', '| Run | Episode | Layout seed | Fall time s | Command vx/vy/yaw | Hold s | Phase proxy |',
        '|---|---:|---:|---:|---|---:|---|']
    for r in rows:lines.append(f"| {r['run']} | {r['episode_index']} | {r['layout_seed']} | {r['time_s']:.2f} | {r['command']} | {r['action_hold_s']} | {r['phase_proxy']} |")
    (out/'previous-training-falls.md').write_text('\n'.join(lines)+'\n')


def offline(root,out):
    previous=root/'results/mission7-overnight-20260920';out.mkdir(exist_ok=True,parents=True)
    rewards={};dynamics={};total_episodes=0
    absent=['distance_progress','upright','velocity','action_penalty','other_shaping']
    for name in ['A1','A2','A3','A4','A5','B1','C2','C3','C4']:
        path=previous/name
        episodes=[json.loads(l) for l in (path/'episodes.jsonl').read_text().splitlines()]
        logs=[json.loads(l) for l in (path/'learning.jsonl').read_text().splitlines()]
        total_episodes+=len(episodes);distances=[];partial=[]
        for ep in episodes:
            goal=np.asarray(ep['layout']['route'][-1])*ep['layout']['cell_m']
            values=[np.linalg.norm(np.asarray(t['xy'])-goal) for t in ep['trace']]
            distances.append(values[-1]);partial.append(min(values)<.36)
        denom=sum(abs(v) for e in episodes for v in e['reward_terms'].values())
        net=sum(e['return_total'] for e in episodes);steps=sum(e['high_level_steps'] for e in episodes)
        table={}
        for term in list(TERMS)+absent:
            values=[e['reward_terms'].get(term,0.) for e in episodes]
            scale={'collision':.2,'failure':5.,'correct_button':2.,'wrong_button':1.,'door_crossing':2.,'acquisition':2.,'success':10.}.get(term,1.)
            counts=[round(abs(v)/scale) for v in values]
            if term=='time':lo=hi=steps
            elif term in ('failure','success','acquisition'):lo=hi=sum(counts)
            else:
                lo=sum(math.ceil(c/5) for c in counts);hi=sum(min(c,e['high_level_steps']) for c,e in zip(counts,episodes))
            table[term]={**describe(values),'present_in_reward':term not in absent,
                'episode_nonzero_fraction':float(np.mean(np.asarray(values)!=0)),
                'decision_nonzero_fraction_bounds':[lo/max(steps,1),hi/max(steps,1)],
                'frequency_exact':lo==hi,'absolute_contribution_fraction':float(np.abs(values).sum()/max(denom,1e-9)),
                'signed_net_return_fraction':float(np.sum(values)/net) if abs(net)>1e-9 else None,
                'correlation_final_distance':correlation(values,distances),
                'correlation_sampled_goal_entry':correlation(values,partial)}
        rewards[name]=dict(episodes=len(episodes),decisions=steps,terms=table,
            sampled_goal_entries=int(sum(partial)),final_distance=describe(distances),
            correlation_limit='Final distances are exact terminal samples; partial entry uses saved ~1 s traces and can miss brief visits. Constant variables have null correlation.')
        keys=('value_function','surrogate','entropy','learning_rate','action_mean_abs','translation_command_mean_norm')
        windows={}
        for lo,hi in [(1,50),(51,75),(76,100),(101,125),(126,150),(151,200)]:
            selected=[l for l in logs if lo<=l['update']<=hi]
            windows[f'{lo}-{hi}']={k:describe([l[k] for l in selected]) for k in keys}
            windows[f'{lo}-{hi}']['action_std_range']=[min(min(l['action_std']) for l in selected),max(max(l['action_std']) for l in selected)]
        dynamics[name]=dict(windows=windows,missing_original_metrics=['KL divergence','clip fraction','explained variance','gradient norm'],
            fixed_stage=True,validation=json.loads((path/'success-trajectory.json').read_text()))
        if name=='C4':
            fig,axes=plt.subplots(3,2,figsize=(11,9),sharex=True)
            for ax,key in zip(axes.flat,('value_function','surrogate','entropy','learning_rate','reward','action_mean_abs')):
                ax.plot([l['update'] for l in logs],[l[key] for l in logs],linewidth=.8)
                ax.axvline(100,color='green',linestyle='--',label='2/8 at update 100');ax.axvline(150,color='red',linestyle=':',label='0/8 at 150')
                ax.set_ylabel(key);ax.grid(alpha=.2)
                if key=='learning_rate':ax.set_yscale('log')
            axes[0,0].legend(fontsize=8);axes[-1,0].set_xlabel('PPO update');axes[-1,1].set_xlabel('PPO update')
            fig.suptitle('C4: recorded optimization dynamics; KL / clip / gradients were not logged')
            fig.tight_layout();fig.savefig(out/'c4-dynamics.png',dpi=160);fig.savefig(out/'c4-dynamics.pdf');plt.close(fig)
            fig,ax=plt.subplots(figsize=(9,3))
            for j in range(5):ax.plot([l['update'] for l in logs],[l['action_std'][j] for l in logs],label=f'action {j}')
            ax.axvline(100,color='green',linestyle='--');ax.axvline(150,color='red',linestyle=':');ax.legend();ax.set(xlabel='Update',ylabel='Gaussian action std',title='C4 exploration scale');fig.tight_layout();fig.savefig(out/'c4-action-std.png',dpi=160);plt.close(fig)
    write(out/'existing-reward-audit.json',dict(complete=True,episodes=total_episodes,runs=rewards,
        limitation='Per-decision term logs were not retained in previous PPO runs. Exact frequencies are reported where recoverable; other frequencies are bounded, never invented. New studies save them.'))
    write(out/'existing-ppo-audit.json',dict(complete=True,runs=dynamics))
    lines=['# Existing-run reward decomposition','', 'All 1,386 recorded training episodes; per-episode rewards. Null correlations mean a constant variable or missing identification, not zero association.','', '| Run | Term | Mean | Median | Std | Nonzero decisions, lower–upper | Absolute contribution | Corr(final distance) | Corr(sampled goal entry) |','|---|---|---:|---:|---:|---|---:|---|---|']
    for name,run in rewards.items():
        for term,v in run['terms'].items():
            lo,hi=v['decision_nonzero_fraction_bounds'];lines.append(f"| {name} | {term} | {v['mean']:.4f} | {v['median']:.4f} | {v['std']:.4f} | {lo:.3%}–{hi:.3%} | {v['absolute_contribution_fraction']:.1%} | {v['correlation_final_distance']} | {v['correlation_sampled_goal_entry']} |")
    lines += ['', 'Distance progress, upright, velocity, action penalty and shaping terms are absent, not small positive rewards. Standing still receives a time penalty; there is no survival bonus. Whether it has better expected return than attempted movement is measured separately in the matched controller comparison.']
    (out/'existing-reward-audit.md').write_text('\n'.join(lines)+'\n')
    # Old full-route traces support coarse phase associations, not exact contact histories.
    failures=[]
    for stage in ('doors','transport'):
        for ep in json.loads((previous/'D2'/f'{stage}.json').read_text())['episodes']:
            if not ep['failure']:continue
            t=ep['trace'][-1];cmd=(np.asarray(t['action'][:3])*[.4,.35,.4]).tolist()
            phase='object interaction / carrying' if t['carrying'] else 'navigation'
            if abs(cmd[2])>.1:phase='turning'
            if np.linalg.norm(cmd[:2])<.04:phase='braking / stationary'
            failures.append(dict(stage=stage,layout_index=ep['layout_index'],reset_seed=ep['reset_seed'],
                failure=ep['failure'],time_s=ep['elapsed_s'],command_before_termination=cmd,
                phase_proxy=phase,opened=ep['open'],crossed=ep['crossed'],acquired=ep['acquired'],
                contacts=ep['collisions'],confidence='coarse ~1 s trace; precise collision type and acceleration phase unavailable'))
    write(out/'previous-route-failures.json',dict(complete=True,failures=failures))
    old_training_falls(previous,out)
    print('OFFLINE_AUDIT_COMPLETE',total_episodes)


def gait_summary(out):
    rows=[]
    for seed in range(3):
        result=json.loads((out/f'gait-{seed}/result.json').read_text());assert result['complete'];rows+=result['episodes']
    groups=defaultdict(list)
    for row in rows:groups[(row['command_name'],row['duration_s'])].append(row)
    summaries=[]
    for (name,duration),g in sorted(groups.items()):
        summary=dict(command_name=name,command=g[0]['command'],duration_s=duration,trials=len(g),
            fall_probability=float(np.mean([r['fall'] for r in g])),final_upright_fraction=float(np.mean([r['final_upright'] for r in g])))
        for key in ('along_displacement_m','actual_speed_m_s','lateral_drift_m','heading_change_rad','heading_error_rad','braking_distance_m','overshoot_m'):
            summary[key]=describe([r[key] for r in g])
        summary['onset_delay_s']=describe([r['onset_delay_s'] for r in g if r['onset_delay_s'] is not None]) if any(r['onset_delay_s'] is not None for r in g) else None
        summary['braking_stop_s']=describe([r['braking_stop_s'] for r in g if r['braking_stop_s'] is not None]) if any(r['braking_stop_s'] is not None for r in g) else None
        latencies=[]
        for trial in g:
            active=[r for r in trial['trace'] if any(r['command'])]
            speed=np.linalg.norm(trial['command'][:2]); onset=None
            for j in range(len(active)-2):
                part=active[j:j+3]
                if speed:
                    direction=np.asarray(trial['command'][:2])/speed
                    c,s=np.cos(part[0]['yaw']),np.sin(part[0]['yaw']);world=np.array([[c,-s],[s,c]])@direction
                    valid=all(np.dot(r['velocity'],world)>max(.05,.5*speed) for r in part)
                else:valid=all(abs(r['yaw_rate'])>.5*abs(trial['command'][2]) for r in part)
                if valid:onset=active[j]['time_s']-1.2;break
            if onset is not None:latencies.append(onset)
        summary['sustained_response_latency_s']=describe(latencies) if latencies else None
        summary['sustained_response_trials']=len(latencies)
        summaries.append(summary)
    # Recommendations must be reviewed against the complete response table.
    forward=[r for r in summaries if r['command_name'].startswith('forward') and r['duration_s']==2. and r['fall_probability']==0]
    viable=[r for r in forward if r['along_displacement_m']['mean']>.15]
    braking_safe=[r for r in viable if r['braking_distance_m']['max']<.10]
    chosen=max(braking_safe or viable,key=lambda r:r['along_displacement_m']['mean']) if viable else None
    yaw=[r for r in summaries if r['command_name'].startswith('yaw') and r['duration_s']==2. and r['fall_probability']==0]
    controller=dict(speed=chosen['command'][0] if chosen else .4,turn_rate=.35,stop_radius=.25,heading_tolerance=.25)
    write(out/'gait-summary.json',dict(complete=True,episodes=len(rows),groups=summaries,controller=controller,
        locomotion_candidate_found=bool(chosen),recommendation_requires_approach_validation=True,
        selection_rule='Fastest no-fall forward response at 2 s with >0.15 m displacement, preferring all-trial brake displacement <0.10 m.',
        latency_rule='Three consecutive 40 ms samples above max(0.05 m/s, 50% commanded speed); yaw uses 50% commanded turn rate. Original onset_delay_s is only the first threshold crossing.'))
    lines=['# Gait response — open floor','', '| Command (vx, vy, yaw) | Hold s | N | Along displacement m ± std | Measured speed m/s | Brake displacement m | Heading error rad | Falls |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['command']} | {r['duration_s']} | {r['trials']} | {r['along_displacement_m']['mean']:.3f} ± {r['along_displacement_m']['std']:.3f} | {r['actual_speed_m_s']['mean']:.3f} | {r['braking_distance_m']['mean']:.3f} | {r['heading_error_rad']['mean']:.3f} | {r['fall_probability']:.1%} |")
    (out/'gait-response.md').write_text('\n'.join(lines)+'\n')
    names=sorted({r['command_name'] for r in summaries});durations=sorted({r['duration_s'] for r in summaries})
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for name in names:
        selected=sorted([r for r in summaries if r['command_name']==name],key=lambda r:r['duration_s'])
        if name.startswith('yaw'):continue
        axes[0,0].errorbar([r['duration_s'] for r in selected],[r['along_displacement_m']['mean'] for r in selected],yerr=[r['along_displacement_m']['std'] for r in selected],label=name,marker='.',linewidth=.8)
    axes[0,0].set(xlabel='Hold (s)',ylabel='Along displacement (m)',title='Duration → displacement');axes[0,0].legend(fontsize=6,ncol=2)
    for duration in durations:
        selected=[r for r in summaries if r['command_name'].startswith('forward') and r['duration_s']==duration]
        axes[0,1].plot([r['command'][0] for r in selected],[r['along_displacement_m']['mean'] for r in selected],marker='.',label=f'{duration}s')
    axes[0,1].set(xlabel='Forward command (m/s)',ylabel='Displacement (m)',title='Command → displacement');axes[0,1].legend(fontsize=7)
    for ax,key,title in [(axes[1,0],'fall_probability','Fall probability'),(axes[1,1],'heading_error_rad','Mean heading error (rad)')]:
        grid=np.array([[next(r for r in summaries if r['command_name']==name and r['duration_s']==d)[key] if key=='fall_probability' else next(r for r in summaries if r['command_name']==name and r['duration_s']==d)[key]['mean'] for d in durations] for name in names])
        im=ax.imshow(grid,aspect='auto',vmin=0,vmax=1) if key=='fall_probability' else ax.imshow(grid,aspect='auto',cmap='coolwarm',vmin=-max(abs(grid.min()),abs(grid.max())),vmax=max(abs(grid.min()),abs(grid.max())))
        ax.set_xticks(range(len(durations)),durations);ax.set_yticks(range(len(names)),names,fontsize=7);ax.set(xlabel='Hold (s)',title=title);fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(out/'gait-response.png',dpi=170);fig.savefig(out/'gait-response.pdf');plt.close(fig)
    write(out/'gait-falls.json',dict(complete=True,falls=[{k:v for k,v in r.items() if k!='trace'} for r in rows if r['fall']]))
    print('GAIT_SUMMARY_COMPLETE',len(rows),controller)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('offline','gait'));p.add_argument('--repo',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);a=p.parse_args()
    offline(a.repo,a.campaign/'offline-audit') if a.mode=='offline' else gait_summary(a.campaign)
