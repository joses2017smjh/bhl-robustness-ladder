"""Bounded, fixed-stage Mission7 diagnosis. No production curriculum promotion."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from bhl_robust.mission.overnight import StudyEnv, OraclePolicy, TERMS, study_matrix
from bhl_robust.mission.policy import MissionPolicy
from bhl_robust.mission.layout import generate, SPLITS, wall_segments
from bhl_robust.mission.sensors import FRAME, HISTORY, PROPRIO, LIDAR, DEPTH
from mission7_diagnostic import oracle_action


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def pack(obs, env, privileged=False):
    data = {'policy': torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)}
    if privileged:
        data['oracle'] = torch.as_tensor(env.oracle_vector()).unsqueeze(0)
    return TensorDict(data, batch_size=[1])


def make_env(a, cfg, folder, split='train', seed=None):
    return StudyEnv(a.repo, folder, arm=cfg.get('arm', 'both'), stage=cfg.get('stage', 'approach'),
                    seed=cfg.get('seed', 0) if seed is None else seed, split=split,
                    approach_distance=cfg.get('approach_distance', .65))


def summary(rows):
    return {'episodes': len(rows), 'successes': sum(r['success'] for r in rows),
            'success_rate': float(np.mean([r['success'] for r in rows])),
            'mean_return': float(np.mean([r['return_total'] for r in rows])),
            'mean_episode_s': float(np.mean([r['elapsed_s'] for r in rows])),
            'falls': sum(r['fall'] for r in rows), 'collisions': sum(r['collisions'] for r in rows),
            'button_activations': sum(sum(r['open']) for r in rows),
            'gate_traversals': sum(sum(r['crossed']) for r in rows),
            'acquisitions': sum(r['acquired'] for r in rows),
            'placements': sum(r['placement_success'] for r in rows),
            'failures': dict(Counter(r['failure'] or 'success' for r in rows))}


def evaluate(a, cfg, policy, out, count):
    env = make_env(a, cfg, out.parent/'eval-cache', split='validation', seed=1000)
    rows = []
    for index in range(count):
        obs = env.reset(index)
        while True:
            with torch.inference_mode():
                action = policy.act_inference(pack(obs, env, cfg['oracle']))[0].numpy()
            obs, _, done, _ = env.step(action)
            if done:
                break
        rows.append(env.metrics())
        write(out, {'complete': False, 'episodes': rows})
    result = {'complete': True, 'split': 'validation', 'summary': summary(rows), 'episodes': rows}
    write(out, result)
    return result['summary']


def train(a, cfg, out):
    if cfg.get('requires'):
        prerequisite = json.loads((a.campaign/cfg['requires']/'result.json').read_text())
        if not prerequisite.get('learning_gate_passed'):
            write(out/'result.json', {'complete': True, 'status': 'SKIPPED_NOT_LEARNABLE',
                  'study': cfg, 'reason': 'C4 did not establish improvement on the matched easier stage'})
            print('SKIPPED_NOT_LEARNABLE', flush=True)
            return
    torch.manual_seed(cfg['seed'])
    env = make_env(a, cfg, out/'train-cache')
    obs = pack(env.reset(), env, cfg['oracle'])
    policy = OraclePolicy(obs) if cfg['oracle'] else MissionPolicy(obs)
    with torch.no_grad():
        policy.log_std.fill_(np.log(cfg['noise']))
    ppo = PPO(policy, num_learning_epochs=cfg['epochs'], num_mini_batches=cfg['minibatches'],
              learning_rate=cfg['lr'], schedule=cfg['schedule'], entropy_coef=cfg['entropy'],
              gamma=cfg['gamma'], lam=cfg['lam'])
    ppo.init_storage('rl', 1, cfg['horizon'], obs, [5])
    validation = []
    if not a.smoke:
        validation.append({'update': 0, **evaluate(a, cfg, policy, out/'validation-000.json', 8)})
    started, steps = time.monotonic(), 0
    with (out/'learning.jsonl').open('x') as log, (out/'episodes.jsonl').open('x') as episode_log:
        for update in range(cfg['updates']):
            completed, actions, reward_sum = [], [], 0.
            with torch.no_grad():
                for _ in range(cfg['horizon']):
                    action = ppo.act(obs)[0].numpy()
                    nxt, reward, done, info = env.step(action)
                    actions.append(np.tanh(action))
                    reward_sum += reward
                    steps += 1
                    if done:
                        row = env.metrics()
                        completed.append(row)
                        episode_log.write(json.dumps({'update': update+1, 'steps': steps, **row}, allow_nan=False)+'\n')
                        episode_log.flush()
                        nxt = env.reset()
                    obs = pack(nxt, env, cfg['oracle'])
                    ppo.process_env_step(obs, torch.tensor([reward]), torch.tensor([done]), {})
                ppo.compute_returns(obs)
            loss = ppo.update()
            if not np.isfinite(list(loss.values())).all() or not all(torch.isfinite(p).all() for p in policy.parameters()):
                raise FloatingPointError('nonfinite PPO')
            row = {'update': update+1, 'steps': steps, 'elapsed_s': time.monotonic()-started,
                   'reward': reward_sum, 'episode_metrics': summary(completed) if completed else None,
                   'action_mean_abs': float(np.mean(np.abs(actions))),
                   'translation_command_mean_norm': float(np.mean(np.linalg.norm(np.asarray(actions)[:, :2]*[.4,.35], axis=1))),
                   'action_std': policy.log_std.exp().detach().tolist(), 'learning_rate': ppo.learning_rate, **loss}
            log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
            if (update+1) % 25 == 0 or update+1 == cfg['updates']:
                torch.save({'model': policy.state_dict(), 'config': cfg, 'update': update+1}, out/f'model_{update+1}.pt')
                print(json.dumps(row), flush=True)
            if not a.smoke and ((update+1) % 50 == 0 or update+1 == cfg['updates']):
                validation.append({'update': update+1, **evaluate(a, cfg, policy, out/f'validation-{update+1:03}.json', 8)})
                write(out/'success-trajectory.json', validation)
    passed = bool(validation and validation[-1]['success_rate'] >= .375
                  and validation[-1]['success_rate'] >= validation[0]['success_rate']+.125)
    result = {'complete': True, 'status': 'COMPLETED_DIAGNOSTIC', 'learning_gate_passed': passed,
              'study': cfg, 'steps': steps, 'validation': validation,
              'gate_rule': '>=3/8 final successes and >=1/8 improvement over untrained; validation only',
              'production_promotion': False, 'smoke_only': a.smoke}
    write(out/'result.json', result)


def directed_action(env, target, speed=.28):
    r, s = env.runner, env.slot
    q = r.d.qpos[s.qpos_adr+3:s.qpos_adr+7]
    yaw = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
    return oracle_action(r.d.xpos[s.body_id,:2].copy(), yaw, target, speed)


def reward_table(rows, samples):
    total_abs = sum(abs(r['reward_terms'][k]) for r in rows for k in TERMS)
    net = sum(r['return_total'] for r in rows)
    result = {}
    for key in TERMS:
        v = np.asarray([r['reward_terms'][key] for r in rows])
        step_values = np.asarray([s[key] for s in samples])
        result[key] = {'episode_mean': float(v.mean()), 'episode_std': float(v.std()),
                       'episode_min': float(v.min()), 'episode_max': float(v.max()),
                       'episode_nonzero_fraction': float(np.mean(v != 0)),
                       'decision_nonzero_fraction': float(np.mean(step_values != 0)),
                       'fraction_total_absolute_contribution': float(np.abs(v).sum()/max(total_abs,1e-9)),
                       'fraction_net_return': float(v.sum()/net) if abs(net)>1e-9 else None}
    return result


def feature_stats(x, valid=None):
    x = np.asarray(x)
    mask = np.isfinite(x)
    if valid is not None:
        mask &= valid
    v = x[mask]
    # Constancy includes explicit mask/zero representation; valid-only moments
    # below avoid mistaking missing measurements for a measured zero distance.
    delta = np.diff(x, axis=0) if len(x)>1 else np.zeros_like(x)
    return {'shape': list(x.shape), 'finite_fraction': float(np.isfinite(x).mean()),
            'valid_fraction': float(mask.mean()), 'invalid_fraction': float(1-mask.mean()),
            'mean': float(v.mean()) if v.size else None, 'std': float(v.std()) if v.size else None,
            'min': float(v.min()) if v.size else None, 'max': float(v.max()) if v.size else None,
            'mean_absolute_temporal_change': float(np.mean(np.abs(delta))),
            'near_constant_feature_fraction': float(np.mean(np.std(x,axis=0)<1e-5))}


def observation_report(trajectories):
    # All arms get the exact same physically collected trajectory. Mask only the
    # unavailable modality, so the offline information comparison is paired.
    report = {}
    for arm in ('blind','lidar','stereo','both'):
        groups = []
        for trajectory in trajectories:
            x = np.asarray(trajectory).copy()
            if arm not in ('lidar','both'):
                x[:,PROPRIO:PROPRIO+2*LIDAR+1] = 0; x[:,PROPRIO+2*LIDAR] = 1
            if arm not in ('stereo','both'):
                x[:,PROPRIO+2*LIDAR+1:] = 0; x[:,-1] = 1
            groups.append(x)
        x = np.concatenate(groups)
        l = x[:,PROPRIO:PROPRIO+LIDAR]
        lm = x[:,PROPRIO+LIDAR:PROPRIO+2*LIDAR]>0
        start = PROPRIO+2*LIDAR+1
        d, dm = x[:,start:start+DEPTH], x[:,start+DEPTH:start+2*DEPTH]>0
        delta = np.concatenate([np.diff(g,axis=0) for g in groups if len(g)>1])
        report[arm] = {'actor_shape': [1,HISTORY*FRAME], 'frame': feature_stats(x),
                       'lidar_valid_values': feature_stats(l,lm), 'depth_valid_values': feature_stats(d,dm),
                       'temporal_change_within_episodes': float(np.mean(np.abs(delta))),
                       'validity_flags_binary': bool(np.isin(x[:,PROPRIO+LIDAR:PROPRIO+2*LIDAR], [0,1]).all()
                                                      and np.isin(x[:,start+DEPTH:start+2*DEPTH],[0,1]).all()),
                       'invalid_values_zero': bool(np.all(l[~lm]==0) and np.all(d[~dm]==0))}
        if arm == 'both':
            left,right = d[:,:64].ravel(),d[:,64:].ravel()
            valid = dm[:,:64].ravel() & dm[:,64:].ravel()
            report[arm]['paired_eye_correlation'] = float(np.corrcoef(left[valid],right[valid])[0,1]) if valid.sum()>2 and left[valid].std()>1e-8 and right[valid].std()>1e-8 else None
    return report


def layout_audit(out):
    rows = []
    edges = {split: [] for split in SPLITS}
    for split, (_, count) in SPLITS.items():
        for index in range(count):
            layout = generate(split,index)
            adjacency = layout.adjacency
            goal = layout.xy(layout.route[-1])
            spawn = goal+(layout.xy(layout.route[-2])-goal)/layout.cell_m*.65
            wall_clearance = min(np.linalg.norm(np.maximum(np.abs(spawn-center)-half,0)) for center,half in wall_segments(layout))
            post_clearance = min(np.linalg.norm(spawn-(goal+[.52,side*.35]))-.045 for side in (-1,1))
            row = {'split':split,'seed':layout.seed,'geometry_sha256':layout.fingerprint,
                   'route_m':(len(layout.route)-1)*layout.cell_m,'decisions':layout.decisions,'turns':layout.turns,
                   'dead_ends':sum(len(v)==1 for v in adjacency.values()),'corridor_clear_width_m':layout.cell_m-.08,
                   'switch_count':4,'correct_switch_count':2,'door_count':2,
                   'object_to_drop_centerline_m':(len(layout.route)-1-layout.object_index)*layout.cell_m,
                   'shortest_open_gate_path_m':(len(layout.route)-1)*layout.cell_m,
                   'approach_spawn_wall_clearance_m':float(wall_clearance),'approach_spawn_post_clearance_m':float(post_clearance),
                   'approach_direction':(np.asarray(layout.route[-1])-layout.route[-2]).tolist()}
            rows.append(row); edges[split].append(set(layout.edges))
    distributions = {}
    keys = [k for k,v in rows[0].items() if isinstance(v,(int,float)) and k!='seed']
    for split in SPLITS:
        selected=[r for r in rows if r['split']==split]
        distributions[split]={k:{'mean':float(np.mean([r[k] for r in selected])),
                                'std':float(np.std([r[k] for r in selected])),
                                'quantiles_0_25_50_75_100':np.quantile([r[k] for r in selected],[0,.25,.5,.75,1]).tolist()}
                              for k in keys}
    train = edges['train']
    overlap={split:{'minimum_edge_symmetric_difference_to_training': min(len(e^t) for e in group for t in train)}
             for split,group in edges.items() if split!='train'}
    result={'complete':True,'kind':'geometry_only_no_test_rollouts','layouts':rows,'distributions':distributions,
            'cross_split_topology':overlap,'all_geometry_hashes_unique':len({r['geometry_sha256'] for r in rows})==len(rows),
            'path_metric_limit':'centerline with open gates; physical switch detours add travel'}
    write(out/'layouts.json',result)
    lines=['# Layout difficulty audit','', 'Held-out entries are static geometry checks, never policy outcomes.','',
           '| Split | N | Route m mean | Turns mean | Dead ends mean | Min spawn-post clearance m |',
           '|---|---:|---:|---:|---:|---:|']
    for split in SPLITS:
        r=[x for x in rows if x['split']==split]
        lines.append(f"| {split} | {len(r)} | {np.mean([x['route_m'] for x in r]):.2f} | {np.mean([x['turns'] for x in r]):.2f} | {np.mean([x['dead_ends'] for x in r]):.2f} | {min(x['approach_spawn_post_clearance_m'] for x in r):.3f} |")
    (out/'layouts.md').write_text('\n'.join(lines)+'\n')


def audits(a, out):
    layout_audit(out)
    cfg=dict(arm='both',stage='approach',seed=0,oracle=False)
    checkpoint_path=a.repo/'results/mission7-20260920/validation-both-s0/model_400.pt'
    checkpoint=torch.load(checkpoint_path,weights_only=False,map_location='cpu')
    all_rows, all_samples, trajectories = {}, {}, []
    for control in ('random','untrained','oracle','trained'):
        torch.manual_seed(0)
        env=make_env(a,cfg,out/f'{control}-cache',seed=1000)
        policy=None; rows=[]; samples=[]; rng=np.random.default_rng(501)
        for index in range(1 if a.smoke else 16):
            obs=env.reset(index)
            if a.smoke: env.max_seconds=.4
            if policy is None:
                policy=MissionPolicy(pack(obs,env))
                if control=='trained': policy.load_state_dict(checkpoint['model'])
            trajectory=[]
            while True:
                trajectory.append(obs.reshape(HISTORY,FRAME)[-1].tolist())
                if control=='random': action=rng.normal(0,.5,5)
                elif control=='oracle': action=directed_action(env,env.layout.xy(env.layout.route[-1])) if env.runner.d.time>=1 else np.zeros(5)
                else:
                    with torch.inference_mode(): action=policy.act_inference(pack(obs,env))[0].numpy()
                obs,_,done,_=env.step(action)
                if done: break
            rows.append(env.metrics()); samples.extend(env.reward_samples)
            if control=='oracle': trajectories.append(trajectory)
            write(out/f'reward-{control}-episodes.json',{'complete':False,'episodes':rows})
        all_rows[control]=rows; all_samples[control]=samples
        write(out/f'reward-{control}-episodes.json',{'complete':True,'summary':summary(rows),'episodes':rows})
    tables={k:reward_table(all_rows[k],all_samples[k]) for k in all_rows}
    write(out/'rewards.json',{'complete':True,'terms':tables,
          'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
          'structural_flags':['Approach has no positive reward except terminal success.',
             'Switch, door and pickup terms are structurally inactive in Approach, not missing signals.',
             'No oracle distance/progress enters the reward. Sparse success is ground-truth scoring.',
             'Collision penalties can favor standing still when discovery never reaches a success.'],
          'empirical_flags':{c:{'success_nonzero_episode_fraction':t['success']['episode_nonzero_fraction'],
              'dominant_absolute_term':max(t,key=lambda k:t[k]['fraction_total_absolute_contribution'])} for c,t in tables.items()}})
    lines=['# Reward signal audit','', '| Controller | Term | Episode mean ± std | Nonzero decisions | Min / max | Absolute contribution |',
           '|---|---|---:|---:|---:|---:|']
    for c,table in tables.items():
        for term,v in table.items():
            lines.append(f"| {c} | {term} | {v['episode_mean']:.4f} ± {v['episode_std']:.4f} | {v['decision_nonzero_fraction']:.3%} | {v['episode_min']:.3f} / {v['episode_max']:.3f} | {v['fraction_total_absolute_contribution']:.1%} |")
    lines += ['', 'The only positive Approach term is terminal success. An absent success signal cannot distinguish useful movement from standing still.',
              'Absolute contribution uses the sum of absolute episode-term totals; signed net-return fractions are separately in JSON.']
    (out/'rewards.md').write_text('\n'.join(lines)+'\n')
    features=observation_report(trajectories)
    sampled=np.concatenate([np.asarray(t) for t in trajectories])[::2][:512]
    with torch.inference_mode():
        frames=torch.as_tensor(sampled,dtype=torch.float32)
        le=policy.actor.lidar(frames[:,PROPRIO:PROPRIO+2*LIDAR+1]).numpy()
        de=policy.actor.stereo(frames[:,PROPRIO+2*LIDAR+1:]).numpy()
        repeated=frames[:,None,:].expand(-1,HISTORY,-1).reshape(-1,HISTORY*FRAME)
        gates=policy.actor.encode(repeated)[1][:,-1].numpy()
    encoder_stats={'lidar':feature_stats(le),'paired_depth':feature_stats(de),
                   'mean_fusion_weights':gates.mean(axis=0).tolist(),
                   'checkpoint_update':checkpoint['update'],
                   'sampling':'static repeated-frame probes drawn from physical trajectories; no learning claim'}
    # Distinguish geometry without movement confounding: actual ray captures at
    # center of a route junction versus center of the destination in each layout.
    import mujoco
    comparisons=[]
    env=make_env(a,cfg,out/'geometry-sensors-cache')
    for index in range(1 if a.smoke else 8):
        env.reset(index); scans=[]
        junction=next(c for c in env.layout.route[1:-1] if len(env.layout.adjacency[c])>=3)
        for cell in (junction,env.layout.route[-1]):
            env.runner.d.qpos[env.slot.qpos_adr:env.slot.qpos_adr+2]=env.layout.xy(cell)
            mujoco.mj_forward(env.model,env.runner.d)
            packet=env.sensors._capture(env.runner.d,0,0.)
            scans.append(packet)
        row={'layout_index':index}
        for name,key in (('lidar','lidar_m'),('depth','paired_depth_m')):
            x,y=scans[0][key],scans[1][key]; valid=np.isfinite(x)&np.isfinite(y)
            row[name]={'joint_valid_fraction':float(valid.mean()),
                       'mean_absolute_geometry_change_m':float(np.mean(np.abs(x[valid]-y[valid]))) if valid.any() else None}
        comparisons.append(row)
    write(out/'observations.json',{'complete':True,'arms':features,'geometry_comparisons':comparisons,'separate_encoder_outputs':encoder_stats,
          'sampling':'same physical oracle trajectories replayed with modality masks; geometry probes deliberately reposition robot',
          'provenance':'actual MuJoCo rays; paired idealized depth, not RGB stereo'})
    write(out/'result.json',{'complete':True,'status':'COMPLETED_DIAGNOSTIC','artifacts':['rewards.json','rewards.md','observations.json','layouts.json','layouts.md']})


class InteractionController:
    """Oracle route controller with measured switch/attachment prerequisites."""
    def __init__(self, env):
        self.waypoint=1
        self.parcel_released=False
        self.release_started=None
        self.env=env

    def action(self):
        env=self.env; state=env.state; r=env.runner
        xy=r.d.xpos[env.slot.body_id,:2].copy()
        if r.d.time<1: return np.zeros(5)
        acquire=0.; activate=0.
        if env.stage=='transport' and not state.acquired:
            while self.waypoint<env.layout.object_index and np.linalg.norm(xy-env.layout.xy(env.layout.route[self.waypoint]))<.16:
                self.waypoint+=1
            target=env.layout.xy(env.layout.route[self.waypoint])
            if self.waypoint>=env.layout.object_index:
                target=r.d.xpos[env.parcel_body,:2].copy()
                if np.linalg.norm(target-xy)<.40:
                    acquire=2.; self.waypoint=env.layout.object_index
        elif env.stage=='transport' and state.drops and not state.carrying:
            target=r.d.xpos[env.parcel_body,:2].copy()
            if np.linalg.norm(target-xy)<.40: acquire=2.
        else:
            while self.waypoint<len(env.layout.route)-1 and np.linalg.norm(xy-env.layout.xy(env.layout.route[self.waypoint]))<.16:
                self.waypoint+=1
            target=env.layout.xy(env.layout.route[self.waypoint])
            for door,k in enumerate(env.layout.door_indices):
                if self.waypoint>=k and not state.open[door]:
                    # First reach the upstream cell, then approach its plate;
                    # cutting diagonally across a wall is not allowed.
                    center=env.layout.xy(env.layout.route[k])
                    if np.linalg.norm(xy-center)<.5:
                        target=env.layout.plate(door,env.layout.correct_sides[door]); activate=2.
                    else: target=center
                    self.waypoint=min(self.waypoint,k+1)
                    break
            if env.stage=='transport' and self.waypoint==len(env.layout.route)-1 and all(state.crossed):
                goal=env.layout.xy(env.layout.route[-1])
                mount=r.d.xmat[env.slot.body_id].reshape(3,3) @ [.32,0,.48]
                target=goal-mount[:2]
                obj=r.d.xpos[env.parcel_body]
                speed=np.linalg.norm(r.d.qvel[env.slot.qvel_adr:env.slot.qvel_adr+2])
                if self.parcel_released or np.linalg.norm(obj[:2]-goal)<.18 and speed<.15:
                    acquire=-2.; self.parcel_released=True
                    action=np.zeros(5); action[4]=acquire
                    return action
        action=directed_action(env,target,.40)
        action[3]=activate; action[4]=acquire
        return action


def component_interactions(a, out):
    """Reset-localized physical trials; never report these as full missions."""
    import mujoco
    for kind in ('door_0', 'door_1', 'transport'):
        rows=[]
        env=make_env(a,dict(stage='transport' if kind=='transport' else 'doors',arm='both',seed=2000),out/f'component-{kind}-cache')
        for index in range(1 if a.smoke else 8):
            env.reset(index)
            r,s=env.runner,env.slot
            goal=env.layout.xy(env.layout.route[-1])
            if kind=='transport':
                direction=(goal-env.layout.xy(env.layout.route[-2]))/env.layout.cell_m
                spawn=goal-direction*1.0
                # The parcel starts within the acquisition radius, on the floor.
                r.d.qpos[r.parcel_q:r.parcel_q+3]=np.r_[spawn+[.32,0], .08]
            else:
                door=int(kind[-1]); k=env.layout.door_indices[door]
                center=env.layout.xy(env.layout.route[k])
                _,direction=env.layout.door(door)
                spawn=center-direction*.15
            r.d.qpos[s.qpos_adr:s.qpos_adr+2]=spawn+env.rng.normal(0,.015,2)
            mujoco.mj_forward(env.model,r.d)
            env.sensors.next_capture[0]=0; env.sensors.capture(r.d,0.)
            env.history[:]=env.observe_frame()
            env.max_seconds=.6 if a.smoke else 40.
            released=False; dwell=0.; passed=False; contact_intervals=0
            acquisition_s=None; release_s=None; traversal_s=None
            while True:
                if kind=='transport':
                    obj=r.d.xpos[env.parcel_body].copy()
                    if not env.state.acquired:
                        action=directed_action(env,obj[:2],.40); action[4]=2.
                    else:
                        mount=r.d.xmat[s.body_id].reshape(3,3) @ [.32,0,.48]
                        action=directed_action(env,goal-mount[:2],.40)
                        if released or (np.linalg.norm(obj[:2]-goal)<.18 and np.linalg.norm(r.d.qvel[s.qvel_adr:s.qvel_adr+2])<.15):
                            released=True; action=np.zeros(5); action[4]=-2.
                else:
                    target=(env.layout.xy(env.layout.route[k+1]) if env.state.open[door]
                            else env.layout.plate(door,env.layout.correct_sides[door]))
                    action=directed_action(env,target,.40); action[3]=2.
                if r.d.time<1.: action=np.zeros(5)
                _,_,done,_=env.step(action)
                contact_intervals+=int(env.touched)
                if env.state.acquired and acquisition_s is None: acquisition_s=float(r.d.time)
                if released and release_s is None: release_s=float(r.d.time)
                if kind=='transport':
                    obj=r.d.xpos[env.parcel_body]
                    valid=(released and env.state.acquired and not env.state.carrying
                           and np.linalg.norm(obj[:2]-goal)<=.29 and .055<=obj[2]<=.095
                           and np.linalg.norm(r.d.qvel[r.parcel_v:r.parcel_v+6])<.15)
                    dwell=dwell+env.dt*env.repeat if valid else 0.
                    passed=bool(dwell>=1.-1e-8 and env.state.carried_m>=.4 and env.state.failure is None)
                else:
                    passed=bool(env.state.open[door] and env.state.crossed[door] and env.state.failure is None)
                    if passed: traversal_s=float(r.d.time)
                if passed or done: break
            row=env.metrics(); row.update(layout_index=index,component=kind,component_success=passed,
                component_dwell_s=dwell,contact_decisions=contact_intervals,acquisition_s=acquisition_s,
                release_s=release_s,traversal_s=traversal_s,reset_localized=True,
                in_episode_teleports=False,gate_preconditions_overridden=False)
            rows.append(row)
            write(out/f'component-{kind}.json',{'complete':False,'episodes':rows})
        write(out/f'component-{kind}.json',{'complete':True,'episodes':rows,
            'component_success_rate':float(np.mean([r['component_success'] for r in rows])),
            'interpretation':'Localized reset and physical rollout; no full-mission claim. Transport dwell sampled at 0.2 s decisions; production dwell is 0.04 s.'})


def interactions(a,out):
    component_interactions(a,out)
    for stage in ('doors','transport'):
        rows=[]
        for seed in range(1 if a.smoke else 2):
            env=make_env(a,dict(stage=stage,arm='both',seed=1000+seed),out/f'{stage}-{seed}-cache')
            for index in range(1 if a.smoke else 8):
                env.reset(index)
                if a.smoke: env.max_seconds=.6
                controller=InteractionController(env)
                first_acquired=None; first_release=None
                while True:
                    action=controller.action()
                    _,_,done,_=env.step(action)
                    if env.state.acquired and first_acquired is None: first_acquired=float(env.runner.d.time)
                    if controller.parcel_released and first_release is None: first_release=float(env.runner.d.time)
                    if done: break
                row=env.metrics(); row.update(reset_seed=1000+seed,layout_index=index,
                       acquisition_s=first_acquired,release_s=first_release,privileged_controller=True,
                       reset_teleports=False,controller_route_index=controller.waypoint)
                rows.append(row)
                write(out/f'{stage}.json',{'complete':False,'episodes':rows})
                print(json.dumps({k:row[k] for k in ('stage','layout_index','success','failure','open','crossed','acquired','carried_m','drops')}),flush=True)
        write(out/f'{stage}.json',{'complete':True,'summary':summary(rows),'episodes':rows,
              'capability':'oracle navigation and kinematic attachment, not learned control or grasping'})
    write(out/'result.json',{'complete':True,'status':'COMPLETED_DIAGNOSTIC','artifacts':['doors.json','transport.json','component-door_0.json','component-door_1.json','component-transport.json']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--index',type=int,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(); matrix=study_matrix()
    if not 0<=a.index<len(matrix): p.error('invalid study index')
    if not a.campaign.resolve().is_relative_to(a.repo.resolve()): p.error('campaign must remain inside repository')
    cfg=matrix[a.index].copy(); out=a.campaign/cfg['name']
    if out.exists(): p.error('study output exists; do not overwrite')
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    if a.smoke and cfg['mode']=='train':
        cfg.update(updates=2,horizon=8,minibatches=2,epochs=1); cfg.pop('requires',None)
    write(out/'config.json',cfg)
    if cfg['mode']=='train': train(a,cfg,out)
    elif cfg['mode']=='audits': audits(a,out)
    else: interactions(a,out)
    print('MISSION7_OVERNIGHT_STUDY_COMPLETE '+cfg['name'],flush=True)


if __name__=='__main__': main()
