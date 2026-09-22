"""Small privileged Approach PPO controls with recorded optimization telemetry."""
import json,time
import numpy as np
import torch
from torch.distributions import Normal
from rsl_rl.algorithms import PPO
from bhl_robust.mission.approach_debug import DebugEnv
from bhl_robust.mission.overnight import OraclePolicy
from mission7_overnight import pack,write


class UpdateTelemetry:
    def __init__(self,ppo):
        self.ppo=ppo;self.raw={};self.gradients=[]
        self.handles=[]
        for name,param in ppo.policy.named_parameters():
            def hook(grad,name=name):self.raw[name]=float(grad.detach().square().sum())
            self.handles.append(param.register_hook(hook))
        def before_step(optimizer,args,kwargs):
            clipped=sum(float(p.grad.square().sum()) for p in ppo.policy.parameters() if p.grad is not None)
            self.gradients.append((sum(self.raw.values())**.5,clipped**.5));self.raw.clear()
        self.handles.append(ppo.optimizer.register_step_pre_hook(before_step))

    def update(self):
        ppo=self.ppo;s=ppo.storage;policy=ppo.policy
        obs=s.observations.flatten(0,1).clone()
        actions=s.actions.flatten(0,1).clone();old_mu=s.mu.flatten(0,1).clone();old_sigma=s.sigma.flatten(0,1).clone()
        old_logp=s.actions_log_prob.flatten().clone();returns=s.returns.flatten().clone()
        old_values=s.values.flatten().clone();before=torch.cat([p.detach().flatten().clone() for p in policy.actor.parameters()])
        self.gradients=[];loss=ppo.update()
        with torch.no_grad():
            mu=policy.act_inference(obs);sigma=policy.log_std.exp().expand_as(mu)
            logp=Normal(mu,sigma).log_prob(actions).sum(-1)
            kl=torch.log(sigma/old_sigma)+(old_sigma.square()+(old_mu-mu).square())/(2*sigma.square())-.5
            ratio=torch.exp(logp-old_logp);values=policy.evaluate(obs).flatten()
            variance=returns.var(unbiased=False)
            ev=lambda prediction:float(1-(returns-prediction).var(unbiased=False)/variance) if variance>1e-10 else None
            after=torch.cat([p.detach().flatten() for p in policy.actor.parameters()])
            diagnostics=dict(analytic_kl_post_update=float(kl.sum(-1).mean()),
                clip_fraction_post_update=float(((ratio<1-ppo.clip_param)|(ratio>1+ppo.clip_param)).float().mean()),
                explained_variance_before=ev(old_values),explained_variance_after=ev(values),
                actor_update_relative_l2=float((after-before).norm()/before.norm().clamp_min(1e-9)),
                gradient_norm_before_clip_mean=float(np.mean([g[0] for g in self.gradients])),
                gradient_norm_before_clip_max=max(g[0] for g in self.gradients),
                gradient_norm_after_clip_mean=float(np.mean([g[1] for g in self.gradients])))
        return {**loss,**diagnostics}


def evaluate(a,cfg,policy,out,count):
    from mission7_debug import summarize
    env=DebugEnv(a.repo,out.parent/'eval-cache',stage='approach',split='validation',seed=1000,
        hold_ticks=cfg['hold_ticks'],dense=cfg['dense'])
    rows=[]
    for index in range(count):
        obs=env.reset(index)
        while True:
            with torch.inference_mode():action=policy.act_inference(pack(obs,env,True))[0].numpy()
            obs,_,done,_=env.step(action)
            if done:break
        rows.append(env.metrics());write(out,dict(complete=False,episodes=rows))
    result=dict(complete=True,summary=summarize(rows),episodes=rows)
    write(out,result);return result['summary']


def train(a,out):
    matrix_name='training-matrix-exploration.json' if a.mode=='train_exploration' else 'training-matrix.json'
    matrix=json.loads((a.campaign/matrix_name).read_text())
    cfg=matrix['rows'][a.index].copy()
    if not a.smoke:
        prerequisite=json.loads((a.campaign/'feasibility-gate.json').read_text())
        rewards=json.loads((a.campaign/'rewards/result.json').read_text())
        assert rewards['complete'] and prerequisite['complete']
        if not prerequisite['reliable']:
            write(out/'result.json',dict(complete=True,status='BLOCKED_FEASIBILITY',reason='Scripted Approach did not meet reliability gate; no PPO performed.'))
            return
    else:cfg.update(updates=2,horizon=8,epochs=1,minibatches=2)
    write(out/'config.json',cfg)
    torch.manual_seed(0)
    env=DebugEnv(a.repo,out/'train-cache',stage='approach',split='train',seed=0,hold_ticks=cfg['hold_ticks'],dense=cfg['dense'])
    obs=pack(env.reset(),env,True);policy=OraclePolicy(obs)
    with torch.no_grad():policy.log_std.fill_(np.log(cfg['noise']))
    ppo=PPO(policy,num_learning_epochs=cfg['epochs'],num_mini_batches=cfg['minibatches'],
        learning_rate=cfg['lr'],schedule=cfg['schedule'],entropy_coef=cfg['entropy'],gamma=cfg['gamma'],lam=cfg['lam'])
    ppo.init_storage('rl',1,cfg['horizon'],obs,[5]);telemetry=UpdateTelemetry(ppo)
    validation=[];steps=0;simulated_s=0.;start=time.monotonic();training_successes=0
    if not a.smoke:validation.append(dict(update=0,**evaluate(a,cfg,policy,out/'validation-000.json',16)))
    with (out/'learning.jsonl').open('x') as learning,(out/'episodes.jsonl').open('x') as episodes:
        for update in range(1,cfg['updates']+1):
            actions=[];rewards=[];finished=[]
            with torch.no_grad():
                for _ in range(cfg['horizon']):
                    action=ppo.act(obs)[0].numpy();before=float(env.runner.d.time)
                    nxt,reward,done,info=env.step(action);simulated_s+=float(env.runner.d.time)-before
                    actions.append(np.tanh(action));rewards.append(reward);steps+=1
                    if done:
                        row=env.metrics();training_successes+=int(row['success']);finished.append(row)
                        episodes.write(json.dumps(dict(update=update,steps=steps,**row),allow_nan=False)+'\n');episodes.flush()
                        nxt=env.reset()
                    obs=pack(nxt,env,True)
                    ppo.process_env_step(obs,torch.tensor([reward]),torch.tensor([done]),{})
                ppo.compute_returns(obs)
            loss=telemetry.update()
            if not all(np.isfinite(v) for v in loss.values() if v is not None):raise FloatingPointError('nonfinite PPO diagnostic')
            if not all(torch.isfinite(p).all() for p in policy.parameters()):raise FloatingPointError('nonfinite model')
            row=dict(update=update,steps=steps,simulated_training_s=simulated_s,elapsed_s=time.monotonic()-start,
                rollout_return=float(sum(rewards)),training_successes=training_successes,
                completed_episodes=len(finished),mean_final_distance=float(np.mean([e['final_goal_distance_m'] for e in finished])) if finished else None,
                learning_rate=ppo.learning_rate,action_std=policy.log_std.exp().detach().tolist(),
                action_mean_abs=float(np.mean(np.abs(actions))),**loss)
            learning.write(json.dumps(row,allow_nan=False)+'\n');learning.flush()
            if update%25==0 or update==cfg['updates']:
                torch.save(dict(model=policy.state_dict(),config=cfg,update=update),out/f'model_{update}.pt')
                print(json.dumps(row),flush=True)
            if not a.smoke and update in (25,50,100,150,200):
                validation.append(dict(update=update,**evaluate(a,cfg,policy,out/f'validation-{update:03}.json',16)))
                write(out/'learning-curve.json',validation)
    stable=bool(len(validation)>2 and all(v['successes']>=5 for v in validation[-2:])
        and validation[-1]['successes']>=validation[0]['successes']+2 and training_successes>=3)
    write(out/'result.json',dict(complete=True,status='COMPLETED_DIAGNOSTIC',config=cfg,validation=validation,
        training_successes=training_successes,steps=steps,simulated_training_s=simulated_s,
        stable_within_seed=stable,sensor_comparison_release=False,
        gate='Last two evaluations >=5/16, final improvement >=2/16 and >=3 training successes. Still one seed, privileged actor and diagnostic shaping.'))
