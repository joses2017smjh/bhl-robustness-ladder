"""Bounded Mission7 feasibility and checkpoint diagnostics; never promote training.

Oracle control uses exact simulator pose/goal and is explicitly privileged.
Checkpoint ablations use validation layouts only, including failed checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from bhl_robust.mission.env import MissionEnv
from bhl_robust.mission.policy import MissionPolicy
from bhl_robust.mission.sensors import HISTORY, FRAME, PROPRIO, LIDAR, DEPTH


def oracle_action(xy, yaw, goal, speed):
    velocity = (np.asarray(goal)-xy)*1.2
    norm = np.linalg.norm(velocity)
    velocity *= min(1., speed/max(norm, 1e-9))
    if np.linalg.norm(np.asarray(goal)-xy) < .10:
        velocity[:] = 0
    c, s = np.cos(yaw), np.sin(yaw)
    command = np.array([c*velocity[0]+s*velocity[1], -s*velocity[0]+c*velocity[1], np.clip(-1.2*yaw, -.3, .3)])
    return np.r_[np.arctanh(np.clip(command/[.4, .35, .4], -.99, .99)), 0., 0.]


def mask_modality(observation, condition):
    x = np.asarray(observation).copy().reshape(HISTORY, FRAME)
    for name, start, width in (("lidar", PROPRIO, 2*LIDAR+1),
                                ("stereo", PROPRIO+2*LIDAR+1, 2*DEPTH+1)):
        if condition in (name+"_missing", "both_missing"):
            x[:, start:start+width] = 0
            x[:, start+width-1] = 1
    return x.ravel()


def packed(x):
    return TensorDict({"policy": torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)}, batch_size=[1])


def write(path, data):
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('feasibility', 'checkpoint'))
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--layouts', type=int, default=16)
    p.add_argument('--seeds', type=int, default=2)
    p.add_argument('--seconds', type=float, default=18.)
    a = p.parse_args()
    if not 1 <= a.layouts <= 32 or not 1 <= a.seeds <= 3 or not .2 <= a.seconds <= 18:
        p.error('bounded layouts (1..32), seeds (1..3), seconds (.2..18) required')
    if not a.out.resolve().is_relative_to(a.repo.resolve()) or a.out.exists():
        p.error('choose a fresh output directory inside the repository')
    torch.set_num_threads(1)
    a.out.mkdir(parents=True)
    cases = ('oracle_slow', 'oracle_fast', 'stationary') if a.mode == 'feasibility' else ('normal', 'lidar_missing', 'stereo_missing', 'both_missing')
    rows, counterfactual = [], []
    report = {'schema': 'bhl-mission7-diagnostic-v1', 'mode': a.mode, 'complete': False,
              'stage': 'approach', 'split': 'train' if a.mode == 'feasibility' else 'validation',
              'privileged_controller': a.mode == 'feasibility', 'training_promotion': False,
              'episodes': rows, 'counterfactual_sensor_action_effects': counterfactual}
    checkpoint = None
    if a.mode == 'checkpoint':
        if not a.checkpoint or not a.checkpoint.is_file():
            p.error('specified checkpoint must exist; no automatic fallback')
        checkpoint = torch.load(a.checkpoint, weights_only=False, map_location='cpu')
        if (checkpoint['stage'], checkpoint['arm'], checkpoint['seed']) != ('approach', 'both', 0):
            p.error('this diagnostic expects the Both/seed-0/Approach checkpoint')
        report.update(checkpoint=str(a.checkpoint.resolve()), checkpoint_update=checkpoint['update'],
                      checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest())
    for case in cases:
        for seed in range(a.seeds):
            env = MissionEnv(a.repo, a.out/f'{case}-{seed}-cache', arm='both', stage='approach',
                             split=report['split'], seed=1000+seed, max_seconds=a.seconds,
                             failure=case if a.mode == 'checkpoint' else 'normal')
            policy = None
            for index in range(a.layouts):
                obs = env.reset(index)
                if checkpoint is not None and policy is None:
                    policy = MissionPolicy(packed(obs))
                    policy.load_state_dict(checkpoint['model'])
                    policy.eval()
                min_distance = float('inf')
                inside_steps, slow_inside_steps, max_hold, sample = 0, 0, 0., 0
                collision_names = set()
                while True:
                    r, slot = env.runner, env.slot
                    xy = r.d.xpos[slot.body_id, :2].copy()
                    goal = env.layout.xy(env.layout.route[-1])
                    distance = float(np.linalg.norm(goal-xy))
                    min_distance = min(min_distance, distance)
                    inside_steps += int(distance < .36)
                    slow_inside_steps += int(distance < .36 and np.linalg.norm(r.d.qvel[slot.qvel_adr:slot.qvel_adr+2]) < .2)
                    if policy is None:
                        q = r.d.qpos[slot.qpos_adr+3:slot.qpos_adr+7]
                        yaw = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
                        action = np.zeros(5) if case == 'stationary' or r.d.time < 1. else oracle_action(xy, yaw, goal, .12 if case == 'oracle_slow' else .28)
                    else:
                        with torch.inference_mode():
                            action = policy.act_inference(packed(obs))[0].numpy()
                            if case == 'normal' and sample % 5 == 0:
                                effects = {}
                                for missing in cases[1:]:
                                    changed = policy.act_inference(packed(mask_modality(obs, missing)))[0].numpy()
                                    effects[missing] = float(np.linalg.norm(np.tanh(changed)-np.tanh(action)))
                                counterfactual.append({'seed': seed, 'layout_index': index, 'time_s': float(r.d.time), **effects})
                    obs, _, done, _ = env.step(action)
                    max_hold = max(max_hold, env.state.hold_s)
                    # Endpoint contact names aid diagnosis; authoritative collision
                    # counts still include every physics substep in MissionRunner.
                    import mujoco
                    for con in env.runner.d.contact:
                        for geom in (int(con.geom1), int(con.geom2)):
                            if geom in env.runner.walls:
                                collision_names.add(mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom))
                    sample += 1
                    if done:
                        break
                row = env.metrics()
                row.update(controller=case, reset_seed=1000+seed, layout_index=index,
                           min_goal_distance_m=min_distance, samples_inside_goal=inside_steps,
                           samples_slow_inside_goal=slow_inside_steps, max_hold_s=max_hold,
                           endpoint_contact_names=sorted(collision_names))
                rows.append(row)
                write(a.out/'report.json', report)
                print(json.dumps({k: row[k] for k in ('controller','reset_seed','layout_index','success','failure','min_goal_distance_m','max_hold_s')}), flush=True)
    summary = {}
    for case in cases:
        selected = [r for r in rows if r['controller'] == case]
        summary[case] = {'episodes': len(selected), 'successes': sum(r['success'] for r in selected),
                         'success_rate': float(np.mean([r['success'] for r in selected])),
                         'collision_failures': sum(r['failure'] == 'excessive_collision' for r in selected),
                         'mean_min_distance_m': float(np.mean([r['min_goal_distance_m'] for r in selected]))}
    report.update(complete=True, summary=summary,
                  interpretation='Diagnostic only. Oracle success is not learned success; action sensitivity is not sensor benefit.')
    write(a.out/'report.json', report)
    print('MISSION7_DIAGNOSTIC_COMPLETE '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
