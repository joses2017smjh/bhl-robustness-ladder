"""Progress-aware, supervised closed-loop dynamics; empirical simulation only.

Inputs: simulated IMU attitude/gyro, joints, previous gait actions and history.
Velocity, local displacement, local sink and next-step termination are training
TARGETS only. Terrain identity, root pose, true velocity and sink are never
runtime model inputs. Future behavior-policy severity is not used as a
counterfactual risk label. Every candidate supplies its own rollout commands.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = 'bhl-progress-dynamics-v3'
LEGACY_SCHEMA = 'bhl-progress-dynamics-v2'
MODEL_METHODS = ('shadow', 'ensemble_speed_fixed', 'dynamics_only', 'ensemble', 'ensemble_curvature')
METHODS = ('baseline', 'yaw_only', 'fixed', 'adaptive', 'adaptive_curvature') + MODEL_METHODS
AUX_DIM = 6  # endpoint yaw-frame velocity (vx,vy,wz), step-local displacement (dx,dy), sink


def proprio(robot_observation, previous_action):
    obs = np.asarray(robot_observation, dtype=np.float32)
    action = np.asarray(previous_action, dtype=np.float32).reshape(-1)
    n = len(action)
    if obs.shape != (11 + 2*n,) or not np.isfinite(obs).all() or not np.isfinite(action).all():
        raise ValueError('finite upstream observation and previous gait action required')
    q = obs[:4].astype(np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-8)
    w, v, g = q[0], q[1:], np.array([0., 0., -1.])
    gravity = g*(2*w*w-1)-2*w*np.cross(v, g)+2*v*np.dot(v, g)
    return np.concatenate([gravity, obs[4:7+2*n], action]).astype(np.float32)


class StateHistory:
    def __init__(self):
        self.previous = None
        self.commands = np.zeros(6, dtype=np.float32)

    def state(self, core):
        return np.concatenate([core, core if self.previous is None else self.previous, self.commands]).astype(np.float32)

    def advance(self, core, command):
        self.previous = np.asarray(core, dtype=np.float32).copy()
        self.commands = np.concatenate([np.asarray(command, dtype=np.float32), self.commands[:3]])


def whole_seed_split(seeds, split_seed=47111, strata=None, balance_tags=None):
    groups = np.unique(np.asarray(seeds, dtype=np.int64))
    if len(groups) < 5:
        raise ValueError('need >=5 complete reset groups for train/calibration/test')
    rng = np.random.default_rng(split_seed)
    result = {'test': [], 'calibration': [], 'train': []}
    buckets = [groups] if strata is None else [np.array([s for s in groups if strata[int(s)] == k])
                                               for k in sorted(set(strata.values()), key=str)]
    for bucket in buckets:
        selected = rng.permutation(bucket)
        if len(selected) < 3:
            raise ValueError('each policy/command-regime stratum needs >=3 reset groups; collect >=12 episodes, preferably16')
        if balance_tags is not None and len(selected) == 4:
            values = [balance_tags[int(s)] for s in selected]
            if all(isinstance(v,tuple) and len(v)==2 for v in values) and len(set(values))==4:
                first = selected[0]
                opposite = next(s for s in selected if all(a!=b for a,b in
                                     zip(balance_tags[int(first)],balance_tags[int(s)])))
                result['train'].extend((first,opposite))
                heldout = rng.permutation([s for s in selected if s not in (first,opposite)])
                result['test'].append(heldout[0]);result['calibration'].append(heldout[1])
                continue
            tagged = [[s for s in selected if balance_tags[int(s)] == t]
                      for t in sorted({balance_tags[int(s)] for s in selected}, key=str)]
            if len(tagged) == 2 and all(len(v) == 2 for v in tagged):
                result['train'].extend(v[0] for v in tagged)
                heldout = rng.permutation([v[1] for v in tagged])
                result['test'].append(heldout[0]); result['calibration'].append(heldout[1])
                continue
        n = max(1, int(round(len(selected)*.2)))
        result['test'].extend(selected[:n]); result['calibration'].extend(selected[n:2*n])
        result['train'].extend(selected[2*n:])
    return {k: np.array(v, dtype=np.int64) for k, v in result.items()}


def model_class():
    import torch
    from torch import nn
    class Network(nn.Module):
        def __init__(self, input_dim, output_dim, hidden=128):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, output_dim+1))
        def forward(self, x):
            out = self.net(x)
            return out[..., :-1], out[..., -1]  # standardized regression, immediate-event logit
    return Network


def tilt_of(gravity):
    g = np.asarray(gravity)
    return np.arccos(np.clip(-g[..., 2]/np.maximum(np.linalg.norm(g, axis=-1), 1e-8), -1., 1.))


def calibration_stratum(state, command):
    """Only current IMU/command observables; no terrain or true velocity label."""
    tilt = float(tilt_of(np.asarray(state)[:3]))
    gyro = float(np.linalg.norm(np.asarray(state)[3:6]))
    yaw = abs(float(np.asarray(command)[2]))
    regime = 'calm' if tilt < .12 and gyro < .8 and yaw < .4 else (
        'turning' if tilt < .20 and gyro < 1.8 else 'recovery')
    motion = 'walking' if np.linalg.norm(np.asarray(command)[:2]) > .05 else 'holding'
    return regime+'_'+motion


class DynamicsEnsemble:
    def __init__(self, checkpoint=None, *, payload=None, consumed_development=False):
        import torch
        torch.set_num_threads(1)
        self.payload = payload if payload is not None else torch.load(checkpoint, map_location='cpu', weights_only=False)
        p = self.payload
        legacy = p.get('schema') == LEGACY_SCHEMA
        if p.get('schema') != SCHEMA and not (legacy and consumed_development):
            raise ValueError('v3 checkpoint required; consumed v2 evidence requires explicit development flag')
        if p.get('consumed_development') and not consumed_development:
            raise ValueError('consumed-data model is development evidence, not a fresh campaign checkpoint')
        self.consumed_development = bool(consumed_development or legacy or p.get('consumed_development'))
        self.core_dim, self.dt = p['core_dim'], p['dt']
        self.gait_name = p['data'][0]['gait_name']
        self.policy_sha256s = {m['checkpoint_sha256'] for m in p['data']}
        self.x_mean, self.x_std, self.y_mean, self.y_std = [p[k] for k in ('x_mean','x_std','y_mean','y_std')]
        self.members = []
        for weights in p['members']:
            net = model_class()(len(self.x_mean), len(self.y_mean), p['hidden'])
            net.load_state_dict(weights); net.eval(); self.members.append(net)
        self.checkpoint = str(Path(checkpoint).resolve()) if checkpoint else None
        self.checkpoint_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None

    def predict_members(self, states, commands):
        import torch
        x = np.concatenate([states, commands], axis=-1).astype(np.float32)
        x = torch.from_numpy((x-self.x_mean)/self.x_std)
        reg, prob = [], []
        event_cal = self.payload.get('event_calibration', {})
        with torch.inference_mode():
            for net in self.members:
                y, logit = net(x)
                reg.append(y.numpy()*self.y_std+self.y_mean)
                prob.append(torch.sigmoid(logit*event_cal.get('scale',1.)+event_cal.get('bias',0.)).numpy())
        return np.stack(reg), np.stack(prob)

    def rollout(self, state, command_sequences, return_trajectory=False):
        import torch
        seq = np.asarray(command_sequences, dtype=np.float32)
        initial = np.asarray(state, dtype=np.float32)
        if initial.ndim == 1:
            initial = np.broadcast_to(initial[None], (len(seq), len(initial)))
        if initial.ndim != 2 or len(initial) != len(seq) or seq.ndim != 3:
            raise ValueError('matching state/candidate batches and command sequences required')
        states = np.broadcast_to(initial[None], (len(self.members), *initial.shape)).copy()
        shape = (len(self.members), len(seq))
        displacement = np.zeros(shape+(2,), np.float32)
        heading = np.zeros(shape, np.float32)
        risk_head = np.zeros(shape, np.float32)
        max_tilt = np.zeros(shape, np.float32)
        max_sink = np.zeros(shape, np.float32)
        uncertainty = np.zeros(len(seq)); input_rms_peak = np.zeros(len(seq))
        velocities, trajectory, aux_trajectory = [], [], []
        event_cal = self.payload.get('event_calibration', {})
        for t in range(seq.shape[1]):
            outputs, events = [], []
            with torch.inference_mode():
                for mi, net in enumerate(self.members):
                    x = np.concatenate([states[mi], seq[:, t]], axis=1)
                    z = (x-self.x_mean)/self.x_std
                    input_rms_peak = np.maximum(input_rms_peak, np.sqrt(np.mean(z*z, axis=1)))
                    y, event = net(torch.from_numpy(z))
                    outputs.append(y.numpy()*self.y_std+self.y_mean)
                    events.append(torch.sigmoid(event*event_cal.get('scale',1.)+event_cal.get('bias',0.)).numpy())
            outputs, events = np.stack(outputs), np.stack(events)
            core = states[:, :, :self.core_dim]
            predicted = core+outputs[:, :, :self.core_dim]
            aux = outputs[:, :, self.core_dim:]
            gravity = predicted[:, :, :3]
            gravity /= np.maximum(np.linalg.norm(gravity, axis=2, keepdims=True), 1e-6)
            c, s = np.cos(heading), np.sin(heading)
            displacement[..., 0] += c*aux[..., 3]-s*aux[..., 4]
            displacement[..., 1] += s*aux[..., 3]+c*aux[..., 4]
            heading += aux[..., 2]*self.dt
            velocities.append(aux[..., :3])
            max_tilt = np.maximum(max_tilt, tilt_of(gravity))
            max_sink = np.maximum(max_sink, aux[..., 5])
            risk_head = np.maximum(risk_head, events)
            uncertainty = np.maximum(uncertainty, np.sqrt(np.mean((predicted/self.y_std[:self.core_dim]).std(0)**2, axis=1)))
            commands = np.broadcast_to(seq[None, :, t], shape+(3,))
            states = np.concatenate([predicted, core, commands, states[:, :, 2*self.core_dim:2*self.core_dim+3]], axis=2)
            if return_trajectory:
                trajectory.append(predicted.copy()); aux_trajectory.append(aux.copy())
        result = {'velocity': np.stack(velocities).mean(axis=(0,1)),
                  'displacement': displacement.mean(0), 'displacement_std': displacement.std(0),
                  'tilt': max_tilt.max(0), 'sink': max_sink.max(0),
                  'immediate_event_probability': np.clip(risk_head.mean(0)+2*risk_head.std(0), 0., 1.),
                  'uncertainty': uncertainty, 'input_rms_peak': input_rms_peak,
                  'next_core_mean': predicted.mean(0)}
        if return_trajectory:
            result['core_trajectory'] = np.stack(trajectory).mean(1)
            result['aux_trajectory'] = np.stack(aux_trajectory).mean(1)
        return result


class Governor:
    def __init__(self, kind='baseline', checkpoint=None, horizon_steps=8, qualification=None,
                 gait_name=None, training_seed=None, consumed_development=False):
        if kind not in METHODS:
            raise ValueError(f'unknown governor {kind}; expected {METHODS}')
        if kind in MODEL_METHODS and checkpoint is None:
            raise ValueError('model methods require a fitted checkpoint')
        self.kind, self.horizon_steps = kind, int(horizon_steps)
        if self.horizon_steps < 1:
            raise ValueError('positive horizon required')
        self.model = DynamicsEnsemble(checkpoint,consumed_development=consumed_development) if kind in MODEL_METHODS else None
        self.gait_name = gait_name or (self.model.gait_name if self.model else None)
        self.training_seed = training_seed
        self.qualification_source = None
        if isinstance(qualification, (str, Path)):
            self.qualification_source = str(Path(qualification).resolve())
            qualification = json.loads(Path(qualification).read_text())
        self.qualification = qualification or {}
        key = f'{self.gait_name}-s{training_seed}'
        policies = self.qualification.get('policies', {})
        self.motion = policies.get(key, {})
        if self.motion.get('qualified') and self.model and self.motion.get('policy_sha256') not in self.model.policy_sha256s:
            raise ValueError('qualified policy digest is absent from model collection provenance')
        self.reference = None
        if kind in MODEL_METHODS or kind in ('adaptive','adaptive_curvature'):
            from adaptive_command import AdaptiveCommand
            self.reference = AdaptiveCommand(qualification=self.qualification,gait_name=self.gait_name,
                                             training_seed=self.training_seed,
                                             option='curvature' if kind in ('adaptive_curvature','ensemble_curvature') else 'balanced')
        self.reset()

    def reset(self):
        self.history = StateHistory(); self.last_command = np.zeros(3); self.lpf_yaw = 0.
        self.last_info = {}; self.pending_forecast = None
        if self.reference is not None:self.reference.reset()
        self.statistics = {'steps':0,'interventions':0,'zero_command_bypasses':0,'fixed_fallbacks':0,
                           'uncertainty_abstentions':0,'unsafe_abstentions':0,'unqualified_abstentions':0,
                           'candidate_index_counts':[0]*10,'abs_raw_wz':0.,'abs_executed_wz':0.,
                           'raw_flips':0,'executed_flips':0,'max_risk':0.,'max_uncertainty':0.,
                           'forecast_samples':0,'forecast_velocity_samples':0,'forecast_displacement_samples':0,
                           'forecast_sink_samples':0,'forecast_velocity_squared_error':0.,
                           'forecast_yaw_rate_squared_error':0.,'forecast_displacement_squared_error':0.,'forecast_sink_squared_error':0.}
        self.statistics.update(reference_fallback_steps=0,reference_fallback_reasons={},
            model_authority_steps=0,model_influence_steps=0,observed_danger_stops=0,
            model_statistical_vetoes=0,shadow_steps=0,reference_steps=0,
            active_steps=0,model_evaluated_steps=0,invalid_input_stops=0,
            model_pure_turn_fallback_steps=0,model_liveness_fallback_steps=0,
            model_zero_output_steps=0,model_nonzero_influence_steps=0,
            model_progress_preserving_influence_steps=0)
        self.last_raw_wz = 0.

    @staticmethod
    def envelope(command, previous, dt):
        # Historical fixed arm retained exactly; v2 planners do not reuse its
        # coupled translational slowdown.
        out = np.asarray(command, dtype=float).copy()
        out[:2] = np.clip(out[:2], -.45, .45); out[2] = np.clip(out[2], -.65, .65)
        out[2] = np.clip(out[2], previous[2]-2*dt, previous[2]+2*dt)
        out[:2] *= max(.35, 1-abs(out[2]))
        return out

    def yaw_only(self, desired, dt):
        self.lpf_yaw += dt/max(.2, dt)*(desired[2]-self.lpf_yaw)
        out = np.asarray(desired, dtype=float).copy(); out[2] = self.lpf_yaw
        return out

    def candidates(self, desired, dt, translation_ceiling=None):
        desired = np.asarray(desired, dtype=float)
        bounded = desired.copy()
        if translation_ceiling is not None:
            ceiling = np.asarray(translation_ceiling,dtype=float)[:2]
            bounded[:2] = np.sign(desired[:2])*np.minimum(np.abs(desired[:2]),np.abs(ceiling))
        pure_turn = np.linalg.norm(desired[:2]) <= 1e-8 and abs(desired[2]) > 1e-8
        translations = (1.,) if self.kind == 'ensemble_speed_fixed' or pure_turn else (1., .85, .65)
        qualified = bool(self.motion.get('qualified'))
        min_cmd = self.motion.get('min_forward_command_m_s', float('inf'))
        max_cmd = self.motion.get('max_forward_command_m_s', 0.)
        sequences, labels = [], []
        for scale in translations:
            target = bounded.copy(); target[:2] *= scale
            translating = np.linalg.norm(target[:2]) > 1e-8
            # No reverse/sideways qualification is inferred from forward tests.
            if translating and (not qualified or target[0] < min_cmd-1e-8 or target[0] > max_cmd+1e-8 or abs(target[1])>.02):
                continue
            # A zero-yaw pure-turn target is the explicit complete stop below,
            # never a second candidate mislabeled as task motion.
            for yaw_scale in ((1., .5) if pure_turn else (1., .5, 0.)):
                cmd = target.copy(); cmd[2] *= yaw_scale
                yaw_cap = self.reference.settings.yaw_cap if self.reference is not None else .65
                cmd[2] = np.clip(cmd[2],-yaw_cap,yaw_cap)
                previous_yaw = float(self.last_command[2]); sequence = []
                for _ in range(self.horizon_steps):
                    previous_yaw = np.clip(cmd[2], previous_yaw-2*dt, previous_yaw+2*dt)
                    command = cmd.copy(); command[2] = previous_yaw; sequence.append(command)
                sequences.append(sequence); labels.append({'speed_scale':scale,'yaw_scale':yaw_scale,'stop':False,
                    'task_motion':'walking' if translating else 'turning'})
        sequences.append(np.zeros((self.horizon_steps,3))); labels.append({'speed_scale':0.,'yaw_scale':0.,'stop':True,
            'task_motion':'complete_stop'})
        return np.asarray(sequences, dtype=np.float32), labels

    def choose(self, state, desired, dt, reference_command=None):
        seq, labels = self.candidates(desired, dt,translation_ceiling=reference_command)
        objective = np.asarray(desired,dtype=float).copy()
        if self.kind=='ensemble_curvature' and reference_command is not None:
            # Track the feasible intended curvature, not a zero yaw-filter
            # transient. Every candidate retains the same reference translation
            # ceiling across this local horizon; the next decision updates it.
            objective[:2] = np.asarray(reference_command)[:2]
            objective[2] = np.clip(objective[2],-self.reference.settings.yaw_cap,self.reference.settings.yaw_cap)
        predicted = self.model.rollout(state, seq)
        finite = all(np.isfinite(predicted[k]).all() for k in ('velocity','displacement','tilt','sink','uncertainty','input_rms_peak'))
        if not finite:
            return np.zeros(3), {'candidate_index':len(seq)-1,'candidate_costs':[], 'uncertainty':0.,
                                 'predicted_risk':None,'abstention':'nonfinite_prediction'}
        cal = self.model.payload.get('calibration', {})
        conditional = self.model.payload.get('conditional_calibration', {})
        strata = [calibration_stratum(state,sequence[0]) for sequence in seq]
        local = [conditional.get('bins',{}).get(key,{}) for key in strata]
        authority = np.array([bool(v.get('authority')) and
                              conditional.get('horizon_steps')==self.horizon_steps for v in local])
        pure_turn = np.linalg.norm(desired[:2]) <= 1e-8 and abs(desired[2]) > 1e-8
        # The frozen V3 dataset has zero pure-turn transitions. Its holding
        # calibration consists of stops, not an angular-progress qualification.
        # Neither calm IMU readings nor a small first slew step confer support
        # for an unobserved turn schedule. Keep its predictions as diagnostics.
        if pure_turn:
            authority[:] = False
        # Unidentified strata retain the original global errors in diagnostics.
        # They cannot gain controller authority merely by ignoring those errors.
        margins = [value if value.get('coverage_valid') else cal for value in local]
        tilt_upper = predicted['tilt']+np.array([v.get('horizon_tilt_error_q95',.15) for v in margins])
        sink_upper = predicted['sink']+np.array([v.get('horizon_sink_error_q95',.05) for v in margins])
        state_risk = np.maximum(tilt_upper/.78, np.maximum(sink_upper,0.)/.25)
        uncertainty_ok = predicted['uncertainty'] <= cal.get('rollout_disagreement_limit', 2.)
        support_ok = predicted['input_rms_peak'] <= cal.get('rollout_input_rms_limit', 6.)
        safety_ok = (tilt_upper < .70) & (sink_upper < .22)
        head_authority = bool(self.model.payload.get('calibration_validity',{}).get('event_head_authority'))
        if self.kind != 'dynamics_only' and head_authority:
            safety_ok &= predicted['immediate_event_probability'] < cal.get('event_probability_threshold', .5)
        scale = max(.1, float(np.linalg.norm(objective[:2])))
        tracking = np.mean(((predicted['velocity']-objective)/np.array([scale,scale,.65]))**2, axis=1)
        target_displacement = np.zeros(2); heading = 0.
        for _ in range(self.horizon_steps):
            c,s = np.cos(heading),np.sin(heading)
            target_displacement += dt*np.array([c*objective[0]-s*objective[1],s*objective[0]+c*objective[1]])
            heading += objective[2]*dt
        progress = np.sum((predicted['displacement']-target_displacement)**2, axis=1)/(scale*self.horizon_steps*dt)**2
        command_change = np.mean(((seq-objective)/np.array([.45,.45,.65]))**2, axis=(1,2))
        # Task progress dominates ordinary variation of upright gait severity.
        # Risk is primarily an eligibility constraint, rather than a reward for
        # reducing all walking and stepping motion toward zero.
        cost = tracking+3*progress+.2*command_change+.05*predicted['uncertainty']
        cost += .2*np.maximum(state_risk-.6,0.)**2
        moving = np.array([not v['stop'] for v in labels])
        eligible = safety_ok & uncertainty_ok & support_ok & authority
        eligible[-1] = True  # stopping remains available even when all motion is rejected
        selected = int(np.argmin(np.where(eligible,cost,np.inf)))
        abstention = None
        if pure_turn:
            selected = len(seq)-1; abstention = 'pure_turn_not_qualified'
        elif not moving.any():
            selected = len(seq)-1; abstention = 'unqualified_motion'
        elif not (authority & moving).any():
            selected = len(seq)-1; abstention = 'calibration_without_authority'
        elif not (eligible & moving).any():
            selected = len(seq)-1
            abstention = 'uncertainty_or_support' if not (uncertainty_ok & support_ok & moving).any() else 'predicted_unsafe'
        elif labels[selected]['stop']:
            abstention = 'progress_or_safety_stop'
        records = []
        for i,label in enumerate(labels):
            records.append({**label,'index':i,'cost':float(cost[i]),'velocity_tracking_cost':float(tracking[i]),
                            'progress_cost':float(progress[i]),'command_change_cost':float(command_change[i]),
                            'predicted_velocity':predicted['velocity'][i].tolist(),'predicted_displacement':predicted['displacement'][i].tolist(),
                            'tilt_upper_rad':float(tilt_upper[i]),'sink_upper_m':float(sink_upper[i]),
                            'immediate_event_probability':float(predicted['immediate_event_probability'][i]),
                            'uncertainty':float(predicted['uncertainty'][i]),'input_rms_peak':float(predicted['input_rms_peak'][i]),
                            'eligible':bool(eligible[i]),'state_safety_eligible':bool(safety_ok[i]),
                            'calibration_stratum':strata[i],'calibration_authority':bool(authority[i]),
                            'calibration_status':local[i].get('authority_reason','UNIDENTIFIED_OR_LEGACY'),
                            'event_head_has_authority':head_authority and self.kind!='dynamics_only',
                            'tilt_error_margin_rad':float(margins[i].get('horizon_tilt_error_q95',.15)),
                            'sink_error_margin_m':float(margins[i].get('horizon_sink_error_q95',.05)),
                            'uncertainty_eligible':bool(uncertainty_ok[i] and support_ok[i])})
        return seq[selected,0].astype(float), {'candidate_index':selected,'candidate_costs':records,
                'predicted_risk':float(state_risk[selected]),'uncertainty':float(predicted['uncertainty'][selected]),
                'predicted_velocity':predicted['velocity'][selected].tolist(),
                'predicted_displacement':predicted['displacement'][selected].tolist(),
                'abstention':abstention,'all_motion_rejected':bool(not (eligible & moving).any()),
                'model_authority':bool(authority[selected] and not labels[selected]['stop']),
                'objective_target':objective.tolist(),
                'candidate_translation_ceiling':np.asarray(reference_command)[:2].tolist() if reference_command is not None else None,
                'statistical_bounds_only':True}

    def filter(self, desired_cmd, robot_observation, previous_action, dt):
        reference = None
        if self.reference is not None:
            reference = self.reference.filter(desired_cmd,robot_observation,previous_action,dt)
            self.statistics['reference_steps'] += 1
            if self.reference.last_info.get('invalid_input_stop'):
                # Preserve the reference's finite stop contract before parsing
                # invalid proprioception, and never poison model history.
                self.pending_forecast = None;self.last_command = np.zeros(3)
                self.statistics['steps'] += 1;self.statistics['observed_danger_stops'] += 1
                self.statistics['invalid_input_stops'] += 1
                self.last_info={'requested':self.reference.last_info.get('requested'),
                    'executed':[0.,0.,0.],'observed_danger_stop':True,'invalid_input_stop':True,
                    'decision_source':'observed_danger_stop','model_authority':False,
                    'model_influence':False,'reference_fallback':False,'predicted_risk':None,
                    'uncertainty':0.,'reference_info':self.reference.last_info.copy()}
                return np.zeros(3)
        desired = np.asarray(desired_cmd, dtype=float)
        if desired.shape != (3,) or not np.isfinite(desired).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError('finite3-vector and positive timestep required')
        core = proprio(robot_observation, previous_action); state = self.history.state(core)
        self.pending_forecast = None
        info = {'predicted_risk':None,'uncertainty':0.,'candidate_index':None,'abstention':None,
                'model_authority':False,'model_influence':False,'reference_fallback':False,
                'model_nonzero_influence':False,'model_progress_preserving_influence':False}
        if self.reference is not None:
            info['reference_command'] = reference.tolist()
            info['reference_info'] = self.reference.last_info.copy()
        if self.kind == 'baseline':
            command = desired.copy()
        elif np.max(np.abs(desired)) < 1e-9:
            command = np.zeros(3); self.lpf_yaw = 0.; self.statistics['zero_command_bypasses'] += 1
            info['stop_bypass'] = True
        elif reference is not None and self.reference.last_info.get('observed_danger_stop'):
            command = reference.copy();info['observed_danger_stop'] = True
            info['decision_source'] = 'observed_danger_stop'
            self.statistics['observed_danger_stops'] += 1
        elif self.kind in ('adaptive','adaptive_curvature'):
            command = reference.copy();info['decision_source'] = 'adaptive_reference'
        elif self.kind == 'yaw_only':
            command = self.yaw_only(desired,dt)
        elif self.kind == 'fixed':
            command = self.envelope(desired,self.last_command,dt)
        else:
            if self.model.core_dim != len(core) or abs(self.model.dt-dt)>1e-8:
                raise ValueError('model dimensions/timestep differ from evaluated gait')
            suggested,decision = self.choose(state,desired,dt,reference_command=reference)
            self.statistics['model_evaluated_steps'] += 1
            info.update(decision)
            info['reference_command'] = reference.tolist()
            info['reference_info'] = self.reference.last_info.copy()
            reason = decision.get('abstention')
            requested_translation = np.linalg.norm(desired[:2]) > 1e-8
            requested_turn = not requested_translation and abs(desired[2]) > 1e-8
            zero_suggestion = np.max(np.abs(suggested)) <= 1e-8
            # Guard the executed task channels as well as candidate labels:
            # a zero output cannot masquerade as a useful learned pure turn.
            killed_task_motion = (requested_translation and np.linalg.norm(suggested[:2])<=1e-8) or (
                requested_turn and abs(suggested[2])<=1e-8)
            envelope_violation = (np.any(np.abs(suggested[:2])>np.abs(reference[:2])+1e-7) or
                np.any(suggested[:2]*desired[:2]<-1e-8) or
                abs(suggested[2])>self.reference.settings.yaw_cap+1e-7 or
                abs(suggested[2]-self.last_command[2])>self.reference.settings.yaw_slew*dt+1e-7)
            if envelope_violation and not reason and decision.get('model_authority'):
                reason = 'reference_envelope_violation'
                info['abstention'] = reason
            # A statistical-model stop is a loss of model authority, not evidence
            # of currently measured physical danger. The competent reference
            # remains responsible, including its own observed-danger stop.
            if not decision.get('model_authority') or reason or killed_task_motion:
                command = reference.copy()
                info['reference_fallback'] = True
                info['reference_fallback_reason'] = reason or (
                    'model_task_motion_veto' if killed_task_motion else 'no_task_motion_model_authority')
                info['model_authority'] = False
                self.statistics['reference_fallback_steps'] += 1
                counts = self.statistics['reference_fallback_reasons']
                key = info['reference_fallback_reason'];counts[key] = counts.get(key,0)+1
                self.statistics['model_statistical_vetoes'] += int(bool(reason))
                self.statistics['model_pure_turn_fallback_steps'] += int(reason=='pure_turn_not_qualified')
                self.statistics['model_liveness_fallback_steps'] += int(killed_task_motion)
            else:
                command = suggested
                self.statistics['model_authority_steps'] += int(self.kind!='shadow')
            if self.kind == 'shadow':
                info['shadow_suggested_command'] = suggested.tolist()
                command = reference.copy();info['shadow_only'] = True
                info['model_authority'] = False
                self.statistics['shadow_steps'] += 1
            info['model_influence'] = bool(self.kind!='shadow' and info['model_authority'] and
                                           np.max(np.abs(command-reference))>1e-6)
            self.statistics['model_influence_steps'] += int(info['model_influence'])
            info['model_nonzero_influence'] = bool(info['model_influence'] and np.max(np.abs(command))>1e-8)
            task_active = (requested_translation and np.linalg.norm(command[:2])>1e-8) or (
                requested_turn and desired[2]*command[2]>1e-8)
            info['model_progress_preserving_influence'] = bool(info['model_nonzero_influence'] and task_active)
            self.statistics['model_zero_output_steps'] += int(info['model_authority'] and zero_suggestion and self.kind!='shadow')
            self.statistics['model_nonzero_influence_steps'] += int(info['model_nonzero_influence'])
            self.statistics['model_progress_preserving_influence_steps'] += int(info['model_progress_preserving_influence'])
            info['decision_source'] = 'learned_model' if info['model_authority'] else 'adaptive_reference'
            index=info.get('candidate_index')
            if index is not None:self.statistics['candidate_index_counts'][index] += 1
            if info.get('predicted_risk') is not None:
                self.statistics['max_risk'] = max(self.statistics['max_risk'],info['predicted_risk'])
            self.statistics['max_uncertainty'] = max(self.statistics['max_uncertainty'],info['uncertainty'])
        # Evaluate predictions for the command actually executed, including
        # reference fallback. These labels never enter controller state.
        if self.model is not None and not info.get('stop_bypass') and not info.get('observed_danger_stop'):
            y,p = self.model.predict_members(state[None],command[None])
            if np.isfinite(y).all():self.pending_forecast = y.mean(0)[0,self.model.core_dim:].copy()
        stats=self.statistics;stats['steps']+=1
        stats['active_steps'] += int(np.max(np.abs(desired))>=1e-9)
        stats['interventions'] += int(np.max(np.abs(command-desired))>1e-6)
        stats['abs_raw_wz']+=abs(float(desired[2]));stats['abs_executed_wz']+=abs(float(command[2]))
        stats['raw_flips']+=int(desired[2]*self.last_raw_wz<0);stats['executed_flips']+=int(command[2]*self.last_command[2]<0)
        self.last_raw_wz=float(desired[2]);self.history.advance(core,command);self.last_command=command.copy()
        if self.reference is not None:
            # Its next slew-limited fallback starts from the command actually
            # executed, including an authorized learned intervention.
            self.reference.last_command=command.copy()
        self.last_info={**info,'requested':desired.tolist(),'executed':command.tolist()}
        return command

    def observe_transition(self, robot_observation=None, previous_action=None, achieved_velocity=None,
                           achieved_displacement=None, sink=None):
        # Diagnostic labels never modify planner state or future commands.
        if self.pending_forecast is None:
            return
        stats=self.statistics; stats['forecast_samples']+=1
        if achieved_velocity is not None:
            achieved_velocity=np.asarray(achieved_velocity)
            stats['forecast_velocity_samples']+=1
            stats['forecast_velocity_squared_error']+=float(np.mean((self.pending_forecast[:2]-achieved_velocity[:2])**2))
            stats['forecast_yaw_rate_squared_error']+=float((self.pending_forecast[2]-achieved_velocity[2])**2)
        if achieved_displacement is not None:
            stats['forecast_displacement_samples']+=1
            stats['forecast_displacement_squared_error']+=float(np.mean((self.pending_forecast[3:5]-achieved_displacement)**2))
        if sink is not None:
            stats['forecast_sink_samples']+=1
            stats['forecast_sink_squared_error']+=float((self.pending_forecast[5]-sink)**2)
        self.pending_forecast=None

    def summary(self):
        stats=dict(self.statistics)
        count=stats['forecast_velocity_samples']
        if count:
            stats['forecast_velocity_rmse']=float(np.sqrt(stats['forecast_velocity_squared_error']/count))
            stats['forecast_velocity_rmse_m_s']=stats['forecast_velocity_rmse']
            stats['forecast_yaw_rate_rmse_rad_s']=float(np.sqrt(stats['forecast_yaw_rate_squared_error']/count))
        count=stats['forecast_displacement_samples']
        if count:stats['forecast_displacement_rmse_m']=float(np.sqrt(stats['forecast_displacement_squared_error']/count))
        count=stats['forecast_sink_samples']
        if count:stats['forecast_sink_rmse_m']=float(np.sqrt(stats['forecast_sink_squared_error']/count))
        return {'mode':self.kind,'statistics':stats,'state_schema':SCHEMA,'horizon_steps':self.horizon_steps,
                'model_checkpoint':self.model.checkpoint if self.model else None,
                'model_sha256':self.model.checkpoint_sha256 if self.model else None,
                'qualification_source':self.qualification_source,'motion_qualification':self.motion,
                'reference':self.reference.summary() if self.reference else None,
                'consumed_development':bool(self.model.consumed_development) if self.model else False,
                'calibration_authority':self.model.payload.get('conditional_calibration',{}) if self.model else None,
                'calibration_validity':self.model.payload.get('calibration_validity',{'status':'LEGACY_UNIDENTIFIED'}) if self.model else None,
                'global_calibration':self.model.payload.get('calibration',{}) if self.model else None,
                'pure_turn_policy':'Adaptive reference only: frozen V3 model has no pure-turn training/calibration qualification.',
                'candidate_envelope':'All model candidates obey current reference translation ceiling, qualified minimum, yaw cap and yaw slew. Ensemble-curvature tracks that feasible translation plus capped intended yaw, not transient filtered yaw; future decisions refresh the envelope.',
                'influence_interpretation':'Model influence is an executed deviation from reference. Nonzero/progress-preserving counters describe command activity only, not causal task benefit. Fallback and complete stops are excluded.',
                'scope':'Empirical simulation only. Statistical model rejection falls back to adaptive reference; observed danger still stops. Fallback success is reference performance, not learned improvement. Truth velocity/displacement/sink remain labels only.'}


def _segments(data,mask,horizon,stride=1):
    starts=[]
    for episode in np.unique(data['episode'][mask]):
        idx=np.flatnonzero((data['episode']==episode)&mask)
        if len(idx)>=horizon:
            selected=idx[:len(idx)-horizon+1:stride].tolist()
            # Include the terminal window: a coarse stride must not silently
            # drop the only fall transition at an episode's end.
            if int(idx[-horizon]) not in selected:selected.append(int(idx[-horizon]))
            starts.extend(selected)
    return np.array(starts,dtype=np.int64)


def fit_conditional_calibration(model,data,mask,horizon=8):
    """Empirical error by observable initial regime, with explicit authority.

    All sampled nonoverlapping windows plus every terminal window are included.
    Coverage counts reset clusters, not independent control steps.
    The global event-enriched error report remains intact and is never clipped.
    """
    starts=_segments(data,mask,horizon,stride=horizon)
    records={}
    for offset in range(0,len(starts),256):
        batch=starts[offset:offset+256]
        sequences=np.stack([data['command'][i:i+horizon] for i in batch])
        prediction=model.rollout(data['state'][batch],sequences,return_trajectory=True)
        true_core=np.stack([data['next_core'][batch+t] for t in range(horizon)])
        true_sink=np.stack([data['sink'][batch+t] for t in range(horizon)])
        true_velocity=np.stack([data['velocity'][batch+t] for t in range(horizon)])
        tilt_error=np.maximum(tilt_of(true_core[...,:3])-tilt_of(prediction['core_trajectory'][...,:3]),0.).max(0)
        sink_error=np.maximum(true_sink-prediction['aux_trajectory'][...,5],0.).max(0)
        velocity_error=((true_velocity[...,:2]-prediction['aux_trajectory'][...,:2])**2).mean((0,2))
        for j,index in enumerate(batch):
            key=calibration_stratum(data['state'][index],data['command'][index])
            row=records.setdefault(key,{'tilt':[],'sink':[],'velocity':[],'seeds':set(),'episodes':set(),'events':[]})
            row['tilt'].append(float(tilt_error[j]));row['sink'].append(float(sink_error[j]))
            row['velocity'].append(float(velocity_error[j]));row['seeds'].add(int(data['seed'][index]))
            row['episodes'].add(str(data['episode'][index]));row['events'].append(bool(data['event'][index:index+horizon].max()>.5))
    bins={}
    for key,row in records.items():
        tilt=float(np.quantile(row['tilt'],.95));sink=float(np.quantile(row['sink'],.95))
        event_mask=np.asarray(row['events'],dtype=bool)
        if event_mask.any():
            tilt=max(tilt,float(np.quantile(np.asarray(row['tilt'])[event_mask],.95)))
            sink=max(sink,float(np.quantile(np.asarray(row['sink'])[event_mask],.95)))
        coverage=len(row['seeds'])>=6 and len(row['episodes'])>=12 and len(row['tilt'])>=120
        # Authority is withheld when prediction error consumes over half of the
        # inner state envelope. Withholding authority means reference fallback,
        # never shrinking a measured error to manufacture an eligible motion.
        usable=tilt<.35 and sink<.11
        bins[key]={'coverage_valid':coverage,'authority':bool(coverage and usable),
            'authority_reason':'EMPIRICAL_SUPPORTED' if coverage and usable else
                 ('INSUFFICIENT_CLUSTER_COVERAGE' if not coverage else 'ERROR_CONSUMES_STATE_ENVELOPE'),
            'horizon_tilt_error_q95':tilt,'horizon_sink_error_q95':sink,
            'velocity_rmse_m_s':float(np.sqrt(np.mean(row['velocity']))),
            'windows':len(row['tilt']),'reset_groups':len(row['seeds']),'episodes':len(row['episodes']),
            'event_windows':int(event_mask.sum())}
    return {'horizon_steps':horizon,'bins':bins,
        'coverage_rule':{'minimum_reset_groups':6,'minimum_episodes':12,'minimum_windows':120},
        'authority_error_budget':{'tilt_rad':.35,'sink_m':.11},
        'features':'Current IMU tilt/gyro and candidate command only; no terrain ID or true velocity.',
        'scope':'Empirical stratified calibration, not a safety certificate. All sampled nonoverlapping windows plus every terminal window retained; test never grants authority.'}


def train(data_paths,out,members=5,epochs=35,seed=91273,hidden=128,unroll_steps=4,consumed_development=False):
    out=Path(out)
    if out.exists() or out.with_suffix('.json').exists():
        raise FileExistsError(out)
    import torch
    fit_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch.set_num_threads(1)
    arrays=[];metadata=[]
    fields=('state','command','next_core','velocity','displacement','sink','event','seed','episode')
    for path in data_paths:
        with np.load(path,allow_pickle=False) as d:
            m=json.loads(str(d['metadata']))
            if m.get('schema')!=SCHEMA and not (consumed_development and m.get('schema')==LEGACY_SCHEMA):
                raise ValueError('v3 data required; consumed v2 data requires explicit development flag')
            arrays.append({k:d[k] for k in fields})
            m.update(source_path=str(Path(path).resolve()),source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
            metadata.append(m)
    if not arrays or len({m['gait_name'] for m in metadata})!=1 or len({m['dt'] for m in metadata})!=1:
        raise ValueError('fit exactly one gait configuration and timestep')
    data={k:np.concatenate([a[k] for a in arrays]) for k in fields if k!='episode'}
    data['episode']=np.concatenate([np.array([f'{j}:{e}' for e in a['episode']]) for j,a in enumerate(arrays)])
    core_dim=data['next_core'].shape[1]
    strata={};tags={}
    for s in np.unique(data['seed']):
        sources=[m for a,m in zip(arrays,metadata) if s in a['seed']]
        regimes={(int(s)-m['collection_seed_start'])%4 for m in sources}
        if len(regimes)!=1:
            raise ValueError('shared reset groups have inconsistent regimes')
        strata[int(s)]=(tuple(sorted(m['training_seed'] for m in sources)),next(iter(regimes)))
        block=(int(s)-sources[0]['collection_seed_start'])//4
        tags[int(s)]=(block%2,(block//2)%2) if sources[0].get('reference_mix_by_four_blocks') else block%2
    split=whole_seed_split(data['seed'],strata=strata,balance_tags=tags)
    masks={k:np.isin(data['seed'],v) for k,v in split.items()}
    x=np.concatenate([data['state'],data['command']],axis=1)
    y=np.concatenate([data['next_core']-data['state'][:,:core_dim],data['velocity'],data['displacement'],data['sink'][:,None]],axis=1)
    xm=x[masks['train']].mean(0);xs=x[masks['train']].std(0).clip(.01)
    ym=y[masks['train']].mean(0);ys=y[masks['train']].std(0).clip(.001)
    xt=torch.from_numpy(data['state'].astype(np.float32));cmd=torch.from_numpy(data['command'].astype(np.float32))
    target=torch.from_numpy(y.astype(np.float32));terminal=torch.from_numpy(data['event'].astype(np.float32))
    xm_t,xs_t,ym_t,ys_t=[torch.from_numpy(v) for v in (xm,xs,ym,ys)]
    train_starts=_segments(data,masks['train'],unroll_steps)
    cal_starts=_segments(data,masks['calibration'],unroll_steps,stride=unroll_steps)
    if not len(train_starts) or not len(cal_starts):
        raise ValueError('complete held-out trajectories shorter than unroll horizon')
    rng=np.random.default_rng(seed);weights=[];logs=[]
    pos=float(data['event'][masks['train']].sum());neg=int(masks['train'].sum())-pos
    pos_weight=torch.tensor(min(30.,max(1.,neg/max(1.,pos))),dtype=torch.float32)

    def loss_for(net,starts,horizon,noise=False):
        state=xt[starts].clone(); losses=[]
        if noise:
            # Sensor-scale perturbation, no pose/velocity/terrain augmentation.
            sigma=torch.zeros_like(state);sigma[:,3:6]=.01
            sigma[:,6:core_dim]=.002
            state=state+torch.randn_like(state)*sigma
        for t in range(horizon):
            idx=starts+t
            normalized,logit=net((torch.cat([state,cmd[idx]],1)-xm_t)/xs_t)
            output=normalized*ys_t+ym_t
            actual_core=torch.from_numpy(data['next_core'][idx])
            predicted_core=state[:,:core_dim]+output[:,:core_dim]
            gravity=predicted_core[:,:3]
            gravity=gravity/torch.linalg.vector_norm(gravity,dim=1,keepdim=True).clamp(min=1e-6)
            predicted_core=torch.cat([gravity,predicted_core[:,3:]],dim=1)
            # Autoregressive state error penalizes drift; auxiliary targets give
            # measured body progress, not commanded-speed proxies.
            state_loss=(((predicted_core-actual_core)/ys_t[:core_dim])**2).mean()
            auxiliary=(((output[:,core_dim:]-target[idx,core_dim:])/ys_t[core_dim:])**2).mean()
            event_loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,terminal[idx],pos_weight=pos_weight)
            losses.append(state_loss+2*auxiliary+.25*event_loss)
            state=torch.cat([predicted_core,state[:,:core_dim],cmd[idx],state[:,2*core_dim:2*core_dim+3]],1)
        return torch.stack(losses).mean()

    for mi in range(members):
        torch.manual_seed(seed+mi);net=model_class()(x.shape[1],y.shape[1],hidden)
        opt=torch.optim.Adam(net.parameters(),lr=.001)
        groups=np.unique(data['seed'][masks['train']]);draws=rng.choice(groups,len(groups),replace=True)
        indices=np.concatenate([train_starts[data['seed'][train_starts]==s] for s in draws])
        best=None;best_loss=float('inf')
        for epoch in range(epochs):
            net.train();rng.shuffle(indices)
            horizon=min(unroll_steps,1+epoch//max(1,epochs//unroll_steps))
            for start in range(0,len(indices),512):
                idx=indices[start:start+512]
                loss=loss_for(net,idx,horizon,noise=True)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
            net.eval(); validation=[]
            with torch.inference_mode():
                for start in range(0,len(cal_starts),1024):
                    validation.append(float(loss_for(net,cal_starts[start:start+1024],unroll_steps)))
            value=float(np.mean(validation))
            if value<best_loss:
                best_loss=value;best={k:v.detach().clone() for k,v in net.state_dict().items()}
        weights.append(best);logs.append({'member':mi,'bootstrap_reset_group_draws':len(draws),'best_calibration_multistep_loss':best_loss})
        print(json.dumps(logs[-1]),flush=True)
    payload={'schema':SCHEMA,'core_dim':core_dim,'hidden':hidden,'dt':metadata[0]['dt'],
             'consumed_development':bool(consumed_development),
             'fit_source_sha256':fit_source_hash,
             'x_mean':xm,'x_std':xs,'y_mean':ym,'y_std':ys,'members':weights,'data':metadata,
             'split_reset_seeds':{k:v.tolist() for k,v in split.items()},'split_rows':{k:int(v.sum()) for k,v in masks.items()},
             'fit':logs,'epochs':epochs,'training_seed':seed,'unroll_steps':unroll_steps,
             'split_method':'whole reset/terrain groups stratified by policy and command regime; train pushed/unpushed balance',
             'risk_interpretation':'Candidate-conditioned learned state rollouts plus one-step event head trained on immediate next-state fall. No future behavior-policy severity targets. Probabilities are empirical, not safety guarantees.',
             'target_privilege':'Simulator yaw-frame velocity/displacement/local sink/termination are labels only. No true velocity, sink, terrain ID, or root pose input.'}
    model=DynamicsEnsemble(payload=payload,consumed_development=consumed_development)
    # Weighted training BCE learns a rare-event classifier. Fit its probability
    # scale/offset on unweighted, whole-group calibration data before choosing
    # thresholds. The untouched test set never fits calibration parameters.
    _,raw_probability=model.predict_members(data['state'][masks['calibration']],data['command'][masks['calibration']])
    cal_labels=torch.from_numpy(data['event'][masks['calibration']].astype(np.float32))
    raw_logits=torch.from_numpy(np.log(np.clip(raw_probability,1e-6,1-1e-6)/(1-np.clip(raw_probability,1e-6,1-1e-6))))
    if 0<float(cal_labels.sum())<len(cal_labels):
        log_scale=torch.zeros((),requires_grad=True);bias=torch.zeros((),requires_grad=True)
        calibration_optimizer=torch.optim.Adam([log_scale,bias],lr=.03)
        for _ in range(200):
            calibrated=torch.sigmoid(raw_logits*log_scale.exp().clamp(.1,10.)+bias.clamp(-10.,10.)).mean(0)
            calibration_loss=torch.nn.functional.binary_cross_entropy(calibrated,cal_labels)
            calibration_optimizer.zero_grad();calibration_loss.backward();calibration_optimizer.step()
        payload['event_calibration']={'scale':float(log_scale.exp().clamp(.1,10.).detach()),'bias':float(bias.clamp(-10.,10.).detach()),
                                      'status':'CALIBRATED_ON_WHOLE_RESET_GROUP_HOLDOUT','positive_rows':int(cal_labels.sum())}
    else:
        payload['event_calibration']={'scale':1.,'bias':0.,'status':'UNIDENTIFIED_CLASS_COVERAGE','positive_rows':int(cal_labels.sum())}
    diagnostics={}; calibration={}
    for name in ('calibration','test'):
        mask=masks[name];prediction,event=model.predict_members(data['state'][mask],data['command'][mask])
        mean=prediction.mean(0);actual=y[mask];prob=event.mean(0)
        diag={'rows':int(mask.sum()),'normalized_delta_rmse':float(np.sqrt(np.mean(((mean[:,:core_dim]-actual[:,:core_dim])/ys[:core_dim])**2))),
              'velocity_rmse_m_s':float(np.sqrt(np.mean((mean[:,core_dim:core_dim+2]-actual[:,core_dim:core_dim+2])**2))),
              'yaw_rate_rmse_rad_s':float(np.sqrt(np.mean((mean[:,core_dim+2]-actual[:,core_dim+2])**2))),
              'displacement_rmse_m':float(np.sqrt(np.mean((mean[:,core_dim+3:core_dim+5]-actual[:,core_dim+3:core_dim+5])**2))),
              'sink_rmse_m':float(np.sqrt(np.mean((mean[:,-1]-actual[:,-1])**2))),
              'immediate_event_brier':float(np.mean((prob-data['event'][mask])**2)),
              'immediate_event_positive_rows':int(data['event'][mask].sum()),'horizons':{}}
        diag['event_reliability_bins']=[{'probability_interval':[low,high],'rows':int(((prob>=low)&(prob<high)).sum()),
             'mean_predicted':float(prob[(prob>=low)&(prob<high)].mean()) if ((prob>=low)&(prob<high)).any() else None,
             'observed_event_fraction':float(data['event'][mask][(prob>=low)&(prob<high)].mean()) if ((prob>=low)&(prob<high)).any() else None}
             for low,high in zip((0.,.01,.05,.1,.25,.5),(.01,.05,.1,.25,.5,1.00001))]
        if name=='calibration':
            positives=prob[data['event'][mask]>.5]
            # At least95% sensitivity on observed calibration positives when
            # possible; clamp prevents zero-positive data silently clearing risk.
            threshold=float(np.quantile(positives,.05,method='lower')) if len(positives) else .25
            calibration['event_probability_threshold']=max(.05,min(.5,threshold))
        threshold=calibration['event_probability_threshold']
        posmask=data['event'][mask]>.5
        diag['immediate_event_miss_fraction']=float(np.mean(prob[posmask]<threshold)) if posmask.any() else None
        diag['immediate_event_false_alarm_fraction']=float(np.mean(prob[~posmask]>=threshold)) if (~posmask).any() else None
        for horizon in (1,4,8,16):
            starts=_segments(data,mask,horizon,stride=horizon)
            if not len(starts):continue
            # Preserve scarce terminal-event windows. Uniformly capping all
            # windows can silently exclude precisely the errors used to assess
            # fall-state predictions. Sample only the remaining normal windows.
            event_windows=np.array([data['event'][i:i+horizon].max()>.5 for i in starts])
            event_starts=starts[event_windows]
            normal_starts=starts[~event_windows]
            normal_count=min(len(normal_starts),max(0,384-len(event_starts)))
            starts=np.concatenate([event_starts,rng.choice(normal_starts,normal_count,replace=False)])
            sequences=np.stack([data['command'][i:i+horizon] for i in starts])
            roll=model.rollout(data['state'][starts],sequences,return_trajectory=True)
            true_core=np.stack([data['next_core'][starts+t] for t in range(horizon)])
            true_sink=np.stack([data['sink'][starts+t] for t in range(horizon)])
            true_velocity=np.stack([data['velocity'][starts+t] for t in range(horizon)])
            true_displacement=np.zeros((len(starts),2));true_heading=np.zeros(len(starts))
            for t in range(horizon):
                delta=data['displacement'][starts+t]
                c,s=np.cos(true_heading),np.sin(true_heading)
                true_displacement[:,0]+=c*delta[:,0]-s*delta[:,1]
                true_displacement[:,1]+=s*delta[:,0]+c*delta[:,1]
                true_heading+=true_velocity[t,:,2]*metadata[0]['dt']
            predicted_tilt=tilt_of(roll['core_trajectory'][...,:3]);true_tilt=tilt_of(true_core[...,:3])
            diag['horizons'][str(horizon)]={'segments':len(starts),'normalized_endpoint_rmse':float(np.sqrt(np.mean(((roll['next_core_mean']-true_core[-1])/ys[:core_dim])**2))),
                'velocity_rmse_m_s':float(np.sqrt(np.mean((roll['aux_trajectory'][...,:2]-true_velocity[...,:2])**2))),
                'yaw_rate_rmse_rad_s':float(np.sqrt(np.mean((roll['aux_trajectory'][...,2]-true_velocity[...,2])**2))),
                'displacement_endpoint_rmse_m':float(np.sqrt(np.mean((roll['displacement']-true_displacement)**2))),
                'tilt_rmse_rad':float(np.sqrt(np.mean((predicted_tilt-true_tilt)**2))),
                'sink_rmse_m':float(np.sqrt(np.mean((roll['aux_trajectory'][...,5]-true_sink)**2)))}
            event_segments=np.array([data['event'][i:i+horizon].max()>.5 for i in starts])
            diag['horizons'][str(horizon)]['event_segments']=int(event_segments.sum())
            if name=='calibration' and horizon==8:
                tilt_error=np.maximum(true_tilt-predicted_tilt,0.).max(0)
                sink_error=np.maximum(true_sink-roll['aux_trajectory'][...,5],0.).max(0)
                tilt_margin=float(np.quantile(tilt_error,.95));sink_margin=float(np.quantile(sink_error,.95))
                if event_segments.any():
                    tilt_margin=max(tilt_margin,float(np.quantile(tilt_error[event_segments],.95)))
                    sink_margin=max(sink_margin,float(np.quantile(sink_error[event_segments],.95)))
                calibration.update(horizon_tilt_error_q95=tilt_margin,
                    horizon_sink_error_q95=sink_margin,
                    rollout_disagreement_limit=max(.25,3*float(np.quantile(roll['uncertainty'],.95))),
                    rollout_input_rms_limit=max(3.,2*float(np.quantile(roll['input_rms_peak'],.99))),
                    segments=len(starts),horizon_steps=8)
        diag['scope']='Complete episode-local segments only. Horizon diagnostics retain all terminal-event windows and sample up to384 total windows, so their errors are event-enriched diagnostics rather than deployment-population estimates. Calibration tunes margins/thresholds; test never selects parameters.'
        diagnostics[name]=diag
    payload['calibration']=calibration
    payload['diagnostics']=diagnostics
    payload['conditional_calibration']=fit_conditional_calibration(model,data,masks['calibration'])
    test_strata=fit_conditional_calibration(model,data,masks['test'])
    for row in test_strata['bins'].values():
        row['authority']=False;row['authority_reason']='TEST_DIAGNOSTIC_ONLY'
    diagnostics['test']['observable_strata']=test_strata
    payload['calibration_validity']={'event_head_positive_rows':int(cal_labels.sum()),
        'event_head_miss_fraction':diagnostics['calibration']['immediate_event_miss_fraction'],
        'event_head_authority':bool(int(cal_labels.sum())>=20 and
             diagnostics['calibration']['immediate_event_miss_fraction'] is not None and
             diagnostics['calibration']['immediate_event_miss_fraction']<=.05),
        'global_errors_preserved':True,
        'scope':'Invalid statistical components cannot force a global physical stop. Reference fallback is counted separately from learned influence.'}
    payload['split_source_strata']={name:[{'training_seed':m['training_seed'],'reset_seeds':sorted(set(a['seed'][np.isin(a['seed'],group)].tolist()))}
               for a,m in zip(arrays,metadata)] for name,group in split.items()}
    payload['split_regime_coverage']={name:[{
        'training_seed':m['training_seed'],'command_regime':regime,
        'reset_groups':len(selected:={int(s) for s in a['seed'] if int(s) in set(group.tolist()) and
                    (int(s)-m['collection_seed_start'])%4==regime}),
        'pushed_reset_groups':sum(((s-m['collection_seed_start'])//(8 if m.get('reference_mix_by_four_blocks') else 4))%2==1 for s in selected),
        'unpushed_reset_groups':sum(((s-m['collection_seed_start'])//(8 if m.get('reference_mix_by_four_blocks') else 4))%2==0 for s in selected),
        'adaptive_reset_groups':sum(((s-m['collection_seed_start'])//4)%2==1 for s in selected) if m.get('reference_mix_by_four_blocks') else None,
        'raw_reset_groups':sum(((s-m['collection_seed_start'])//4)%2==0 for s in selected) if m.get('reference_mix_by_four_blocks') else None}
        for a,m in zip(arrays,metadata) for regime in range(4)] for name,group in split.items()}
    out.parent.mkdir(parents=True,exist_ok=True)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=fit_source_hash:
        raise RuntimeError('Fitting source changed during training; checkpoint not published')
    with out.open('xb') as f:torch.save(payload,f)
    report={k:v for k,v in payload.items() if k not in ('members','x_mean','x_std','y_mean','y_std')}
    report['checkpoint_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
    with out.with_suffix('.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'checkpoint':str(out),'diagnostics':diagnostics,'calibration':calibration}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('train',));parser.add_argument('--data',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--members',type=int,default=5)
    parser.add_argument('--epochs',type=int,default=35);parser.add_argument('--seed',type=int,default=91273)
    parser.add_argument('--unroll-steps',type=int,default=4)
    parser.add_argument('--consumed-development',action='store_true')
    args=parser.parse_args()
    if args.members<2 or args.epochs<1 or args.unroll_steps<1:parser.error('>=2members, >=1epoch andpositive unroll required')
    train(args.data,args.out,args.members,args.epochs,args.seed,unroll_steps=args.unroll_steps,
          consumed_development=args.consumed_development)


if __name__=='__main__':main()
