"""Causal IMU turn moderation with no blanket forward-speed penalty.

These are two predeclared development laws, shared across all policy/maze seeds.
The controller is an empirical simulated command filter, not a safety proof.
It never increases either component of the upstream translation command.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Settings:
    yaw_cap: float
    yaw_tau: float
    yaw_slew: float
    lookahead_s: float = .15
    risk_start_rad: float = .18
    risk_full_rad: float = .48
    yaw_min_fraction: float = .35
    risk_release_s: float = .45
    translation_start_rad: float = .38
    translation_full_rad: float = .60
    translation_min_fraction: float = .80
    danger_tilt_rad: float = .65
    danger_rate_rad_s: float = 5.
    danger_rate_min_tilt_rad: float = .25
    recovery_tilt_rad: float = .18
    recovery_rate_rad_s: float = .75
    recovery_s: float = .40


OPTIONS = {
    'balanced': Settings(yaw_cap=.65,yaw_tau=.18,yaw_slew=2.),
    'guarded': Settings(yaw_cap=.55,yaw_tau=.24,yaw_slew=1.5),
    # Shared settings: this ablation changes translation curvature only.
    'curvature': Settings(yaw_cap=.65,yaw_tau=.18,yaw_slew=2.),
}

CURVATURE_ONSET_S = .30


def attitude(robot_observation, previous_action):
    """Return tilt and its outward rate from WXYZ attitude and body gyro.

    The upstream observation is quaternion4, gyro3, joints2N, mode1, command3.
    Gravity's body-frame derivative is -omega cross gravity. IMU inputs only.
    """
    obs = np.asarray(robot_observation,dtype=float)
    action = np.asarray(previous_action,dtype=float).reshape(-1)
    if obs.shape!=(11+2*len(action),) or not np.isfinite(obs).all() or not np.isfinite(action).all():
        raise ValueError('finite upstream robot observation/action required')
    if np.max(np.abs(obs[4:7]))>1e4:
        raise ValueError('unphysical IMU gyro magnitude')
    q = obs[:4]
    norm = float(np.linalg.norm(q))
    if norm<1e-8 or not math.isfinite(norm):
        raise ValueError('nondegenerate IMU quaternion required')
    w,x,y,z = q/norm
    gravity = np.array([2*(w*y-x*z),-2*(w*x+y*z),2*(x*x+y*y)-1.])
    tilt = float(np.arccos(np.clip(-gravity[2],-1.,1.)))
    gravity_rate = -np.cross(obs[4:7],gravity)
    outward = max(0.,float(gravity_rate[2])/max(math.sin(tilt),.05))
    rate = float(np.linalg.norm(obs[4:6]))
    if not all(math.isfinite(v) for v in (tilt,outward,rate)):
        raise ValueError('nonfinite derived IMU attitude')
    return tilt,outward,rate


class AdaptiveCommand:
    def __init__(self,qualification=None,gait_name=None,training_seed=None,option='balanced'):
        if option not in OPTIONS:
            raise ValueError(f'unknown adaptive option {option!r}; expected {tuple(OPTIONS)}')
        self.option,self.settings = option,OPTIONS[option]
        self.qualification_source = None
        if isinstance(qualification,(str,Path)):
            self.qualification_source = str(Path(qualification).resolve())
            qualification = json.loads(Path(qualification).read_text())
        self.motion = (qualification or {}).get('policies',{}).get(f'{gait_name}-s{training_seed}',{})
        lower = self.motion.get('min_forward_command_m_s',.15)
        self.minimum_forward = float(lower) if isinstance(lower,(int,float)) and math.isfinite(lower) and lower>0 else .15
        self.reset()

    def reset(self):
        self.last_command = np.zeros(3)
        self.last_raw_yaw = 0.
        self.risk = 0.
        self.danger_latched = False
        self.healthy_s = 0.
        self.turn_hold_s = 0.
        self.turn_sign = 0.
        self.last_info = {}
        self.statistics = dict(steps=0,interventions=0,zero_command_bypasses=0,
            observed_danger_stops=0,invalid_input_stops=0,below_min_translation_preserved=0,
            translation_attenuations=0,raw_yaw_flips=0,executed_yaw_flips=0,
            abs_raw_wz=0.,abs_executed_wz=0.,max_tilt_rad=0.,max_predictive_tilt_rad=0.)
        self.statistics.update(curvature_attenuations=0,curvature_active_steps=0)

    def _record(self,desired,command,info):
        stats = self.statistics
        stats['steps'] += 1
        if desired is not None:
            stats['interventions'] += int(np.max(np.abs(command-desired))>1e-9)
            stats['raw_yaw_flips'] += int(desired[2]*self.last_raw_yaw<0)
            stats['abs_raw_wz'] += abs(float(desired[2]))
            self.last_raw_yaw = float(desired[2])
        stats['executed_yaw_flips'] += int(command[2]*self.last_command[2]<0)
        stats['abs_executed_wz'] += abs(float(command[2]))
        self.last_command = command.copy()
        self.last_info = {'observed_danger_stop':False,'below_min_translation_preserved':False,
            **info,'requested':desired.tolist() if desired is not None else None,'executed':command.tolist()}
        return command

    def filter(self,desired_cmd,robot_observation,previous_action,dt):
        desired = None
        try:
            desired = np.asarray(desired_cmd,dtype=float)
            if desired.shape!=(3,) or not np.isfinite(desired).all():
                desired = None
                raise ValueError('finite command3 required')
            # Far outside every declared task; stop rather than overflow metrics
            # or feed an unphysical command to the frozen gait.
            if np.any(np.abs(desired)>np.array([2.,2.,10.])):
                desired = None
                raise ValueError('unphysical upstream command magnitude')
            if not isinstance(dt,(int,float,np.floating)) or not math.isfinite(dt) or dt<=0:
                raise ValueError('finite positive dt required')
            tilt,outward,rate = attitude(robot_observation,previous_action)
        except (ValueError,TypeError,OverflowError):
            self.danger_latched,self.healthy_s,self.risk = True,0.,1.
            self.statistics['invalid_input_stops'] += 1
            return self._record(desired,np.zeros(3),dict(observed_danger_stop=True,invalid_input_stop=True))
        s = self.settings
        predicted_tilt = min(math.pi,tilt+s.lookahead_s*outward)
        instantaneous = float(np.clip((predicted_tilt-s.risk_start_rad)/(s.risk_full_rad-s.risk_start_rad),0.,1.))
        self.risk = max(instantaneous,self.risk*math.exp(-dt/s.risk_release_s))
        dangerous = tilt>=s.danger_tilt_rad or (tilt>=s.danger_rate_min_tilt_rad and rate>=s.danger_rate_rad_s)
        if dangerous:
            self.danger_latched,self.healthy_s = True,0.
        elif self.danger_latched:
            self.healthy_s = self.healthy_s+dt if tilt<=s.recovery_tilt_rad and rate<=s.recovery_rate_rad_s else 0.
            if self.healthy_s+1e-12>=s.recovery_s:
                self.danger_latched,self.healthy_s = False,0.
        info = dict(measured_tilt_rad=tilt,outward_tilt_rate_rad_s=outward,
                    predictive_tilt_rad=predicted_tilt,measured_rp_rate_rad_s=rate,
                    turn_risk=self.risk,option=self.option)
        self.statistics['max_tilt_rad'] = max(self.statistics['max_tilt_rad'],tilt)
        self.statistics['max_predictive_tilt_rad'] = max(self.statistics['max_predictive_tilt_rad'],predicted_tilt)
        if self.danger_latched:
            self.turn_hold_s,self.turn_sign = 0.,0.
            self.statistics['observed_danger_stops'] += 1
            return self._record(desired,np.zeros(3),dict(info,observed_danger_stop=True))
        if not np.any(desired):
            self.turn_hold_s,self.turn_sign = 0.,0.
            self.statistics['zero_command_bypasses'] += 1
            return self._record(desired,np.zeros(3),dict(info,stop_bypass=True))
        command = desired.copy()
        yaw_fraction = 1.-(1.-s.yaw_min_fraction)*self.risk
        target_yaw = float(np.clip(desired[2],-s.yaw_cap,s.yaw_cap))*yaw_fraction
        # Exact exponential smoothing has consistent behavior across control dt.
        yaw = self.last_command[2]+(-math.expm1(-dt/s.yaw_tau))*(target_yaw-self.last_command[2])
        command[2] = np.clip(yaw,self.last_command[2]-s.yaw_slew*dt,self.last_command[2]+s.yaw_slew*dt)
        below_min = 0.<abs(desired[0])<self.minimum_forward
        factor = 1.
        curvature_factor = 1.
        if self.option == 'curvature':
            # Stable-sign sustained turns only. The magnitude is the upstream
            # request, never the near-zero yaw transient after a reversal/stop.
            sign = float(np.sign(desired[2]))
            high_yaw = abs(desired[2]) > s.yaw_cap
            if high_yaw:
                self.turn_hold_s = self.turn_hold_s+dt if sign==self.turn_sign else dt
                self.turn_sign = sign
                onset = min(1.,self.turn_hold_s/CURVATURE_ONSET_S)
                curvature_factor = 1.-onset*(1.-s.yaw_cap/abs(desired[2]))
            else:
                self.turn_hold_s,self.turn_sign = 0.,0.
        if not below_min:
            severity = float(np.clip((tilt-s.translation_start_rad)/(s.translation_full_rad-s.translation_start_rad),0.,1.))
            factor = 1.-(1.-s.translation_min_fraction)*severity
            factor = min(factor,curvature_factor)
            # A reduction must not turn already effective qualified forward motion
            # into the known deadband. This can only increase factor toward1.
            if desired[0]>=self.minimum_forward:
                factor = max(factor,self.minimum_forward/desired[0])
            command[:2] *= factor
        else:
            self.statistics['below_min_translation_preserved'] += 1
        self.statistics['translation_attenuations'] += int(factor<1.-1e-12)
        curvature_active = self.option=='curvature' and not below_min and curvature_factor<1.-1e-12
        self.statistics['curvature_active_steps'] += int(curvature_active)
        self.statistics['curvature_attenuations'] += int(curvature_active and factor<1.-1e-12)
        return self._record(desired,command,dict(info,translation_fraction=factor,
                yaw_fraction=yaw_fraction,below_min_translation_preserved=bool(below_min),
                curvature_fraction=curvature_factor,curvature_turn_hold_s=self.turn_hold_s))

    def summary(self):
        return dict(mode='adaptive',option=self.option,settings=asdict(self.settings),
            statistics=dict(self.statistics),model_sha256=None,
            qualification_source=self.qualification_source,motion_qualification=self.motion,
            minimum_forward_command_m_s=self.minimum_forward,
            curvature_rule={'onset_s':CURVATURE_ONSET_S,
                'formula':'stable-sign abs(upstream_yaw)>yaw_cap: ramp factor1 to yaw_cap/abs(upstream_yaw); preserve qualified minimum and below-min brakes',
                'status':'predeclared engineering ablation; benefit not established'} if self.option=='curvature' else None,
            scope='Shared causal IMU command filter; no per-policy/maze outcome lookup. Upstream translation never boosted; finite-sample simulation validation required.')
