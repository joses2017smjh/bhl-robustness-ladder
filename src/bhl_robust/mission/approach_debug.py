"""Privileged diagnostics; no changes to production Mission7 scoring or rewards."""
from __future__ import annotations
import numpy as np
import mujoco
from bhl_robust.mission.overnight import StudyEnv


def yaw_of(env):
    q=env.runner.d.qpos[env.slot.qpos_adr+3:env.slot.qpos_adr+7]
    return float(np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2)))


def wrap(angle):
    return float(np.arctan2(np.sin(angle),np.cos(angle)))


def command_action(command):
    return np.r_[np.arctanh(np.clip(np.asarray(command)/[.4,.35,.4],-.999999,.999999)),0.,0.]


def physical_sample(env, command, phase):
    r,s=env.runner,env.slot
    contacts=[]
    for contact in r.d.contact:
        for geom in (contact.geom1,contact.geom2):
            if int(geom) in r.walls:
                contacts.append(mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,int(geom)))
    yaw = yaw_of(env)
    velocity = r.d.qvel[s.qvel_adr:s.qvel_adr+2].copy()
    rotation = np.array([[np.cos(yaw), np.sin(yaw)],
                         [-np.sin(yaw), np.cos(yaw)]])
    command = np.asarray(command, dtype=float)
    return dict(time_s=float(r.d.time),xy=r.d.xpos[s.body_id,:2].tolist(),
                velocity=velocity.tolist(),velocity_body=(rotation @ velocity).tolist(),
                command_world=(rotation.T @ command[:2]).tolist(),command=command.tolist(),yaw=yaw,
                yaw_rate=float(r.d.qvel[s.qvel_adr+5]),tilt=float(r.tilt(0)),
                phase=phase,contacts=sorted(set(contacts)))


def classify_fall(trace):
    if not trace:return None
    hit=next((i for i,r in enumerate(trace) if r['tilt']>=.78),None)
    if hit is None:return None
    recent=trace[max(0,hit-10):hit+1]; last=recent[-1]
    names={name for row in recent for name in row['contacts']}
    if any(n.startswith('door_') for n in names): cause='door collision'
    elif names: cause='wall/post collision'
    elif last['phase'] in ('turn','brake','accelerate','acquire','release','recover'):
        cause={'turn':'during turning','brake':'during braking','accelerate':'during acceleration',
               'acquire':'object interaction','release':'object interaction','recover':'post-interaction recovery'}[last['phase']]
    elif abs(last['command'][2])>.1:cause='during turning'
    else:cause='other'
    return dict(category=cause,time_s=last['time_s'],command=last['command'],phase=last['phase'],
                contacts=sorted(names),classification='temporal association, not established causality')


class DebugEnv(StudyEnv):
    def __init__(self,*args,hold_ticks=5,dense=False,**kwargs):
        super().__init__(*args,**kwargs)
        if hold_ticks not in (1,2,5,10,20,25,50):raise ValueError('unsupported diagnostic hold')
        self.repeat=hold_ticks;self.dense=dense

    def distance(self):
        return float(np.linalg.norm(self.runner.d.xpos[self.slot.body_id,:2]-self.layout.xy(self.layout.route[-1])))

    def reset(self,layout_index=None):
        obs=super().reset(layout_index)
        self.initial_distance=self.minimum_distance=self.distance();self.closest_s=0.
        self.phase='policy'; self.active_command=np.zeros(3);self.diagnostic_trace=[]
        self.progress_reward=0.;self.decision_rewards=[]
        update=self.state.update
        def monitored_update(**kwargs):
            reward=update(**kwargs)
            distance=self.distance()
            if distance<self.minimum_distance:self.minimum_distance=distance;self.closest_s=float(self.runner.d.time)
            self.diagnostic_trace.append(physical_sample(self,self.active_command,self.phase))
            return reward
        self.state.update=monitored_update
        return obs

    def step(self,action):
        distance=self.distance();self.active_command=np.tanh(action[:3])*[.4,.35,.4]
        obs,reward,done,info=super().step(action)
        progress=10.*(distance-self.distance()) if self.dense else 0.
        self.progress_reward+=progress
        terms={**info['reward_terms'],'privileged_distance_progress':progress}
        self.decision_rewards.append(dict(time_s=float(self.runner.d.time),distance_before_m=distance,
            distance_after_m=self.distance(),inside_goal=self.distance()<.36,reward_terms=terms))
        return obs,reward+progress,done,{**info,'diagnostic_reward_terms':terms}

    def metrics(self):
        row=super().metrics()
        row.update(initial_goal_distance_m=self.initial_distance,minimum_goal_distance_m=self.minimum_distance,
            final_goal_distance_m=self.distance(),fraction_distance_closed=(self.initial_distance-self.minimum_distance)/max(self.initial_distance,1e-9),
            final_fraction_distance_closed=(self.initial_distance-self.distance())/max(self.initial_distance,1e-9),
            time_closest_s=self.closest_s,goal_region_entered=self.minimum_distance<.36,
            diagnostic_return_total=row['return_total']+self.progress_reward,
            privileged_distance_progress=self.progress_reward,action_hold_s=self.repeat*self.dt,
            fall_classification=classify_fall(self.diagnostic_trace),diagnostic_trace=self.diagnostic_trace,
            decision_rewards=self.decision_rewards)
        return row


class BearingController:
    """Simple turn-forward-brake oracle. Parameters come from the gait study."""
    def __init__(self,env,speed=.4,turn_rate=.35,stop_radius=.25,heading_tolerance=.25):
        self.env=env;self.speed=speed;self.turn_rate=turn_rate
        self.stop_radius=stop_radius;self.heading_tolerance=heading_tolerance

    def action(self,target=None,allow_goal_stop=True):
        env=self.env; r=env.runner
        if r.d.time<1.2:env.phase='settle';return np.zeros(5)
        target=env.layout.xy(env.layout.route[-1]) if target is None else np.asarray(target)
        delta=target-r.d.xpos[env.slot.body_id,:2];distance=np.linalg.norm(delta)
        error=wrap(np.arctan2(delta[1],delta[0])-yaw_of(env))
        if distance<self.stop_radius:
            env.phase='brake';return np.zeros(5)
        if abs(error)>self.heading_tolerance:
            env.phase='turn';return command_action([0,0,np.sign(error)*self.turn_rate])
        env.phase='accelerate' if r.d.time<2. else 'advance'
        # A low speed can fall into the frozen gait's dead zone. Use measured
        # movement speed until the brake radius, rather than guessing a taper.
        return command_action([self.speed,0,np.clip(error,-self.turn_rate,self.turn_rate)])


class TranslationController:
    """Privilege-labeled planar control within the unchanged Mission7 bounds."""
    def __init__(self,env,speed=.4,stop_radius=.28,settle_s=1.2):
        self.env=env;self.speed=speed;self.stop_radius=stop_radius;self.settle_s=settle_s

    def action(self,target=None):
        env=self.env;r=env.runner
        if r.d.time<self.settle_s:env.phase='settle';return np.zeros(5)
        target=env.layout.xy(env.layout.route[-1]) if target is None else np.asarray(target)
        delta=target-r.d.xpos[env.slot.body_id,:2];distance=np.linalg.norm(delta)
        if distance<self.stop_radius:env.phase='brake';return np.zeros(5)
        velocity=delta/max(distance,1e-9)*self.speed;y=yaw_of(env)
        body=np.array([[np.cos(y),np.sin(y)],[-np.sin(y),np.cos(y)]])@velocity
        env.phase='advance'
        return command_action([body[0],body[1],np.clip(-1.2*y,-.35,.35)])


class RecoveryTranslationController(TranslationController):
    """One measured forward pulse if positive lateral movement fails to start."""
    def __init__(self,env,speed=.4,stop_radius=.28,settle_s=1.2):
        super().__init__(env,speed=speed,stop_radius=stop_radius,settle_s=settle_s);self.kicked=False;self.kick_until=0.

    def action(self,target=None):
        env=self.env;r=env.runner
        target=env.layout.xy(env.layout.route[-1]) if target is None else np.asarray(target)
        delta=target-r.d.xpos[env.slot.body_id,:2]
        if not self.kicked and r.d.time>=2.2 and delta[1]>.35 and env.initial_distance-env.distance()<.05:
            self.kicked=True;self.kick_until=float(r.d.time)+.4
        if r.d.time<self.kick_until:
            env.phase='accelerate';return command_action([.3,0,0])
        return super().action(target)


class GoalPostGuardController(RecoveryTranslationController):
    """Translation controller that centers before crossing the goal-post line."""
    def __init__(self, env, speed=.50, stop_radius=.28, settle_s=1.2):
        super().__init__(env, speed=speed, stop_radius=stop_radius, settle_s=settle_s)
        self.guard_until = 0.

    def action(self, target=None):
        env, r, s = self.env, self.env.runner, self.env.slot
        now = float(r.d.time)
        goal = env.layout.xy(env.layout.route[-1])
        incoming = env.layout.xy(env.layout.route[-1])-env.layout.xy(env.layout.route[-2])
        incoming /= max(np.linalg.norm(incoming), 1e-9)
        xy = r.d.xpos[s.body_id, :2].copy()
        if incoming[0] < -.9:
            post_x = goal[0] + .52
            nearest_y = goal[1] - .35 if xy[1] <= goal[1] else goal[1] + .35
            close_to_line = post_x < xy[0] < post_x + .48
            near_post = np.linalg.norm(xy-np.array([post_x, nearest_y])) < .46
            if now < self.guard_until or (close_to_line and near_post):
                if now >= self.guard_until:
                    dy = xy[1]-goal[1]
                    if abs(dy) > .02:
                        sign = -1. if dy > 0 else 1.
                    else:
                        sign = 0.
                    self.guard_until = now + .40
                    self.guard_sign = sign
                yaw = yaw_of(env)
                rot = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
                # Apply a full lateral gait command while the guard is active;
                # this centers an initially close spawn before forward motion.
                body = rot @ np.array([0., self.guard_sign*.35])
                env.phase = 'post_guard'
                return command_action([body[0], body[1], 0.])
        return super().action(target)


class PulseApproachController:
    """Privileged approach controller built from measured gait pulses.

    Commands stay at the measured 0.30 m/s onset threshold.  Motion is made
    from short pulses followed by standstill intervals, with a final slow
    dwell.  Negative-X approaches detour around the goal posts before the
    final straight-in translation; the posts and all benchmark geometry stay
    enabled.
    """
    def __init__(self, env, *, speed=.30, pulse_s=.40, pause_s=.20,
                 stop_radius=.29, side_offset=.90):
        self.env = env
        self.speed = float(speed)
        self.pulse_s = float(pulse_s)
        self.pause_s = float(pause_s)
        self.stop_radius = float(stop_radius)
        self.side_offset = float(side_offset)
        self.pulse_until = 0.
        self.pause_until = 0.
        self.kick_until = 0.
        self.last_target_distance = float("inf")
        self.last_progress_s = 1.2
        self.waypoint = 0
        self.phase_history = []
        r, s = env.runner, env.slot
        self.goal = env.layout.xy(env.layout.route[-1]).astype(float)
        prev = env.layout.xy(env.layout.route[-2]).astype(float)
        self.goal_direction = (self.goal-prev) / max(np.linalg.norm(self.goal-prev), 1e-9)
        current = r.d.xpos[s.body_id, :2].copy()
        self.waypoints = [self.goal]
        # The goal posts are at goal+[+.52, +/- .35].  A negative-X start
        # crosses their x coordinate, so move to the interior side first.
        if self.goal_direction[0] < -.9:
            goal_cell_y = env.layout.route[-1][1]
            side = 1. if goal_cell_y <= 2 else -1.
            if 2 <= goal_cell_y <= 3:
                side = 1. if current[1] <= self.goal[1] else -1.
            side_point = self.goal + np.array([0., side*self.side_offset])
            side_point[0] = current[0]
            self.waypoints = [side_point, self.goal + np.array([0., side*self.side_offset]), self.goal]

    def _set_phase(self, phase):
        self.env.phase = phase
        self.phase_history.append((float(self.env.runner.d.time), phase))

    def _body_command(self, direction):
        yaw = yaw_of(self.env)
        rot = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        body = rot @ (np.asarray(direction) * self.speed)
        return command_action([np.clip(body[0], -.4, .4), np.clip(body[1], -.35, .35), 0.])

    def action(self):
        env, r, s = self.env, self.env.runner, self.env.slot
        now = float(r.d.time)
        xy = r.d.xpos[s.body_id, :2].copy()
        if now < 1.2:
            self._set_phase('settle')
            return np.zeros(5)
        if np.linalg.norm(xy-self.goal) <= self.stop_radius:
            self._set_phase('dwell')
            return np.zeros(5)
        while self.waypoint < len(self.waypoints)-1 and np.linalg.norm(xy-self.waypoints[self.waypoint]) < .13:
            self.waypoint += 1
            self.pause_until = max(self.pause_until, now+.30)
            self.pulse_until = 0.
        target = self.waypoints[self.waypoint]
        delta = target-xy
        target_distance = float(np.linalg.norm(delta))
        if target_distance + .02 < self.last_target_distance:
            self.last_target_distance = target_distance
            self.last_progress_s = now
        elif self.last_target_distance == float("inf"):
            self.last_target_distance = target_distance
            self.last_progress_s = now
        if np.linalg.norm(delta) < .08:
            self._set_phase('brake')
            return np.zeros(5)
        direction = delta/max(np.linalg.norm(delta), 1e-9)
        # A one-pulse recovery is bounded and only used after a sustained
        # standstill, which handles the measured lateral dead zone.
        if now >= 2.2 and now-self.last_progress_s > .8 and np.linalg.norm(r.d.qvel[s.qvel_adr:s.qvel_adr+2]) < .08:
            self.kick_until = max(self.kick_until, now+.40)
            self.last_progress_s = now
        if now < self.kick_until:
            self._set_phase('recover')
            # The calibrated gait study showed a forward 0.30 m/s pulse is
            # the reliable way to restart a stalled lateral gait.
            return command_action([.30, 0., 0.])
        if now < self.pause_until:
            self._set_phase('brake')
            return np.zeros(5)
        if now >= self.pulse_until:
            self.pulse_until = now+self.pulse_s
            self.pause_until = self.pulse_until+self.pause_s
        if now < self.pulse_until:
            self._set_phase('translate')
            return self._body_command(direction)
        self._set_phase('brake')
        return np.zeros(5)


class MeasuredRouteController:
    """Route feasibility: measured speed bounds and a pause at cell transitions."""
    def __init__(self,env):
        self.env=env;self.waypoint=1;self.brake_until=1.2;self.released=False
        self.previous_open=tuple(env.state.open);self.previous_carry=False

    def action(self):
        env=self.env;r=env.runner;s=env.state;xy=r.d.xpos[env.slot.body_id,:2].copy()
        opened=tuple(s.open)
        if opened!=self.previous_open or s.carrying!=self.previous_carry:
            self.brake_until=float(r.d.time)+.4
        self.previous_open=opened;self.previous_carry=s.carrying
        if r.d.time<self.brake_until:env.phase='recover';return np.zeros(5)
        activate=acquire=0.;phase='advance';tolerance=.18
        if env.stage=='transport' and not s.acquired:
            limit=env.layout.object_index
            if self.waypoint<limit and np.linalg.norm(xy-env.layout.xy(env.layout.route[self.waypoint]))<.22:
                self.waypoint+=1;self.brake_until=float(r.d.time)+.4;env.phase='brake';return np.zeros(5)
            target=env.layout.xy(env.layout.route[self.waypoint])
            if self.waypoint>=limit:
                target=r.d.xpos[env.parcel_body,:2].copy();phase='acquire';tolerance=.08
                if np.linalg.norm(target-xy)<.40:acquire=2.
        else:
            if self.waypoint<len(env.layout.route)-1 and np.linalg.norm(xy-env.layout.xy(env.layout.route[self.waypoint]))<.22:
                self.waypoint+=1;self.brake_until=float(r.d.time)+.4;env.phase='brake';return np.zeros(5)
            target=env.layout.xy(env.layout.route[self.waypoint])
            for door,k in enumerate(env.layout.door_indices):
                if self.waypoint>=k and not s.open[door]:
                    center=env.layout.xy(env.layout.route[k])
                    if np.linalg.norm(xy-center)<.5:
                        target=env.layout.plate(door,env.layout.correct_sides[door]);activate=2.;phase='switch';tolerance=.08
                    else:target=center
                    self.waypoint=min(self.waypoint,k+1);break
            if env.stage=='transport' and self.waypoint==len(env.layout.route)-1 and all(s.crossed):
                goal=env.layout.xy(env.layout.route[-1]);mount=r.d.xmat[env.slot.body_id].reshape(3,3)@[.32,0,.48]
                target=goal-mount[:2];obj=r.d.xpos[env.parcel_body];phase='release';tolerance=.08
                if self.released or np.linalg.norm(obj[:2]-goal)<.18:
                    self.released=True;env.phase='release';return np.array([0.,0.,0.,0.,-2.])
        delta=target-xy;dist=np.linalg.norm(delta);velocity=delta/max(dist,1e-9)*.4
        if dist<tolerance:velocity[:]=0
        yaw=yaw_of(env);body=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])@velocity
        body[0]=np.clip(body[0],-.3,.3);body[1]=np.clip(body[1],-.3,.35)
        env.phase=phase
        action=command_action([body[0],body[1],np.clip(-1.2*yaw,-.35,.35)])
        action[3]=activate;action[4]=acquire
        return action


class RecoveryRouteController(MeasuredRouteController):
    """Reuse the measured Approach pulse once per stalled route waypoint/phase."""
    def __init__(self,env):
        super().__init__(env);self.kick_until=0.;self.recovered=set()

    def action(self):
        env=self.env;now=float(env.runner.d.time)
        if now<self.kick_until:
            env.phase='accelerate';return command_action([.3,0,0])
        action=super().action();key=(self.waypoint,env.phase)
        recent=[r for r in env.diagnostic_trace if now-1.2<=r['time_s']<=now]
        if (key not in self.recovered and env.phase in ('advance','switch','acquire')
                and len(recent)>=29 and recent[-1]['time_s']-recent[0]['time_s']>=1.1
                and all(r['command'][1]>.25 for r in recent)
                and np.linalg.norm(np.array(recent[-1]['xy'])-recent[0]['xy'])<.05):
            self.recovered.add(key);self.kick_until=now+.4
            env.phase='accelerate';return command_action([.3,0,0])
        return action


class PlateSafeRouteController(MeasuredRouteController):
    """Measured route controller with a contact-triggered plate brake.

    The plate remains a normal colliding, activating geom.  Once a physical
    button contact is observed, the controller holds standstill long enough
    for the measured gait to settle before leaving the plate.  Gate-opening
    transitions receive the same longer brake interval.
    """
    def __init__(self, env, contact_hold_s=1.0):
        super().__init__(env)
        self.contact_hold_s = float(contact_hold_s)
        self.contact_brake_until = 0.
        self.plate_contacts_seen = []

    def action(self):
        env = self.env
        now = float(env.runner.d.time)
        if now < self.contact_brake_until:
            env.phase = 'plate_brake'
            return np.zeros(5)
        before_open = tuple(env.state.open)
        action = super().action()
        current_contacts = set(getattr(env.runner, 'button_contacts', set()))
        new_contacts = current_contacts - set(self.plate_contacts_seen)
        if new_contacts:
            self.contact_brake_until = now + self.contact_hold_s
            self.plate_contacts_seen.extend(sorted(new_contacts))
            env.phase = 'plate_brake'
            return np.zeros(5)
        if tuple(env.state.open) != before_open:
            self.contact_brake_until = now + self.contact_hold_s
            env.phase = 'plate_brake'
            return np.zeros(5)
        # Reduce the entry command only in the final 45 cm before the target
        # plate.  The command stays above the measured gait onset threshold;
        # this changes approach timing while preserving activation semantics.
        if env.phase == 'switch':
            for door, correct in enumerate(env.layout.correct_sides):
                if env.state.open[door]:
                    continue
                plate = env.layout.plate(door, correct)
                xy = env.runner.d.xpos[env.slot.body_id, :2]
                if np.linalg.norm(xy-plate) < .45:
                    command = np.tanh(action[:3])*[.4, .35, .4]
                    norm = float(np.linalg.norm(command[:2]))
                    if norm > .30:
                        command[:2] *= .30/max(norm, 1e-9)
                    action[:3] = np.arctanh(np.clip(command/[.4, .35, .4], -.999999, .999999))
                    break
        return action
