"""Full BHL MuJoCo mission; high-level actions have no route supervisor."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import mujoco
import numpy as np
import onnxruntime as ort
from omegaconf import OmegaConf
from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from bhl_robust.eval.multi_robot import MultiRunner, _WORLDS, build_multi
from bhl_robust.mission.layout import generate, world_xml, SPLITS, STAGES
from bhl_robust.mission.sensors import MissionSensors, FRAME, HISTORY
from bhl_robust.mission.state import MissionState
from bhl_robust.sensor_io import imu_features


class CpuPolicy:
    def __init__(self, path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.key = self.session.get_inputs()[0].name

    def forward(self, obs):
        return self.session.run(None, {self.key: obs})[0]


class MissionRunner(MultiRunner):
    # Diagnostics may attach a list here to receive per-physics-step contact
    # records.  The default None path does no work and leaves the benchmark
    # dynamics unchanged.
    contact_trace = None

    def step(self, targets_per_robot):
        self.wall_contact = False
        self.button_contacts = set()
        slot = self.slots[0]
        force = np.zeros(6)
        for _ in range(self.substeps):
            if self.disabled_parcel:
                self.d.qpos[self.parcel_q:self.parcel_q+3] = [0, 0, -5]
                self.d.qvel[self.parcel_v:self.parcel_v+6] = 0
            if self.carrying:
                rotation = self.d.xmat[slot.body_id].reshape(3, 3)
                self.d.qpos[self.parcel_q:self.parcel_q+3] = self.d.xpos[slot.body_id] + rotation @ [.32, 0, .48]
                self.d.qpos[self.parcel_q+3:self.parcel_q+7] = [1, 0, 0, 0]
                self.d.qvel[self.parcel_v:self.parcel_v+6] = 0
            tau = self.kp*(targets_per_robot[0]-self.d.sensordata[slot.jpos_adr]) - self.kd*self.d.sensordata[slot.jvel_adr]
            self.d.ctrl[slot.ctrl] = np.clip(tau, -self.eff, self.eff)
            mujoco.mj_step(self.m, self.d)
            for idx in range(self.d.ncon):
                contact = self.d.contact[idx]
                a, b = int(contact.geom1), int(contact.geom2)
                if self.own[a] == self.own[b]:
                    continue
                world = b if self.own[a] else a
                self.wall_contact |= world in self.walls
                if world in self.plates:
                    mujoco.mj_contactForce(self.m, self.d, idx, force)
                    if force[0] >= 1.:
                        self.button_contacts.add(self.plates[world])
                if self.contact_trace is not None and (world in self.walls or world in self.plates):
                    body1 = int(self.m.geom_bodyid[a])
                    body2 = int(self.m.geom_bodyid[b])
                    world_body = int(self.m.geom_bodyid[world])
                    base_q = self.d.qpos[slot.qpos_adr+3:slot.qpos_adr+7]
                    base_yaw = float(np.arctan2(
                        2*(base_q[0]*base_q[3] + base_q[1]*base_q[2]),
                        1 - 2*(base_q[2]**2 + base_q[3]**2)))
                    self.contact_trace.append({
                        "time_s": float(self.d.time),
                        "geom1": mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, a) or "",
                        "geom2": mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, b) or "",
                        "world_geom": mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, world) or "",
                        "body1": mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, body1) or "",
                        "body2": mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, body2) or "",
                        "world_body": mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, world_body) or "",
                        "base_xy": self.d.xpos[slot.body_id, :2].astype(float).tolist(),
                        "base_yaw": base_yaw,
                        "distance_m": float(contact.dist),
                        "normal_force_N": float(force[0]) if world in self.plates else 0.,
                        "tangent_force_N": float(np.linalg.norm(force[1:3])) if world in self.plates else 0.,
                        "normal": [float(x) for x in contact.frame[:3]],
                        "position": [float(x) for x in contact.pos],
                        "friction_geom1": [float(x) for x in self.m.geom_friction[a]],
                        "friction_geom2": [float(x) for x in self.m.geom_friction[b]],
                        "solref": [float(x) for x in contact.solref],
                        "solimp": [float(x) for x in contact.solimp],
                        "margin_m": float(self.m.geom_margin[a]),
                    })


class MissionEnv:
    def __init__(self, repo, cache, *, arm="both", stage="approach", split="train", seed=0,
                 failure="normal", max_seconds=None):
        if stage not in STAGES or split not in SPLITS:
            raise ValueError("unknown curriculum stage or split")
        self.repo, self.cache = Path(repo), Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.upstream = self.repo / "external/Berkeley-Humanoid-Lite"
        deploy = self.upstream / "logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml"
        self.cfg = OmegaConf.load(deploy)
        self.deploy = deploy
        if (self.cfg.num_actions, self.cfg.num_observations) != (22, 75):
            raise ValueError("full-humanoid gait contract mismatch")
        self.gait = CpuPolicy(deploy.parent / "policy.onnx")
        self.arm, self.stage, self.split, self.seed, self.failure = arm, stage, split, seed, failure
        self.rng = np.random.default_rng(seed)
        self.max_seconds = max_seconds or {"approach": 18, "branches": 50}.get(stage, 180)
        self.dt = float(self.cfg.policy_dt)
        self.repeat = 5
        self.episode = 0
        self.history = np.zeros((HISTORY, FRAME), np.float32)

    def reset(self, layout_index=None):
        index = int(self.rng.integers(SPLITS[self.split][1])) if layout_index is None else layout_index
        self.layout = generate(self.split, index)
        _WORLDS["mission7"] = world_xml(self.layout)
        model, slots = build_multi(self.upstream, self.cache, 1, ["mission7"], variant="humanoid", world="mission7")
        controller = RlController(self.cfg)
        controller.policy = self.gait
        self.runner = MissionRunner(model, slots, [self.cfg], [controller])
        self.controller, self.slot, self.model = controller, slots[0], model
        r, s = self.runner, self.slot
        r.reset(self.rng)
        start = -2 if self.stage == "approach" else -5 if self.stage == "branches" else 0
        self.start_index = len(self.layout.route)+start if start < 0 else start
        spawn = self.layout.xy(self.layout.route[start])
        if self.stage == "approach":
            goal = self.layout.xy(self.layout.route[-1])
            spawn = goal + (spawn-goal)/self.layout.cell_m*.65
        r.d.qpos[s.qpos_adr:s.qpos_adr+2] = spawn + self.rng.normal(0, .015, 2)
        r.own = np.array([(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b)) or "").startswith(s.prefix)
                          for b in model.geom_bodyid])
        owners = np.where(r.own, 0, -1)
        r.walls, r.plates = set(), {}
        for g in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith(("wall_", "door_", "goal_post")):
                r.walls.add(g)
            if name.startswith("plate_"):
                _, door, side = name.split("_")
                r.plates[g] = (int(door), int(side) == self.layout.correct_sides[int(door)])
        self.gates = [int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"gate_{i}")]) for i in range(2)]
        self.parcel_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "parcel")
        joint = int(model.body_jntadr[self.parcel_body])
        r.parcel_q, r.parcel_v = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        self.parcel_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "parcel_geom")
        r.carrying = False
        r.disabled_parcel = self.stage != "transport"
        self.state = MissionState(self.stage)
        self.previous_action = np.zeros(5)
        self.touched = False
        self.visited = set()
        self.last_cell = None
        self.trace = []
        self.sensors = MissionSensors(model, slots, owners, self.seed+self.episode*1709, self.arm, self.failure)
        self.episode += 1
        self._gates()
        if self.stage != "transport":
            r.d.qpos[r.parcel_q:r.parcel_q+3] = [0, 0, -5]
            model.geom_contype[self.parcel_geom] = model.geom_conaffinity[self.parcel_geom] = 0
            model.geom_group[self.parcel_geom] = 5
        mujoco.mj_forward(model, r.d)
        self.sensors.capture(r.d, 0.)
        frame = self.observe_frame()
        self.history[:] = frame
        return self.history.ravel().copy()

    def _gates(self):
        for i, mid in enumerate(self.gates):
            opened = self.stage not in ("doors", "transport") or self.state.open[i]
            self.runner.d.mocap_pos[mid, 2] = 2. if opened else .55

    def observe_frame(self):
        r, s = self.runner, self.slot
        sd = r.d.sensordata
        q = sd[s.quat_adr:s.quat_adr+4]
        acc = self.sensors.accel_adrs[0]
        imu = imu_features(q[[1, 2, 3, 0]], sd[s.gyro_adr:s.gyro_adr+3], sd[acc:acc+3],
                           stamp_s=r.d.time, now_s=r.d.time)
        # Only gravity, gyro, accelerometer, encoders, previous action and locally
        # measurable contact/attachment bits. No yaw, translation, target or stage.
        proprio = np.r_[imu, sd[s.jpos_adr], sd[s.jvel_adr]*.1,
                       self.previous_action, float(self.state.carrying), float(self.touched)]
        return np.r_[proprio, *self.sensors.features(r.d.time)].astype(np.float32)

    def step(self, action):
        action = np.asarray(action, float)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("five finite high-level actions required")
        if self.state.completed_s is not None or self.state.failure is not None:
            raise RuntimeError("reset required after terminal state")
        # Smooth bounded action transformation; PPO stores the untransformed action.
        bounded = np.tanh(action)
        command = bounded[:3]*[.4, .35, .4]
        reward = 0.
        r, s, m = self.runner, self.slot, self.state
        for _ in range(self.repeat):
            before = r.d.xpos[s.body_id, :2].copy()
            if self.stage == "transport":
                obj = r.d.xpos[self.parcel_body]
                if bounded[4] > .5 and np.linalg.norm(obj[:2]-before) < .45 and not m.carrying:
                    reward += m.acquire()
                if bounded[4] < -.5:
                    # Counts as a drop unless released over the destination.
                    m.release(accidental=np.linalg.norm(obj[:2]-self.layout.xy(self.layout.route[-1])) > .29)
                if r.tilt(0) > .5:
                    m.release(accidental=True)
            r.carrying = m.carrying
            self.model.geom_contype[self.parcel_geom] = self.model.geom_conaffinity[self.parcel_geom] = int(self.stage == "transport" and not m.carrying)
            targets = self.controller.update(r.observe(0, command))
            if not np.isfinite(targets).all():
                raise FloatingPointError("nonfinite gait output")
            r.step([targets])
            if not np.isfinite(r.d.qpos).all() or not np.isfinite(r.d.qvel).all() or r.d.warning.number.sum():
                raise FloatingPointError("nonfinite or warned MuJoCo physics")
            after = r.d.xpos[s.body_id, :2].copy()
            cross = []
            for i in range(2):
                center, normal = self.layout.door(i)
                if np.dot(before-center, normal) <= 0 < np.dot(after-center, normal) and np.linalg.norm(after-center) < self.layout.cell_m/2:
                    cross.append(i)
            goal = self.layout.xy(self.layout.route[-1])
            obj = r.d.xpos[self.parcel_body]
            object_inside = bool(np.linalg.norm(obj[:2]-goal) <= .29 and .055 <= obj[2] <= .095)
            reward += m.update(now=float(r.d.time), dt=self.dt, upright=r.tilt(0) < .78,
                               collision=r.wall_contact, button_contacts=r.button_contacts,
                               activate=bounded[3] > .5, crossing=cross,
                               xy_delta=float(np.linalg.norm(after-before)),
                               goal_inside=np.linalg.norm(after-goal) < .36,
                               slow=np.linalg.norm(r.d.qvel[s.qvel_adr:s.qvel_adr+2]) < .2,
                               object_inside=object_inside,
                               object_slow=np.linalg.norm(r.d.qvel[r.parcel_v:r.parcel_v+6]) < .15)
            self.touched = bool(r.button_contacts)
            self._gates()
            self.sensors.capture(r.d, float(r.d.time))
            cell = tuple(np.floor(after/self.layout.cell_m+.5).astype(int))
            if cell != self.last_cell and cell in self.layout.adjacency:
                if len(self.layout.adjacency[cell]) == 1 and cell not in (self.layout.route[0], self.layout.route[-1]):
                    m.dead_end_entries += 1
                self.last_cell = cell
                self.visited.add(cell)
            if m.completed_s is not None or m.failure is not None:
                break
        self.previous_action = bounded
        self.history[:-1] = self.history[1:]
        self.history[-1] = self.observe_frame()
        timeout = r.d.time+1e-8 >= self.max_seconds and m.completed_s is None and m.failure is None
        if timeout:
            m.failure = "timeout"
        done = m.completed_s is not None or m.failure is not None
        if len(self.trace) == 0 or r.d.time-self.trace[-1]["time_s"] >= .99 or done:
            self.trace.append({"time_s": float(r.d.time), "xy": after.tolist(),
                               "action": bounded.tolist(), "gates": m.open.copy(),
                               "carrying": m.carrying, "parcel_xyz": obj.tolist(),
                               "lidar_valid": float(self.history[-1, 61+36:61+72].mean()),
                               "depth_valid": float(self.history[-1, 134+128:134+256].mean())})
        return self.history.ravel().copy(), reward, done, {"timeout": timeout}

    def metrics(self):
        m, r = self.state, self.runner
        shortest = .65 if self.stage == "approach" else (len(self.layout.route)-1-self.start_index)*self.layout.cell_m
        obj = r.d.xpos[self.parcel_body]
        state = asdict(m)
        state.pop("previous_buttons")
        return {**state, "success": m.completed_s is not None and m.failure is None,
                "layout": self.layout.metadata(), "arm": self.arm, "seed": self.seed,
                "sensor_failure": self.failure, "elapsed_s": float(r.d.time),
                "fall": m.failure == "fall", "timeout": m.failure == "timeout",
                "button_success": all(m.open), "door_success": all(m.crossed),
                "pickup_success": m.acquired, "placement_success": self.stage == "transport" and m.completed_s is not None,
                "wall_contact_intervals": m.collisions,
                "shortest_centerline_m": shortest,
                "path_efficiency": min(1., shortest/max(m.distance_m, 1e-9)) if m.completed_s is not None else None,
                "object_xyz": obj.tolist(), "placement_error_m": float(np.linalg.norm(obj[:2]-self.layout.xy(self.layout.route[-1]))),
                "trace": self.trace, "transport_kind": "kinematic_attachment_no_grasp_or_payload_dynamics"}
