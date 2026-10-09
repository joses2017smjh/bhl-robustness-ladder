#!/usr/bin/env python3
"""Actual FAST-LIO2 in the learned-gait navigation loop; no pose oracle.

The static external route and initial start registration are declared inputs.
Simulator poses enter ray/IMU generation and independent outcomes only.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from bhl_robust.research.native_lio import NativeLioClient, sha256
from bhl_robust.research.pose_metrics import transform

CASES = ("straight", "dogleg", "occluders")
ROUTES = {"straight": [[5., 0.]], "dogleg": [[2.2, 0.], [2.5, 2.8]],
          "occluders": [[1.5, -.25], [3., .25], [5., 0.]]}


class DelayedNativeState:
    """A computed response is invisible until its charged physics delay ends."""
    def __init__(self):
        self.available=None;self.raw=np.empty((0,3));self.available_at_s=None
        self.pending=None;self.history=[]

    def stage(self,packet,raw,step,wall_seconds,dt):
        if self.pending is not None:raise ValueError("native response already pending")
        if not math.isfinite(wall_seconds) or wall_seconds<0 or not math.isfinite(dt) or dt<=0:
            raise ValueError("finite native wall latency and positive gait period required")
        ticks=max(1,int(math.ceil(wall_seconds/dt)))
        arrival=step+ticks
        self.pending={"packet":packet,"raw":raw,"arrival_step":arrival,"arrival_s":arrival*dt,
                      "capture_s":packet["timestamp_s"],"request_s":step*dt,"wall_seconds":wall_seconds}
        return ticks

    def advance(self,step):
        if self.pending is None or step<self.pending["arrival_step"]:return False
        value=self.pending;self.pending=None
        self.available=value["packet"];self.raw=value["raw"];self.available_at_s=value["arrival_s"]
        self.history.append({k:v for k,v in value.items() if k not in ("packet","raw")})
        return True


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as f: json.dump(data, f, indent=2, allow_nan=False); f.write("\n")


def navigation_command(native, registration, t_body_imu, route, waypoint,
                       now, raw_points_b, *, maximum_age_s=.35):
    """Pure inference boundary: native pose, fixed calibration/route and rays."""
    zero = np.zeros(3)
    if native is None or native.get("tracked") is not True:
        return zero, waypoint, "native_untracked_stop", None
    if native.get("map_reset_id") != 0 or registration is None:
        return zero, waypoint, "native_unregistered_or_reset_stop", None
    age = now - float(native["timestamp_s"])
    if not math.isfinite(age) or age < -1e-8 or age > maximum_age_s:
        return zero, waypoint, "native_stale_stop", None
    pose = transform(registration) @ transform(native["T_W_I"]) @ np.linalg.inv(transform(t_body_imu))
    position = pose[:2, 3]
    route = np.asarray(route, dtype=float)
    if route.ndim != 2 or route.shape[1] != 2 or not np.isfinite(route).all(): raise ValueError("finite external waypoints required")
    while waypoint < len(route)-1 and np.linalg.norm(route[waypoint] - position) < .25: waypoint += 1
    delta = route[waypoint] - position
    if waypoint == len(route)-1 and np.linalg.norm(delta) < .25:
        return zero, waypoint, "estimated_goal_stop", pose
    points = np.asarray(raw_points_b, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 20 or not np.isfinite(points).all():
        return zero, waypoint, "lidar_unknown_stop", pose
    blocked = ((points[:,0] > .10) & (points[:,0] < .45) & (np.abs(points[:,1]) < .18)
               & (points[:,2] > .10) & (points[:,2] < .70))
    if blocked.any(): return zero, waypoint, "raw_obstacle_stop", pose
    yaw = math.atan2(pose[1,0], pose[0,0])
    angle = (math.atan2(delta[1],delta[0])-yaw+math.pi)%(2*math.pi)-math.pi
    speed = .30 * max(0., math.cos(angle)) if abs(angle) < .75 else 0.
    return np.array([speed, 0., np.clip(1.2*angle, -.45, .45)]), waypoint, "estimated_route", pose


def world_xml(case, group):
    """Fixed external route fixtures; geometry never enters command inference."""
    if case not in CASES: raise ValueError("unknown predeclared navigation case")
    from bhl_robust.eval.multi_robot import _FLAT_WORLD
    import xml.etree.ElementTree as ET
    root = ET.fromstring(_FLAT_WORLD)
    world = root.find("worldbody")
    rng = np.random.default_rng(group)
    shade = rng.uniform(.4,.7)
    walls = []
    if case in ("straight", "occluders"):
        walls = [(2.5,-.775,3.0,.025),(2.5,.775,3.0,.025),(-.525,0,.025,.8),(5.525,0,.025,.8)]
    else:
        walls = [(1.,-.775,1.55,.025),(.65,.775,1.2,.025),(-.525,0,.025,.8),
                 (3.275,1.4,.025,2.2),(1.725,2.2,.025,1.45),(2.5,3.575,.8,.025)]
    for index,(x,y,sx,sy) in enumerate(walls):
        ET.SubElement(world,"geom",name=f"wall_native_{index}",type="box",pos=f"{x} {y} .5",size=f"{sx} {sy} .5",
                      group="0",rgba=f"{shade} .55 .65 1",material="groundplane")
    # Asymmetric surfaces provide genuine scan-to-map geometry beyond two
    # parallel walls. Their fixed locations are not passed to the estimator.
    for index,(x,y) in enumerate(((1.,-.62),(2.7,.62),(4.,-.62))):
        if case == "dogleg" and index>0: continue
        ET.SubElement(world,"geom",name=f"wall_feature_{index}",type="box",pos=f"{x} {y} .25",size=".10 .05 .25",group="0",rgba=".2 .4 .7 1")
    if case=="occluders":
        for index,(x,y) in enumerate(((2.,.50),(3.5,-.50))):
            ET.SubElement(world,"geom",name=f"wall_occluder_{index}",type="box",pos=f"{x} {y} .2",size=".12 .18 .2",group="0",rgba=".7 .3 .2 1")
    return ET.tostring(root,encoding="unicode")


def fixed_site_transform(model, base_body, site):
    """Static base-to-site calibration, including fixed compiled child bodies."""
    def local(position, quaternion):
        w,x,y,z=np.asarray(quaternion,dtype=float)
        rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                           [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                           [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        value=np.eye(4);value[:3,:3]=rotation;value[:3,3]=position
        return transform(value)
    value=local(model.site_pos[site],model.site_quat[site])
    body=int(model.site_bodyid[site]);visited=set()
    while body!=base_body:
        if body<=0 or body in visited or int(model.body_jntnum[body])!=0:
            raise ValueError("IMU must have a fixed body chain to the robot base")
        if hasattr(model,"body_mocapid") and int(model.body_mocapid[body])>=0:
            raise ValueError("mocap IMU body is not a static calibration")
        visited.add(body)
        value=local(model.body_pos[body],model.body_quat[body])@value
        body=int(model.body_parentid[body])
    return transform(value)


class CausalSensors:
    """Physics-substep IMU and moving-rig timed rays; no retrospective poses."""
    def __init__(self, runner, slot, owners, *, start_s=2., rate_hz=5.):
        import mujoco
        self.runner,self.slot,self.owners=runner,slot,owners
        self.start_s,self.period=start_s,1/rate_hz
        dt=float(runner.m.opt.timestep)
        if not math.isclose(round(.005/dt)*dt,.005,rel_tol=0,abs_tol=1e-9):
            raise ValueError("200Hz IMU requires a physics timestep dividing5ms")
        self.imu_every=max(1,int(round(.005/dt)));self.substep=0;self.physics_dt=dt
        sid=mujoco.mj_name2id(runner.m,mujoco.mjtObj.mjOBJ_SENSOR,slot.prefix+"imu_acc")
        if sid<0: raise ValueError("actual MuJoCo accelerometer required")
        self.acc_adr=int(runner.m.sensor_adr[sid]);site=int(runner.m.sensor_objid[sid])
        self.t_body_imu=fixed_site_transform(runner.m,slot.body_id,site)
        self.t_body_lidar=np.eye(4);self.t_body_lidar[:3,3]=[.10,0,.38]
        self.t_imu_lidar=np.linalg.inv(self.t_body_imu)@self.t_body_lidar
        self.imu=[];self.imu_cursor=0;self.ready=deque();self.recorded=[]
        self.cycle=0;self.column=0;self.active_points=[];self.active_times=[];self.active_rings=[]
        self.azimuth=np.deg2rad(np.linspace(-80,80,128));self.elevation=np.deg2rad(np.linspace(-35,25,16))
        # Group0 contains world geoms. All robot geoms are additionally excluded
        # by assigning them group1 in the sensor model only; contacts unchanged.
        runner.m.geom_group[np.asarray(owners)>=0]=1
        runner.substep_hook=self.on_substep

    def on_substep(self,data):
        import mujoco
        # mj_step computes xmat/xpos/sensordata before integrating qpos and
        # incrementing time. The retained sensor state therefore belongs to
        # the preceding physics timestamp, not the incremented clock.
        self.substep+=1;t=float(data.time)-self.physics_dt
        if t<self.start_s-1e-9:return
        if self.substep%self.imu_every==0:
            self.imu.append((t,data.sensordata[self.slot.gyro_adr:self.slot.gyro_adr+3].copy(),
                             data.sensordata[self.acc_adr:self.acc_adr+3].copy()))
        beginning=self.start_s+(self.cycle+1)*self.period-.05
        end=self.start_s+(self.cycle+1)*self.period
        if t<beginning-1e-9:return
        target=min(128,int(math.floor((t-beginning)/.05*128+1e-7)))
        rotation=np.asarray(data.xmat[self.slot.body_id]).reshape(3,3)
        origin=data.xpos[self.slot.body_id]+rotation@self.t_body_lidar[:3,3]
        geom=np.zeros(1,np.int32)
        for col in range(self.column,target):
            az=self.azimuth[col]
            for ring,el in enumerate(self.elevation):
                ray=np.array([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)])
                distance=mujoco.mj_ray(self.runner.m,data,origin,rotation@ray,np.array([1,0,0,0,0,0],np.uint8),1,-1,geom)
                if .1<distance<=8.:
                    self.active_points.append((ray*distance).astype(np.float32));self.active_times.append(t);self.active_rings.append(ring)
        self.column=target
        if t>=end-1e-9:
            if len(self.active_points)>1:
                scan={"points_xyz_m":np.asarray(self.active_points),"point_time_s":np.asarray(self.active_times),
                      "ring_index":np.asarray(self.active_rings,np.uint16),"frame_timestamp_s":float(max(self.active_times)),"origin":"simulation"}
                self.ready.append(scan);self.recorded.append(scan)
            self.cycle+=1;self.column=0;self.active_points=[];self.active_times=[];self.active_rings=[]

    def newest(self):
        if not self.ready:return None
        scan=self.ready[-1];dropped=len(self.ready)-1;self.ready.clear()
        tail=scan["frame_timestamp_s"];end=self.imu_cursor
        while end<len(self.imu) and self.imu[end][0]<=tail+1e-9:end+=1
        selected=self.imu[self.imu_cursor:end];self.imu_cursor=end
        if not selected:raise ValueError("causal scan has no actual IMU coverage")
        imu={"timestamp_s":np.asarray([x[0] for x in selected]),"gyro_rad_s":np.asarray([x[1] for x in selected]),
             "specific_force_m_s2":np.asarray([x[2] for x in selected])}
        return scan,imu,dropped

    def save(self,path):
        counts=[len(s["points_xyz_m"]) for s in self.recorded]
        np.savez_compressed(path,points_xyz_m=np.concatenate([s["points_xyz_m"] for s in self.recorded]) if counts else np.empty((0,3)),
            point_time_s=np.concatenate([s["point_time_s"] for s in self.recorded]) if counts else np.empty(0),
            ring_index=np.concatenate([s["ring_index"] for s in self.recorded]) if counts else np.empty(0,np.uint16),
            scan_offsets=np.r_[0,np.cumsum(counts)],scan_tail_s=np.asarray([s["frame_timestamp_s"] for s in self.recorded]),
            imu_timestamp_s=np.asarray([x[0] for x in self.imu]),gyro_rad_s=np.asarray([x[1] for x in self.imu]),
            specific_force_m_s2=np.asarray([x[2] for x in self.imu]))


def episode(upstream,actor,runtime,case,seed,output,*,seconds=40.,
            client_factory=NativeLioClient,world_factory=world_xml,
            build_factory=None,sensor_factory=CausalSensors,
            method="fast_lio2_lidar_imu",scope=None):
    import mujoco
    from bhl_robust.eval.multi_robot import _WORLDS,build_multi
    from team_airlock import ContactRunner
    from team_airlock import CpuPolicy
    from omegaconf import OmegaConf
    from bhl_robust.eval.gait_clock import make_controller
    cfg=OmegaConf.load(actor["deploy"]);cfg.policy_checkpoint_path=actor["checkpoint"]
    controller=make_controller(cfg);controller.policy=CpuPolicy(actor["checkpoint"])
    output.mkdir(parents=True,exist_ok=False)
    _WORLDS["native_navigation"]=world_factory(case,seed)
    cache=Path(tempfile.mkdtemp(prefix="bhl-native-nav-assets-"))
    build_factory=build_factory or build_multi
    model,slots=build_factory(upstream,cache,1,[method],variant="biped",world="native_navigation")
    (output/"world.xml").write_text(_WORLDS["native_navigation"])
    slot=slots[0];runner=ContactRunner(model,slots,[cfg],[controller]);runner.reset(np.random.default_rng(seed))
    owners=np.array([0 if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])) or "").startswith(slot.prefix) else -1 for g in range(model.ngeom)])
    runner.configure_contacts(owners);sensors=sensor_factory(runner,slot,owners)
    policy_dt=float(cfg.policy_dt)
    known_start=np.eye(4);known_start[:3,3]=model.qpos0[slot.qpos_adr:slot.qpos_adr+3]
    registration=None;delivery=DelayedNativeState();waypoint=0;command=np.zeros(3)
    traces=[];native_records=[];client_latencies=[];additional_sensor_latencies=[];dropped_scans=0;goals=False;fell=collision=nonfinite=False
    latency_steps=0;goal_time=None;truth_imu=[];last_consumed_s=None
    initial_z=float(runner.d.qpos[slot.qpos_adr+2])
    maximum_steps=int(round(seconds/policy_dt))
    with client_factory(runtime,sensors.t_imu_lidar,output/"native",max_frames=int(seconds*5)+10) as client:
        for step in range(maximum_steps):
            now=float(runner.d.time)
            arrived=delivery.advance(step)
            latest=delivery.available
            if arrived and latest["tracked"] and registration is None:
                registration=known_start@sensors.t_body_imu@np.linalg.inv(transform(latest["T_W_I"]))
            if delivery.pending is not None:
                reason="measured_native_latency_hold";latency_steps+=1
            else:
                packet=None if arrived else sensors.newest()
                if packet is not None:
                    scan,imu,dropped=packet;dropped_scans+=dropped
                    started=time.monotonic();computed=client.track(scan,imu);elapsed=time.monotonic()-started
                    client_latencies.append(elapsed);native_records.append(computed)
                    extra=float(computed.get("additional_sensor_compute_seconds",0.))
                    if not math.isfinite(extra) or extra<0:raise ValueError("additional sensor compute latency must be finite and nonnegative")
                    additional_sensor_latencies.append(extra)
                    body_points=scan["points_xyz_m"]+sensors.t_body_lidar[:3,3]
                    delivery.stage(computed,body_points,step,elapsed+extra,policy_dt)
                    reason="measured_native_latency_hold";latency_steps+=1
                else:
                    command,waypoint,reason,_=navigation_command(latest,registration,sensors.t_body_imu,ROUTES[case],waypoint,now,delivery.raw)
                    last_consumed_s=now if latest is not None else None
            if now<5.:command=np.zeros(3);reason="declared_stationary_initialization"
            # Loss/reset can affect commands only after the response arrives.
            # The older available pose may independently become stale during
            # computation, which correctly stops the held command.
            if latest is None or not latest["tracked"] or latest.get("map_reset_id")!=0: command=np.zeros(3)
            if latest is not None and now-latest["timestamp_s"]>.35:command=np.zeros(3);reason="native_stale_stop"
            target=controller.update(runner.observe(0,command))
            if not np.isfinite(target).all():nonfinite=True;break
            runner.step([target])
            nonfinite=not np.isfinite(runner.d.qpos).all() or not np.isfinite(runner.d.qvel).all()
            if nonfinite:break
            # All simulator truth below is evaluator-only and never fed back.
            position=runner.d.qpos[slot.qpos_adr:slot.qpos_adr+3].copy();tilt=runner.tilt(0)
            collision=collision or runner.hit_wall;fell=tilt>=.78 or position[2]<initial_z-.25
            distance=float(np.linalg.norm(position[:2]-np.asarray(ROUTES[case][-1])))
            if distance<=.30 and not goals:goals=True;goal_time=float(runner.d.time)
            true_imu=np.eye(4);rotation=np.empty(9)
            mujoco.mju_quat2Mat(rotation,runner.d.qpos[slot.qpos_adr+3:slot.qpos_adr+7])
            true_imu[:3,:3]=rotation.reshape(3,3);true_imu[:3,3]=position
            truth_imu.append((float(runner.d.time),true_imu@sensors.t_body_imu))
            traces.append({"timestamp_s":float(runner.d.time),"command":command.tolist(),"reason":reason,
                "waypoint_index":waypoint,"native_timestamp_s":latest["timestamp_s"] if latest else None,
                "native_available_at_s":delivery.available_at_s,"native_last_consumed_at_s":last_consumed_s,
                "pending_capture_s":delivery.pending["capture_s"] if delivery.pending else None,
                "pending_arrival_s":delivery.pending["arrival_s"] if delivery.pending else None,
                "native_tracked":latest["tracked"] if latest else False,"evaluator_xy":position[:2].tolist(),
                "evaluator_goal_distance_m":distance,"evaluator_tilt_rad":tilt,"evaluator_contact":bool(runner.hit_wall)})
            if fell or collision:break
    sensors.save(output/"inference_sensors.npz")
    np.savez_compressed(output/"evaluator_truth.npz",timestamp_s=np.asarray([x[0] for x in truth_imu]),T_W_I=np.asarray([x[1] for x in truth_imu]))
    actual_seconds=float(runner.d.time)
    result={"status":"PASS","scientific_status":"MEASURED_DEVELOPMENT_EPISODE","method":method,
        "case":case,"seed":seed,"success":bool(goals and not(fell or collision or nonfinite) and actual_seconds>=seconds-1e-6),
        "goal_reached_evaluator":goals,"first_goal_time_s":goal_time,"fell":bool(fell),"collision":bool(collision),"nonfinite":bool(nonfinite),
        "elapsed_simulation_s":actual_seconds,"full_horizon":actual_seconds>=seconds-1e-6,"controller_goal_stop":"estimated_pose_only",
        "native_frames":len(native_records),"native_tracked_frames":sum(r["tracked"] for r in native_records),
        "native_states":{state:sum(r["state"]==state for r in native_records) for state in sorted({r["state"] for r in native_records})},
        "client_wall_p95_ms":float(np.quantile(client_latencies,.95)*1000) if client_latencies else None,
        "additional_sensor_compute_total_s":float(sum(additional_sensor_latencies)),
        "compute_latency_policy_steps":latency_steps,"latency_quantization_s":policy_dt,"dropped_scans":dropped_scans,
        "latency_semantics":"ceil((measured client.track wallseconds + declared additional sensor capture/encoding cost)/policy_dt) physics/gait updates holding previous command and old available pose; native response and loss/reset status become visible only at arrival; old-pose staleness independently stops commands",
        "response_deliveries":delivery.history,"responses_still_pending_at_horizon":int(delivery.pending is not None),
        "initial_registration":registration.tolist() if registration is not None else None,"known_start_pose":known_start.tolist(),
        "T_body_imu":sensors.t_body_imu.tolist(),"T_body_lidar":sensors.t_body_lidar.tolist(),"route_external":ROUTES[case],
        "trace":traces,"inference_sensors_sha256":sha256(output/"inference_sensors.npz"),"evaluator_truth_sha256":sha256(output/"evaluator_truth.npz"),
        "stream_receipt_sha256":sha256(output/"native/stream_receipt.json"),"ground_truth_inputs_to_estimator_or_planner":[],
        "scope":scope or "LIO-only learned12DoF gait simulation; fixed prescribed route and start station; ideal timed ray lidar and native MuJoCo IMU; no physical validation or NavGym actor claim"}
    write_json(output/"episode.json",result)
    import shutil
    shutil.rmtree(cache)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream",type=Path,default=Path(os.environ.get("UPSTREAM","external/Berkeley-Humanoid-Lite")))
    p.add_argument("--deploy",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True)
    runtime_group=p.add_mutually_exclusive_group(required=True)
    runtime_group.add_argument("--runtime",type=Path)
    runtime_group.add_argument("--runtime-archive",type=Path)
    p.add_argument("--runtime-archive-sha256")
    p.add_argument("--phase",choices=["smoke","development"],required=True)
    p.add_argument("--output",type=Path,default=Path(os.environ.get("H34_OUTPUT_DIR","output")))
    a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=True)
    if a.runtime_archive:
        if not a.runtime_archive_sha256 or sha256(a.runtime_archive)!=a.runtime_archive_sha256:
            raise ValueError("frozen runtime archive checksum required")
        spec=importlib.util.spec_from_file_location("native_navigation_unpack",ROOT/"scripts/bench/h34_campaign.py")
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        unpack=Path(os.environ["H34_WORK_DIR"] if "H34_WORK_DIR" in os.environ else tempfile.mkdtemp(prefix="bhl-native-nav-runtime-"))/"lio-runtime"
        unpack.mkdir(parents=True,exist_ok=False);launcher.safe_extract(a.runtime_archive,unpack)
        a.runtime=unpack/"runtime"
        runtime_receipt=json.loads((a.runtime/"runtime.json").read_text())
        binary=a.runtime/"fastlio_headless"
        if sha256(binary)!=runtime_receipt["binary_sha256"]:raise ValueError("unpacked native binary hash differs")
        binary.chmod(0o700)
    actor={"name":"frozen_dr_default","deploy":str(a.deploy.resolve()),"checkpoint":str(a.checkpoint.resolve())}
    inventory={str(path.resolve()):sha256(path) for path in [a.deploy,a.checkpoint,a.runtime/"runtime.json",a.runtime/"fastlio_headless"]}
    cases=("straight",) if a.phase=="smoke" else CASES
    seeds=(370000,370001) if a.phase=="smoke" else (380000,380001,380002)
    rows=[];started=time.monotonic()
    for case in cases:
        for seed in seeds:
            row=episode(a.upstream.resolve(),actor,a.runtime.resolve(),case,seed,a.output/f"{case}-s{seed}",seconds=8. if a.phase=="smoke" else 40.)
            rows.append(row);print(json.dumps({k:row[k] for k in("case","seed","success","native_tracked_frames","fell","collision")}),flush=True)
    result={"status":"PASS","scientific_status":"SMOKE_ONLY" if a.phase=="smoke" else "DEVELOPMENT_ONLY_NO_CONFIRMATION",
        "route":"native_estimated_pose_navigation","method":"fast_lio2_lidar_imu","phase":a.phase,"episodes":len(rows),
        "goals":sum(r["success"] for r in rows),"falls":sum(r["fell"] for r in rows),"collisions":sum(r["collision"] for r in rows),
        "native_tracked_frames":sum(r["native_tracked_frames"] for r in rows),"input_sha256":inventory,
        "gate_to_confirmation":bool(a.phase=="development" and len(rows)==9 and all(r["success"] for r in rows)),
        "wall_seconds":time.monotonic()-started,"orb_slam3_navigation":"NOT_RUN_IN_THIS_LIO_ONLY_ROUTE",
        "ground_truth_inputs_to_estimator_or_planner":[],"episode_files":[f"{r['case']}-s{r['seed']}/episode.json" for r in rows]}
    for name,expected in inventory.items():
        if sha256(name)!=expected:raise ValueError("native navigation input mutated during run")
    if a.phase=="smoke" and not all(r["native_tracked_frames"]>0 and not r["nonfinite"] for r in rows):result["status"]="INCOMPLETE"
    if a.phase=="development" and not result["gate_to_confirmation"]:result["status"]="NEGATIVE"
    write_json(a.output/"campaign_result.json",result)
    print(json.dumps(result),flush=True)
    return 0 if result["status"] in ("PASS","NEGATIVE") else 1


if __name__=="__main__":raise SystemExit(main())
