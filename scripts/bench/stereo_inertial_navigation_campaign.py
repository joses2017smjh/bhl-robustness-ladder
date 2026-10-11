#!/usr/bin/env python3
"""Separate native stereo / stereo–IMU navigation comparison with initialization gates."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from bhl_robust.research.native_orb import NativeOrbClient, TRACKING_STATES, sha256, tracking_summary
from bhl_robust.research.native_orb_inertial import write_stereo_inertial_settings
from bhl_robust.research.pose_metrics import transform
import native_stereo_navigation_campaign as STEREO

NAV = STEREO.NAV
ARMS = ("stereo", "stereo_inertial")


def inertial_navigation_packet(raw, t_imu_lidar, *, map_changed):
    """Only native inertially initialized poses can command this navigation arm."""
    initialized = raw["inertial"]["map_imu_initialized"]
    accepted = bool(raw["tracked"] and initialized and raw["map_id"] is not None and not map_changed)
    state = ("MAP_CHANGED_STOP" if map_changed else
             "IMU_INITIALIZING_STOP" if not initialized else
             "MAP_ID_UNAVAILABLE_STOP" if raw["tracked"] and raw["map_id"] is None else
             TRACKING_STATES[raw["tracking_state"]])
    return {"timestamp_s": raw["timestamp_s"], "tracked": accepted,
            "T_W_I": STEREO.native_camera_to_imu(raw["T_W_C"], t_imu_lidar).tolist() if accepted else None,
            "state": state, "map_reset_id": int(map_changed), "compute_seconds": raw["compute_seconds"],
            "native_ORB_T_W_C": raw["T_W_C"], "native_ORB_map_id": raw["map_id"],
            "native_ORB_tracking_state": raw["tracking_state"], "native_inertial": raw["inertial"]}


class InertialOrbNavigationClient(STEREO.OrbNavigationClient):
    """Actual body-attached cameras and MuJoCo sensor IMU, without evaluator poses."""
    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        t_b_l = np.eye(4); t_b_l[:3, 3] = [.10, 0., .38]
        t_b_i = t_b_l @ np.linalg.inv(self.t_imu_lidar)
        t_i_c = np.linalg.inv(t_b_i) @ STEREO.body_camera_transform(1)
        focal = STEREO.HEIGHT / 2 / math.tan(math.radians(STEREO.VFOV_DEG / 2))
        self.configuration = write_stereo_inertial_settings(self.directory / "stereo.yaml", t_imu_camera=t_i_c,
            fx=focal, fy=focal, cx=(STEREO.WIDTH-1)/2, cy=(STEREO.HEIGHT-1)/2,
            width=STEREO.WIDTH, height=STEREO.HEIGHT, fps=STEREO.CAMERA_HZ, baseline_m=STEREO.BASELINE)
        environment = dict(os.environ, LD_LIBRARY_PATH=str(self.runtime / "lib"), OMP_NUM_THREADS="1")
        self.client = NativeOrbClient(self.runtime / "bin/orb_native", self.runtime / "ORBvoc.txt",
            self.directory / "stereo.yaml", self.directory / "process", environment=environment, sensor_mode="stereo_inertial")
        self.client.__enter__()
        return self

    def track(self, scan, imu):
        if len(self.frames) >= self.maximum: raise ValueError("Bounded native inertial frame count exceeded")
        original = scan["stereo"]
        stamps = np.asarray(imu["timestamp_s"])
        gyro, acceleration = np.asarray(imu["gyro_rad_s"]), np.asarray(imu["specific_force_m_s2"])
        if stamps.ndim != 1 or gyro.shape != (len(stamps), 3) or acceleration.shape != gyro.shape:
            raise ValueError("Original causal IMU packet shapes differ")
        batch = np.column_stack((stamps, gyro, acceleration))
        raw = self.client.track(original["timestamp_s"], original["left"], original["right"], imu_samples=batch)
        if raw["tracked"] and raw["inertial"]["map_imu_initialized"] and raw["map_id"] is not None:
            if self.original_map is None: self.original_map = raw["map_id"]
            elif raw["map_id"] != self.original_map: self.map_changed = True
        packet = inertial_navigation_packet(raw, self.t_imu_lidar, map_changed=self.map_changed)
        packet.update(additional_sensor_compute_seconds=float(scan.get("additional_sensor_compute_seconds", 0.)),
                      original_left_sha256=original["left_sha256"], original_right_sha256=original["right_sha256"])
        self.frames.append(packet)
        return packet

    def __exit__(self, kind, value, traceback):
        try:
            return self.client.__exit__(kind, value, traceback)
        finally:
            with (self.directory / "native_frames.jsonl").open("x") as stream:
                for row in self.frames: stream.write(json.dumps(row, allow_nan=False) + "\n")
            initialized = [r for r in self.client.responses if r["inertial"]["map_imu_initialized"]]
            NAV.write_json(self.directory / "stream_receipt.json", {
                "schema": "bhl-native-orb-inertial-navigation-stream-v1",
                "native_runtime_sha256": sha256(self.runtime / "runtime.json"),
                "binary_sha256": sha256(self.runtime / "bin/orb_native"),
                "vocabulary_sha256": sha256(self.runtime / "ORBvoc.txt"),
                "settings_sha256": sha256(self.directory / "stereo.yaml"), "configuration": self.configuration,
                "frames": len(self.frames), "tracking": tracking_summary(self.client.responses),
                "native_imu_initialized_frames": len(initialized),
                "first_native_imu_initialized_timestamp_s": initialized[0]["timestamp_s"] if initialized else None,
                "navigation_accepted_frames": sum(r["tracked"] for r in self.frames),
                "first_map_id": self.original_map, "map_changed": self.map_changed,
                "T_I_L": self.t_imu_lidar.tolist(), "T_B_C_left": STEREO.body_camera_transform(1).tolist(),
                "estimator_inputs": "original stereo PNGs and timestamped 200 Hz MuJoCo gyro/specific-force samples",
                "ground_truth_inputs": [], "no_lidar_to_stereo_inertial_estimator": True,
                "planner_sensor_scope": "native inertially initialized camera pose plus raw LiDAR obstacle brake",
                "preinertial_pose_policy": "stop; visual OK before native map IMU initialization is not accepted"})


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("smoke", "development"), required=True)
    for name in ("upstream", "deploy", "checkpoint", "runtime-archive", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--runtime-archive-sha256", required=True)
    parser.add_argument("--runtime-archive-url", help="Optional frozen public GitHub release asset; downloaded only to a new worker-local path")
    parser.add_argument("--qualified-smoke", type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"schema": "bhl-stereo-inertial-navigation-campaign-v1", "status": "INCOMPLETE", "phase": args.phase,
              "episodes": 0, "problems": [], "ground_truth_inputs_to_estimator_or_planner": []}
    try:
        if args.runtime_archive_url:
            from urllib.request import urlopen
            from urllib.parse import urlparse
            url = urlparse(args.runtime_archive_url)
            if (url.scheme != "https" or url.netloc != "github.com" or
                    not url.path.startswith("/joses2017smjh/bhl-robustness-ladder/releases/download/")):
                raise ValueError("Runtime URL must be a frozen project GitHub release asset")
            if args.runtime_archive.exists(): raise ValueError("Downloaded runtime destination must be new")
            args.runtime_archive.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(args.runtime_archive_url, timeout=60) as source, args.runtime_archive.open("xb") as target:
                size = 0
                while True:
                    block = source.read(1024*1024)
                    if not block: break
                    size += len(block)
                    if size > 256*1024**2: raise ValueError("Native runtime exceeds the predeclared 256 MiB download bound")
                    target.write(block)
            if sha256(args.runtime_archive) != args.runtime_archive_sha256:
                raise ValueError("Downloaded runtime SHA-256 differs from the frozen runtime pin")
        own_sources = sorted({p for subtree in (ROOT / "src", ROOT / "scripts/bench", ROOT / "scripts/native")
                              for p in subtree.rglob("*.py") if p.is_file()})
        source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in own_sources}
        input_hashes = {key: sha256(getattr(args, key)) for key in ("deploy", "checkpoint", "runtime_archive")}
        if args.phase == "development":
            if args.qualified_smoke is None: raise ValueError("A matching actual inertial-initialization smoke is required")
            smoke = json.loads(args.qualified_smoke.read_text())
            if (smoke.get("status") != "PASS" or smoke.get("phase") != "smoke" or smoke.get("gate_to_development") is not True
                    or smoke.get("source_sha256") != source_hashes or smoke.get("input_sha256") != input_hashes):
                raise ValueError("Stereo–IMU development requires matching source/input and actual initialized navigation smoke")
        work = Path(os.getenv("H34_WORK_DIR", tempfile.mkdtemp(prefix="bhl-orb-inertial-nav-")))
        runtime = STEREO.unpack_runtime(args.runtime_archive, args.runtime_archive_sha256, work / "orb")
        actor = {"name": "frozen_dr_default_s0", "deploy": str(args.deploy.resolve()), "checkpoint": str(args.checkpoint.resolve())}
        seeds = (570000, 570001) if args.phase == "smoke" else (580000, 580001, 580002)
        cases = ("straight",) if args.phase == "smoke" else NAV.CASES
        rows = []
        for case in cases:
            for seed in seeds:
                for arm in ARMS:
                    output = args.output / f"{arm}-{case}-s{seed}"
                    sensors_created = []
                    def sensor_factory(runner, slot, owners):
                        sensor = STEREO.StereoSensors(runner, slot, owners, output=output / "stereo")
                        sensors_created.append(sensor); return sensor
                    try:
                        row = NAV.episode(args.upstream.resolve(), actor, runtime, case, seed, output,
                            seconds=20. if args.phase == "smoke" else 40.,
                            client_factory=STEREO.OrbNavigationClient if arm == "stereo" else InertialOrbNavigationClient,
                            world_factory=lambda c, s: STEREO.textured_world(c, s, output / "visual_assets"),
                            build_factory=STEREO.build_stereo_model, sensor_factory=sensor_factory,
                            method=arm, scope="Matched native stereo versus native stereo–IMU, shared fixed cameras and raw LiDAR obstacle brake; inertial arm stops until actual IMU initialization; simulated frozen gait")
                    finally:
                        for sensor in sensors_created: sensor.close()
                    native = json.loads((output / "native/stream_receipt.json").read_text())
                    row["native_imu_initialized_frames"] = native.get("native_imu_initialized_frames", 0)
                    row["episode_file"] = f"{arm}-{case}-s{seed}/episode.json"
                    rows.append(row)
                    print(json.dumps({k: row[k] for k in ("method", "seed", "success", "native_tracked_frames", "native_imu_initialized_frames")}), flush=True)
        for key, checksum in input_hashes.items():
            if sha256(getattr(args, key)) != checksum: raise ValueError("Frozen navigation input changed")
        gate = all(r["native_tracked_frames"] > 0 and not r["nonfinite"] and not r["fell"] and not r["collision"] and r["full_horizon"]
                   and (r["method"] != "stereo_inertial" or r["native_imu_initialized_frames"] > 0) for r in rows)
        result.update(status="PASS" if gate else "NEGATIVE", source_sha256=source_hashes, input_sha256=input_hashes,
            gate_to_development=gate if args.phase == "smoke" else None,
            scientific_status="ACTUAL_NATIVE_INITIALIZATION_SMOKE" if args.phase == "smoke" else "DEVELOPMENT_NO_CONFIRMATION",
            episodes=len(rows), episode_files=[r["episode_file"] for r in rows],
            arms={arm: {"episodes": sum(r["method"] == arm for r in rows),
                        "goals": sum(r["success"] for r in rows if r["method"] == arm),
                        "falls": sum(r["fell"] for r in rows if r["method"] == arm),
                        "contacts": sum(r["collision"] for r in rows if r["method"] == arm),
                        "native_imu_initialized_frames": sum(r["native_imu_initialized_frames"] for r in rows if r["method"] == arm)} for arm in ARMS},
            next_gate="Full paired development is eligible only if both fresh-seed inertial smoke episodes actually initialize; no automatic parameter tuning")
    except Exception as error:
        result["problems"].append(type(error).__name__ + ": " + str(error))
    NAV.write_json(args.output / "campaign_result.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] in ("PASS", "NEGATIVE") else 1


if __name__ == "__main__": raise SystemExit(main())
